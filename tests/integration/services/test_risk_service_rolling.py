"""RiskService.get_rolling_volatility() / get_drawdown_series() testleri."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import InsufficientDataError
from src.domain.models.transaction import Transaction
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.repositories.sqlite.cash_ledger_repository import SQLiteCashLedgerRepository
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.infrastructure.repositories.sqlite.transaction_repository import SQLiteTransactionRepository
from src.services.price_sync_service import PriceSyncService
from src.services.risk_service import RiskService

pytestmark = pytest.mark.integration


class FakeProvider:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
        mask = pd.Series(True, index=self._df.index)
        if start_date is not None:
            mask &= self._df.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= self._df.index <= pd.Timestamp(end_date)
        return self._df[mask]

    def get_provider_name(self) -> str:
        return "mock"


def _make_ohlcv(prices, n=150):
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D").normalize()
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1000] * n,
    }, index=dates)


@pytest.fixture()
def env(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'rolling_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sf() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="Rolling Test", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    tx_repo = SQLiteTransactionRepository(sf)
    tx_repo.add_transaction(pid, "BIST_STOCK", Transaction(
        symbol="THYAO", transaction_type=TransactionType.BUY,
        timestamp=datetime.today() - timedelta(days=200),
        quantity=Decimal("100"), price=Decimal("100.00"),
    ))
    yield pid, tx_repo, sf
    engine.dispose()


def _build_service(tx_repo, sf, prices, n_days=150):
    cash_repo = SQLiteCashLedgerRepository(sf)
    price_repo = SQLitePriceRepository(sf)
    price_sync = PriceSyncService(price_repo, FakeProvider(_make_ohlcv(prices, n_days)))
    return RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo, price_sync_service=price_sync,
        risk_calculator=RiskCalculator(min_data_points=5), return_calculator=ReturnCalculator(),
        risk_free_rate_annual=0.45,
    )


# ── get_rolling_volatility ───────────────────────────────────────────────────

def test_rolling_volatility_returns_series_not_scalar(env):
    pid, tx_repo, sf = env
    prices = list(100 + np.cumsum(np.random.default_rng(1).normal(0.05, 1.5, 150)))
    service = _build_service(tx_repo, sf, prices, 150)

    result = service.get_rolling_volatility(pid, lookback_days=100, window=20, include_cash=False)

    assert isinstance(result, pd.Series)
    assert len(result) > 0
    assert (result >= 0).all()  # volatilite asla negatif olamaz


def test_rolling_volatility_matches_point_calculation_for_full_window(env):
    """
    KRİTİK doğrulama: rolling serinin SON değeri, AYNI veriyle
    compute_risk_profile()'ın hesapladığı annualized_volatility'ye
    YAKIN olmalı (tam pencere kullanıldığında iki yöntem MATEMATİKSEL
    OLARAK örtüşmeli — farklı formüller kullanıyor OLMAMALILAR).
    """
    pid, tx_repo, sf = env
    prices = list(100 + np.cumsum(np.random.default_rng(2).normal(0.05, 1.5, 150)))
    service = _build_service(tx_repo, sf, prices, 150)

    profile = service.compute_risk_profile(pid, lookback_days=100, include_cash=False)
    rolling = service.get_rolling_volatility(pid, lookback_days=100, window=99, include_cash=False)

    # window=99 ~ tüm veriye YAKIN bir pencere -> son değer, nokta
    # tahminine YAKLAŞIK eşit olmalı (aynı ddof=1 formülü, aynı veri).
    assert abs(rolling.iloc[-1] - profile.annualized_volatility) < 0.05


def test_rolling_volatility_window_larger_than_lookback_raises(env):
    pid, tx_repo, sf = env
    prices = list(100 + np.cumsum(np.random.default_rng(3).normal(0.05, 1.5, 150)))
    service = _build_service(tx_repo, sf, prices, 150)

    with pytest.raises(InsufficientDataError):
        service.get_rolling_volatility(pid, lookback_days=10, window=30, include_cash=False)


# ── get_drawdown_series ──────────────────────────────────────────────────────

def test_drawdown_series_never_positive(env):
    pid, tx_repo, sf = env
    prices = list(100 + np.cumsum(np.random.default_rng(4).normal(0.05, 1.5, 150)))
    service = _build_service(tx_repo, sf, prices, 150)

    result = service.get_drawdown_series(pid, lookback_days=100, include_cash=False)

    assert isinstance(result, pd.Series)
    assert (result <= 1e-9).all()  # asla pozitif (float yuvarlama payı)


def test_drawdown_series_first_value_is_zero(env):
    """İlk gün, TANIM GEREĞİ kendi zirvesidir — drawdown = 0."""
    pid, tx_repo, sf = env
    prices = list(100 + np.cumsum(np.random.default_rng(5).normal(0.05, 1.5, 150)))
    service = _build_service(tx_repo, sf, prices, 150)

    result = service.get_drawdown_series(pid, lookback_days=100, include_cash=False)
    assert abs(result.iloc[0]) < 1e-9


def test_drawdown_series_minimum_matches_point_calculation(env):
    """
    KRİTİK doğrulama: drawdown serisinin MİNİMUMU, compute_risk_profile()'ın
    hesapladığı max_drawdown ile TUTARLI olmalı — AYNI formül, farklı
    granülerlik (nokta vs. seri), UYUŞMAZLIK matematiksel bir hata
    olurdu.
    """
    pid, tx_repo, sf = env
    prices = list(100 + np.cumsum(np.random.default_rng(6).normal(-0.1, 3, 150)))
    service = _build_service(tx_repo, sf, prices, 150)

    profile = service.compute_risk_profile(pid, lookback_days=100, include_cash=False)
    drawdown_series = service.get_drawdown_series(pid, lookback_days=100, include_cash=False)

    assert abs(drawdown_series.min() - profile.max_drawdown.max_drawdown) < 0.02


def test_drawdown_series_monotonically_recovers_after_new_high(env):
    """Fiyat yeni bir zirveye ulaştığında drawdown TAM OLARAK sıfır olmalı."""
    pid, tx_repo, sf = env
    # Kasıtlı: düş, sonra YENİ ZİRVEYE çık
    prices = [100.0] * 10 + [90.0] * 10 + [110.0] * 130  # son kısım açık yeni zirve
    service = _build_service(tx_repo, sf, prices, 150)

    result = service.get_drawdown_series(pid, lookback_days=100, include_cash=False)
    assert abs(result.iloc[-1]) < 1e-6  # en son gün, yeni zirvede -> drawdown ~0
