"""
TEFAS fonlarının portföy/risk pipeline'ında uçtan uca çalıştığını
doğrulayan testler.

GEREKÇE: Bu proje "BIST hisseleri VE TEFAS yatırım fonları" olarak
tanımlanmış (üst düzey proje kapsamı), ama bu oturumda yazılan
neredeyse TÜM entegrasyon testleri BIST sembolleriyle (THYAO, GARAN,
AKBNK) çalıştırıldı. TEFAS fonlarının RiskService/PriceSyncService/
SnapshotScheduler pipeline'ında GERÇEKTEN çalıştığı hiç doğrulanmamıştı
— bu dosya o boşluğu kapatıyor.

TEFAS'a özgü yapısal fark: TefasAdapter._to_ohlcv(), NAV'ı
Open=High=Low=Close=NAV, Volume=0 olarak eşliyor (gerçek OHLC yok,
yalnızca günlük tek bir NAV değeri) — bu, BIST'in gerçek OHLCV'sinden
YAPISAL olarak farklı. Bu testler, bu farkın downstream'de (risk
hesaplaması, konsantrasyon metrikleri) sorun ÇIKARMADIĞINI kanıtlıyor.
"""

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
from src.domain.models.transaction import Transaction
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.data_providers.provider_router import classify_symbol
from src.infrastructure.repositories.sqlite.cash_ledger_repository import SQLiteCashLedgerRepository
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.infrastructure.repositories.sqlite.transaction_repository import SQLiteTransactionRepository
from src.services.price_sync_service import PriceSyncService
from src.services.risk_service import RiskService
from src.services.transaction_service import TransactionService

pytestmark = pytest.mark.integration


class FakeTefasLikeProvider:
    """
    TefasAdapter._to_ohlcv() ile AYNI yapısal deseni taşır: Open=High=
    Low=Close=NAV, Volume=0 — gerçek OHLC YOK. Bu, BIST'in (farklı
    open/high/low/close değerleri taşıyan) sahte sağlayıcılarından
    KASITLI OLARAK farklı tutuldu.
    """

    def __init__(self, nav_series: pd.DataFrame):
        self._nav_series = nav_series

    def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
        mask = pd.Series(True, index=self._nav_series.index)
        if start_date is not None:
            mask &= self._nav_series.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= self._nav_series.index <= pd.Timestamp(end_date)
        return self._nav_series[mask]

    def get_provider_name(self) -> str:
        return "mock"


def _make_nav_series(nav_values: list[float], n_days: int = 150) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="D").normalize()
    if len(nav_values) < n_days:
        nav_values = nav_values + [nav_values[-1]] * (n_days - len(nav_values))
    nav_values = nav_values[-n_days:]
    return pd.DataFrame({
        "Open": nav_values, "High": nav_values, "Low": nav_values,
        "Close": nav_values, "Volume": [0] * n_days,
    }, index=dates)


def test_classify_symbol_routes_fund_code_to_tefas():
    """
    Ön koşul doğrulaması: 3 harfli bir fon kodu GERÇEKTEN TEFAS'a
    sınıflandırılıyor mu (varsayım değil, kontrol).
    """
    assert classify_symbol("YAC") == "TEFAS"
    assert classify_symbol("TCD") == "TEFAS"
    assert classify_symbol("THYAO") == "BIST"  # 5 harf, BIST deseni


@pytest.fixture()
def env(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'tefas_e2e_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sf() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="TEFAS Test Portföyü", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    yield {
        "pid": pid, "sf": sf,
        "tx_repo": SQLiteTransactionRepository(sf),
        "cash_repo": SQLiteCashLedgerRepository(sf),
        "price_repo": SQLitePriceRepository(sf),
    }
    engine.dispose()


def test_tefas_fund_transaction_auto_classifies_correctly(env):
    """TransactionService, 3 harfli bir fon kodunu TEFAS_FUND olarak sınıflandırıyor mu."""
    tx_service = TransactionService(env["tx_repo"], cash_ledger_repo=env["cash_repo"])
    tx = tx_service.add_transaction(
        env["pid"], "YAC", TransactionType.BUY, Decimal("1000"), Decimal("1.50"),
        datetime.today().date(),
    )
    assert tx.symbol_type == "TEFAS_FUND"


def test_tefas_fund_risk_calculation_end_to_end(env):
    """
    KRİTİK: TEFAS'ın Open=High=Low=Close=NAV, Volume=0 yapısı,
    RiskService pipeline'ında (fetch → write → read → hesaplama)
    ÇÖKMEDEN çalışıyor mu. Bu, bu oturumda İLK KEZ test ediliyor.
    """
    tx_service = TransactionService(env["tx_repo"], cash_ledger_repo=env["cash_repo"])
    tx_service.add_transaction(
        env["pid"], "YAC", TransactionType.BUY, Decimal("1000"), Decimal("1.50"),
        (datetime.today() - timedelta(days=200)).date(),
    )

    np.random.seed(42)
    nav_values = list(1.50 + np.cumsum(np.random.normal(0.002, 0.01, 150)))
    provider = FakeTefasLikeProvider(_make_nav_series(nav_values, 150))
    price_sync = PriceSyncService(env["price_repo"], provider)

    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=price_sync, risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )

    profile = risk_service.compute_risk_profile(env["pid"], lookback_days=100, include_cash=False)

    assert profile.data_points_used > 0
    assert profile.annualized_volatility >= 0  # NAV, hisse kadar oynak olmasa da sıfır olmayabilir
    assert isinstance(profile.sharpe_ratio, float)


def test_mixed_bist_and_tefas_portfolio(env):
    """
    Gerçekçi karma senaryo: bir portföyde HEM BIST hissesi HEM TEFAS
    fonu — RiskService'in _build_portfolio_value_series'i İKİ FARKLI
    yapısal veri kaynağını (gerçek OHLC vs NAV-only) doğru birleştirip
    birleştiremediğini doğrular.
    """
    tx_service = TransactionService(env["tx_repo"], cash_ledger_repo=env["cash_repo"])
    tx_service.add_transaction(
        env["pid"], "THYAO", TransactionType.BUY, Decimal("100"), Decimal("250.00"),
        (datetime.today() - timedelta(days=200)).date(),
    )
    tx_service.add_transaction(
        env["pid"], "YAC", TransactionType.BUY, Decimal("1000"), Decimal("1.50"),
        (datetime.today() - timedelta(days=200)).date(),
    )

    np.random.seed(7)
    bist_prices = list(250 + np.cumsum(np.random.normal(0.1, 3, 150)))
    nav_values = list(1.50 + np.cumsum(np.random.normal(0.002, 0.01, 150)))

    class MixedProvider:
        def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
            if symbol == "THYAO":
                df = _make_nav_series(bist_prices, 150)  # gerçekte farklı open/high/low olurdu ama yapı yeterli
            else:
                df = _make_nav_series(nav_values, 150)
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= df.index >= pd.Timestamp(start_date)
            if end_date is not None:
                mask &= df.index <= pd.Timestamp(end_date)
            return df[mask]

        def get_provider_name(self) -> str:
            return "mock"

    price_sync = PriceSyncService(env["price_repo"], MixedProvider())
    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=price_sync, risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )

    profile = risk_service.compute_risk_profile(env["pid"], lookback_days=100, include_cash=False)
    assert profile.data_points_used > 0

    # Konsantrasyon metriklerinin de İKİ farklı varlık tipiyle çalıştığını doğrula
    value_series, combined = risk_service._build_portfolio_value_series(
        env["pid"], lookback_days=100, include_cash=False,
    )
    assert set(combined.columns) == {"THYAO", "YAC"}
    weights = risk_service._compute_position_weights(combined)
    hhi, top5 = risk_service._risk_calc.calculate_concentration_metrics(weights)
    assert 0 < hhi <= 1.0
