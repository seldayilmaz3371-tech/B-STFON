"""
SnapshotScheduler testleri.

Gerçek PortfolioService/RiskService/repository'ler kullanılıyor — yalnızca
dış network (MarketDataProvider) sahte. Bu, "job doğru servisleri doğru
parametrelerle çağırıyor mu" sorusuna GERÇEK bir cevap verir (mock'lanmış
servislerle test etseydik yalnızca "mock'u doğru çağırdım" kanıtlanırdı,
GERÇEK entegrasyonu değil).
"""

from __future__ import annotations

import time
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
from src.infrastructure.repositories.sqlite.cash_ledger_repository import SQLiteCashLedgerRepository
from src.infrastructure.repositories.sqlite.portfolio_repository import SQLitePortfolioRepository
from src.infrastructure.repositories.sqlite.risk_snapshot_repository import SQLiteRiskSnapshotRepository
from src.infrastructure.repositories.sqlite.transaction_repository import SQLiteTransactionRepository
from src.infrastructure.scheduler.snapshot_scheduler import SnapshotScheduler
from src.services.portfolio_service import PortfolioService
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.services.price_sync_service import PriceSyncService
from src.services.risk_service import RiskService

pytestmark = pytest.mark.integration


class FakeConstantProvider:
    def __init__(self, price_map):
        self._price_map = price_map

    def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
        df = self._price_map.get(symbol, pd.DataFrame())
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        if start_date is not None:
            mask &= df.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= df.index <= pd.Timestamp(end_date)
        return df[mask]

    def get_provider_name(self) -> str:
        return "mock"


def _make_price_series(prices, n_days=150):
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="D").normalize()
    if len(prices) < n_days:
        prices = prices + [prices[-1]] * (n_days - len(prices))
    prices = prices[-n_days:]
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices,
        "Volume": [1_000_000] * n_days,
    }, index=dates)


@pytest.fixture()
def env(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'scheduler_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)

    portfolio_repo = SQLitePortfolioRepository(sf)
    tx_repo = SQLiteTransactionRepository(sf)
    cash_repo = SQLiteCashLedgerRepository(sf)
    snapshot_repo = SQLiteRiskSnapshotRepository(sf)
    portfolio_service = PortfolioService(
        transaction_repo=tx_repo, market_data_service=None, portfolio_repo=portfolio_repo,
    )
    yield {
        "sf": sf, "portfolio_repo": portfolio_repo, "tx_repo": tx_repo,
        "cash_repo": cash_repo, "snapshot_repo": snapshot_repo,
        "portfolio_service": portfolio_service,
    }
    engine.dispose()


def _create_portfolio_with_holding(env, name, symbol="THYAO", benchmark_code=None):
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with env["sf"]() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name=name, currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
            benchmark_code=benchmark_code,
        ))
        session.commit()
    env["tx_repo"].add_transaction(pid, "BIST_STOCK", Transaction(
        symbol=symbol, transaction_type=TransactionType.BUY,
        timestamp=datetime.today() - timedelta(days=200),
        quantity=Decimal("100"), price=Decimal("10.00"),
    ))
    return pid


def test_run_now_computes_snapshot_for_all_portfolios(env):
    pid1 = _create_portfolio_with_holding(env, "Portföy 1", "THYAO")
    pid2 = _create_portfolio_with_holding(env, "Portföy 2", "GARAN")

    np.random.seed(5)
    provider = FakeConstantProvider({
        "THYAO": _make_price_series(list(100 + np.cumsum(np.random.normal(0.05, 1.5, 150)))),
        "GARAN": _make_price_series(list(50 + np.cumsum(np.random.normal(0.03, 1.0, 150)))),
    })
    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=PriceSyncService(SQLitePriceRepository(env["sf"]), provider), risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    scheduler = SnapshotScheduler(
        portfolio_service=env["portfolio_service"], risk_service=risk_service,
        risk_snapshot_repo=env["snapshot_repo"], lookback_days=100,
    )

    result = scheduler.run_now()

    assert result == {"succeeded": 2, "failed": 0}
    assert env["snapshot_repo"].get_latest_snapshot(pid1) is not None
    assert env["snapshot_repo"].get_latest_snapshot(pid2) is not None


def test_run_now_isolates_per_portfolio_failures(env):
    """
    KRİTİK: Bir portföyün (fiyat verisi eksik -> InsufficientDataError)
    başarısız olması, DİĞER portföyün başarıyla işlenmesini
    ENGELLEMEMELİ.
    """
    good_pid = _create_portfolio_with_holding(env, "İyi Portföy", "THYAO")
    bad_pid = _create_portfolio_with_holding(env, "Veri Eksik Portföy", "BILINMEYEN")

    np.random.seed(6)
    # Yalnızca THYAO için veri var — BILINMEYEN için provider boş DataFrame döner
    provider = FakeConstantProvider({
        "THYAO": _make_price_series(list(100 + np.cumsum(np.random.normal(0.05, 1.5, 150)))),
    })
    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=PriceSyncService(SQLitePriceRepository(env["sf"]), provider), risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    scheduler = SnapshotScheduler(
        portfolio_service=env["portfolio_service"], risk_service=risk_service,
        risk_snapshot_repo=env["snapshot_repo"], lookback_days=100,
    )

    result = scheduler.run_now()

    assert result == {"succeeded": 1, "failed": 1}
    assert env["snapshot_repo"].get_latest_snapshot(good_pid) is not None
    assert env["snapshot_repo"].get_latest_snapshot(bad_pid) is None


def test_start_is_idempotent(env):
    """start() birden fazla kez çağrılsa da job ÇOĞALMAMALI (tek job, aynı id)."""
    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=PriceSyncService(SQLitePriceRepository(env["sf"]), FakeConstantProvider({})), risk_calculator=RiskCalculator(),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    scheduler = SnapshotScheduler(
        portfolio_service=env["portfolio_service"], risk_service=risk_service,
        risk_snapshot_repo=env["snapshot_repo"], interval_hours=24,
    )
    try:
        scheduler.start()
        scheduler.start()  # ikinci çağrı NO-OP olmalı
        scheduler.start()  # üçüncü çağrı da

        jobs = scheduler._scheduler.get_jobs()
        assert len(jobs) == 1  # ÇOĞALMADI
        assert scheduler.is_running is True
    finally:
        scheduler.shutdown()


def test_shutdown_stops_scheduler(env):
    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=PriceSyncService(SQLitePriceRepository(env["sf"]), FakeConstantProvider({})), risk_calculator=RiskCalculator(),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    scheduler = SnapshotScheduler(
        portfolio_service=env["portfolio_service"], risk_service=risk_service,
        risk_snapshot_repo=env["snapshot_repo"], interval_hours=24,
    )
    scheduler.start()
    assert scheduler.is_running is True
    scheduler.shutdown()
    assert scheduler.is_running is False


def test_run_now_with_no_portfolios(env):
    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=PriceSyncService(SQLitePriceRepository(env["sf"]), FakeConstantProvider({})), risk_calculator=RiskCalculator(),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    scheduler = SnapshotScheduler(
        portfolio_service=env["portfolio_service"], risk_service=risk_service,
        risk_snapshot_repo=env["snapshot_repo"],
    )
    result = scheduler.run_now()
    assert result == {"succeeded": 0, "failed": 0}


def test_run_now_passes_benchmark_code_from_portfolio(env):
    """
    Portföyün benchmark_code'u varsa, job'ın bunu risk_service'e
    doğru şekilde geçirdiğini (ve snapshot'a yansıdığını) doğrular.
    """
    pid = _create_portfolio_with_holding(env, "Benchmarklı", "THYAO", benchmark_code="XU100.IS")
    np.random.seed(7)
    provider = FakeConstantProvider({
        "THYAO": _make_price_series(list(100 + np.cumsum(np.random.normal(0.05, 1.5, 150)))),
        "XU100.IS": _make_price_series(list(1000 + np.cumsum(np.random.normal(0.04, 10, 150)))),
    })
    risk_service = RiskService(
        transaction_repo=env["tx_repo"], cash_ledger_repo=env["cash_repo"],
        price_sync_service=PriceSyncService(SQLitePriceRepository(env["sf"]), provider), risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    scheduler = SnapshotScheduler(
        portfolio_service=env["portfolio_service"], risk_service=risk_service,
        risk_snapshot_repo=env["snapshot_repo"], lookback_days=100,
    )
    result = scheduler.run_now()
    assert result["succeeded"] == 1

    snapshot = env["snapshot_repo"].get_latest_snapshot(pid)
    assert snapshot.benchmark_code == "XU100.IS"
    assert snapshot.beta is not None
