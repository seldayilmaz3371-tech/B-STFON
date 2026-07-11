"""RiskSnapshot repository + RiskService.compute_and_persist_snapshot testleri."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.enums.transaction_type import TransactionType
from src.domain.enums.var_method import VaRMethod
from src.domain.models.risk_snapshot import RiskSnapshot
from src.domain.models.transaction import Transaction
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.repositories.sqlite.cash_ledger_repository import SQLiteCashLedgerRepository
from src.infrastructure.repositories.sqlite.risk_snapshot_repository import SQLiteRiskSnapshotRepository
from src.infrastructure.repositories.sqlite.transaction_repository import SQLiteTransactionRepository
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
def portfolio_setup(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'snapshot_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sf() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="Snapshot Test", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    tx_repo = SQLiteTransactionRepository(sf)
    tx_repo.add_transaction(pid, "BIST_STOCK", Transaction(
        symbol="THYAO", transaction_type=TransactionType.BUY,
        timestamp=datetime.today() - timedelta(days=200),
        quantity=__import__("decimal").Decimal("100"), price=__import__("decimal").Decimal("100.00"),
    ))
    tx_repo.add_transaction(pid, "BIST_STOCK", Transaction(
        symbol="GARAN", transaction_type=TransactionType.BUY,
        timestamp=datetime.today() - timedelta(days=180),
        quantity=__import__("decimal").Decimal("50"), price=__import__("decimal").Decimal("50.00"),
    ))
    yield pid, tx_repo, sf
    engine.dispose()


def test_compute_and_persist_snapshot_full_fields(portfolio_setup):
    pid, tx_repo, sf = portfolio_setup
    cash_repo = SQLiteCashLedgerRepository(sf)
    snapshot_repo = SQLiteRiskSnapshotRepository(sf)

    np.random.seed(11)
    thyao_prices = list(100 + np.cumsum(np.random.normal(0.05, 1.5, 150)))
    garan_prices = list(50 + np.cumsum(np.random.normal(0.03, 1.0, 150)))
    provider = FakeConstantProvider({
        "THYAO": _make_price_series(thyao_prices, 150),
        "GARAN": _make_price_series(garan_prices, 150),
    })

    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo, price_sync_service=PriceSyncService(SQLitePriceRepository(sf), provider),
        risk_calculator=RiskCalculator(min_data_points=30), return_calculator=ReturnCalculator(),
        risk_free_rate_annual=0.45,
    )

    snapshot = service.compute_and_persist_snapshot(
        pid, snapshot_repo, lookback_days=100, include_cash=False,
    )

    assert snapshot.snapshot_id is not None
    assert snapshot.is_stale is False
    assert snapshot.portfolio_volatility is not None
    assert snapshot.calmar_ratio is not None or snapshot.max_drawdown == 0.0
    assert snapshot.var_99 is not None
    assert snapshot.cvar_99 is not None
    assert snapshot.herfindahl_index is not None
    assert snapshot.top5_concentration is not None
    # 2 pozisyon var -> top5 = tüm portföy = 1.0
    assert abs(snapshot.top5_concentration - 1.0) < 0.01


def test_snapshot_marks_previous_as_stale(portfolio_setup):
    """
    KRİTİK doğrulama: yeni bir snapshot eklendiğinde, ÖNCEKİ snapshot
    silinmiyor, yalnızca is_stale=1 işaretleniyor (audit trail).
    """
    pid, tx_repo, sf = portfolio_setup
    cash_repo = SQLiteCashLedgerRepository(sf)
    snapshot_repo = SQLiteRiskSnapshotRepository(sf)

    np.random.seed(22)
    prices = list(100 + np.cumsum(np.random.normal(0.05, 1.5, 150)))
    provider = FakeConstantProvider({
        "THYAO": _make_price_series(prices, 150),
        "GARAN": _make_price_series(prices, 150),
    })
    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo, price_sync_service=PriceSyncService(SQLitePriceRepository(sf), provider),
        risk_calculator=RiskCalculator(min_data_points=30), return_calculator=ReturnCalculator(),
        risk_free_rate_annual=0.45,
    )

    first = service.compute_and_persist_snapshot(pid, snapshot_repo, lookback_days=100, include_cash=False)
    second = service.compute_and_persist_snapshot(pid, snapshot_repo, lookback_days=100, include_cash=False)

    latest = snapshot_repo.get_latest_snapshot(pid)
    assert latest.snapshot_id == second.snapshot_id

    history = snapshot_repo.get_history(pid)
    assert len(history) == 2  # HER İKİSİ de hâlâ DB'de (silinmedi)
    stale_ids = {s.snapshot_id for s in history if s.is_stale}
    assert stale_ids == {first.snapshot_id}


def test_get_latest_snapshot_none_when_no_snapshots(portfolio_setup):
    pid, _, sf = portfolio_setup
    snapshot_repo = SQLiteRiskSnapshotRepository(sf)
    assert snapshot_repo.get_latest_snapshot(pid) is None


def test_snapshot_with_benchmark_populates_relative_metrics(portfolio_setup):
    pid, tx_repo, sf = portfolio_setup
    cash_repo = SQLiteCashLedgerRepository(sf)
    snapshot_repo = SQLiteRiskSnapshotRepository(sf)

    np.random.seed(33)
    thyao_prices = list(100 + np.cumsum(np.random.normal(0.05, 1.5, 150)))
    garan_prices = list(50 + np.cumsum(np.random.normal(0.03, 1.0, 150)))
    bench_prices = list(1000 + np.cumsum(np.random.normal(0.04, 10, 150)))
    provider = FakeConstantProvider({
        "THYAO": _make_price_series(thyao_prices, 150),
        "GARAN": _make_price_series(garan_prices, 150),
        "XU100.IS": _make_price_series(bench_prices, 150),
    })
    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo, price_sync_service=PriceSyncService(SQLitePriceRepository(sf), provider),
        risk_calculator=RiskCalculator(min_data_points=30), return_calculator=ReturnCalculator(),
        risk_free_rate_annual=0.45,
    )
    snapshot = service.compute_and_persist_snapshot(
        pid, snapshot_repo, lookback_days=100, include_cash=False, benchmark_code="XU100.IS",
    )
    assert snapshot.beta is not None
    assert snapshot.benchmark_code == "XU100.IS"
