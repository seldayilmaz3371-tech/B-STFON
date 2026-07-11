"""BacktestService testleri — gerçek PriceSyncService, sahte provider."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.backtest_engine import BacktestEngine
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.exceptions.domain_exceptions import InsufficientDataError
from src.domain.strategies.sma_crossover_strategy import SMACrossoverStrategy
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.services.backtest_service import BacktestService
from src.services.price_sync_service import PriceSyncService

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


def _make_series(n=200):
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D").normalize()
    prices = list(100 + np.cumsum(np.random.default_rng(3).normal(0.1, 2, n)))
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1000] * n,
    }, index=dates)


@pytest.fixture()
def service(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'backtest_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    price_repo = SQLitePriceRepository(sf)
    price_sync = PriceSyncService(price_repo, FakeProvider(_make_series(200)))
    backtest_engine = BacktestEngine(ReturnCalculator(), RiskCalculator(min_data_points=5))
    yield BacktestService(price_sync, backtest_engine)
    engine.dispose()


def test_backtest_service_runs_end_to_end(service):
    result = service.run(
        symbol="THYAO", strategy=SMACrossoverStrategy(),
        start=date.today() - timedelta(days=150), end=date.today(),
        strategy_params={"fast_window": 10, "slow_window": 30},
        initial_capital=Decimal("10000"),
    )
    assert result.symbol == "THYAO"
    assert result.final_value > Decimal("0")


def test_backtest_service_no_data_raises(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'empty_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    price_sync = PriceSyncService(SQLitePriceRepository(sf), FakeProvider(pd.DataFrame()))
    backtest_engine = BacktestEngine(ReturnCalculator(), RiskCalculator())
    service = BacktestService(price_sync, backtest_engine)

    with pytest.raises(InsufficientDataError):
        service.run(
            symbol="BILINMEYEN", strategy=SMACrossoverStrategy(),
            start=date.today() - timedelta(days=100), end=date.today(),
        )
    engine.dispose()
