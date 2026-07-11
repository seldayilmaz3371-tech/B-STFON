"""BuyAndHoldStrategy ve BacktestService.run_with_benchmark() testleri."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.backtest_engine import BacktestEngine
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from src.domain.strategies.sma_crossover_strategy import SMACrossoverStrategy
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.services.backtest_service import BacktestComparison, BacktestService
from src.services.price_sync_service import PriceSyncService

pytestmark = pytest.mark.integration


def _make_price_df(prices: list[float], n_days: int | None = None) -> pd.DataFrame:
    n = n_days or len(prices)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": prices[:n]}, index=dates)


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


def _make_ohlcv_series(prices, n=200):
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D").normalize()
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1000] * n,
    }, index=dates)


@pytest.fixture()
def service(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'buyhold_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    price_repo = SQLitePriceRepository(sf)

    np.random.seed(11)
    prices = list(100 + np.cumsum(np.random.default_rng(11).normal(0.1, 2, 200)))
    price_sync = PriceSyncService(price_repo, FakeProvider(_make_ohlcv_series(prices, 200)))
    backtest_engine = BacktestEngine(ReturnCalculator(), RiskCalculator(min_data_points=5))
    yield BacktestService(price_sync, backtest_engine)
    engine.dispose()


# ── BuyAndHoldStrategy domain testi (saf, provider gerektirmez) ────────────

def test_buy_and_hold_generates_single_buy_signal():
    df = _make_price_df([100.0, 101, 102, 103, 104], 5)
    signals = BuyAndHoldStrategy().generate_signals(df, {})

    assert signals.iloc[0] == 1  # ilk gün AL
    assert (signals.iloc[1:] == 0).all()  # SONRASINDA hiç sinyal YOK


def test_buy_and_hold_matches_simple_price_appreciation():
    """Buy-and-hold'un total_return'ü, KOMİSYON SIFIRKEN basit fiyat artışıyla eşleşmeli."""
    prices = [100.0] + list(100 + np.cumsum(np.random.default_rng(1).normal(0.1, 1, 99)))
    df = _make_price_df(prices, 100)
    signals = BuyAndHoldStrategy().generate_signals(df, {})

    engine = BacktestEngine(ReturnCalculator(), RiskCalculator(min_data_points=5))
    result = engine.run("TEST", df, signals, Decimal("10000"), commission_rate=Decimal("0"))

    expected_return = Decimal(str(prices[99] / prices[0] - 1))
    assert abs(result.total_return - expected_return) < Decimal("0.01")


# ── BacktestService.run_with_benchmark() ────────────────────────────────────

def test_run_with_benchmark_returns_both_results(service):
    comparison = service.run_with_benchmark(
        symbol="THYAO", strategy=SMACrossoverStrategy(),
        start=date.today() - timedelta(days=150), end=date.today(),
        strategy_params={"fast_window": 10, "slow_window": 30},
    )
    assert isinstance(comparison, BacktestComparison)
    assert comparison.strategy_result.symbol == "THYAO"
    assert comparison.benchmark_result.symbol == "THYAO"


def test_run_with_benchmark_uses_same_price_data_for_both(service):
    """
    KRİTİK: strateji ve benchmark, AYNI start_date/end_date'e sahip
    olmalı — farklı tarih aralıklarıyla karşılaştırma HAKSIZ olurdu.
    """
    comparison = service.run_with_benchmark(
        symbol="THYAO", strategy=SMACrossoverStrategy(),
        start=date.today() - timedelta(days=150), end=date.today(),
        strategy_params={"fast_window": 10, "slow_window": 30},
    )
    assert comparison.strategy_result.start_date == comparison.benchmark_result.start_date
    assert comparison.strategy_result.end_date == comparison.benchmark_result.end_date


def test_run_with_benchmark_fetches_price_data_only_once(service):
    """
    DRY doğrulaması: run_with_benchmark(), fiyat verisini YALNIZCA
    BİR KEZ çekmeli (iki AYRI PriceSyncService çağrısı YAPMAMALI).
    """
    call_count = {"n": 0}
    original_get_price_history = service._price_sync.get_price_history

    def counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original_get_price_history(*args, **kwargs)

    service._price_sync.get_price_history = counting_wrapper

    service.run_with_benchmark(
        symbol="THYAO", strategy=SMACrossoverStrategy(),
        start=date.today() - timedelta(days=150), end=date.today(),
        strategy_params={"fast_window": 10, "slow_window": 30},
    )
    assert call_count["n"] == 1  # YALNIZCA BİR fetch, iki DEĞİL


def test_buy_and_hold_used_as_strategy_directly_via_run(service):
    """BuyAndHoldStrategy, herhangi bir strateji gibi doğrudan run()'a da geçirilebilmeli."""
    result = service.run(
        symbol="THYAO", strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
    )
    assert result.total_trades == 1  # yalnızca BİR alım, hiç satış yok
