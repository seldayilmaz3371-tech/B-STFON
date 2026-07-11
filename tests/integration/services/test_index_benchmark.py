"""BacktestService.run_with_benchmark()'ün index_benchmark_symbol özelliği testleri."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.backtest_engine import BacktestEngine
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.services.backtest_service import BacktestComparison, BacktestService
from src.services.price_sync_service import PriceSyncService

pytestmark = pytest.mark.integration


def _make_ohlcv(prices, n=150):
    dates = pd.date_range(end=date.today(), periods=n, freq="D")
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1000] * n,
    }, index=pd.DatetimeIndex(dates))


def _price_series(seed, base=100.0, n=150):
    rng = np.random.default_rng(seed)
    return list(base + np.cumsum(rng.normal(0.1, 2, n)))


class TwoSymbolProvider:
    def __init__(self, series_by_symbol, failing_symbols=None):
        self._series = series_by_symbol
        self._failing = failing_symbols or set()

    def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
        if symbol in self._failing:
            raise ConnectionError(f"{symbol} için simüle edilmiş ağ hatası")
        df = self._series[symbol]
        mask = pd.Series(True, index=df.index)
        if start_date is not None:
            mask &= df.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= df.index <= pd.Timestamp(end_date)
        return df[mask]

    def get_provider_name(self) -> str:
        return "mock"


@pytest.fixture()
def service_factory(tmp_path):
    def _build(series_by_symbol, failing_symbols=None):
        engine = create_db_engine(f"sqlite:///{tmp_path / 'index_bench_test.db'}")
        initialize_database(engine)
        sf = create_session_factory(engine)
        price_repo = SQLitePriceRepository(sf)
        provider = TwoSymbolProvider(series_by_symbol, failing_symbols)
        price_sync = PriceSyncService(price_repo, provider)
        backtest_engine = BacktestEngine(ReturnCalculator(), RiskCalculator(min_data_points=5))
        return BacktestService(price_sync, backtest_engine), engine
    return _build


def test_run_with_benchmark_without_index_symbol_leaves_it_none(service_factory):
    """Geriye dönük UYUMLULUK: index_benchmark_symbol verilmezse, alan None kalmalı."""
    series = {"THYAO": _make_ohlcv(_price_series(1))}
    service, engine = service_factory(series)

    comparison = service.run_with_benchmark(
        symbol="THYAO", strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
    )
    engine.dispose()

    assert comparison.index_benchmark_result is None
    assert comparison.index_benchmark_symbol is None


def test_run_with_benchmark_with_index_symbol_computes_it(service_factory):
    series = {
        "THYAO": _make_ohlcv(_price_series(1)),
        "XU100.IS": _make_ohlcv(_price_series(99)),
    }
    service, engine = service_factory(series)

    comparison = service.run_with_benchmark(
        symbol="THYAO", strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
        index_benchmark_symbol="XU100.IS",
    )
    engine.dispose()

    assert comparison.index_benchmark_result is not None
    assert comparison.index_benchmark_symbol == "XU100.IS"
    assert comparison.index_benchmark_result.symbol == "XU100.IS"


def test_index_benchmark_fetch_failure_does_not_break_main_comparison(service_factory):
    """
    KRİTİK doğrulama (hata izolasyonu): endeks verisi çekilemezse, ANA
    karşılaştırma (strateji vs. aynı-sembol Buy&Hold) YİNE DE başarılı
    dönmeli — yalnızca index_benchmark_result None kalmalı.
    """
    series = {
        "THYAO": _make_ohlcv(_price_series(1)),
        "XU100.IS": _make_ohlcv(_price_series(99)),
    }
    service, engine = service_factory(series, failing_symbols={"XU100.IS"})

    comparison = service.run_with_benchmark(
        symbol="THYAO", strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
        index_benchmark_symbol="XU100.IS",
    )
    engine.dispose()

    assert isinstance(comparison, BacktestComparison)
    assert comparison.strategy_result is not None  # ANA sonuç ETKİLENMEDİ
    assert comparison.benchmark_result is not None
    assert comparison.index_benchmark_result is None  # yalnızca İKİNCİL sonuç None
    assert comparison.index_benchmark_symbol is None
