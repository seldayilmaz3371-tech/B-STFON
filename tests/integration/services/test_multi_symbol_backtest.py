"""BacktestService.run_multi_symbol() testleri."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.backtest_engine import BacktestEngine
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.exceptions.domain_exceptions import BusinessRuleError, InsufficientDataError
from src.domain.strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.services.backtest_service import BacktestService, MultiSymbolBacktestResult
from src.services.price_sync_service import PriceSyncService

pytestmark = pytest.mark.integration


def _make_ohlcv(prices, n=150):
    dates = pd.date_range(end=date.today(), periods=n, freq="D")
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1000] * n,
    }, index=pd.DatetimeIndex(dates))


class MultiSymbolFakeProvider:
    """Her sembol için FARKLI fiyat serisi döner — sembol bazlı davranışı test etmek için."""

    def __init__(self, series_by_symbol: dict[str, pd.DataFrame], failing_symbols: set[str] | None = None):
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
        engine = create_db_engine(f"sqlite:///{tmp_path / 'multi_symbol_test.db'}")
        initialize_database(engine)
        sf = create_session_factory(engine)
        price_repo = SQLitePriceRepository(sf)
        provider = MultiSymbolFakeProvider(series_by_symbol, failing_symbols)
        price_sync = PriceSyncService(price_repo, provider)
        backtest_engine = BacktestEngine(ReturnCalculator(), RiskCalculator(min_data_points=5))
        return BacktestService(price_sync, backtest_engine), engine
    return _build


def _price_series(seed, base=100.0, n=150):
    rng = np.random.default_rng(seed)
    return list(base + np.cumsum(rng.normal(0.1, 2, n)))


# ── Temel işlevsellik ────────────────────────────────────────────────────────

def test_run_multi_symbol_returns_result_per_symbol(service_factory):
    series = {
        "THYAO": _make_ohlcv(_price_series(1)),
        "GARAN": _make_ohlcv(_price_series(2)),
    }
    service, engine = service_factory(series)

    result = service.run_multi_symbol(
        symbols=["THYAO", "GARAN"], strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
    )
    engine.dispose()

    assert isinstance(result, MultiSymbolBacktestResult)
    assert set(result.symbol_results.keys()) == {"THYAO", "GARAN"}
    assert result.failed_symbols == ()


def test_run_multi_symbol_splits_capital_equally(service_factory):
    """
    KRİTİK doğrulama: her sembole initial_capital / N kadar sermaye
    ayrılmalı — bu, MVP'nin BİLİNÇLİ 'eşit ağırlık' kararının GERÇEKTEN
    uygulandığını kanıtlıyor.
    """
    series = {
        "THYAO": _make_ohlcv(_price_series(1)),
        "GARAN": _make_ohlcv(_price_series(2)),
        "AKBNK": _make_ohlcv(_price_series(3)),
    }
    service, engine = service_factory(series)

    result = service.run_multi_symbol(
        symbols=["THYAO", "GARAN", "AKBNK"], strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
        initial_capital=Decimal("30000"),
    )
    engine.dispose()

    for symbol_result in result.symbol_results.values():
        assert symbol_result.initial_capital == Decimal("10000")  # 30000 / 3


def test_run_multi_symbol_combined_initial_capital_matches_input(service_factory):
    series = {
        "THYAO": _make_ohlcv(_price_series(1)),
        "GARAN": _make_ohlcv(_price_series(2)),
    }
    service, engine = service_factory(series)

    result = service.run_multi_symbol(
        symbols=["THYAO", "GARAN"], strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
        initial_capital=Decimal("20000"),
    )
    engine.dispose()

    assert result.combined_initial_capital == Decimal("20000")


# ── Hata izolasyonu ───────────────────────────────────────────────────────────

def test_run_multi_symbol_isolates_failing_symbol(service_factory):
    """
    KRİTİK doğrulama: BİR sembol veri çekemezse (ağ hatası), TÜM
    backtest BAŞARISIZ OLMAMALI — yalnızca o sembol atlanmalı.
    """
    series = {
        "THYAO": _make_ohlcv(_price_series(1)),
        "GARAN": _make_ohlcv(_price_series(2)),
    }
    service, engine = service_factory(series, failing_symbols={"GARAN"})

    result = service.run_multi_symbol(
        symbols=["THYAO", "GARAN"], strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
    )
    engine.dispose()

    assert "THYAO" in result.symbol_results
    assert "GARAN" not in result.symbol_results
    assert result.failed_symbols == ("GARAN",)


def test_run_multi_symbol_all_symbols_failing_raises(service_factory):
    series = {"THYAO": _make_ohlcv(_price_series(1))}
    service, engine = service_factory(series, failing_symbols={"THYAO"})

    with pytest.raises(InsufficientDataError):
        service.run_multi_symbol(
            symbols=["THYAO"], strategy=BuyAndHoldStrategy(),
            start=date.today() - timedelta(days=100), end=date.today(),
        )
    engine.dispose()


def test_run_multi_symbol_empty_symbol_list_raises(service_factory):
    service, engine = service_factory({})
    with pytest.raises(BusinessRuleError):
        service.run_multi_symbol(
            symbols=[], strategy=BuyAndHoldStrategy(),
            start=date.today() - timedelta(days=100), end=date.today(),
        )
    engine.dispose()


# ── Birleştirilmiş equity serisi ─────────────────────────────────────────────

def test_combined_equity_curve_is_sum_of_individual_curves(service_factory):
    """
    MATEMATİKSEL DEĞİŞMEZ: combined_equity_curve'ün HER GÜNÜ, o günkü
    TÜM sembollerin equity değerlerinin TOPLAMINA eşit olmalı.
    """
    series = {
        "THYAO": _make_ohlcv(_price_series(1)),
        "GARAN": _make_ohlcv(_price_series(2)),
    }
    service, engine = service_factory(series)

    result = service.run_multi_symbol(
        symbols=["THYAO", "GARAN"], strategy=BuyAndHoldStrategy(),
        start=date.today() - timedelta(days=100), end=date.today(),
    )
    engine.dispose()

    thyao_curve = result.symbol_results["THYAO"].portfolio_value_series
    garan_curve = result.symbol_results["GARAN"].portfolio_value_series

    # Ortak bir tarihte toplamı doğrula
    common_date = thyao_curve.index[-1]
    if common_date in garan_curve.index:
        expected = thyao_curve.loc[common_date] + garan_curve.loc[common_date]
        actual = result.combined_equity_curve.loc[common_date]
        assert abs(expected - actual) < 0.01
