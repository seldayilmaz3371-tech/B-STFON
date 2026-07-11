"""
Aşama 9 unit testleri: TefasAdapter, SlidingWindowRateLimiter, ProviderRouter.

Çalıştırma (bağımlılıklar kuruluysa):
    pytest tests/unit/infrastructure/test_tefas_adapter.py -v

Mock stratejisi:
  - pytefas kurulu değilse _download_nav() mock'lanır
  - Rate limit testleri gerçek pytefas'ı stub'layarak çalışır
  - sleep_fn inject edilerek gerçek bekleme olmaz
"""

from __future__ import annotations

import sys
import threading
import time
import types
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.exceptions.domain_exceptions import (
    DataValidationError,
    NoDataError,
    ProviderUnavailableError,
    SymbolNotFoundError,
)
from src.infrastructure.data_providers.base_provider import MarketDataProvider
from src.infrastructure.data_providers.provider_router import ProviderRouter, classify_symbol
from src.infrastructure.data_providers.rate_limiter import SlidingWindowRateLimiter
from src.infrastructure.data_providers.tefas_adapter import TefasAdapter


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _make_nav_df(nav_values: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(nav_values), freq="D")
    return pd.DataFrame({"nav": nav_values}, index=idx)


def _no_sleep(_s: float) -> None:
    pass


def _make_adapter(
    rate_limiter=None,
    chunk_size_days: int = 25,
    min_bars: int = 1,
) -> TefasAdapter:
    rl = rate_limiter or SlidingWindowRateLimiter(
        max_calls=6, window_seconds=60.0, sleep_fn=_no_sleep
    )
    return TefasAdapter(
        rate_limit_per_minute=6,
        chunk_size_days=chunk_size_days,
        retry_count=1,
        base_delay_seconds=0.01,
        max_delay_seconds=0.1,
        min_bars_required=min_bars,
        _rate_limiter=rl,
    )


def _fake_pytefas_module() -> types.ModuleType:
    """pytefas stub — gerçek network olmadan."""
    fake = types.ModuleType("pytefas")

    class FakeTefas:
        def fetch(self, fund_code, start_date, end_date):
            idx = pd.date_range(start=start_date, end=end_date, freq="D")
            return pd.DataFrame({"price": [15.0] * len(idx)}, index=idx)

    fake.Tefas = FakeTefas
    return fake


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowRateLimiter
# ─────────────────────────────────────────────────────────────────────────────

class TestSlidingWindowRateLimiter:

    def test_allows_calls_under_limit(self):
        limiter = SlidingWindowRateLimiter(max_calls=6, window_seconds=60.0, sleep_fn=_no_sleep)
        for _ in range(6):
            limiter.acquire()

    def test_exceeds_limit_triggers_sleep(self):
        sleep_calls = []
        limiter = SlidingWindowRateLimiter(
            max_calls=3, window_seconds=60.0,
            sleep_fn=lambda s: sleep_calls.append(s)
        )
        for _ in range(3):
            limiter.acquire()
        limiter.acquire()
        assert len(sleep_calls) == 1
        assert sleep_calls[0] > 0

    def test_current_call_count_accurate(self):
        limiter = SlidingWindowRateLimiter(max_calls=6, window_seconds=60.0, sleep_fn=_no_sleep)
        assert limiter.current_call_count == 0
        limiter.acquire()
        limiter.acquire()
        assert limiter.current_call_count == 2

    def test_reset_clears_history(self):
        limiter = SlidingWindowRateLimiter(max_calls=2, window_seconds=60.0, sleep_fn=_no_sleep)
        limiter.acquire()
        limiter.acquire()
        limiter.reset()
        assert limiter.current_call_count == 0

    def test_invalid_max_calls_raises(self):
        with pytest.raises(ValueError, match="max_calls"):
            SlidingWindowRateLimiter(max_calls=0)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window_seconds"):
            SlidingWindowRateLimiter(max_calls=6, window_seconds=0)

    def test_thread_safe_concurrent_acquire(self):
        limiter = SlidingWindowRateLimiter(
            max_calls=100, window_seconds=60.0, sleep_fn=_no_sleep
        )
        errors = []

        def worker():
            try:
                limiter.acquire()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert limiter.current_call_count == 20


# ─────────────────────────────────────────────────────────────────────────────
# classify_symbol
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifySymbol:

    @pytest.mark.parametrize("symbol,expected", [
        ("THYAO",    "BIST"),
        ("GARAN",    "BIST"),
        ("AKBNK",    "BIST"),
        ("EREGL",    "BIST"),
        ("ASELS",    "BIST"),
        ("THYAO.IS", "BIST"),
        ("thyao",    "BIST"),
        ("BIMAS",    "BIST"),
    ])
    def test_bist_symbols(self, symbol, expected):
        assert classify_symbol(symbol) == expected

    @pytest.mark.parametrize("symbol,expected", [
        ("AFA", "TEFAS"),
        ("MAC", "TEFAS"),
        ("TI2", "TEFAS"),
        ("YAS", "TEFAS"),
        ("GHO", "TEFAS"),
        ("A01", "TEFAS"),
    ])
    def test_tefas_symbols(self, symbol, expected):
        assert classify_symbol(symbol) == expected

    def test_is_suffix_always_bist(self):
        assert classify_symbol("XYZ.IS") == "BIST"

    def test_normalize_lowercase(self):
        assert classify_symbol("thyao") == "BIST"
        assert classify_symbol("afa") == "TEFAS"


# ─────────────────────────────────────────────────────────────────────────────
# ProviderRouter
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderRouter:

    def _make_router(self):
        bist = MagicMock(spec=MarketDataProvider)
        bist.get_provider_name.return_value = "yfinance"
        bist.fetch_ohlcv.return_value = pd.DataFrame({"Close": [100.0]})
        tefas = MagicMock(spec=MarketDataProvider)
        tefas.get_provider_name.return_value = "tefas"
        tefas.fetch_ohlcv.return_value = pd.DataFrame({"Close": [15.0]})
        return ProviderRouter(bist_provider=bist, tefas_provider=tefas), bist, tefas

    def test_bist_symbol_routed_to_bist(self):
        router, bist, tefas = self._make_router()
        router.fetch_ohlcv("THYAO", "1d")
        bist.fetch_ohlcv.assert_called_once_with(
            symbol="THYAO", timeframe="1d", start_date=None, end_date=None
        )
        tefas.fetch_ohlcv.assert_not_called()

    def test_tefas_symbol_routed_to_tefas(self):
        router, bist, tefas = self._make_router()
        router.fetch_ohlcv("AFA", "1d")
        tefas.fetch_ohlcv.assert_called_once_with(
            symbol="AFA", timeframe="1d", start_date=None, end_date=None
        )
        bist.fetch_ohlcv.assert_not_called()

    def test_is_suffix_goes_to_bist(self):
        router, bist, _ = self._make_router()
        router.fetch_ohlcv("THYAO.IS", "1d")
        bist.fetch_ohlcv.assert_called_once()

    def test_router_implements_market_data_provider(self):
        router, _, _ = self._make_router()
        assert isinstance(router, MarketDataProvider)

    def test_get_provider_name(self):
        router, _, _ = self._make_router()
        assert router.get_provider_name() == "provider_router"

    def test_get_provider_for_bist(self):
        router, bist, _ = self._make_router()
        assert router.get_provider_for("THYAO") is bist

    def test_get_provider_for_tefas(self):
        router, _, tefas = self._make_router()
        assert router.get_provider_for("AFA") is tefas

    def test_exceptions_propagate(self):
        bist = MagicMock(spec=MarketDataProvider)
        bist.fetch_ohlcv.side_effect = SymbolNotFoundError("INVALID", "yfinance")
        router = ProviderRouter(bist, MagicMock(spec=MarketDataProvider))
        with pytest.raises(SymbolNotFoundError):
            router.fetch_ohlcv("THYAO", "1d")

    def test_date_args_forwarded(self):
        router, bist, _ = self._make_router()
        s, e = date(2024, 1, 1), date(2024, 6, 1)
        router.fetch_ohlcv("THYAO", "1d", start_date=s, end_date=e)
        bist.fetch_ohlcv.assert_called_once_with(
            symbol="THYAO", timeframe="1d", start_date=s, end_date=e
        )


# ─────────────────────────────────────────────────────────────────────────────
# TefasAdapter — NAV→OHLCV
# ─────────────────────────────────────────────────────────────────────────────

class TestTefasAdapterNavToOhlcv:

    def test_nav_maps_to_all_ohlcv_columns(self):
        result = TefasAdapter._to_ohlcv(_make_nav_df([15.234567, 15.35]), "AFA")
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        for col in ["Open", "High", "Low", "Close"]:
            assert (result[col] == result["Close"]).all()
        assert (result["Volume"] == 0.0).all()

    def test_nav_value_preserved_precisely(self):
        result = TefasAdapter._to_ohlcv(_make_nav_df([1.234567]), "AFA")
        assert abs(float(result["Close"].iloc[0]) - 1.234567) < 1e-6

    def test_zero_nav_filtered(self):
        result = TefasAdapter._to_ohlcv(_make_nav_df([0.0, 15.0, -1.0, 16.0]), "AFA")
        assert len(result) == 2
        assert (result["Close"] > 0).all()

    def test_all_zero_raises_data_validation_error(self):
        with pytest.raises(DataValidationError):
            TefasAdapter._to_ohlcv(_make_nav_df([0.0, 0.0]), "AFA")

    def test_index_is_datetime(self):
        result = TefasAdapter._to_ohlcv(_make_nav_df([15.0, 16.0]), "AFA")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_timezone_aware_converted(self):
        nav_df = _make_nav_df([15.0])
        nav_df.index = nav_df.index.tz_localize("Europe/Istanbul")
        result = TefasAdapter._to_ohlcv(nav_df, "AFA")
        assert result.index.tz is None

    def test_detect_nav_column_price(self):
        assert TefasAdapter._detect_nav_column(pd.DataFrame({"price": [1.0]})) == "price"

    def test_detect_nav_column_fiyat(self):
        assert TefasAdapter._detect_nav_column(pd.DataFrame({"fiyat": [1.0]})) == "fiyat"

    def test_detect_nav_column_unknown(self):
        assert TefasAdapter._detect_nav_column(pd.DataFrame({"xyz": [1.0]})) is None


# ─────────────────────────────────────────────────────────────────────────────
# TefasAdapter — Happy Path
# ─────────────────────────────────────────────────────────────────────────────

class TestTefasAdapterHappyPath:

    def test_fetch_returns_valid_ohlcv(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_download_nav", return_value=_make_nav_df([15.0, 15.5, 16.0])):
            result = adapter.fetch_ohlcv("AFA", "1d",
                                          start_date=date(2024,1,2),
                                          end_date=date(2024,1,4))
        assert len(result) == 3
        assert "Close" in result.columns
        assert (result["Volume"] == 0.0).all()

    def test_provider_name(self):
        assert _make_adapter().get_provider_name() == "tefas"

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError, match="1d"):
            _make_adapter().fetch_ohlcv("AFA", "1h")

    def test_symbol_uppercased(self):
        adapter = _make_adapter()
        calls = []
        def mock_dl(fund_code, s, e):
            calls.append(fund_code)
            return _make_nav_df([15.0])
        with patch.object(adapter, "_download_nav", side_effect=mock_dl):
            adapter.fetch_ohlcv("afa", "1d",
                                 start_date=date(2024,1,2), end_date=date(2024,1,2))
        assert calls[0] == "AFA"

    def test_default_date_range(self):
        """start_date=None → _fetch_in_chunks'a end=bugün, start=bugün-30."""
        adapter = _make_adapter()
        captured = {}

        def mock_chunks(fund_code, start_dt, end_dt):
            captured["start"] = start_dt
            captured["end"] = end_dt
            return []

        with patch.object(adapter, "_fetch_in_chunks", side_effect=mock_chunks):
            try:
                adapter.fetch_ohlcv("AFA", "1d")
            except SymbolNotFoundError:
                pass

        today = date.today()
        assert captured["end"] == today
        assert captured["start"] == today - timedelta(days=30)

    def test_empty_response_raises_symbol_not_found(self):
        adapter = _make_adapter()
        with patch.object(adapter, "_download_nav", return_value=None):
            with pytest.raises(SymbolNotFoundError) as exc_info:
                adapter.fetch_ohlcv("INVALID", "1d",
                                     start_date=date(2024,1,2), end_date=date(2024,1,4))
        assert exc_info.value.symbol == "INVALID"
        assert exc_info.value.provider == "tefas"

    def test_below_min_bars_raises_no_data(self):
        adapter = TefasAdapter(
            _rate_limiter=SlidingWindowRateLimiter(6, 60.0, _no_sleep),
            min_bars_required=10, retry_count=1, base_delay_seconds=0.01,
        )
        with patch.object(adapter, "_download_nav", return_value=_make_nav_df([15.0, 16.0])):
            with pytest.raises(NoDataError):
                adapter.fetch_ohlcv("AFA", "1d",
                                     start_date=date(2024,1,2), end_date=date(2024,1,3))


# ─────────────────────────────────────────────────────────────────────────────
# TefasAdapter — Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestTefasAdapterChunking:

    def test_long_range_split_into_chunks(self):
        adapter = _make_adapter(chunk_size_days=25)
        calls = []
        def mock_dl(fc, s, e):
            calls.append((s, e))
            return _make_nav_df([15.0]*5)
        with patch.object(adapter, "_download_nav", side_effect=mock_dl):
            adapter.fetch_ohlcv("AFA", "1d",
                                 start_date=date(2024,1,1), end_date=date(2024,3,1))
        assert len(calls) == 3

    def test_short_range_single_chunk(self):
        adapter = _make_adapter(chunk_size_days=25)
        calls = []
        def mock_dl(fc, s, e):
            calls.append((s, e))
            return _make_nav_df([15.0]*5)
        with patch.object(adapter, "_download_nav", side_effect=mock_dl):
            adapter.fetch_ohlcv("AFA", "1d",
                                 start_date=date(2024,1,1), end_date=date(2024,1,10))
        assert len(calls) == 1

    def test_chunks_dont_overlap(self):
        adapter = _make_adapter(chunk_size_days=10)
        calls = []
        def mock_dl(fc, s, e):
            calls.append((s, e))
            return _make_nav_df([15.0]*3)
        with patch.object(adapter, "_download_nav", side_effect=mock_dl):
            adapter.fetch_ohlcv("AFA", "1d",
                                 start_date=date(2024,1,1), end_date=date(2024,1,21))
        for i in range(1, len(calls)):
            assert calls[i][0] == calls[i-1][1] + timedelta(days=1)

    def test_results_concatenated_sorted(self):
        adapter = _make_adapter(chunk_size_days=5)
        n = {"v": 0}
        def mock_dl(fc, s, e):
            n["v"] += 1
            return _make_nav_df([10.0 + n["v"], 10.5 + n["v"]],
                                 start=s.strftime("%Y-%m-%d"))
        with patch.object(adapter, "_download_nav", side_effect=mock_dl):
            result = adapter.fetch_ohlcv("AFA", "1d",
                                          start_date=date(2024,1,1), end_date=date(2024,1,12))
        assert result.index.is_monotonic_increasing


# ─────────────────────────────────────────────────────────────────────────────
# TefasAdapter — Rate Limit (gerçek pytefas stub)
# ─────────────────────────────────────────────────────────────────────────────

class TestTefasAdapterRateLimitIntegration:

    def test_rate_limiter_called_per_chunk(self):
        acquire_count = {"n": 0}
        rl = SlidingWindowRateLimiter(max_calls=6, window_seconds=60.0, sleep_fn=_no_sleep)
        original = rl.acquire

        def counting():
            acquire_count["n"] += 1
            original()

        rl.acquire = counting
        adapter = _make_adapter(rate_limiter=rl, chunk_size_days=10)

        sys.modules["pytefas"] = _fake_pytefas_module()
        try:
            adapter.fetch_ohlcv("AFA", "1d",
                                 start_date=date(2024,1,1), end_date=date(2024,1,21))
        finally:
            sys.modules.pop("pytefas", None)

        assert acquire_count["n"] >= 2

    def test_acquire_before_download(self):
        call_order = []
        rl = SlidingWindowRateLimiter(max_calls=6, window_seconds=60.0, sleep_fn=_no_sleep)
        original = rl.acquire

        def tracking():
            call_order.append("acquire")
            original()

        rl.acquire = tracking
        adapter = _make_adapter(rate_limiter=rl)

        fake = types.ModuleType("pytefas")
        class FT:
            def fetch(self, fc, start_date, end_date):
                call_order.append("download")
                idx = pd.date_range(start=start_date, periods=1, freq="D")
                return pd.DataFrame({"price": [15.0]}, index=idx)
        fake.Tefas = FT
        sys.modules["pytefas"] = fake
        try:
            adapter.fetch_ohlcv("AFA", "1d",
                                 start_date=date(2024,1,1), end_date=date(2024,1,1))
        finally:
            sys.modules.pop("pytefas", None)

        assert call_order[0] == "acquire"
        assert "download" in call_order


# ─────────────────────────────────────────────────────────────────────────────
# TefasAdapter — Error Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestTefasAdapterErrorHandling:

    def test_retry_exhausted_raises_provider_unavailable(self):
        adapter = TefasAdapter(
            _rate_limiter=SlidingWindowRateLimiter(6, 60.0, _no_sleep),
            retry_count=2, base_delay_seconds=0.001, max_delay_seconds=0.01,
        )
        with patch.object(adapter, "_download_nav",
                          side_effect=ConnectionError("ağ hatası")):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                adapter.fetch_ohlcv("AFA", "1d",
                                     start_date=date(2024,1,1), end_date=date(2024,1,1))
        assert exc_info.value.provider == "tefas"

    def test_non_retryable_calls_download_once(self):
        adapter = _make_adapter()
        count = {"n": 0}
        def raise_ve(*a, **kw):
            count["n"] += 1
            raise ValueError("geçersiz")
        with patch.object(adapter, "_download_nav", side_effect=raise_ve):
            try:
                adapter.fetch_ohlcv("AFA","1d",
                                     start_date=date(2024,1,1),end_date=date(2024,1,1))
            except (ValueError, ProviderUnavailableError):
                pass
        assert count["n"] == 1

    def test_partial_chunk_failure_raises_provider_unavailable(self):
        adapter = _make_adapter(chunk_size_days=5)
        count = {"n": 0}
        def flaky(fc, s, e):
            count["n"] += 1
            if count["n"] >= 2:
                raise ConnectionError("2. chunk hatası")
            return _make_nav_df([15.0]*3)
        with patch.object(adapter, "_download_nav", side_effect=flaky):
            with pytest.raises(ProviderUnavailableError):
                adapter.fetch_ohlcv("AFA","1d",
                                     start_date=date(2024,1,1),end_date=date(2024,1,12))
