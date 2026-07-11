"""
MarketDataService unit testleri.

Test kapsamı:
  1. Mutlu yol (happy path) — provider ve calculator doğru sırada çağrılıyor mu?
  2. DTO doğruluğu — MarketAnalysisResult alanları doğru dolduruluyor mu?
  3. Hata dönüşümü — ProviderError ailesi → MarketDataServiceError
  4. Hata dönüşümü — CalculationError/KeyError/ValueError → MarketDataServiceError
  5. Boş/None veri savunma kontrolü
  6. Parametrelerin (period, ma_period) doğru iletildiği
  7. Stateless doğrulaması — aynı instance ile ardışık çağrılar birbirini etkilemiyor
  8. Exception chaining — orijinal hata __cause__ zincirinde korunuyor

KRİTİK TASARIM KARARI: Hem MarketDataProvider hem TechnicalCalculator
tamamen mock'lanır. Gerçek yfinance çağrısı, gerçek pandas hesaplaması
zorunlu değildir (calculator çağrıları da Mock ile izlenir) — ancak bazı
testlerde gerçek TechnicalCalculator kullanılarak uçtan-uca entegrasyon
da ayrıca doğrulanır (iki seviyeli test stratejisi: izole + entegre).
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, call

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.technical_calculator import TechnicalCalculator
from src.domain.exceptions.domain_exceptions import (
    InsufficientDataError,
    MarketDataServiceError,
    NoDataError,
    ProviderUnavailableError,
    SymbolNotFoundError,
)
from src.infrastructure.data_providers.base_provider import MarketDataProvider
from src.services.market_data_service import MarketAnalysisResult, MarketDataService


# ── Test Veri Yardımcıları ────────────────────────────────────────────────────

def _make_ohlcv(bars: int = 30, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=bars, freq="D")
    close = np.full(bars, base)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.full(bars, 1_000_000.0),
        },
        index=idx,
    )


def _make_mock_provider(return_df: pd.DataFrame | None = None) -> MagicMock:
    """MarketDataProvider arayüzünü taklit eden mock."""
    mock = MagicMock(spec=MarketDataProvider)
    mock.fetch_ohlcv.return_value = (
        return_df if return_df is not None else _make_ohlcv()
    )
    mock.get_provider_name.return_value = "mock_provider"
    return mock


def _make_mock_calculator(
    rsi: pd.Series | None = None,
    rvol: pd.Series | None = None,
    atr: pd.Series | None = None,
    bars: int = 30,
) -> MagicMock:
    """TechnicalCalculator'ı taklit eden mock — calculate_* metodları izlenir."""
    mock = MagicMock(spec=TechnicalCalculator)
    mock.calculate_rsi.return_value = (
        rsi if rsi is not None else pd.Series(np.full(bars, 55.0))
    )
    mock.calculate_rvol.return_value = (
        rvol if rvol is not None else pd.Series(np.full(bars, 1.2))
    )
    mock.calculate_atr.return_value = (
        atr if atr is not None else pd.Series(np.full(bars, 2.5))
    )
    return mock


# ── Mutlu Yol (Happy Path) ────────────────────────────────────────────────────

class TestGetMarketAnalysisHappyPath:

    def test_returns_market_analysis_result(self):
        """Başarılı akışta MarketAnalysisResult döner."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        assert isinstance(result, MarketAnalysisResult)
        assert result.symbol == "THYAO"
        assert result.timeframe == "1d"

    def test_provider_fetch_ohlcv_called_with_correct_args(self):
        """provider.fetch_ohlcv() doğru parametrelerle çağrılmalı."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        service.get_market_analysis("GARAN", timeframe="1h")

        provider.fetch_ohlcv.assert_called_once_with(
            symbol="GARAN",
            timeframe="1h",
            start_date=None,
            end_date=None,
        )

    def test_provider_called_before_calculator(self):
        """
        Orkestrasyon sırası: provider ÖNCE, calculator SONRA çağrılmalı.
        Bir mock manager ile çağrı sırası izlenir.
        """
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()

        manager = Mock()
        manager.attach_mock(provider.fetch_ohlcv, "fetch_ohlcv")
        manager.attach_mock(calculator.calculate_rsi, "calculate_rsi")

        service = MarketDataService(provider=provider, calculator=calculator)
        service.get_market_analysis("THYAO", timeframe="1d")

        expected_order = [call.fetch_ohlcv(
            symbol="THYAO", timeframe="1d", start_date=None, end_date=None
        )]
        actual_calls = [c for c in manager.mock_calls if c[0] in ("fetch_ohlcv", "calculate_rsi")]
        assert actual_calls[0][0] == "fetch_ohlcv"
        assert actual_calls[1][0] == "calculate_rsi"

    def test_all_three_indicators_calculated(self):
        """RSI, RVOL, ATR — üçü de hesaplanmalı."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        service.get_market_analysis("THYAO", timeframe="1d")

        calculator.calculate_rsi.assert_called_once()
        calculator.calculate_rvol.assert_called_once()
        calculator.calculate_atr.assert_called_once()

    def test_calculator_receives_correct_close_series(self):
        """calculate_rsi, OHLCV'nin Close sütununu almalı."""
        ohlcv = _make_ohlcv(bars=10, base=150.0)
        provider = _make_mock_provider(return_df=ohlcv)
        calculator = _make_mock_calculator(bars=10)
        service = MarketDataService(provider=provider, calculator=calculator)

        service.get_market_analysis("THYAO", timeframe="1d")

        called_close_series = calculator.calculate_rsi.call_args.args[0]
        pd.testing.assert_series_equal(called_close_series, ohlcv["Close"])

    def test_calculator_receives_correct_volume_series(self):
        """calculate_rvol, OHLCV'nin Volume sütununu almalı."""
        ohlcv = _make_ohlcv(bars=10)
        provider = _make_mock_provider(return_df=ohlcv)
        calculator = _make_mock_calculator(bars=10)
        service = MarketDataService(provider=provider, calculator=calculator)

        service.get_market_analysis("THYAO", timeframe="1d")

        called_volume_series = calculator.calculate_rvol.call_args.args[0]
        pd.testing.assert_series_equal(called_volume_series, ohlcv["Volume"])

    def test_calculator_receives_full_ohlcv_for_atr(self):
        """calculate_atr, tüm OHLCV DataFrame'ini almalı (High/Low/Close gerekir)."""
        ohlcv = _make_ohlcv(bars=10)
        provider = _make_mock_provider(return_df=ohlcv)
        calculator = _make_mock_calculator(bars=10)
        service = MarketDataService(provider=provider, calculator=calculator)

        service.get_market_analysis("THYAO", timeframe="1d")

        called_df = calculator.calculate_atr.call_args.args[0]
        pd.testing.assert_frame_equal(called_df, ohlcv)


# ── DTO Doğruluğu ──────────────────────────────────────────────────────────────

class TestMarketAnalysisResultDTO:

    def test_dto_contains_raw_ohlcv(self):
        ohlcv = _make_ohlcv(bars=15)
        provider = _make_mock_provider(return_df=ohlcv)
        calculator = _make_mock_calculator(bars=15)
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        pd.testing.assert_frame_equal(result.ohlcv, ohlcv)

    def test_dto_contains_computed_indicators(self):
        rsi_series = pd.Series(np.full(20, 65.0))
        rvol_series = pd.Series(np.full(20, 1.8))
        atr_series = pd.Series(np.full(20, 3.2))

        provider = _make_mock_provider(return_df=_make_ohlcv(bars=20))
        calculator = _make_mock_calculator(
            rsi=rsi_series, rvol=rvol_series, atr=atr_series, bars=20
        )
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        pd.testing.assert_series_equal(result.rsi, rsi_series)
        pd.testing.assert_series_equal(result.rvol, rvol_series)
        pd.testing.assert_series_equal(result.atr, atr_series)

    def test_dto_bar_count_property(self):
        provider = _make_mock_provider(return_df=_make_ohlcv(bars=42))
        calculator = _make_mock_calculator(bars=42)
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        assert result.bar_count == 42

    def test_dto_latest_close_property(self):
        ohlcv = _make_ohlcv(bars=10, base=123.45)
        provider = _make_mock_provider(return_df=ohlcv)
        calculator = _make_mock_calculator(bars=10)
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        assert result.latest_close == pytest.approx(float(ohlcv["Close"].iloc[-1]))

    def test_dto_latest_rsi_property(self):
        rsi_series = pd.Series([50.0, 60.0, 70.0])
        provider = _make_mock_provider(return_df=_make_ohlcv(bars=3))
        calculator = _make_mock_calculator(rsi=rsi_series, bars=3)
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        assert result.latest_rsi == 70.0

    def test_dto_latest_rsi_nan_returns_none(self):
        """Son RSI değeri NaN ise latest_rsi None döndürmeli (crash değil)."""
        rsi_series = pd.Series([50.0, 60.0, np.nan])
        provider = _make_mock_provider(return_df=_make_ohlcv(bars=3))
        calculator = _make_mock_calculator(rsi=rsi_series, bars=3)
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        assert result.latest_rsi is None

    def test_dto_is_immutable(self):
        """MarketAnalysisResult frozen dataclass — alan ataması engellenir."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        with pytest.raises((AttributeError, Exception)):
            result.symbol = "DEGISTI"


# ── Hata Dönüşümü: Provider Katmanı Hataları ─────────────────────────────────

class TestProviderErrorTranslation:
    """
    Infrastructure katmanından yükselen ProviderError ailesi hatalar
    MarketDataServiceError'a sarmalanmalı — ham hata çağırana sızmamalı.
    """

    def test_symbol_not_found_wrapped_as_service_error(self):
        provider = _make_mock_provider()
        provider.fetch_ohlcv.side_effect = SymbolNotFoundError(
            symbol="GECERSIZ", provider="yfinance"
        )
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError) as exc_info:
            service.get_market_analysis("GECERSIZ", timeframe="1d")

        assert exc_info.value.symbol == "GECERSIZ"

    def test_provider_unavailable_wrapped_as_service_error(self):
        provider = _make_mock_provider()
        provider.fetch_ohlcv.side_effect = ProviderUnavailableError(
            provider="yfinance", reason="ağ hatası"
        )
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

    def test_no_data_error_wrapped_as_service_error(self):
        provider = _make_mock_provider()
        provider.fetch_ohlcv.side_effect = NoDataError(
            symbol="THYAO", provider="yfinance"
        )
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

    def test_original_exception_preserved_in_cause_chain(self):
        """
        KRİTİK: Orijinal hata kaybolmamalı — __cause__ zincirinde
        korunmalı (loglama/hata ayıklama için).
        """
        provider = _make_mock_provider()
        original_error = ProviderUnavailableError(provider="yfinance", reason="timeout")
        provider.fetch_ohlcv.side_effect = original_error
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError) as exc_info:
            service.get_market_analysis("THYAO", timeframe="1d")

        assert exc_info.value.__cause__ is original_error

    def test_calculator_not_called_when_provider_fails(self):
        """Provider başarısız olursa calculator HİÇ çağrılmamalı."""
        provider = _make_mock_provider()
        provider.fetch_ohlcv.side_effect = ProviderUnavailableError(
            provider="yfinance", reason="hata"
        )
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

        calculator.calculate_rsi.assert_not_called()
        calculator.calculate_rvol.assert_not_called()
        calculator.calculate_atr.assert_not_called()


# ── Hata Dönüşümü: Calculator/Domain Katmanı Hataları ─────────────────────────

class TestCalculationErrorTranslation:
    """
    Domain katmanından yükselen CalculationError ailesi (veya KeyError/
    ValueError gibi beklenmeyen hesaplama hataları) MarketDataServiceError'a
    sarmalanmalı.
    """

    def test_insufficient_data_error_wrapped(self):
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        calculator.calculate_rsi.side_effect = InsufficientDataError(
            required=14, available=3, metric="RSI"
        )
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError) as exc_info:
            service.get_market_analysis("THYAO", timeframe="1d")

        assert exc_info.value.symbol == "THYAO"

    def test_key_error_from_missing_column_wrapped(self):
        """ATR hesabı eksik sütun yüzünden KeyError fırlatırsa sarmalanmalı."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        calculator.calculate_atr.side_effect = KeyError("High")
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

    def test_value_error_wrapped(self):
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        calculator.calculate_rvol.side_effect = ValueError("ma_period pozitif olmalı")
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

    def test_calculation_error_cause_preserved(self):
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        original = InsufficientDataError(required=14, available=2, metric="RSI")
        calculator.calculate_rsi.side_effect = original
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError) as exc_info:
            service.get_market_analysis("THYAO", timeframe="1d")

        assert exc_info.value.__cause__ is original


# ── Boş/None Veri Savunma Kontrolü ────────────────────────────────────────────

class TestEmptyDataDefense:
    """
    Provider sözleşmesi gereği None/boş veri döndürmemeli (kendi içinde
    SymbolNotFoundError/NoDataError fırlatmalı) — ancak servis katmanı
    bunu varsaymaz, kendi savunma kontrolünü de yapar (defense in depth).
    """

    def test_none_dataframe_raises_service_error(self):
        provider = _make_mock_provider()
        provider.fetch_ohlcv.return_value = None
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

    def test_empty_dataframe_raises_service_error(self):
        provider = _make_mock_provider()
        provider.fetch_ohlcv.return_value = pd.DataFrame()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

    def test_calculator_not_called_on_empty_data(self):
        provider = _make_mock_provider()
        provider.fetch_ohlcv.return_value = pd.DataFrame()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

        calculator.calculate_rsi.assert_not_called()


# ── Parametre İletimi (Hard-code Değil, Inject) ───────────────────────────────

class TestParameterInjection:

    def test_default_periods_used_when_not_specified(self):
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        service.get_market_analysis("THYAO", timeframe="1d")

        rsi_kwargs = calculator.calculate_rsi.call_args.kwargs
        rvol_kwargs = calculator.calculate_rvol.call_args.kwargs
        atr_kwargs = calculator.calculate_atr.call_args.kwargs

        assert rsi_kwargs.get("period") == 14
        assert rvol_kwargs.get("ma_period") == 20
        assert atr_kwargs.get("period") == 14

    def test_custom_periods_injected_via_constructor(self):
        """Periyot değerleri constructor'dan override edilebilmeli (hard-code yasak)."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(
            provider=provider,
            calculator=calculator,
            rsi_period=9,
            rvol_ma_period=50,
            atr_period=21,
        )

        service.get_market_analysis("THYAO", timeframe="1d")

        assert calculator.calculate_rsi.call_args.kwargs.get("period") == 9
        assert calculator.calculate_rvol.call_args.kwargs.get("ma_period") == 50
        assert calculator.calculate_atr.call_args.kwargs.get("period") == 21

    def test_start_end_date_forwarded_to_provider(self):
        from datetime import date

        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        start = date(2024, 1, 1)
        end = date(2024, 6, 1)
        service.get_market_analysis("THYAO", timeframe="1d", start_date=start, end_date=end)

        provider.fetch_ohlcv.assert_called_once_with(
            symbol="THYAO", timeframe="1d", start_date=start, end_date=end
        )


# ── Stateless Doğrulaması ──────────────────────────────────────────────────────

class TestStatelessness:
    """
    Servis instance üzerinde hiçbir mutable durum tutulmamalı.
    Aynı instance ile ardışık çağrılar birbirinden tamamen izole olmalı.
    """

    def test_consecutive_calls_do_not_leak_state(self):
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        result1 = service.get_market_analysis("THYAO", timeframe="1d")
        result2 = service.get_market_analysis("GARAN", timeframe="1h")

        assert result1.symbol == "THYAO"
        assert result2.symbol == "GARAN"
        assert result1.symbol != result2.symbol

    def test_failed_call_does_not_affect_subsequent_call(self):
        """Bir çağrı hata fırlatsa bile, sonraki çağrı temiz başlamalı."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        provider.fetch_ohlcv.side_effect = ProviderUnavailableError(
            provider="yfinance", reason="geçici"
        )
        with pytest.raises(MarketDataServiceError):
            service.get_market_analysis("THYAO", timeframe="1d")

        # İkinci çağrıda provider normale dönsün
        provider.fetch_ohlcv.side_effect = None
        provider.fetch_ohlcv.return_value = _make_ohlcv()

        result = service.get_market_analysis("THYAO", timeframe="1d")
        assert isinstance(result, MarketAnalysisResult)

    def test_same_instance_reusable_across_symbols(self):
        """Tek bir servis instance'ı farklı semboller için güvenle kullanılabilir."""
        provider = _make_mock_provider()
        calculator = _make_mock_calculator()
        service = MarketDataService(provider=provider, calculator=calculator)

        symbols = ["THYAO", "GARAN", "AKBNK", "ASELS"]
        results = [service.get_market_analysis(s, timeframe="1d") for s in symbols]

        assert [r.symbol for r in results] == symbols


# ── Entegrasyon: Gerçek TechnicalCalculator ile ──────────────────────────────

class TestIntegrationWithRealCalculator:
    """
    Mock yerine gerçek TechnicalCalculator kullanılarak servis-domain
    entegrasyonunun uçtan uca doğru çalıştığı doğrulanır. Provider hâlâ
    mock'lanır (ağ çağrısı yapılmaz) ama hesaplama gerçektir.
    """

    def test_real_calculator_produces_valid_rsi_range(self):
        ohlcv = _make_ohlcv(bars=50)
        provider = _make_mock_provider(return_df=ohlcv)
        service = MarketDataService(provider=provider, calculator=TechnicalCalculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        valid_rsi = result.rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_real_calculator_rvol_is_dimensionless(self):
        """BUG-11 entegrasyon doğrulaması: servis üzerinden de RVOL boyutsuz davranır."""
        ohlcv = _make_ohlcv(bars=30, base=100.0)
        ohlcv.loc[ohlcv.index[-1], "Volume"] = 2_000_000.0  # son bar 2x hacim

        provider = _make_mock_provider(return_df=ohlcv)
        service = MarketDataService(provider=provider, calculator=TechnicalCalculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        assert result.latest_rvol is not None
        assert result.latest_rvol > 1.0

    def test_real_calculator_with_nan_bars_does_not_crash(self):
        """Aşama 1'in bıraktığı NaN barlar gerçek calculator ile de sorun çıkarmamalı."""
        ohlcv = _make_ohlcv(bars=40)
        ohlcv.iloc[10, ohlcv.columns.get_loc("Close")] = np.nan
        ohlcv.iloc[20, ohlcv.columns.get_loc("Volume")] = np.nan

        provider = _make_mock_provider(return_df=ohlcv)
        service = MarketDataService(provider=provider, calculator=TechnicalCalculator)

        result = service.get_market_analysis("THYAO", timeframe="1d")

        assert result.bar_count == 40
        assert len(result.rsi) == 40
        assert len(result.rvol) == 40
        assert len(result.atr) == 40


# ── Servis Domain İzolasyonu Doğrulaması ──────────────────────────────────────

class TestArchitecturalIsolation:
    """Mimari kısıtların statik/davranışsal doğrulaması."""

    def test_service_does_not_import_concrete_yfinance_adapter(self):
        """
        Servis modülü, somut YFinanceAdapter sınıfını import etmemeli —
        yalnızca soyut MarketDataProvider arayüzünü bilmeli.

        AST tabanlı kontrol kullanılır (yalnızca gerçek `import` ifadeleri
        taranır) — docstring içinde örnek/açıklama amaçlı geçen
        "YFinanceAdapter" kelimesi yanlış pozitif üretmemelidir.
        """
        import ast
        import inspect

        import src.services.market_data_service as svc_module

        source = inspect.getsource(svc_module)
        tree = ast.parse(source)

        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.update(alias.name for alias in node.names)
                if node.module:
                    imported_names.add(node.module)

        assert not any("yfinance" in name.lower() for name in imported_names), (
            f"Servis somut yfinance modülünü/sınıfını import etmemeli. "
            f"Bulunan import'lar: {imported_names}"
        )
        assert not any("YFinanceAdapter" in name for name in imported_names)

    def test_service_accepts_any_provider_implementation(self):
        """
        Herhangi bir MarketDataProvider implementasyonu (yalnızca arayüzü
        sağlayan herhangi bir nesne) servise enjekte edilebilmeli —
        somut tipe bağımlılık olmamalı.
        """

        class CustomFakeProvider(MarketDataProvider):
            def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
                return _make_ohlcv(bars=5)

            def get_provider_name(self) -> str:
                return "custom_fake"

        service = MarketDataService(
            provider=CustomFakeProvider(), calculator=_make_mock_calculator(bars=5)
        )
        result = service.get_market_analysis("XYZ", timeframe="1d")

        assert result.symbol == "XYZ"
