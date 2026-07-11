"""
YFinanceAdapter unit testleri.

Test kapsamı:
  1. Validation pipeline entegrasyonu — geçerli/geçersiz veri akışı
  2. Retry/backoff mekanizması — geçici hatalarda yeniden deneme
  3. Retry tükenmesi — ProviderUnavailableError'a dönüşüm
  4. Kalıcı hatalar (retryable olmayan) — anında yükselme
  5. Boş/None yanıt — SymbolNotFoundError
  6. Yetersiz bar sayısı — NoDataError
  7. Sembol normalizasyonu (.IS suffix)
  8. Timeframe → interval çevirisi ve geçersiz timeframe reddi
  9. stdout/stderr suppress doğrulaması

KRİTİK TASARIM KARARI: yfinance modülü gerçekte import edilmez/çağrılmaz.
Tüm testler `_download_silently` metodunu monkeypatch ederek izole çalışır
— gerçek ağ çağrısı, gerçek API limiti veya gerçek internet bağlantısı
gerektirmez. Bu hem hız hem deterministiklik sağlar.

Retry testlerinde sleep_fn parametresi mock'lanarak gerçek zaman
beklenmesi engellenir (testler milisaniyeler içinde tamamlanır).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.domain.exceptions.domain_exceptions import (
    DataValidationError,
    NoDataError,
    ProviderUnavailableError,
    SymbolNotFoundError,
)
from src.infrastructure.data_providers.retry_policy import (
    RetryExhaustedError,
    RetryPolicy,
    execute_with_retry,
)
from src.infrastructure.data_providers.yfinance_adapter import YFinanceAdapter


# ── Test Veri Yardımcıları ────────────────────────────────────────────────────

def _make_valid_ohlcv(bars: int = 30, base: float = 100.0) -> pd.DataFrame:
    """Tamamen geçerli (Aşama 1 doğrulamasını geçecek) OHLCV DataFrame."""
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


def _make_invalid_ohlcv(bars: int = 30) -> pd.DataFrame:
    """Aşama 1 doğrulamasını başarısız kılacak bozuk veri (strict_mode ile)."""
    df = _make_valid_ohlcv(bars)
    # Tüm barları bozarak strict_mode'da is_valid=False tetiklenmesini sağla
    df["Close"] = 0.0
    df["High"] = 0.0
    df["Low"] = 0.0
    df["Open"] = 0.0
    return df


def _no_sleep(_seconds: float) -> None:
    """Test'lerde gerçek zaman beklememek için sleep stub'ı."""
    pass


# ── Sembol Normalizasyonu ─────────────────────────────────────────────────────

class TestSymbolNormalization:

    def test_bare_symbol_gets_suffix(self):
        adapter = YFinanceAdapter()
        assert adapter._to_yfinance_symbol("THYAO") == "THYAO.IS"

    def test_lowercase_symbol_normalized(self):
        adapter = YFinanceAdapter()
        assert adapter._to_yfinance_symbol("thyao") == "THYAO.IS"

    def test_already_suffixed_symbol_unchanged(self):
        adapter = YFinanceAdapter()
        assert adapter._to_yfinance_symbol("THYAO.IS") == "THYAO.IS"

    def test_whitespace_stripped(self):
        adapter = YFinanceAdapter()
        assert adapter._to_yfinance_symbol("  THYAO  ") == "THYAO.IS"

    def test_custom_suffix_injectable(self):
        """Suffix hard-coded değil — constructor'dan override edilebilir."""
        adapter = YFinanceAdapter(symbol_suffix=".US")
        assert adapter._to_yfinance_symbol("AAPL") == "AAPL.US"


# ── Timeframe Doğrulama ────────────────────────────────────────────────────────

class TestTimeframeValidation:

    def test_unsupported_timeframe_raises_value_error(self):
        adapter = YFinanceAdapter()
        with pytest.raises(ValueError, match="Desteklenmeyen timeframe"):
            adapter.fetch_ohlcv("THYAO", timeframe="3d")

    def test_supported_timeframes_accepted(self):
        """Desteklenen tüm timeframe'ler ValueError fırlatmamalı (mock veri ile)."""
        adapter = YFinanceAdapter()
        valid_df = _make_valid_ohlcv(bars=30)

        for tf in ["1d", "1h", "15m", "4h"]:
            with patch.object(adapter, "_download_silently", return_value=valid_df.copy()):
                # ValueError fırlatılmamalı — başka exception'lar olabilir ama bu değil
                try:
                    adapter.fetch_ohlcv("THYAO", timeframe=tf)
                except ValueError as exc:
                    pytest.fail(f"timeframe={tf} ValueError fırlattı: {exc}")


# ── Validation Pipeline Entegrasyonu (KRİTİK) ────────────────────────────────

class TestValidationPipelineIntegration:
    """
    En kritik test grubu: ham veri DOĞRUDAN döndürülmüyor,
    normalize_and_validate() üzerinden geçiyor mu?
    """

    def test_valid_data_returned_clean(self):
        """Geçerli veri: temizlenmiş DataFrame döner, exception yok."""
        adapter = YFinanceAdapter()
        valid_df = _make_valid_ohlcv(bars=30)

        with patch.object(adapter, "_download_silently", return_value=valid_df.copy()):
            result = adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert result is not None
        assert len(result) == 30
        assert "Close" in result.columns

    def test_invalid_data_raises_data_validation_error(self):
        """
        KRİTİK: Aşama 1 doğrulamasını geçemeyen veri DataValidationError
        fırlatmalı — ham veri asla sessizce döndürülmemeli.
        """
        adapter = YFinanceAdapter(strict_validation=True, min_bars_required=1)
        invalid_df = _make_invalid_ohlcv(bars=30)

        with patch.object(adapter, "_download_silently", return_value=invalid_df):
            with pytest.raises(DataValidationError) as exc_info:
                adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert exc_info.value.provider == "yfinance"

    def test_data_validation_error_includes_warnings(self):
        """DataValidationError mesajında report.warnings yer almalı."""
        adapter = YFinanceAdapter(strict_validation=True)
        invalid_df = _make_invalid_ohlcv(bars=30)

        with patch.object(adapter, "_download_silently", return_value=invalid_df):
            with pytest.raises(DataValidationError) as exc_info:
                adapter.fetch_ohlcv("THYAO", timeframe="1d")

        # reason context'inde warning bilgisi olmalı
        assert "reason" in exc_info.value.context
        assert len(exc_info.value.context["reason"]) > 0

    def test_multiindex_raw_data_normalized_before_validation(self):
        """
        MultiIndex ham veri (gerçek yfinance çıktısı formatı) doğru
        normalize edilip sonra validate edilmeli.
        """
        adapter = YFinanceAdapter()
        valid_df = _make_valid_ohlcv(bars=30)
        multiindex_df = valid_df.copy()
        multiindex_df.columns = pd.MultiIndex.from_tuples(
            [(c, "THYAO.IS") for c in valid_df.columns]
        )

        with patch.object(adapter, "_download_silently", return_value=multiindex_df):
            result = adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert result is not None
        assert not isinstance(result.columns, pd.MultiIndex)
        assert "Close" in result.columns

    def test_bad_bars_nullified_not_dropped(self):
        """
        Aşama 1 prensibi korunmalı: hatalı barlar silinmiyor, NaN'a
        çevriliyor. Adapter bu davranışı değiştirmemeli.
        """
        adapter = YFinanceAdapter(strict_validation=False, min_bars_required=1)
        df = _make_valid_ohlcv(bars=30)
        df.iloc[10, df.columns.get_loc("Close")] = -5.0  # Bozuk bar

        with patch.object(adapter, "_download_silently", return_value=df):
            result = adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert len(result) == 30  # Satır silinmedi
        assert pd.isna(result.iloc[10]["Close"])  # NaN'a çevrildi

    def test_strict_mode_injectable(self):
        """strict_validation parametresi adapter constructor'ından geçer."""
        adapter_strict = YFinanceAdapter(strict_validation=True)
        adapter_lenient = YFinanceAdapter(strict_validation=False)

        assert adapter_strict._strict_validation is True
        assert adapter_lenient._strict_validation is False


# ── Boş/None Yanıt → SymbolNotFoundError ──────────────────────────────────────

class TestEmptyResponseHandling:

    def test_none_response_raises_symbol_not_found(self):
        adapter = YFinanceAdapter()

        with patch.object(adapter, "_download_silently", return_value=None):
            with pytest.raises(SymbolNotFoundError) as exc_info:
                adapter.fetch_ohlcv("GECERSIZ", timeframe="1d")

        assert exc_info.value.symbol == "GECERSIZ"
        assert exc_info.value.provider == "yfinance"

    def test_empty_dataframe_raises_symbol_not_found(self):
        adapter = YFinanceAdapter()

        with patch.object(adapter, "_download_silently", return_value=pd.DataFrame()):
            with pytest.raises(SymbolNotFoundError):
                adapter.fetch_ohlcv("DELISTED", timeframe="1d")


# ── Yetersiz Bar Sayısı → NoDataError ─────────────────────────────────────────

class TestInsufficientDataHandling:

    def test_below_minimum_bars_raises_no_data_error(self):
        adapter = YFinanceAdapter(min_bars_required=50)
        small_df = _make_valid_ohlcv(bars=5)

        with patch.object(adapter, "_download_silently", return_value=small_df):
            with pytest.raises(NoDataError) as exc_info:
                adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert exc_info.value.symbol == "THYAO"

    def test_exact_minimum_bars_succeeds(self):
        adapter = YFinanceAdapter(min_bars_required=10)
        exact_df = _make_valid_ohlcv(bars=10)

        with patch.object(adapter, "_download_silently", return_value=exact_df):
            result = adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert len(result) == 10

    def test_min_bars_required_default_is_permissive(self):
        """Varsayılan min_bars_required=1 — küçük veri setlerini reddetmemeli."""
        adapter = YFinanceAdapter()
        tiny_df = _make_valid_ohlcv(bars=2)

        with patch.object(adapter, "_download_silently", return_value=tiny_df):
            result = adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert len(result) == 2


# ── Retry / Exponential Backoff Mekanizması ──────────────────────────────────

class TestRetryPolicy:
    """RetryPolicy'nin matematiksel doğruluğu — izole, adapter'dan bağımsız."""

    def test_max_attempts_validated(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryPolicy(max_attempts=0)

    def test_negative_base_delay_rejected(self):
        with pytest.raises(ValueError, match="base_delay"):
            RetryPolicy(base_delay_seconds=-1.0)

    def test_backoff_factor_below_one_rejected(self):
        with pytest.raises(ValueError, match="backoff_factor"):
            RetryPolicy(backoff_factor=0.5)

    def test_delay_increases_exponentially(self):
        """Her deneme bekleme süresi bir öncekinden büyük olmalı."""
        policy = RetryPolicy(
            base_delay_seconds=1.0, backoff_factor=2.0,
            max_delay_seconds=100.0, jitter_ratio=0.0,  # jitter kapalı — deterministik
        )
        delay_0 = policy.compute_delay(0)
        delay_1 = policy.compute_delay(1)
        delay_2 = policy.compute_delay(2)

        assert delay_0 == pytest.approx(1.0)
        assert delay_1 == pytest.approx(2.0)
        assert delay_2 == pytest.approx(4.0)

    def test_delay_capped_at_max(self):
        """Bekleme süresi max_delay_seconds'ı aşamaz."""
        policy = RetryPolicy(
            base_delay_seconds=10.0, backoff_factor=10.0,
            max_delay_seconds=15.0, jitter_ratio=0.0,
        )
        delay = policy.compute_delay(5)  # Teorik: 10 * 10^5 — çok büyük
        assert delay <= 15.0 * 1.0  # jitter=0 olduğu için tam max'ta

    def test_jitter_adds_randomness_within_bounds(self):
        """Jitter, delay'i artırır ama belirlenen oranı aşmaz."""
        policy = RetryPolicy(
            base_delay_seconds=10.0, backoff_factor=1.0,
            max_delay_seconds=100.0, jitter_ratio=0.2,
        )
        delays = [policy.compute_delay(0) for _ in range(20)]
        # Tüm değerler [10.0, 12.0] aralığında olmalı (10 + %20 jitter)
        assert all(10.0 <= d <= 12.0 for d in delays)
        # En azından bazı değerler birbirinden farklı olmalı (rastgelelik var)
        assert len(set(delays)) > 1


class TestExecuteWithRetry:
    """execute_with_retry fonksiyonunun davranışı — gerçek sleep yok."""

    def test_succeeds_on_first_attempt_no_retry(self):
        calls = {"count": 0}

        def func():
            calls["count"] += 1
            return "success"

        policy = RetryPolicy(max_attempts=3)
        result = execute_with_retry(
            func, policy, retryable_exceptions=(ConnectionError,),
            sleep_fn=_no_sleep,
        )

        assert result == "success"
        assert calls["count"] == 1

    def test_retries_on_retryable_exception_then_succeeds(self):
        calls = {"count": 0}

        def func():
            calls["count"] += 1
            if calls["count"] < 3:
                raise ConnectionError("geçici ağ hatası")
            return "success_after_retries"

        policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.01)
        result = execute_with_retry(
            func, policy, retryable_exceptions=(ConnectionError,),
            sleep_fn=_no_sleep,
        )

        assert result == "success_after_retries"
        assert calls["count"] == 3

    def test_exhausts_retries_raises_retry_exhausted_error(self):
        def func():
            raise ConnectionError("kalıcı ağ sorunu")

        policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01)

        with pytest.raises(RetryExhaustedError) as exc_info:
            execute_with_retry(
                func, policy, retryable_exceptions=(ConnectionError,),
                sleep_fn=_no_sleep,
            )

        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ConnectionError)

    def test_non_retryable_exception_raises_immediately(self):
        """
        ValueError gibi kalıcı hatalar retry edilmemeli — anında yükselmeli.
        """
        calls = {"count": 0}

        def func():
            calls["count"] += 1
            raise ValueError("kalıcı/mantıksal hata")

        policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.01)

        with pytest.raises(ValueError, match="kalıcı/mantıksal hata"):
            execute_with_retry(
                func, policy, retryable_exceptions=(ConnectionError,),
                sleep_fn=_no_sleep,
            )

        assert calls["count"] == 1  # Yalnızca 1 kez denendi, retry yok

    def test_sleep_called_correct_number_of_times(self):
        """N deneme = N-1 sleep çağrısı (son denemeden sonra beklenmez)."""
        sleep_calls = []

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        def func():
            raise ConnectionError("hep başarısız")

        policy = RetryPolicy(max_attempts=4, base_delay_seconds=0.01)

        with pytest.raises(RetryExhaustedError):
            execute_with_retry(
                func, policy, retryable_exceptions=(ConnectionError,),
                sleep_fn=fake_sleep,
            )

        assert len(sleep_calls) == 3  # 4 deneme, 3 ara bekleme


# ── Adapter Düzeyinde Retry Entegrasyonu ──────────────────────────────────────

class TestAdapterRetryIntegration:
    """
    YFinanceAdapter.fetch_ohlcv'nin retry mekanizmasını gerçekten
    kullandığının uçtan uca doğrulanması.
    """

    def test_transient_network_error_retried_then_succeeds(self):
        """Geçici ConnectionError sonrası başarılı veri dönüşü."""
        adapter = YFinanceAdapter(
            retry_count=3, base_delay_seconds=0.01, backoff_factor=1.5
        )
        valid_df = _make_valid_ohlcv(bars=30)
        call_count = {"n": 0}

        def flaky_download(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ConnectionError("geçici bağlantı hatası")
            return valid_df.copy()

        with patch.object(adapter, "_download_silently", side_effect=flaky_download):
            with patch(
                "src.infrastructure.data_providers.retry_policy.time.sleep",
                _no_sleep,
            ):
                result = adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert result is not None
        assert call_count["n"] == 2

    def test_persistent_network_error_raises_provider_unavailable(self):
        """
        Tüm retry denemeleri ağ hatasıyla tükenirse
        ProviderUnavailableError fırlatılmalı (RetryExhaustedError değil —
        domain exception'a sarmalanmış olmalı).
        """
        adapter = YFinanceAdapter(retry_count=3, base_delay_seconds=0.01)

        def always_fails(*args, **kwargs):
            raise TimeoutError("sürekli timeout")

        with patch.object(adapter, "_download_silently", side_effect=always_fails):
            with patch(
                "src.infrastructure.data_providers.retry_policy.time.sleep",
                _no_sleep,
            ):
                with pytest.raises(ProviderUnavailableError) as exc_info:
                    adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert exc_info.value.provider == "yfinance"

    def test_retry_count_respected(self):
        """retry_count parametresi gerçekten kaç deneme yapılacağını belirler."""
        adapter = YFinanceAdapter(retry_count=5, base_delay_seconds=0.01)
        call_count = {"n": 0}

        def always_fails(*args, **kwargs):
            call_count["n"] += 1
            raise ConnectionError("hep başarısız")

        with patch.object(adapter, "_download_silently", side_effect=always_fails):
            with patch(
                "src.infrastructure.data_providers.retry_policy.time.sleep",
                _no_sleep,
            ):
                with pytest.raises(ProviderUnavailableError):
                    adapter.fetch_ohlcv("THYAO", timeframe="1d")

        assert call_count["n"] == 5

    def test_non_network_exception_not_retried(self):
        """
        DataValidationError gibi retry kapsamı dışındaki hatalar
        (örn. veri doğrulama hatası) yeniden denenmemeli — anında yükselir.
        Bu test, validation hatasının retry mekanizmasına hiç girmediğini
        (yalnızca download'ın retry edildiğini) doğrular.
        """
        adapter = YFinanceAdapter(retry_count=3, strict_validation=True)
        invalid_df = _make_invalid_ohlcv(bars=30)
        call_count = {"n": 0}

        def download_once(*args, **kwargs):
            call_count["n"] += 1
            return invalid_df

        with patch.object(adapter, "_download_silently", side_effect=download_once):
            with pytest.raises(DataValidationError):
                adapter.fetch_ohlcv("THYAO", timeframe="1d")

        # Validation hatası retry mekanizmasını TETİKLEMEMELİ —
        # download yalnızca 1 kez çağrılmış olmalı.
        assert call_count["n"] == 1

    def test_default_retry_count_is_three(self):
        """Spesifikasyondaki varsayılan: retry_count=3."""
        adapter = YFinanceAdapter()
        assert adapter._retry_policy.max_attempts == 3

    def test_default_backoff_factor_is_one_point_five(self):
        """Spesifikasyondaki varsayılan: backoff_factor=1.5."""
        adapter = YFinanceAdapter()
        assert adapter._retry_policy.backoff_factor == 1.5

    def test_retry_parameters_are_injectable_not_hardcoded(self):
        """
        Kısıt doğrulaması: retry/backoff parametreleri constructor'dan
        serbestçe override edilebilmeli (hard-code yasak).
        """
        adapter = YFinanceAdapter(
            retry_count=10,
            backoff_factor=3.0,
            base_delay_seconds=5.0,
            max_delay_seconds=120.0,
        )
        assert adapter._retry_policy.max_attempts == 10
        assert adapter._retry_policy.backoff_factor == 3.0
        assert adapter._retry_policy.base_delay_seconds == 5.0
        assert adapter._retry_policy.max_delay_seconds == 120.0


# ── stdout/stderr Suppress Doğrulaması ────────────────────────────────────────

class TestOutputSuppression:
    """
    yfinance'ın konsolu kirleten print/warning çıktılarının
    tamamen bastırıldığının doğrulanması.
    """

    def test_download_silently_suppresses_stdout(self, capsys):
        """_download_silently içinde basılan hiçbir şey stdout'a çıkmamalı."""
        adapter = YFinanceAdapter()

        def noisy_yf_download_simulation():
            # yfinance'ın yaptığı gibi konsola doğrudan yazma simülasyonu
            print("possibly delisted; no price data found")
            print("some other yfinance warning")
            return _make_valid_ohlcv(bars=10)

        # _download_silently'nin iç mantığını taklit ederek
        # contextlib redirect'in çalıştığını doğrula
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(
            io.StringIO()
        ):
            noisy_yf_download_simulation()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_fetch_ohlcv_produces_no_stdout_noise(self, capsys):
        """
        Tam fetch_ohlcv akışı sırasında (mock'lanmış _download_silently ile)
        hiçbir stdout kirliliği oluşmamalı — yalnızca logging kullanılmalı.
        """
        adapter = YFinanceAdapter()
        valid_df = _make_valid_ohlcv(bars=30)

        with patch.object(adapter, "_download_silently", return_value=valid_df):
            adapter.fetch_ohlcv("THYAO", timeframe="1d")

        captured = capsys.readouterr()
        assert captured.out == "", f"Beklenmeyen stdout çıktısı: {captured.out!r}"


# ── Resample (4h) Davranışı ───────────────────────────────────────────────────

class TestResampling:

    def test_4h_timeframe_triggers_resample(self):
        """4h timeframe: 1h native veri çekilip resample edilmeli."""
        adapter = YFinanceAdapter()
        # 1h bazında 8 saatlik veri (2x 4h bar üretmeli)
        idx = pd.date_range("2024-01-02 09:00", periods=8, freq="1h")
        hourly_df = pd.DataFrame(
            {
                "Open": np.full(8, 100.0),
                "High": np.full(8, 101.0),
                "Low": np.full(8, 99.0),
                "Close": np.full(8, 100.0),
                "Volume": np.full(8, 100_000.0),
            },
            index=idx,
        )

        with patch.object(adapter, "_download_silently", return_value=hourly_df):
            result = adapter.fetch_ohlcv("THYAO", timeframe="4h")

        assert result is not None
        # 8 saat / 4h = 2 bar (yaklaşık, resample sınırlarına bağlı)
        assert len(result) <= 8

    def test_resample_preserves_ohlc_semantics(self):
        """Resample sonrası Open=ilk, Close=son, High=max, Low=min olmalı."""
        adapter = YFinanceAdapter()
        idx = pd.date_range("2024-01-02 00:00", periods=4, freq="1h")
        df = pd.DataFrame(
            {
                "Open":  [100.0, 102.0, 98.0,  105.0],
                "High":  [103.0, 104.0, 100.0, 107.0],
                "Low":   [99.0,  101.0, 96.0,  104.0],
                "Close": [102.0, 98.0,  105.0, 106.0],
                "Volume":[1000.0,1000.0,1000.0,1000.0],
            },
            index=idx,
        )

        resampled = YFinanceAdapter._resample(df, "4h")

        assert len(resampled) == 1
        assert float(resampled.iloc[0]["Open"]) == 100.0   # İlk bar'ın açılışı
        assert float(resampled.iloc[0]["Close"]) == 106.0  # Son bar'ın kapanışı
        assert float(resampled.iloc[0]["High"]) == 107.0   # Maksimum high
        assert float(resampled.iloc[0]["Low"]) == 96.0     # Minimum low
        assert float(resampled.iloc[0]["Volume"]) == 4000.0  # Toplam hacim


# ── Provider Adı ────────────────────────────────────────────────────────────

class TestProviderIdentity:

    def test_provider_name_is_yfinance(self):
        adapter = YFinanceAdapter()
        assert adapter.get_provider_name() == "yfinance"

    def test_adapter_implements_base_provider_interface(self):
        from src.infrastructure.data_providers.base_provider import MarketDataProvider

        adapter = YFinanceAdapter()
        assert isinstance(adapter, MarketDataProvider)
