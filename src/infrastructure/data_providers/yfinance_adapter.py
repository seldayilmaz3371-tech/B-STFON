"""
yfinance veri sağlayıcı adapter'ı.

Legacy kaynak: BistKokpit V23.5 _yf_get() fonksiyonunun production-grade
dönüşümü.

Veri Akışı (KRİTİK — sözleşme):
  1. yfinance.download() ile ham veri çekilir (stdout/stderr tamamen
     bastırılmış halde — yfinance'ın "possibly delisted" gibi konsol
     kirletici uyarıları sistemin loglarına asla karışmaz).
  2. Ham veri DOĞRUDAN döndürülmez. Aşama 1'de yazılan
     normalize_and_validate() pipeline'ından geçirilir
     (MultiIndex flatten + OHLCV bütünlük kontrolü).
  3. report.is_valid == False ise DataValidationError fırlatılır
     (mesajında report.warnings yer alır).
  4. Yalnızca is_valid == True ise temizlenmiş DataFrame döndürülür.

Dayanıklılık:
  Ağ kaynaklı geçici hatalar (timeout, connection error) exponential
  backoff ile retry edilir (retry_policy.py). Retry sayısı ve backoff
  parametreleri constructor'dan inject edilir — hard-coded değer yoktur.

Domain İzolasyonu (kesin kural):
  Bu sınıf src/domain altındaki hiçbir hesaplama sınıfını
  (TechnicalCalculator vb.) import etmez veya çağırmaz. Sorumluluğu
  yalnızca "temiz, doğrulanmış ham veri" sağlamaktır.
"""

from __future__ import annotations

import contextlib
import io
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.domain.exceptions.domain_exceptions import (
    DataValidationError,
    NoDataError,
    ProviderUnavailableError,
    SymbolNotFoundError,
)
from src.infrastructure.data_providers.base_provider import MarketDataProvider
from src.infrastructure.data_providers.retry_policy import (
    RetryExhaustedError,
    RetryPolicy,
    execute_with_retry,
)
from src.infrastructure.logging_config import get_logger
from src.infrastructure.validators.market_data_validator import normalize_and_validate

logger = get_logger(__name__)

# yfinance'da ağ/geçici hata olarak kabul edilen exception tipleri.
# Bunlar retry'a tabidir. ValueError gibi "kalıcı" hatalar
# (geçersiz sembol formatı vb.) retry edilmez.
#
# DÜZELTME (bu turda, GERÇEK bir test/araştırmayla bulundu — daha önce
# xfail ile ertelenmiş bilinen bir sınıflandırma hatası): yfinance,
# ağ hatalarını (HTTP 403, bağlantı sorunları) KENDİ İÇİNDE yutup
# EXCEPTION FIRLATMADAN boş bir DataFrame döndürüyor — bu, retry
# mekanizmasının HİÇ tetiklenmemesine yol açıyordu (yalnızca exception'lar
# retry ediliyordu, "başarılı ama boş sonuç" değil). Bu proje boyunca
# İKİ farklı yaklaşım denendi ve İKİSİ DE başarısız oldu:
#   1) yf.shared._ERRORS sözlüğünden gerçek hata nedenini okumak —
#      bu senaryoda dict BOŞ kalıyor (doğrudan test edilerek doğrulandı).
#   2) Bastırılan stderr metnini örüntü eşleştirmeyle sınıflandırmak
#      ("HTTP Error" vs "possibly delisted") — GÜVENİLMEZ: "possibly
#      delisted" mesajı bazen AĞ HATASI durumunda da ortaya çıkıyor
#      (gerçek AppTest loglarında gözlemlendi).
#
# KARAR: Sahte kesinlik iddia eden kırılgan bir heuristik YAZILMADI.
# Bunun yerine _EmptyOhlcvResponseError, boş yanıtı RETRY EDİLEBİLİR
# hale getiriyor (geçici ağ sorunlarını GERÇEKTEN çözer) — ve retry'lar
# tükendiğinde hata mesajı, "sembol kesin bulunamadı" yerine belirsizliği
# DÜRÜSTÇE yansıtacak şekilde güncellendi (bkz. fetch_ohlcv'deki
# except RetryExhaustedError bloğu).
class _EmptyOhlcvResponseError(Exception):
    """
    İç kullanım — yfinance boş/None DataFrame döndürdüğünde fırlatılır,
    böylece execute_with_retry() bunu bir "gerçek hata" gibi retry
    edebilir. Domain katmanına SIZMAZ (her zaman burada, adapter
    içinde yakalanıp domain exception'larına çevrilir).
    """

    def __init__(self, captured_diagnostic_text: str = "") -> None:
        self.captured_diagnostic_text = captured_diagnostic_text
        super().__init__(captured_diagnostic_text or "yfinance boş yanıt döndürdü")


_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,  # Soket düzeyi ağ hataları genelde OSError alt sınıfıdır
    _EmptyOhlcvResponseError,
)

# BIST sembolleri için yfinance suffix'i.
# Config-driven yapıya geçişte settings'ten okunacak; şimdilik
# adapter constructor'ından override edilebilir (hard-code DEĞİL).
_DEFAULT_BIST_SUFFIX = ".IS"

# timeframe → yfinance interval eşlemesi.
# Adapter'a özgü bir çeviri katmanı: domain "1d"/"1h"/"15m"/"4h" gibi
# kendi sözlüğünü kullanır, yfinance'ın "1d"/"60m" gibi kendi
# adlandırmasına burada çevrilir.
_TIMEFRAME_TO_YF_INTERVAL: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "60m",   # yfinance native 4h sağlamaz — 1h çekilip resample edilir
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

# 4h gibi yfinance'da doğrudan desteklenmeyen timeframe'ler için
# hangi native interval'dan resample yapılacağını belirtir.
_RESAMPLE_SOURCE: dict[str, str] = {
    "4h": "1h",
}

_RESAMPLE_RULE: dict[str, str] = {
    "4h": "4h",
}


class YFinanceAdapter(MarketDataProvider):
    """
    yfinance kütüphanesi üzerinden BIST hisse senedi verisi sağlayan adapter.

    Kullanım:
        adapter = YFinanceAdapter(retry_count=3, backoff_factor=1.5)
        df = adapter.fetch_ohlcv("THYAO", timeframe="1d",
                                  start_date=date(2024,1,1), end_date=date(2024,6,1))

    Constructor parametreleri tamamen dependency injection'a açıktır;
    hiçbir retry/timeout değeri kod içinde sabitlenmemiştir.
    """

    def __init__(
        self,
        retry_count: int = 3,
        backoff_factor: float = 1.5,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        symbol_suffix: str = _DEFAULT_BIST_SUFFIX,
        strict_validation: bool = False,
        min_bars_required: int = 1,
    ) -> None:
        """
        Args:
            retry_count: Maksimum deneme sayısı (ilk deneme dahil).
            backoff_factor: Her denemede bekleme süresinin çarpanı.
            base_delay_seconds: İlk retry'da beklenecek temel süre.
            max_delay_seconds: Bekleme süresinin üst sınırı.
            symbol_suffix: BIST sembolüne eklenecek borsa son eki
                            (varsayılan ".IS"). Test'lerde veya farklı
                            borsalar için override edilebilir.
            strict_validation: True ise MarketDataValidator'da kötü bar
                                oranı %5'i aşarsa is_valid=False kabul edilir
                                (bkz. Aşama 1 — strict_mode parametresi).
            min_bars_required: Geçerli sayılması için minimum bar sayısı.
                                Bundan az veri dönerse NoDataError fırlatılır.
        """
        self._retry_policy = RetryPolicy(
            max_attempts=retry_count,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            backoff_factor=backoff_factor,
        )
        self._symbol_suffix = symbol_suffix
        self._strict_validation = strict_validation
        self._min_bars_required = min_bars_required

    def get_provider_name(self) -> str:
        return "yfinance"

    # ── Public API ───────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> pd.DataFrame:
        """
        BIST sembolü için OHLCV verisi çek, doğrula ve döndür.

        Args:
            symbol: Normalize sembol (ör: "THYAO"). ".IS" eklenmemiş olmalı —
                    suffix bu adapter tarafından otomatik eklenir.
            timeframe: "1d", "1h", "15m", "4h" gibi desteklenen aralıklardan biri.
            start_date: Başlangıç tarihi (None ise yfinance varsayılan period kullanılır).
            end_date: Bitiş tarihi (None ise bugüne kadar).

        Returns:
            pd.DataFrame: Validate edilmiş, temizlenmiş OHLCV verisi.

        Raises:
            SymbolNotFoundError: yfinance sembolü tanımıyor / hiç veri yok.
            NoDataError: İstenen aralıkta yetersiz veri (min_bars_required altı).
            ProviderUnavailableError: Tüm retry denemeleri ağ hatasıyla tükendi.
            DataValidationError: Çekilen veri Aşama 1 doğrulamasını geçemedi.
            ValueError: Desteklenmeyen timeframe.
        """
        if timeframe not in _TIMEFRAME_TO_YF_INTERVAL:
            raise ValueError(
                f"Desteklenmeyen timeframe: {timeframe!r}. "
                f"Geçerli değerler: {sorted(_TIMEFRAME_TO_YF_INTERVAL)}"
            )

        yf_symbol = self._to_yfinance_symbol(symbol)
        yf_interval = _TIMEFRAME_TO_YF_INTERVAL[timeframe]

        logger.debug(
            "fetch_ohlcv_started",
            symbol=symbol,
            yf_symbol=yf_symbol,
            timeframe=timeframe,
            yf_interval=yf_interval,
            start_date=str(start_date) if start_date else None,
            end_date=str(end_date) if end_date else None,
        )

        try:
            raw_df = execute_with_retry(
                func=lambda: self._download_silently(
                    yf_symbol, yf_interval, start_date, end_date
                ),
                policy=self._retry_policy,
                retryable_exceptions=_RETRYABLE_EXCEPTIONS,
                operation_name=f"yfinance.download({yf_symbol})",
            )
        except RetryExhaustedError as exc:
            if isinstance(exc.last_exception, _EmptyOhlcvResponseError):
                # DÜZELTME (bu turda): Artık RETRY EDİLMİŞ (geçici ağ
                # sorunları bu noktaya gelmeden ÖNCE çözülmüş olabilir)
                # ama hâlâ boş sonuç alınıyor. KÖK NEDEN BELİRSİZ —
                # gerçekten geçersiz bir sembol MÜ, yoksa sağlayıcıya
                # KALICI OLARAK mı ulaşılamıyor, GÜVENİLİR ŞEKİLDE
                # AYIRT EDİLEMİYOR (bkz. _RETRYABLE_EXCEPTIONS
                # docstring'indeki araştırma notu — iki yaklaşım
                # denenip İKİSİ DE güvenilmez bulundu). Mesaj bu
                # belirsizliği DÜRÜSTÇE yansıtıyor.
                logger.warning(
                    "fetch_ohlcv_empty_after_retries",
                    symbol=symbol,
                    yf_symbol=yf_symbol,
                    attempts=exc.attempts,
                    diagnostic=exc.last_exception.captured_diagnostic_text[:500],
                )
                raise SymbolNotFoundError(
                    symbol=symbol,
                    provider=self.get_provider_name(),
                    note=(
                        f"{exc.attempts} denemeden sonra veri alınamadı. "
                        "Sembol geçersiz OLABİLİR VEYA sağlayıcıya şu an "
                        "ulaşılamıyor OLABİLİR — bu ikisi güvenilir şekilde "
                        "ayırt edilemiyor (yfinance'in kendi sınırlaması)."
                    ),
                ) from exc

            logger.error(
                "fetch_ohlcv_provider_unavailable",
                symbol=symbol,
                attempts=exc.attempts,
                last_error=str(exc.last_exception),
            )
            raise ProviderUnavailableError(
                provider=self.get_provider_name(),
                reason=f"{exc.attempts} deneme sonrası ulaşılamadı: {exc.last_exception}",
                symbol=symbol,
            ) from exc

        # Savunmacı ikinci kontrol — normal akışta BURAYA ULAŞILMAMALI
        # (boş yanıt artık _download_silently içinde exception'a
        # çevriliyor, yukarıdaki except bloğu tarafından yakalanıyor).
        # Yine de execute_with_retry'nin gelecekte değişebilecek iç
        # davranışına karşı bir güvenlik ağı olarak KORUNUYOR.
        if raw_df is None or raw_df.empty:  # pragma: no cover
            logger.warning(
                "fetch_ohlcv_empty_response_unexpected_path",
                symbol=symbol,
                yf_symbol=yf_symbol,
            )
            raise SymbolNotFoundError(symbol=symbol, provider=self.get_provider_name())

        # 4h gibi native desteklenmeyen timeframe'ler için resample
        if timeframe in _RESAMPLE_SOURCE:
            raw_df = self._resample(raw_df, timeframe)

        # ── KRİTİK: Aşama 1 doğrulama pipeline'ı ────────────────────────────
        clean_df, report = normalize_and_validate(
            raw_df,
            symbol=symbol,
            strict_mode=self._strict_validation,
        )

        if not report.is_valid:
            logger.error(
                "fetch_ohlcv_validation_failed",
                symbol=symbol,
                warnings=report.warnings,
                bad_bar_count=report.bad_bar_count,
                duplicate_count=report.duplicate_count,
            )
            raise DataValidationError(
                provider=self.get_provider_name(),
                reason="; ".join(report.warnings) or "Bilinmeyen doğrulama hatası",
                symbol=symbol,
            )

        if clean_df is None or len(clean_df) < self._min_bars_required:
            actual_count = 0 if clean_df is None else len(clean_df)
            logger.warning(
                "fetch_ohlcv_insufficient_bars",
                symbol=symbol,
                required=self._min_bars_required,
                actual=actual_count,
            )
            raise NoDataError(symbol=symbol, provider=self.get_provider_name())

        logger.info(
            "fetch_ohlcv_succeeded",
            symbol=symbol,
            timeframe=timeframe,
            bar_count=len(clean_df),
            bad_bar_count=report.bad_bar_count,
        )

        return clean_df

    # ── Private Yardımcılar ──────────────────────────────────────────────────

    def _to_yfinance_symbol(self, symbol: str) -> str:
        """THYAO → THYAO.IS dönüşümü (zaten suffix'liyse dokunmaz)."""
        clean = symbol.upper().strip()
        if clean.endswith(self._symbol_suffix):
            return clean
        return clean + self._symbol_suffix

    def _download_silently(
        self,
        yf_symbol: str,
        yf_interval: str,
        start_date: date | datetime | None,
        end_date: date | datetime | None,
    ) -> pd.DataFrame:
        """
        yfinance.download()'ı stdout/stderr tamamen bastırılmış şekilde çalıştır.

        yfinance kütüphanesi "possibly delisted", "no price data found" gibi
        mesajları doğrudan konsola (print) basar — bu, sistemin yapılandırılmış
        loglama disiplinini bozar. contextlib.redirect_stdout/stderr ile bu
        çıktılar tamamen yutulur ama İÇERİĞİ ATILMAZ — bkz. aşağıdaki
        DÜZELTME notu.

        DÜZELTME (bu turda): Önceden boş/None DataFrame SESSİZCE
        döndürülüyordu — çağıran kod bunu doğrudan SymbolNotFoundError'a
        çeviriyordu, retry mekanizması HİÇ devreye girmiyordu (yalnızca
        exception'lar retry ediliyor). Artık boş yanıt _EmptyOhlcvResponseError
        olarak FIRLATILIYOR — bu, execute_with_retry() tarafından retry
        edilebilir hale geliyor (geçici ağ sorunları GERÇEKTEN çözülüyor,
        yalnızca "sessizce kabul ediliyor" değil). Bastırılan stderr metni,
        hata mesajına context olarak EKLENİYOR (tamamen ATILMIYOR) — bu,
        sınıflandırma için GÜVENİLİR bir sinyal DEĞİL (bkz. yukarıdaki
        _RETRYABLE_EXCEPTIONS docstring'i — "possibly delisted" ağ
        hatasında da çıkabiliyor) ama en azından DEBUG/log seviyesinde
        insan tarafından okunabilir bir ipucu sağlıyor.
        """
        download_kwargs: dict[str, Any] = {
            "interval": yf_interval,
            "progress": False,
            "auto_adjust": False,
        }

        if start_date is not None or end_date is not None:
            if start_date is not None:
                download_kwargs["start"] = start_date
            if end_date is not None:
                download_kwargs["end"] = end_date
        else:
            # Hiç tarih verilmediyse makul bir varsayılan period kullan
            download_kwargs["period"] = self._default_period_for_interval(yf_interval)

        import yfinance as yf

        captured_stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(captured_stderr),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            raw = yf.download(yf_symbol, **download_kwargs)

        if raw is None or raw.empty:
            raise _EmptyOhlcvResponseError(captured_stderr.getvalue().strip())

        return raw

    @staticmethod
    def _default_period_for_interval(yf_interval: str) -> str:
        """
        Tarih aralığı verilmediğinde interval'a göre makul varsayılan period.

        yfinance intraday veriler için period'u sınırlı tutar
        (ör: 1m verisi yalnızca son 7 gün için sağlanır).
        """
        intraday_short = {"1m", "5m", "15m", "30m"}
        intraday_long = {"60m"}

        if yf_interval in intraday_short:
            return "5d"
        if yf_interval in intraday_long:
            return "60d"
        return "1y"

    @staticmethod
    def _resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Native desteklenmeyen timeframe için (örn. 4h) resample uygula.

        Resample, normalize_and_validate() ÇAĞRILMADAN ÖNCE yapılır —
        yani doğrulama her zaman nihai (resample edilmiş) bar yapısı
        üzerinde çalışır.
        """
        rule = _RESAMPLE_RULE[timeframe]
        try:
            resampled = (
                df.resample(rule)
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna(subset=["Close"])
            )
            return resampled
        except Exception as exc:
            logger.warning(
                "resample_failed",
                timeframe=timeframe,
                rule=rule,
                error=str(exc),
            )
            return df
