"""
Piyasa verisi orkestrasyon servisi.

Bu servis, sistemin "Service Layer Pattern" uygulamasının ilk somut
örneğidir: Infrastructure (veri sağlama) ile Domain (saf hesaplama)
katmanlarını, hiçbiri diğerini bilmeden, tek bir iş akışında birleştirir.

Orkestrasyon akışı (get_market_analysis):
  1. MarketDataProvider.fetch_ohlcv() çağrılır — bu çağrı zaten
     Aşama 1'deki normalize_and_validate() pipeline'ından geçmiş
     temiz veri döndürür (adapter'ın kendi sorumluluğu).
  2. Dönen veri boş/None ise MarketDataServiceError fırlatılır.
  3. Veri geçerliyse TechnicalCalculator'ın saf fonksiyonları
     (RSI, RVOL, ATR, ADX) sırayla çağrılır.
  4. Ham OHLCV + hesaplanmış indikatörler tek bir DTO
     (MarketAnalysisResult) içinde birleştirilip döndürülür.

Hata Yönetimi Prensibi (kritik kısıt):
  Alttaki katmanlardan (Infrastructure: ProviderError ailesi; Domain:
  CalculationError ailesi) yükselen HİÇBİR exception doğrudan çağırana
  sızdırılmaz. Hepsi yakalanıp MarketDataServiceError'a sarmalanır.
  Orijinal hata `raise ... from exc` ile zincire eklenir — kaybolmaz,
  yalnızca üst katmana (UI/API) sızması engellenir ve tek tip, öngörülebilir
  bir hata sözleşmesi sunulur.

Katman İzolasyonu (kesin kurallar):
  - Bu servis YFinanceAdapter'ı, yfinance'ı veya herhangi bir somut
    provider implementasyonunu İMPORT ETMEZ. Yalnızca
    src.infrastructure.data_providers.base_provider.MarketDataProvider
    soyut arayüzüyle konuşur (constructor injection).
  - Servis STATELESS'tir: instance üzerinde hiçbir mutable veri
    tutulmaz. Her get_market_analysis() çağrısı kendi içinde tamamen
    izoledir; aynı servis instance'ı eşzamanlı (concurrent) çağrılarda
    güvenle paylaşılabilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.domain.calculators.technical_calculator import TechnicalCalculator
from src.domain.exceptions.domain_exceptions import (
    CalculationError,
    MarketDataServiceError,
    ProviderError,
)
from src.infrastructure.cache.ttl_cache import TTLCache
from src.infrastructure.data_providers.base_provider import MarketDataProvider
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DTO — Data Transfer Object
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketAnalysisResult:
    """
    get_market_analysis() çağrısının sonucu — UI/API katmanına sunulmaya hazır.

    Bu sınıf bilinçli olarak Domain modeli DEĞİLDİR (örn. PriceSeries gibi).
    Bir Application/Service katmanı read-model'idir: ham veriyi ve
    hesaplanmış indikatörleri, çağıranın (Streamlit sayfası, API endpoint'i)
    tek seferde tüketebileceği biçimde bir araya getirir.

    Tüm seri alanlar (rsi, rvol, atr) pandas.Series'tir — DatetimeIndex'i
    ohlcv ile birebir hizalıdır, böylece çağıran taraf ek bir join/align
    işlemi yapmadan doğrudan grafik veya tabloya aktarabilir.
    """

    symbol: str
    timeframe: str
    ohlcv: pd.DataFrame             # Aşama 1'den geçmiş, temizlenmiş ham veri
    rsi: pd.Series
    rvol: pd.Series
    atr: pd.Series
    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def bar_count(self) -> int:
        return len(self.ohlcv)

    @property
    def latest_close(self) -> float | None:
        if self.ohlcv.empty:
            return None
        return float(self.ohlcv["Close"].iloc[-1])

    @property
    def latest_rsi(self) -> float | None:
        if self.rsi.empty or pd.isna(self.rsi.iloc[-1]):
            return None
        return float(self.rsi.iloc[-1])

    @property
    def latest_rvol(self) -> float | None:
        if self.rvol.empty or pd.isna(self.rvol.iloc[-1]):
            return None
        return float(self.rvol.iloc[-1])

    def __repr__(self) -> str:
        return (
            f"MarketAnalysisResult(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"bars={self.bar_count}, latest_close={self.latest_close}, "
            f"latest_rsi={self.latest_rsi}, latest_rvol={self.latest_rvol})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────


class MarketDataService:
    """
    Piyasa verisi + teknik analiz orkestrasyon servisi.

    Bağımlılıklar (her ikisi de constructor üzerinden inject edilir,
    asla bu sınıf içinde oluşturulmaz):
      - provider:   MarketDataProvider (Infrastructure ABC) — somut adapter
                     (YFinanceAdapter, ileride TEFASAdapter vb.) çağıranın
                     sorumluluğundadır; servis yalnızca arayüzü bilir.
      - calculator: TechnicalCalculator (Domain) — saf, stateless hesaplama
                     sınıfı; tüm metodları zaten @staticmethod'dur, bu yüzden
                     instance enjeksiyonu testlerde kolay mock'lanabilirlik
                     ve gelecekte calculator'ın da bir interface'e bağlanma
                     ihtimaline (örn. alternatif bir indikatör motoru)
                     açıklık sağlar.

    Kullanım:
        service = MarketDataService(provider=yfinance_adapter, calculator=TechnicalCalculator())
        result = service.get_market_analysis("THYAO", timeframe="1d")
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        calculator: TechnicalCalculator | type[TechnicalCalculator] = TechnicalCalculator,
        rsi_period: int = 14,
        rvol_ma_period: int = 20,
        atr_period: int = 14,
        cache: TTLCache | None = None,
    ) -> None:
        """
        Args:
            provider: MarketDataProvider arayüzünü implemente eden somut adapter.
            calculator: TechnicalCalculator sınıfı veya instance'ı.
            rsi_period: RSI hesabı için periyot (inject edilir, hard-code değil).
            rvol_ma_period: RVOL hareketli ortalama periyodu.
            atr_period: ATR hesabı için periyot.
            cache: TTLCache instance'ı. None → cache devre dışı (her çağrı provider'a
                   gider). Container'dan TTLCache(ttl_seconds=60) inject edilebilir.
                   Test'lerde None bırakarak cache etkisi izole edilebilir.
        """
        self._provider = provider
        self._calculator = calculator
        self._rsi_period = rsi_period
        self._rvol_ma_period = rvol_ma_period
        self._atr_period = atr_period
        self._cache = cache

    def get_market_analysis(
        self,
        symbol: str,
        timeframe: str,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> MarketAnalysisResult:
        """
        Belirtilen sembol için ham veri çek, teknik indikatörleri hesapla,
        tek bir DTO içinde birleştirip döndür.

        Args:
            symbol: Normalize sembol (ör: "THYAO").
            timeframe: "1d", "1h", "15m", "4h" gibi desteklenen aralıklardan biri.
            start_date: Başlangıç tarihi (None ise provider varsayılanı kullanılır).
            end_date: Bitiş tarihi (None ise bugüne kadar).

        Returns:
            MarketAnalysisResult: Ham OHLCV + RSI + RVOL + ATR birleşik sonucu.

        Raises:
            MarketDataServiceError: Veri çekme, doğrulama veya hesaplama
                                     aşamalarından herhangi birinde hata
                                     oluşursa — orijinal sebep `__cause__`
                                     zincirinde korunur, ancak çağırana
                                     yalnızca bu tek tip exception sızar.
        """
        logger.debug(
            "market_analysis_started",
            symbol=symbol,
            timeframe=timeframe,
        )

        # Cache lookup (start_date/end_date içeren çağrılar cache'lenmez —
        # tarihli sorgular tekil, portföy döngüsündeki günlük çağrılar cacheable)
        cache_key = f"{symbol}|{timeframe}" if (start_date is None and end_date is None) else None
        if cache_key and self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("market_analysis_cache_hit", symbol=symbol, timeframe=timeframe)
                return cached

        ohlcv = self._fetch_ohlcv(symbol, timeframe, start_date, end_date)
        self._ensure_data_present(ohlcv, symbol)

        rsi, rvol, atr = self._compute_indicators(ohlcv, symbol)

        result = MarketAnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            ohlcv=ohlcv,
            rsi=rsi,
            rvol=rvol,
            atr=atr,
        )

        logger.info(
            "market_analysis_completed",
            symbol=symbol,
            timeframe=timeframe,
            bar_count=result.bar_count,
            latest_close=result.latest_close,
            latest_rsi=result.latest_rsi,
            latest_rvol=result.latest_rvol,
        )

        # Cache write (yalnızca tarihsiz çağrılar cache'lenir)
        if cache_key and self._cache is not None:
            self._cache.set(cache_key, result)

        return result

    # ── Private Orkestrasyon Adımları ────────────────────────────────────────

    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: date | datetime | None,
        end_date: date | datetime | None,
    ) -> pd.DataFrame:
        """
        Infrastructure katmanından veri çek; tüm ProviderError ailesi
        hatalarını MarketDataServiceError'a sarmalayarak yükselt.
        """
        try:
            return self._provider.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        except ProviderError as exc:
            logger.error(
                "market_analysis_fetch_failed",
                symbol=symbol,
                timeframe=timeframe,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            # DÜZELTME (bu turda, GERÇEK bir test ile bulundu): Bu
            # sarmalama, orijinal exception'ın .context dict'ini
            # KAYBEDİYORDU — yalnızca str(exc) (ana mesaj) taşınıyordu.
            # yfinance_adapter.py artık bazı SymbolNotFoundError'larda
            # context['note'] alanında dürüst bir belirsizlik uyarısı
            # taşıyor ("sembol geçersiz OLABİLİR VEYA sağlayıcıya
            # ulaşılamıyor OLABİLİR") — bu, UI'a kadar ulaşabilmesi için
            # BURADA da AKTARILIYOR (getattr ile duck-typing, MarketData
            # Service zaten ProviderError'ı import ediyor, katman
            # izolasyonu ihlali YOK).
            original_note = getattr(exc, "context", {}).get("note")
            raise MarketDataServiceError(
                message=f"{symbol} için veri çekilemedi: {exc}",
                symbol=symbol,
                reason=str(exc),
                cause_type=type(exc).__name__,
                note=original_note,
            ) from exc

    def _ensure_data_present(self, ohlcv: pd.DataFrame, symbol: str) -> None:
        """
        Provider'dan dönen veri None/boş ise (savunma katmanı — provider
        zaten kendi içinde NoDataError/SymbolNotFoundError fırlatmış
        olmalı, ama bu kontrol "asla None döndürme" garantisini servis
        seviyesinde de tekrar güvence altına alır).
        """
        if ohlcv is None or ohlcv.empty:
            logger.error("market_analysis_empty_data", symbol=symbol)
            raise MarketDataServiceError(
                message=f"{symbol} için geçerli piyasa verisi alınamadı (boş sonuç).",
                symbol=symbol,
                reason="empty_or_none_dataframe",
            )

    def _compute_indicators(
        self,
        ohlcv: pd.DataFrame,
        symbol: str,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Domain katmanı hesaplamalarını sırayla çalıştır.

        CalculationError ailesi (örn. InsufficientDataError) veya
        beklenmeyen KeyError/ValueError (eksik sütun, geçersiz periyot)
        burada yakalanıp MarketDataServiceError'a sarmalanır.
        """
        try:
            rsi = self._calculator.calculate_rsi(
                ohlcv["Close"], period=self._rsi_period
            )
            rvol = self._calculator.calculate_rvol(
                ohlcv["Volume"], ma_period=self._rvol_ma_period
            )
            atr = self._calculator.calculate_atr(ohlcv, period=self._atr_period)

            return rsi, rvol, atr

        except (CalculationError, KeyError, ValueError) as exc:
            logger.error(
                "market_analysis_calculation_failed",
                symbol=symbol,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise MarketDataServiceError(
                message=f"{symbol} için teknik analiz hesaplanamadı: {exc}",
                symbol=symbol,
                reason=str(exc),
                cause_type=type(exc).__name__,
            ) from exc
