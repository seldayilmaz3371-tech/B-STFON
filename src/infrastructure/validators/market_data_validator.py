"""
Piyasa verisi doğrulama katmanı.

Bu modül, BistKokpit V23.5'teki iki legacy fonksiyonun
production-grade dönüşümüdür:

  Legacy kaynak:
    _validate_ohlcv()  → OHLCVValidator.validate()
    _flatten()         → OHLCVNormalizer.flatten()

Entegrasyon noktası:
  YFinanceAdapter ve diğer tüm veri sağlayıcı adapter'ları,
  ham veriyi döndürmeden önce bu validator'dan geçirir.

  Infrastructure katmanı sorumluluğu — Domain veya Service
  katmanı bu sınıfı doğrudan kullanmaz.

BUG-12 (Legacy notundan):
  ÖNCEKİ: yfinance'ten gelen veri doğrudan kullanılıyordu.
  Korumasız: Close<=0, Volume<0, High<Low, satır duplikasyonu.
  YENİSİ: Her adapter çağrısında validate() zorunlu.

Tasarım kararları:
  - Hatalı barlar SİLİNMEZ — NaN'a çevrilir.
    Sebep: Index bütünlüğünü korur. Downstream dropna() temizler.
  - Validator stateless — instance state tutmaz.
  - Her doğrulama kuralı ayrı metod → test edilebilirlik.
  - Tüm işlemler loglanır: kaç bar hatalı, hangi kural tetiklendi.
  - Pydantic modeli ValidationReport ile sonuçlar yapılandırılmış döner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

# Zorunlu OHLCV sütun seti
_REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

# yfinance MultiIndex'te fiyat sütun adları (her iki level için)
_OHLCV_NAMES = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}


# ── Validation Sonuç Modeli ───────────────────────────────────────────────────

class OHLCVValidationReport(BaseModel):
    """
    Doğrulama işleminin yapılandırılmış sonucu.

    is_valid: True → DataFrame kullanılabilir (temizlenmiş).
    bad_bar_count: NaN'a çevrilen bar sayısı.
    duplicate_count: Kaldırılan duplikat satır sayısı.
    warnings: İnsan okunabilir uyarı mesajları.
    """

    is_valid: bool
    total_bars: int
    bad_bar_count: int = 0
    duplicate_count: int = 0
    warnings: list[str] = []

    @property
    def clean_bar_count(self) -> int:
        return self.total_bars - self.bad_bar_count


# ── Normalizer ────────────────────────────────────────────────────────────────

class OHLCVNormalizer:
    """
    yfinance MultiIndex sütun yapısını normalize eder.

    Legacy karşılığı: _flatten()

    yfinance 0.2.x, sürüme ve sembol sayısına göre farklı
    MultiIndex formatları döndürebilir:
      - ("Price", "Ticker"): level-0'da fiyat adları
      - ("Ticker", "Price"): level-1'de fiyat adları

    Her iki format da handle edilir. "Adj Close" → "Close" yeniden
    adlandırılır (auto_adjust=False çağrılarında).

    Timezone-aware index, timezone-naive'e dönüştürülür
    (SQLAlchemy TEXT storage uyumu).
    """

    @staticmethod
    def flatten(df: pd.DataFrame | None) -> pd.DataFrame | None:
        """
        Ham yfinance DataFrame'ini normalize et.

        Args:
            df: yfinance.download() çıktısı (MultiIndex veya flat).

        Returns:
            Normalize edilmiş DataFrame. None döner eğer:
            - df None veya boş
            - Zorunlu OHLCV sütunları mevcut değil
        """
        if df is None or df.empty:
            logger.debug("normalizer_skipped", reason="empty_or_none")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            lvl0_names = set(df.columns.get_level_values(0))
            lvl1_names = set(df.columns.get_level_values(1))

            if lvl0_names & _OHLCV_NAMES:
                # ("Price", "Ticker") yapısı — level-0 al
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
                logger.debug("multiindex_flattened", level_used=0)
            elif lvl1_names & _OHLCV_NAMES:
                # ("Ticker", "Price") yapısı — level-1 al
                df = df.copy()
                df.columns = df.columns.get_level_values(1)
                logger.debug("multiindex_flattened", level_used=1)
            else:
                # Bilinmeyen yapı — level-0 ile dene
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
                logger.warning(
                    "multiindex_unknown_structure",
                    level0_sample=list(lvl0_names)[:3],
                    level1_sample=list(lvl1_names)[:3],
                )

        # "Adj Close" → "Close" (auto_adjust=False durumu)
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
            logger.debug("adj_close_renamed")

        # Duplikat sütunları kaldır
        df = df.loc[:, ~df.columns.duplicated()]

        # Zorunlu sütun kontrolü
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.error(
                "normalizer_missing_required_columns",
                missing=sorted(missing),
                available=sorted(df.columns.tolist()),
            )
            return None

        # Timezone-aware → naive (SQLAlchemy TEXT uyumu)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)
            logger.debug("timezone_converted_to_naive")

        result = df if not df.empty else None
        if result is None:
            logger.warning("normalizer_result_empty_after_processing")

        return result


# ── Validator ─────────────────────────────────────────────────────────────────

class MarketDataValidator:
    """
    OHLCV veri kalitesi doğrulama sınıfı.

    Legacy karşılığı: _validate_ohlcv()

    Kontrol edilen koşullar (BUG-12'den alınan, genişletilmiş):
      1. Duplicate index satırları
      2. Close <= 0  (silinmiş/hatalı sembol fiyatları)
      3. Volume < 0  (veri sağlayıcı hatası)
      4. High < Low  (OHLC tutarsızlığı)
      5. High < Close veya High < Open  (bar bütünlüğü bozuk)
      6. Low > Close veya Low > Open    (bar bütünlüğü bozuk)

    Yaklaşım:
      Hatalı barların OHLCV değerleri NaN'a çevrilir.
      Satır silinmez — index bütünlüğü korunur.
      Downstream dropna(subset=["Close"]) veya
      dropna(subset=["SMA20","ATR","RSI"]) temizler.

    Kullanım:
        validator = MarketDataValidator()
        df_clean, report = validator.validate(df_raw)
        if not report.is_valid:
            logger.error("validation_failed", ...)
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """
        Args:
            strict_mode: True → kötü bar oranı %5'i aşarsa
                         is_valid=False döner (pipeline için).
                         False → sadece loglama, her zaman True.
        """
        self._strict_mode = strict_mode

    def validate(
        self,
        df: pd.DataFrame | None,
        symbol: str = "UNKNOWN",
    ) -> tuple[pd.DataFrame | None, OHLCVValidationReport]:
        """
        OHLCV DataFrame'ini doğrula ve temizle.

        Args:
            df: Normalize edilmiş (flatten sonrası) OHLCV DataFrame.
            symbol: Log mesajlarına dahil edilecek sembol adı (debug için).

        Returns:
            (temizlenmiş_df, ValidationReport) tuple'ı.
            df None ise (None, invalid_report) döner.
        """
        if df is None or df.empty:
            report = OHLCVValidationReport(
                is_valid=False,
                total_bars=0,
                warnings=["DataFrame None veya boş."],
            )
            logger.warning("validation_skipped_empty", symbol=symbol)
            return None, report

        df = df.copy()
        total_bars = len(df)
        warnings: list[str] = []
        duplicate_count = 0

        # ── Kural 1: Duplicate index ──────────────────────────────────────────
        dup_mask = df.index.duplicated()
        if dup_mask.any():
            duplicate_count = int(dup_mask.sum())
            df = df[~df.index.duplicated(keep="first")]
            msg = f"Duplikat index: {duplicate_count} satır kaldırıldı."
            warnings.append(msg)
            logger.warning(
                "ohlcv_duplicate_index_removed",
                symbol=symbol,
                count=duplicate_count,
            )

        # ── Kural 2–6: Bar bütünlük kontrolleri ──────────────────────────────
        bad_mask = self._compute_bad_mask(df, symbol, warnings)

        bad_bar_count = int(bad_mask.sum())
        if bad_bar_count > 0:
            # Hatalı barları NaN'a çevir — silme!
            ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df.loc[bad_mask, ohlcv_cols] = np.nan

            logger.warning(
                "ohlcv_bad_bars_nullified",
                symbol=symbol,
                bad_count=bad_bar_count,
                total_count=len(df),
                bad_ratio=round(bad_bar_count / len(df), 4),
            )

        # ── is_valid kararı ───────────────────────────────────────────────────
        bad_ratio = bad_bar_count / len(df) if len(df) > 0 else 0.0
        is_valid: bool

        if len(df) == 0:
            is_valid = False
            warnings.append("Duplikat temizlemesi sonrası DataFrame boşaldı.")
        elif self._strict_mode and bad_ratio > 0.05:
            is_valid = False
            warnings.append(
                f"Strict mode: Kötü bar oranı %{bad_ratio * 100:.1f} > %5 eşiği."
            )
        else:
            is_valid = True

        report = OHLCVValidationReport(
            is_valid=is_valid,
            total_bars=total_bars,
            bad_bar_count=bad_bar_count,
            duplicate_count=duplicate_count,
            warnings=warnings,
        )

        if is_valid:
            logger.debug(
                "ohlcv_validation_passed",
                symbol=symbol,
                total_bars=len(df),
                bad_bars=bad_bar_count,
                duplicates=duplicate_count,
            )
        else:
            logger.error(
                "ohlcv_validation_failed",
                symbol=symbol,
                warnings=warnings,
            )

        return df, report

    def _compute_bad_mask(
        self,
        df: pd.DataFrame,
        symbol: str,
        warnings: list[str],
    ) -> pd.Series:
        """
        Tüm bar bütünlük kurallarını uygula ve birleşik hata maskesi döndür.

        Her kural ayrı olarak değerlendirilir ve loglanır.
        Bu yapı bireysel kural testlerini kolaylaştırır.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_ = df["Open"]
        volume = df["Volume"]

        # Kural 2: Close <= 0
        mask_close_zero = close <= 0
        count = int(mask_close_zero.sum())
        if count:
            warnings.append(f"Close ≤ 0: {count} bar.")
            logger.warning("ohlcv_close_zero_or_negative", symbol=symbol, count=count)

        # Kural 3: Volume < 0
        mask_vol_negative = volume < 0
        count = int(mask_vol_negative.sum())
        if count:
            warnings.append(f"Volume < 0: {count} bar.")
            logger.warning("ohlcv_volume_negative", symbol=symbol, count=count)

        # Kural 4: High < Low
        mask_hl_inversion = high < low
        count = int(mask_hl_inversion.sum())
        if count:
            warnings.append(f"High < Low (OHLC tutarsız): {count} bar.")
            logger.warning("ohlcv_high_below_low", symbol=symbol, count=count)

        # Kural 5: High < Close veya High < Open (bar bütünlüğü bozuk)
        mask_high_violation = (high < close) | (high < open_)
        count = int(mask_high_violation.sum())
        if count:
            warnings.append(f"High < Close veya High < Open: {count} bar.")
            logger.warning("ohlcv_high_violation", symbol=symbol, count=count)

        # Kural 6: Low > Close veya Low > Open
        mask_low_violation = (low > close) | (low > open_)
        count = int(mask_low_violation.sum())
        if count:
            warnings.append(f"Low > Close veya Low > Open: {count} bar.")
            logger.warning("ohlcv_low_violation", symbol=symbol, count=count)

        # Tüm maskeleri OR ile birleştir
        return (
            mask_close_zero
            | mask_vol_negative
            | mask_hl_inversion
            | mask_high_violation
            | mask_low_violation
        )


# ── Convenience fonksiyonu ────────────────────────────────────────────────────

def normalize_and_validate(
    df: pd.DataFrame | None,
    symbol: str = "UNKNOWN",
    strict_mode: bool = False,
) -> tuple[pd.DataFrame | None, OHLCVValidationReport]:
    """
    Normalize + Validate pipeline — tek çağrı.

    Tüm adapter'larda kullanılacak standart pipeline:
      1. OHLCVNormalizer.flatten()   → MultiIndex düzleştirme
      2. MarketDataValidator.validate() → Bar bütünlük kontrolü

    Args:
        df: Ham yfinance.download() çıktısı.
        symbol: Log mesajları için sembol adı.
        strict_mode: Kötü bar oranı >%5 ise is_valid=False.

    Returns:
        (temizlenmiş_df, ValidationReport) tuple'ı.

    Kullanım (adapter içinde):
        raw = yf.download(yf_symbol, ...)
        df, report = normalize_and_validate(raw, symbol=ticker)
        if not report.is_valid:
            raise DataValidationError(provider="yfinance", reason=str(report.warnings))
    """
    flattened = OHLCVNormalizer.flatten(df)

    validator = MarketDataValidator(strict_mode=strict_mode)
    clean_df, report = validator.validate(flattened, symbol=symbol)

    return clean_df, report
