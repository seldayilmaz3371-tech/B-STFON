"""
MarketDataValidator ve OHLCVNormalizer unit testleri.

Her doğrulama kuralı ayrı test sınıfında.
Legacy BistKokpit'teki _validate_ohlcv() ve _flatten() davranışları
burada doğrulanır.

Tasarım prensibi (legacy'den korunan):
  - Hatalı barlar SİLİNMEZ — NaN'a çevrilir
  - Index bütünlüğü her zaman korunur
  - Duplicate satırlar kaldırılır (ilk occurrence korunur)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.validators.market_data_validator import (
    MarketDataValidator,
    OHLCVNormalizer,
    OHLCVValidationReport,
    normalize_and_validate,
)


# ── Test Veri Yardımcıları ────────────────────────────────────────────────────

def _make_ohlcv(
    bars: int = 10,
    base_price: float = 100.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Geçerli OHLCV DataFrame üret."""
    start = start or datetime(2024, 1, 2, 10, 0)
    idx = pd.date_range(start=start, periods=bars, freq="15min")

    close = np.full(bars, base_price)
    return pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.01,
            "Low":    close * 0.98,
            "Close":  close,
            "Volume": np.full(bars, 1_000_000.0),
        },
        index=idx,
    )


def _inject_bad_bar(df: pd.DataFrame, idx: int, **kwargs) -> pd.DataFrame:
    """DataFrame'in belirli satırına bozuk değer enjekte et."""
    df = df.copy()
    for col, val in kwargs.items():
        df.iloc[idx, df.columns.get_loc(col)] = val
    return df


# ── OHLCVNormalizer Testleri ──────────────────────────────────────────────────

class TestOHLCVNormalizer:

    def test_flat_dataframe_passthrough(self):
        """Zaten düz DataFrame değişmeden geçmeli."""
        df = _make_ohlcv()
        result = OHLCVNormalizer.flatten(df)
        assert result is not None
        assert set(result.columns) >= {"Open", "High", "Low", "Close", "Volume"}

    def test_none_input_returns_none(self):
        result = OHLCVNormalizer.flatten(None)
        assert result is None

    def test_empty_dataframe_returns_none(self):
        result = OHLCVNormalizer.flatten(pd.DataFrame())
        assert result is None

    def test_multiindex_level0_price_names(self):
        """MultiIndex: level-0'da fiyat adları → level-0 alınmalı."""
        df = _make_ohlcv()
        # (Price, Ticker) yapısı simülasyonu
        df.columns = pd.MultiIndex.from_tuples(
            [(c, "THYAO") for c in df.columns]
        )
        result = OHLCVNormalizer.flatten(df)
        assert result is not None
        assert "Close" in result.columns
        assert not isinstance(result.columns, pd.MultiIndex)

    def test_multiindex_level1_price_names(self):
        """MultiIndex: level-1'de fiyat adları → level-1 alınmalı."""
        df = _make_ohlcv()
        # (Ticker, Price) yapısı simülasyonu
        df.columns = pd.MultiIndex.from_tuples(
            [("THYAO", c) for c in df.columns]
        )
        result = OHLCVNormalizer.flatten(df)
        assert result is not None
        assert "Close" in result.columns

    def test_adj_close_renamed_to_close(self):
        """'Adj Close' sütunu 'Close' olarak yeniden adlandırılmalı."""
        df = _make_ohlcv()
        df = df.rename(columns={"Close": "Adj Close"})
        result = OHLCVNormalizer.flatten(df)
        assert result is not None
        assert "Close" in result.columns
        assert "Adj Close" not in result.columns

    def test_missing_required_column_returns_none(self):
        """Zorunlu sütun eksikse None dönmeli."""
        df = _make_ohlcv().drop(columns=["Close"])
        result = OHLCVNormalizer.flatten(df)
        assert result is None

    def test_timezone_aware_index_converted(self):
        """Timezone-aware index, naive'e çevrilmeli."""
        df = _make_ohlcv()
        df.index = df.index.tz_localize("Europe/Istanbul")
        result = OHLCVNormalizer.flatten(df)
        assert result is not None
        assert result.index.tz is None

    def test_duplicate_columns_removed(self):
        """Duplikat sütunlar kaldırılmalı."""
        df = _make_ohlcv()
        df = pd.concat([df, df[["Close"]].rename(columns={"Close": "Close"})], axis=1)
        # Aynı isimde iki Close sütunu
        result = OHLCVNormalizer.flatten(df)
        assert result is not None
        assert result.columns.tolist().count("Close") == 1


# ── MarketDataValidator Testleri ──────────────────────────────────────────────

class TestMarketDataValidatorCleanData:

    def setup_method(self):
        self.validator = MarketDataValidator()

    def test_clean_data_passes(self):
        """Temiz veri: is_valid=True, bad_bar_count=0."""
        df = _make_ohlcv()
        result, report = self.validator.validate(df, symbol="THYAO")
        assert result is not None
        assert report.is_valid is True
        assert report.bad_bar_count == 0
        assert report.duplicate_count == 0
        assert len(report.warnings) == 0

    def test_none_input_invalid(self):
        """None input: is_valid=False."""
        result, report = self.validator.validate(None)
        assert result is None
        assert report.is_valid is False

    def test_empty_dataframe_invalid(self):
        """Boş DataFrame: is_valid=False."""
        result, report = self.validator.validate(pd.DataFrame())
        assert result is None
        assert report.is_valid is False


class TestMarketDataValidatorCloseLEZero:

    def setup_method(self):
        self.validator = MarketDataValidator()

    def test_close_zero_nullified(self):
        """Close = 0 olan bar NaN'a çevrilmeli."""
        df = _inject_bad_bar(_make_ohlcv(), idx=3, Close=0.0)
        result, report = self.validator.validate(df, symbol="TEST")

        assert report.bad_bar_count == 1
        assert pd.isna(result.iloc[3]["Close"])
        # Diğer barlar etkilenmemeli
        assert not pd.isna(result.iloc[2]["Close"])
        assert not pd.isna(result.iloc[4]["Close"])

    def test_close_negative_nullified(self):
        """Close < 0 olan bar NaN'a çevrilmeli."""
        df = _inject_bad_bar(_make_ohlcv(), idx=5, Close=-10.0)
        result, report = self.validator.validate(df, symbol="TEST")
        assert report.bad_bar_count == 1
        assert pd.isna(result.iloc[5]["Close"])

    def test_close_zero_does_not_delete_row(self):
        """
        KRİTİK: Hatalı bar silinmemeli — sadece NaN'a çevrilmeli.
        Legacy tasarım kararı: index bütünlüğü korunmalı.
        """
        original_len = 10
        df = _inject_bad_bar(_make_ohlcv(bars=original_len), idx=3, Close=0.0)
        result, report = self.validator.validate(df)
        assert len(result) == original_len  # Uzunluk DEĞİŞMEMELİ


class TestMarketDataValidatorVolumeNegative:

    def setup_method(self):
        self.validator = MarketDataValidator()

    def test_volume_negative_nullified(self):
        """Volume < 0 olan bar NaN'a çevrilmeli."""
        df = _inject_bad_bar(_make_ohlcv(), idx=2, Volume=-500.0)
        result, report = self.validator.validate(df, symbol="TEST")
        assert report.bad_bar_count == 1
        assert pd.isna(result.iloc[2]["Volume"])

    def test_volume_zero_allowed(self):
        """Volume = 0 geçerli (işlem olmayan bar — tatil gönü değil ama mümkün)."""
        df = _inject_bad_bar(_make_ohlcv(), idx=2, Volume=0.0)
        result, report = self.validator.validate(df)
        assert report.bad_bar_count == 0


class TestMarketDataValidatorHighLowInversion:

    def setup_method(self):
        self.validator = MarketDataValidator()

    def test_high_below_low_nullified(self):
        """High < Low: OHLC tutarsızlığı — NaN'a çevrilmeli."""
        df = _make_ohlcv()
        # High ve Low'u ters çevir
        df = df.copy()
        df.iloc[4, df.columns.get_loc("High")] = 95.0   # < Low (98)
        result, report = self.validator.validate(df)
        assert report.bad_bar_count == 1
        assert pd.isna(result.iloc[4]["Close"])

    def test_high_below_close_nullified(self):
        """High < Close: bar bütünlüğü bozuk."""
        df = _make_ohlcv()
        df = df.copy()
        df.iloc[6, df.columns.get_loc("High")] = 99.0   # < Close (100)
        result, report = self.validator.validate(df)
        assert report.bad_bar_count == 1

    def test_high_below_open_nullified(self):
        """High < Open: bar bütünlüğü bozuk."""
        df = _make_ohlcv()
        df = df.copy()
        # Open = 99, High = 98 → High < Open
        df.iloc[1, df.columns.get_loc("Open")] = 99.0
        df.iloc[1, df.columns.get_loc("High")] = 98.0
        result, report = self.validator.validate(df)
        assert report.bad_bar_count == 1

    def test_low_above_close_nullified(self):
        """Low > Close: bar bütünlüğü bozuk."""
        df = _make_ohlcv()
        df = df.copy()
        df.iloc[3, df.columns.get_loc("Low")] = 101.0   # > Close (100)
        result, report = self.validator.validate(df)
        assert report.bad_bar_count == 1

    def test_low_above_open_nullified(self):
        """Low > Open: bar bütünlüğü bozuk."""
        df = _make_ohlcv()
        df = df.copy()
        # Open = 99, Low = 100 → Low > Open
        df.iloc[7, df.columns.get_loc("Open")] = 99.0
        df.iloc[7, df.columns.get_loc("Low")] = 100.0
        result, report = self.validator.validate(df)
        assert report.bad_bar_count == 1


class TestMarketDataValidatorDuplicateIndex:

    def setup_method(self):
        self.validator = MarketDataValidator()

    def test_duplicate_index_removed(self):
        """Duplikat index: ilk occurrence korunur, diğerleri kaldırılır."""
        df = _make_ohlcv(bars=5)
        # 3. satırın timestamp'ini 2. satırla aynı yap
        new_index = df.index.tolist()
        new_index[2] = new_index[1]  # Duplikat
        df.index = pd.DatetimeIndex(new_index)

        result, report = self.validator.validate(df, symbol="TEST")

        assert report.duplicate_count == 1
        assert len(result) == 4   # 5 - 1 duplikat = 4
        assert not result.index.duplicated().any()

    def test_first_occurrence_kept(self):
        """Duplikat index: keep='first' — ilk satır korunmalı."""
        df = _make_ohlcv(bars=4)
        original_close_val = float(df.iloc[1]["Close"])

        # 3. satırı 2. ile aynı timestamp'e al, Close değerini değiştir
        df_dup = df.copy()
        df_dup.iloc[2, df_dup.columns.get_loc("Close")] = 999.0
        new_index = df_dup.index.tolist()
        new_index[2] = new_index[1]
        df_dup.index = pd.DatetimeIndex(new_index)

        result, report = self.validator.validate(df_dup)

        # İlk occurrence (orijinal değer) korunmalı
        dup_ts = df_dup.index[1]
        assert float(result.loc[dup_ts, "Close"]) == pytest.approx(original_close_val)


class TestMarketDataValidatorMultipleErrors:

    def setup_method(self):
        self.validator = MarketDataValidator()

    def test_multiple_bad_bars_counted(self):
        """Birden fazla hatalı bar: hepsi sayılmalı."""
        df = _make_ohlcv(bars=10)
        df = _inject_bad_bar(df, idx=2, Close=0.0)
        df = _inject_bad_bar(df, idx=5, Volume=-100.0)
        df = _inject_bad_bar(df, idx=8, Close=-5.0)

        result, report = self.validator.validate(df, symbol="MULTI")
        assert report.bad_bar_count == 3

        # Her hatalı bar NaN'a çevrilmiş olmalı
        for bad_idx in [2, 5, 8]:
            assert pd.isna(result.iloc[bad_idx]["Close"]) or pd.isna(result.iloc[bad_idx]["Volume"])

    def test_good_bars_untouched(self):
        """İyi barlar NaN'a çevrilmemeli."""
        df = _make_ohlcv(bars=10)
        df = _inject_bad_bar(df, idx=3, Close=0.0)

        result, report = self.validator.validate(df)

        # Bar 3 dışındakiler etkilenmemeli
        for i in range(10):
            if i != 3:
                assert not pd.isna(result.iloc[i]["Close"]), f"Bar {i} yanlışlıkla NaN oldu!"

    def test_combined_duplicate_and_bad_bars(self):
        """Hem duplikat hem hatalı bar aynı anda."""
        df = _make_ohlcv(bars=6)
        # Duplikat ekle
        new_idx = df.index.tolist()
        new_idx[4] = new_idx[3]
        df.index = pd.DatetimeIndex(new_idx)
        # Hatalı bar ekle
        df = _inject_bad_bar(df, idx=1, Close=0.0)

        result, report = self.validator.validate(df)

        assert report.duplicate_count >= 1
        assert report.bad_bar_count >= 1
        assert report.is_valid is True  # Temizlenmiş ama valid


class TestMarketDataValidatorStrictMode:

    def test_strict_mode_fails_on_high_bad_ratio(self):
        """Strict mode: kötü bar oranı >%5 ise is_valid=False."""
        validator = MarketDataValidator(strict_mode=True)
        df = _make_ohlcv(bars=20)
        # 2 bar boz → %10 kötü bar oranı
        df = _inject_bad_bar(df, idx=5, Close=0.0)
        df = _inject_bad_bar(df, idx=10, Close=-1.0)

        _, report = validator.validate(df)
        assert report.is_valid is False
        assert any("Strict mode" in w for w in report.warnings)

    def test_normal_mode_passes_despite_bad_bars(self):
        """Normal mode: kötü bar oranı >%5 olsa bile is_valid=True."""
        validator = MarketDataValidator(strict_mode=False)
        df = _make_ohlcv(bars=20)
        df = _inject_bad_bar(df, idx=5, Close=0.0)
        df = _inject_bad_bar(df, idx=10, Close=-1.0)

        _, report = validator.validate(df)
        assert report.is_valid is True


# ── normalize_and_validate Pipeline Testi ────────────────────────────────────

class TestNormalizeAndValidatePipeline:

    def test_clean_pipeline(self):
        """Temiz veri: pipeline başarılı."""
        df = _make_ohlcv()
        result, report = normalize_and_validate(df, symbol="THYAO")
        assert result is not None
        assert report.is_valid is True

    def test_none_input_pipeline(self):
        """None input: pipeline is_valid=False döndürmeli."""
        result, report = normalize_and_validate(None, symbol="TEST")
        assert result is None
        assert report.is_valid is False

    def test_multiindex_then_validate(self):
        """MultiIndex normalize + validate pipeline tam akış."""
        df = _make_ohlcv()
        # MultiIndex yap, sonra kötü bar ekle
        df.columns = pd.MultiIndex.from_tuples([(c, "THYAO") for c in df.columns])
        # Flatten sonrasında bozulacak — önce flatten uygulanacak

        result, report = normalize_and_validate(df, symbol="THYAO")
        assert result is not None
        assert not isinstance(result.columns, pd.MultiIndex)

    def test_pipeline_preserves_index_length_on_bad_bars(self):
        """
        KRİTİK: Pipeline sonrası index uzunluğu korunmalı.
        (Duplikat kaldırma hariç)
        """
        df = _make_ohlcv(bars=20)
        df = _inject_bad_bar(df, idx=3, Close=0.0)
        df = _inject_bad_bar(df, idx=10, Volume=-100.0)

        result, report = normalize_and_validate(df)
        assert result is not None
        assert len(result) == 20  # Uzunluk değişmemeli


# ── ValidationReport Model Testleri ──────────────────────────────────────────

class TestOHLCVValidationReport:

    def test_clean_bar_count_property(self):
        """clean_bar_count = total_bars - bad_bar_count."""
        report = OHLCVValidationReport(
            is_valid=True,
            total_bars=100,
            bad_bar_count=5,
        )
        assert report.clean_bar_count == 95

    def test_report_with_warnings(self):
        """Warnings listesi doğru saklanmalı."""
        report = OHLCVValidationReport(
            is_valid=False,
            total_bars=10,
            bad_bar_count=10,
            warnings=["Close ≤ 0: 10 bar."],
        )
        assert len(report.warnings) == 1
        assert "Close" in report.warnings[0]
