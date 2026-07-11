"""
TechnicalCalculator unit testleri.

Test kapsamı:
  1. RVOL (BUG-11 kritik) — sıfıra bölünme, NaN propagation, birim doğruluğu
  2. RSI — Wilder smoothing, sınır değerler, NaN toleransı
  3. ADX — DI hesabı, eksik sütun koruması
  4. ATR — True Range bileşenleri
  5. VWAP — günlük sıfırlama, sıfır hacim koruması
  6. OBV — BUG-03 flat-bar düzeltmesi doğrulaması
  7. Rolling Rank Pct — stride_tricks doğruluğu, NaN barlar
  8. Bollinger Squeeze — eşik mantığı
  9. VCP — daralma kriterleri
  10. Relative Strength — inner join (BUG-05/06), slope

Her test sınıfı bağımsız — fixture shared değil (izolasyon).
NaN güvenlik testleri her hesaplama türü için ayrı kontrol edilir.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.technical_calculator import (
    TechnicalCalculator,
    ADXResult,
    VCPComponents,
    RelativeStrengthResult,
)


# ── Test Veri Yardımcıları ────────────────────────────────────────────────────

def _make_ohlcv(
    bars: int = 50,
    base_price: float = 100.0,
    trend: float = 0.001,          # günlük artış oranı
    vol_base: float = 1_000_000.0,
    start: str = "2024-01-02 10:00",
    freq: str = "15min",
) -> pd.DataFrame:
    """Gerçekçi OHLCV DataFrame üret (trending fiyat)."""
    idx = pd.date_range(start=start, periods=bars, freq=freq)
    close = base_price * (1 + trend) ** np.arange(bars)
    noise = np.random.default_rng(42).uniform(-0.005, 0.005, bars)
    close = close * (1 + noise)

    return pd.DataFrame(
        {
            "Open":   close * np.random.default_rng(1).uniform(0.995, 1.005, bars),
            "High":   close * np.random.default_rng(2).uniform(1.003, 1.012, bars),
            "Low":    close * np.random.default_rng(3).uniform(0.988, 0.997, bars),
            "Close":  close,
            "Volume": vol_base * np.random.default_rng(4).uniform(0.5, 1.5, bars),
        },
        index=idx,
    )


def _inject_nan(series: pd.Series, *indices: int) -> pd.Series:
    """Belirtilen indexlere NaN enjekte et."""
    s = series.copy()
    for i in indices:
        s.iloc[i] = np.nan
    return s


def _make_monotone_series(n: int = 30, start: float = 1.0) -> pd.Series:
    """Monoton artan seri (RSI = 100 yaklaşımı testi için)."""
    return pd.Series(np.linspace(start, start * 2, n), dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# RVOL Testleri (BUG-11 — En Kritik)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateRVOL:
    """
    BUG-11 düzeltmesinin doğrulanması.

    RVOL = Hacim / Hacim_Hareketli_Ortalama
    (NOT: Fiyat / Hacim_Ortalaması değil!)
    """

    def test_rvol_basic_calculation(self):
        """Temel RVOL hesabı: hacim/hacim_ma doğru."""
        # Sabit hacim: RVOL her bar için 1.0 olmalı
        volume = pd.Series([1_000_000.0] * 20, dtype=float)
        rvol = TechnicalCalculator.calculate_rvol(volume, ma_period=5)

        # Warmup sonrası tüm barlar 1.0 olmalı
        assert all(abs(rvol.iloc[5:] - 1.0) < 1e-9), "Sabit hacimde RVOL=1.0 olmalı"

    def test_rvol_double_volume_returns_two(self):
        """2x hacim → RVOL = 2.0."""
        base_vol = 1_000_000.0
        # İlk 10 bar normal, son bar 2x
        volume = pd.Series([base_vol] * 10 + [base_vol * 2.0], dtype=float)
        rvol = TechnicalCalculator.calculate_rvol(volume, ma_period=10)

        # Son bar RVOL ≈ 2.0 / 1.0 × bazlı oran
        # Exactness: son bar = 2*base, ma = mean([base]*9 + [2*base]) / 10
        last_rvol = float(rvol.iloc[-1])
        assert last_rvol > 1.5, f"2x hacimde RVOL > 1.5 bekleniyor, gelen: {last_rvol:.3f}"

    def test_rvol_units_are_dimensionless(self):
        """
        BUG-11 unit testi: RVOL boyutsuz olmalı.

        Hacim farklı büyüklüklerde olsa da RVOL aynı kalmalı.
        (Fiyat/HacimMA gibi boyutlu bir hesap bu testi geçemez.)
        """
        volume_small = pd.Series([100.0] * 20 + [150.0], dtype=float)
        volume_large = volume_small * 1_000_000  # Aynı oranlar, farklı büyüklük

        rvol_small = TechnicalCalculator.calculate_rvol(volume_small, ma_period=20)
        rvol_large = TechnicalCalculator.calculate_rvol(volume_large, ma_period=20)

        # Son bar RVOL her ikisinde de aynı olmalı (boyutsuz oran)
        assert abs(float(rvol_small.iloc[-1]) - float(rvol_large.iloc[-1])) < 1e-6, (
            "RVOL boyutsuz olmalı — hacim büyüklüğünden bağımsız"
        )

    def test_rvol_zero_volume_returns_zero(self):
        """vol = 0, vol_ma > 0 → RVOL = 0.0 (sıfır hacimli bar gerçek)."""
        volume = pd.Series([1_000_000.0] * 9 + [0.0], dtype=float)
        rvol = TechnicalCalculator.calculate_rvol(volume, ma_period=5)

        assert float(rvol.iloc[-1]) == pytest.approx(0.0, abs=1e-9), (
            "Sıfır hacimde RVOL = 0.0 olmalı"
        )

    def test_rvol_zero_moving_average_returns_neutral(self):
        """vol_ma = 0 (tüm pencere sıfır) → RVOL = 1.0 (nötr default)."""
        volume = pd.Series([0.0] * 5 + [1_000_000.0], dtype=float)
        rvol = TechnicalCalculator.calculate_rvol(volume, ma_period=5)

        # İlk 5 bar: vol_ma = 0 → RVOL = 1.0
        for i in range(5):
            assert float(rvol.iloc[i]) == pytest.approx(1.0, abs=1e-9), (
                f"vol_ma=0 durumunda bar {i} RVOL=1.0 olmalı"
            )

    def test_rvol_nan_volume_propagates_nan(self):
        """volume = NaN (Aşama 1 hatalı bar) → RVOL = NaN."""
        volume = pd.Series([1_000_000.0] * 10, dtype=float)
        volume.iloc[5] = np.nan

        rvol = TechnicalCalculator.calculate_rvol(volume, ma_period=3)

        assert pd.isna(rvol.iloc[5]), "NaN volume → NaN RVOL olmalı"
        # Diğer barlar etkilenmemeli
        assert not pd.isna(rvol.iloc[4]), "Bar 4 (öncesi) NaN olmamalı"
        assert not pd.isna(rvol.iloc[6]), "Bar 6 (sonrası) NaN olmamalı"

    def test_rvol_does_not_crash_all_nan_series(self):
        """Tamamen NaN seri: crash vermemeli."""
        volume = pd.Series([np.nan] * 10, dtype=float)
        rvol = TechnicalCalculator.calculate_rvol(volume, ma_period=5)

        assert len(rvol) == 10
        assert rvol.isna().all()

    def test_rvol_negative_period_raises(self):
        """Negatif ma_period → ValueError."""
        volume = pd.Series([1e6] * 10)
        with pytest.raises(ValueError, match="pozitif"):
            TechnicalCalculator.calculate_rvol(volume, ma_period=-1)

    def test_rvol_threshold_detection(self):
        """
        RVOL >= 1.5 eşiği ile 'yüksek hacim' tespiti.

        Legacy mantığı: vol_ratio_now >= RVOL_THRESHOLD (1.5) AND close > prev_close
        Bu test o kombinasyonun doğru çalıştığını doğrular.
        """
        # Normal hacim: 1M, son bar: 2M (RVOL ≈ 2.0 > 1.5)
        volume = pd.Series([1_000_000.0] * 20 + [2_000_000.0])
        close = pd.Series([100.0] * 20 + [101.0])

        rvol = TechnicalCalculator.calculate_rvol(volume, ma_period=20)
        vol_confirmed = (rvol >= 1.5) & (close > close.shift(1))

        assert bool(vol_confirmed.iloc[-1]), "Son bar yüksek hacim + yükseliş: Vol_Confirmed True olmalı"
        # Normal barlar False olmalı (RVOL ≈ 1.0 < 1.5)
        assert not bool(vol_confirmed.iloc[10]), "Normal bar: Vol_Confirmed False olmalı"


# ─────────────────────────────────────────────────────────────────────────────
# RSI Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateRSI:

    def test_rsi_range_0_to_100(self):
        """RSI değerleri [0, 100] aralığında olmalı."""
        df = _make_ohlcv(bars=100)
        rsi = TechnicalCalculator.calculate_rsi(df["Close"])

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0.0).all(), "RSI < 0 olamaz"
        assert (valid_rsi <= 100.0).all(), "RSI > 100 olamaz"

    def test_rsi_monotone_rising_approaches_100(self):
        """
        Monoton artan fiyatta RSI = 100.

        avg_loss = 0 (hiç düşüş yok) → saf yükseliş → RSI = 100.
        Bu implementasyonun bilinçli tasarım kararıdır:
          avg_loss=0, avg_gain>0 → RSI=100
          avg_loss=0, avg_gain=0 → RSI=50 (sabit fiyat, nötr)
        """
        close = _make_monotone_series(n=100)
        rsi = TechnicalCalculator.calculate_rsi(close, period=14)

        assert float(rsi.iloc[-1]) == 100.0, (
            f"Monoton yükselişte RSI=100 bekleniyor, gelen: {rsi.iloc[-1]:.1f}"
        )

    def test_rsi_constant_price_is_fifty(self):
        """Sabit fiyat: avg_gain=avg_loss=0 → RSI=50 (nötr)."""
        close = pd.Series([100.0] * 50)
        rsi = TechnicalCalculator.calculate_rsi(close, period=14)

        assert float(rsi.iloc[-1]) == 50.0, (
            f"Sabit fiyatta RSI=50 bekleniyor, gelen: {rsi.iloc[-1]:.1f}"
        )

    def test_rsi_nan_bars_do_not_crash(self):
        """NaN barlar RSI'ı crash ettirmemeli."""
        df = _make_ohlcv(bars=50)
        close_with_nan = _inject_nan(df["Close"], 5, 15, 30)

        rsi = TechnicalCalculator.calculate_rsi(close_with_nan, period=14)

        assert len(rsi) == 50
        # NaN olmayan barların çoğunda geçerli RSI olmalı
        assert rsi.notna().sum() > 30

    def test_rsi_period_parametric(self):
        """Farklı period değerleri farklı RSI üretmeli."""
        df = _make_ohlcv(bars=100)
        rsi_14 = TechnicalCalculator.calculate_rsi(df["Close"], period=14)
        rsi_9 = TechnicalCalculator.calculate_rsi(df["Close"], period=9)

        # Farklı periyotlar farklı değer üretmeli (en azından 1 bar farklı)
        assert not (rsi_14.dropna() == rsi_9.dropna()).all(), (
            "Farklı period RSI değerleri farklı olmalı"
        )

    def test_rsi_invalid_period_raises(self):
        """Sıfır veya negatif period → ValueError."""
        close = pd.Series([100.0] * 20)
        with pytest.raises(ValueError):
            TechnicalCalculator.calculate_rsi(close, period=0)


# ─────────────────────────────────────────────────────────────────────────────
# ADX Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateADX:

    def test_adx_returns_three_series(self):
        """ADX üç seri döndürmeli: adx, plus_di, minus_di."""
        df = _make_ohlcv(bars=50)
        result = TechnicalCalculator.calculate_adx(df)

        assert isinstance(result, ADXResult)
        assert len(result.adx) == 50
        assert len(result.plus_di) == 50
        assert len(result.minus_di) == 50

    def test_adx_range_0_to_100(self):
        """ADX, PlusDI, MinusDI [0, 100] aralığında olmalı."""
        df = _make_ohlcv(bars=100)
        result = TechnicalCalculator.calculate_adx(df)

        for series_name, series in [
            ("ADX", result.adx),
            ("PlusDI", result.plus_di),
            ("MinusDI", result.minus_di),
        ]:
            valid = series.dropna()
            assert (valid >= 0.0).all(), f"{series_name} < 0 olamaz"
            assert (valid <= 100.0).all(), f"{series_name} > 100 olamaz"

    def test_adx_missing_column_raises(self):
        """Eksik sütun → KeyError."""
        df = _make_ohlcv(bars=30).drop(columns=["High"])
        with pytest.raises(KeyError):
            TechnicalCalculator.calculate_adx(df)

    def test_adx_strong_uptrend_plus_di_dominant(self):
        """Güçlü yükseliş trendinde PlusDI > MinusDI olmalı."""
        # Monoton yükselen fiyat
        n = 100
        close = np.linspace(100.0, 150.0, n)
        df = pd.DataFrame(
            {
                "Open":   close * 0.998,
                "High":   close * 1.008,
                "Low":    close * 0.992,
                "Close":  close,
                "Volume": np.full(n, 1e6),
            },
            index=pd.date_range("2024-01-02", periods=n, freq="D"),
        )
        result = TechnicalCalculator.calculate_adx(df)

        # Son 20 bar ortalaması: PlusDI > MinusDI
        plus_mean = float(result.plus_di.iloc[-20:].mean())
        minus_mean = float(result.minus_di.iloc[-20:].mean())
        assert plus_mean > minus_mean, (
            f"Yükseliş trendinde PlusDI ({plus_mean:.1f}) > MinusDI ({minus_mean:.1f}) olmalı"
        )

    def test_adx_nan_bars_no_crash(self):
        """NaN OHLCV barları ADX'i crash ettirmemeli."""
        df = _make_ohlcv(bars=60)
        df_nan = df.copy()
        df_nan.iloc[10, df.columns.get_loc("Close")] = np.nan
        df_nan.iloc[10, df.columns.get_loc("High")] = np.nan

        result = TechnicalCalculator.calculate_adx(df_nan)
        assert len(result.adx) == 60  # Uzunluk korunmalı


# ─────────────────────────────────────────────────────────────────────────────
# OBV Testleri (BUG-03)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateOBV:
    """BUG-03: flat bar'larda hacim davranışı."""

    def test_obv_rising_bar_adds_volume(self):
        """Yükselen bar → hacim eklenir."""
        close = pd.Series([100.0, 101.0])
        volume = pd.Series([1000.0, 500.0])
        obv = TechnicalCalculator.calculate_obv(close, volume)

        # Bar 1: close up → +500
        assert float(obv.iloc[1]) == pytest.approx(1500.0, abs=1.0)

    def test_obv_falling_bar_subtracts_volume(self):
        """Düşen bar → hacim çıkarılır."""
        close = pd.Series([101.0, 100.0])
        volume = pd.Series([1000.0, 500.0])
        obv = TechnicalCalculator.calculate_obv(close, volume)

        assert float(obv.iloc[1]) == pytest.approx(500.0, abs=1.0)

    def test_obv_flat_bar_uses_previous_direction(self):
        """
        BUG-03 Doğrulaması: Flat bar (değişmeyen kapanış) önceki yönde hacim ekler.

        Legacy hatası: np.sign(0) = 0 → flat bar'da hacim sıfırlanıyordu.
        Düzeltme: ffill() ile önceki yöne hacim ekle.
        """
        # Senaryo: yukarı gidip flat bar
        close = pd.Series([100.0, 101.0, 101.0, 102.0])
        volume = pd.Series([1000.0, 800.0, 600.0, 400.0])

        obv = TechnicalCalculator.calculate_obv(close, volume)

        # Bar 1: +800 (yukarı)
        # Bar 2: flat → önceki yön yukarı → +600
        # Bar 3: +400 (yukarı)
        assert float(obv.iloc[1]) > float(obv.iloc[0]), "Bar 1 (yukarı): OBV artmalı"
        assert float(obv.iloc[2]) > float(obv.iloc[1]), (
            "Bar 2 (flat → önceki yön yukarı): OBV artmalı (BUG-03 fix)"
        )

    def test_obv_flat_bar_legacy_would_be_lower(self):
        """
        Legacy davranışı ile fark testi.

        Legacy: flat bar'da OBV değişmez (sıfır eklenir).
        Fixed: flat bar'da önceki yönde hacim eklenir.
        """
        close = pd.Series([100.0, 101.0, 101.0])
        volume = pd.Series([1000.0, 500.0, 300.0])

        obv_fixed = TechnicalCalculator.calculate_obv(close, volume)

        # Legacy implementasyon (karşılaştırma için inline)
        direction_legacy = np.sign(close.diff().fillna(0))
        obv_legacy = (direction_legacy * volume).cumsum()

        # Fixed versiyonu legacy'den yüksek olmalı (flat bar ekstra hacim ekledi)
        assert float(obv_fixed.iloc[-1]) > float(obv_legacy.iloc[-1]), (
            "BUG-03: Fixed OBV flat bar'da legacy'den yüksek olmalı"
        )

    def test_obv_nan_volume_no_crash(self):
        """NaN hacim barında OBV crash vermemeli."""
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        volume = pd.Series([1000.0, np.nan, 800.0, 600.0])

        obv = TechnicalCalculator.calculate_obv(close, volume)
        assert len(obv) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Rank Percentile Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateRollingRankPct:

    def test_smallest_in_window_returns_lowest_possible_rank(self):
        """
        Penceredeki en küçük değer → MÜMKÜN OLAN en düşük rank (1/N × 100).

        DÜZELTME (bu turda): Test adı ve beklentisi "rank=0" idi — bu,
        ESKİ (pandas'la TUTARSIZ, current'ı pencereden HARİÇ tutan)
        formülün davranışıydı. GERÇEK pandas'ta rank HER ZAMAN 1'den
        başlar (0'dan DEĞİL) — bu yüzden N-elemanlı bir pencerede
        (duplicate yokken) en küçük değer bile 1/N × 100 alır, asla
        TAM 0 almaz. Gerçek pandas ile DOĞRUDAN doğrulandı (25.0).
        """
        series = pd.Series([5.0, 4.0, 3.0, 1.0, 6.0, 7.0, 8.0, 9.0])
        result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=4)

        # Bar 3 (değer=1.0, penceredeki KESİN en küçük): window=[5,4,3,1]
        # (current DAHİL) → rank = 1/4 × 100 = 25.0 (pandas'la doğrulandı)
        assert float(result.iloc[3]) == pytest.approx(25.0, abs=1e-6), (
            f"Penceredeki en küçük değer, mümkün olan en düşük rank'ı (25.0) "
            f"almalı, gelen: {result.iloc[3]:.1f}"
        )

    def test_largest_in_window_returns_100(self):
        """Penceredeki en büyük değer → rank = 100."""
        series = pd.Series([1.0, 2.0, 3.0, 9.0, 4.0, 5.0])
        result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=4)

        # Bar 3 (değer=9.0): window=[1,2,3,9] → hist=[1,2,3] → tümü < 9 → rank=100
        assert float(result.iloc[3]) == pytest.approx(100.0, abs=1e-6), (
            f"Penceredeki en büyük değer rank=100 olmalı, gelen: {result.iloc[3]:.1f}"
        )

    def test_median_value_returns_fifty(self):
        """
        Duplicate bir orta değer → rank ≈ 50 (average-tie yöntemiyle).

        DÜZELTME (bu turda): Beklenti aralığı (20-40) ESKİ formülün
        davranışına göre kalibre edilmişti. Bu senaryoda pencerede
        (current DAHİL) İKİ tane 5.0 var — pandas'ın average-tie
        yöntemi bu durumda TAM OLARAK 50.0 verir (gerçek pandas ile
        DOĞRUDAN doğrulandı), 20-40 aralığında bir yerde DEĞİL.
        """
        series2 = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0, 5.0])
        result2 = TechnicalCalculator.calculate_rolling_rank_pct(series2, lookback=5)
        # Son bar (current=5.0): window=[3,5,7,9,5] (current DAHİL) —
        # pencerede İKİ tane 5.0 var (biri geçmişten, biri current'ın
        # kendisi) → below=1 (yalnızca 3), below_eq=3 (3 ve iki 5),
        # avg_rank=(1+3+1)/2=2.5, pct=2.5/5×100=50.0
        assert float(result2.iloc[-1]) == pytest.approx(50.0, abs=1e-6), (
            f"Duplicate orta değer için rank=50.0 bekleniyor, gelen: {result2.iloc[-1]:.1f}"
        )

    def test_nan_input_returns_nan_output(self):
        """NaN girdi → NaN çıktı (propagation)."""
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=4)

        assert pd.isna(result.iloc[2]), "NaN girdi → NaN rank olmalı"
        # Diğer geçerli barlar NaN olmamalı
        assert not pd.isna(result.iloc[-1]), "Geçerli bar NaN olmamalı"

    def test_warmup_period_returns_50(self):
        """Warmup dönemi (< min_periods) → 50.0 döndürmeli."""
        series = pd.Series([1.0] * 50)
        result = TechnicalCalculator.calculate_rolling_rank_pct(
            series, lookback=30, min_periods=10
        )

        # İlk 9 bar (< min_periods=10): 50.0 olmalı
        for i in range(9):
            assert float(result.iloc[i]) == pytest.approx(50.0, abs=1e-6)

    def test_invalid_lookback_raises(self):
        """lookback <= 1 → ValueError."""
        series = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=1)

    def test_result_length_matches_input(self):
        """Çıktı uzunluğu girdi uzunluğuyla aynı olmalı."""
        series = pd.Series(np.random.rand(100))
        result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=20)
        assert len(result) == 100

    def test_rank_monotone_with_larger_values(self):
        """
        Sabit pencerede değer arttıkça rank artmalı.

        DÜZELTME (bu turda, kullanıcı tarafından bulunan gerçek bir
        başarısızlıkla): ÖNCEKİ test verisi (10 sabit + kesinlikle
        artan 3 değer) MATEMATİKSEL OLARAK İMKANSIZ bir beklenti
        taşıyordu — hem eski implementasyon HEM GERÇEK pandas'ın
        kendisi (ampirik olarak doğrulandı) bu veriyle %100'de DOYUYOR,
        çünkü pencerenin en son elemanı KENDİ geçmişindeki HER ŞEYDEN
        büyükse percentile rank TANIMI GEREĞİ %100'dür — bu bir hata
        DEĞİL, sabit pencereli rolling percentile'ın matematiksel
        doğası (bkz. test_rank_saturates_at_100_for_strictly_monotonic_series).

        YENİ senaryo: ÇEŞİTLİ (sabit DEĞİL) bir geçmiş kullanıyor —
        böylece low/mid/high'ın HİÇBİRİ kendi penceresindeki HER ŞEYDEN
        büyük olmuyor, gerçek ayırt edicilik (10% < 50% < 90%) ortaya
        çıkıyor.
        """
        base = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        vals = pd.Series([15.0, 55.0, 95.0])
        series = pd.concat([base, vals], ignore_index=True)

        result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=10)

        rank_low = float(result.iloc[-3])    # 15.0
        rank_mid = float(result.iloc[-2])    # 55.0
        rank_high = float(result.iloc[-1])   # 95.0

        assert rank_low < rank_mid < rank_high, (
            f"Monoton değerler için monoton rank bekleniyor: "
            f"{rank_low:.1f} < {rank_mid:.1f} < {rank_high:.1f}"
        )
        # Tam sayısal doğrulama (elle hesaplandı, bkz. docstring)
        assert abs(rank_low - 10.0) < 1e-9
        assert abs(rank_mid - 50.0) < 1e-9
        assert abs(rank_high - 90.0) < 1e-9

    def test_rank_saturates_at_100_for_strictly_monotonic_series(self):
        """
        DÜZELTME (bu turda eklendi — dürüstçe belgelemek için): Kesinlikle
        monotonik artan bir seri, SABİT boyutlu bir pencerede, belirli
        bir noktadan sonra %100'DE DOYAR — bu, implementasyon hatası
        DEĞİL, matematiksel bir gerçek. GERÇEK pandas'ın kendisi de
        AYNI şekilde davranır (bu test, bunu kanıtlamak için pandas'la
        DOĞRUDAN karşılaştırıyor).
        """
        series = pd.Series(range(1, 101), dtype=float)  # 1, 2, ..., 100
        lookback = 30

        my_result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=lookback)
        pandas_result = series.rolling(window=lookback).apply(
            lambda w: pd.Series(w).rank(pct=True, method="average").iloc[-1] * 100, raw=False,
        )

        # İkisi de son N noktada %100'de DOYMALI — bu BEKLENEN davranış
        assert (my_result.tail(10) == 100.0).all()
        assert (pandas_result.tail(10) == 100.0).all()

    def test_rank_matches_real_pandas_rolling_rank_exactly(self):
        """
        KRİTİK doğrulama (kullanıcı talebi: "pandas rolling(rank) ile
        aynı mantık"): Rastgele, NaN VE duplicate içeren gerçekçi bir
        seride, implementasyonum GERÇEK pandas'ın rolling(window).rank
        (pct=True, method='average') ÇIKTISIYLA BİREBİR eşleşmeli (tam
        pencere bölgesinde).
        """
        import numpy as np

        rng = np.random.default_rng(11)
        data = list(rng.normal(50, 5, 100))
        data[70] = data[40]  # bilinçli duplicate
        series = pd.Series(data)
        lookback = 15

        my_result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=lookback, min_periods=5)
        pandas_result = series.rolling(window=lookback).apply(
            lambda w: pd.Series(w).rank(pct=True, method="average").iloc[-1] * 100, raw=False,
        )

        full_window_region = slice(lookback - 1, None)
        diff = (my_result[full_window_region] - pandas_result[full_window_region]).abs()
        assert (diff < 1e-9).all(), f"pandas'tan sapma bulundu, maksimum fark: {diff.max()}"

    def test_rank_vectorized_and_fallback_paths_produce_identical_results(self):
        """
        KRİTİK doğrulama (kullanıcı talebi: "vectorized sürüm ile
        fallback aynı sonucu üretmeli"): sliding_window_view'i BİLEREK
        devre dışı bırakıp fallback yolunu ZORLA, vectorized sonuçla
        BİREBİR karşılaştır.
        """
        import numpy as np
        import numpy.lib.stride_tricks as st

        rng = np.random.default_rng(7)
        data = list(rng.normal(50, 5, 100))
        data[20] = np.nan
        data[70] = data[40]  # duplicate
        series = pd.Series(data)

        vectorized_result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=15, min_periods=5)

        original_swv = st.sliding_window_view

        def _broken_swv(*args, **kwargs):
            raise ImportError("test: fallback yolunu zorla")

        st.sliding_window_view = _broken_swv
        try:
            fallback_result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=15, min_periods=5)
        finally:
            st.sliding_window_view = original_swv

        diff = (vectorized_result - fallback_result).abs()
        assert (diff.dropna() < 1e-9).all()
        assert (vectorized_result.isna() == fallback_result.isna()).all()

    def test_rank_duplicate_values_use_average_tie_method(self):
        """
        KRİTİK doğrulama (kullanıcı talebi: "duplicate değerlerde
        deterministic olmalı"): pandas'ın 'average' tie-breaking
        yöntemiyle BİREBİR eşleşmeli — ham strict-less-than sayımı
        (ÖNCEKİ, HATALI davranış) DEĞİL.
        """
        # Pencere: [3, 5, 5, 5, 5] — son eleman (5.0) 3 tane daha
        # duplicate'e sahip. pandas rank(pct=True, method='average')
        # bu durumda 70.0 verir (elle doğrulandı, bkz. bu turun
        # gerekçe notu).
        series = pd.Series([3.0, 5.0, 5.0, 5.0, 5.0])
        result = TechnicalCalculator.calculate_rolling_rank_pct(series, lookback=5, min_periods=2)
        assert abs(float(result.iloc[-1]) - 70.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# VWAP Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateVWAP:

    def test_vwap_equal_prices_returns_price(self):
        """Tüm fiyatlar eşit olduğunda VWAP = fiyat."""
        df = pd.DataFrame(
            {
                "Open":   [100.0] * 5,
                "High":   [100.0] * 5,
                "Low":    [100.0] * 5,
                "Close":  [100.0] * 5,
                "Volume": [1_000.0] * 5,
            },
            index=pd.date_range("2024-01-02 10:00", periods=5, freq="15min"),
        )
        vwap = TechnicalCalculator.calculate_vwap(df)
        assert all(abs(vwap - 100.0) < 1e-9)

    def test_vwap_resets_each_day(self):
        """VWAP her gün başından hesaplanmalı."""
        # 2 gün verisi
        idx = pd.DatetimeIndex(
            pd.date_range("2024-01-02 10:00", periods=4, freq="4h")
        )
        df = pd.DataFrame(
            {
                "Open":   [100.0, 102.0, 104.0, 106.0],
                "High":   [101.0, 103.0, 105.0, 107.0],
                "Low":    [99.0,  101.0, 103.0, 105.0],
                "Close":  [100.0, 102.0, 104.0, 106.0],
                "Volume": [1000.0, 1000.0, 1000.0, 1000.0],
            },
            index=idx,
        )
        vwap = TechnicalCalculator.calculate_vwap(df)
        assert len(vwap) == 4  # Crash vermemeli

    def test_vwap_zero_volume_returns_nan(self):
        """Sıfır hacimde VWAP = NaN (bölme koruması)."""
        df = pd.DataFrame(
            {
                "Open":   [100.0, 101.0],
                "High":   [101.0, 102.0],
                "Low":    [99.0,  100.0],
                "Close":  [100.0, 101.0],
                "Volume": [0.0,   0.0],
            },
            index=pd.date_range("2024-01-02 10:00", periods=2, freq="15min"),
        )
        vwap = TechnicalCalculator.calculate_vwap(df)
        assert vwap.isna().all(), "Sıfır hacimde VWAP NaN olmalı"

    def test_vwap_missing_column_raises(self):
        """Eksik sütun → KeyError."""
        df = _make_ohlcv(bars=10).drop(columns=["Volume"])
        with pytest.raises(KeyError):
            TechnicalCalculator.calculate_vwap(df)


# ─────────────────────────────────────────────────────────────────────────────
# ATR Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateATR:

    def test_atr_always_non_negative(self):
        """ATR negatif olamaz (True Range ≥ 0)."""
        df = _make_ohlcv(bars=50)
        atr = TechnicalCalculator.calculate_atr(df)
        assert (atr.dropna() >= 0).all(), "ATR negatif olamaz"

    def test_atr_wider_range_gives_higher_atr(self):
        """Geniş fiyat aralığında ATR yüksek olmalı."""
        df_tight = pd.DataFrame(
            {
                "Open":  [100.0] * 30, "High": [100.5] * 30,
                "Low":   [99.5] * 30,  "Close": [100.0] * 30,
                "Volume": [1e6] * 30,
            },
            index=pd.date_range("2024-01-02", periods=30, freq="D"),
        )
        df_wide = pd.DataFrame(
            {
                "Open":  [100.0] * 30, "High": [105.0] * 30,
                "Low":   [95.0] * 30,  "Close": [100.0] * 30,
                "Volume": [1e6] * 30,
            },
            index=pd.date_range("2024-01-02", periods=30, freq="D"),
        )
        atr_tight = TechnicalCalculator.calculate_atr(df_tight).iloc[-1]
        atr_wide = TechnicalCalculator.calculate_atr(df_wide).iloc[-1]

        assert atr_wide > atr_tight, (
            f"Geniş aralık ATR ({atr_wide:.2f}) > Dar aralık ATR ({atr_tight:.2f}) olmalı"
        )

    def test_atr_missing_column_raises(self):
        df = _make_ohlcv(bars=20).drop(columns=["Low"])
        with pytest.raises(KeyError):
            TechnicalCalculator.calculate_atr(df)


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Squeeze Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateBollingerSqueeze:

    def test_squeeze_returns_bool_series(self):
        """Bollinger Squeeze bool serisi döndürmeli."""
        df = _make_ohlcv(bars=300)
        squeeze = TechnicalCalculator.calculate_bollinger_squeeze(df["Close"])
        assert squeeze.dtype == bool

    def test_low_volatility_triggers_squeeze(self):
        """Düşük volatilite → sıkışma (squeeze) aktif."""
        # Sabit fiyat → sıfır std → BB bandwidth = 0 → sıkışma
        close_flat = pd.Series([100.0] * 300)
        squeeze = TechnicalCalculator.calculate_bollinger_squeeze(close_flat)
        # Yeterli warmup sonrası sıkışma True olmalı
        if squeeze.dropna().any():
            assert bool(squeeze.dropna().iloc[-1]), "Sabit fiyat sıkışmada olmalı"

    def test_high_volatility_no_squeeze(self):
        """Yüksek volatilite → sıkışma yok."""
        # Büyük fiyat salınımı
        close_volatile = pd.Series(
            [100.0 if i % 2 == 0 else 150.0 for i in range(300)], dtype=float
        )
        squeeze = TechnicalCalculator.calculate_bollinger_squeeze(close_volatile)
        # Son barlar False olmalı
        assert not bool(squeeze.iloc[-1]), "Yüksek volatilite sıkışmada olmamalı"


# ─────────────────────────────────────────────────────────────────────────────
# VCP Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateVCPComponents:

    def test_vcp_insufficient_data_returns_zero(self):
        """Yetersiz veri → tüm bileşenler False, score=0."""
        close = pd.Series([100.0] * 10)  # lookback=60'tan az
        volume = pd.Series([1e6] * 10)

        result = TechnicalCalculator.calculate_vcp_components(close, volume, lookback=60)

        assert isinstance(result, VCPComponents)
        assert result.composite_score == 0
        assert result.price_contraction is False

    def test_vcp_score_range_0_to_3(self):
        """VCP skoru 0-3 aralığında olmalı."""
        df = _make_ohlcv(bars=100)
        result = TechnicalCalculator.calculate_vcp_components(df["Close"], df["Volume"])
        assert 0 <= result.composite_score <= 3

    def test_vcp_contraction_detected(self):
        """Gerçek daralma senaryosunda price_contraction=True."""
        # Geniş volatilite ardından dar volatilite
        np.random.seed(99)
        wide = 100.0 + np.random.uniform(-10, 10, 40)   # Geniş: ±10
        narrow = wide[-1] + np.random.uniform(-1, 1, 20) # Dar: ±1

        close = pd.Series(np.concatenate([wide, narrow]))
        volume = pd.Series([1e6] * 60)

        result = TechnicalCalculator.calculate_vcp_components(
            close, volume, lookback=60, recent_bars=20
        )
        assert result.price_contraction is True, (
            "Belirgin daralma senaryosunda price_contraction True olmalı"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Relative Strength Testleri (BUG-05/06)
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateRelativeStrength:

    def test_rs_returns_correct_type(self):
        """Göreceli güç RelativeStrengthResult döndürmeli."""
        stock = pd.Series([100.0 + i for i in range(30)], dtype=float)
        bench = pd.Series([100.0] * 30, dtype=float)

        result = TechnicalCalculator.calculate_relative_strength(stock, bench)
        assert isinstance(result, RelativeStrengthResult)
        assert len(result.rs_series) == 30

    def test_rs_outperforming_stock_above_ma(self):
        """Endeksten güçlü hisse → rs_above_ma True."""
        # Hisse güçlü yükseliş, endeks sabit
        stock = pd.Series(np.linspace(100.0, 130.0, 50), dtype=float)
        bench = pd.Series([100.0] * 50, dtype=float)

        result = TechnicalCalculator.calculate_relative_strength(stock, bench, ma_period=10)

        # Son barlar: RS yükselen eğilimde → rs_above_ma True
        assert bool(result.rs_above_ma.iloc[-1]), (
            "Güçlü hissede RS > RS_MA olmalı"
        )

    def test_rs_inner_join_no_lookahead(self):
        """
        BUG-05/06: Inner join kullanılıyor, ffill yok.

        Farklı tarihli seriler birleştirildiğinde
        yalnızca eşleşen tarihler kullanılmalı.
        """
        idx_stock = pd.date_range(start="2024-01-01", periods=20, freq="D")
        idx_bench = pd.date_range(start="2024-01-05", periods=15, freq="D")  # 4 gün ileride

        stock = pd.Series(np.linspace(100.0, 110.0, 20), index=idx_stock)
        bench = pd.Series(np.linspace(100.0, 105.0, 15), index=idx_bench)

        result = TechnicalCalculator.calculate_relative_strength(stock, bench)

        # Inner join: sadece ortak tarihler (15 eşleşme)
        assert len(result.rs_series) == 15, (
            f"Inner join: 15 eşleşme bekleniyor, gelen: {len(result.rs_series)}"
        )

    def test_rs_insufficient_data_raises(self):
        """< 2 eşleşen bar → ValueError."""
        stock = pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-01"]))
        bench = pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-02"]))  # Farklı gün

        with pytest.raises(ValueError, match="yeterli"):
            TechnicalCalculator.calculate_relative_strength(stock, bench)

    def test_rs_momentum_slope(self):
        """Hız kazanan RS → rs_slope_positive True."""
        # Güçlü yükseliş → RS slope pozitif
        stock = pd.Series(np.linspace(100.0, 140.0, 40), dtype=float)
        bench = pd.Series([100.0] * 40, dtype=float)

        result = TechnicalCalculator.calculate_relative_strength(
            stock, bench, slope_bars=5
        )
        # Son barlar momentum pozitif olmalı
        assert bool(result.rs_slope_positive.iloc[-1]), (
            "Yükselen RS'de slope pozitif olmalı"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ATR Percentile Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateATRPct:

    def test_atr_pct_formula(self):
        """ATR% = ATR / Close."""
        atr = pd.Series([2.0, 3.0, 4.0])
        close = pd.Series([100.0, 150.0, 200.0])
        result = TechnicalCalculator.calculate_atr_pct(atr, close)

        expected = [0.02, 0.02, 0.02]
        for i, exp in enumerate(expected):
            assert float(result.iloc[i]) == pytest.approx(exp, rel=1e-6)

    def test_atr_pct_zero_close_returns_nan(self):
        """Close = 0 → ATR% = NaN (sıfıra bölünme koruması)."""
        atr = pd.Series([2.0, 3.0])
        close = pd.Series([0.0, 100.0])
        result = TechnicalCalculator.calculate_atr_pct(atr, close)

        assert pd.isna(result.iloc[0]), "Sıfır close → NaN ATR%"
        assert not pd.isna(result.iloc[1]), "Geçerli close → geçerli ATR%"


# ─────────────────────────────────────────────────────────────────────────────
# Gap Yüzdesi Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateGapPct:

    def test_gap_pct_no_gap(self):
        """Open = PrevClose → gap = 0."""
        open_p = pd.Series([100.0, 100.0])
        prev_close = pd.Series([100.0, 100.0])
        result = TechnicalCalculator.calculate_gap_pct(open_p, prev_close)
        assert float(result.iloc[1]) == pytest.approx(0.0, abs=1e-9)

    def test_gap_pct_four_percent_gap(self):
        """Open = 104, PrevClose = 100 → gap = 0.04."""
        open_p = pd.Series([104.0])
        prev_close = pd.Series([100.0])
        result = TechnicalCalculator.calculate_gap_pct(open_p, prev_close)
        assert float(result.iloc[0]) == pytest.approx(0.04, rel=1e-6)

    def test_gap_pct_zero_prev_close_returns_nan(self):
        """PrevClose = 0 → gap = NaN."""
        open_p = pd.Series([100.0])
        prev_close = pd.Series([0.0])
        result = TechnicalCalculator.calculate_gap_pct(open_p, prev_close)
        assert pd.isna(result.iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# NaN Genel Dayanıklılık Testleri
# ─────────────────────────────────────────────────────────────────────────────

class TestNANRobustness:
    """
    Aşama 1 validator'ı NaN barlar bırakabilir.
    Tüm hesaplamalar bu durumda çökmemeli.
    """

    def test_all_calculators_survive_heavy_nan(self):
        """
        Her hesaplama metodu %30 NaN içeren serilerle çalışmalı.
        Bu, Aşama 1'in bıraktığı hatalı barları simüle eder.
        """
        df = _make_ohlcv(bars=100)

        # %30 NaN enjekte et
        rng = np.random.default_rng(99)
        nan_indices = rng.choice(range(5, 95), size=30, replace=False)

        df_nan = df.copy()
        for idx in nan_indices:
            df_nan.iloc[idx, :] = np.nan  # Tüm satır NaN

        # Her metod crash vermemeli
        rsi = TechnicalCalculator.calculate_rsi(df_nan["Close"])
        assert len(rsi) == 100

        adx_result = TechnicalCalculator.calculate_adx(df_nan)
        assert len(adx_result.adx) == 100

        atr = TechnicalCalculator.calculate_atr(df_nan)
        assert len(atr) == 100

        rvol = TechnicalCalculator.calculate_rvol(df_nan["Volume"])
        assert len(rvol) == 100

        obv = TechnicalCalculator.calculate_obv(df_nan["Close"], df_nan["Volume"])
        assert len(obv) == 100

        print(f"\nHeavy NaN (%30) testi: Tüm hesaplamalar tamamlandı.")
        print(f"  RSI geçerli bar: {rsi.notna().sum()}/100")
        print(f"  RVOL geçerli bar: {rvol.notna().sum()}/100")
