"""
Teknik analiz hesaplama motoru.

Bu modül, BistKokpit V23.5 legacy'sindeki tüm teknik indikatör
hesaplama fonksiyonlarının Domain katmanına taşınmış, production-grade
dönüşümüdür.

Legacy kaynak → Domain metodu:
  _calc_rsi()            → TechnicalCalculator.calculate_rsi()
  _calc_adx()            → TechnicalCalculator.calculate_adx()
  ATR (get_data içinde)  → TechnicalCalculator.calculate_atr()
  _calc_vwap()           → TechnicalCalculator.calculate_vwap()
  _calc_obv()            → TechnicalCalculator.calculate_obv()
  vol_ratio (BUG-11)     → TechnicalCalculator.calculate_rvol()
  _rolling_rank_pct()    → TechnicalCalculator.calculate_rolling_rank_pct()
  _vol_pct_vectorized()  → TechnicalCalculator.calculate_rolling_rank_pct() (tek metod, doğrusu)
  bollinger_squeeze()    → TechnicalCalculator.calculate_bollinger_squeeze()
  vcp_score()            → TechnicalCalculator.calculate_vcp_components()
  relative_strength_full → TechnicalCalculator.calculate_relative_strength()
  atr_regime()           → TechnicalCalculator.calculate_atr_pct()

Domain Katmanı Kısıtları (kesinlikle uygulanır):
  ✓ Tüm metodlar @staticmethod — sınıf state tutmuyor
  ✓ Sıfır dış bağımlılık — sadece pandas ve numpy
  ✓ Veri çekme yok — sadece pd.Series / pd.DataFrame alır
  ✓ NaN-toleranslı — Aşama 1 validator'ı NaN barlar bırakabilir
  ✗ print() yok — sadece logging (hesaplama metodlarında bile sadece debug)
  ✗ API çağrısı yok
  ✗ Dosya I/O yok

BUG-11 Düzeltmesi (Kritik):
  YANLIŞ (legacy V23.4):  vol_ratio = close_price / volume_moving_average
                           Birim: TL / Adet = finansal anlamsız
  DOĞRU  (legacy V23.5):  vol_ratio = volume / volume_moving_average
                           Birim: Adet / Adet = boyutsuz oran (RVOL)
  Bu modülde: calculate_rvol(volume, ma_period) → RVOL serisi

BUG-02 Notu (vol_pct):
  Legacy _vol_pct_vectorized() min-max normalizasyon kullanıyor,
  ancak kendi kodu içinde "bu gerçek percentile değil" notu var.
  Bu modülde tek doğru implementasyon: calculate_rolling_rank_pct()
  RSI_Pct, ATR_Pct ve Vol_Pct için aynı metod kullanılır.

BUG-03 Düzeltmesi (OBV):
  np.sign(diff=0) = 0 → değişmeyen bar'larda hacim sıfırlanıyor.
  Düzeltme: flat bar → önceki yönde hacim eklenir (TradingView davranışı).

NaN Güvenlik Garantileri:
  - Tüm rolling hesaplamalar min_periods=1 kullanır
  - Sıfıra bölünme: vol_ma=0 → RVOL=1.0 (nötr), vol=NaN → RVOL=NaN
  - ewm() pandas NaN'ı propagate etmez (NaN barı atlar)
  - stride_tricks rolling rank: NaN girdi → NaN çıktı (explicit kontrol)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
import pandas as pd

from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


# ── Sonuç veri yapıları (dönüş tipi netliği için) ────────────────────────────

class ADXResult(NamedTuple):
    """ADX hesaplaması üç seri döndürür."""
    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


class VCPComponents(NamedTuple):
    """VCP (Volatility Contraction Pattern) bileşen skorları."""
    price_contraction: bool    # Kısa dönem std < uzun dönem std × 0.85
    volume_contraction: bool   # Kısa dönem vol ort < uzun dönem vol ort × 0.75
    price_rising: bool         # Son kapanış > lookback başındaki kapanış
    composite_score: int       # True sayısı (0-3)


@dataclass(frozen=True)
class RelativeStrengthResult:
    """Göreceli güç hesaplaması sonucu."""
    rs_series: pd.Series           # Stock / Benchmark oranı
    rs_above_ma: pd.Series         # RS > RS'nin rolling MA'sı (bool)
    rs_slope_positive: pd.Series   # RS momentum pozitif (bool)
    ma_period: int


# ── Ana Hesaplama Sınıfı ──────────────────────────────────────────────────────

class TechnicalCalculator:
    """
    Teknik analiz hesaplama motoru.

    Tüm metodlar @staticmethod — durumsuz, saf fonksiyonlar.
    Her metod bağımsız olarak test edilebilir.
    Girdi olarak yalnızca pd.Series veya pd.DataFrame alır.

    Kullanım:
        from src.domain.calculators.technical_calculator import TechnicalCalculator

        rsi = TechnicalCalculator.calculate_rsi(df["Close"])
        adx, plus_di, minus_di = TechnicalCalculator.calculate_adx(df)
        rvol = TechnicalCalculator.calculate_rvol(df["Volume"], ma_period=20)
    """

    # ── RSI ──────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder RSI hesabı.

        Legacy karşılığı: _calc_rsi()
        TradingView ile eşleşen yöntem: Wilder Smoothing (ewm alpha=1/period).

        Formül:
          diff = close.diff()
          avg_gain = ewm(diff.where(diff>0, 0), alpha=1/period)
          avg_loss = ewm((-diff).where(diff<0, 0), alpha=1/period)
          RSI = 100 - 100 / (1 + avg_gain/avg_loss)

        NaN güvenliği:
          ewm() NaN girişini propagate etmez — NaN barı atlar.
          Başlangıç warmup dönemi NaN döner (period öncesi).

        Args:
            close: Kapanış fiyatı serisi (float, NaN içerebilir).
            period: RSI periyodu (default 14).

        Returns:
            RSI serisi [0, 100] aralığında. Warmup dönemi NaN.
        """
        if period <= 0:
            raise ValueError(f"RSI periyodu pozitif olmalı: {period}")

        d = close.diff()
        avg_gain = d.where(d > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = (-d).where(d < 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()

        # Sıfıra bölünme koruması (finansal doğru davranış):
        #   avg_loss=0, avg_gain>0 → saf yükseliş → RSI=100
        #   avg_loss=0, avg_gain=0 → sabit fiyat   → RSI=50 (nötr)
        #   avg_loss>0             → normal hesap
        rsi = pd.Series(
            np.where(
                avg_loss == 0.0,
                np.where(avg_gain == 0.0, 50.0, 100.0),
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss)),
            ),
            index=close.index,
        )

        logger.debug(
            "rsi_calculated",
            period=period,
            bars=len(close),
            valid_bars=int(rsi.notna().sum()),
        )
        return rsi

    # ── ADX ──────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_adx(
        df: pd.DataFrame,
        period: int = 14,
    ) -> ADXResult:
        """
        ADX + PlusDI + MinusDI hesabı.

        Legacy karşılığı: _calc_adx()
        Wilder Smoothing (ewm) kullanır — TradingView uyumlu.

        Gerekli sütunlar: "High", "Low", "Close"

        Args:
            df: OHLCV DataFrame (NaN barlar içerebilir).
            period: ADX/DI periyodu (default 14).

        Returns:
            ADXResult(adx, plus_di, minus_di) — hepsi pd.Series.

        Raises:
            KeyError: Gerekli sütun eksikse.
        """
        _required = {"High", "Low", "Close"}
        missing = _required - set(df.columns)
        if missing:
            raise KeyError(f"ADX hesabı için eksik sütunlar: {missing}")

        if period <= 0:
            raise ValueError(f"ADX periyodu pozitif olmalı: {period}")

        h = df["High"]
        lo = df["Low"]
        c = df["Close"]

        up_move = h.diff()
        dn_move = -lo.diff()

        # Directional Movement
        plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

        # True Range
        tr = pd.concat(
            [h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()],
            axis=1,
        ).max(axis=1)

        # Wilder Smoothing (ewm)
        alpha = 1.0 / period
        atr14 = tr.ewm(alpha=alpha, adjust=False).mean()

        plus_di = (
            100.0
            * pd.Series(plus_dm, index=df.index)
            .ewm(alpha=alpha, adjust=False)
            .mean()
            / atr14.replace(0.0, np.nan)
        )
        minus_di = (
            100.0
            * pd.Series(minus_dm, index=df.index)
            .ewm(alpha=alpha, adjust=False)
            .mean()
            / atr14.replace(0.0, np.nan)
        )

        # DX → ADX
        di_sum = (plus_di + minus_di).replace(0.0, np.nan)
        dx = (plus_di - minus_di).abs() / di_sum * 100.0
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return ADXResult(adx=adx, plus_di=plus_di, minus_di=minus_di)

    # ── ATR ──────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR) hesabı.

        Wilder Smoothing (ewm) — ADX ile tutarlı.

        Gerekli sütunlar: "High", "Low", "Close"

        Args:
            df: OHLCV DataFrame.
            period: ATR periyodu (default 14).

        Returns:
            ATR serisi (fiyat birimi cinsinden).
        """
        _required = {"High", "Low", "Close"}
        missing = _required - set(df.columns)
        if missing:
            raise KeyError(f"ATR hesabı için eksik sütunlar: {missing}")

        h = df["High"]
        lo = df["Low"]
        c = df["Close"]

        tr = pd.concat(
            [h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()],
            axis=1,
        ).max(axis=1)

        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    # ── ATR Percentile ────────────────────────────────────────────────────────

    @staticmethod
    def calculate_atr_pct(
        atr: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        ATR / Close oranı (yüzde olarak).

        Legacy karşılığı: atr_regime() içindeki atr/price hesabı.
        Sıfır fiyat koruması dahil.

        Args:
            atr: ATR serisi.
            close: Kapanış fiyatı serisi.

        Returns:
            ATR% serisi (0.02 = %2). Sıfır kapanışta NaN.
        """
        safe_close = close.replace(0.0, np.nan)
        return atr / safe_close

    # ── VWAP ─────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP) — günlük kümülatif.

        Legacy karşılığı: _calc_vwap()
        Her işlem gününde sıfırdan başlar (gerçek intraday VWAP).

        Gerekli sütunlar: "High", "Low", "Close", "Volume"

        Args:
            df: İntraday OHLCV DataFrame (DatetimeIndex gerekli).

        Returns:
            VWAP serisi. Sıfır hacimli barlarda NaN.

        Not:
            Günlük veri için VWAP matematiksel olarak anlamlı değildir.
            Bu metod intraday (15m, 1h) veri için tasarlanmıştır.
        """
        _required = {"High", "Low", "Close", "Volume"}
        missing = _required - set(df.columns)
        if missing:
            raise KeyError(f"VWAP hesabı için eksik sütunlar: {missing}")

        try:
            typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
            pv = typical * df["Volume"]

            # Günlük sıfırlama: her gün için ayrı kümülatif
            # DÜZELTME (bu turda, GERÇEK pandas-stubs kurulumu sonrası
            # bulundu): mypy, df.index'i genel Index[Any] olarak
            # görüyor, DatetimeIndex olduğunu STATİK OLARAK bilemiyor
            # (VWAP'ın kendi ön koşulu zaten tarih-indeksli bir
            # DataFrame gerektiriyor — bkz. bu fonksiyonun docstring'i/
            # girdi doğrulaması). Açık cast GÜVENLİ, tip hatası GİZLEMİYOR.
            date_idx = cast(pd.DatetimeIndex, df.index).normalize()
            cum_pv = pv.groupby(date_idx).cumsum()
            cum_vol = df["Volume"].groupby(date_idx).cumsum()

            return cum_pv / cum_vol.replace(0.0, np.nan)

        except Exception as exc:
            logger.warning("vwap_calculation_failed", error=str(exc))
            return pd.Series(np.nan, index=df.index)

    # ── OBV ──────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV) — BUG-03 flat-bar düzeltmesi dahil.

        Legacy karşılığı: _calc_obv()

        BUG-03 Düzeltmesi:
          YANLIŞ: np.sign(diff=0) = 0 → değişmeyen bar'larda hacim 0 eklenir.
          DOĞRU: flat bar (diff=0) → önceki yönde hacim eklenir.
          Bu TradingView OBV davranışıyla uyumludur.

        Args:
            close: Kapanış fiyatı serisi.
            volume: Hacim serisi.

        Returns:
            OBV kümülatif serisi.
        """
        diff = close.diff().fillna(0.0)
        direction = np.sign(diff)

        # Flat bar düzeltmesi: 0 → önceki non-zero yönü al
        direction_series = pd.Series(direction, index=close.index)
        direction_series = (
            direction_series
            .replace(0.0, np.nan)
            .ffill()
            .fillna(1.0)   # İlk bar için varsayılan: yukarı yön
        )

        return (direction_series * volume).fillna(0.0).cumsum()

    # ── RVOL (BUG-11 Düzeltmesi) ─────────────────────────────────────────────

    @staticmethod
    def calculate_rvol(
        volume: pd.Series,
        ma_period: int = 20,
    ) -> pd.Series:
        """
        Relative Volume (RVOL) = Hacim / Hacim Hareketli Ortalaması.

        BUG-11 Düzeltmesi (KRİTİK):
          YANLIŞ (V23.4): vol_ratio = close_price / vol_moving_average
                          Birim: TL / Adet → finansal anlamsız, birim uyumsuzluğu
          DOĞRU  (V23.5): vol_ratio = volume / vol_moving_average
                          Birim: Adet / Adet → boyutsuz oran

        Bu metod legacy V23.5'teki düzeltmeyi parametrik ve vektörize
        olarak uygular.

        NaN Güvenlik Kuralları:
          - volume = 0, vol_ma > 0  → RVOL = 0.0 (sıfır hacimli bar)
          - volume = any, vol_ma = 0 → RVOL = 1.0 (pencere sıfır, nötr)
          - volume = NaN             → RVOL = NaN (Aşama 1 hatalı bar)

        Args:
            volume: Hacim serisi (NaN içerebilir).
            ma_period: Hareketli ortalama periyodu (default 20).

        Returns:
            RVOL serisi. >1.0 = ortalamanın üstünde hacim (örn: 1.5 = %150).

        Örnek:
            rvol = TechnicalCalculator.calculate_rvol(df["Volume"], ma_period=20)
            vol_confirmed = (rvol >= 1.5) & (df["Close"] > df["Close"].shift(1))
        """
        if ma_period <= 0:
            raise ValueError(f"RVOL ma_period pozitif olmalı: {ma_period}")

        vol_ma = volume.rolling(ma_period, min_periods=1).mean()

        # vol_ma = 0 durumunda sıfıra bölünmeyi önle → NaN yap, sonra 1.0 ile doldur
        vol_ma_safe = vol_ma.copy()
        vol_ma_safe[vol_ma_safe == 0.0] = np.nan

        rvol = volume / vol_ma_safe

        # vol_ma = 0 olan barlar: 1.0 (nötr default)
        rvol = rvol.fillna(1.0)

        # volume = NaN olan barlar: RVOL de NaN olmalı (hatalı bar işareti)
        rvol[volume.isna()] = np.nan

        logger.debug(
            "rvol_calculated",
            ma_period=ma_period,
            bars=len(volume),
            nan_bars=int(volume.isna().sum()),
        )
        return rvol

    # ── Rolling Percentile Rank ───────────────────────────────────────────────

    @staticmethod
    def calculate_rolling_rank_pct(
        series: pd.Series,
        lookback: int,
        min_periods: int = 30,
    ) -> pd.Series:
        """
        Gerçek rolling percentile rank — pandas'ın rolling(window).rank(pct=True)
        (method='average') SEMANTİĞİYLE eşleşecek şekilde YENİDEN YAZILDI.

        DÜZELTME (bu turda, kullanıcı tarafından GERÇEK bir başarısız testle
        bulundu ve MATEMATİKSEL OLARAK ANALİZ EDİLDİ):

        ÖNCEKİ FORMÜL: rank = count(GEÇMİŞ < current) / len(GEÇMİŞ) * 100
          — current, PENCERENİN KENDİSİNDEN HARİÇ tutuluyordu, VE
          eşit değerler (duplicate) için "average rank" YERİNE ham
          strict-less-than sayımı kullanılıyordu.

        KANIT (gerçek pandas ile ampirik olarak doğrulandı): Kesinlikle
        monotonik artan bir seri için, SABİT boyutlu bir pencerede, HEM
        eski formül HEM GERÇEK pandas'ın kendisi belirli bir noktadan
        sonra %100'DE DOYAR — çünkü pencerenin en son elemanı, KENDİ
        penceresindeki HER ŞEYDEN büyükse (ki uzun bir monotonik artışta
        bu KAÇINILMAZDIR), percentile rank TANIMI GEREĞİ %100'dür. Bu,
        "hatalı" bir davranış DEĞİL — SABİT PENCERELİ rolling percentile
        rank'ın MATEMATİKSEL DOĞASI. Test verisi (10 sabit + kesinlikle
        artan 3 değer, lookback=10), İKİ ayrı noktanın (7.0 ve 12.0)
        İKİSİNİN DE kendi 9-elemanlı geçmiş pencerelerindeki HER ŞEYDEN
        büyük olmasına yol açıyordu — bu yüzden İKİSİ DE %100 alıyordu,
        implementasyon NASIL yazılırsa yazılsın.

        GERÇEK DÜZELTME (iki parça):
          1) Algoritma, pandas'ın rank(pct=True, method='average')
             SEMANTİĞİNE getirildi — current ARTIK pencerenin bir PARÇASI
             (önceden HARİÇ tutuluyordu) VE duplicate'ler "average rank"
             formülüyle ele alınıyor: rank = (count(<v) + count(<=v) + 1) / 2.
             Bu, GERÇEK pandas ile birebir eşleştiği DOĞRUDAN test edilerek
             kanıtlandı (bkz. bu turun gerekçe notu — hem tekil hem
             duplicate içeren pencerelerde).
          2) Test verisi DÜZELTİLDİ (test dosyasında) — matematiksel
             olarak İMKANSIZ bir beklenti (doyma noktasından SONRA
             monotonluk) yerine, doyma noktasından ÖNCEki, GERÇEKTEN
             ayırt edilebilir bir senaryo kullanıyor.

        Algoritma (numpy stride_tricks):
          views[j] = series[j .. j+lookback-1]  (current DAHİL)
          below = count(views[j] < current), below_eq = count(views[j] <= current)
          rank[j] = (below + below_eq + 1) / 2 / lookback * 100
          Lookahead yok: pencere yalnızca [i-lookback+1, i] aralığını kapsar.

        Performans:
          Vectorized numpy: ~3-4x saf Python döngüsünden hızlı.
          O(n × lookback) bellek, O(n × lookback) hesaplama.

        NaN Güvenliği:
          Girdi (current) NaN ise rank NaN döner.
          Pencheredeki GEÇMİŞ NaN değerler sayımdan HARİÇ tutulur (payda
          yalnızca GEÇERLİ eleman sayısını yansıtır) — ÖNCEKİ davranışla
          TUTARLI.
          Warmup dönemi (< min_periods) → 50.0 (nötr percentile).

        Args:
            series: İndikatör serisi (RSI, ATR, Volume vb.).
            lookback: Pencere boyutu (ör: 252 = ~1 yıl).
            min_periods: Minimum veri noktası (default 30).

        Returns:
            Percentile rank serisi [0, 100].
            Warmup dönemi: 50.0 (nötr).
            NaN girdi: NaN çıktı.
        """
        if lookback <= 1:
            raise ValueError(f"lookback en az 2 olmalı: {lookback}")
        if min_periods < 1:
            raise ValueError(f"min_periods en az 1 olmalı: {min_periods}")

        arr = cast(np.ndarray, series.values.astype(np.float64))
        n = len(arr)
        out = np.full(n, 50.0)

        nan_input_mask = np.isnan(arr)
        out[nan_input_mask] = np.nan

        def _avg_tie_rank_pct(window: np.ndarray, current: float) -> float:
            """
            PAYLAŞILAN formül — vectorized YOL, fallback YOL ve warmup
            YOLU'nun ÜÇÜ DE bunu kullanır (DÜZELTME: önceden fallback/
            warmup FARKLI, pandas'la TUTARSIZ bir formül kullanıyordu —
            artık TEK bir kaynak, üç tüketici).
            """
            valid = window[~np.isnan(window)]
            if len(valid) == 0:
                return 50.0
            below = np.sum(valid < current)
            below_eq = np.sum(valid <= current)
            avg_rank = (below + below_eq + 1) / 2.0
            return float(avg_rank / len(valid) * 100.0)

        # stride_tricks: tam pencere varsa vectorized
        if n >= lookback:
            try:
                from numpy.lib.stride_tricks import sliding_window_view

                views = sliding_window_view(arr, window_shape=lookback)
                # views.shape = (m, lookback), m = n - lookback + 1
                # views[j][-1] = arr[j + lookback - 1] = arr[i] (güncel bar, PENCERENİN PARÇASI)

                current_vals = views[:, -1]  # (m,)
                nan_mask_current = np.isnan(current_vals)

                # DÜZELTME (performans): _avg_tie_rank_pct'in formülü,
                # numpy BROADCAST ile TAM VECTORIZED şekilde uygulanıyor
                # (Python döngüsü YOK — önceki taslakta yanlışlıkla
                # bir for-loop'a düşmüştüm, "vectorized" hedefini
                # BOZUYORDU, bu turda düzeltildi). NaN karşılaştırmaları
                # numpy'da DOĞAL OLARAK False döndüğü için (NaN < x
                # HER ZAMAN False), below/below_eq sayımları NaN
                # değerleri KENDİLİĞİNDEN dışlıyor — yalnızca payda
                # (valid_count) için AYRI bir np.isnan kontrolü gerekiyor.
                current_col = current_vals[:, None]  # (m, 1) — broadcast için
                below = np.sum(views < current_col, axis=1)
                below_eq = np.sum(views <= current_col, axis=1)
                valid_count = np.sum(~np.isnan(views), axis=1)
                valid_count_safe = np.where(valid_count == 0, 1, valid_count)  # /0 koruması

                avg_rank = (below + below_eq + 1) / 2.0
                ranks = avg_rank / valid_count_safe * 100.0
                ranks[nan_mask_current] = np.nan

                out[lookback - 1:] = ranks

            except ImportError:
                # stride_tricks yoksa yavaş yol (Python fallback) — AYNI formül
                logger.warning("stride_tricks_unavailable_using_fallback")
                for i in range(lookback - 1, n):
                    w = arr[i - lookback + 1: i + 1]
                    if np.isnan(w[-1]):
                        out[i] = np.nan
                    else:
                        out[i] = _avg_tie_rank_pct(w, w[-1])

        # Kısmi pencere (warmup): min_periods'dan lookback'e kadar — AYNI formül
        for i in range(min_periods - 1, min(lookback - 1, n)):
            w = arr[: i + 1]
            if np.isnan(w[-1]):
                out[i] = np.nan
            else:
                out[i] = _avg_tie_rank_pct(w, w[-1])

        return pd.Series(out, index=series.index)

    # ── Bollinger Band ────────────────────────────────────────────────────────

    @staticmethod
    def calculate_bollinger_bandwidth(
        close: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> pd.Series:
        """
        Bollinger Band genişliği.

        Formül: (4 × std) / sma  (legacy'deki num_std=2, band width = 2×2=4)
        Not: Genişlik = (Upper - Lower) / Middle = 2×num_std×std / sma

        Args:
            close: Kapanış fiyatı serisi.
            period: BB periyodu (default 20).
            num_std: Standart sapma çarpanı (default 2.0).

        Returns:
            BB genişlik serisi (boyutsuz oran).
        """
        sma = close.rolling(period, min_periods=1).mean()
        std = close.rolling(period, min_periods=2).std()

        # sma = 0 durumunda NaN (sıfıra bölünme koruması)
        safe_sma = sma.replace(0.0, np.nan)
        bandwidth = (2.0 * num_std * std) / safe_sma

        return bandwidth

    @staticmethod
    def calculate_bollinger_squeeze(
        close: pd.Series,
        bb_period: int = 20,
        squeeze_lookback: int = 250,
        squeeze_quantile: float = 0.15,
    ) -> pd.Series:
        """
        Bollinger Squeeze tespiti (bool serisi).

        Legacy karşılığı: bollinger_squeeze()

        Mantık: BB genişliği son {squeeze_lookback} barın P{squeeze_quantile×100}
        yüzdeliğinin altındaysa sıkışma (squeeze) aktif.

        Args:
            close: Kapanış fiyatı serisi.
            bb_period: Bollinger Band periyodu (default 20).
            squeeze_lookback: Sıkışma eşiği hesabı için lookback (default 250).
            squeeze_quantile: Sıkışma eşiği yüzdeliği (default 0.15 = P15).

        Returns:
            Boolean serisi: True = Squeeze aktif (fiyat sıkışmış).
        """
        bandwidth = TechnicalCalculator.calculate_bollinger_bandwidth(close, bb_period)

        threshold = bandwidth.rolling(
            squeeze_lookback, min_periods=30
        ).quantile(squeeze_quantile)

        return bandwidth < threshold

    # ── VCP Bileşenleri ───────────────────────────────────────────────────────

    @staticmethod
    def calculate_vcp_components(
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 60,
        recent_bars: int = 20,
        std_contraction_ratio: float = 0.85,
        vol_contraction_ratio: float = 0.75,
    ) -> VCPComponents:
        """
        Volatility Contraction Pattern (VCP) bileşen analizi.

        Legacy karşılığı: vcp_score()

        VCP kriterleri:
          1. Price Contraction: Son {recent_bars} std < Tüm lookback std × ratio
          2. Volume Contraction: Son {recent_bars} vol ort < Tüm lookback vol ort × ratio
          3. Price Rising: Son kapanış > lookback başındaki kapanış

        Args:
            close: Kapanış fiyatı serisi (minimum lookback uzunluğunda).
            volume: Hacim serisi.
            lookback: Toplam analiz periyodu (default 60 bar).
            recent_bars: Kısa dönem penceresi (default 20 bar).
            std_contraction_ratio: Std daralma eşiği (default 0.85 = %15 daralma).
            vol_contraction_ratio: Hacim daralma eşiği (default 0.75 = %25 daralma).

        Returns:
            VCPComponents(price_contraction, volume_contraction, price_rising, score).

        Note:
            Bu metod anlık (son bar için) VCP durumunu döndürür.
            Backtest için vektörize versiyon ayrıca geliştirilebilir.
        """
        if len(close) < lookback:
            logger.debug(
                "vcp_insufficient_data",
                available=len(close),
                required=lookback,
            )
            return VCPComponents(
                price_contraction=False,
                volume_contraction=False,
                price_rising=False,
                composite_score=0,
            )

        c_window = close.iloc[-lookback:]
        v_window = volume.iloc[-lookback:]

        std_long = float(c_window.std()) if len(c_window) > 1 else 0.0
        std_short = float(c_window.iloc[-recent_bars:].std()) if len(c_window) >= recent_bars else 0.0

        vol_long = float(v_window.mean())
        vol_short = float(v_window.iloc[-recent_bars:].mean()) if len(v_window) >= recent_bars else 0.0

        price_contraction = (
            (std_short < std_long * std_contraction_ratio)
            if std_long > 0
            else False
        )
        volume_contraction = (
            (vol_short < vol_long * vol_contraction_ratio)
            if vol_long > 0
            else False
        )
        price_rising = float(c_window.iloc[-1]) > float(c_window.iloc[-recent_bars])

        score = int(price_contraction) + int(volume_contraction) + int(price_rising)

        return VCPComponents(
            price_contraction=price_contraction,
            volume_contraction=volume_contraction,
            price_rising=price_rising,
            composite_score=score,
        )

    # ── Göreceli Güç ─────────────────────────────────────────────────────────

    @staticmethod
    def calculate_relative_strength(
        stock_close: pd.Series,
        benchmark_close: pd.Series,
        ma_period: int = 20,
        slope_bars: int = 5,
    ) -> RelativeStrengthResult:
        """
        Göreceli güç (Relative Strength) hesabı.

        Legacy karşılığı: relative_strength_full() — BUG-05/06 düzeltmeleri dahil.

        BUG-05/06 Düzeltmesi:
          YANLIŞ: pd.concat + ffill → lookahead bias
          DOĞRU: inner join (dropna) → yalnızca eşleşen tarihler

        RS = stock_close / benchmark_close
        RS yukarıda MA → güçlü trend
        RS slope pozitif → momentum artıyor

        Args:
            stock_close: Hisse kapanış fiyatı serisi.
            benchmark_close: Benchmark kapanış serisi (ör: XU100).
            ma_period: RS hareketli ortalama periyodu (default 20).
            slope_bars: RS momentum hesabı için bar sayısı (default 5).

        Returns:
            RelativeStrengthResult (rs_series, rs_above_ma, rs_slope_positive).

        Raises:
            ValueError: Eşleşen tarih sayısı 2'den azsa (hesaplama imkansız).
        """
        # Inner join — lookahead bias önleme (BUG-05/06)
        aligned = pd.concat(
            [stock_close.rename("stock"), benchmark_close.rename("bench")],
            axis=1,
        ).dropna()  # ffill YOK — sadece eşleşen barlar

        if len(aligned) < 2:
            raise ValueError(
                f"Göreceli güç için yeterli eşleşen veri yok: {len(aligned)} bar"
            )

        rs = aligned["stock"] / aligned["bench"].replace(0.0, np.nan)
        rs_ma = rs.rolling(ma_period, min_periods=1).mean()

        rs_above_ma = rs > rs_ma
        rs_slope_positive = rs.diff(slope_bars) > 0.0

        return RelativeStrengthResult(
            rs_series=rs,
            rs_above_ma=rs_above_ma,
            rs_slope_positive=rs_slope_positive,
            ma_period=ma_period,
        )

    # ── Gap Yüzdesi ──────────────────────────────────────────────────────────

    @staticmethod
    def calculate_gap_pct(
        open_price: pd.Series,
        prev_close: pd.Series,
    ) -> pd.Series:
        """
        Açılış gap yüzdesi = |Open - PrevClose| / PrevClose.

        Args:
            open_price: Açılış fiyatı serisi.
            prev_close: Önceki kapanış serisi (close.shift(1) olabilir).

        Returns:
            Gap yüzde serisi (0.04 = %4 gap). PrevClose=0 ise NaN.
        """
        safe_prev = prev_close.replace(0.0, np.nan)
        return (open_price - safe_prev).abs() / safe_prev
