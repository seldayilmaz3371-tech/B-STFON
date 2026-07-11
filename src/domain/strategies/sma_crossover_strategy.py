"""
SMA Crossover — design doc Bölüm 3.10 "strategies/sma_crossover.py"
ile uyumlu örnek strateji.

BU BİR YATIRIM TAVSİYESİ DEĞİL — yalnızca Strategy arayüzünün ve
BacktestEngine'in DOĞRU çalıştığını KANITLAMAK için referans bir
implementasyon. Gerçek dünyada SMA crossover stratejilerinin
karlılığı TARTIŞMALIDIR, bu proje bir strateji ÖNERİSİ SUNMUYOR.

LOOK-AHEAD BIAS KORUNMASI: `.shift(1)` KULLANILIYOR — gün T'deki
sinyal, gün T'nin KAPANIŞINDAN SONRA hesaplanan crossover'ı bir gün
ERTELEYEREK uyguluyor (gerçekte T'nin kapanışı bilinmeden T'de işlem
YAPILAMAZ, T+1'in açılışında işlem yapılabilir — bu basitleştirme
"T+1 kapanışında işlem" varsayıyor, gerçekte "T+1 açılışı" daha
gerçekçi olurdu ama bu MVP için kabul edilebilir bir yaklaşım).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class SMACrossoverStrategy:
    """
    fast_window'un slow_window'u YUKARI kestiği gün: AL (1)
    fast_window'un slow_window'u AŞAĞI kestiği gün: SAT (-1)
    Diğer günler: BEKLE (0)
    """

    def generate_signals(self, price_data: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        fast_window = params.get("fast_window", 10)
        slow_window = params.get("slow_window", 50)

        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) slow_window'dan ({slow_window}) "
                "küçük olmalı."
            )

        close = price_data["close"]
        fast_sma = close.rolling(window=fast_window).mean()
        slow_sma = close.rolling(window=slow_window).mean()

        # LOOK-AHEAD BIAS KORUNMASI: .shift(1) — bkz. modül docstring'i.
        #
        # DÜZELTME (bu turda, GERÇEK bir DeprecationWarning ile bulundu):
        # .shift(1), bool dtype'lı bir Series'i (NaN taşıyabilmek için)
        # OBJECT dtype'a yükseltiyor — .fillna(False) sonrası bile dtype
        # 'object' olarak KALIYOR (otomatik bool'a dönmüyor). Bu durumda
        # `~` (bitwise NOT) OBJECT-dtype Series üzerinde her elemana
        # Python'ın ÇIPLAK ~ operatörünü uyguluyor — bu, bool için
        # BİTWİSE tersine çevirme yapar (~True == -2), MANTIKSAL DEĞİL,
        # ve Python 3.12+'ta DeprecationWarning veriyor (gelecekte
        # muhtemelen TypeError'a dönüşecek). `.astype(bool)` ile dtype
        # GERİ ZORLANIYOR, `~` artık DOĞRU (numpy vectorized, mantıksal)
        # negasyon yapıyor.
        fast_above = (fast_sma > slow_sma).shift(1).fillna(False).astype(bool)
        fast_above_prev = fast_above.shift(1).fillna(False).astype(bool)

        signals = pd.Series(0, index=price_data.index)
        signals[fast_above & ~fast_above_prev] = 1   # yukarı kesişim: AL
        signals[~fast_above & fast_above_prev] = -1  # aşağı kesişim: SAT

        return signals
