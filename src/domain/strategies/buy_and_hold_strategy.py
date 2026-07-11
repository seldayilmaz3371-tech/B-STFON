"""
Buy and Hold — herhangi bir aktif stratejinin karşılaştırılacağı
TEMEL (baseline) referans stratejisi.

GEREKÇE: Backtesting'in standart pratiği — bir stratejinin "iyi"
olduğunu iddia etmeden ÖNCE, "basitçe alıp hiç satmasaydım ne olurdu"
ile karşılaştırılması ZORUNLU bir sağlık kontrolü. Aktif bir strateji,
komisyon+slippage maliyetini karşılayacak kadar buy-and-hold'u
YENEMİYORSA, o stratejinin GERÇEK bir değeri yok demektir. Bu proje
bir strateji ÖNERMİYOR (bkz. sma_crossover_strategy.py'nin AYNI
uyarısı) — yalnızca KARŞILAŞTIRMA için gerekli bir referans noktası
sağlıyor.

DAVRANIŞ: İlk gün AL (1) sinyali, SONRASINDA HİÇBİR ZAMAN SAT
sinyali üretmez (tüm dizi boyunca 0). BacktestEngine bunu ALL-IN
BUY olarak işler (bkz. backtest_engine.py "Position sizing"
gerekçesi) ve pozisyonu asla kapatmaz — bu, GERÇEK bir "buy and
hold" davranışının doğru simülasyonu.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class BuyAndHoldStrategy:
    """
    İlk gün AL, sonrasında HİÇ SAT sinyali üretmez.

    params: KULLANILMIYOR (bu stratejinin hiçbir parametresi yok) —
    Strategy Protocol sözleşmesini korumak için imzada TUTULUYOR
    (backtest_service.py, TÜM stratejileri AYNI arayüzle çağırıyor).
    """

    def generate_signals(self, price_data: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        signals = pd.Series(0, index=price_data.index)
        if len(signals) > 0:
            signals.iloc[0] = 1
        return signals
