"""
Strategy arayüz sözleşmesi — design doc Bölüm 3.10 "strategies/
base_strategy.py" ile uyumlu.

MİMARİ KARAR (bu turda, Backtest Engine mimarisi için verildi):
  vectorbt DEĞİL, custom event-driven engine — GEREKÇE: mevcut
  CostBasisCalculator (WAVG/FIFO) DOĞRUDAN kullanılabiliyor, backtest
  sonuçları CANLI muhasebe mantığıyla TUTARLI oluyor. vectorbt kendi
  vectorized portföy simülasyon modelini kullanır — FARKLI bir PnL
  hesaplama yöntemi anlamına gelir. Ayrıca "event-driven sistemler"
  proje başından beri AÇIKÇA istenen bir mimari yaklaşım.

  Lisans notu (ayrıca araştırıldı): Açık kaynak vectorbt, Apache 2.0 +
  Commons Clause lisanslı (yazılımı SATAMAZSIN kısıtlaması) — kişisel
  kullanım için sorun değildi, ama bu KARARIN birincil gerekçesi
  DEĞİL (birincil gerekçe: muhasebe tutarlılığı).

KAPSAM KARARI: Sinyal üretimi VECTORIZED (design doc'un "generate_signals
DataFrame ile -1/0/1 sinyalleri" tanımıyla TUTARLI) ama sinyallerin
GERÇEK İŞLEME dönüştürülmesi (BacktestEngine.run) EVENT-DRIVEN bir
döngüde, CostBasisCalculator kullanarak yapılıyor — iki yaklaşımın
GÜÇLÜ yanlarını birleştiren bir hibrit.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class Strategy(Protocol):
    """
    generate_signals(): Fiyat verisinden -1 (SAT), 0 (BEKLE), 1 (AL)
    sinyalleri üretir — design doc'un "signals: DataFrame ile -1/0/1
    sinyalleri" tanımıyla BİREBİR.

    KRİTİK — LOOK-AHEAD BIAS KORUNMASI: Bir stratejinin gün T'deki
    sinyali, YALNIZCA T ve ÖNCESİNDEKİ veriyi kullanmalıdır (T+1'in
    kapanış fiyatını GÖRMEMELİDİR). Bu KURAL, Strategy implementasyonunun
    SORUMLULUĞUDUR — BacktestEngine bunu ZORLAYAMAZ (bir stratejinin
    pandas.shift() kullanmayı UNUTMASI, engine seviyesinde YAKALANAMAZ).
    Bu, açıkça İŞARETLENMESİ gereken bir RİSK — bkz. test dosyasındaki
    "look-ahead bias" testi (yalnızca ÖRNEK stratejinin doğru
    davrandığını kanıtlıyor, TÜM gelecek stratejileri garanti ETMİYOR).
    """

    def generate_signals(self, price_data: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """
        Args:
            price_data: date-indexed DataFrame, en az 'close' kolonu.
            params: Strateji parametreleri (örn. {"fast_window": 10, "slow_window": 50}).

        Returns:
            price_data ile AYNI date index'e sahip, değerleri {-1, 0, 1}
            olan bir pd.Series.
        """
        ...
