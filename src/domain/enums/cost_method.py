"""
Maliyet hesaplama yöntemi enum'u.

ADR-001 referansı: WAVG varsayılan, FIFO config'den açılabilir.
Bu enum, Portfolio entity'sindeki `cost_method` alanının değer kümesi
ve CostBasisCalculatorFactory'nin (Faz C'de container.py entegrasyonunda
eklenecek) strateji seçim anahtarıdır.

LIFO bilinçli olarak dışarıda bırakıldı: ADR-001 yalnızca WAVG/FIFO
kararlaştırıyor, LIFO hiçbir golden dataset'te tanımlı değil. İleride
gerekirse bu enum'a eklenmesi mevcut kodu bozmaz (Open/Closed).
"""

from __future__ import annotations

from enum import Enum


class CostMethod(str, Enum):
    WAVG = "WAVG"
    FIFO = "FIFO"
