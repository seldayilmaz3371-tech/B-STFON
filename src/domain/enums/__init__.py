"""
domain/enums paketi — dışa aktarılan enum'lar.

Not: asset_type, currency_code, fund_type, var_method enum'ları
bilinçli olarak bu turda YAZILMADI. Gerekçe: YAGNI — hiçbir mevcut
modül (cost_basis_calculator, Transaction) bunları şu an tüketmiyor.
Faz B'de (TEFAS fon modelleri, çoklu para birimi) veya Faz D'de
(VaR yöntemi seçimi) ilk gerçek tüketici ortaya çıktığında eklenecekler.
Şimdiden yazmak, kullanılmayan / varsayımsal alanlarla enum'ları
kirletme riski taşır.
"""

from __future__ import annotations

from src.domain.enums.cost_method import CostMethod
from src.domain.enums.transaction_type import TransactionType

__all__ = ["CostMethod", "TransactionType"]
