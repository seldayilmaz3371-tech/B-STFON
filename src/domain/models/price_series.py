"""
PriceSeries domain modeli.

Kaynak: BIST_TEFAS_Master_Design_Document.md Bölüm 2.7 — alan alan
birebir uygulandı, icat edilmedi.

KRİTİK TASARIM NOTU (design doc'tan aynen alındı, bu ayrım hayati):
  close_price: Ham fiyat (o gün işlem gören gerçek fiyat) — maliyet
    bazı hesaplaması için kullanılır.
  adjusted_close: Corporate action'lar için geriye dönük düzeltilmiş
    fiyat — GETİRİ hesaplaması için kullanılır.
  Bu ikisinin karıştırılması yanlış PnL/getiri hesaplamalarına yol
  açar. CostBasisCalculator'a HİÇBİR ZAMAN adjusted_close verilmemeli
  — yalnızca close_price veya gerçek işlem fiyatı (Transaction.price).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from src.domain.enums.asset_type import AssetType


@dataclass(frozen=True)
class PriceSeries:
    symbol: str
    symbol_type: AssetType
    date: date
    close_price: Decimal  # DDL: NOT NULL — TEFAS için NAV = close
    source: str  # "yfinance" | "tefas" | "isyatirim" | "manual" | "mock"
    open_price: Decimal | None = None
    high_price: Decimal | None = None
    low_price: Decimal | None = None
    volume: Decimal | None = None
    adjusted_close: Decimal | None = None
    is_holiday: bool = False
    price_id: str | None = None

    def __post_init__(self) -> None:
        if self.close_price < Decimal("0"):
            raise ValueError(f"close_price negatif olamaz: {self.close_price}")
        if not self.symbol_type.is_valid_for_price_series():
            raise ValueError(
                f"{self.symbol_type} bir PriceSeries için geçerli değil "
                "(bkz. AssetType.is_valid_for_price_series)."
            )
