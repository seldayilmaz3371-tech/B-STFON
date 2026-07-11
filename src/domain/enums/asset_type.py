"""
Varlık tipi enum'u.

DDL TUTARSIZLIĞI TESPİTİ (açıkça işaretleniyor, sessizce çözülmedi):
  BIST_TEFAS_Master_Design_Document.md'deki DDL'de İKİ FARKLI CHECK
  constraint kümesi var:

    transactions.symbol_type CHECK:
      BIST_STOCK, TEFAS_FUND, BIST_ETF, BOND, CASH, OTHER

    price_series.symbol_type CHECK:
      BIST_STOCK, TEFAS_FUND, BIST_ETF, BENCHMARK

  Yani CASH/BOND/OTHER yalnızca transactions'ta anlamlı (örn. DEPOSIT/
  WITHDRAWAL/FEE işlemleri symbol_type='CASH' ile kayıtlı), price_series'te
  bunların bir fiyat serisi OLAMAZ (nakidin "kapanış fiyatı" olmaz).
  BENCHMARK ise tam tersi — yalnızca price_series'te anlamlı (BIST100
  gibi endeksler işlem görmez ama fiyat serisi tutulur, transactions'ta
  hiç yer almaz).

  KARAR: TEK bir AssetType enum'u, ikisinin BİRLEŞİMİ (union) olarak
  tanımlanıyor. Python enum seviyesinde her tabloya özel kısıtlama
  UYGULANMIYOR — bu sorumluluk DB'nin CHECK constraint'lerinde kalıyor
  (bkz. orm_models.py, her tablo kendi CheckConstraint'ini taşıyor).
  Gerekçe: iki ayrı enum tanımlamak (TransactionSymbolType,
  PriceSeriesSymbolType) tip sistemini gereksiz karmaşıklaştırır ve
  ADR-001 tarzı "tek kaynak/tek sözleşme" prensibiyle çelişir; DB
  seviyesinde zaten doğru kısıtlama var, Python enum'u yalnızca ortak
  bir kelime dağarcığı sağlıyor. RİSK: Bir geliştirici yanlışlıkla
  AssetType.CASH'i bir PriceSeries'e atarsa, bunu yakalayan tek yer
  DB INSERT anındaki CheckConstraint olur (Python tip sisteminde değil)
  — bu kabul edilmiş bir risk, mypy bunu yakalayamaz.
"""

from __future__ import annotations

from enum import Enum


class AssetType(str, Enum):
    BIST_STOCK = "BIST_STOCK"
    TEFAS_FUND = "TEFAS_FUND"
    BIST_ETF = "BIST_ETF"
    BOND = "BOND"
    CASH = "CASH"
    OTHER = "OTHER"
    BENCHMARK = "BENCHMARK"

    def is_valid_for_price_series(self) -> bool:
        """price_series.symbol_type CHECK constraint'iyle senkron."""
        return self in (
            AssetType.BIST_STOCK,
            AssetType.TEFAS_FUND,
            AssetType.BIST_ETF,
            AssetType.BENCHMARK,
        )

    def is_valid_for_transaction(self) -> bool:
        """transactions.symbol_type CHECK constraint'iyle senkron."""
        return self in (
            AssetType.BIST_STOCK,
            AssetType.TEFAS_FUND,
            AssetType.BIST_ETF,
            AssetType.BOND,
            AssetType.CASH,
            AssetType.OTHER,
        )
