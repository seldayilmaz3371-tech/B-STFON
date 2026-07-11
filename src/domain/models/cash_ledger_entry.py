"""
CashLedgerEntry domain modeli.

Kaynak: BIST_TEFAS_Master_Design_Document.md Bölüm 2.10 — birebir.

SORUMLULUK SINIRI (açıkça belirtiliyor):
  balance_after alanı DENORMALİZE bir alandır ve BU SINIF TARAFINDAN
  HESAPLANMAZ — çağıran (henüz yazılmamış bir CashLedgerService,
  Faz C+ kapsamı) mevcut bakiyeyi okuyup (CashLedgerRepository.
  get_balance) yeni bakiyeyi hesaplayıp bu alanı DOLU olarak
  CashLedgerEntry'yi oluşturmalı. Bu sınıf yalnızca veri taşıyıcısıdır
  — iş kuralı (DEBIT mi CREDIT mi, hangi işlem tipi nakti nasıl
  etkiler) burada YOK. Bu ayrım bilinçli: Single Responsibility —
  "nakit hareketi kaydı" ile "nakit hareketi iş kuralı" farklı
  katmanlara ait.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from src.domain.enums.ledger_entry_type import LedgerEntryType


@dataclass(frozen=True)
class CashLedgerEntry:
    portfolio_id: str
    entry_type: LedgerEntryType
    amount: Decimal  # DDL CHECK: her zaman pozitif
    entry_date: date
    description: str
    balance_after: Decimal
    currency: str = "TRY"
    transaction_id: str | None = None
    entry_id: str | None = None

    def __post_init__(self) -> None:
        if self.amount <= Decimal("0"):
            raise ValueError(
                f"amount her zaman pozitif olmalı (DDL CHECK), alınan: {self.amount}"
            )
