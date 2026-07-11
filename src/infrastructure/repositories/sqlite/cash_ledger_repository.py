"""
Nakit ledger repository'si.

API sözleşmesi — VARSAYIM DEĞİL, BIST_TEFAS_Master_Design_Document.md
Bölüm 1.2 "CashLedgerRepository" tam interface contract'ından alındı:
    add_entry(entry) -> CashLedgerEntry
    get_balance(portfolio_id, as_of=None) -> Decimal
    get_entries(portfolio_id, start=None, end=None) -> list[CashLedgerEntry]
    verify_balance(portfolio_id) -> BalanceVerification

SORUMLULUK SINIRI (tekrar vurgulanıyor — CashLedgerEntry docstring'inde
de var): add_entry() balance_after'ı HESAPLAMAZ, yalnızca persist eder.
balance_after'ı hesaplamak (mevcut bakiye + / - amount) bir servis
katmanı sorumluluğu (CashLedgerService, henüz yazılmadı). Bu
repository'nin add_entry()'si "sana ne verilirse onu yazarım" ilkesiyle
çalışır — bu YANLIŞ bir tasarım değil, Repository Pattern'in doğru
uygulanışı: repository iş kuralı bilmez, yalnızca kalıcılık sağlar.

verify_balance() TAM OLARAK bu yüzden var: iş kuralının repository
dışında doğru uygulandığını SONRADAN doğrulamak için bir güvenlik ağı.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.enums.ledger_entry_type import LedgerEntryType
from src.domain.models.cash_ledger_entry import CashLedgerEntry
from src.infrastructure.database.orm_models import cash_ledger_entries_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

ZERO = Decimal("0")


@dataclass(frozen=True)
class BalanceVerification:
    """
    verify_balance() sonucu.

    SUM(CREDIT) - SUM(DEBIT) == son entry'nin balance_after değeri mi?
    Tutmuyorsa bu, add_entry()'yi çağıran servis katmanında bir
    hesaplama hatası olduğunun kanıtıdır (repository'nin kendi hatası
    DEĞİL — repository yalnızca ne verilirse onu yazdı).
    """

    is_consistent: bool
    expected: Decimal
    actual: Decimal
    discrepancy: Decimal


def _row_to_entry(row: Any) -> CashLedgerEntry:
    return CashLedgerEntry(
        portfolio_id=row.portfolio_id,
        entry_type=LedgerEntryType(row.entry_type),
        amount=Decimal(row.amount),
        entry_date=date.fromisoformat(row.entry_date),
        description=row.description,
        balance_after=Decimal(row.balance_after),
        currency=row.currency,
        transaction_id=row.transaction_id,
        entry_id=row.id,
    )


class SQLiteCashLedgerRepository:
    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def add_entry(self, entry: CashLedgerEntry) -> CashLedgerEntry:
        new_id = str(uuid.uuid4())
        with self._session_factory() as session:
            session.execute(
                cash_ledger_entries_table.insert().values(
                    id=new_id,
                    portfolio_id=entry.portfolio_id,
                    transaction_id=entry.transaction_id,
                    entry_type=entry.entry_type.value,
                    amount=str(entry.amount),
                    currency=entry.currency,
                    entry_date=entry.entry_date.isoformat(),
                    description=entry.description,
                    balance_after=str(entry.balance_after),
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            session.commit()
        logger.info(
            "cash_ledger_entry_added",
            entry_id=new_id,
            portfolio_id=entry.portfolio_id,
            entry_type=entry.entry_type.value,
            amount=str(entry.amount),
        )
        return CashLedgerEntry(
            portfolio_id=entry.portfolio_id, entry_type=entry.entry_type,
            amount=entry.amount, entry_date=entry.entry_date,
            description=entry.description, balance_after=entry.balance_after,
            currency=entry.currency, transaction_id=entry.transaction_id,
            entry_id=new_id,
        )

    def get_balance(self, portfolio_id: str, as_of: date | None = None) -> Decimal:
        """
        as_of=None: en güncel balance_after.
        as_of=date: o tarihe kadar (dahil) olan son entry'nin balance_after'ı.

        Hiç entry yoksa: Decimal('0') döner (boş ledger = sıfır bakiye).

        KRİTİK DÜZELTME (bu turda, GERÇEK bir tutarsızlık testiyle
        bulundu — test_reversal_keeps_cash_balance_consistent):
          İlk tasarımda as_of=None durumu da entry_date'i birincil
          sıralama anahtarı olarak kullanıyordu. Bu, GERİYE DÖNÜK
          TARİHLİ kayıtlarla (örn. bugün yapılan ama geçmişi düzelten
          bir reversal, ya da kullanıcının geçmiş bir işlemi bugün
          girmesi) YANLIŞ sonuç veriyordu — "ekleniş sırası" ile
          "iş tarihi sırası" birbirinden ayrıştığında, aradaki (iş
          tarihi daha ileri ama daha ÖNCE eklenmiş) bir kayıt yanlışlıkla
          "daha güncel" sayılıyordu. Somut etkisi: bir reversal sonrası
          hesaplanan bir sonraki işlemin bakiyesi, reversal'ın etkisini
          GÖRMÜYORDU — sistematik bir tutarsızlık.

          DOĞRU MUHASEBE PRENSİBİ: "güncel bakiye" HER ZAMAN deftere
          EN SON YAZILAN (insertion order) satırın taşıdığı değerdir,
          o satırın hangi iş tarihini etiketlediği ÖNEMLİ DEĞİL — gerçek
          defter sistemleri böyle çalışır (her yeni kayıt, önceki TÜM
          kayıtların üzerine ekleniş sırasına göre inşa edilir).

          as_of=<belirli tarih> sorgusu FARKLI bir semantik taşıyor
          ("o iş tarihinde defter ne gösteriyordu" — bitemporal/historical
          sorgu) — bu durumda entry_date'in birincil anahtar olması
          DOĞRU ve DEĞİŞTİRİLMEDİ.
        """
        stmt = select(cash_ledger_entries_table.c.balance_after).where(
            cash_ledger_entries_table.c.portfolio_id == portfolio_id
        )
        if as_of is not None:
            stmt = stmt.where(
                cash_ledger_entries_table.c.entry_date <= as_of.isoformat()
            ).order_by(
                cash_ledger_entries_table.c.entry_date.desc(),
                cash_ledger_entries_table.c.created_at.desc(),
            )
        else:
            # "Güncel bakiye" = insertion-order'a göre EN SON kayıt.
            # id (UUID4) sıralamaya uygun değil — yalnızca created_at
            # (ISO 8601 string, lexicographic sıralama = kronolojik
            # sıralama) güvenilir tek anahtar.
            stmt = stmt.order_by(cash_ledger_entries_table.c.created_at.desc())
        stmt = stmt.limit(1)

        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return Decimal(row.balance_after) if row is not None else ZERO

    def get_entries(
        self, portfolio_id: str, start: date | None = None, end: date | None = None
    ) -> list[CashLedgerEntry]:
        stmt = select(cash_ledger_entries_table).where(
            cash_ledger_entries_table.c.portfolio_id == portfolio_id
        )
        if start is not None:
            stmt = stmt.where(cash_ledger_entries_table.c.entry_date >= start.isoformat())
        if end is not None:
            stmt = stmt.where(cash_ledger_entries_table.c.entry_date <= end.isoformat())
        stmt = stmt.order_by(
            cash_ledger_entries_table.c.entry_date.asc(),
            cash_ledger_entries_table.c.created_at.asc(),
        )
        with self._session_factory() as session:
            return [_row_to_entry(row) for row in session.execute(stmt)]

    def verify_balance(self, portfolio_id: str) -> BalanceVerification:
        """
        DÜZELTME (bu turda): `actual` daha önce get_entries()'in
        entry_date-sıralı listesinin SON elemanından alınıyordu — bu,
        get_balance()'daki AYNI hatayı taşıyordu (bkz. get_balance()
        docstring'indeki tam gerekçe). Artık düzeltilmiş get_balance()
        çağrılıyor — TEK bir doğru "güncel bakiye" kaynağı (DRY,
        iki farklı yerde iki farklı yanlış sonuç riski ORTADAN KALKTI).
        """
        entries = self.get_entries(portfolio_id)
        if not entries:
            return BalanceVerification(
                is_consistent=True, expected=ZERO, actual=ZERO, discrepancy=ZERO
            )

        expected = ZERO
        for e in entries:
            expected += e.amount if e.entry_type is LedgerEntryType.CREDIT else -e.amount

        actual = self.get_balance(portfolio_id)
        discrepancy = expected - actual
        return BalanceVerification(
            is_consistent=discrepancy == ZERO,
            expected=expected,
            actual=actual,
            discrepancy=discrepancy,
        )
