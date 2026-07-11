"""
İşlem (transaction) repository — SQLite/SQLAlchemy Core implementasyonu.

get_portfolio_symbols/get_by_symbol sözleşmesi — VARSAYIM DEĞİL,
portfolio_service.py'nin ÇALIŞTIRILARAK tersine mühendisliğiyle çıkarıldı.

add_transaction/get_by_id/list_by_portfolio/reverse_transaction — bu
turda TransactionService için eklendi ve TransactionService üzerinden
GERÇEKTEN test edildi (bkz. test_transaction_service.py).

REVERSAL TASARIMI (mimari karar, gerekçeli):
  Transaction immutable (frozen dataclass) — UPDATE metodu YOK ve
  OLMAYACAK. Yanlış girilen bir işlem şöyle düzeltilir:
    1. Orijinal satır `is_active=0` yapılır (hesaplamalardan ÇIKARILIR
       — CostBasisCalculator yalnızca is_active=1 görür).
    2. YENİ bir "reversal marker" satırı eklenir: `is_reversal=1`,
       `reversal_of=<orijinal_id>`, `reversal_reason=<sebep>`,
       KENDİSİ DE is_active=0 (hesaplamaları ETKİLEMEZ — yalnızca
       audit trail için var).

  NEDEN offsetting-entry (ters işlem eklemek) DEĞİL, is_active=0 +
  marker: Bir SPLIT veya BONUS_SHARE işleminin "tersini" almak
  (ör. "SPLIT'i geri al") kavramsal olarak type-specific karmaşık
  bir ters mantık gerektirir (ratio'nun tersini almak, vs.) — bu,
  her transaction_type için AYRI bir "reverse etkisi" formülü
  yazmayı gerektirir ve YANLIŞ yapılması kolay bir alan. is_active=0
  + marker deseni ise TÜM transaction_type'lar için AYNI, basit,
  hatasız bir mekanizma: "bu satır artık geçerli değil, işte neden."
  Audit trail DB seviyesinde tam korunuyor (satır silinmiyor).

  FK ETKİSİ: cash_ledger_entries.transaction_id bir transaction'a
  işaret ediyorsa ve o transaction reverse edilirse, ilgili cash
  ledger entry'si OTOMATİK OLARAK güncellenmez/silinmez — bu bilinçli
  bir kapsam dışı bırakma, CashLedgerService (henüz yazılmadı) bu
  senaryoyu ele almalı. Şimdilik bu, kabul edilmiş bir tutarlılık
  riski.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import AlreadyReversedError, NotFoundError
from src.domain.models.transaction import Transaction
from src.infrastructure.database.orm_models import transactions_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _row_to_transaction(row: Any) -> Transaction:
    """
    ORM satırı → domain Transaction dönüşümü.

    Decimal alanlarda TEXT → Decimal dönüşümü HER ZAMAN str() üzerinden
    (satırda zaten TEXT olarak saklandığı için doğrudan Decimal(value)
    güvenli).
    """
    return Transaction(
        symbol=row.symbol,
        transaction_type=TransactionType(row.transaction_type),
        timestamp=datetime.fromisoformat(row.trade_date),
        quantity=Decimal(row.quantity),
        price=Decimal(row.price),
        split_ratio=Decimal(row.split_ratio) if row.split_ratio is not None else None,
        net_amount=Decimal(row.net_amount) if row.net_amount is not None else None,
        transaction_id=row.id,
        symbol_type=row.symbol_type,
        portfolio_id=row.portfolio_id,
    )


class SQLiteTransactionRepository:
    """
    Repository Pattern — domain katmanı bu sınıfın SQLAlchemy
    kullandığını bilmez, yalnızca dönen `Transaction` nesnelerini görür.
    """

    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def get_portfolio_symbols(self, portfolio_id: str) -> list[str]:
        """is_active=1 filtresi: reversal edilmiş işlemler hesaba katılmaz."""
        stmt = (
            select(transactions_table.c.symbol)
            .where(
                transactions_table.c.portfolio_id == portfolio_id,
                transactions_table.c.is_active == 1,
            )
            .distinct()
        )
        with self._session_factory() as session:
            result = session.execute(stmt)
            return [row.symbol for row in result]

    def get_by_symbol(self, portfolio_id: str, symbol: str) -> list[Transaction]:
        """Bir portföy+sembol için TÜM (aktif) işlem geçmişini döndürür."""
        stmt = (
            select(transactions_table)
            .where(
                transactions_table.c.portfolio_id == portfolio_id,
                transactions_table.c.symbol == symbol,
                transactions_table.c.is_active == 1,
            )
            .order_by(transactions_table.c.trade_date.asc())
        )
        with self._session_factory() as session:
            result = session.execute(stmt)
            return [_row_to_transaction(row) for row in result]

    def list_by_portfolio(
        self, portfolio_id: str, include_inactive: bool = False
    ) -> list[Transaction]:
        """
        Bir portföydeki TÜM sembollerin TÜM işlemlerini döndürür (İşlem
        Girişi UI'ının "geçmiş işlemler" listesi için).

        include_inactive=True: reversal edilmiş VE reversal marker
        satırlarını da içerir — audit/denetim görünümü için.
        """
        stmt = select(transactions_table).where(
            transactions_table.c.portfolio_id == portfolio_id
        )
        if not include_inactive:
            stmt = stmt.where(transactions_table.c.is_active == 1)
        stmt = stmt.order_by(transactions_table.c.trade_date.desc())

        with self._session_factory() as session:
            return [_row_to_transaction(row) for row in session.execute(stmt)]

    def get_by_id(self, transaction_id: str) -> Transaction | None:
        stmt = select(transactions_table).where(transactions_table.c.id == transaction_id)
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_transaction(row) if row is not None else None

    def add_transaction(
        self, portfolio_id: str, symbol_type: str, transaction: Transaction
    ) -> str:
        """
        Yeni bir işlem kaydeder. Immutable — UPDATE metodu YOK, bilinçli
        olarak (reversal ile düzeltme kuralı).

        Returns:
            Oluşturulan transaction id (UUID4, str).
        """
        new_id = str(uuid.uuid4())
        stmt = transactions_table.insert().values(
            id=new_id,
            portfolio_id=portfolio_id,
            symbol=transaction.symbol,
            symbol_type=symbol_type,
            transaction_type=transaction.transaction_type.value,
            quantity=str(transaction.quantity),
            price=str(transaction.price),
            net_amount=str(transaction.net_amount) if transaction.net_amount is not None else None,
            split_ratio=str(transaction.split_ratio) if transaction.split_ratio is not None else None,
            trade_date=transaction.timestamp.isoformat(),
            is_active=1,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._session_factory() as session:
            session.execute(stmt)
            session.commit()
        logger.info(
            "transaction_added",
            transaction_id=new_id,
            portfolio_id=portfolio_id,
            symbol=transaction.symbol,
            transaction_type=transaction.transaction_type.value,
        )
        return new_id

    def reverse_transaction(self, transaction_id: str, reason: str) -> str:
        """
        Bkz. modül docstring'indeki "REVERSAL TASARIMI" bölümü.

        Raises:
            NotFoundError: transaction_id yoksa.
            AlreadyReversedError: İşlem zaten reverse edilmişse (is_active=0).
        """
        with self._session_factory() as session:
            row = session.execute(
                select(transactions_table).where(transactions_table.c.id == transaction_id)
            ).first()
            if row is None:
                raise NotFoundError("Transaction", transaction_id)
            if row.is_active == 0:
                raise AlreadyReversedError(transaction_id)

            session.execute(
                transactions_table.update()
                .where(transactions_table.c.id == transaction_id)
                .values(is_active=0)
            )

            marker_id = str(uuid.uuid4())
            session.execute(
                transactions_table.insert().values(
                    id=marker_id,
                    portfolio_id=row.portfolio_id,
                    symbol=row.symbol,
                    symbol_type=row.symbol_type,
                    transaction_type=row.transaction_type,
                    quantity=row.quantity,
                    price=row.price,
                    net_amount=row.net_amount,
                    split_ratio=row.split_ratio,
                    trade_date=row.trade_date,
                    is_active=0,  # yalnızca audit — hesaplamaları ETKİLEMEZ
                    created_at=datetime.now(timezone.utc).isoformat(),
                    is_reversal=1,
                    reversal_of=transaction_id,
                    reversal_reason=reason,
                )
            )
            session.commit()

        logger.info(
            "transaction_reversed",
            original_transaction_id=transaction_id,
            reversal_marker_id=marker_id,
            reason=reason,
        )
        return marker_id
