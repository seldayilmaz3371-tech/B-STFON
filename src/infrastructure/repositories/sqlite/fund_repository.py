"""Fund repository — SQLite/SQLAlchemy Core implementasyonu."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.exceptions.domain_exceptions import DuplicateError, NotFoundError
from src.domain.models.fund import Fund
from src.infrastructure.database.orm_models import funds_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _dec(value: Any) -> Decimal | None:
    return Decimal(value) if value is not None else None


def _row_to_fund(row: Any) -> Fund:
    return Fund(
        fund_code=row.fund_code, fund_name=row.fund_name, fund_type=row.fund_type,
        currency=row.currency, umbrella_type=row.umbrella_type, founder=row.founder,
        stock_pct=_dec(row.stock_pct), bond_pct=_dec(row.bond_pct), repo_pct=_dec(row.repo_pct),
        foreign_stock_pct=_dec(row.foreign_stock_pct), gold_pct=_dec(row.gold_pct),
        other_pct=_dec(row.other_pct),
        allocation_date=date.fromisoformat(row.allocation_date) if row.allocation_date else None,
        last_nav=_dec(row.last_nav),
        last_nav_date=date.fromisoformat(row.last_nav_date) if row.last_nav_date else None,
        ytd_return=_dec(row.ytd_return), management_fee=_dec(row.management_fee),
        is_active=bool(row.is_active),
        last_updated=datetime.fromisoformat(row.last_updated), created_at=datetime.fromisoformat(row.created_at),
    )


class SQLiteFundRepository:
    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def upsert(self, fund: Fund) -> Fund:
        """
        Fund'lar TEFAS'tan periyodik senkronize edilecek varlıklar
        (design doc'un "last_nav (cache)" alanı bunu ima ediyor) —
        bu yüzden Transaction'ın AKSİNE upsert semantiği DOĞRU
        (fund_code zaten TEFAS'ın kendi benzersiz kodu, çakışma
        BEKLENEN bir durum, hata DEĞİL).
        """
        now = datetime.now(timezone.utc).isoformat()
        values = dict(
            fund_name=fund.fund_name, fund_type=fund.fund_type,
            umbrella_type=fund.umbrella_type, founder=fund.founder, currency=fund.currency,
            stock_pct=str(fund.stock_pct) if fund.stock_pct is not None else None,
            bond_pct=str(fund.bond_pct) if fund.bond_pct is not None else None,
            repo_pct=str(fund.repo_pct) if fund.repo_pct is not None else None,
            foreign_stock_pct=str(fund.foreign_stock_pct) if fund.foreign_stock_pct is not None else None,
            gold_pct=str(fund.gold_pct) if fund.gold_pct is not None else None,
            other_pct=str(fund.other_pct) if fund.other_pct is not None else None,
            allocation_date=fund.allocation_date.isoformat() if fund.allocation_date else None,
            last_nav=str(fund.last_nav) if fund.last_nav is not None else None,
            last_nav_date=fund.last_nav_date.isoformat() if fund.last_nav_date else None,
            ytd_return=str(fund.ytd_return) if fund.ytd_return is not None else None,
            management_fee=str(fund.management_fee) if fund.management_fee is not None else None,
            is_active=int(fund.is_active), last_updated=now,
        )
        with self._session_factory() as session:
            existing = session.execute(
                select(funds_table.c.fund_code).where(funds_table.c.fund_code == fund.fund_code)
            ).first()
            if existing is not None:
                session.execute(
                    funds_table.update().where(funds_table.c.fund_code == fund.fund_code).values(**values)
                )
            else:
                session.execute(funds_table.insert().values(fund_code=fund.fund_code, created_at=now, **values))
            session.commit()
        logger.info("fund_upserted", fund_code=fund.fund_code)
        return Fund(fund_code=fund.fund_code, **{k: v for k, v in fund.__dict__.items() if k != "fund_code"})

    def get_by_code(self, fund_code: str) -> Fund | None:
        stmt = select(funds_table).where(funds_table.c.fund_code == fund_code)
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_fund(row) if row is not None else None

    def list_by_type(self, fund_type: str | None = None, include_inactive: bool = False) -> list[Fund]:
        stmt = select(funds_table)
        if fund_type is not None:
            stmt = stmt.where(funds_table.c.fund_type == fund_type)
        if not include_inactive:
            stmt = stmt.where(funds_table.c.is_active == 1)
        stmt = stmt.order_by(funds_table.c.fund_name.asc())
        with self._session_factory() as session:
            return [_row_to_fund(row) for row in session.execute(stmt)]
