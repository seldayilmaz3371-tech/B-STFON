"""
CorporateAction repository — yalnızca veri saklama.

KAPSAM: create/get/list/mark_confirmed/mark_applied — "portföye
uygula" mantığı YOK (bkz. corporate_action.py modül docstring'i).
mark_applied() yalnızca BAYRAĞI günceller, GERÇEKTEN hiçbir portföy
hesaplamasını TETİKLEMEZ — çağıran taraf (ileride yazılacak bir
CorporateActionApplicationService) bu bayrağı, işi GERÇEKTEN
yaptıktan SONRA set etmekle sorumlu olacak.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.exceptions.domain_exceptions import NotFoundError
from src.domain.models.corporate_action import CorporateAction
from src.infrastructure.database.orm_models import corporate_actions_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _row_to_action(row: Any) -> CorporateAction:
    return CorporateAction(
        symbol=row.symbol, action_type=row.action_type,
        ex_date=date.fromisoformat(row.ex_date),
        action_data=json.loads(row.action_data) if row.action_data else {},
        announcement_date=date.fromisoformat(row.announcement_date) if row.announcement_date else None,
        record_date=date.fromisoformat(row.record_date) if row.record_date else None,
        payment_date=date.fromisoformat(row.payment_date) if row.payment_date else None,
        is_confirmed=bool(row.is_confirmed), is_applied=bool(row.is_applied),
        source=row.source, notes=row.notes, raw_data=row.raw_data,
        action_id=row.id, created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
    )


class SQLiteCorporateActionRepository:
    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def create(self, action: CorporateAction) -> CorporateAction:
        new_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._session_factory() as session:
            session.execute(
                corporate_actions_table.insert().values(
                    id=new_id, symbol=action.symbol, action_type=action.action_type,
                    announcement_date=action.announcement_date.isoformat() if action.announcement_date else None,
                    ex_date=action.ex_date.isoformat(),
                    record_date=action.record_date.isoformat() if action.record_date else None,
                    payment_date=action.payment_date.isoformat() if action.payment_date else None,
                    action_data=json.dumps(action.action_data),
                    is_confirmed=int(action.is_confirmed), is_applied=int(action.is_applied),
                    source=action.source, notes=action.notes, raw_data=action.raw_data,
                    created_at=now, updated_at=now,
                )
            )
            session.commit()
        logger.info("corporate_action_created", action_id=new_id, symbol=action.symbol, action_type=action.action_type)
        return CorporateAction(
            **{**action.__dict__, "action_id": new_id, "created_at": datetime.fromisoformat(now), "updated_at": datetime.fromisoformat(now)}
        )

    def get_by_id(self, action_id: str) -> CorporateAction | None:
        stmt = select(corporate_actions_table).where(corporate_actions_table.c.id == action_id)
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_action(row) if row is not None else None

    def list_by_symbol(self, symbol: str) -> list[CorporateAction]:
        stmt = (
            select(corporate_actions_table)
            .where(corporate_actions_table.c.symbol == symbol)
            .order_by(corporate_actions_table.c.ex_date.desc())
        )
        with self._session_factory() as session:
            return [_row_to_action(row) for row in session.execute(stmt)]

    def list_pending(self) -> list[CorporateAction]:
        """is_applied=0 olan TÜM kayıtlar — design doc'un idx_ca_pending index'iyle TUTARLI sorgu deseni."""
        stmt = (
            select(corporate_actions_table)
            .where(corporate_actions_table.c.is_applied == 0)
            .order_by(corporate_actions_table.c.ex_date.asc())
        )
        with self._session_factory() as session:
            return [_row_to_action(row) for row in session.execute(stmt)]

    def mark_confirmed(self, action_id: str) -> None:
        self._set_flag(action_id, "is_confirmed", 1)

    def mark_applied(self, action_id: str) -> None:
        """
        YALNIZCA bayrağı işaretler — GERÇEKTEN hiçbir portföy
        hesaplamasını tetiklemiyor (bkz. modül docstring'i).
        """
        self._set_flag(action_id, "is_applied", 1)

    def _set_flag(self, action_id: str, column: str, value: int) -> None:
        with self._session_factory() as session:
            existing = session.execute(
                select(corporate_actions_table.c.id).where(corporate_actions_table.c.id == action_id)
            ).first()
            if existing is None:
                raise NotFoundError("CorporateAction", action_id)
            session.execute(
                corporate_actions_table.update()
                .where(corporate_actions_table.c.id == action_id)
                .values(**{column: value, "updated_at": datetime.now(timezone.utc).isoformat()})
            )
            session.commit()
