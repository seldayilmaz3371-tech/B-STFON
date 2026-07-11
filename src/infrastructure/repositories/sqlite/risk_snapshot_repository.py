"""
Risk snapshot repository.

API sözleşmesi — design doc Bölüm 1.3'ün "RiskService.get_latest_snapshot"
metodunu destekleyecek şekilde tasarlandı.

MİMARİ KARAR — is_stale ile "audit trail korunan cache" deseni:
  add() çağrıldığında, o portföy için ÖNCEDEN VAR OLAN tüm is_stale=0
  snapshot'lar ÖNCE is_stale=1 yapılır, SONRA yeni snapshot eklenir.
  Bu, TEK ATOMIC OPERASYON değil (aynı session içinde iki adım) — ama
  AYNI session/transaction'da olduğu için (tek `with` bloğu, tek
  commit) DB-seviyesinde atomik. Bu, Transaction/CashLedgerEntry
  arasındaki (FARKLI repository'ler, FARKLI session'lar) atomicity
  sınırlamasından FARKLI ve DAHA GÜÇLÜ bir garanti — çünkü burada
  tek bir repository, tek bir session kullanıyor.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.enums.var_method import VaRMethod
from src.domain.models.risk_snapshot import RiskSnapshot
from src.infrastructure.database.orm_models import risk_snapshots_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _row_to_snapshot(row: Any) -> RiskSnapshot:
    return RiskSnapshot(
        portfolio_id=row.portfolio_id,
        as_of_date=date.fromisoformat(row.as_of_date),
        lookback_days=row.lookback_days,
        risk_free_rate=row.risk_free_rate,
        portfolio_volatility=row.portfolio_volatility,
        sharpe_ratio=row.sharpe_ratio,
        sortino_ratio=row.sortino_ratio,
        calmar_ratio=row.calmar_ratio,
        max_drawdown=row.max_drawdown,
        max_drawdown_start=date.fromisoformat(row.max_drawdown_start) if row.max_drawdown_start else None,
        max_drawdown_end=date.fromisoformat(row.max_drawdown_end) if row.max_drawdown_end else None,
        current_drawdown=row.current_drawdown,
        var_95=row.var_95, var_99=row.var_99, cvar_95=row.cvar_95, cvar_99=row.cvar_99,
        var_method=VaRMethod(row.var_method),
        beta=row.beta, alpha=row.alpha, r_squared=row.r_squared,
        information_ratio=row.information_ratio, tracking_error=row.tracking_error,
        herfindahl_index=row.herfindahl_index, top5_concentration=row.top5_concentration,
        benchmark_code=row.benchmark_code, is_stale=bool(row.is_stale),
        snapshot_id=row.id, computed_at=datetime.fromisoformat(row.computed_at),
    )


class SQLiteRiskSnapshotRepository:
    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def add(self, snapshot: RiskSnapshot) -> RiskSnapshot:
        """
        Önceki (is_stale=0) snapshot'ları AYNI session'da stale işaretler,
        sonra yenisini ekler — bkz. modül docstring'i.
        """
        new_id = str(uuid.uuid4())
        computed_at = datetime.now(timezone.utc)
        with self._session_factory() as session:
            session.execute(
                risk_snapshots_table.update()
                .where(
                    risk_snapshots_table.c.portfolio_id == snapshot.portfolio_id,
                    risk_snapshots_table.c.is_stale == 0,
                )
                .values(is_stale=1)
            )
            session.execute(
                risk_snapshots_table.insert().values(
                    id=new_id,
                    portfolio_id=snapshot.portfolio_id,
                    computed_at=computed_at.isoformat(),
                    as_of_date=snapshot.as_of_date.isoformat(),
                    lookback_days=snapshot.lookback_days,
                    portfolio_volatility=snapshot.portfolio_volatility,
                    sharpe_ratio=snapshot.sharpe_ratio,
                    sortino_ratio=snapshot.sortino_ratio,
                    calmar_ratio=snapshot.calmar_ratio,
                    max_drawdown=snapshot.max_drawdown,
                    max_drawdown_start=snapshot.max_drawdown_start.isoformat() if snapshot.max_drawdown_start else None,
                    max_drawdown_end=snapshot.max_drawdown_end.isoformat() if snapshot.max_drawdown_end else None,
                    current_drawdown=snapshot.current_drawdown,
                    var_95=snapshot.var_95, var_99=snapshot.var_99,
                    cvar_95=snapshot.cvar_95, cvar_99=snapshot.cvar_99,
                    var_method=snapshot.var_method.value,
                    beta=snapshot.beta, alpha=snapshot.alpha, r_squared=snapshot.r_squared,
                    information_ratio=snapshot.information_ratio,
                    tracking_error=snapshot.tracking_error,
                    herfindahl_index=snapshot.herfindahl_index,
                    top5_concentration=snapshot.top5_concentration,
                    risk_free_rate=snapshot.risk_free_rate,
                    benchmark_code=snapshot.benchmark_code,
                    is_stale=0,
                )
            )
            session.commit()

        logger.info(
            "risk_snapshot_added", snapshot_id=new_id, portfolio_id=snapshot.portfolio_id,
            as_of_date=snapshot.as_of_date.isoformat(),
        )
        return RiskSnapshot(
            **{**snapshot.__dict__, "snapshot_id": new_id, "computed_at": computed_at},
        )

    def get_latest_snapshot(self, portfolio_id: str) -> RiskSnapshot | None:
        """is_stale=0 olan (yalnızca bir tane olmalı, add()'in garantisi) snapshot'ı döner."""
        stmt = select(risk_snapshots_table).where(
            risk_snapshots_table.c.portfolio_id == portfolio_id,
            risk_snapshots_table.c.is_stale == 0,
        ).order_by(risk_snapshots_table.c.computed_at.desc()).limit(1)
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_snapshot(row) if row is not None else None

    def get_history(self, portfolio_id: str, limit: int = 30) -> list[RiskSnapshot]:
        """Stale dahil TÜM geçmiş snapshot'lar — trend analizi için (en yeniden en eskiye)."""
        stmt = select(risk_snapshots_table).where(
            risk_snapshots_table.c.portfolio_id == portfolio_id,
        ).order_by(risk_snapshots_table.c.computed_at.desc()).limit(limit)
        with self._session_factory() as session:
            return [_row_to_snapshot(row) for row in session.execute(stmt)]
