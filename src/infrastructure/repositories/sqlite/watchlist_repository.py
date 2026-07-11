"""
Watchlist repository — SQLite/SQLAlchemy Core implementasyonu.

API sözleşmesi — design doc'ta bir WatchlistService/Repository arayüzü
BELGELENMEDİĞİ için (yalnızca DDL var), bu proje boyunca kurulan
TransactionRepository/PortfolioRepository desenleriyle TUTARLI olacak
şekilde TASARLANDI (icat değil, mevcut desenin uygulanması).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.enums.asset_type import AssetType
from src.domain.exceptions.domain_exceptions import DuplicateError, NotFoundError
from src.domain.models.watchlist import Watchlist, WatchlistItem
from src.infrastructure.database.orm_models import watchlist_items_table, watchlists_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _row_to_watchlist(row: Any) -> Watchlist:
    return Watchlist(
        name=row.name, portfolio_id=row.portfolio_id, description=row.description,
        is_active=bool(row.is_active),
        created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
        watchlist_id=row.id,
    )


def _row_to_item(row: Any) -> WatchlistItem:
    return WatchlistItem(
        watchlist_id=row.watchlist_id, symbol=row.symbol,
        symbol_type=AssetType(row.symbol_type),
        alert_price_low=Decimal(row.alert_price_low) if row.alert_price_low is not None else None,
        alert_price_high=Decimal(row.alert_price_high) if row.alert_price_high is not None else None,
        alert_pct_change=Decimal(row.alert_pct_change) if row.alert_pct_change is not None else None,
        notes=row.notes, added_at=datetime.fromisoformat(row.added_at), item_id=row.id,
    )


class SQLiteWatchlistRepository:
    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def create_watchlist(self, watchlist: Watchlist) -> Watchlist:
        """
        Raises:
            DuplicateError: Aynı isimde bir watchlist zaten varsa.

        NOT: DDL'de watchlists.name için UNIQUE constraint YOK (yalnızca
        watchlist_items.(watchlist_id, symbol) UNIQUE) — bu yüzden
        aynı isimde birden fazla watchlist DB seviyesinde TEKNİK OLARAK
        mümkün. Burada uygulama seviyesinde (SELECT-then-INSERT) bir
        isim benzersizliği kontrolü ekleniyor — kullanıcı deneyimi için
        makul bir kısıtlama, ama DB CHECK constraint kadar GÜÇLÜ bir
        garanti DEĞİL (TOCTOU riski teorik olarak var, kişisel/tek-
        kullanıcılı ölçekte pratik etkisi yok).
        """
        with self._session_factory() as session:
            existing = session.execute(
                select(watchlists_table.c.id).where(watchlists_table.c.name == watchlist.name)
            ).first()
            if existing is not None:
                raise DuplicateError("Watchlist", watchlist.name)

            new_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            session.execute(
                watchlists_table.insert().values(
                    id=new_id, name=watchlist.name, portfolio_id=watchlist.portfolio_id,
                    description=watchlist.description, is_active=int(watchlist.is_active),
                    created_at=now, updated_at=now,
                )
            )
            session.commit()

        logger.info("watchlist_created", watchlist_id=new_id, name=watchlist.name)
        return Watchlist(
            name=watchlist.name, portfolio_id=watchlist.portfolio_id,
            description=watchlist.description, is_active=watchlist.is_active,
            watchlist_id=new_id,
        )

    def list_watchlists(self, include_inactive: bool = False) -> list[Watchlist]:
        stmt = select(watchlists_table)
        if not include_inactive:
            stmt = stmt.where(watchlists_table.c.is_active == 1)
        stmt = stmt.order_by(watchlists_table.c.name.asc())
        with self._session_factory() as session:
            return [_row_to_watchlist(row) for row in session.execute(stmt)]

    def get_watchlist(self, watchlist_id: str) -> Watchlist | None:
        stmt = select(watchlists_table).where(watchlists_table.c.id == watchlist_id)
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_watchlist(row) if row is not None else None

    def add_item(self, item: WatchlistItem) -> WatchlistItem:
        """
        Raises:
            DuplicateError: Bu watchlist'te bu sembol ZATEN varsa (DDL
                UNIQUE(watchlist_id, symbol) — DB seviyesinde zorlanıyor).
        """
        new_id = str(uuid.uuid4())
        try:
            with self._session_factory() as session:
                session.execute(
                    watchlist_items_table.insert().values(
                        id=new_id, watchlist_id=item.watchlist_id, symbol=item.symbol,
                        symbol_type=item.symbol_type.value,
                        alert_price_low=str(item.alert_price_low) if item.alert_price_low is not None else None,
                        alert_price_high=str(item.alert_price_high) if item.alert_price_high is not None else None,
                        alert_pct_change=str(item.alert_pct_change) if item.alert_pct_change is not None else None,
                        notes=item.notes, added_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
                session.commit()
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc) or "uq_watchlist_symbol" in str(exc):
                raise DuplicateError("WatchlistItem", f"{item.watchlist_id}:{item.symbol}") from exc
            raise

        logger.info("watchlist_item_added", watchlist_id=item.watchlist_id, symbol=item.symbol)
        return WatchlistItem(
            watchlist_id=item.watchlist_id, symbol=item.symbol, symbol_type=item.symbol_type,
            alert_price_low=item.alert_price_low, alert_price_high=item.alert_price_high,
            alert_pct_change=item.alert_pct_change, notes=item.notes, item_id=new_id,
        )

    def remove_item(self, item_id: str) -> None:
        """
        Raises:
            NotFoundError: item_id yoksa.

        NOT: Transaction'ın AKSİNE, watchlist item'ları GERÇEK muhasebe
        kaydı DEĞİL — bu yüzden burada reversal deseni (is_active=0 +
        marker) YERİNE doğrudan DELETE kullanılıyor. Bir izleme
        listesinden sembol çıkarmanın "audit trail" gerektiren bir
        muhasebe anlamı yok.
        """
        with self._session_factory() as session:
            existing = session.execute(
                select(watchlist_items_table.c.id).where(watchlist_items_table.c.id == item_id)
            ).first()
            if existing is None:
                raise NotFoundError("WatchlistItem", item_id)
            session.execute(
                watchlist_items_table.delete().where(watchlist_items_table.c.id == item_id)
            )
            session.commit()
        logger.info("watchlist_item_removed", item_id=item_id)

    def list_items(self, watchlist_id: str) -> list[WatchlistItem]:
        stmt = (
            select(watchlist_items_table)
            .where(watchlist_items_table.c.watchlist_id == watchlist_id)
            .order_by(watchlist_items_table.c.added_at.asc())
        )
        with self._session_factory() as session:
            return [_row_to_item(row) for row in session.execute(stmt)]
