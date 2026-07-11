"""
Portföy repository — SQLite/SQLAlchemy Core implementasyonu.

list_all() sözleşmesi — VARSAYIM DEĞİL, portfolio_service.py'nin
ÇALIŞTIRILARAK tersine mühendisliğiyle çıkarıldı (bkz. PortfolioRecord).

add()/get_by_id()/get_by_name() — design doc Bölüm 1.2 "PortfolioRepository"
tam interface contract'ından alındı (bu turda eklendi, çünkü sistemde
şimdiye kadar HİÇBİR kullanıcı arayüzünden portföy oluşturma yolu
yoktu — tüm test verisi doğrudan SQL ile ekleniyordu, bu gerçek bir
kullanılabilirlik boşluğuydu).

tags/metadata serileştirme: JSON-as-TEXT (projenin "Decimal as TEXT"
konvansiyonuyla tutarlı bir "karmaşık tip as TEXT" yaklaşımı). NULL
değer boş liste/dict olarak yorumlanıyor (round-trip'te veri kaybı yok,
None ile boş koleksiyon arasındaki fark bilinçli olarak ÖNEMSİZ
sayılıyor — bir portföyün "hiç tag'i yok" ile "boş tag listesi" arasında
iş mantığı açısından fark yok).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.enums.cost_method import CostMethod
from src.domain.enums.currency_code import CurrencyCode
from src.domain.exceptions.domain_exceptions import DuplicateError
from src.domain.models.portfolio import Portfolio
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PortfolioRecord:
    """
    list_all()'ın döndürdüğü minimal DTO — portfolio_service.py yalnızca
    .id ve .name okuyor, bu yüzden bilinçli olarak dar tutuldu (tam
    Portfolio nesnesi değil; o get_by_id()/get_by_name()'den gelir).
    """

    id: str
    name: str
    is_active: bool


def _row_to_portfolio(row: Any) -> Portfolio:
    return Portfolio(
        id=row.id,
        name=row.name,
        description=row.description,
        currency=CurrencyCode(row.currency),
        cost_method=CostMethod(row.cost_method),
        inception_date=date.fromisoformat(row.inception_date),
        benchmark_code=row.benchmark_code,
        is_active=bool(row.is_active),
        created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
        tags=tuple(json.loads(row.tags)) if row.tags else (),
        metadata=json.loads(row.metadata) if row.metadata else {},
    )


class SQLitePortfolioRepository:
    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def list_all(self, include_inactive: bool = False) -> list[PortfolioRecord]:
        stmt = select(
            portfolios_table.c.id,
            portfolios_table.c.name,
            portfolios_table.c.is_active,
        )
        if not include_inactive:
            stmt = stmt.where(portfolios_table.c.is_active == 1)
        stmt = stmt.order_by(portfolios_table.c.name.asc())

        with self._session_factory() as session:
            result = session.execute(stmt)
            return [
                PortfolioRecord(id=row.id, name=row.name, is_active=bool(row.is_active))
                for row in result
            ]

    def get_by_id(self, portfolio_id: str) -> Portfolio | None:
        stmt = select(portfolios_table).where(portfolios_table.c.id == portfolio_id)
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_portfolio(row) if row is not None else None

    def get_by_name(self, name: str) -> Portfolio | None:
        stmt = select(portfolios_table).where(portfolios_table.c.name == name)
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_portfolio(row) if row is not None else None

    def add(self, portfolio: Portfolio) -> Portfolio:
        """
        Raises:
            DuplicateError: Aynı isimde AKTİF ya da PASİF bir portföy
                zaten var (DDL UNIQUE(name) — is_active'e bakmaksızın
                tüm isimler için geçerli, DB seviyesinde zorlanıyor).

        Atomicity notu: DB'nin UNIQUE constraint'ine güveniyoruz
        (SELECT-then-INSERT yerine) — bu, iki eşzamanlı isteğin aynı
        isimle portföy oluşturmaya çalıştığı TOCTOU (time-of-check to
        time-of-use) yarışını DB seviyesinde önlüyor; SELECT-then-INSERT
        deseni bu korumayı SAĞLAMAZ (iki request aynı anda SELECT'i
        "yok" görüp ikisi de INSERT edebilir).
        """
        try:
            with self._session_factory() as session:
                session.execute(
                    portfolios_table.insert().values(
                        id=portfolio.id,
                        name=portfolio.name,
                        description=portfolio.description,
                        currency=portfolio.currency.value,
                        cost_method=portfolio.cost_method.value,
                        inception_date=portfolio.inception_date.isoformat(),
                        benchmark_code=portfolio.benchmark_code,
                        is_active=int(portfolio.is_active),
                        created_at=portfolio.created_at.isoformat(),
                        updated_at=portfolio.updated_at.isoformat(),
                        tags=json.dumps(list(portfolio.tags)) if portfolio.tags else None,
                        metadata=json.dumps(portfolio.metadata) if portfolio.metadata else None,
                    )
                )
                session.commit()
        except Exception as exc:
            # SQLAlchemy/SQLite UNIQUE constraint ihlali IntegrityError
            # olarak gelir — mesaj içeriğine bakmak kırılgan olsa da
            # (dialect'e bağlı), DB seviyesindeki tek gerçek kaynak bu.
            # Alternatif: SELECT-then-INSERT — ama bu TOCTOU riski taşır
            # (yukarıdaki docstring notuna bkz.), bu yüzden BİLİNÇLİ
            # olarak exception-tabanlı yaklaşım tercih edildi.
            if "UNIQUE constraint failed" in str(exc) or "uq_portfolio_name" in str(exc):
                raise DuplicateError("Portfolio", portfolio.name) from exc
            raise

        logger.info(
            "portfolio_created",
            portfolio_id=portfolio.id,
            name=portfolio.name,
            currency=portfolio.currency.value,
            cost_method=portfolio.cost_method.value,
        )
        return portfolio
