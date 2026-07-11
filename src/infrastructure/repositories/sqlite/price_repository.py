"""
Fiyat serisi repository'si.

API sözleşmesi — VARSAYIM DEĞİL, BIST_TEFAS_Master_Design_Document.md
Bölüm 1.2 "PriceRepository" tam interface contract'ından alındı
(upsert, upsert_batch, get_ohlcv, get_latest_price, get_missing_dates,
get_price_on_date — hepsi dokümante edilmiş imzalarla).

BİLİNEN SINIRLAMA — get_missing_dates() (açıkça işaretleniyor):
  Design doc "trading_days_only=True: tatil günleri missing sayılmaz"
  diyor. Bunu doğru uygulamak GERÇEK bir BIST tatil takvimi gerektirir
  (resmi tatiller + yarım günler). Bu proje kapsamında böyle bir
  takvim VERİSİ YOK (bkz. tasarım belgesi Ek G.4 — "KAP makine
  okunabilir API sunmuyor" benzeri bir gerçeklik, resmi tatil takvimi
  için de otomatik bir kaynak entegre edilmedi). Bu implementasyon
  yalnızca HAFTA SONLARINI (Cumartesi/Pazar) tatil sayıyor — resmi
  BIST tatilleri (örn. 23 Nisan, 1 Mayıs) İSE "missing" olarak
  raporlanacak (yanlış pozitif). Gerçek kullanımda bu, gereksiz
  "eksik veri" uyarılarına yol açar ama veri bütünlüğünü YANLIŞ
  YÖNDE etkilemez (olması gerekenden FAZLA gap raporlar, AZ değil —
  bu daha güvenli bir hata yönüdür). Gerçek tatil takvimi kaynağı
  (örn. `pandas_market_calendars` kütüphanesi, XIST borsası desteği
  var) entegre edilene kadar bu sınırlama geçerli.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.domain.enums.asset_type import AssetType
from src.domain.models.price_series import PriceSeries
from src.infrastructure.database.orm_models import price_series_table
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BatchResult:
    """
    upsert_batch() sonucu.

    failed: (sembol, tarih, hata_mesajı) üçlüleri — exception fırlatmak
    yerine loglanıp burada toplanır (design doc: "hatalar loglanır,
    exception değil" — bir sembolün başarısız yazılması, batch'teki
    diğer 999 kaydın yazılmasını engellememeli).
    """

    inserted: int = 0
    updated: int = 0
    failed: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)

    @property
    def success_count(self) -> int:
        return self.inserted + self.updated


def _row_to_price_series(row: Any) -> PriceSeries:
    return PriceSeries(
        symbol=row.symbol,
        symbol_type=AssetType(row.symbol_type),
        date=date.fromisoformat(row.date),
        close_price=Decimal(row.close_price),
        source=row.source,
        open_price=Decimal(row.open_price) if row.open_price is not None else None,
        high_price=Decimal(row.high_price) if row.high_price is not None else None,
        low_price=Decimal(row.low_price) if row.low_price is not None else None,
        volume=Decimal(row.volume) if row.volume is not None else None,
        adjusted_close=Decimal(row.adjusted_close) if row.adjusted_close is not None else None,
        is_holiday=bool(row.is_holiday),
        price_id=row.id,
    )


class SQLitePriceRepository:
    def __init__(self, session_factory: "sessionmaker[Session]") -> None:
        self._session_factory = session_factory

    def upsert(self, price: PriceSeries) -> PriceSeries:
        """
        (symbol, date) unique constraint üzerinden upsert.

        SQLite'a özgü `INSERT ... ON CONFLICT DO UPDATE` yerine
        BİLİNÇLİ OLARAK select-then-insert/update kullanılıyor
        (SQLAlchemy Core'un dialect-agnostic yolu) — gerekçe: Faz 3'te
        PostgreSQL'e geçişte bu kod DEĞİŞMEDEN çalışmalı; SQLite'a özgü
        ON CONFLICT syntax'ı bu taşınabilirliği bozar. Bedel: 2 sorgu
        (1 select + 1 insert/update) yerine 1 native upsert — kişisel
        portföy ölçeğinde (günlük birkaç yüz fiyat güncellemesi)
        performans farkı ölçülebilir değil.
        """
        with self._session_factory() as session:
            existing = session.execute(
                select(price_series_table.c.id).where(
                    price_series_table.c.symbol == price.symbol,
                    price_series_table.c.date == price.date.isoformat(),
                )
            ).first()

            values = dict(
                symbol=price.symbol,
                symbol_type=price.symbol_type.value,
                date=price.date.isoformat(),
                open_price=str(price.open_price) if price.open_price is not None else None,
                high_price=str(price.high_price) if price.high_price is not None else None,
                low_price=str(price.low_price) if price.low_price is not None else None,
                close_price=str(price.close_price),
                adjusted_close=str(price.adjusted_close) if price.adjusted_close is not None else None,
                volume=str(price.volume) if price.volume is not None else None,
                source=price.source,
                is_holiday=int(price.is_holiday),
            )

            if existing is not None:
                price_id = existing.id
                session.execute(
                    price_series_table.update()
                    .where(price_series_table.c.id == price_id)
                    .values(**values)
                )
            else:
                price_id = str(uuid.uuid4())
                session.execute(
                    price_series_table.insert().values(
                        id=price_id,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        **values,
                    )
                )
            session.commit()

        return PriceSeries(
            symbol=price.symbol, symbol_type=price.symbol_type, date=price.date,
            close_price=price.close_price, source=price.source,
            open_price=price.open_price, high_price=price.high_price,
            low_price=price.low_price, volume=price.volume,
            adjusted_close=price.adjusted_close, is_holiday=price.is_holiday,
            price_id=price_id,
        )

    def upsert_batch(self, prices: list[PriceSeries]) -> BatchResult:
        """
        DÜZELTME (bu turda, RiskService'e ThreadPoolExecutor eklenirken
        GERÇEK bir yük testiyle bulundu — KRİTİK performans hatası):

        İLK TASARIM: Her fiyat kaydı için AYRI bir session açıp AYRI
        commit ediyordu (aslında `exists` kontrolü için BİR session,
        sonra `self.upsert()` içinde İKİNCİ bir session — item başına
        2 session + 1 commit). 150 kayıtlık bir batch, 5 sembol için
        PARALEL çalıştırıldığında toplam ~1.0 SANİYE sürdüğü ÖLÇÜLEREK
        tespit edildi (saf fetch işlemi ~0.2s'de bitmesine rağmen).
        Kök neden: SQLite WAL modu tek-yazarlıdır — her commit bir
        fsync-benzeri senkronizasyon içerir; 750 kayıt (150×5 sembol)
        için 750 ayrı commit, thread'ler arası yazma kilidi çekişmesiyle
        (busy_timeout ile BEKLEYEREK, ama yine de SIRAYLA) birikip
        gerçek bir darboğaz oluşturdu.

        DÜZELTME: TEK session, TEK commit (batch sonunda). Her kayıt
        için per-item hata izolasyonu (bir kaydın CHECK constraint
        ihlali DİĞERLERİNİ etkilememeli) artık `session.begin_nested()`
        (SAVEPOINT) ile sağlanıyor — bir savepoint'in rollback'i yalnızca
        O kaydı geri alır, session'daki ÖNCEKİ başarılı kayıtları
        ETKİLEMEZ. Bu, önceki "her item ayrı session" izolasyonuyla
        AYNI garantiyi, çok daha düşük maliyetle sağlıyor.

        Design doc performans hedefi: 1000 kayıt < 500ms. Bu düzeltme
        SONRASI hâlâ TAM olarak benchmark edilmedi (yalnızca doğruluk +
        "eskisinden çok daha hızlı" karşılaştırması yapıldı) — kesin
        1000-kayıt ölçümü Faz H (sertleştirme) kapsamında yapılmalı.
        """
        inserted = 0
        updated = 0
        failed: list[tuple[str, str, str]] = []

        with self._session_factory() as session:
            for price in prices:
                try:
                    with session.begin_nested():  # SAVEPOINT — izole rollback alanı
                        existing = session.execute(
                            select(price_series_table.c.id).where(
                                price_series_table.c.symbol == price.symbol,
                                price_series_table.c.date == price.date.isoformat(),
                            )
                        ).first()

                        values = dict(
                            symbol=price.symbol,
                            symbol_type=price.symbol_type.value,
                            date=price.date.isoformat(),
                            open_price=str(price.open_price) if price.open_price is not None else None,
                            high_price=str(price.high_price) if price.high_price is not None else None,
                            low_price=str(price.low_price) if price.low_price is not None else None,
                            close_price=str(price.close_price),
                            adjusted_close=str(price.adjusted_close) if price.adjusted_close is not None else None,
                            volume=str(price.volume) if price.volume is not None else None,
                            source=price.source,
                            is_holiday=int(price.is_holiday),
                        )

                        if existing is not None:
                            session.execute(
                                price_series_table.update()
                                .where(price_series_table.c.id == existing.id)
                                .values(**values)
                            )
                        else:
                            session.execute(
                                price_series_table.insert().values(
                                    id=str(uuid.uuid4()),
                                    created_at=datetime.now(timezone.utc).isoformat(),
                                    **values,
                                )
                            )

                    if existing is not None:
                        updated += 1
                    else:
                        inserted += 1
                except Exception as exc:  # noqa: BLE001 — kasıtlı: bir hata batch'i durdurmamalı
                    logger.error(
                        "price_upsert_failed",
                        symbol=price.symbol,
                        date=price.date.isoformat(),
                        error=str(exc),
                    )
                    failed.append((price.symbol, price.date.isoformat(), str(exc)))

            session.commit()  # TEK commit — tüm başarılı kayıtlar için

        return BatchResult(inserted=inserted, updated=updated, failed=tuple(failed))

    def get_ohlcv(
        self, symbol: str, start: date, end: date, adjusted: bool = False
    ) -> Any:
        """
        Dönüş tipi gerçekte pandas.DataFrame'dir; `Any` olarak
        işaretlenmesinin nedeni pandas-stubs projede henüz kurulu
        değil (technical_calculator.py'da da aynı durum tespit edildi
        — mypy "Library stubs not installed for pandas" uyarısı
        veriyor). Bu, tip iddia edip yanlış çıkma riskini almaktansa
        dürüst bir Any tercih edilmesi.

        adjusted=True: close kolonu adjusted_close'dan doldurulur,
        adjusted_close NULL olan satırlarda close_price'a DÜŞER (design
        doc: "adjusted=True: adjusted_close kolonu kullan, None ise close").
        """
        import pandas as pd  # yerel import — domain/infra ayrımı: bu
        # repository infrastructure katmanında, pandas bağımlılığı
        # burada sorun değil (yalnızca domain katmanı pandas'tan bağımsız
        # kalmalı — cost_basis_calculator.py'da olduğu gibi).

        if start > end:
            return pd.DataFrame()

        stmt = (
            select(price_series_table)
            .where(
                price_series_table.c.symbol == symbol,
                price_series_table.c.date >= start.isoformat(),
                price_series_table.c.date <= end.isoformat(),
            )
            .order_by(price_series_table.c.date.asc())
        )
        with self._session_factory() as session:
            rows = list(session.execute(stmt))

        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            close = Decimal(row.close_price)
            adj_close = Decimal(row.adjusted_close) if row.adjusted_close is not None else None
            effective_close = (adj_close if adj_close is not None else close) if adjusted else close
            records.append({
                "date": date.fromisoformat(row.date),
                "open": Decimal(row.open_price) if row.open_price is not None else None,
                "high": Decimal(row.high_price) if row.high_price is not None else None,
                "low": Decimal(row.low_price) if row.low_price is not None else None,
                "close": effective_close,
                "adjusted_close": adj_close,
                "volume": Decimal(row.volume) if row.volume is not None else None,
            })

        df = pd.DataFrame.from_records(records)
        return df.set_index("date")

    def get_latest_price(self, symbol: str) -> PriceSeries | None:
        stmt = (
            select(price_series_table)
            .where(price_series_table.c.symbol == symbol)
            .order_by(price_series_table.c.date.desc())
            .limit(1)
        )
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return _row_to_price_series(row) if row is not None else None

    def get_price_on_date(self, symbol: str, on_date: date) -> Decimal | None:
        stmt = select(price_series_table.c.close_price).where(
            price_series_table.c.symbol == symbol,
            price_series_table.c.date == on_date.isoformat(),
        )
        with self._session_factory() as session:
            row = session.execute(stmt).first()
        return Decimal(row.close_price) if row is not None else None

    def get_missing_dates(
        self, symbol: str, start: date, end: date, trading_days_only: bool = True
    ) -> list[date]:
        """BİLİNEN SINIRLAMA için modül docstring'ine bkz. (yalnızca hafta sonu hariç tutulur)."""
        stmt = select(price_series_table.c.date).where(
            price_series_table.c.symbol == symbol,
            price_series_table.c.date >= start.isoformat(),
            price_series_table.c.date <= end.isoformat(),
        )
        with self._session_factory() as session:
            existing_dates = {date.fromisoformat(row.date) for row in session.execute(stmt)}

        missing: list[date] = []
        current = start
        one_day = date.resolution
        while current <= end:
            is_weekend = current.weekday() >= 5  # 5=Cumartesi, 6=Pazar
            if current not in existing_dates and not (trading_days_only and is_weekend):
                missing.append(current)
            current += one_day
        return missing
