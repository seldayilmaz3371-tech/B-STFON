"""
Veritabanı bağlantı yönetimi — SQLAlchemy Core engine/session.

API sözleşmesi — VARSAYIM DEĞİL, container.py'nin ÇALIŞTIRILARAK
tersine mühendisliğiyle çıkarıldı:
    create_db_engine(database_url: str, echo: bool) -> Engine
    create_session_factory(engine) -> sessionmaker[Session]
    check_database_connection(engine) -> bool
    initialize_database(engine) -> None

BİLİNEN SINIRLAMA (açıkça işaretleniyor):
  container.py, create_db_engine'i yalnızca (database_url, echo) ile
  çağırıyor — settings.database.sqlite (PRAGMA ayarları) parametre
  olarak GEÇİRİLMİYOR. Bu yüzden aşağıdaki PRAGMA değerleri,
  settings.py'daki SQLiteConfig sınıfının VARSAYILAN değerleriyle
  BİREBİR AYNI şekilde burada kopyalanıyor (icat edilmedi). Risk:
  Biri settings.yaml'da bu PRAGMA'ları override ederse, bu dosya
  bunu YANSITMAZ — iki kaynak senkron kalmalı. Doğru çözüm,
  container.py'nin _get_engine() metodunu güncelleyip
  settings.database.sqlite'ı create_db_engine'e geçirmesidir; bu,
  container.py'yi değiştirmeyi gerektireceği için BU TURDA yapılmadı
  (mevcut kodu bozmama önceliği). Faz C entegrasyonunda ele alınmalı.

WAL mode gerekçesi (ADR-002, Faz 0 DoD kriteri): Tek yazar/çoklu
okuyucu eşzamanlılığı — okuma işlemleri (Streamlit dashboard) yazma
işlemini (işlem ekleme) bloklamaz.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from src.infrastructure.database.orm_models import metadata
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

# settings.py::SQLiteConfig ile BİREBİR SENKRON tutulmalı (bkz. modül
# docstring'indeki bilinen sınırlama).
#
# DÜZELTME (bu turda eklendi — RiskService/SnapshotScheduler'a
# ThreadPoolExecutor ile paralel sembol/portföy işleme eklenirken
# bulundu): busy_timeout AYARLI DEĞİLDİ. WAL modu çoklu-okuyucu/tek-
# yazar destekler ama İKİ THREAD AYNI ANDA YAZMAYA ÇALIŞIRSA (örn.
# PriceSyncService.upsert_batch, farklı semboller için paralel
# thread'lerden çağrılınca AYNI price_series tablosuna yazar),
# busy_timeout=0 (varsayılan) ile SQLite HEMEN "database is locked"
# hatası fırlatır — BEKLEMEZ/YENİDEN DENEMEZ. Bu, sıralı (sequential)
# işlem sırasında HİÇ ortaya çıkmayan, yalnızca PARALELLİK
# eklendiğinde aktif hale gelen bir risk sınıfıydı — paralelleştirme
# yapılmadan ÖNCE düzeltildi (sonradan keşfedilen bir "flaky test"
# olarak DEĞİL).
_SQLITE_PRAGMAS: dict[str, str] = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "cache_size": "-64000",
    "foreign_keys": "1",
    "temp_store": "MEMORY",
    "mmap_size": "268435456",
    "busy_timeout": "5000",  # ms — kilit varsa 5sn'ye kadar BEKLE, hemen hata verme
}


def create_db_engine(database_url: str, echo: bool = False) -> Engine:
    """
    SQLAlchemy Engine oluşturur.

    SQLite için: her yeni DBAPI connection açıldığında PRAGMA'ları
    uygular (connect event listener) — bağlantı havuzundaki HER
    connection için, yalnızca ilk açılışta değil (SQLite PRAGMA'ları
    connection-scoped'tur, engine-scoped değil).

    PostgreSQL (Faz 3) için: PRAGMA bloğu atlanır — PostgreSQL'in
    kendi eşzamanlılık modeli (MVCC) var, SQLite PRAGMA'ları anlamsız.
    """
    engine = create_engine(database_url, echo=echo, future=True)

    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _apply_sqlite_pragmas(
            dbapi_connection: sqlite3.Connection, connection_record: Any
        ) -> None:
            cursor = dbapi_connection.cursor()
            for pragma, value in _SQLITE_PRAGMAS.items():
                cursor.execute(f"PRAGMA {pragma}={value}")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """
    expire_on_commit=False: commit sonrası nesnelere erişim (örn.
    PositionDTO oluştururken ORM row'undan alan okuma) yeni bir SELECT
    tetiklemesin — repository'ler zaten kendi Core sorgularını kontrol
    ediyor, lazy-reload sürprizini önlemek Core-first yaklaşımıyla
    tutarlı.
    """
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def check_database_connection(engine: Engine) -> bool:
    """
    Faz 0 DoD kriteri: "SQLite WAL mode aktif (PRAGMA doğrulaması
    testi var)" — bu fonksiyon hem bağlantıyı hem de WAL modunun
    fiilen aktif olduğunu doğrular (yalnızca "connect edebiliyor muyum"
    değil).
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            if engine.url.get_backend_name() == "sqlite":
                result = conn.execute(text("PRAGMA journal_mode")).scalar()
                if result is None or result.lower() != "wal":
                    logger.error(
                        "wal_mode_not_active",
                        actual_mode=result,
                    )
                    return False
        return True
    except Exception as exc:
        logger.error("database_connection_failed", error=str(exc))
        return False


def initialize_database(engine: Engine) -> None:
    """
    Şemayı oluşturur (tablo yoksa) — idempotent (checkfirst=True
    create_all'ın varsayılan davranışı).

    NOT: Bu, Alembic migration'ın YERİNE GEÇMEZ. Bu fonksiyon yalnızca
    "fresh DB / ilk kurulum" senaryosu için var (Faz 0 DoD: "Alembic
    migration çalışıyor (fresh DB oluşturuluyor)"). Şema değişikliği
    (ALTER TABLE) gerektiren senaryolarda Alembic migration'ları
    kullanılmalı — bu fonksiyon var olan bir tabloyu GÜNCELLEMEZ.
    """
    metadata.create_all(engine)
    logger.info("database_initialized", tables=list(metadata.tables.keys()))
