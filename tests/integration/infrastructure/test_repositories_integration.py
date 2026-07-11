"""
Persistence katmanı entegrasyon testleri — GERÇEK SQLite dosyası ile
(mock yok). Her test izole geçici bir DB dosyası kullanır (tmp_path).

Bu testler unit test DEĞİL — @pytest.mark.integration ile işaretli.
Tasarım belgesi Faz 1 DoD kriteri: "Integration testler @pytest.mark.network
ile işaretli (CI'da skip)" — burada network değil disk I/O var, ama
aynı ayrım mantığı: hızlı unit test paketinden ayrı çalıştırılabilmeli.

pytest.ini / pyproject.toml'a şu marker kaydı eklenmeli (henüz
eklenmedi, bu bilinçli bir TODO — pyproject.toml Faz H'de sonlanacak):
    [tool.pytest.ini_options]
    markers = ["integration: gerçek DB/dosya sistemi kullanır"]
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.domain.calculators.cost_basis_calculator import WAVGCostBasisCalculator
from src.domain.enums.transaction_type import TransactionType
from src.domain.models.transaction import Transaction
from src.infrastructure.database.connection import (
    check_database_connection,
    create_db_engine,
    create_session_factory,
    initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.repositories.sqlite.portfolio_repository import (
    SQLitePortfolioRepository,
)
from src.infrastructure.repositories.sqlite.transaction_repository import (
    SQLiteTransactionRepository,
)

pytestmark = pytest.mark.integration


@pytest.fixture()
def db_engine(tmp_path: Path):
    """Her test için izole, gerçek bir SQLite dosyası."""
    db_path = tmp_path / "test.db"
    engine = create_db_engine(f"sqlite:///{db_path}", echo=False)
    initialize_database(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def session_factory(db_engine):
    return create_session_factory(db_engine)


@pytest.fixture()
def portfolio_id(session_factory) -> str:
    """Testler için hazır bir portföy satırı."""
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with session_factory() as session:
        session.execute(
            portfolios_table.insert().values(
                id=pid, name=f"Test-{pid[:8]}", currency="TRY",
                cost_method="WAVG", inception_date="2024-01-01",
                is_active=1, created_at=now, updated_at=now,
            )
        )
        session.commit()
    return pid


# ── connection.py ────────────────────────────────────────────────────────────

def test_check_database_connection_true_after_init(db_engine):
    assert check_database_connection(db_engine) is True


def test_wal_mode_actually_active_on_disk(tmp_path):
    """
    Faz 0 DoD: 'SQLite WAL mode aktif (PRAGMA doğrulaması testi var)'.
    check_database_connection'a değil, HAM sqlite3'e sorarak doğrular
    — bu, connection.py'daki iddiayı bağımsız bir kanaldan teyit eder.
    """
    import sqlite3

    db_path = tmp_path / "wal_test.db"
    engine = create_db_engine(f"sqlite:///{db_path}", echo=False)
    initialize_database(engine)
    engine.dispose()

    raw = sqlite3.connect(str(db_path))
    mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
    raw.close()
    assert mode.lower() == "wal"


def test_initialize_database_is_idempotent(db_engine):
    """İkinci kez çağrıldığında hata fırlatmamalı (create_all checkfirst)."""
    initialize_database(db_engine)  # fixture zaten bir kez çağırdı
    assert check_database_connection(db_engine) is True


# ── SQLitePortfolioRepository ───────────────────────────────────────────────

def test_portfolio_repository_list_all_empty(session_factory):
    repo = SQLitePortfolioRepository(session_factory)
    assert repo.list_all() == []


def test_portfolio_repository_list_all_returns_created(session_factory, portfolio_id):
    repo = SQLitePortfolioRepository(session_factory)
    records = repo.list_all()
    assert len(records) == 1
    assert records[0].id == portfolio_id
    assert records[0].is_active is True


def test_portfolio_repository_excludes_inactive_by_default(session_factory):
    now = datetime.now(timezone.utc).isoformat()
    active_id, inactive_id = str(uuid.uuid4()), str(uuid.uuid4())
    with session_factory() as session:
        session.execute(portfolios_table.insert().values(
            id=active_id, name="Aktif", currency="TRY", cost_method="WAVG",
            inception_date="2024-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.execute(portfolios_table.insert().values(
            id=inactive_id, name="Pasif", currency="TRY", cost_method="WAVG",
            inception_date="2024-01-01", is_active=0, created_at=now, updated_at=now,
        ))
        session.commit()

    repo = SQLitePortfolioRepository(session_factory)
    assert [r.id for r in repo.list_all()] == [active_id]
    assert {r.id for r in repo.list_all(include_inactive=True)} == {active_id, inactive_id}


# ── SQLiteTransactionRepository ─────────────────────────────────────────────

def test_transaction_roundtrip_preserves_decimal_precision(session_factory, portfolio_id):
    """
    KRİTİK finansal doğruluk testi: TEXT storage → Decimal round-trip
    hassasiyet kaybetmemeli. 1/3 gibi tekrarlı ondalıklar YERİNE gerçek
    parasal bir değer (11.333333 — GD-001'deki WAVG avg_cost) kullanılıyor.
    """
    repo = SQLiteTransactionRepository(session_factory)
    original = Transaction(
        symbol="THYAO", transaction_type=TransactionType.BUY,
        timestamp=datetime(2024, 1, 2), quantity=Decimal("100"),
        price=Decimal("11.333333"),
    )
    repo.add_transaction(portfolio_id, "BIST_STOCK", original)

    fetched = repo.get_by_symbol(portfolio_id, "THYAO")
    assert len(fetched) == 1
    assert fetched[0].price == Decimal("11.333333")  # tam eşitlik, tolerans YOK
    assert fetched[0].quantity == Decimal("100")


def test_get_portfolio_symbols_distinct(session_factory, portfolio_id):
    repo = SQLiteTransactionRepository(session_factory)
    for symbol in ("THYAO", "THYAO", "GARAN"):
        repo.add_transaction(portfolio_id, "BIST_STOCK", Transaction(
            symbol=symbol, transaction_type=TransactionType.BUY,
            timestamp=datetime(2024, 1, 2), quantity=Decimal("10"), price=Decimal("1"),
        ))
    assert sorted(repo.get_portfolio_symbols(portfolio_id)) == ["GARAN", "THYAO"]


def test_get_by_symbol_ordered_by_trade_date(session_factory, portfolio_id):
    repo = SQLiteTransactionRepository(session_factory)
    # Kasıtlı olarak TERS sırada ekleniyor — repository'nin ORDER BY
    # ile doğru sıraya koyduğunu doğrulamak için.
    repo.add_transaction(portfolio_id, "BIST_STOCK", Transaction(
        symbol="THYAO", transaction_type=TransactionType.BUY,
        timestamp=datetime(2024, 3, 1), quantity=Decimal("1"), price=Decimal("1")))
    repo.add_transaction(portfolio_id, "BIST_STOCK", Transaction(
        symbol="THYAO", transaction_type=TransactionType.BUY,
        timestamp=datetime(2024, 1, 1), quantity=Decimal("2"), price=Decimal("1")))

    fetched = repo.get_by_symbol(portfolio_id, "THYAO")
    assert [t.timestamp for t in fetched] == sorted(t.timestamp for t in fetched)


def test_end_to_end_gd001_from_real_database(session_factory, portfolio_id):
    """
    UÇTAN UCA: DB'ye yazılan gerçek satırlardan okunan Transaction'lar
    CostBasisCalculator'a verildiğinde GD-001 sonucunu üretmeli.
    Bu, repository + domain calculator entegrasyonunun tek testte
    kanıtı.
    """
    repo = SQLiteTransactionRepository(session_factory)
    for symbol_tx in [
        Transaction(symbol="THYAO", transaction_type=TransactionType.BUY,
                    timestamp=datetime(2024, 1, 2), quantity=Decimal("100"), price=Decimal("10.00")),
        Transaction(symbol="THYAO", transaction_type=TransactionType.BUY,
                    timestamp=datetime(2024, 2, 1), quantity=Decimal("50"), price=Decimal("14.00")),
        Transaction(symbol="THYAO", transaction_type=TransactionType.SELL,
                    timestamp=datetime(2024, 3, 1), quantity=Decimal("80"), price=Decimal("16.00")),
    ]:
        repo.add_transaction(portfolio_id, "BIST_STOCK", symbol_tx)

    fetched = repo.get_by_symbol(portfolio_id, "THYAO")
    result = WAVGCostBasisCalculator().calculate(fetched)

    assert result.total_quantity == Decimal("70")
    assert result.total_cost_basis.quantize(Decimal("0.01")) == Decimal("793.33")
    assert result.realized_pnl.quantize(Decimal("0.01")) == Decimal("373.33")


def test_reversed_transaction_excluded_from_symbols(session_factory, portfolio_id):
    """
    is_active=0 olan (reversal edilmiş) işlemler get_portfolio_symbols
    ve get_by_symbol'den DIŞLANMALI. Şu an add_transaction() reversal
    mekanizması sağlamıyor (Faz C+ kapsamı) — bu test doğrudan SQL ile
    is_active=0 yaparak repository'nin FİLTRELEME davranışını izole
    doğruluyor.
    """
    from src.infrastructure.database.orm_models import transactions_table

    repo = SQLiteTransactionRepository(session_factory)
    tx_id = repo.add_transaction(portfolio_id, "BIST_STOCK", Transaction(
        symbol="THYAO", transaction_type=TransactionType.BUY,
        timestamp=datetime(2024, 1, 2), quantity=Decimal("10"), price=Decimal("1")))

    with session_factory() as session:
        session.execute(
            transactions_table.update()
            .where(transactions_table.c.id == tx_id)
            .values(is_active=0)
        )
        session.commit()

    assert repo.get_portfolio_symbols(portfolio_id) == []
    assert repo.get_by_symbol(portfolio_id, "THYAO") == []
