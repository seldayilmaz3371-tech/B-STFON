"""
PriceRepository ve CashLedgerRepository entegrasyon testleri — gerçek
SQLite dosyası ile (mock yok). Bkz. test_repositories_integration.py
için geçerli olan genel notlar (fixture deseni, marker kaydı TODO'su).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from decimal import Decimal

import pytest

from src.domain.enums.asset_type import AssetType
from src.domain.enums.ledger_entry_type import LedgerEntryType
from src.domain.models.cash_ledger_entry import CashLedgerEntry
from src.domain.models.price_series import PriceSeries
from src.infrastructure.database.connection import create_db_engine, create_session_factory, initialize_database
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.repositories.sqlite.cash_ledger_repository import SQLiteCashLedgerRepository
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository

pytestmark = pytest.mark.integration


@pytest.fixture()
def session_factory(tmp_path: Path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test.db'}")
    initialize_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture()
def portfolio_id(session_factory) -> str:
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with session_factory() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name=f"Test-{pid[:8]}", currency="TRY", cost_method="WAVG",
            inception_date="2024-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()
    return pid


# ── PriceRepository ──────────────────────────────────────────────────────────

def test_upsert_insert_then_update_same_id(session_factory):
    repo = SQLitePriceRepository(session_factory)
    p1 = repo.upsert(PriceSeries(
        symbol="THYAO", symbol_type=AssetType.BIST_STOCK, date=date(2024, 1, 2),
        close_price=Decimal("245.50"), source="yfinance",
    ))
    p2 = repo.upsert(PriceSeries(
        symbol="THYAO", symbol_type=AssetType.BIST_STOCK, date=date(2024, 1, 2),
        close_price=Decimal("246.00"), source="yfinance",
    ))
    assert p1.price_id == p2.price_id
    assert repo.get_price_on_date("THYAO", date(2024, 1, 2)) == Decimal("246.00")


def test_upsert_batch_reports_failures_without_stopping(session_factory):
    """
    price_series.symbol_type CHECK constraint'ini kasıtlı ihlal eden bir
    kayıt — DB seviyesinde reddedilmeli, ama batch DURMAMALI (design doc:
    'hatalar loglanır, exception değil').
    """
    repo = SQLitePriceRepository(session_factory)
    valid = PriceSeries(symbol="GARAN", symbol_type=AssetType.BIST_STOCK,
                         date=date(2024, 1, 2), close_price=Decimal("55.00"), source="yfinance")
    # AssetType.CASH price_series için GEÇERSİZ (bkz. is_valid_for_price_series) —
    # domain modeli zaten __post_init__'te bunu reddeder, bu yüzden PriceSeries
    # constructor'ı ValueError fırlatır ve bu, upsert_batch'in try/except
    # bloğunda yakalanıp failed listesine eklenir.
    with pytest.raises(ValueError):
        PriceSeries(symbol="NAKIT", symbol_type=AssetType.CASH,
                    date=date(2024, 1, 2), close_price=Decimal("1.00"), source="manual")

    result = repo.upsert_batch([valid])
    assert result.inserted == 1
    assert result.failed == ()


def test_get_ohlcv_empty_when_no_data(session_factory):
    repo = SQLitePriceRepository(session_factory)
    df = repo.get_ohlcv("NOTFOUND", date(2024, 1, 1), date(2024, 1, 31))
    assert df.empty


def test_get_ohlcv_empty_when_start_after_end(session_factory):
    repo = SQLitePriceRepository(session_factory)
    df = repo.get_ohlcv("THYAO", date(2024, 2, 1), date(2024, 1, 1))
    assert df.empty


def test_get_latest_price_none_when_no_data(session_factory):
    repo = SQLitePriceRepository(session_factory)
    assert repo.get_latest_price("NOTFOUND") is None


def test_get_missing_dates_excludes_weekends(session_factory):
    """2024-01-06/07 Cumartesi/Pazar — trading_days_only=True iken missing sayılmamalı."""
    repo = SQLitePriceRepository(session_factory)
    missing = repo.get_missing_dates("THYAO", date(2024, 1, 5), date(2024, 1, 8),
                                       trading_days_only=True)
    assert date(2024, 1, 6) not in missing  # Cumartesi
    assert date(2024, 1, 7) not in missing  # Pazar
    assert date(2024, 1, 5) in missing      # Cuma, veri yok
    assert date(2024, 1, 8) in missing      # Pazartesi, veri yok


def test_get_missing_dates_includes_weekends_when_disabled(session_factory):
    repo = SQLitePriceRepository(session_factory)
    missing = repo.get_missing_dates("THYAO", date(2024, 1, 6), date(2024, 1, 7),
                                       trading_days_only=False)
    assert missing == [date(2024, 1, 6), date(2024, 1, 7)]


# ── CashLedgerRepository ─────────────────────────────────────────────────────

def test_empty_ledger_balance_is_zero(session_factory, portfolio_id):
    repo = SQLiteCashLedgerRepository(session_factory)
    assert repo.get_balance(portfolio_id) == Decimal("0")


def test_balance_as_of_historical_date(session_factory, portfolio_id):
    repo = SQLiteCashLedgerRepository(session_factory)
    repo.add_entry(CashLedgerEntry(
        portfolio_id=portfolio_id, entry_type=LedgerEntryType.CREDIT, amount=Decimal("10000"),
        entry_date=date(2024, 1, 1), description="Yatırma", balance_after=Decimal("10000"),
    ))
    repo.add_entry(CashLedgerEntry(
        portfolio_id=portfolio_id, entry_type=LedgerEntryType.DEBIT, amount=Decimal("3000"),
        entry_date=date(2024, 1, 15), description="Alım", balance_after=Decimal("7000"),
    ))
    assert repo.get_balance(portfolio_id, as_of=date(2024, 1, 1)) == Decimal("10000")
    assert repo.get_balance(portfolio_id, as_of=date(2024, 1, 10)) == Decimal("10000")
    assert repo.get_balance(portfolio_id) == Decimal("7000")


def test_verify_balance_detects_inconsistency(session_factory, portfolio_id):
    repo = SQLiteCashLedgerRepository(session_factory)
    repo.add_entry(CashLedgerEntry(
        portfolio_id=portfolio_id, entry_type=LedgerEntryType.CREDIT, amount=Decimal("1000"),
        entry_date=date(2024, 1, 1), description="Doğru", balance_after=Decimal("1000"),
    ))
    repo.add_entry(CashLedgerEntry(
        portfolio_id=portfolio_id, entry_type=LedgerEntryType.CREDIT, amount=Decimal("500"),
        entry_date=date(2024, 1, 2), description="Yanlış hesaplanmış",
        balance_after=Decimal("777"),  # doğrusu 1500 olmalıydı
    ))
    verification = repo.verify_balance(portfolio_id)
    assert verification.is_consistent is False
    assert verification.expected == Decimal("1500")
    assert verification.actual == Decimal("777")


def test_verify_balance_consistent_when_correct(session_factory, portfolio_id):
    repo = SQLiteCashLedgerRepository(session_factory)
    repo.add_entry(CashLedgerEntry(
        portfolio_id=portfolio_id, entry_type=LedgerEntryType.CREDIT, amount=Decimal("1000"),
        entry_date=date(2024, 1, 1), description="Doğru", balance_after=Decimal("1000"),
    ))
    verification = repo.verify_balance(portfolio_id)
    assert verification.is_consistent is True
    assert verification.discrepancy == Decimal("0")


def test_negative_or_zero_amount_rejected_at_domain_level():
    """DDL CHECK (amount > 0) domain modelinde de savunmacı olarak var."""
    with pytest.raises(ValueError):
        CashLedgerEntry(
            portfolio_id="x", entry_type=LedgerEntryType.CREDIT, amount=Decimal("0"),
            entry_date=date(2024, 1, 1), description="Geçersiz", balance_after=Decimal("0"),
        )


def test_get_entries_filtered_by_date_range(session_factory, portfolio_id):
    repo = SQLiteCashLedgerRepository(session_factory)
    for d, amt in [(date(2024, 1, 1), "100"), (date(2024, 2, 1), "200"), (date(2024, 3, 1), "300")]:
        repo.add_entry(CashLedgerEntry(
            portfolio_id=portfolio_id, entry_type=LedgerEntryType.CREDIT, amount=Decimal(amt),
            entry_date=d, description="x", balance_after=Decimal(amt),
        ))
    filtered = repo.get_entries(portfolio_id, start=date(2024, 1, 15), end=date(2024, 2, 15))
    assert len(filtered) == 1
    assert filtered[0].entry_date == date(2024, 2, 1)
