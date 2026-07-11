"""FundRepository ve CorporateActionRepository testleri — gerçek SQLite ile."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.domain.exceptions.domain_exceptions import NotFoundError
from src.domain.models.corporate_action import CorporateAction
from src.domain.models.fund import Fund
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.repositories.sqlite.corporate_action_repository import (
    SQLiteCorporateActionRepository,
)
from src.infrastructure.repositories.sqlite.fund_repository import SQLiteFundRepository

pytestmark = pytest.mark.integration


@pytest.fixture()
def repos(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'fund_ca_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    yield {
        "fund": SQLiteFundRepository(sf),
        "ca": SQLiteCorporateActionRepository(sf),
    }
    engine.dispose()


# ── Fund domain modeli doğrulaması ──────────────────────────────────────────

def test_fund_invalid_type_raises():
    with pytest.raises(ValueError):
        Fund(fund_code="YAC", fund_name="Yatırım Fonu", fund_type="GECERSIZ")


def test_fund_empty_code_raises():
    with pytest.raises(ValueError):
        Fund(fund_code="  ", fund_name="Test", fund_type="YAT")


def test_fund_allocation_out_of_range_raises():
    with pytest.raises(ValueError):
        Fund(fund_code="YAC", fund_name="Test", fund_type="YAT", stock_pct=Decimal("150"))


def test_fund_partial_allocation_is_valid():
    """TEFAS her zaman TAM allocation vermez — kısmi veri GEÇERLİ olmalı."""
    fund = Fund(fund_code="YAC", fund_name="Test", fund_type="YAT", stock_pct=Decimal("60"))
    assert fund.bond_pct is None  # eksik ama hata değil


# ── FundRepository ───────────────────────────────────────────────────────────

def test_fund_upsert_and_get(repos):
    fund = Fund(
        fund_code="YAC", fund_name="Yapı Kredi Portföy Test Fonu", fund_type="YAT",
        founder="Yapı Kredi Portföy", stock_pct=Decimal("70.5"), last_nav=Decimal("1.523"),
    )
    repos["fund"].upsert(fund)

    retrieved = repos["fund"].get_by_code("YAC")
    assert retrieved is not None
    assert retrieved.fund_name == "Yapı Kredi Portföy Test Fonu"
    assert retrieved.stock_pct == Decimal("70.5")


def test_fund_upsert_updates_existing(repos):
    """Design doc'un 'last_nav (cache)' semantiği — İKİNCİ upsert, İLKİNİ günceller (çakışma hatası DEĞİL)."""
    fund_v1 = Fund(fund_code="YAC", fund_name="Test Fonu", fund_type="YAT", last_nav=Decimal("1.500"))
    repos["fund"].upsert(fund_v1)

    fund_v2 = Fund(fund_code="YAC", fund_name="Test Fonu", fund_type="YAT", last_nav=Decimal("1.600"))
    repos["fund"].upsert(fund_v2)

    retrieved = repos["fund"].get_by_code("YAC")
    assert retrieved.last_nav == Decimal("1.600")  # GÜNCELLENDİ, ÇOĞALMADI

    all_funds = repos["fund"].list_by_type()
    assert len(all_funds) == 1  # hâlâ TEK kayıt


def test_fund_get_nonexistent_returns_none(repos):
    assert repos["fund"].get_by_code("YOK") is None


def test_fund_list_by_type_filters_correctly(repos):
    repos["fund"].upsert(Fund(fund_code="YAC", fund_name="Yatırım Fonu", fund_type="YAT"))
    repos["fund"].upsert(Fund(fund_code="TCD", fund_name="Emeklilik Fonu", fund_type="EMK"))

    yat_funds = repos["fund"].list_by_type(fund_type="YAT")
    assert len(yat_funds) == 1
    assert yat_funds[0].fund_code == "YAC"


# ── CorporateAction domain modeli doğrulaması ───────────────────────────────

def test_corporate_action_invalid_type_raises():
    with pytest.raises(ValueError):
        CorporateAction(symbol="THYAO", action_type="GECERSIZ", ex_date=date.today())


def test_corporate_action_empty_symbol_raises():
    with pytest.raises(ValueError):
        CorporateAction(symbol="", action_type="SPLIT", ex_date=date.today())


# ── CorporateActionRepository ────────────────────────────────────────────────

def test_corporate_action_create_and_get(repos):
    action = CorporateAction(
        symbol="THYAO", action_type="BONUS_SHARE", ex_date=date(2024, 6, 15),
        action_data={"ratio": "0.25"}, source="KAP",
    )
    created = repos["ca"].create(action)

    assert created.action_id is not None
    retrieved = repos["ca"].get_by_id(created.action_id)
    assert retrieved.action_data == {"ratio": "0.25"}
    assert retrieved.is_applied is False  # varsayılan: HENÜZ uygulanmadı


def test_corporate_action_list_pending_excludes_applied(repos):
    a1 = repos["ca"].create(CorporateAction(symbol="THYAO", action_type="SPLIT", ex_date=date(2024, 1, 1), action_data={"ratio": "2.0"}))
    a2 = repos["ca"].create(CorporateAction(symbol="GARAN", action_type="DIVIDEND", ex_date=date(2024, 2, 1), action_data={"dividend_per_share": "1.50"}))

    repos["ca"].mark_applied(a1.action_id)

    pending = repos["ca"].list_pending()
    pending_ids = {p.action_id for p in pending}
    assert a1.action_id not in pending_ids  # uygulandı, artık bekleyen listede DEĞİL
    assert a2.action_id in pending_ids


def test_corporate_action_mark_applied_does_not_change_data(repos):
    """
    KRİTİK: mark_applied() YALNIZCA bayrağı değiştirir — action_data'ya
    HİÇ dokunmaz (bkz. repository docstring'i, 'GERÇEKTEN hiçbir
    hesaplamayı tetiklemiyor' garantisi).
    """
    action = repos["ca"].create(CorporateAction(
        symbol="THYAO", action_type="SPLIT", ex_date=date(2024, 1, 1), action_data={"ratio": "2.0"},
    ))
    repos["ca"].mark_applied(action.action_id)

    retrieved = repos["ca"].get_by_id(action.action_id)
    assert retrieved.is_applied is True
    assert retrieved.action_data == {"ratio": "2.0"}  # DEĞİŞMEDİ


def test_corporate_action_mark_applied_nonexistent_raises(repos):
    with pytest.raises(NotFoundError):
        repos["ca"].mark_applied("var-olmayan-id")


def test_corporate_action_list_by_symbol(repos):
    repos["ca"].create(CorporateAction(symbol="THYAO", action_type="SPLIT", ex_date=date(2024, 1, 1), action_data={}))
    repos["ca"].create(CorporateAction(symbol="THYAO", action_type="DIVIDEND", ex_date=date(2024, 6, 1), action_data={}))
    repos["ca"].create(CorporateAction(symbol="GARAN", action_type="DIVIDEND", ex_date=date(2024, 3, 1), action_data={}))

    thyao_actions = repos["ca"].list_by_symbol("THYAO")
    assert len(thyao_actions) == 2
    # en yeni önce (ex_date DESC)
    assert thyao_actions[0].ex_date == date(2024, 6, 1)
