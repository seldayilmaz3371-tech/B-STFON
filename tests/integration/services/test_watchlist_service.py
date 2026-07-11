"""WatchlistService testleri — gerçek SQLite repository ile."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.exceptions.domain_exceptions import (
    BusinessRuleError,
    DuplicateError,
    NotFoundError,
)
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.repositories.sqlite.watchlist_repository import (
    SQLiteWatchlistRepository,
)
from src.services.watchlist_service import WatchlistService

pytestmark = pytest.mark.integration


@pytest.fixture()
def service(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'watchlist_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    yield WatchlistService(SQLiteWatchlistRepository(sf))
    engine.dispose()


# ── create_watchlist ─────────────────────────────────────────────────────────

def test_create_watchlist(service):
    wl = service.create_watchlist("BIST Favorilerim", description="Takip ettiğim hisseler")
    assert wl.watchlist_id is not None
    assert wl.name == "BIST Favorilerim"


def test_create_watchlist_empty_name_raises(service):
    with pytest.raises(BusinessRuleError):
        service.create_watchlist("   ")


def test_create_watchlist_duplicate_name_raises(service):
    service.create_watchlist("Aynı İsim")
    with pytest.raises(DuplicateError):
        service.create_watchlist("Aynı İsim")


def test_list_watchlists(service):
    service.create_watchlist("Liste 1")
    service.create_watchlist("Liste 2")
    watchlists = service.list_watchlists()
    assert len(watchlists) == 2
    assert {w.name for w in watchlists} == {"Liste 1", "Liste 2"}


# ── add_symbol / remove_symbol ───────────────────────────────────────────────

def test_add_symbol_auto_classifies_bist(service):
    wl = service.create_watchlist("Test")
    item = service.add_symbol(wl.watchlist_id, "THYAO")
    assert item.symbol_type.value == "BIST_STOCK"
    assert item.item_id is not None


def test_add_symbol_auto_classifies_tefas(service):
    wl = service.create_watchlist("Test")
    item = service.add_symbol(wl.watchlist_id, "YAC")
    assert item.symbol_type.value == "TEFAS_FUND"


def test_add_symbol_empty_raises(service):
    wl = service.create_watchlist("Test")
    with pytest.raises(BusinessRuleError):
        service.add_symbol(wl.watchlist_id, "   ")


def test_add_duplicate_symbol_raises(service):
    wl = service.create_watchlist("Test")
    service.add_symbol(wl.watchlist_id, "THYAO")
    with pytest.raises(DuplicateError):
        service.add_symbol(wl.watchlist_id, "THYAO")


def test_add_symbol_with_price_alerts(service):
    wl = service.create_watchlist("Test")
    item = service.add_symbol(
        wl.watchlist_id, "THYAO",
        alert_price_low=Decimal("200.00"), alert_price_high=Decimal("300.00"),
    )
    assert item.alert_price_low == Decimal("200.00")
    assert item.alert_price_high == Decimal("300.00")


def test_add_symbol_invalid_alert_range_raises(service):
    """alert_price_low > alert_price_high — domain modelinin __post_init__'i yakalamalı."""
    wl = service.create_watchlist("Test")
    with pytest.raises(BusinessRuleError):
        service.add_symbol(
            wl.watchlist_id, "THYAO",
            alert_price_low=Decimal("300.00"), alert_price_high=Decimal("200.00"),
        )


def test_list_symbols(service):
    wl = service.create_watchlist("Test")
    service.add_symbol(wl.watchlist_id, "THYAO")
    service.add_symbol(wl.watchlist_id, "GARAN")
    items = service.list_symbols(wl.watchlist_id)
    assert len(items) == 2
    assert {i.symbol for i in items} == {"THYAO", "GARAN"}


def test_remove_symbol(service):
    wl = service.create_watchlist("Test")
    item = service.add_symbol(wl.watchlist_id, "THYAO")
    service.remove_symbol(item.item_id)
    assert service.list_symbols(wl.watchlist_id) == []


def test_remove_nonexistent_symbol_raises(service):
    with pytest.raises(NotFoundError):
        service.remove_symbol("var-olmayan-id")


def test_two_watchlists_can_have_same_symbol(service):
    """UNIQUE constraint (watchlist_id, symbol) — FARKLI watchlist'lerde AYNI sembol OLABİLİR."""
    wl1 = service.create_watchlist("Liste 1")
    wl2 = service.create_watchlist("Liste 2")
    service.add_symbol(wl1.watchlist_id, "THYAO")
    item2 = service.add_symbol(wl2.watchlist_id, "THYAO")  # ÇÖKMEMELİ
    assert item2.item_id is not None


# ── get_items_with_current_price (bu turda eklenen özellik) ────────────────

class _FakeMarketDataService:
    """MarketDataService.get_market_analysis() sözleşmesine uygun sahte servis."""

    def __init__(self, prices: dict[str, float | None], errors: dict[str, str] | None = None):
        self._prices = prices
        self._errors = errors or {}

    def get_market_analysis(self, symbol: str, timeframe: str):
        if symbol in self._errors:
            raise RuntimeError(self._errors[symbol])

        class _Result:
            latest_close = self._prices.get(symbol)

        return _Result()


def test_get_items_with_current_price_no_alarm(tmp_path):
    from src.infrastructure.database.connection import create_db_engine, create_session_factory, initialize_database
    from src.infrastructure.repositories.sqlite.watchlist_repository import SQLiteWatchlistRepository

    engine = create_db_engine(f"sqlite:///{tmp_path / 'wl_price_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    fake_mds = _FakeMarketDataService({"THYAO": 250.0})
    service = WatchlistService(SQLiteWatchlistRepository(sf), market_data_service=fake_mds)

    wl = service.create_watchlist("Test")
    service.add_symbol(wl.watchlist_id, "THYAO", alert_price_low=Decimal("200"), alert_price_high=Decimal("300"))

    statuses = service.get_items_with_current_price(wl.watchlist_id)
    assert len(statuses) == 1
    assert statuses[0].current_price == Decimal("250.0")
    assert statuses[0].alarm_triggered is False
    engine.dispose()


def test_get_items_with_current_price_low_alarm_triggered(tmp_path):
    from src.infrastructure.database.connection import create_db_engine, create_session_factory, initialize_database
    from src.infrastructure.repositories.sqlite.watchlist_repository import SQLiteWatchlistRepository

    engine = create_db_engine(f"sqlite:///{tmp_path / 'wl_price_test2.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    fake_mds = _FakeMarketDataService({"THYAO": 190.0})  # alt eşiğin ALTINDA
    service = WatchlistService(SQLiteWatchlistRepository(sf), market_data_service=fake_mds)

    wl = service.create_watchlist("Test")
    service.add_symbol(wl.watchlist_id, "THYAO", alert_price_low=Decimal("200"), alert_price_high=Decimal("300"))

    statuses = service.get_items_with_current_price(wl.watchlist_id)
    assert statuses[0].alarm_triggered is True
    assert statuses[0].alarm_direction == "low"
    engine.dispose()


def test_get_items_with_current_price_high_alarm_triggered(tmp_path):
    from src.infrastructure.database.connection import create_db_engine, create_session_factory, initialize_database
    from src.infrastructure.repositories.sqlite.watchlist_repository import SQLiteWatchlistRepository

    engine = create_db_engine(f"sqlite:///{tmp_path / 'wl_price_test3.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    fake_mds = _FakeMarketDataService({"THYAO": 310.0})  # üst eşiğin ÜSTÜNDE
    service = WatchlistService(SQLiteWatchlistRepository(sf), market_data_service=fake_mds)

    wl = service.create_watchlist("Test")
    service.add_symbol(wl.watchlist_id, "THYAO", alert_price_low=Decimal("200"), alert_price_high=Decimal("300"))

    statuses = service.get_items_with_current_price(wl.watchlist_id)
    assert statuses[0].alarm_triggered is True
    assert statuses[0].alarm_direction == "high"
    engine.dispose()


def test_get_items_with_current_price_isolates_per_symbol_failure(tmp_path):
    """
    KRİTİK: Bir sembolün fiyatı alınamazsa (ağ hatası), DİĞER
    sembollerin durumu ETKİLENMEMELİ — tüm liste tek bir hatadan
    dolayı gösterilemez hale gelmemeli.
    """
    from src.infrastructure.database.connection import create_db_engine, create_session_factory, initialize_database
    from src.infrastructure.repositories.sqlite.watchlist_repository import SQLiteWatchlistRepository

    engine = create_db_engine(f"sqlite:///{tmp_path / 'wl_price_test4.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    fake_mds = _FakeMarketDataService(
        {"GARAN": 55.0}, errors={"THYAO": "Ağ hatası simülasyonu"},
    )
    service = WatchlistService(SQLiteWatchlistRepository(sf), market_data_service=fake_mds)

    wl = service.create_watchlist("Test")
    service.add_symbol(wl.watchlist_id, "THYAO")
    service.add_symbol(wl.watchlist_id, "GARAN")

    statuses = service.get_items_with_current_price(wl.watchlist_id)
    assert len(statuses) == 2  # HER İKİSİ de listede, biri hata ile

    by_symbol = {s.item.symbol: s for s in statuses}
    assert by_symbol["THYAO"].fetch_error is not None
    assert by_symbol["THYAO"].current_price is None
    assert by_symbol["GARAN"].fetch_error is None
    assert by_symbol["GARAN"].current_price == Decimal("55.0")
    engine.dispose()


def test_get_items_with_current_price_no_market_data_service_injected(service):
    """market_data_service inject edilmezse (opsiyonel), ÇÖKMEMELİ."""
    wl = service.create_watchlist("Test")
    service.add_symbol(wl.watchlist_id, "THYAO")
    statuses = service.get_items_with_current_price(wl.watchlist_id)
    assert len(statuses) == 1
    assert statuses[0].fetch_error is not None
    assert statuses[0].current_price is None
