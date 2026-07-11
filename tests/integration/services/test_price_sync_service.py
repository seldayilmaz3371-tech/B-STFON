"""
PriceSyncService testleri — gerçek SQLite PriceRepository, sahte
(ama sözleşmeye uygun) MarketDataProvider.

Üç senaryo AYRI AYRI doğrulanıyor (bkz. modül docstring'i "gizlenmiş
karmaşıklık kabul edilemez"):
  1. Cache TAM → canlı sağlayıcıya HİÇ dokunmaz (call sayacı ile kanıtlanır)
  2. Cache KISMEN eksik → yalnızca eksik aralığı çeker
  3. Cache BOŞ → tam pencereyi çeker
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.services.price_sync_service import PriceSyncService

pytestmark = pytest.mark.integration


class CountingProvider:
    """fetch_ohlcv çağrı sayısını VE hangi aralıkla çağrıldığını izler."""

    def __init__(self, full_series: pd.DataFrame):
        self._full_series = full_series
        self.call_count = 0
        self.last_call_range: tuple | None = None

    def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
        self.call_count += 1
        self.last_call_range = (start_date, end_date)
        mask = pd.Series(True, index=self._full_series.index)
        if start_date is not None:
            mask &= self._full_series.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= self._full_series.index <= pd.Timestamp(end_date)
        return self._full_series[mask]

    def get_provider_name(self) -> str:
        return "mock"


def _make_series(n_days=30, base_price=100.0):
    dates = pd.date_range(end=date.today(), periods=n_days, freq="D").normalize()
    return pd.DataFrame({
        "Open": [base_price] * n_days, "High": [base_price + 1] * n_days,
        "Low": [base_price - 1] * n_days, "Close": [base_price] * n_days,
        "Volume": [1_000_000] * n_days,
    }, index=dates)


@pytest.fixture()
def price_repo(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'price_sync_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    yield SQLitePriceRepository(sf)
    engine.dispose()


# ── Senaryo 1: Cache TAM ─────────────────────────────────────────────────────

def test_cache_hit_does_not_touch_provider(price_repo):
    provider = CountingProvider(_make_series(30))
    service = PriceSyncService(price_repo, provider)

    start, end = date.today() - timedelta(days=20), date.today()
    # İlk çağrı: cache boş, provider'a gidilir
    service.get_price_history("THYAO", start, end)
    assert provider.call_count == 1

    # İKİNCİ çağrı: AYNI aralık, artık cache'te TAM veri var
    service.get_price_history("THYAO", start, end)
    assert provider.call_count == 1  # HİÇ ARTMADI — provider'a dokunulmadı


# ── Senaryo 2: Cache KISMEN eksik ────────────────────────────────────────────

def test_partial_cache_fetches_only_missing_range(price_repo):
    provider = CountingProvider(_make_series(60))
    service = PriceSyncService(price_repo, provider)

    # Önce eski bir aralığı senkronize et (30-20 gün önce)
    old_start = date.today() - timedelta(days=30)
    old_end = date.today() - timedelta(days=20)
    service.get_price_history("THYAO", old_start, old_end)
    first_call_range = provider.last_call_range

    # Şimdi DAHA GENİŞ bir aralık iste — yalnızca YENİ kısmı çekmeli
    provider.call_count = 0  # sayaç sıfırla
    wide_start = date.today() - timedelta(days=30)
    wide_end = date.today()
    df = service.get_price_history("THYAO", wide_start, wide_end)

    assert provider.call_count >= 1  # eksik kısım için EN AZ bir çağrı
    # Kritik: tam pencereyi (30 gün) DEĞİL, yalnızca eksik kısmı istemeli
    called_start, called_end = provider.last_call_range
    assert called_start.date() > wide_start  # baştan itibaren DEĞİL


# ── Senaryo 3: Cache BOŞ ─────────────────────────────────────────────────────

def test_empty_cache_fetches_full_window(price_repo):
    provider = CountingProvider(_make_series(30))
    service = PriceSyncService(price_repo, provider)

    start, end = date.today() - timedelta(days=25), date.today()
    df = service.get_price_history("GARAN", start, end)

    assert provider.call_count == 1
    assert not df.empty


# ── sync_symbol (scheduler tarafından çağrılan proaktif yol) ───────────────

def test_sync_symbol_populates_cache(price_repo):
    provider = CountingProvider(_make_series(200))
    service = PriceSyncService(price_repo, provider)

    result = service.sync_symbol("THYAO", lookback_days=90)
    assert result["cached"] > 0

    # Artık price_repo'da GERÇEKTEN veri var mı doğrula
    latest = price_repo.get_latest_price("THYAO")
    assert latest is not None


def test_sync_symbol_noop_when_already_synced(price_repo):
    """
    NOT: sync_symbol(lookback_days=90) takvim günü olarak 180 gün
    geriye gidiyor (hafta sonu payı) — bu yüzden sahte seri EN AZ
    180 gün kapsamalı, aksi halde "sağlayıcıda hiç olmayan tarihler"
    KALICI bir boşluk olarak görünür ve her senkronizasyonda tekrar
    denenir (zararsız ama israf — gerçek dünyada yakın zamanda halka
    arz olmuş bir sembol için beklenen davranış, bkz. modül docstring'i).
    Bu test, o senaryoyu DEĞİL, "cache zaten tam" senaryosunu test
    ediyor — bu yüzden seri yeterince geniş (200 gün) tutuldu.
    """
    provider = CountingProvider(_make_series(200))
    service = PriceSyncService(price_repo, provider)

    service.sync_symbol("THYAO", lookback_days=90)
    provider.call_count = 0
    result = service.sync_symbol("THYAO", lookback_days=90)

    assert result == {"fetched": 0, "cached": 0}
    assert provider.call_count == 0  # zaten güncel, tekrar çekmedi


# ── Hata durumları ───────────────────────────────────────────────────────────

def test_provider_failure_does_not_crash_sync(price_repo):
    class FailingProvider:
        def fetch_ohlcv(self, *a, **kw):
            raise ConnectionError("Ağ hatası simülasyonu")

        def get_provider_name(self):
            return "failing"

    service = PriceSyncService(price_repo, FailingProvider())
    result = service.sync_symbol("THYAO", lookback_days=30)
    assert result == {"fetched": 0, "cached": 0}  # ÇÖKMEDİ, boş sonuç döndü


def test_empty_provider_response_handled_gracefully(price_repo):
    class EmptyProvider:
        def fetch_ohlcv(self, *a, **kw):
            return pd.DataFrame()

        def get_provider_name(self):
            return "empty"

    service = PriceSyncService(price_repo, EmptyProvider())
    result = service.sync_symbol("BILINMEYEN", lookback_days=30)
    assert result == {"fetched": 0, "cached": 0}
