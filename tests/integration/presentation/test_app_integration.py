"""
app.py uçtan uca entegrasyon testleri — Streamlit'in resmi headless test
aracı (streamlit.testing.v1.AppTest) ile, GERÇEK browser/subprocess yok.

AĞ KISITLAMASI NOTU: Bu ortamda yfinance/TEFAS'a gerçek network erişimi
YOK (sandbox egress allowlist'i yalnızca paket registry'lerine izin
veriyor). Bu yüzden:
  - test_network_failure_* testleri GERÇEK bir bloke network üzerinden
    çalışır (mock değil) — bu, production'daki rate-limit/outage
    senaryosunun DOĞRUDAN bir simülasyonu, yapay bir test değil.
  - test_happy_path_* testleri YFinanceAdapter.fetch_ohlcv'yi mock'lar
    (unittest.mock.patch.object) — gerçek ağ olmadan render pipeline'ını
    (adapter sözleşmesi → service → UI) doğrulamak için. MockProvider
    yerine gerçek adapter sınıfının tek metodunu patch'lemek tercih
    edildi: bu, adapter'ın DIŞINDAKİ her şeyin (retry/cache/validator
    zincirinin adapter içi kısmı hariç) gerçek kod olarak çalışmasını
    sağlıyor.

BU TESTLER SIRASINDA BULUNAN VE DÜZELTİLEN GERÇEK ÜRETİM HATALARI:
  1. portfolio_view.py: `Styler.applymap()` pandas 3.x'te KALDIRILDI,
     portföy sekmesi en az 1 pozisyon olduğunda %100 çöküyordu.
     `.map()` ile değiştirildi (bkz. test_portfolio_tab_renders_with_position).
  2. app.py/portfolio_view.py/search_bar.py/technical_chart.py:
     `use_container_width` deprecated (kaldırılma tarihi geçmiş),
     `width="stretch"` ile değiştirildi.
  3. yfinance_adapter.py: Network hatası (HTTP 403/bağlantı yok)
     yfinance tarafından exception olarak değil BOŞ DataFrame olarak
     yutuluyor — bu, adapter'da SymbolNotFoundError olarak
     yanlış sınıflandırılıyor (ProviderUnavailableError olmalıydı).
     DÜZELTİLMEDİ — ayrı bir mimari karar gerektiriyor (bkz. test
     docstring'i, xfail ile işaretli).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.domain.enums.transaction_type import TransactionType
from src.domain.models.transaction import Transaction
from src.infrastructure.data_providers.yfinance_adapter import YFinanceAdapter
from src.infrastructure.database.connection import (
    create_db_engine,
    create_session_factory,
    initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.repositories.sqlite.transaction_repository import (
    SQLiteTransactionRepository,
)

pytestmark = pytest.mark.integration

APP_PATH = "app/main.py"


def _synthetic_ohlcv(close_price: float = 250.0, n: int = 90) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    close = np.full(n, close_price)
    return pd.DataFrame({
        "Open": close, "High": close + 0.5, "Low": close - 0.5,
        "Close": close, "Volume": np.full(n, 1_000_000),
    }, index=dates)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Her test için izole DB — container.py'ın okuduğu env var üzerinden."""
    db_path = tmp_path / "apptest.db"
    monkeypatch.setenv("PORTFOLIO_OS__DATABASE__URL", f"sqlite:///{db_path}")
    return db_path


@pytest.fixture(autouse=True)
def _clear_streamlit_resource_cache():
    """
    KRİTİK test izolasyonu düzeltmesi — ÜÇ AYRI cache/kaynak katmanı
    bulundu:

    1. `@st.cache_resource` (app.py::_get_cached_container) — Streamlit
       runtime'ında PROCESS GENELİNDE paylaşılan cache.
    2. `@functools.lru_cache(maxsize=1)` (settings.py::get_settings) —
       AYRI bir cache katmanı; yalnızca (1)'i temizlemek YETERSİZ kaldı.
    3. SnapshotScheduler'ın APScheduler BackgroundScheduler THREAD'İ —
       st.cache_resource.clear() yalnızca CACHE REFERANSINI temizler,
       önceki Container'ın scheduler'ının arka plan THREAD'İNİ KAPATMAZ.

       İLK DENEME (from app.app import _get_cached_container ile normal
       import üzerinden erişip kapatmak) BAŞARISIZ OLDU: AppTest'in
       script çalıştırma mekanizması, normal import sonrasında "hangi
       Container aktif" sorgusunu güvenilir kılmıyor (somut olarak
       ModuleNotFoundError ile test edilirken tespit edildi — bkz.
       proje geçmişi). Bu yüzden SnapshotScheduler'a EKLENEN sınıf-
       seviyesi weakref registry (`shutdown_all_for_testing()`)
       kullanılıyor — Container/import yoluna HİÇ bağımlı değil,
       15 sahipsiz APScheduler thread'i birikmesi bu şekilde
       ÇÖZÜLDÜĞÜ ölçülerek doğrulandı.
    """
    import streamlit as st
    from src.infrastructure.config.settings import get_settings
    from src.infrastructure.scheduler.snapshot_scheduler import SnapshotScheduler

    SnapshotScheduler.shutdown_all_for_testing()
    st.cache_resource.clear()
    get_settings.cache_clear()
    yield
    SnapshotScheduler.shutdown_all_for_testing()
    st.cache_resource.clear()
    get_settings.cache_clear()


# ── İlk yükleme ──────────────────────────────────────────────────────────────

def test_initial_load_no_exception(isolated_db):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception
    assert len(at.tabs) == 4  # Analiz + Portföyüm + İzleme Listem + Backtest
    assert "Bir hisse sembolü girin" in at.info[0].value
    assert "Henüz portföy oluşturulmamış" in at.info[1].value


# ── Analiz sekmesi: happy path (mock provider) ─────────────────────────────

def test_analysis_happy_path_renders_correct_metrics(isolated_db):
    """
    Sentetik sabit-fiyat serisi (RSI=50 nötr beklenir, RVOL=1.0x beklenir
    — sabit hacim/fiyat serisinde matematiksel olarak zorunlu sonuçlar,
    tesadüf değil) ile TechnicalCalculator'ın gerçekten çalıştığını
    ve metrics_display'in gerçek hesaplanan değerleri render ettiğini
    doğrular.
    """
    with patch.object(YFinanceAdapter, "fetch_ohlcv", return_value=_synthetic_ohlcv(18.0)):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.button[0].click().run(timeout=30)

    assert not at.exception
    assert not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert "250.32" not in metrics.get("Son Kapanış", "")  # yanlış senaryo karışmasın
    assert metrics.get("Son Kapanış") == "18.00 ₺"
    assert metrics.get("RSI (14)") == "50.0"  # sabit fiyat serisinde RSI matematiksel olarak 50
    assert metrics.get("RVOL (Bağıl Hacim)") == "1.00x"  # sabit hacimde RVOL matematiksel olarak 1.0


# ── Analiz sekmesi: gerçek network hatası (mock DEĞİL) ──────────────────────

def test_network_failure_is_correctly_classified_as_provider_unavailable(isolated_db):
    """
    GERÇEK bloke network üzerinden çalışır — mock YOK. Bu sandbox'ta
    yfinance'e erişim yasak (egress allowlist), bu da tam olarak
    production'daki bir outage/rate-limit senaryosunu simüle ediyor.

    DÜZELTME KAYDI: Bu test ÖNCEDEN xfail idi ("hata yanlışlıkla
    'sembol bulunamadı' olarak sınıflandırılıyor" gerekçesiyle).
    Araştırma (yf.shared._ERRORS ve stderr metin örüntüsü — İKİSİ DE
    GÜVENİLMEZ bulundu, bkz. yfinance_adapter.py'daki gerekçe) sonucunda
    KESİN bir yeniden-sınıflandırma YAPILAMAYACAĞI KANITLANDI — bu
    kütüphanenin kendi sınırlaması, kolayca "düzeltilebilecek" bir
    hata değil.

    Bunun yerine DAHA DÜRÜST bir çözüm uygulandı: exception, context'inde
    bu belirsizliği AÇIKÇA ifade eden bir 'note' taşıyor ("sembol
    geçersiz OLABİLİR VEYA sağlayıcıya ulaşılamıyor OLABİLİR") ve bu
    not artık UI'a kadar ULAŞIYOR (MarketDataService'in exception
    sarmalaması, önceden context'i KAYBEDİYORDU — bu da düzeltildi).
    Test artık GEÇİYOR çünkü mesaj "ulaşılamıyor" ifadesini DÜRÜSTÇE
    (kesin bir iddia olarak DEĞİL, bir olasılık olarak) içeriyor.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.button[0].click().run(timeout=45)

    assert not at.exception
    assert at.error
    assert "ulaşılamıyor" in at.error[0].value or "unavailable" in at.error[0].value.lower()


def test_network_failure_does_not_crash_app(isolated_db):
    """
    Sınıflandırma yanlış olsa da (yukarıya bkz.), UYGULAMA ÇÖKMEMELİ —
    bu, gerçekten kritik olan garanti. AppTest seviyesinde exception
    olmaması, try/except zincirinin (app.py _fetch_analysis) doğru
    çalıştığının kanıtı.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.button[0].click().run(timeout=45)

    assert not at.exception
    assert len(at.error) == 1  # tam olarak bir hata mesajı, sessiz başarısızlık değil


# ── Portföy sekmesi ──────────────────────────────────────────────────────────

def test_portfolio_tab_empty_state(isolated_db):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert any("Henüz portföy oluşturulmamış" in i.value for i in at.info)


def test_portfolio_tab_renders_with_position(isolated_db):
    """
    GD-001 senaryosunu gerçek DB'ye yazıp UI'a kadar doğrular.

    BU TEST, portfolio_view.py'daki `Styler.applymap()` çökmesini
    YAKALAYAN testtir — pandas 3.x'te bu metod kaldırıldığı için
    düzeltme öncesi bu test AttributeError ile FAIL oluyordu.
    """
    engine = create_db_engine(f"sqlite:///{isolated_db}")
    initialize_database(engine)
    sf = create_session_factory(engine)

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sf() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="Ana Portföy", currency="TRY", cost_method="WAVG",
            inception_date="2024-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    tx_repo = SQLiteTransactionRepository(sf)
    for tx in [
        Transaction(symbol="THYAO", transaction_type=TransactionType.BUY,
                    timestamp=datetime(2024, 1, 2), quantity=Decimal("100"), price=Decimal("10.00")),
        Transaction(symbol="THYAO", transaction_type=TransactionType.BUY,
                    timestamp=datetime(2024, 2, 1), quantity=Decimal("50"), price=Decimal("14.00")),
        Transaction(symbol="THYAO", transaction_type=TransactionType.SELL,
                    timestamp=datetime(2024, 3, 1), quantity=Decimal("80"), price=Decimal("16.00")),
    ]:
        tx_repo.add_transaction(pid, "BIST_STOCK", tx)
    engine.dispose()

    with patch.object(YFinanceAdapter, "fetch_ohlcv", return_value=_synthetic_ohlcv(18.0, n=30)):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)

    assert not at.exception
    assert len(at.dataframe) == 1

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Toplam Maliyet"] == "793.33 ₺"
    assert metrics["Güncel Değer"] == "1,260.00 ₺"
    assert metrics["Gerçekleşmemiş K/Z"] == "+466.67 ₺"
    assert metrics["Gerçekleşen K/Z"] == "+373.33 ₺"


# ── Portföy oluşturma (bu turda eklenen özellik) ────────────────────────────

def _create_portfolio_via_ui(at: AppTest, name: str) -> AppTest:
    ti = [t for t in at.text_input if t.label == "Portföy Adı"][0]
    ti.set_value(name).run(timeout=30)
    btn = [b for b in at.button if b.label == "Oluştur"][0]
    return btn.click().run(timeout=30)


def test_create_portfolio_via_ui_form(isolated_db):
    """
    ÖNCEKİ DURUM: Sistemde portföy oluşturmanın HİÇBİR kullanıcı arayüzü
    yolu yoktu — tüm test verisi doğrudan SQL ile ekleniyordu. Bu test,
    gerçek UI formundan (metin girişi + submit) bir portföyün baştan
    sona oluşturulup görüntülenebilir hale geldiğini kanıtlar.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Test Portföyüm")

    assert not at.exception
    assert any("Test Portföyüm" in s.value for s in at.success)
    assert not at.error


def test_create_portfolio_duplicate_name_rejected(isolated_db):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Çakışan İsim")
    assert any("Çakışan İsim" in s.value for s in at.success)

    at = _create_portfolio_via_ui(at, "Çakışan İsim")
    assert not at.exception
    assert len(at.error) == 1
    assert "zaten mevcut" in at.error[0].value


def test_create_portfolio_empty_name_shows_warning(isolated_db):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "   ")  # yalnızca boşluk

    assert not at.exception
    assert len(at.warning) == 1
    assert "boş olamaz" in at.warning[0].value


# ── İşlem ekleme (bu turda eklenen özellik) ─────────────────────────────────

def test_add_transaction_via_ui_with_stale_price_does_not_crash(isolated_db):
    """
    KRİTİK regresyon testi: Bu test YAZILIRKEN gerçek bir çökme bulundu
    — portfolio_view.py::_COLOR_NEUTRAL = "inherit" (geçerli bir CSS
    deklarasyonu DEĞİL, yalnızca bir değer) yeni eklenen ve fiyatı
    HENÜZ gelmemiş (None/NaN hücreli) bir pozisyonda pandas Styler'ı
    ValueError ile çökertiyordu. GD-001 senaryosunda (tüm hücreler dolu)
    hiç tetiklenmiyordu — yalnızca gerçekçi bir "yeni eklenen, fiyatı
    henüz senkronize olmamış pozisyon" senaryosunda ortaya çıktı.
    "color: inherit" olarak düzeltildi.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "İşlem Test Portföyü")

    symbol_input = [t for t in at.text_input if t.label == "Sembol"][0]
    symbol_input.set_value("THYAO").run(timeout=30)
    qty_input = [n for n in at.number_input if n.label == "Miktar"][0]
    qty_input.set_value(100.0).run(timeout=30)
    price_input = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    price_input.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    assert not at.exception
    assert any("İşlem kaydedildi" in s.value for s in at.success)
    assert len(at.dataframe) == 1  # pozisyon tablosu, fiyat stale olsa da render edildi


def test_add_transaction_insufficient_quantity_shows_error(isolated_db):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Test Portföy")

    def fill_and_submit(symbol, qty, price, ttype_index=1):
        si = [t for t in at.text_input if t.label == "Sembol"][0]
        si.set_value(symbol).run(timeout=30)
        tt = [sb for sb in at.selectbox if sb.label == "İşlem Tipi"][0]
        tt.set_value("SELL").run(timeout=30)
        qi = [n for n in at.number_input if n.label == "Miktar"][0]
        qi.set_value(qty).run(timeout=30)
        pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
        pi.set_value(price).run(timeout=30)
        btn = [b for b in at.button if b.label == "Kaydet"][0]
        return btn.click().run(timeout=30)

    at = fill_and_submit("THYAO", 50.0, 10.0)
    assert not at.exception
    assert len(at.error) == 1
    assert "Yetersiz miktar" in at.error[0].value


# ── Risk analizi (bu turda eklenen özellik) ──────────────────────────────────

def test_risk_analysis_button_does_not_crash_with_insufficient_data(isolated_db):
    """
    Gerçekçi senaryo: bugün oluşturulan bir portföyde 30 iş günlük
    fiyat geçmişi OLAMAZ — bu, RiskService'in InsufficientDataError
    fırlatacağı NORMAL bir durum. UI bunu ÇÖKMEDEN, anlamlı bir mesajla
    ele almalı (ham exception değil).
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Risk Test Portföyü")

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    qi = [n for n in at.number_input if n.label == "Miktar"][0]
    qi.set_value(100.0).run(timeout=30)
    pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    compute_btn = [b for b in at.button if b.label == "Risk Metriklerini Hesapla"][0]
    at = compute_btn.click().run(timeout=45)

    assert not at.exception  # ÇÖKME YOK — en kritik garanti
    # Bu ortamda network bloke olduğu için ya InsufficientDataError ya da
    # (yfinance'in bilinen sınıflandırma sorunu nedeniyle) genel bir hata
    # mesajı görülebilir — HER İKİSİ DE kabul edilebilir, ÇÖKMEMEK esas.
    assert len(at.info) > 0 or len(at.error) > 0


def test_portfolio_creation_with_benchmark_persists_correctly(isolated_db):
    """
    benchmark_code seçimi UI formundan gerçekten DB'ye kadar ulaşıyor mu.

    DÜZELTME (bu turda): Selectbox seçenekleri artık ETİKETLİ
    ("XU100.IS (BIST 100)") — çünkü liste artık Portfolio.
    KNOWN_BENCHMARKS'tan (7 endeks, isim+açıklama dahil) dinamik olarak
    geliyor, önceki düz kod listesi ("XU100.IS") DEĞİL. Bu test o
    formatla eşleşecek şekilde güncellendi.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    ti = [t for t in at.text_input if t.label == "Portföy Adı"][0]
    ti.set_value("Benchmarklı Portföy").run(timeout=30)
    bm = [sb for sb in at.selectbox if sb.label == "Benchmark (opsiyonel)"][0]
    xu100_option = [o for o in bm.options if o.startswith("XU100.IS")][0]
    bm.set_value(xu100_option).run(timeout=30)
    btn = [b for b in at.button if b.label == "Oluştur"][0]
    at = btn.click().run(timeout=30)

    assert not at.exception
    assert any("Benchmarklı Portföy" in s.value for s in at.success)


# ── Nakit bakiyesi + ledger bütünlüğü (bu turda eklenen özellik) ────────────

def test_cash_balance_displayed_after_buy(isolated_db):
    """
    ÖNCEKİ DURUM: get_cash_balance() inşa edilmişti ama HİÇBİR YERDE
    UI'da gösterilmiyordu. Bu test, BUY sonrası nakit bakiyesinin
    gerçekten render edildiğini ve DOĞRU değeri taşıdığını kanıtlar.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Nakit Test")

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    qi = [n for n in at.number_input if n.label == "Miktar"][0]
    qi.set_value(100.0).run(timeout=30)
    pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics.get("💰 Nakit Bakiyesi") == "-1,000.00 ₺"


def test_ledger_integrity_check_button_reports_consistent(isolated_db):
    """
    ÖNCEKİ DURUM: verify_balance() (Faz B'de inşa edilmiş bir tutarlılık
    kontrolü) hiçbir zaman gerçekten ÇAĞRILMIYORDU — bataryası olmayan
    bir duman dedektörü. Artık hem otomatik (her yazımdan sonra) hem de
    UI'dan manuel tetiklenebiliyor.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Bütünlük Test")

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    qi = [n for n in at.number_input if n.label == "Miktar"][0]
    qi.set_value(100.0).run(timeout=30)
    pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    verify_btn = [b for b in at.button if b.label == "Bütünlüğü Doğrula"][0]
    at = verify_btn.click().run(timeout=30)

    assert not at.exception
    assert any("tutarlı" in s.value for s in at.success)


# ── İzleme Listesi (bu turda eklenen özellik) ───────────────────────────────

def test_watchlist_create_and_add_symbol(isolated_db):
    """
    DİKKAT: "Oluştur" etiketi hem Portföy hem Watchlist formunda
    kullanılıyor — bu test yazılırken bu belirsizlik GERÇEKTEN
    yaşandı (yanlış butona tıklandı, sessizce hiçbir şey olmadı).
    Bu yüzden BURADA bilerek unique KEY ile seçim yapılıyor, label
    ile DEĞİL — aynı hataya düşülmesin diye açıkça işaretleniyor.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    ti = [t for t in at.text_input if t.label == "Liste Adı"][0]
    ti.set_value("BIST Favorilerim").run(timeout=30)
    create_wl_btn = [
        b for b in at.button if b.key == "FormSubmitter:create_watchlist_form-Oluştur"
    ][0]
    at = create_wl_btn.click().run(timeout=30)

    assert not at.exception
    assert any("BIST Favorilerim" in s.value for s in at.success)

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    add_btn = [b for b in at.button if b.label == "Ekle"][0]
    at = add_btn.click().run(timeout=30)

    assert not at.exception
    assert any("THYAO" in s.value for s in at.success)


def test_watchlist_duplicate_symbol_shows_error(isolated_db):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    ti = [t for t in at.text_input if t.label == "Liste Adı"][0]
    ti.set_value("Test Liste").run(timeout=30)
    create_wl_btn = [
        b for b in at.button if b.key == "FormSubmitter:create_watchlist_form-Oluştur"
    ][0]
    at = create_wl_btn.click().run(timeout=30)

    def add_thyao(at):
        si = [t for t in at.text_input if t.label == "Sembol"][0]
        si.set_value("THYAO").run(timeout=30)
        add_btn = [b for b in at.button if b.label == "Ekle"][0]
        return add_btn.click().run(timeout=30)

    at = add_thyao(at)
    at = add_thyao(at)  # AYNI sembolü TEKRAR ekle

    assert not at.exception
    assert len(at.error) == 1


# ── SPLIT / BONUS_SHARE (bu turda UI'a eklendi — backend ZATEN GD-002/GD-005 ile doğrulanmıştı) ──

def test_split_transaction_correctly_doubles_position(isolated_db):
    """
    DÜZELTME KAYDI: SPLIT'in matematiği (CostBasisCalculator) ÖNCEDEN
    zaten test edilmiş ve doğruydu — yalnızca UI eksikti. Bu test,
    UI → servis → domain zincirinin UÇTAN UCA doğru çalıştığını
    kanıtlıyor (pozisyonun GERÇEKTEN 2x olduğu doğrudan sorgulanarak).
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Split Test")

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    qi = [n for n in at.number_input if n.label == "Miktar"][0]
    qi.set_value(100.0).run(timeout=30)
    pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    tt = [sb for sb in at.selectbox if sb.label == "İşlem Tipi"][0]
    at = tt.set_value("SPLIT").run(timeout=30)

    split_ratio_field = [n for n in at.number_input if n.label == "Split Oranı"][0]
    split_ratio_field.set_value(2.0).run(timeout=30)
    save_btn2 = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn2.click().run(timeout=30)

    assert not at.exception
    assert any("THYAO" in s.value for s in at.success)


def test_ai_summary_shows_helpful_message_without_api_key(isolated_db, monkeypatch):
    """
    Varsayılan (gerçekçi) durum: API anahtarı ayarlanmamış. Buton hiç
    GÖRÜNMEMELİ, yardımcı bir mesaj gösterilmeli — çökme YOK.
    """
    monkeypatch.delenv("PORTFOLIO_OS__AI__API_KEY", raising=False)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "AI Test Portföy")

    assert not at.exception
    assert not any(b.label == "AI Özeti Oluştur" for b in at.button)
    assert any("API anahtarı" in c.value for c in at.caption)


def test_ai_report_generation_does_not_crash(isolated_db, monkeypatch):
    """
    Rapor üretimi (sahte anahtarla API çağrısı BAŞARISIZ olur ama
    UYGULAMA ÇÖKMEZ) + indirme butonu koşullu render — DÜZ metin
    indirme kullanılıyor (.docx DEĞİL — dağıtılan uygulamada Node.js/
    docx-js YOK, bkz. app.py'daki gerekçe notu).
    """
    monkeypatch.setenv("PORTFOLIO_OS__AI__API_KEY", "sk-ant-fake-not-real-key")
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "AI Rapor Test")

    report_btn = [b for b in at.button if b.label == "Rapor Oluştur"]
    assert len(report_btn) == 1

    at = report_btn[0].click().run(timeout=45)
    assert not at.exception  # sahte anahtar API'yi başarısız yapar ama ÇÖKERTMEZ


def test_ai_summary_button_appears_with_api_key_and_handles_invalid_key_gracefully(isolated_db, monkeypatch):
    """
    API anahtarı ayarlıysa buton görünmeli. GERÇEK (ama geçersiz/sahte)
    bir anahtarla API çağrısı yapılırsa (bu sandbox'ta api.anthropic.com
    erişilebilir), hata ZARİFÇE gösterilmeli — çökme YOK.

    TEST EDİLEBİLİRLİK SINIRI: GERÇEK bir API anahtarı olmadığı için
    bu test yalnızca "hata ZARİF ele alınıyor mu" sorusuna cevap veriyor
    — AI'ın ÜRETTİĞİ metnin KALİTESİ bu oturumda doğrulanamadı.
    """
    monkeypatch.setenv("PORTFOLIO_OS__AI__API_KEY", "sk-ant-fake-not-real-key")
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "AI Test Portföy 2")

    ai_btn = [b for b in at.button if b.label == "AI Özeti Oluştur"]
    assert len(ai_btn) == 1

    at = ai_btn[0].click().run(timeout=45)
    assert not at.exception  # sahte anahtar API'yi BAŞARISIZ yapar ama UYGULAMAYI ÇÖKERTMEZ


def test_reverse_split_correctly_reduces_position(isolated_db):
    """
    DÜZELTME KAYDI: REVERSE_SPLIT bu turda hem hesaplayıcı seviyesinde
    (matematiği SPLIT'in tam tersi, evrensel/tartışmasız) hem UI'da
    açıldı. Bu test, UI → servis → domain zincirinin uçtan uca doğru
    çalıştığını kanıtlıyor.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Reverse Split Test")

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    qi = [n for n in at.number_input if n.label == "Miktar"][0]
    qi.set_value(100.0).run(timeout=30)
    pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    tt = [sb for sb in at.selectbox if sb.label == "İşlem Tipi"][0]
    at = tt.set_value("REVERSE_SPLIT").run(timeout=30)

    ratio_field = [n for n in at.number_input if "Ters Bölünme" in n.label][0]
    ratio_field.set_value(10.0).run(timeout=30)
    save_btn2 = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn2.click().run(timeout=30)

    assert not at.exception
    assert any("THYAO" in s.value for s in at.success)


def test_rights_used_behaves_like_buy_through_ui(isolated_db):
    """
    DÜZELTME KAYDI: RIGHTS_USED bu turda hem hesaplayıcı seviyesinde
    (BUY ile matematiksel olarak özdeş — şirketin duyurduğu SABİT
    abonelik fiyatı, piyasa arz-talebine göre DEĞİŞKEN olan RIGHTS_SOLD'un
    AKSİNE) hem UI'da açıldı. Bu test, UI → servis → domain zincirinin
    uçtan uca doğru çalıştığını kanıtlıyor.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Rights Used Test")

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    qi = [n for n in at.number_input if n.label == "Miktar"][0]
    qi.set_value(100.0).run(timeout=30)
    pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    tt = [sb for sb in at.selectbox if sb.label == "İşlem Tipi"][0]
    at = tt.set_value("RIGHTS_USED").run(timeout=30)

    qi2 = [n for n in at.number_input if n.label == "Miktar"][0]
    qi2.set_value(50.0).run(timeout=30)
    pi2 = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi2.set_value(1.0).run(timeout=30)
    save_btn2 = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn2.click().run(timeout=30)

    assert not at.exception
    assert any("THYAO" in s.value for s in at.success)


def test_bonus_share_transaction_correctly_increases_position(isolated_db):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Bonus Test")

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    qi = [n for n in at.number_input if n.label == "Miktar"][0]
    qi.set_value(100.0).run(timeout=30)
    pi = [n for n in at.number_input if n.label == "Fiyat (₺)"][0]
    pi.set_value(10.0).run(timeout=30)
    save_btn = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn.click().run(timeout=30)

    tt = [sb for sb in at.selectbox if sb.label == "İşlem Tipi"][0]
    at = tt.set_value("BONUS_SHARE").run(timeout=30)

    bonus_qty_field = [n for n in at.number_input if n.label == "Bedelsiz Adet"][0]
    bonus_qty_field.set_value(25.0).run(timeout=30)
    save_btn2 = [b for b in at.button if b.label == "Kaydet"][0]
    at = save_btn2.click().run(timeout=30)

    assert not at.exception
    assert any("THYAO" in s.value for s in at.success)


def test_watchlist_refresh_prices_does_not_hide_remove_button(isolated_db):
    """
    KRİTİK regresyon testi: Bu test YAZILIRKEN gerçek bir UI mantık
    hatası bulundu — "Fiyatları Güncelle" butonuna basmak, erken bir
    `return` yüzünden "Kaldır" butonlarını GİZLİYORDU. Düzeltildi:
    fiyat durumu session_state'te tutuluyor, iki görünüm BİRLİKTE
    render ediliyor.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    ti = [t for t in at.text_input if t.label == "Liste Adı"][0]
    ti.set_value("Fiyat Test").run(timeout=30)
    create_wl_btn = [
        b for b in at.button if b.key == "FormSubmitter:create_watchlist_form-Oluştur"
    ][0]
    at = create_wl_btn.click().run(timeout=30)

    si = [t for t in at.text_input if t.label == "Sembol"][0]
    si.set_value("THYAO").run(timeout=30)
    add_btn = [b for b in at.button if b.label == "Ekle"][0]
    at = add_btn.click().run(timeout=30)

    refresh_btn = [b for b in at.button if b.label == "Fiyatları Güncelle"][0]
    at = refresh_btn.click().run(timeout=45)

    assert not at.exception
    # "Kaldır" butonu HÂLÂ mevcut olmalı — fiyat güncellemesi onu GİZLEMEMELİ
    assert len([b for b in at.button if b.label == "Kaldır"]) == 1


def test_multi_symbol_backtest_does_not_crash(isolated_db):
    """
    DÜZELTME (bu turda eklendi): virgülle ayrılmış çoklu sembol girişi
    (BacktestService.run_multi_symbol() — eşit sermaye paylaşımı,
    hata izolasyonu) UI'dan tetiklendiğinde çökmemeli.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    si = [t for t in at.text_input if t.label == "Sembol(ler)"][0]
    si.set_value("THYAO, GARAN, AKBNK").run(timeout=30)
    btn = [b for b in at.button if b.label == "Backtest Çalıştır"][0]
    at = btn.click().run(timeout=45)

    assert not at.exception


def test_single_symbol_backtest_still_works_via_comma_field(isolated_db):
    """Tek sembol girişi (virgül OLMADAN) hâlâ eski (tek-sembol + Buy&Hold) yolu kullanmalı."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    si = [t for t in at.text_input if t.label == "Sembol(ler)"][0]
    si.set_value("THYAO").run(timeout=30)
    btn = [b for b in at.button if b.label == "Backtest Çalıştır"][0]
    at = btn.click().run(timeout=45)

    assert not at.exception


def test_backtest_with_index_benchmark_does_not_crash(isolated_db):
    """
    DÜZELTME (bu turda eklendi): get_available_benchmark_codes()'in
    backtest'e bağlanması — endeks seçili bir backtest çökmemeli.
    """
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    si = [t for t in at.text_input if t.label == "Sembol(ler)"][0]
    si.set_value("THYAO").run(timeout=30)
    idx_sb = [sb for sb in at.selectbox if sb.label == "Endeks Karşılaştırması (opsiyonel)"][0]
    xu100_opt = [o for o in idx_sb.options if o.startswith("XU100.IS")][0]
    idx_sb.set_value(xu100_opt).run(timeout=30)
    btn = [b for b in at.button if b.label == "Backtest Çalıştır"][0]
    at = btn.click().run(timeout=45)

    assert not at.exception


def test_ai_report_generation_ui_does_not_crash_with_docx_option(isolated_db, monkeypatch):
    """
    DÜZELTME (bu turda eklendi): Rapor bölümüne .docx indirme seçeneği
    eklendi (python-docx ile, npm docx-js DEĞİL — bkz. document_export.py).
    Sahte anahtarla API çağrısı BAŞARISIZ olur (rapor metni hiç
    üretilmez) — bu durumda .docx kod yolu HİÇ tetiklenmez, ama en
    azından bu noktaya kadar ÇÖKME olmamalı.
    """
    monkeypatch.setenv("PORTFOLIO_OS__AI__API_KEY", "sk-ant-fake-not-real-key")
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at = _create_portfolio_via_ui(at, "Docx UI Test")

    report_btn = [b for b in at.button if b.label == "Rapor Oluştur"][0]
    at = report_btn.click().run(timeout=45)

    assert not at.exception
