"""
RiskService testleri — GERÇEK repository/provider yok (network bloke,
bkz. bu projede daha önce kanıtlanan sandbox kısıtlaması). Bunun yerine
gerçek SQLite repository'leri (transaction_repo, cash_ledger_repo —
GERÇEK, mock değil) + sözleşmeye uygun sahte bir MarketDataProvider
(yalnızca fetch_ohlcv'yi sentetik veriyle dolduran) kullanılıyor.

Bu, "hangi kısmın gerçek hangi kısmın sahte olduğu" konusunda tam
şeffaflık sağlıyor: DB katmanı ve pozisyon zaman serisi inşası %100
gerçek kod; yalnızca dış ağ sınırı (yfinance/TEFAS) sahte.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import InsufficientDataError
from src.domain.models.transaction import Transaction
from src.infrastructure.database.connection import (
    create_db_engine,
    create_session_factory,
    initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.repositories.sqlite.cash_ledger_repository import (
    SQLiteCashLedgerRepository,
)
from src.infrastructure.repositories.sqlite.transaction_repository import (
    SQLiteTransactionRepository,
)
from src.infrastructure.repositories.sqlite.price_repository import SQLitePriceRepository
from src.services.price_sync_service import PriceSyncService
from src.services.risk_service import RiskService

pytestmark = pytest.mark.integration


class FakeConstantProvider:
    """
    Sözleşmeye uygun (MarketDataProvider arayüzü) ama SABİT/DETERMİNİSTİK
    fiyat serisi döndüren sahte sağlayıcı. Gerçek network YOK.
    """

    def __init__(self, price_map: dict[str, pd.DataFrame]) -> None:
        self._price_map = price_map

    def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
        df = self._price_map.get(symbol, pd.DataFrame())
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        if start_date is not None:
            mask &= df.index >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= df.index <= pd.Timestamp(end_date)
        return df[mask]

    def get_provider_name(self) -> str:
        return "mock"


def _make_price_series(prices: list[float], n_days: int = 120) -> pd.DataFrame:
    """n_days'lık bir tarih aralığında verilen fiyat dizisini tekrarlayarak/genişleterek DataFrame üretir."""
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="D").normalize()
    if len(prices) < n_days:
        # Son değeri tekrarlayarak uzat (sabit devam)
        prices = prices + [prices[-1]] * (n_days - len(prices))
    prices = prices[-n_days:]
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices,
        "Volume": [1_000_000] * n_days,
    }, index=dates)


@pytest.fixture()
def db_session_factory(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'risk_test.db'}")
    initialize_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture()
def portfolio_with_single_holding(db_session_factory):
    """
    THYAO'da GERÇEKÇİ BİR pozisyon açan, GERÇEK DB'ye yazılmış bir portföy.
    Fiyat serisi, benchmark ile TAM AYNI olacak şekilde kurulacak
    (test senaryosuna göre) — bkz. ilgili testler.
    """
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with db_session_factory() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="Risk Test Portföyü", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    tx_repo = SQLiteTransactionRepository(db_session_factory)
    tx_repo.add_transaction(pid, "BIST_STOCK", Transaction(
        symbol="THYAO", transaction_type=TransactionType.BUY,
        timestamp=datetime.today() - timedelta(days=200),
        quantity=Decimal("100"), price=Decimal("100.00"),
    ))
    return pid, tx_repo, db_session_factory


# ── compute_risk_profile ─────────────────────────────────────────────────────

def test_compute_risk_profile_with_random_walk_prices(portfolio_with_single_holding):
    pid, tx_repo, sf = portfolio_with_single_holding
    cash_repo = SQLiteCashLedgerRepository(sf)

    np.random.seed(7)
    prices = list(100 + np.cumsum(np.random.normal(0.1, 2, 150)))
    provider = FakeConstantProvider({"THYAO": _make_price_series(prices, n_days=150)})

    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo,
        price_sync_service=PriceSyncService(SQLitePriceRepository(sf), provider), risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    profile = service.compute_risk_profile(pid, lookback_days=100, include_cash=False)

    assert profile.portfolio_id == pid
    assert profile.data_points_used > 0
    assert profile.annualized_volatility > 0
    assert isinstance(profile.sharpe_ratio, float)
    assert profile.max_drawdown.max_drawdown <= 0  # drawdown her zaman <=0


def test_compute_risk_profile_no_holdings_raises(db_session_factory):
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with db_session_factory() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="Boş Portföy", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    tx_repo = SQLiteTransactionRepository(sf := db_session_factory)
    cash_repo = SQLiteCashLedgerRepository(sf)
    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo,
        price_sync_service=PriceSyncService(SQLitePriceRepository(sf), FakeConstantProvider({})), risk_calculator=RiskCalculator(),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    with pytest.raises(InsufficientDataError):
        service.compute_risk_profile(pid)


# ── calculate_relative_performance ──────────────────────────────────────────

def test_relative_performance_identical_to_benchmark_gives_beta_one(portfolio_with_single_holding):
    """
    KRİTİK doğrulama: Portföyün TEK holding'i THYAO ve THYAO'nun fiyat
    serisi benchmark ile NEREDEYSE AYNI (çok küçük bağımsız gürültü
    eklenmiş — TAM AYNI değil, çünkü tracking_error=0 durumu
    RiskCalculator'da BİLİNÇLİ olarak CalculationError fırlatıyor,
    bkz. bu oturumda daha önce eklenen epsilon koruması) ise
    beta≈1.0, r_squared≈1.0, alpha≈0 olmalı.
    """
    pid, tx_repo, sf = portfolio_with_single_holding
    cash_repo = SQLiteCashLedgerRepository(sf)

    np.random.seed(3)
    base_prices = list(100 + np.cumsum(np.random.normal(0.05, 1.5, 150)))
    thyao_series = _make_price_series(base_prices, n_days=150)
    # Benchmark, THYAO'nun NEREDEYSE aynısı ama sıfırdan farklı (çok küçük)
    # bağımsız gürültü ile — tracking_error>0 garantili, gerçekçi bir
    # "yüksek korelasyonlu ama özdeş olmayan" senaryo.
    np.random.seed(99)
    tiny_noise = np.random.normal(0, 0.01, 150)
    benchmark_prices = [p + n for p, n in zip(base_prices, tiny_noise)]
    benchmark_series = _make_price_series(benchmark_prices, n_days=150)

    provider = FakeConstantProvider({"THYAO": thyao_series, "XU100.IS": benchmark_series})

    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo,
        price_sync_service=PriceSyncService(SQLitePriceRepository(sf), provider), risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    perf = service.calculate_relative_performance(
        pid, "XU100.IS", lookback_days=100, include_cash=False,
    )

    assert abs(perf.beta - 1.0) < 0.05
    assert abs(perf.r_squared - 1.0) < 0.01
    assert abs(perf.alpha_daily) < 0.001
    assert 0 < perf.tracking_error < 0.001  # küçük ama sıfır DEĞİL
    assert abs(perf.up_capture - 1.0) < 0.05
    assert abs(perf.down_capture - 1.0) < 0.05


def test_relative_performance_missing_benchmark_raises(portfolio_with_single_holding):
    pid, tx_repo, sf = portfolio_with_single_holding
    cash_repo = SQLiteCashLedgerRepository(sf)
    provider = FakeConstantProvider({"THYAO": _make_price_series([100.0] * 150, n_days=150)})
    # "XU100.IS" provider'da YOK

    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo,
        price_sync_service=PriceSyncService(SQLitePriceRepository(sf), provider), risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )
    with pytest.raises(InsufficientDataError):
        service.calculate_relative_performance(pid, "XU100.IS", lookback_days=100)


# ── Cache davranışı ──────────────────────────────────────────────────────────

def test_cache_avoids_recomputation(portfolio_with_single_holding):
    from src.infrastructure.cache.ttl_cache import TTLCache

    pid, tx_repo, sf = portfolio_with_single_holding
    cash_repo = SQLiteCashLedgerRepository(sf)

    call_count = {"n": 0}

    class CountingProvider(FakeConstantProvider):
        def fetch_ohlcv(self, *args, **kwargs):
            call_count["n"] += 1
            return super().fetch_ohlcv(*args, **kwargs)

    provider = CountingProvider({"THYAO": _make_price_series(
        list(100 + np.cumsum(np.random.default_rng(5).normal(0.05, 1.5, 150))), n_days=150
    )})
    cache = TTLCache(ttl_seconds=3600)
    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo,
        price_sync_service=PriceSyncService(SQLitePriceRepository(sf), provider), risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45, cache=cache,
    )

    service.compute_risk_profile(pid, lookback_days=100, include_cash=False)
    first_call_count = call_count["n"]
    service.compute_risk_profile(pid, lookback_days=100, include_cash=False)

    assert call_count["n"] == first_call_count  # ikinci çağrı provider'a HİÇ dokunmadı


def test_multi_symbol_fetch_is_actually_parallel(tmp_path):
    """
    KRİTİK doğrulama: ThreadPoolExecutor'ın yalnızca DOĞRU sonuç
    üretmediğini, GERÇEKTEN paralel çalıştığını kanıtlar. Yapay bir
    gecikme (0.2s) taşıyan sahte bir provider ile: 5 sembol SIRALI
    işlenseydi ~1.0s sürerdi, PARALEL işlenirse ~0.2-0.3s sürmeli.
    """
    import time
    import threading

    engine = create_db_engine(f"sqlite:///{tmp_path / 'parallel_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sf() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="Parallel Test", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    tx_repo = SQLiteTransactionRepository(sf)
    symbols = ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"]
    for sym in symbols:
        tx_repo.add_transaction(pid, "BIST_STOCK", Transaction(
            symbol=sym, transaction_type=TransactionType.BUY,
            timestamp=datetime.today() - timedelta(days=200),
            quantity=Decimal("10"), price=Decimal("10.00"),
        ))

    # DÜZELTME (bu turda bulundu): Bu testin İLK VERSİYONU toplam
    # end-to-end süreyi ölçüyordu — ama bu, İKİ FARKLI endişeyi
    # (fetch paralelliği vs write throughput) karıştırıyordu. Gerçek
    # ölçümle bulundu: write_batch() (750 kayıt, TEK thread'de,
    # BİLİNÇLİ olarak serileştirilmiş — bkz. risk_service.py'daki
    # "database is locked" gerekçesi) ~0.76s sürüyor — bu bir HATA
    # DEĞİL, nadir bir "soğuk başlangıç" (5 sembol × 150 günlük TAM
    # backfill) senaryosunun kabul edilebilir maliyeti. Gerçek dünyada
    # incremental sync (~1-5 gün/sembol) çok daha hızlı olur.
    # Bu yüzden test, FETCH AŞAMASININ gerçekten paralel olduğunu
    # (start/end zaman damgalarının ÖRTÜŞTÜĞÜNÜ) doğrudan ölçüyor —
    # toplam süreye değil, fetch'in ÖRTÜŞMESİNE bakıyor.
    call_timestamps: list[tuple[float, float]] = []
    lock = threading.Lock()

    class SlowProvider:
        """Her çağrıda 0.2s gecikme ekler VE başlangıç/bitiş zamanını kaydeder."""
        def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
            t_start = time.time()
            time.sleep(0.2)
            t_end = time.time()
            with lock:
                call_timestamps.append((t_start, t_end))
            import numpy as np
            prices = list(100 + np.cumsum(np.random.default_rng(hash(symbol) % 1000).normal(0.05, 1.5, 150)))
            return _make_price_series(prices, n_days=150)

        def get_provider_name(self) -> str:
            return "mock"

    cash_repo = SQLiteCashLedgerRepository(sf)
    price_repo = SQLitePriceRepository(sf)
    service = RiskService(
        transaction_repo=tx_repo, cash_ledger_repo=cash_repo,
        price_sync_service=PriceSyncService(price_repo, SlowProvider()),
        risk_calculator=RiskCalculator(min_data_points=30),
        return_calculator=ReturnCalculator(), risk_free_rate_annual=0.45,
    )

    service.compute_risk_profile(pid, lookback_days=100, include_cash=False)
    engine.dispose()

    assert len(call_timestamps) == 5, f"5 sembol için 5 fetch bekleniyor, gelen: {len(call_timestamps)}"
    # Paralellik kanıtı: İKİNCİ fetch'in başlangıcı, İLK fetch'in
    # BİTİŞİNDEN ÖNCE olmalı (örtüşme = gerçek paralellik). Sıralı
    # çalışsaydı, her fetch bir öncekinin bitişinden SONRA başlardı.
    sorted_calls = sorted(call_timestamps, key=lambda x: x[0])
    overlaps = sum(
        1 for i in range(1, len(sorted_calls))
        if sorted_calls[i][0] < sorted_calls[i - 1][1]
    )
    assert overlaps >= 3, (
        f"Fetch'ler ÖRTÜŞMÜYOR — paralellik çalışmıyor olabilir. "
        f"Zaman damgaları: {sorted_calls}"
    )
