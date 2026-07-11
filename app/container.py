"""
Dependency Injection Container.

Tüm servis ve repository instance'larını yönetir.
Streamlit'te @st.cache_resource ile singleton olarak kullanılır.

Tasarım kararı (ADR-003, ADR-004):
  - Tüm bağımlılıklar constructor injection ile sağlanır
  - Container tek noktada konfigüre edilir
  - Test'lerde mock repository'ler inject edilebilir
  - Streamlit rerun'larında yeniden oluşturulmaz (@st.cache_resource)

Kullanım (app.py içinde):
    container = get_container()
    portfolio_service = container.portfolio_service
"""

from __future__ import annotations

from sqlalchemy.orm import sessionmaker, Session

from src.domain.calculators.cost_basis_calculator import (
    FIFOCostBasisCalculator,
    WAVGCostBasisCalculator,
)
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.calculators.backtest_engine import BacktestEngine
from src.services.backtest_service import BacktestService
from src.domain.enums.var_method import VaRMethod
from src.services.risk_service import RiskService
from src.services.price_sync_service import PriceSyncService
from src.domain.calculators.technical_calculator import TechnicalCalculator
from src.infrastructure.config.settings import Settings, get_settings
from src.infrastructure.data_providers.yfinance_adapter import YFinanceAdapter
from src.infrastructure.data_providers.tefas_adapter import TefasAdapter
from src.infrastructure.data_providers.provider_router import ProviderRouter
from sqlalchemy.engine import Engine
from src.infrastructure.database.connection import (
    check_database_connection,
    create_db_engine,
    create_session_factory,
    initialize_database,
)
from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus
from src.services.transaction_service import TransactionService
from src.infrastructure.logging_config import configure_from_settings, get_logger
from src.infrastructure.repositories.sqlite.cash_ledger_repository import (
    SQLiteCashLedgerRepository,
)
from src.infrastructure.repositories.sqlite.portfolio_repository import (
    SQLitePortfolioRepository,
)
from src.infrastructure.repositories.sqlite.price_repository import (
    SQLitePriceRepository,
)
from src.infrastructure.repositories.sqlite.transaction_repository import (
    SQLiteTransactionRepository,
)
from src.infrastructure.repositories.sqlite.risk_snapshot_repository import (
    SQLiteRiskSnapshotRepository,
)
from src.infrastructure.repositories.sqlite.watchlist_repository import (
    SQLiteWatchlistRepository,
)
from src.infrastructure.repositories.sqlite.fund_repository import SQLiteFundRepository
from src.infrastructure.repositories.sqlite.corporate_action_repository import (
    SQLiteCorporateActionRepository,
)
from src.services.watchlist_service import WatchlistService
from src.services.ai_insight_service import AIInsightService
from src.infrastructure.scheduler.snapshot_scheduler import SnapshotScheduler
from src.infrastructure.cache.ttl_cache import TTLCache
from src.services.market_data_service import MarketDataService
from src.services.portfolio_service import PortfolioService

logger = get_logger(__name__)


class Container:
    """
    Uygulama bağımlılık konteyneri.

    Tüm instance'lar lazy initialization ile oluşturulur.
    İlk erişimde oluşturulur, sonraki erişimlerde cache'den döner.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: Engine | None = None
        self._session_factory: "sessionmaker[Session]" | None = None
        self._event_bus: InMemoryEventBus | None = None

        # Repositories
        self._portfolio_repo: SQLitePortfolioRepository | None = None
        self._transaction_repo: SQLiteTransactionRepository | None = None
        self._transaction_service: TransactionService | None = None
        self._risk_service: RiskService | None = None
        self._risk_cache: TTLCache | None = None
        self._risk_snapshot_repo: SQLiteRiskSnapshotRepository | None = None
        self._scheduler_instance: SnapshotScheduler | None = None
        self._price_sync_service: PriceSyncService | None = None
        self._watchlist_repo: SQLiteWatchlistRepository | None = None
        self._fund_repo: SQLiteFundRepository | None = None
        self._corporate_action_repo: SQLiteCorporateActionRepository | None = None
        self._watchlist_service: WatchlistService | None = None
        self._ai_insight_service: AIInsightService | None = None
        self._backtest_engine: BacktestEngine | None = None
        self._backtest_service: BacktestService | None = None
        self._price_repo: SQLitePriceRepository | None = None
        self._cash_ledger_repo: SQLiteCashLedgerRepository | None = None

        # Market data (Infrastructure adapter + Service orkestrasyonu)
        self._yfinance_adapter: YFinanceAdapter | None = None
        self._tefas_adapter: TefasAdapter | None = None
        self._provider_router: ProviderRouter | None = None
        self._market_data_service: MarketDataService | None = None
        self._portfolio_service: PortfolioService | None = None
        self._analysis_cache: TTLCache | None = None  # TTLCache — MarketDataService için

        # Calculators (stateless — her çağrıda yeniden kullanılabilir)
        self._wavg_calculator = WAVGCostBasisCalculator()
        self._fifo_calculator = FIFOCostBasisCalculator()
        self._return_calculator = ReturnCalculator()
        self._risk_calculator = RiskCalculator(
            trading_days=settings.financial.risk.trading_days_per_year,
            min_data_points=settings.financial.risk.min_data_points_for_risk,
        )

    def _get_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_db_engine(
                database_url=self._settings.database.url,
                echo=self._settings.app.debug,
            )
        return self._engine

    def _get_session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = create_session_factory(self._get_engine())
        return self._session_factory

    @property
    def event_bus(self) -> InMemoryEventBus:
        if self._event_bus is None:
            self._event_bus = InMemoryEventBus()
        return self._event_bus

    @property
    def portfolio_repository(self) -> SQLitePortfolioRepository:
        if self._portfolio_repo is None:
            self._portfolio_repo = SQLitePortfolioRepository(self._get_session_factory())
        return self._portfolio_repo

    @property
    def transaction_repository(self) -> SQLiteTransactionRepository:
        if self._transaction_repo is None:
            self._transaction_repo = SQLiteTransactionRepository(self._get_session_factory())
        return self._transaction_repo

    @property
    def transaction_service(self) -> TransactionService:
        """
        DÜZELTME (bu turda): Bu property daha önce hiç YOKTU —
        SQLiteTransactionRepository.add_transaction() gerçek bir
        orkestrasyon katmanı olmadan doğrudan çağrılamıyordu (validasyon,
        symbol_type sınıflandırma, event yayını — hiçbiri repository'nin
        sorumluluğu değil). container.event_bus DAHA ÖNCE HİÇ
        KULLANILMIYORDU (Faz B'de kurulmuş ama hiçbir servise inject
        edilmemişti) — şimdi ilk gerçek tüketicisine kavuşuyor.
        """
        if self._transaction_service is None:
            self._transaction_service = TransactionService(
                transaction_repo=self.transaction_repository,
                cash_ledger_repo=self.cash_ledger_repository,
                event_bus=self.event_bus,
            )
        return self._transaction_service

    @property
    def price_repository(self) -> SQLitePriceRepository:
        if self._price_repo is None:
            self._price_repo = SQLitePriceRepository(self._get_session_factory())
        return self._price_repo

    @property
    def cash_ledger_repository(self) -> SQLiteCashLedgerRepository:
        if self._cash_ledger_repo is None:
            self._cash_ledger_repo = SQLiteCashLedgerRepository(self._get_session_factory())
        return self._cash_ledger_repo

    @property
    def yfinance_adapter(self) -> YFinanceAdapter:
        """
        BIST piyasa verisi sağlayıcı adapter'ı.

        Tüm retry/timeout/sembol-suffix parametreleri settings.yaml'dan
        okunur — hard-coded değer yoktur (bkz. config/settings.yaml
        data_providers.bist).
        """
        if self._yfinance_adapter is None:
            bist_cfg = self._settings.data_providers.bist
            self._yfinance_adapter = YFinanceAdapter(
                retry_count=bist_cfg.retry.max_attempts,
                backoff_factor=bist_cfg.retry.backoff_factor,
                base_delay_seconds=bist_cfg.retry.base_delay_seconds,
                max_delay_seconds=bist_cfg.retry.max_delay_seconds,
                symbol_suffix=bist_cfg.symbol_suffix,
            )
        return self._yfinance_adapter

    @property
    def tefas_adapter(self) -> TefasAdapter:
        """
        TEFAS piyasa verisi sağlayıcı adapter'ı.

        DÜZELTME (bu turda bulundu): Bu property ve provider_router
        DAHA ÖNCE HİÇ VAR DEĞİLDİ — market_data_service doğrudan
        yfinance_adapter'a bağlıydı, yani TEFAS fonları (örn. "YAC")
        sessizce yfinance'e gönderiliyordu ve muhtemelen
        SymbolNotFoundError alıyordu. ProviderRouter sınıfı
        (provider_router.py) YAZILMIŞTI ama HİÇ BAĞLANMAMIŞTI.
        """
        if self._tefas_adapter is None:
            tefas_cfg = self._settings.data_providers.tefas
            self._tefas_adapter = TefasAdapter(
                rate_limit_per_minute=tefas_cfg.rate_limit_per_minute,
                chunk_size_days=tefas_cfg.chunk_size_days,
                retry_count=tefas_cfg.retry.max_attempts,
                base_delay_seconds=tefas_cfg.retry.base_delay_seconds,
                max_delay_seconds=tefas_cfg.retry.max_delay_seconds,
            )
        return self._tefas_adapter

    @property
    def provider_router(self) -> ProviderRouter:
        if self._provider_router is None:
            self._provider_router = ProviderRouter(
                bist_provider=self.yfinance_adapter,
                tefas_provider=self.tefas_adapter,
            )
        return self._provider_router

    @property
    def market_data_service(self) -> MarketDataService:
        """
        Piyasa verisi + teknik analiz orkestrasyon servisi.

        Aşama 8: TTLCache(60s) inject edilmiş — aynı sembol+timeframe için
        60 saniye içindeki tekrar çağrılar yfinance'a gitmeden cache'den döner.
        Bu, portföy sekmesinde concurrent 20 sembol çekiminde N→1 çağrıya düşer.
        """
        if self._market_data_service is None:
            if self._analysis_cache is None:
                self._analysis_cache = TTLCache(ttl_seconds=60.0)
            self._market_data_service = MarketDataService(
                provider=self.provider_router,
                calculator=TechnicalCalculator,
                cache=self._analysis_cache,
            )
        return self._market_data_service

    @property
    def portfolio_service(self) -> PortfolioService:
        """
        Portföy PnL ve pozisyon durumu servisi (Aşama 8: concurrent).

        Bağımlılıklar:
          - transaction_repository  (Infrastructure — pozisyon okuma)
          - portfolio_repository    (Infrastructure — portföy listesi)
          - market_data_service     (Service — concurrent fiyat çekme)
          - wavg_calculator         (Domain — maliyet bazı; finansal formül buraya sızmaz)
          - max_workers             (settings'ten; hard-code değil)
        """
        if self._portfolio_service is None:
            self._portfolio_service = PortfolioService(
                transaction_repo=self.transaction_repository,
                market_data_service=self.market_data_service,
                portfolio_repo=self.portfolio_repository,
                calculator=self.wavg_calculator,
                max_workers=self._settings.data_providers.bist.max_concurrent_requests,
            )
        return self._portfolio_service

    @property
    def wavg_calculator(self) -> WAVGCostBasisCalculator:
        return self._wavg_calculator

    @property
    def fifo_calculator(self) -> FIFOCostBasisCalculator:
        return self._fifo_calculator

    @property
    def return_calculator(self) -> ReturnCalculator:
        return self._return_calculator

    @property
    def watchlist_repository(self) -> SQLiteWatchlistRepository:
        if self._watchlist_repo is None:
            self._watchlist_repo = SQLiteWatchlistRepository(self._get_session_factory())
        return self._watchlist_repo

    @property
    def fund_repository(self) -> SQLiteFundRepository:
        if self._fund_repo is None:
            self._fund_repo = SQLiteFundRepository(self._get_session_factory())
        return self._fund_repo

    @property
    def corporate_action_repository(self) -> SQLiteCorporateActionRepository:
        if self._corporate_action_repo is None:
            self._corporate_action_repo = SQLiteCorporateActionRepository(self._get_session_factory())
        return self._corporate_action_repo

    @property
    def watchlist_service(self) -> WatchlistService:
        if self._watchlist_service is None:
            self._watchlist_service = WatchlistService(
                self.watchlist_repository, market_data_service=self.market_data_service,
            )
        return self._watchlist_service

    @property
    def ai_insight_service(self) -> AIInsightService:
        if self._ai_insight_service is None:
            ai_cfg = self._settings.ai
            self._ai_insight_service = AIInsightService(
                portfolio_service=self.portfolio_service,
                risk_service=self.risk_service,
                api_key=ai_cfg.api_key,
                model=ai_cfg.model,
                max_tokens=ai_cfg.max_tokens,
            )
        return self._ai_insight_service

    @property
    def risk_calculator(self) -> RiskCalculator:
        return self._risk_calculator

    @property
    def backtest_engine(self) -> BacktestEngine:
        if self._backtest_engine is None:
            risk_cfg = self._settings.financial.risk
            self._backtest_engine = BacktestEngine(
                return_calculator=self.return_calculator,
                risk_calculator=self.risk_calculator,
                trading_days=risk_cfg.trading_days_per_year,
                risk_free_rate_annual=risk_cfg.risk_free_rate_annual,
            )
        return self._backtest_engine

    @property
    def backtest_service(self) -> BacktestService:
        if self._backtest_service is None:
            self._backtest_service = BacktestService(
                price_sync_service=self.price_sync_service,
                backtest_engine=self.backtest_engine,
            )
        return self._backtest_service

    @property
    def risk_snapshot_repository(self) -> SQLiteRiskSnapshotRepository:
        if self._risk_snapshot_repo is None:
            self._risk_snapshot_repo = SQLiteRiskSnapshotRepository(self._get_session_factory())
        return self._risk_snapshot_repo

    @property
    def scheduler_enabled(self) -> bool:
        """
        app.py'ın container._settings'e DOĞRUDAN ERİŞMEDEN scheduler'ın
        açık/kapalı durumunu okuyabilmesi için — katman izolasyonu
        kuralı (app.py yalnızca Container'ın PUBLIC arayüzünü kullanır).
        """
        result: bool = self._settings.scheduler.enabled
        return result

    @property
    def scheduler_interval_hours(self) -> int:
        result: int = self._settings.scheduler.snapshot_interval_hours
        return result

    @property
    def scheduler(self) -> SnapshotScheduler:
        """
        DÜZELTME (bu turda): Faz F kapsamı — APScheduler entegrasyonu
        daha önce hiç yoktu. @st.cache_resource ile Container tekil
        olduğu için, bu property de doğal olarak process-genelinde
        TEK bir SnapshotScheduler instance'ı garanti ediyor (Container
        ile AYNI desen, tekrar icat edilmedi).
        """
        if self._scheduler_instance is None:
            scheduler_cfg = self._settings.scheduler
            self._scheduler_instance = SnapshotScheduler(
                portfolio_service=self.portfolio_service,
                risk_service=self.risk_service,
                risk_snapshot_repo=self.risk_snapshot_repository,
                interval_hours=scheduler_cfg.snapshot_interval_hours,
                lookback_days=scheduler_cfg.snapshot_lookback_days,
                max_portfolios_per_run=scheduler_cfg.max_portfolios_per_run,
            )
        return self._scheduler_instance

    @property
    def price_sync_service(self) -> PriceSyncService:
        """
        DÜZELTME (bu turda bulundu — ÜÇÜNCÜ "unutulmuş entegrasyon"):
        PriceRepository (Faz B'de upsert/upsert_batch/get_missing_dates
        ile inşa edildi) HİÇBİR ZAMAN gerçek bir yazıcıya sahip değildi.
        RiskService artık CANLI sağlayıcıya (provider_router) doğrudan
        değil, bu Cache-Aside katmanı üzerinden erişiyor.
        """
        if self._price_sync_service is None:
            self._price_sync_service = PriceSyncService(
                price_repo=self.price_repository,
                market_data_provider=self.provider_router,
            )
        return self._price_sync_service

    @property
    def risk_service(self) -> RiskService:
        """
        DÜZELTME (bu turda bulundu — "bir şey unutmayalım" prensibiyle
        kontrol edildi): RiskService sınıfı yazılmıştı ama container.py'a
        HİÇ BAĞLANMAMIŞTI — tam bir "unutulmuş entegrasyon" örneği.

        Ayrı bir TTLCache instance'ı (_risk_cache) kullanıyor —
        _analysis_cache (MarketDataService için, 60s TTL) ile
        PAYLAŞILMIYOR çünkü risk hesaplamaları çok daha uzun TTL
        gerektiriyor (1 saat) — farklı tazelik gereksinimleri farklı
        cache instance'ları haklı çıkarıyor (bkz. risk_service.py
        modül docstring'i).
        """
        if self._risk_service is None:
            if self._risk_cache is None:
                self._risk_cache = TTLCache(ttl_seconds=3600.0)
            risk_cfg = self._settings.financial.risk
            self._risk_service = RiskService(
                transaction_repo=self.transaction_repository,
                cash_ledger_repo=self.cash_ledger_repository,
                price_sync_service=self.price_sync_service,
                risk_calculator=self.risk_calculator,
                return_calculator=self.return_calculator,
                risk_free_rate_annual=risk_cfg.risk_free_rate_annual,
                trading_days=risk_cfg.trading_days_per_year,
                var_confidence=risk_cfg.var_confidence_level,
                var_method=VaRMethod(risk_cfg.var_method),
                cache=self._risk_cache,
            )
        return self._risk_service

    @property
    def settings(self) -> Settings:
        return self._settings

    def initialize(self) -> None:
        """
        Veritabanını başlat ve bağlantıyı doğrula.

        Uygulama başlangıcında bir kez çağrılır.
        """
        engine = self._get_engine()

        if not check_database_connection(engine):
            raise RuntimeError("Veritabanına bağlanılamadı!")

        initialize_database(engine)

        logger.info(
            "container_initialized",
            database_url=self._settings.database.url,
            environment=self._settings.app.environment,
        )


def build_container(settings: Settings | None = None) -> Container:
    """
    Container oluştur ve başlat.

    Args:
        settings: None → get_settings() kullanılır.
    """
    if settings is None:
        settings = get_settings()

    # Logging'i settings'e göre konfigüre et
    configure_from_settings()

    container = Container(settings)
    container.initialize()
    return container


# ── Streamlit Singleton ───────────────────────────────────────────────────────
# Bu fonksiyon app.py içinde @st.cache_resource dekoratörüyle sarılır.
# Doğrudan çağırmak her seferinde yeni container oluşturur.

def get_container() -> Container:
    """
    Singleton container döndür.

    Streamlit'te kullanım (app.py):
        @st.cache_resource
        def _get_container():
            return get_container()

        container = _get_container()
    """
    return build_container()
