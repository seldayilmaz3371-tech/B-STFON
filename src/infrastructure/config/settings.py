"""
Uygulama konfigürasyon sistemi.

pydantic-settings kullanır: YAML dosyası + environment variable override.
Environment variable format: PORTFOLIO_OS__SECTION__KEY (double underscore)

Örnek:
  PORTFOLIO_OS__FINANCIAL__RISK__RISK_FREE_RATE_ANNUAL=0.50
  PORTFOLIO_OS__DATABASE__URL=postgresql+psycopg2://...

Kullanım:
  from src.infrastructure.config.settings import get_settings
  settings = get_settings()
  db_url = settings.database.url
"""

from __future__ import annotations

import functools
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Alt Konfigürasyon Sınıfları ───────────────────────────────────────────────

class SQLiteConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000
    foreign_keys: bool = True
    temp_store: str = "MEMORY"
    mmap_size: int = 268435456
    busy_timeout_ms: int = 5000


class PoolConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    size: int = 5
    max_overflow: int = 10
    timeout: int = 30
    recycle: int = 3600


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    url: str = "sqlite:///./data/portfolio.db"
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)


class L1CacheConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    default_ttl_seconds: int = 300
    price_ttl_market_open: int = 300
    price_ttl_market_closed: int = 86400
    portfolio_value_ttl: int = 300
    max_size: int = 1000


class L2CacheConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    historical_price_never_expires: bool = True
    fund_metadata_ttl_days: int = 7


class CacheConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    l1: L1CacheConfig = Field(default_factory=L1CacheConfig)
    l2: L2CacheConfig = Field(default_factory=L2CacheConfig)


class RetryConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_factor: float = 2.0


class BISTProviderConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    primary: str = "yfinance"
    fallback: list[str] = Field(default_factory=list)
    symbol_suffix: str = ".IS"
    request_timeout_seconds: int = 15
    max_concurrent_requests: int = 8  # ThreadPoolExecutor max_workers için
    retry: RetryConfig = Field(default_factory=RetryConfig)


class TEFASProviderConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    library: str = "pytefas"
    rate_limit_per_minute: int = 6
    chunk_size_days: int = 25
    request_timeout_seconds: int = 30
    fund_limit_per_request: int = 50
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(
            max_attempts=3,
            base_delay_seconds=10.0,
            max_delay_seconds=60.0,
        )
    )


class HealthMonitoringConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    error_threshold_1h: float = 0.20
    log_retention_days: int = 30


class DataProvidersConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    bist: BISTProviderConfig = Field(default_factory=BISTProviderConfig)
    tefas: TEFASProviderConfig = Field(default_factory=TEFASProviderConfig)
    health_monitoring: HealthMonitoringConfig = Field(default_factory=HealthMonitoringConfig)


class BISTPortfolioConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    commission_rate: float = 0.0018
    commission_vat_rate: float = 0.20
    bsmv_rate: float = 0.001
    bsmv_exempt_asset_types: list[str] = Field(
        default_factory=lambda: ["BIST_STOCK", "BIST_ETF"]
    )
    settlement_days: int = 2
    min_lot: int = 1


class TEFASPortfolioConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    commission_rate: float = 0.0
    stamp_duty_rate: float = 0.002
    settlement_days: int = 0
    nav_decimal_places: int = 6


class PortfolioConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    default_currency: str = "TRY"
    default_cost_method: str = "WAVG"
    default_benchmark: str = "XU100.IS"
    bist: BISTPortfolioConfig = Field(default_factory=BISTPortfolioConfig)
    tefas: TEFASPortfolioConfig = Field(default_factory=TEFASPortfolioConfig)


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    risk_free_rate_annual: float = 0.45
    trading_days_per_year: int = 252
    var_confidence_level: float = 0.95
    cvar_confidence_level: float = 0.95
    var_method: str = "HISTORICAL"
    default_lookback_days: int = 252
    min_data_points_for_risk: int = 30


class SchedulerConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = True
    snapshot_interval_hours: int = 6
    snapshot_lookback_days: int = 252
    max_portfolios_per_run: int = 100


class AIConfig(BaseSettings):
    """
    AI destekli portföy özeti — bu turda eklendi.

    KARAR (kullanıcıdan alındı, varsayılmadı): API anahtarı KULLANICININ
    KENDİ .env dosyasına girilir (PORTFOLIO_OS__AI__API_KEY) — maliyeti
    kullanıcı üstlenir, sistem varsayılan bir anahtar İÇERMEZ.
    api_key boşsa (None), AIInsightService özelliği KENDİLİĞİNDEN devre
    dışı kalır (çökme değil, UI'da "API anahtarı ayarlanmamış" mesajı).
    """
    model_config = SettingsConfigDict(extra="ignore")

    api_key: str | None = None
    model: str = "claude-sonnet-5"
    # NOT: Model string'i GERÇEKTEN doğrulandı (web araması ile, bu
    # turda) — "claude-sonnet-5", tahmin/hafızadan YAZILMADI. Sonnet
    # sınıfı seçildi çünkü bu görev ("verilen sayıları doğal dile çevir")
    # Opus'un karmaşık ajan görevleri için optimize edilmiş gücünü
    # GEREKTİRMİYOR — maliyet/kalite dengesi için yeterli.
    max_tokens: int = 1024


class PrecisionConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    price_decimal_places: int = 6
    quantity_decimal_places: int = 6
    money_decimal_places: int = 2
    pct_decimal_places: int = 6


class WithholdingTaxConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    individual_rate: float = 0.10
    corporate_rate: float = 0.15


class FinancialConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    risk: RiskConfig = Field(default_factory=RiskConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    withholding_tax: WithholdingTaxConfig = Field(default_factory=WithholdingTaxConfig)


class LoggingOutputConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    console: bool = True
    file: bool = True
    file_path: str = "logs/portfolio_os.log"
    file_rotation: str = "1 day"
    file_retention: str = "30 days"


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    level: str = "INFO"
    format: str = "console"  # "console" | "json"
    output: LoggingOutputConfig = Field(default_factory=LoggingOutputConfig)
    include_caller_info: bool = False
    overrides: dict[str, str] = Field(
        default_factory=lambda: {
            "sqlalchemy.engine": "WARNING",
            "apscheduler": "WARNING",
            "yfinance": "WARNING",
        }
    )


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    name: str = "Portfolio OS"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    timezone: str = "Europe/Istanbul"
    locale: str = "tr_TR"


# ── Ana Settings Sınıfı ───────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Uygulama ana konfigürasyon sınıfı.

    Öncelik sırası (yüksekten düşüğe):
      1. Environment variables (PORTFOLIO_OS__*)
      2. .env dosyası
      3. config/settings.yaml
      4. Pydantic field defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="PORTFOLIO_OS__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    data_providers: DataProvidersConfig = Field(default_factory=DataProvidersConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    financial: FinancialConfig = Field(default_factory=FinancialConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    ai: AIConfig = Field(default_factory=AIConfig)

    @property
    def is_production(self) -> bool:
        return self.app.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.app.environment == "development"

    @property
    def is_testing(self) -> bool:
        return self.app.environment == "testing"

    @property
    def risk_free_rate(self) -> float:
        """Güncel TCMB politika faizi (config'den)."""
        return self.financial.risk.risk_free_rate_annual

    @property
    def trading_days(self) -> int:
        return self.financial.risk.trading_days_per_year


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """YAML config dosyasını yükle. Dosya yoksa boş dict döner."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_file: str = "config/settings.yaml") -> Settings:
    """
    Ayarları YAML + environment variable kombinasyonundan yükle.

    YAML dosyası temel değerleri sağlar.
    Environment variables her şeyi override edebilir.
    """
    yaml_config = _load_yaml_config(Path(config_file))

    # YAML'dan gelen değerlerle Settings oluştur
    # Environment variables pydantic-settings tarafından otomatik override edilir
    return Settings(**yaml_config)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton settings instance.

    Test'lerde get_settings.cache_clear() ile cache temizlenebilir.
    """
    return load_settings()
