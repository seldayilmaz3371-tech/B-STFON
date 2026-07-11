"""
Domain katmanı exception sınıfları.

Hiyerarşi:
  PortfolioOSError (taban)
  ├── DomainError (iş mantığı hataları)
  │   ├── ValidationError
  │   ├── InsufficientQuantityError
  │   ├── InsufficientFundsError
  │   ├── DuplicateError
  │   ├── NotFoundError
  │   ├── AlreadyReversedError
  │   ├── ImmutableFieldError
  │   └── CurrencyMismatchError
  ├── CalculationError (hesaplama hataları)
  │   ├── ConvergenceError
  │   └── InsufficientDataError
  └── ProviderError (veri sağlayıcı hataları — infrastructure'dan yükselen)
      ├── ProviderUnavailableError
      ├── SymbolNotFoundError
      ├── NoDataError
      ├── RateLimitError
      ├── StaleDataError
      └── DataValidationError
"""

from __future__ import annotations

from typing import Any


class PortfolioOSError(Exception):
    """Tüm uygulama hatalarının taban sınıfı."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __repr__(self) -> str:
        ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.__class__.__name__}({self.message!r}, {ctx})"


# ─────────────────────────────────────────────────────────────────────────────
# Domain Hataları
# ─────────────────────────────────────────────────────────────────────────────


class DomainError(PortfolioOSError):
    """İş mantığı kuralı ihlali."""
    pass


class ValidationError(DomainError):
    """
    Model veya işlem validasyon hatası.

    Kullanım: Geçersiz field değerleri, format hataları,
    iş kuralı constraint ihlalleri.
    """

    def __init__(self, message: str, field: str | None = None, **context: Any) -> None:
        super().__init__(message, field=field, **context)
        self.field = field


class InsufficientQuantityError(DomainError):
    """
    Satış miktarı mevcut pozisyonu aşıyor.

    Kritik: BIST T+2 settlement nedeniyle available_quantity
    total_quantity'den az olabilir.
    """

    def __init__(
        self,
        symbol: str,
        requested: Any,
        available: Any,
        **context: Any,
    ) -> None:
        super().__init__(
            f"{symbol}: İstenen miktar ({requested}) mevcut pozisyonu ({available}) aşıyor.",
            symbol=symbol,
            requested=str(requested),
            available=str(available),
            **context,
        )
        self.symbol = symbol
        self.requested = requested
        self.available = available


class InsufficientFundsError(DomainError):
    """Nakit bakiyesi yetersiz."""

    def __init__(self, required: Any, available: Any, **context: Any) -> None:
        super().__init__(
            f"Nakit yetersiz: Gereken {required}, Mevcut {available}.",
            required=str(required),
            available=str(available),
            **context,
        )


class DuplicateError(DomainError):
    """Aynı kayıt zaten mevcut."""

    def __init__(self, entity: str, identifier: Any, **context: Any) -> None:
        super().__init__(
            f"{entity} zaten mevcut: {identifier}",
            entity=entity,
            identifier=str(identifier),
            **context,
        )


class NotFoundError(DomainError):
    """İstenen kayıt bulunamadı."""

    def __init__(self, entity: str, identifier: Any, **context: Any) -> None:
        super().__init__(
            f"{entity} bulunamadı: {identifier}",
            entity=entity,
            identifier=str(identifier),
            **context,
        )


class PortfolioNotFoundError(NotFoundError):
    """Portföy bulunamadı."""

    def __init__(self, portfolio_id: Any) -> None:
        super().__init__("Portfolio", portfolio_id)


class TransactionNotFoundError(NotFoundError):
    """İşlem bulunamadı."""

    def __init__(self, transaction_id: Any) -> None:
        super().__init__("Transaction", transaction_id)


class AlreadyReversedError(DomainError):
    """İşlem zaten iptal edilmiş."""

    def __init__(self, transaction_id: Any) -> None:
        super().__init__(
            f"İşlem zaten iptal edilmiş: {transaction_id}",
            transaction_id=str(transaction_id),
        )


class ImmutableFieldError(DomainError):
    """
    Değiştirilemez alan üzerinde değişiklik girişimi.

    Örnek: transaction.id, portfolio.inception_date
    """

    def __init__(self, entity: str, field: str, **context: Any) -> None:
        super().__init__(
            f"{entity}.{field} alanı değiştirilemez.",
            entity=entity,
            field=field,
            **context,
        )


class CurrencyMismatchError(DomainError):
    """
    Farklı para birimlerinde işlem yapılmaya çalışıldı.

    Money value object'lerde farklı currency toplanmaya çalışıldığında.
    """

    def __init__(self, expected: str, actual: str, **context: Any) -> None:
        super().__init__(
            f"Para birimi uyumsuzluğu: Beklenen {expected}, Gelen {actual}",
            expected=expected,
            actual=actual,
            **context,
        )


class BusinessRuleError(DomainError):
    """Genel iş kuralı ihlali."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Hesaplama Hataları
# ─────────────────────────────────────────────────────────────────────────────


class CalculationError(PortfolioOSError):
    """Finansal hesaplama hatası."""
    pass


class ConvergenceError(CalculationError):
    """
    Sayısal optimizasyon yakınsamadı.

    MWRR (XIRR) hesabında nadir görülür.
    Çok düzensiz nakit akışı patterns'larında oluşabilir.
    """

    def __init__(self, algorithm: str, **context: Any) -> None:
        super().__init__(
            f"{algorithm} yakınsamadı. Nakit akışı yapısını kontrol edin.",
            algorithm=algorithm,
            **context,
        )


class InsufficientDataError(CalculationError):
    """
    Hesaplama için yeterli veri noktası yok.

    Risk metrikleri minimum 30 iş günü (tercihen 252) gerektirir.
    """

    def __init__(self, required: int, available: int, metric: str, **context: Any) -> None:
        super().__init__(
            f"{metric} hesabı için {required} veri noktası gerekli, {available} mevcut.",
            required=required,
            available=available,
            metric=metric,
            **context,
        )
        # DÜZELTME (NoDataError'da bulunan aynı desen hatası — bu turda
        # önceden tespit edildi, burada da PROAKTİF olarak düzeltiliyor):
        # context dict'e koymak self.required gibi doğrudan attribute
        # erişimi SAĞLAMAZ. RiskService UI entegrasyonunda bu alanlara
        # doğrudan erişim gerektiği için eklendi.
        self.required = required
        self.available = available
        self.metric = metric


# ─────────────────────────────────────────────────────────────────────────────
# Provider Hataları (Infrastructure'dan yükselen, domain'de yakalanabilen)
# ─────────────────────────────────────────────────────────────────────────────


class ProviderError(PortfolioOSError):
    """Veri sağlayıcı hatası."""

    def __init__(self, message: str, provider: str, **context: Any) -> None:
        super().__init__(message, provider=provider, **context)
        self.provider = provider


class ProviderUnavailableError(ProviderError):
    """Veri sağlayıcısına ulaşılamıyor."""

    def __init__(self, provider: str, reason: str = "", **context: Any) -> None:
        super().__init__(
            f"{provider} erişilemiyor. {reason}".strip(),
            provider=provider,
            **context,
        )


class SymbolNotFoundError(ProviderError):
    """Sembol veri sağlayıcıda bulunamadı."""

    def __init__(self, symbol: str, provider: str, **context: Any) -> None:
        super().__init__(
            f"{symbol} sembolü {provider}'da bulunamadı.",
            provider=provider,
            symbol=symbol,
            **context,
        )
        self.symbol = symbol


class NoDataError(ProviderError):
    """İstenen tarih aralığında veri yok."""

    def __init__(self, symbol: str, provider: str, **context: Any) -> None:
        super().__init__(
            f"{symbol} için istenen tarih aralığında veri bulunamadı ({provider}).",
            provider=provider,
            symbol=symbol,
            **context,
        )
        # DÜZELTME: self.symbol/self.provider EKSİKTİ — test_yfinance_adapter.py
        # exc_info.value.symbol şeklinde doğrudan erişim BEKLİYORDU (diğer
        # exception'larda — örn. InsufficientQuantityError — kurulan
        # tutarlı desen). context dict'e symbol/provider koymak yeterli
        # DEĞİL, PortfolioOSError bunları otomatik attribute yapmıyor.
        self.symbol = symbol
        self.provider = provider


class RateLimitError(ProviderError):
    """
    Veri sağlayıcı rate limit aşıldı.

    retry_after_seconds: Kaç saniye sonra tekrar denenmeli.
    Bu exception retry mekanizması tarafından yakalanır.
    """

    def __init__(
        self,
        provider: str,
        retry_after_seconds: float = 60.0,
        **context: Any,
    ) -> None:
        super().__init__(
            f"{provider} rate limit aşıldı. {retry_after_seconds}s sonra tekrar deneyin.",
            provider=provider,
            retry_after_seconds=retry_after_seconds,
            **context,
        )
        self.retry_after_seconds = retry_after_seconds


class StaleDataError(ProviderError):
    """
    Veri taze değil — piyasa saatlerinde 15 dakikadan eski.

    Sistem otomatik olarak fallback provider'a geçer.
    """

    def __init__(self, symbol: str, provider: str, age_minutes: float, **context: Any) -> None:
        super().__init__(
            f"{symbol} verisi {age_minutes:.0f} dakika eski ({provider}).",
            provider=provider,
            symbol=symbol,
            age_minutes=age_minutes,
            **context,
        )


class DataValidationError(ProviderError):
    """Provider'dan gelen veri schema doğrulamasını geçemedi."""

    def __init__(self, provider: str, reason: str, **context: Any) -> None:
        super().__init__(
            f"{provider} veri doğrulama hatası: {reason}",
            provider=provider,
            reason=reason,
            **context,
        )


class AllProvidersFailedError(PortfolioOSError):
    """Tüm fallback provider'lar başarısız oldu."""

    def __init__(self, symbol: str, providers: list[str] | None = None) -> None:
        provider_list = ", ".join(providers) if providers else "tümü"
        super().__init__(
            f"{symbol} için hiçbir provider veri sağlayamadı ({provider_list}).",
            symbol=symbol,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Servis Katmanı Hataları
# ─────────────────────────────────────────────────────────────────────────────


class ServiceError(PortfolioOSError):
    """
    Servis (orkestrasyon) katmanı hatası.

    Service katmanı, Infrastructure/Domain katmanlarından yükselen ham
    exception'ları (ProviderError, DomainError, CalculationError vb.)
    UI/API katmanına sızdırmaz — bunun yerine bu sınıf veya alt sınıflarına
    sarmalayarak (wrap) raporlar. Orijinal hata `__cause__` üzerinden
    (Python'ın doğal exception chaining mekanizmasıyla) korunur; loglama ve
    hata ayıklama için kaybolmaz, yalnızca üst katmana sızması engellenir.
    """

    def __init__(self, message: str, operation: str, **context: Any) -> None:
        super().__init__(message, operation=operation, **context)
        self.operation = operation


class MarketDataServiceError(ServiceError):
    """
    MarketDataService orkestrasyonunda oluşan hata.

    Altta yatan sebep (provider erişilemedi, veri doğrulanamadı, hesaplama
    başarısız oldu vb.) `reason` alanında insan-okunabilir biçimde taşınır.
    """

    def __init__(
        self,
        message: str,
        symbol: str,
        operation: str = "get_market_analysis",
        **context: Any,
    ) -> None:
        super().__init__(message, operation=operation, symbol=symbol, **context)
        self.symbol = symbol


class DuplicatePortfolioNameError(DuplicateError):
    """Aynı isimde aktif bir portföy zaten var (DDL: uq_portfolio_name)."""

    def __init__(self, name: str) -> None:
        super().__init__("Portfolio", name)
