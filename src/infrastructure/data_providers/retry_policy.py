"""
Yeniden deneme (retry) politikası — Exponential Backoff.

Legacy bağlamı:
  BistKokpit V23.5'teki _yf_get() çağıranları (get_xu100, get_4h vb.)
  yalnızca "farklı period/interval kombinasyonu dene, olmazsa devamı geç"
  (try/except/continue) deseniyle çalışıyordu — gerçek bir ağ-hatası
  retry/backoff mekanizması yoktu. Bu modül o boşluğu, konfigüre edilebilir
  exponential backoff ile dolduran production-grade bir bileşendir.

Tasarım kararı:
  Retry mantığı YFinanceAdapter'a gömülmek yerine ayrı bir modülde
  tutulur. Gerekçe: TEFAS adapter'ı (gelecek faz) ve IS Yatirim adapter'ı
  da rate-limit/network hatalarına karşı aynı retry iskeletini
  kullanacak. Kod tekrarını önlemek ve her adapter'da tutarlı backoff
  davranışı garanti etmek için bu modül tüm data_providers/ katmanının
  ortak altyapısıdır.

Sıfır dış bağımlılık: yalnızca stdlib (time, logging, random).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """
    Exponential backoff retry konfigürasyonu.

    Hard-coded değer yok — her alan constructor'dan inject edilir.

    Bekleme süresi formülü:
      delay = min(base_delay * (backoff_factor ** attempt), max_delay)
      + jitter (0 ile delay*jitter_ratio arası rastgele ek süre)

    Jitter, "thundering herd" sorununu önler: birden fazla istemci aynı
    anda retry yaparsa hepsi aynı saniyede tekrar denemez.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_factor: float = 1.5
    jitter_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts en az 1 olmalı: {self.max_attempts}")
        if self.base_delay_seconds < 0:
            raise ValueError(f"base_delay_seconds negatif olamaz: {self.base_delay_seconds}")
        if self.backoff_factor < 1.0:
            raise ValueError(f"backoff_factor en az 1.0 olmalı: {self.backoff_factor}")

    def compute_delay(self, attempt: int) -> float:
        """
        Verilen deneme numarası (0-indexed) için bekleme süresini hesapla.

        Args:
            attempt: Kaçıncı deneme (0 = ilk retry, 1 = ikinci retry, ...).

        Returns:
            float: Saniye cinsinden bekleme süresi (jitter dahil).
        """
        raw_delay = self.base_delay_seconds * (self.backoff_factor**attempt)
        capped_delay = min(raw_delay, self.max_delay_seconds)
        jitter = random.uniform(0, capped_delay * self.jitter_ratio)
        return capped_delay + jitter


class RetryExhaustedError(Exception):
    """Tüm retry denemeleri tükendi — son hatayı taşır."""

    def __init__(self, attempts: int, last_exception: Exception) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"{attempts} deneme sonrası başarısız. Son hata: {last_exception}"
        )


def execute_with_retry(
    func: Callable[[], T],
    policy: RetryPolicy,
    retryable_exceptions: tuple[type[Exception], ...],
    operation_name: str = "operation",
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """
    Verilen fonksiyonu exponential backoff ile yeniden deneyerek çalıştır.

    Yalnızca retryable_exceptions tuple'ında belirtilen exception tipleri
    yakalanıp retry edilir. Diğer tüm exception'lar (örn. ValueError,
    DataValidationError) anında yükselir — retry mantığı yalnızca
    *geçici/ağ kaynaklı* hatalar için tasarlanmıştır.

    Args:
        func: Argümansız, çağrıldığında T döndüren fonksiyon (closure/lambda).
        policy: Retry konfigürasyonu.
        retryable_exceptions: Hangi exception tiplerinin retry'a tabi
                               olacağını belirten tuple.
        operation_name: Log mesajlarında görünecek işlem adı.
        sleep_fn: Test edilebilirlik için inject edilebilir sleep fonksiyonu
                  (testlerde gerçek zaman beklememek için mock'lanır).

    Returns:
        T: func()'un başarılı dönüş değeri.

    Raises:
        RetryExhaustedError: Tüm denemeler tükendi.
        Exception: retryable_exceptions dışındaki bir hata oluşursa
                   anında (retry yapılmadan) yükselir.
    """
    last_exception: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            result = func()
            if attempt > 0:
                logger.info(
                    "retry_succeeded",
                    operation=operation_name,
                    attempt=attempt + 1,
                    max_attempts=policy.max_attempts,
                )
            return result

        except retryable_exceptions as exc:
            last_exception = exc
            is_last_attempt = attempt == policy.max_attempts - 1

            if is_last_attempt:
                logger.error(
                    "retry_exhausted",
                    operation=operation_name,
                    attempts=policy.max_attempts,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                break

            delay = policy.compute_delay(attempt)
            logger.warning(
                "retry_attempt_failed",
                operation=operation_name,
                attempt=attempt + 1,
                max_attempts=policy.max_attempts,
                error_type=type(exc).__name__,
                error=str(exc),
                next_delay_seconds=round(delay, 2),
            )
            sleep_fn(delay)

    assert last_exception is not None  # mypy için — buraya yalnızca hata varsa gelinir
    raise RetryExhaustedError(attempts=policy.max_attempts, last_exception=last_exception)
