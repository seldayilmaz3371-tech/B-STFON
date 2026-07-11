"""
In-process, synchronous event bus — ADR-004 uygulaması.

Mimari karar (bu projede daha önce gerekçelendirildi):
  Handler hataları LOGLANIR ama yukarı FIRLATILMAZ ("best-effort").
  Gerekçe: Bir event handler'ın hatası (örn. cache invalidation
  başarısız), transaction'ın DB'ye yazılmasını geri almamalı —
  muhasebe kaydının atomicity'si, türetilmiş bir cache'in tutarlılığından
  DAHA DEĞERLİ. Publisher (örn. TransactionRepository.add()), event
  publish edilmeden ÖNCE kendi yazma işlemini commit etmiş olmalı;
  bu bus'ın publish() metodu her zaman commit SONRASI çağrılmalı
  (bu, çağıran koda bırakılan bir disiplin — bus bunu zorlayamaz,
  bu yüzden repository implementasyonlarında açıkça yorumlanacak).

Neden hâlâ tek process, senkron (Kafka/RabbitMQ değil):
  ADR-004: external message broker bu ölçekte sıfır operasyonel kazanç
  sağlar, zero network latency/serialization overhead + exception
  propagation kolaylığı senkron modelin avantajı.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, DefaultDict

from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

EventHandler = Callable[["Event"], None]


@dataclass(frozen=True)
class Event:
    """
    Tüm domain event'lerinin taşıyıcısı.

    name: "transaction.added", "position.invalidated" gibi nokta-ayraçlı
      isimlendirme — subscribe() bu isimle eşleşir.
    payload: Event'e özgü veri (örn. {"portfolio_id": ..., "symbol": ...}).
      Bilinçli olarak `dict[str, Any]` — her event tipi için ayrı
      dataclass tanımlamak bu aşamada aşırı mühendislik (henüz yalnızca
      1-2 event tipi var); event sayısı arttıkça (Faz D+) tip-güvenli
      payload dataclass'larına geçiş değerlendirilmeli.
    """

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryEventBus:
    """
    Observer pattern — subscribe/publish, tek process içinde.

    Thread safety notu: Bu sınıf THREAD-SAFE DEĞİL. portfolio_service.py
    ThreadPoolExecutor kullanıyor ama event publish/subscribe akışı
    şu an yalnızca ana thread'den (repository write path) çağrılıyor;
    ThreadPoolExecutor worker'ları yalnızca OKUMA (fiyat çekme) yapıyor,
    event publish etmiyor. Bu varsayım ihlal edilirse (örn. bir worker
    thread'den event publish edilirse) `_subscribers` dict'ine
    concurrent yazma riski oluşur — o senaryo ortaya çıkarsa
    threading.Lock eklenmeli. Şimdiden eklemiyorum çünkü mevcut
    kullanım deseni bunu gerektirmiyor (YAGNI) ve gereksiz lock,
    senkron event dispatch'te ölçülebilir olmayan bir performans
    maliyeti ekler.
    """

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        self._subscribers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Test'lerde izolasyon için — handler yoksa sessizce no-op."""
        handlers = self._subscribers.get(event_name, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """
        Kayıtlı tüm handler'ları SENKRON çağırır.

        Bir handler exception fırlatırsa: loglanır, sonraki handler'lar
        yine de çalıştırılır (bir handler'ın hatası diğerlerini
        engellememeli), ve publish() çağırana hiçbir exception sızmaz.
        """
        for handler in self._subscribers.get(event.name, []):
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 — kasıtlı geniş yakalama
                logger.error(
                    "event_handler_failed",
                    event_name=event.name,
                    handler=getattr(handler, "__name__", repr(handler)),
                    error=str(exc),
                )
