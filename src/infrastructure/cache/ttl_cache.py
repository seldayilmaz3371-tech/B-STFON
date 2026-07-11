"""
Minimal thread-safe TTL cache.

Sıfır dış bağımlılık — yalnızca stdlib (threading, time, typing).
cachetools kurulu olmadığı ortamlarda çalışır.

Tasarım kararı:
  MarketDataService'in get_market_analysis() çağrısını önbelleğe almak için
  kullanılır. TTL=60s: BIST'te 15 dakikalık veri gecikmesi kabul görgörken
  1 dakikalık cache yatırımcıyı yanıltmaz; gereksiz yfinance çağrısını önler.

Thread safety:
  threading.Lock ile okuma/yazma operasyonları atomik yapılır.
  Farklı thread'ler aynı anda cache'e güvenle yazabilir/okuyabilir.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Generic, TypeVar

V = TypeVar("V")


class TTLCache:
    """
    (key → value) TTL cache. Süresi dolmuş entry'ler get() sırasında temizlenir.

    Kullanım:
        cache = TTLCache(ttl_seconds=60)
        cache.set("THYAO|1d", result)
        cached = cache.get("THYAO|1d")  # None if expired
    """

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds pozitif olmalı: {ttl_seconds}")
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Geçerli cache değeri veya None (expire/miss)."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        """Değeri TTL süresiyle cache'e yaz."""
        with self._lock:
            self._store[key] = (value, time.monotonic() + self._ttl)

    def invalidate(self, key: str) -> None:
        """Belirli bir key'i cache'den sil."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Tüm cache'i temizle (test ve force-refresh senaryoları için)."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Cache'deki mevcut (henüz expire olmamış dahil) entry sayısı."""
        with self._lock:
            return len(self._store)
