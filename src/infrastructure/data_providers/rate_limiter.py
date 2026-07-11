"""
TEFAS API hız sınırlayıcı — Sliding Window Rate Limiter.

6 req/min kısıtı, thread-safe, sıfır dış bağımlılık.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(
        self,
        max_calls: int = 6,
        window_seconds: float = 60.0,
        sleep_fn: callable = time.sleep,
    ) -> None:
        if max_calls <= 0:
            raise ValueError(f"max_calls pozitif olmalı: {max_calls}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds pozitif olmalı: {window_seconds}")
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._sleep_fn = sleep_fn
        self._call_timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._call_timestamps and (now - self._call_timestamps[0]) >= self._window_seconds:
                self._call_timestamps.popleft()

            if len(self._call_timestamps) >= self._max_calls:
                oldest = self._call_timestamps[0]
                wait_secs = (oldest + self._window_seconds) - now
                if wait_secs > 0:
                    logger.warning(
                        "tefas_rate_limit_wait",
                        wait_seconds=round(wait_secs, 2),
                        current_calls=len(self._call_timestamps),
                        max_calls=self._max_calls,
                    )
                    self._sleep_fn(wait_secs)
                    now = time.monotonic()
                    while self._call_timestamps and (now - self._call_timestamps[0]) >= self._window_seconds:
                        self._call_timestamps.popleft()

            self._call_timestamps.append(time.monotonic())

    @property
    def current_call_count(self) -> int:
        with self._lock:
            now = time.monotonic()
            return sum(1 for t in self._call_timestamps if (now - t) < self._window_seconds)

    def reset(self) -> None:
        with self._lock:
            self._call_timestamps.clear()
