"""
TEFAS yatırım fonu NAV veri adapter'ı.

NAV → OHLCV: Open=High=Low=Close=NAV, Volume=0
Rate limit: 6 req/min (SlidingWindowRateLimiter)
Chunk: maksimum 25 gün/çağrı
Test edilebilirlik: tüm pytefas çağrıları _download_nav() içine izole
"""

from __future__ import annotations

import contextlib
import io
from datetime import date, datetime, timedelta

import pandas as pd

from src.domain.exceptions.domain_exceptions import (
    DataValidationError,
    NoDataError,
    ProviderUnavailableError,
    SymbolNotFoundError,
)
from src.infrastructure.data_providers.base_provider import MarketDataProvider
from src.infrastructure.data_providers.rate_limiter import SlidingWindowRateLimiter
from src.infrastructure.data_providers.retry_policy import (
    RetryExhaustedError,
    RetryPolicy,
    execute_with_retry,
)
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class TefasAdapter(MarketDataProvider):
    """TEFAS (pytefas) üzerinden yatırım fonu NAV verisi sağlayan adapter."""

    def __init__(
        self,
        rate_limit_per_minute: int = 6,
        chunk_size_days: int = 25,
        retry_count: int = 3,
        base_delay_seconds: float = 10.0,
        max_delay_seconds: float = 60.0,
        backoff_factor: float = 2.0,
        min_bars_required: int = 1,
        _rate_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self._chunk_size_days = chunk_size_days
        self._min_bars_required = min_bars_required
        self._retry_policy = RetryPolicy(
            max_attempts=retry_count,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            backoff_factor=backoff_factor,
        )
        self._rate_limiter = _rate_limiter or SlidingWindowRateLimiter(
            max_calls=rate_limit_per_minute,
            window_seconds=60.0,
        )

    def get_provider_name(self) -> str:
        return "tefas"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> pd.DataFrame:
        if timeframe not in ("1d",):
            raise ValueError(
                f"TEFAS yalnızca günlük ('1d') veri sağlar. İstenen: {timeframe!r}"
            )

        fund_code = symbol.upper().strip()
        end_dt = (
            end_date if isinstance(end_date, date) and not isinstance(end_date, datetime)
            else end_date.date() if isinstance(end_date, datetime)
            else date.today()
        )
        start_dt = (
            start_date if isinstance(start_date, date) and not isinstance(start_date, datetime)
            else start_date.date() if isinstance(start_date, datetime)
            else end_dt - timedelta(days=30)
        )

        logger.debug("tefas_fetch_started", fund_code=fund_code,
                     start_date=str(start_dt), end_date=str(end_dt))

        raw_frames = self._fetch_in_chunks(fund_code, start_dt, end_dt)

        if not raw_frames:
            raise SymbolNotFoundError(symbol=fund_code, provider=self.get_provider_name())

        combined = pd.concat(raw_frames).sort_index().drop_duplicates()

        if len(combined) < self._min_bars_required:
            raise NoDataError(symbol=fund_code, provider=self.get_provider_name())

        ohlcv = self._to_ohlcv(combined, fund_code)
        logger.info("tefas_fetch_succeeded", fund_code=fund_code, bar_count=len(ohlcv))
        return ohlcv

    def _fetch_in_chunks(
        self,
        fund_code: str,
        start_dt: date,
        end_dt: date,
    ) -> list[pd.DataFrame]:
        chunks: list[pd.DataFrame] = []
        current = start_dt

        while current <= end_dt:
            chunk_end = min(current + timedelta(days=self._chunk_size_days - 1), end_dt)

            try:
                df = execute_with_retry(
                    func=lambda s=current, e=chunk_end: self._download_nav(fund_code, s, e),
                    policy=self._retry_policy,
                    retryable_exceptions=_RETRYABLE_EXCEPTIONS,
                    operation_name=f"tefas.download({fund_code},{current}:{chunk_end})",
                )
                if df is not None and not df.empty:
                    chunks.append(df)
            except RetryExhaustedError as exc:
                raise ProviderUnavailableError(
                    provider=self.get_provider_name(),
                    reason=f"TEFAS API ulaşılamaz ({fund_code}): {exc.last_exception}",
                    symbol=fund_code,
                ) from exc

            current = chunk_end + timedelta(days=1)

        return chunks

    def _download_nav(
        self,
        fund_code: str,
        start_dt: date,
        end_dt: date,
    ) -> pd.DataFrame | None:
        """Gerçek pytefas çağrısı — testlerde mock'lanır."""
        self._rate_limiter.acquire()

        import pytefas  # noqa: F401 — testlerde mock'lanır

        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            tefas = pytefas.Tefas()
            df = tefas.fetch(
                fund_code,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
            )

        if df is None or df.empty:
            return None

        nav_col = self._detect_nav_column(df)
        if nav_col is None:
            logger.warning("tefas_unknown_column_format",
                           fund_code=fund_code, columns=df.columns.tolist())
            return None

        result = pd.DataFrame({"nav": df[nav_col]})
        result.index = pd.to_datetime(result.index)
        return result

    @staticmethod
    def _detect_nav_column(df: pd.DataFrame) -> str | None:
        candidates = ["price", "nav", "fiyat", "fund_price", "close"]
        for col in candidates:
            if col in df.columns:
                return col
        lower_map = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in lower_map:
                return lower_map[c]
        return None

    @staticmethod
    def _to_ohlcv(nav_df: pd.DataFrame, fund_code: str) -> pd.DataFrame:
        """NAV → Open=High=Low=Close=NAV, Volume=0."""
        nav = nav_df["nav"].astype(float)
        nav = nav[nav > 0]

        if nav.empty:
            raise DataValidationError(
                provider="tefas",
                reason=f"{fund_code} için geçerli NAV değeri bulunamadı (tümü <= 0)",
            )

        df = pd.DataFrame({
            "Open":   nav,
            "High":   nav,
            "Low":    nav,
            "Close":  nav,
            "Volume": 0.0,
        }, index=nav.index)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        return df
