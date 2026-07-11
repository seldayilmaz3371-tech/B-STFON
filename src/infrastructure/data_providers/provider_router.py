"""
Provider Router — BIST ve TEFAS sembollerini doğru adapter'a yönlendirir.

Routing kuralı:
  '.IS' suffix → BIST
  4-5 büyük harf (rakam yok) → BIST  (örn: THYAO, GARAN)
  Diğer tüm formatlar → TEFAS         (örn: AFA, MAC, TI2)

ProviderRouter, MarketDataProvider ABC'sini implemente eder;
MarketDataService routing mantığından habersiz kalır (LSP).
"""

from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

from src.infrastructure.data_providers.base_provider import MarketDataProvider
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

_BIST_PATTERN = re.compile(r"^[A-Z]{4,5}$")


def classify_symbol(symbol: str) -> str:
    """
    Sembolü 'BIST' veya 'TEFAS' olarak sınıflandır.

    Args:
        symbol: Ham sembol (büyük/küçük harf, boşluk olabilir).

    Returns:
        'BIST' veya 'TEFAS'
    """
    normalized = symbol.upper().strip()
    if normalized.endswith(".IS"):
        return "BIST"
    if _BIST_PATTERN.match(normalized):
        return "BIST"
    return "TEFAS"


class ProviderRouter(MarketDataProvider):
    """
    BIST ve TEFAS adapter'larını tek MarketDataProvider arayüzü arkasında birleştirir.

    Kullanım:
        router = ProviderRouter(
            bist_provider=yfinance_adapter,
            tefas_provider=tefas_adapter,
        )
        service = MarketDataService(provider=router, cache=cache)
    """

    def __init__(
        self,
        bist_provider: MarketDataProvider,
        tefas_provider: MarketDataProvider,
    ) -> None:
        self._providers: dict[str, MarketDataProvider] = {
            "BIST": bist_provider,
            "TEFAS": tefas_provider,
        }

    def get_provider_name(self) -> str:
        return "provider_router"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> pd.DataFrame:
        asset_class = classify_symbol(symbol)
        provider = self._providers[asset_class]

        logger.debug(
            "provider_router_dispatch",
            symbol=symbol,
            asset_class=asset_class,
            provider=provider.get_provider_name(),
            timeframe=timeframe,
        )

        return provider.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

    def get_provider_for(self, symbol: str) -> MarketDataProvider:
        """Sembol için kullanılacak provider'ı döndür (test/debug)."""
        return self._providers[classify_symbol(symbol)]
