"""Para birimi enum'u — portfolios.currency DDL CHECK ile senkron (TRY/USD/EUR)."""

from __future__ import annotations

from enum import Enum


class CurrencyCode(str, Enum):
    TRY = "TRY"
    USD = "USD"
    EUR = "EUR"
