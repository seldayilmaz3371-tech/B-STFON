"""Nakit ledger giriş tipi — DDL cash_ledger_entries.entry_type CHECK ile senkron."""

from __future__ import annotations

from enum import Enum


class LedgerEntryType(str, Enum):
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"
