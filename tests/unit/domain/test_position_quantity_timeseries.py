"""compute_quantity_timeseries / quantity_on_date testleri."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.domain.calculators.position_quantity_timeseries import (
    compute_quantity_timeseries,
    quantity_on_date,
)
from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import BusinessRuleError
from src.domain.models.transaction import Transaction


def tx(ttype, day, quantity="0", price="0", split_ratio=None, net_amount=None):
    return Transaction(
        symbol="THYAO", transaction_type=ttype,
        timestamp=datetime.fromisoformat(day),
        quantity=Decimal(quantity), price=Decimal(price),
        split_ratio=Decimal(split_ratio) if split_ratio else None,
        net_amount=Decimal(net_amount) if net_amount is not None else None,
    )


def test_empty_transactions_returns_empty_series():
    result = compute_quantity_timeseries([])
    assert result.empty


def test_simple_buy_sell_step_function():
    transactions = [
        tx(TransactionType.BUY, "2024-01-02", "100"),
        tx(TransactionType.BUY, "2024-02-01", "50"),
        tx(TransactionType.SELL, "2024-03-01", "80"),
    ]
    series = compute_quantity_timeseries(transactions)
    assert list(series.values) == [Decimal("100"), Decimal("150"), Decimal("70")]


def test_bonus_share_and_split_affect_quantity():
    transactions = [
        tx(TransactionType.BUY, "2024-01-02", "100"),
        tx(TransactionType.BONUS_SHARE, "2024-02-01", "50"),
        tx(TransactionType.SPLIT, "2024-03-01", split_ratio="2"),
    ]
    series = compute_quantity_timeseries(transactions)
    assert list(series.values) == [Decimal("100"), Decimal("150"), Decimal("300")]


def test_cash_only_types_do_not_appear_in_series():
    transactions = [
        tx(TransactionType.BUY, "2024-01-02", "100"),
        tx(TransactionType.DIVIDEND, "2024-02-01", "100", "1.5", net_amount="135.00"),
        tx(TransactionType.SELL, "2024-03-01", "50"),
    ]
    series = compute_quantity_timeseries(transactions)
    # DIVIDEND serüde bir "nokta" olarak GÖRÜNMEZ (quantity değişmiyor)
    assert len(series) == 2
    assert list(series.values) == [Decimal("100"), Decimal("50")]


@pytest.mark.parametrize(
    "ttype", [TransactionType.RIGHTS_SOLD, TransactionType.MERGER]
)
def test_unsupported_types_raise(ttype):
    """DÜZELTME: REVERSE_SPLIT ve RIGHTS_USED bu turlarda listeden ÇIKARILDI — artık desteklenen tipler."""
    transactions = [tx(TransactionType.BUY, "2024-01-02", "100"), tx(ttype, "2024-02-01", "10")]
    with pytest.raises(BusinessRuleError):
        compute_quantity_timeseries(transactions)


def test_rights_used_increases_quantity_like_buy():
    """DÜZELTME (bu turda eklendi): RIGHTS_USED, BUY gibi miktarı artırmalı."""
    transactions = [
        tx(TransactionType.BUY, "2024-01-02", "100"),
        tx(TransactionType.RIGHTS_USED, "2024-02-01", "50", "1.00"),
    ]
    series = compute_quantity_timeseries(transactions)
    assert series.iloc[-1] == Decimal("150")


def test_reverse_split_reduces_quantity():
    """DÜZELTME (bu turda eklendi): 1:10 ters bölünme, miktarı 100'den 10'a düşürmeli."""
    transactions = [
        tx(TransactionType.BUY, "2024-01-02", "100"),
        tx(TransactionType.REVERSE_SPLIT, "2024-03-15", split_ratio="10"),
    ]
    series = compute_quantity_timeseries(transactions)
    assert series.iloc[-1] == Decimal("10")


def test_bonus_share_without_position_raises():
    with pytest.raises(BusinessRuleError):
        compute_quantity_timeseries([tx(TransactionType.BONUS_SHARE, "2024-01-02", "50")])


def test_split_without_position_raises():
    with pytest.raises(BusinessRuleError):
        compute_quantity_timeseries([tx(TransactionType.SPLIT, "2024-01-02", split_ratio="2")])


# ── quantity_on_date ─────────────────────────────────────────────────────────

def test_quantity_on_date_before_first_transaction_is_zero():
    series = compute_quantity_timeseries([tx(TransactionType.BUY, "2024-03-01", "100")])
    result = quantity_on_date(series, pd.Timestamp("2024-01-01"))
    assert result == Decimal("0")


def test_quantity_on_date_forward_fills_between_transactions():
    transactions = [
        tx(TransactionType.BUY, "2024-01-02", "100"),
        tx(TransactionType.SELL, "2024-03-01", "40"),
    ]
    series = compute_quantity_timeseries(transactions)
    # 2024-02-01: son işlem 2024-01-02 (100), henüz satış olmamış
    assert quantity_on_date(series, pd.Timestamp("2024-02-01")) == Decimal("100")
    # 2024-04-01: son işlem 2024-03-01 sonrası (60)
    assert quantity_on_date(series, pd.Timestamp("2024-04-01")) == Decimal("60")
    # tam işlem gününde
    assert quantity_on_date(series, pd.Timestamp("2024-01-02")) == Decimal("100")


def test_quantity_on_date_empty_series_is_zero():
    result = quantity_on_date(compute_quantity_timeseries([]), pd.Timestamp("2024-01-01"))
    assert result == Decimal("0")
