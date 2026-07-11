"""
BacktestEngine testleri.

En kritik doğrulama: "buy and hold" stratejisinin ürettiği total_return,
BASİT fiyat artışıyla (komisyon/miktar yuvarlaması toleransında)
EŞLEŞMELİ — bu, motorun matematiksel olarak doğru çalıştığının
GD-tarzı bir kanıtı (golden dataset değil ama aynı doğrulama felsefesi:
bağımsız hesaplanan bir referans değerle karşılaştırma).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.domain.calculators.backtest_engine import BacktestEngine
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.exceptions.domain_exceptions import BusinessRuleError
from src.domain.strategies.sma_crossover_strategy import SMACrossoverStrategy


def _make_price_df(prices: list[float], n_days: int | None = None) -> pd.DataFrame:
    n = n_days or len(prices)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": prices[:n]}, index=dates)


class BuyAndHoldStrategy:
    """Yalnızca İLK gün AL sinyali üretir, HİÇBİR ZAMAN SAT vermez."""

    def generate_signals(self, price_data: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        signals = pd.Series(0, index=price_data.index)
        signals.iloc[0] = 1
        return signals


@pytest.fixture()
def engine():
    return BacktestEngine(
        return_calculator=ReturnCalculator(), risk_calculator=RiskCalculator(min_data_points=5),
    )


# ── Matematiksel doğrulama: buy-and-hold ────────────────────────────────────

def test_buy_and_hold_matches_simple_price_appreciation(engine):
    """
    KRİTİK: Komisyon SIFIR iken, buy-and-hold'un total_return'ü,
    (final_price/initial_price - 1) ile NEREDEYSE TAM eşleşmeli
    (yalnızca miktar yuvarlamasından kaynaklanan ihmal edilebilir fark).
    """
    prices = [100.0] * 1 + list(100 + np.cumsum(np.random.default_rng(1).normal(0.1, 1, 99)))
    df = _make_price_df(prices, 100)
    signals = BuyAndHoldStrategy().generate_signals(df, {})

    result = engine.run(
        symbol="TEST", price_data=df, signals=signals,
        initial_capital=Decimal("10000"), commission_rate=Decimal("0"),
    )

    expected_return = Decimal(str(prices[99] / prices[0] - 1))
    assert abs(result.total_return - expected_return) < Decimal("0.01")  # %1 tolerans (yuvarlama)


def test_buy_and_hold_with_commission_reduces_return(engine):
    """Komisyon eklenince getiri, komisyonsuz senaryodan DAHA DÜŞÜK olmalı."""
    prices = list(100 + np.cumsum(np.random.default_rng(2).normal(0.1, 1, 100)))
    df = _make_price_df(prices, 100)
    signals = BuyAndHoldStrategy().generate_signals(df, {})

    no_commission = engine.run("TEST", df, signals, Decimal("10000"), commission_rate=Decimal("0"))
    with_commission = engine.run("TEST", df, signals, Decimal("10000"), commission_rate=Decimal("0.01"))

    assert with_commission.total_return < no_commission.total_return


def test_realized_pnl_matches_manual_calculation_for_simple_buy_sell(engine):
    """
    Basit senaryo: gün 0'da AL, gün 5'te SAT (SMA crossover TETİKLEMEDEN,
    doğrudan bir strateji ile) — realized_pnl'in WAVGCostBasisCalculator
    üzerinden DOĞRU hesaplandığını kanıtlar.
    """
    class BuyThenSellStrategy:
        def generate_signals(self, price_data, params):
            signals = pd.Series(0, index=price_data.index)
            signals.iloc[0] = 1
            signals.iloc[5] = -1
            return signals

    prices = [100.0, 101, 102, 103, 104, 110.0] + [110.0] * 94
    df = _make_price_df(prices, 100)
    signals = BuyThenSellStrategy().generate_signals(df, {})

    result = engine.run("TEST", df, signals, Decimal("10000"), commission_rate=Decimal("0"))

    # 10000 / 100 = 100 adet alındı, 110'dan satıldı -> realized_pnl = 100 * (110-100) = 1000
    assert abs(result.realized_pnl - Decimal("1000")) < Decimal("1")
    assert result.total_trades == 2


# ── Look-ahead bias kontrolü ─────────────────────────────────────────────────

def test_sma_crossover_signal_does_not_use_future_data():
    """
    KRİTİK doğrulama: Gün T'deki sinyal, T'DEN SONRAKİ fiyatları
    DEĞİŞTİRSEK BİLE AYNI KALMALI — bu, .shift(1) korumasının
    GERÇEKTEN çalıştığının kanıtı.
    """
    np.random.seed(5)
    base_prices = list(100 + np.cumsum(np.random.normal(0.1, 2, 100)))
    df1 = _make_price_df(base_prices, 100)

    # Son 20 günü TAMAMEN farklı (çok yüksek) değerlerle değiştir
    modified_prices = base_prices[:80] + [500.0] * 20
    df2 = _make_price_df(modified_prices, 100)

    strategy = SMACrossoverStrategy()
    signals1 = strategy.generate_signals(df1, {"fast_window": 5, "slow_window": 20})
    signals2 = strategy.generate_signals(df2, {"fast_window": 5, "slow_window": 20})

    # İlk 79 GÜNÜN sinyalleri AYNI olmalı (80. günden sonraki değişiklik
    # bu günleri ETKİLEMEMELİ — look-ahead bias YOKSA).
    assert (signals1.iloc[:79] == signals2.iloc[:79]).all(), (
        "LOOK-AHEAD BIAS TESPİT EDİLDİ — geçmiş sinyaller gelecekteki "
        "fiyat değişikliğinden ETKİLENDİ."
    )


def test_sma_crossover_runs_through_engine_without_crashing(engine):
    np.random.seed(9)
    prices = list(100 + np.cumsum(np.random.normal(0.1, 2, 150)))
    df = _make_price_df(prices, 150)
    strategy = SMACrossoverStrategy()
    signals = strategy.generate_signals(df, {"fast_window": 10, "slow_window": 30})

    result = engine.run("TEST", df, signals, Decimal("10000"))
    assert result.total_trades >= 0
    assert result.final_value > Decimal("0")


def test_sma_crossover_invalid_windows_raises():
    strategy = SMACrossoverStrategy()
    df = _make_price_df([100.0] * 50, 50)
    with pytest.raises(ValueError):
        strategy.generate_signals(df, {"fast_window": 50, "slow_window": 10})


# ── Girdi doğrulama ──────────────────────────────────────────────────────────

def test_empty_price_data_raises(engine):
    df = pd.DataFrame({"close": []})
    signals = pd.Series([], dtype=int)
    with pytest.raises(BusinessRuleError):
        engine.run("TEST", df, signals, Decimal("10000"))


def test_mismatched_index_raises(engine):
    df = _make_price_df([100.0] * 10, 10)
    signals = pd.Series([0] * 5, index=pd.date_range("2024-01-01", periods=5))
    with pytest.raises(BusinessRuleError):
        engine.run("TEST", df, signals, Decimal("10000"))


def test_invalid_position_size_pct_raises(engine):
    df = _make_price_df([100.0] * 10, 10)
    signals = pd.Series(0, index=df.index)
    with pytest.raises(BusinessRuleError):
        engine.run("TEST", df, signals, Decimal("10000"), position_size_pct=Decimal("1.5"))


def test_no_trades_when_no_buy_signal(engine):
    df = _make_price_df([100.0, 101.0, 102.0], 3)
    signals = pd.Series(0, index=df.index)  # hiç sinyal yok
    result = engine.run("TEST", df, signals, Decimal("10000"))
    assert result.total_trades == 0
    assert result.final_value == Decimal("10000")  # hiç işlem yok, nakit sabit
