"""
BacktestEngine — design doc Bölüm 3.10/4.7 "BacktestEngine.run(signals,
price_data, params)" ile uyumlu.

MİMARİ KARAR (gerekçe TEKRARLANIYOR, bkz. backtest_strategy.py):
  Bu engine, gün-gün (event-driven) bir döngüde simüle edilmiş
  Transaction'lar üretir ve bunları GERÇEK WAVGCostBasisCalculator'a
  besler — backtest'in realized_pnl/cost_basis hesaplaması, CANLI
  portföy muhasebesiyle AYNI KOD YOLUNU kullanır (ayrı, muhtemelen
  SAPAN bir vectorized hesaplama YERİNE).

KAPSAM SINIRLARI (bilinçli, açıkça işaretleniyor):
  - Yalnızca TEK SEMBOL desteklenir (çoklu sembol/portföy-seviyeli
    backtest, position sizing arası etkileşim gerektirir — AYRI ve
    daha büyük bir problem, bu turun kapsamı DIŞINDA).
  - Slippage modeli YOK — yalnızca DÜZ ORANLI komisyon (commission_rate)
    simüle ediliyor. Gerçek BIST bid-ask spread modeli (design doc'ta
    bahsedilen) AYRI bir araştırma/veri gerektiriyor.
  - Survivorship bias uyarısı YOK — "delisted sembol listesi" hiçbir
    dosyada yok, uydurulmadı.
  - Position sizing: TEK bir parametre (position_size_pct) — her AL
    sinyalinde mevcut nakdin bu yüzdesi kullanılır. Kademeli/karmaşık
    position sizing stratejileri (Kelly criterion vb.) desteklenmiyor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, cast

import numpy as np
import pandas as pd

from src.domain.calculators.cost_basis_calculator import WAVGCostBasisCalculator
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.calculators.risk_calculator import RiskCalculator
from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import BusinessRuleError, InsufficientDataError
from src.domain.models.transaction import Transaction


@dataclass(frozen=True)
class BacktestTrade:
    trade_date: date
    transaction_type: TransactionType
    quantity: Decimal
    price: Decimal
    commission: Decimal


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    start_date: date
    end_date: date
    initial_capital: Decimal
    final_value: Decimal
    total_return: Decimal
    annualized_return: float | None
    sharpe_ratio: float | None
    max_drawdown: float | None
    realized_pnl: Decimal
    total_trades: int
    portfolio_value_series: pd.Series = field(repr=False)
    trades: tuple[BacktestTrade, ...] = field(default_factory=tuple, repr=False)


class BacktestEngine:
    def __init__(
        self,
        return_calculator: ReturnCalculator,
        risk_calculator: RiskCalculator,
        trading_days: int = 252,
        risk_free_rate_annual: float = 0.0,
    ) -> None:
        self._return_calc = return_calculator
        self._risk_calc = risk_calculator
        self._trading_days = trading_days
        self._risk_free_rate = risk_free_rate_annual

    def run(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: Decimal,
        commission_rate: Decimal = Decimal("0.001"),
        position_size_pct: Decimal = Decimal("1.0"),
    ) -> BacktestResult:
        """
        Raises:
            BusinessRuleError: price_data boşsa, signals ile index
                uyuşmuyorsa, veya position_size_pct (0, 1] aralığında
                değilse.
        """
        if price_data.empty:
            raise BusinessRuleError("price_data boş olamaz.")
        if not price_data.index.equals(signals.index):
            raise BusinessRuleError("price_data ve signals AYNI date index'e sahip olmalı.")
        if not (Decimal("0") < position_size_pct <= Decimal("1")):
            raise BusinessRuleError(
                f"position_size_pct (0, 1] aralığında olmalı, alınan: {position_size_pct}"
            )

        cash = initial_capital
        quantity = Decimal("0")
        trades: list[BacktestTrade] = []
        simulated_transactions: list[Transaction] = []
        portfolio_values: list[Decimal] = []

        for idx, sig in signals.items():
            # DÜZELTME (bu turda, GERÇEK pandas-stubs kurulumu sonrası
            # bulundu): signals.items()'ın idx tipi mypy için genel
            # Hashable — ama run()'ın başında price_data.index.equals(
            # signals.index) DOĞRULANDI (bkz. yukarıdaki kontrol), bu
            # yüzden idx'in GERÇEKTEN price_data'nın (DatetimeIndex)
            # bir üyesi olduğu ÇALIŞMA ZAMANINDA GARANTİ. Açık cast
            # GÜVENLİ, tip hatası GİZLEMİYOR.
            ts_idx = cast(pd.Timestamp, idx)
            price = Decimal(str(price_data.loc[ts_idx, "close"]))
            trade_date = pd.Timestamp(ts_idx).date()

            if sig == 1 and quantity == Decimal("0") and cash > Decimal("0"):
                # AL — mevcut nakdin position_size_pct'i kullanılır
                available = cash * position_size_pct
                buy_quantity = (available / (price * (Decimal("1") + commission_rate))).quantize(
                    Decimal("0.0001")
                )
                if buy_quantity > Decimal("0"):
                    commission = buy_quantity * price * commission_rate
                    cash -= buy_quantity * price + commission
                    quantity += buy_quantity
                    trades.append(BacktestTrade(trade_date, TransactionType.BUY, buy_quantity, price, commission))
                    simulated_transactions.append(Transaction(
                        symbol=symbol, transaction_type=TransactionType.BUY,
                        timestamp=datetime.combine(trade_date, datetime.min.time()),
                        quantity=buy_quantity, price=price,
                    ))

            elif sig == -1 and quantity > Decimal("0"):
                # SAT — TÜM pozisyon kapatılır (kademeli satış desteklenmiyor)
                commission = quantity * price * commission_rate
                cash += quantity * price - commission
                trades.append(BacktestTrade(trade_date, TransactionType.SELL, quantity, price, commission))
                simulated_transactions.append(Transaction(
                    symbol=symbol, transaction_type=TransactionType.SELL,
                    timestamp=datetime.combine(trade_date, datetime.min.time()),
                    quantity=quantity, price=price,
                ))
                quantity = Decimal("0")

            portfolio_values.append(cash + quantity * price)

        portfolio_value_series = pd.Series(
            [float(v) for v in portfolio_values], index=price_data.index, name="portfolio_value",
        )

        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # KRİTİK: realized_pnl, GERÇEK WAVGCostBasisCalculator ÜZERİNDEN
        # hesaplanıyor — bkz. modül docstring'indeki mimari gerekçe.
        realized_pnl = Decimal("0")
        if simulated_transactions:
            try:
                cb_result = WAVGCostBasisCalculator().calculate(simulated_transactions)
                realized_pnl = cb_result.realized_pnl
            except BusinessRuleError:
                pass  # açık pozisyon kapatılmadan bitmiş olabilir — realized_pnl kısmi kalır

        daily_returns = portfolio_value_series.pct_change().dropna().to_numpy()
        annualized_return: float | None = None
        sharpe_ratio: float | None = None
        max_drawdown: float | None = None
        if len(daily_returns) > 0 and np.std(daily_returns) > 1e-12:
            try:
                annualized_return = float(self._return_calc.calculate_annualized_return(
                    Decimal(str(total_return)), len(price_data),
                ))
            except Exception:
                pass
            try:
                sharpe_ratio = self._risk_calc.calculate_sharpe_ratio(
                    daily_returns, self._risk_free_rate, self._trading_days,
                )
            except (InsufficientDataError, Exception):
                pass
            try:
                cumulative = (1.0 + pd.Series(daily_returns)).cumprod().to_numpy()
                max_drawdown = self._risk_calc.calculate_max_drawdown(cumulative).max_drawdown
            except Exception:
                pass

        return BacktestResult(
            symbol=symbol,
            start_date=pd.Timestamp(price_data.index[0]).date(),
            end_date=pd.Timestamp(price_data.index[-1]).date(),
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            realized_pnl=realized_pnl,
            total_trades=len(trades),
            portfolio_value_series=portfolio_value_series,
            trades=tuple(trades),
        )
