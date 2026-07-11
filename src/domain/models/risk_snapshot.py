"""
RiskSnapshot domain modeli.

Kaynak: BIST_TEFAS_Master_Design_Document.md risk_snapshots DDL'i —
alan alan birebir. Tüm risk metrikleri float (Decimal DEĞİL) — bu
proje boyunca RiskCalculator/ReturnCalculator için zaten gerekçeli
karar: istatistiksel tahminler, muhasebe kaydı değil, float64
hassasiyeti yeterli.

is_stale ALANI — "cache invalidation" DEĞİL, "audit trail" deseni:
  Yeni bir snapshot hesaplandığında ESKİ snapshot SİLİNMEZ, yalnızca
  is_stale=1 işaretlenir (bkz. repository). Bu, Transaction'ın
  reversal deseniyle TUTARLI bir felsefe: geçmiş veri asla silinmez,
  yalnızca "artık geçerli değil" diye işaretlenir — audit/geriye
  dönük analiz için.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from src.domain.enums.var_method import VaRMethod


@dataclass(frozen=True)
class RiskSnapshot:
    portfolio_id: str
    as_of_date: date
    lookback_days: int
    risk_free_rate: float
    portfolio_volatility: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    max_drawdown: float | None = None
    max_drawdown_start: date | None = None
    max_drawdown_end: date | None = None
    current_drawdown: float | None = None
    var_95: float | None = None
    var_99: float | None = None
    cvar_95: float | None = None
    cvar_99: float | None = None
    var_method: VaRMethod = VaRMethod.HISTORICAL
    beta: float | None = None
    alpha: float | None = None
    r_squared: float | None = None
    information_ratio: float | None = None
    tracking_error: float | None = None
    herfindahl_index: float | None = None
    top5_concentration: float | None = None
    benchmark_code: str | None = None
    is_stale: bool = False
    snapshot_id: str | None = None
    computed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError(f"lookback_days pozitif olmalı: {self.lookback_days}")
