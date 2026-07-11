"""
CostBasisStrategy arayüz sözleşmesi ve CostBasisResult DTO'su.

REVİZYON NOTU (ilk taslaktan farkı):
  İlk taslakta CostBasisResult bir `Protocol` idi ve WAVG/FIFO için
  ayrı somut sınıflar (WAVGCostBasisResult/FIFOCostBasisResult)
  planlanmıştı. Mevcut (önceden teslim edilmiş) test_portfolio_service.py
  dosyasını gerçekten çalıştırarak şunu tespit ettim:

    from src.domain.calculators.cost_basis_calculator import (
        CostBasisResult, WAVGCostBasisCalculator,
    )
    ...
    return CostBasisResult(
        total_quantity=..., average_cost=..., total_cost_basis=...,
        realized_pnl=..., total_dividends=ZERO,
    )

  Yani CostBasisResult zaten SOMUT, INSTANTIATE EDİLEBİLİR bir sınıf
  olarak bekleniyor — Protocol değil. Ayrıca `total_dividends` adında
  BENİM İLK TASARIMIMDA OLMAYAN bir alan gerekiyor — bu, DIVIDEND
  işlemlerinin "tamamen no-op" olduğu ilk varsayımımın YANLIŞ
  olduğunu kanıtladı (bkz. cost_basis_calculator.py'daki revize
  gerekçe). Varsayımla ilerlemek yerine, var olan kodu çalıştırıp
  gerçek sözleşmeyi tersine mühendislikle çıkardım ve tasarımı buna
  göre düzelttim.

  Karar: TEK bir somut `CostBasisResult` dataclass, hem WAVG hem FIFO
  tarafından üretiliyor (ayrı alt sınıflar değil). `lots` alanı FIFO'ya
  özgü şeffaflık için var ama varsayılan boş tuple — bu sayede WAVG da
  dahil her yerde tip olarak birebir aynı sınıf kullanılabiliyor
  (Liskov Substitution burada bir disiplin değil, otomatik bir sonuç).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from src.domain.calculators.cost_basis_calculator import TaxLot
    from src.domain.models.transaction import Transaction


@dataclass(frozen=True)
class CostBasisResult:
    """
    WAVG ve FIFO'nun ürettiği TEK ortak sonuç DTO'su.

    total_dividends: DIVIDEND işlemlerinden biriken NET temettü toplamı
      (brüt değil — GD-004'teki "Total return (temettü dahil)"
      hesabı net temettü kullanıyor: 270 TL, 300 TL brüt değil).
      Pozisyon miktarını/maliyetini ETKİLEMEZ, yalnızca bilgi amaçlı
      taşınır; CashLedger'a yazma sorumluluğu bu DTO'nun değil,
      servis katmanının (Faz B/C).

    lots: Yalnızca FIFOCostBasisCalculator tarafından doldurulur.
      WAVG için her zaman boş tuple — WAVG lot bazlı takip yapmaz.
    """

    total_quantity: Decimal
    average_cost: Decimal
    total_cost_basis: Decimal
    realized_pnl: Decimal
    total_dividends: Decimal = Decimal("0")
    lots: tuple["TaxLot", ...] = field(default_factory=tuple)

    def current_value(self, market_price: Decimal) -> Decimal:
        return self.total_quantity * market_price

    def unrealized_pnl(self, market_price: Decimal) -> Decimal:
        return self.current_value(market_price) - self.total_cost_basis

    def total_return(self, market_price: Decimal) -> Decimal:
        """
        Unrealized PnL + gerçekleşmiş kâr/zarar + temettü.

        GD-004 "Total return (temettü dahil)" tanımıyla birebir:
        870 TL = 600 (unrealized) + 270 (dividend); realized_pnl bu
        senaryoda 0 olduğu için örnekte görünmüyor ama genel formül
        realized_pnl'i de kapsamalı — pozisyon kısmen satılmış bir
        portföyde temettü + gerçekleşmiş kâr birlikte değerlendirilmeli.
        """
        return (
            self.unrealized_pnl(market_price)
            + self.realized_pnl
            + self.total_dividends
        )


class CostBasisStrategy(Protocol):
    """
    portfolio_service.py'ın `self._calculator` alanı için hedef tip.

    calculate() HER ZAMAN bir sembolün TÜM işlem geçmişinden SIFIRDAN
    hesaplama yapar (incremental/stateful değil) — bkz. gerekçe
    cost_basis_calculator.py modül docstring'inde.
    """

    def calculate(self, transactions: Sequence["Transaction"]) -> CostBasisResult:
        """
        Raises:
            InsufficientQuantityError: Bir SELL, mevcut pozisyonu aşarsa.
            BusinessRuleError: Pozisyon yokken BONUS_SHARE/SPLIT gelirse,
                veya transactions boşsa.
        """
        ...
