"""
domain/interfaces paketi — dışa aktarılan sözleşmeler.

repositories.py, data_providers.py, event_bus.py bu turda YAZILMADI
(Faz B'nin kapsamı — henüz somut bir tüketicileri yok, bkz. proje
yol haritası Faz B.1-B.4).
"""

from __future__ import annotations

from src.domain.interfaces.cost_basis_strategy import (
    CostBasisResult,
    CostBasisStrategy,
)

__all__ = ["CostBasisResult", "CostBasisStrategy"]
