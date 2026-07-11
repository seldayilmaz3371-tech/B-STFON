"""VaR hesaplama yöntemi — RiskCalculator.calculate_var() için."""

from __future__ import annotations

from enum import Enum


class VaRMethod(str, Enum):
    HISTORICAL = "HISTORICAL"    # Percentile tabanlı, dağılım varsayımı yok
    PARAMETRIC = "PARAMETRIC"    # Normal dağılım varsayımı (mean, std)
    MONTECARLO = "MONTECARLO"    # Bootstrap resampling (dağılım varsayımı yok,
                                  # yalnızca "geçmiş gelecekte de geçerli" varsayımı)
