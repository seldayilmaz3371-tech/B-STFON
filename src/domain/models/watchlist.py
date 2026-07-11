"""
Watchlist / WatchlistItem domain modelleri.

Kaynak: BIST_TEFAS_Master_Design_Document.md watchlists/watchlist_items
DDL'i — alan alan birebir.

KAPSAM KARARI (bilinçli sınırlama): DDL'de alert_price_low/high/
alert_pct_change alanları var (fiyat alarmı için) ama bu turda
YALNIZCA VERİ TAŞIYICISI olarak modelleniyor — GERÇEK alarm TETİKLEME
mekanizması (periyodik fiyat kontrolü + bildirim) BU TURUN KAPSAMI
DIŞINDA. Gerekçe: Bu, ayrı bir scheduler job'ı + bildirim altyapısı
(email? UI banner? push?) gerektiren, kendi başına büyük bir özellik —
"watchlist'e sembol ekle/çıkar" (CRUD) ile "fiyat alarmı çalıştır"
(aktif izleme) FARKLI olgunluk seviyelerinde işler. Alanlar şemada
hazır tutuluyor ki ileride üzerine inşa etmek (yeni migration
gerektirmeden) mümkün olsun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from src.domain.enums.asset_type import AssetType


@dataclass(frozen=True)
class WatchlistItem:
    watchlist_id: str
    symbol: str
    symbol_type: AssetType
    alert_price_low: Decimal | None = None
    alert_price_high: Decimal | None = None
    alert_pct_change: Decimal | None = None
    notes: str | None = None
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    item_id: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol boş olamaz.")
        if self.alert_price_low is not None and self.alert_price_low < Decimal("0"):
            raise ValueError(f"alert_price_low negatif olamaz: {self.alert_price_low}")
        if self.alert_price_high is not None and self.alert_price_high < Decimal("0"):
            raise ValueError(f"alert_price_high negatif olamaz: {self.alert_price_high}")
        if (
            self.alert_price_low is not None and self.alert_price_high is not None
            and self.alert_price_low > self.alert_price_high
        ):
            raise ValueError(
                f"alert_price_low ({self.alert_price_low}) alert_price_high'tan "
                f"({self.alert_price_high}) büyük olamaz."
            )


@dataclass(frozen=True)
class Watchlist:
    name: str
    portfolio_id: str | None = None  # DDL: opsiyonel, "portföye bağlı" olabilir
    description: str | None = None
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    watchlist_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Watchlist adı boş olamaz.")
