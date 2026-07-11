"""
WatchlistService — TransactionService ile TUTARLI desen (symbol
auto-classification, validasyon orkestrasyonu, katman izolasyonu için
string-tabanlı public API).

KAPSAM: CRUD + on-demand fiyat/alarm durumu görüntüleme (bu turda
eklendi). Arka planda OTOMATİK alarm kontrolü + bildirim (email/push)
BİLİNÇLİ OLARAK KAPSAM DIŞI — bkz. watchlist.py modül docstring'i ve
get_items_with_current_price()'ın kendi gerekçesi.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.domain.exceptions.domain_exceptions import BusinessRuleError
from src.domain.models.watchlist import Watchlist, WatchlistItem
from src.infrastructure.data_providers.provider_router import classify_symbol
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _map_asset_class(symbol: str) -> str:
    """TransactionService._map_asset_class ile AYNI mantık — bkz. o dosyanın gerekçesi."""
    return "BIST_STOCK" if classify_symbol(symbol) == "BIST" else "TEFAS_FUND"


@dataclass(frozen=True)
class WatchlistItemStatus:
    """
    get_items_with_current_price()'ın döndürdüğü, fiyat + alarm
    durumunu birleştiren read-model (Domain modeli DEĞİL —
    MarketAnalysisResult ile AYNI felsefe: UI'a sunulmaya hazır bir
    DTO, kalıcı bir varlık değil).
    """

    item: WatchlistItem
    current_price: Decimal | None  # None = fiyat alınamadı (ağ hatası vb.)
    alarm_triggered: bool
    alarm_direction: str | None  # "low" | "high" | None
    fetch_error: str | None = None


class WatchlistService:
    def __init__(self, watchlist_repo: Any, market_data_service: Any | None = None) -> None:
        self._repo = watchlist_repo
        self._market_data_service = market_data_service

    def create_watchlist(
        self, name: str, description: str | None = None, portfolio_id: str | None = None,
    ) -> Watchlist:
        """
        Raises:
            BusinessRuleError: name boşsa.
            DuplicateError: Aynı isimde watchlist zaten varsa.
        """
        if not name or not name.strip():
            raise BusinessRuleError("Watchlist adı boş olamaz.")
        watchlist = Watchlist(
            name=name.strip(), description=description.strip() if description else None,
            portfolio_id=portfolio_id,
        )
        result: Watchlist = self._repo.create_watchlist(watchlist)
        return result

    def list_watchlists(self, include_inactive: bool = False) -> list[Watchlist]:
        result: list[Watchlist] = self._repo.list_watchlists(include_inactive)
        return result

    def add_symbol(
        self, watchlist_id: str, symbol: str,
        alert_price_low: Decimal | None = None, alert_price_high: Decimal | None = None,
        alert_pct_change: Decimal | None = None, notes: str | None = None,
    ) -> WatchlistItem:
        """
        Raises:
            BusinessRuleError: symbol boşsa VEYA fiyat alarmı validasyonu
                başarısızsa (bkz. WatchlistItem.__post_init__).
            DuplicateError: Bu watchlist'te bu sembol zaten varsa.
        """
        if not symbol or not symbol.strip():
            raise BusinessRuleError("Sembol boş olamaz.")

        from src.domain.enums.asset_type import AssetType
        normalized = symbol.strip().upper()
        try:
            item = WatchlistItem(
                watchlist_id=watchlist_id, symbol=normalized,
                symbol_type=AssetType(_map_asset_class(normalized)),
                alert_price_low=alert_price_low, alert_price_high=alert_price_high,
                alert_pct_change=alert_pct_change, notes=notes.strip() if notes else None,
            )
        except ValueError as exc:
            raise BusinessRuleError(str(exc)) from exc

        result: WatchlistItem = self._repo.add_item(item)
        return result

    def remove_symbol(self, item_id: str) -> None:
        """Raises: NotFoundError: item_id yoksa."""
        self._repo.remove_item(item_id)

    def list_symbols(self, watchlist_id: str) -> list[WatchlistItem]:
        result: list[WatchlistItem] = self._repo.list_items(watchlist_id)
        return result

    def get_items_with_current_price(self, watchlist_id: str) -> list[WatchlistItemStatus]:
        """
        DÜZELTME (bu turda bulundu): Kullanıcının girdiği alarm eşikleri
        (alert_price_low/high) HİÇBİR YERDE kontrol edilmiyordu —
        "veri toplanıyor ama hiç kullanılmıyor" deseninin YENİ bir
        örneği olma riski taşıyordu (CashLedgerRepository/PriceRepository/
        verify_balance()'da bulunanlarla AYNI kategori).

        BİLİNÇLİ KAPSAM SINIRI: Bu, ON-DEMAND bir görüntüleme —
        kullanıcı bu metodu (UI'da "Fiyatları Güncelle" butonuyla)
        AÇIKÇA tetiklediğinde çalışır, arka planda OTOMATİK
        çalışmaz/bildirim GÖNDERMEZ. Tam bir "alarm sistemi" (periyodik
        kontrol + email/push bildirimi) AYRI ve BÜYÜK bir özellik —
        bu, yalnızca "kullanıcı baktığında anlamlı bir özet görsün"
        sorununu çözüyor.

        Hata izolasyonu: Bir sembolün fiyatı alınamazsa (ağ hatası vb.),
        DİĞER sembollerin durumu etkilenmez — fetch_error alanında
        taşınır, exception FIRLATILMAZ (UI'ın TEK bir başarısız sembol
        yüzünden TÜM listeyi gösterememesi kabul edilemez).
        """
        items = self.list_symbols(watchlist_id)
        if self._market_data_service is None:
            return [
                WatchlistItemStatus(
                    item=item, current_price=None, alarm_triggered=False,
                    alarm_direction=None, fetch_error="market_data_service inject edilmemiş.",
                )
                for item in items
            ]

        statuses = []
        for item in items:
            try:
                analysis = self._market_data_service.get_market_analysis(item.symbol, "1d")
                price = Decimal(str(analysis.latest_close)) if analysis.latest_close is not None else None
            except Exception as exc:
                logger.warning(
                    "watchlist_price_fetch_failed", symbol=item.symbol, error=str(exc),
                )
                statuses.append(WatchlistItemStatus(
                    item=item, current_price=None, alarm_triggered=False,
                    alarm_direction=None, fetch_error=str(exc),
                ))
                continue

            alarm_triggered = False
            alarm_direction: str | None = None
            if price is not None:
                if item.alert_price_low is not None and price <= item.alert_price_low:
                    alarm_triggered, alarm_direction = True, "low"
                elif item.alert_price_high is not None and price >= item.alert_price_high:
                    alarm_triggered, alarm_direction = True, "high"

            statuses.append(WatchlistItemStatus(
                item=item, current_price=price, alarm_triggered=alarm_triggered,
                alarm_direction=alarm_direction,
            ))
        return statuses
