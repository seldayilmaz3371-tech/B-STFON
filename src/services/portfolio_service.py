"""
Portföy durum servisi — concurrent PnL ve maliyet bazı orkestratörü.

Aşama 8 değişiklikleri:
  - Senkron sembol döngüsü → ThreadPoolExecutor ile concurrent piyasa verisi çekme
  - stale_symbols için threading.Lock (thread-safe append)
  - portfolio_repository injection (çoklu portföy desteği)
  - "default" hardcoded ID tamamen kaldırıldı

Concurrency tasarım gerekçesi:
  yfinance HTTP+parse I/O-bound'dur; GIL I/O bekleme sırasında serbest
  bırakılır. ThreadPoolExecutor(max_workers=N) 20 sembol için ~20s seri
  süreyi ~2-3s'ye düşürür. asyncio tercih edilmedi çünkü yfinance'in
  iç bağımlılığı (requests) async değildir; run_in_executor ile hybrid
  yaklaşım Streamlit event loop ile çakışma riski taşır.

  max_workers=min(8, sembol_sayısı): 8 thread yfinance sunucu-taraflı
  rate-limit'i tetiklemeden I/O eşzamanlılığı sağlar. Config-driven
  (constructor injection) — hard-code değil.

Thread safety:
  _stale_lock (threading.Lock): stale_symbols listesine concurrent
  append'ler güvenlidir. Python'da list.append CPython'da görece atomik
  olsa da bu CPython implementation detail'idir — Lock kullanmak
  davranışı hem doğru hem taşınabilir kılar.

Hata toleransı (Aşama 7'den korunur):
  Bir sembolün fiyatı alınamazsa (herhangi bir Exception) o sembol
  stale_symbols'e eklenir, pozisyonun fiyat alanları None kalır,
  diğer semboller hesaplanmaya devam eder.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.domain.calculators.cost_basis_calculator import WAVGCostBasisCalculator
from src.domain.enums.cost_method import CostMethod
from src.domain.enums.currency_code import CurrencyCode
from src.domain.exceptions.domain_exceptions import InsufficientQuantityError
from src.domain.models.portfolio import Portfolio
from src.infrastructure.logging_config import get_logger
from src.services.market_data_service import MarketDataService

if TYPE_CHECKING:
    from src.infrastructure.repositories.sqlite.portfolio_repository import (
        SQLitePortfolioRepository,
    )
    from src.infrastructure.repositories.sqlite.transaction_repository import (
        SQLiteTransactionRepository,
    )

logger = get_logger(__name__)

ZERO = Decimal("0")

# Optimal thread sayısı: yfinance rate-limit'i aşmadan maksimum paralellik.
# 8 genellikle güvenli üst sınır; config-driven injection ile override edilebilir.
_DEFAULT_MAX_WORKERS = 8


# ── Read-Model DTO'ları ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PositionDTO:
    """
    Tek bir hisse pozisyonunun anlık özeti.

    current_price, pnl_amount, pnl_percentage alanları None olabilir:
    piyasa fiyatı çekilemezse (ağ hatası, geçersiz sembol vb.) bu üç
    alan None kalır; maliyet/miktar alanları DB'den hesaplanmış geçerli
    değerleri taşır.
    """
    symbol: str
    total_quantity: Decimal
    average_cost: Decimal
    total_cost_basis: Decimal
    realized_pnl: Decimal
    current_price: Decimal | None
    current_value: Decimal | None
    unrealized_pnl: Decimal | None
    pnl_percentage: Decimal | None


@dataclass(frozen=True)
class PortfolioSummaryDTO:
    """
    Tüm portföyün anlık özeti.

    Yalnızca total_quantity > 0 olan (açık) pozisyonlar dahildir.
    total_current_value ve total_unrealized_pnl yalnızca fiyatı başarıyla
    çekilen pozisyonları içerir — stale pozisyonlar bu toplamları çarpıtmaz.
    """
    portfolio_id: str
    positions: list[PositionDTO]
    position_count: int
    total_cost_basis: Decimal
    total_realized_pnl: Decimal
    total_current_value: Decimal
    total_unrealized_pnl: Decimal
    stale_symbols: list[str]


# ── Service ───────────────────────────────────────────────────────────────────

class PortfolioService:
    """
    Portföy PnL ve pozisyon durumu orkestrasyon servisi.

    Aşama 8'de eklenen:
      - Concurrent fiyat çekme (ThreadPoolExecutor)
      - portfolio_repository injection (çoklu portföy desteği)
      - max_workers config injection
    """

    def __init__(
        self,
        transaction_repo: Any,          # SQLiteTransactionRepository at runtime
        market_data_service: MarketDataService,
        portfolio_repo: Any = None,      # SQLitePortfolioRepository at runtime; opsiyonel
        calculator: WAVGCostBasisCalculator | None = None,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ) -> None:
        """
        Args:
            transaction_repo: Kalıcı işlem deposu.
            market_data_service: Anlık fiyat sağlayıcı.
            portfolio_repo: Portföy listesi için repository (list_portfolios için şart).
                             None geçilirse list_portfolios() çağrısı ValueError fırlatır.
            calculator: Maliyet bazı motoru. None → WAVGCostBasisCalculator().
            max_workers: Concurrent thread üst sınırı (default 8).
        """
        self._tx_repo = transaction_repo
        self._market_svc = market_data_service
        self._portfolio_repo = portfolio_repo
        self._calculator = calculator or WAVGCostBasisCalculator()
        self._max_workers = max_workers

    # ── Public API ────────────────────────────────────────────────────────────

    def list_portfolios(self) -> list[dict[str, str]]:
        """
        Aktif portföylerin id/name çiftlerini döndürür.

        Returns:
            [{"id": "...", "name": "..."}, ...] — UI selectbox için hazır.

        Raises:
            ValueError: portfolio_repo inject edilmemişse.
        """
        if self._portfolio_repo is None:
            raise ValueError(
                "list_portfolios() için portfolio_repo inject edilmemiş. "
                "Container.portfolio_service property'sini kontrol edin."
            )
        portfolios = self._portfolio_repo.list_all(include_inactive=False)
        return [{"id": p.id, "name": p.name} for p in portfolios]

    def get_available_benchmark_codes(self) -> list[dict[str, str]]:
        """
        DÜZELTME (bu turda): create_portfolio formundaki UI selectbox
        listesi ÖNCEDEN hardcoded 3 endeksti (Portfolio.py'daki ESKİ
        listeyle bile SENKRON DEĞİLDİ — 3 AYRI yerde bağımsız listeler
        riski GERÇEKLEŞMİŞTİ). Artık TEK kanonik kaynaktan (Portfolio.
        KNOWN_BENCHMARKS) besleniyor. app.py domain'i DOĞRUDAN import
        EDEMEZ (katman izolasyonu) — bu yüzden burada dict listesi
        olarak (yalnızca str/str, enum/NamedTuple DEĞİL) dışa veriliyor.
        """
        from src.domain.models.portfolio import KNOWN_BENCHMARKS
        return [{"code": b.code, "name": b.name, "description": b.description} for b in KNOWN_BENCHMARKS]

    def get_portfolio(self, portfolio_id: str) -> Any:
        """
        DÜZELTME (bu turda bulundu): list_portfolios() yalnızca dar bir
        {id, name} DTO'su döndürüyordu — hiçbir yerde TAM Portfolio
        nesnesini (benchmark_code, currency, cost_method dahil) geri
        almanın bir yolu yoktu. RiskService UI entegrasyonu için
        seçili portföyün benchmark_code'unu okumak gerekince bu
        boşluk ortaya çıktı.

        Returns:
            Portfolio | None — portföy yoksa None (exception değil,
            çağıran taraf "portföy bulunamadı" durumunu kendi UI
            akışında ele almalı).
        """
        if self._portfolio_repo is None:
            raise ValueError(
                "get_portfolio() için portfolio_repo inject edilmemiş."
            )
        return self._portfolio_repo.get_by_id(portfolio_id)

    def create_portfolio(
        self,
        name: str,
        currency: str = "TRY",
        cost_method: str = "WAVG",
        inception_date: date | None = None,
        benchmark_code: str | None = None,
        description: str | None = None,
    ) -> Portfolio:
        """
        Yeni bir portföy oluşturur.

        API sözleşmesi — BIST_TEFAS_Master_Design_Document.md Bölüm 1.3
        "PortfolioService.create_portfolio" interface contract'ından
        alındı, TEK FARKLA: currency/cost_method dokümante edilen
        imzada enum tipi (CurrencyCode/CostMethod) — burada BİLİNÇLİ
        OLARAK str'e çevrildi.

        GEREKÇE (mimari sınır ihlali önlendi): app.py'ın kendi
        docstring'i "src.domain veya src.infrastructure altından HİÇBİR
        ŞEY import etmez" diyor (katman izolasyonu, bu projede en
        başından beri kesin kural). Eğer bu metod CurrencyCode/CostMethod
        enum'larını parametre olarak zorunlu kılsaydı, UI katmanı
        (app.py) bir portföy oluşturma formu render etmek için
        domain enum'larını import ETMEK ZORUNDA kalırdı — bu, tam olarak
        önlemeye çalıştığımız katman ihlali olurdu. Bunun yerine enum
        dönüşümü BURADA (Service katmanı, Domain'e bağımlı olması
        İZİNLİ — bkz. Bölüm 5.1 bağımlılık matrisi) yapılıyor.

        Raises:
            ValueError: portfolio_repo inject edilmemişse, VEYA
                currency/cost_method geçersiz bir string ise (enum'a
                çevrilemiyorsa) — bu durumda hata mesajı geçerli
                seçenekleri listeler.
            ValidationError: Portfolio.__post_init__ kurallarından biri
                ihlal edilirse (boş isim, gelecek tarihli inception_date,
                desteklenmeyen cost_method, bilinmeyen benchmark_code).
            DuplicateError: Aynı isimde portföy zaten varsa (DB UNIQUE
                constraint, TOCTOU-güvenli — bkz. repository.add()).
        """
        if self._portfolio_repo is None:
            raise ValueError(
                "create_portfolio() için portfolio_repo inject edilmemiş. "
                "Container.portfolio_service property'sini kontrol edin."
            )

        try:
            currency_enum = CurrencyCode(currency)
        except ValueError as exc:
            valid = [c.value for c in CurrencyCode]
            raise ValueError(f"Geçersiz currency: '{currency}'. Geçerli: {valid}") from exc

        try:
            cost_method_enum = CostMethod(cost_method)
        except ValueError as exc:
            valid = [c.value for c in CostMethod]
            raise ValueError(f"Geçersiz cost_method: '{cost_method}'. Geçerli: {valid}") from exc

        portfolio = Portfolio(
            id=str(uuid.uuid4()),
            name=name,
            currency=currency_enum,
            cost_method=cost_method_enum,
            inception_date=inception_date or date.today(),
            benchmark_code=benchmark_code,
            description=description,
        )
        created = self._portfolio_repo.add(portfolio)
        created_portfolio: Portfolio = created  # Any → Portfolio (repo tip
        # imzası Any olarak bırakıldı, çağıranlar TYPE_CHECKING altında
        # gerçek sınıfı görüyor ama runtime'da Any kalıyor — bkz. mevcut
        # transaction_repo/portfolio_repo alanlarının aynı deseni)
        logger.info(
            "portfolio_created_via_service",
            portfolio_id=created_portfolio.id,
            name=created_portfolio.name,
        )
        return created_portfolio

    def get_portfolio_status(self, portfolio_id: str) -> PortfolioSummaryDTO:
        """
        Portföydeki tüm açık pozisyonları concurrent fiyat çekme ile hesapla.

        Seri (Aşama 7): O(N × RTT)
        Concurrent (Aşama 8): O(RTT) — max_workers thread'i ile

        Hata toleransı: Her sembol için hata bağımsız yakalanır.
        Bu metod hiçbir Exception çağırana sızdırmaz.
        """
        symbols = self._tx_repo.get_portfolio_symbols(portfolio_id)

        if not symbols:
            logger.info("portfolio_status_empty", portfolio_id=portfolio_id)
            return PortfolioSummaryDTO(
                portfolio_id=portfolio_id,
                positions=[], position_count=0,
                total_cost_basis=ZERO, total_realized_pnl=ZERO,
                total_current_value=ZERO, total_unrealized_pnl=ZERO,
                stale_symbols=[],
            )

        # ── Adım 1: Her sembol için maliyet bazı hesapla (senkron — DB okuma) ──
        # DB çağrıları zaten hızlı (local SQLite); bunları seri yapmak doğru.
        # Paralelize etmek SQLite'ın WAL modunda bile lock çakışması riski taşır.
        cb_results: dict[str, Any] = {}
        skipped: list[str] = []
        for symbol in symbols:
            transactions = self._tx_repo.get_by_symbol(portfolio_id, symbol)
            if not transactions:
                continue
            try:
                cb_results[symbol] = self._calculator.calculate(transactions)
            except InsufficientQuantityError as exc:
                logger.error(
                    "cost_basis_calculation_failed",
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    error=str(exc),
                )
                skipped.append(symbol)

        if not cb_results:
            return PortfolioSummaryDTO(
                portfolio_id=portfolio_id,
                positions=[], position_count=0,
                total_cost_basis=ZERO, total_realized_pnl=ZERO,
                total_current_value=ZERO, total_unrealized_pnl=ZERO,
                stale_symbols=skipped,
            )

        # ── Adım 2: Fiyat çekme — ThreadPoolExecutor ile concurrent ──────────
        stale_symbols: list[str] = list(skipped)
        stale_lock = threading.Lock()
        price_results: dict[str, tuple] = {}
        price_lock = threading.Lock()

        # Kapatılmış pozisyonları (qty=0) fiyat sorgusundan çıkar
        active_symbols = [
            sym for sym, cbr in cb_results.items()
            if cbr.total_quantity > ZERO
        ]

        actual_workers = min(self._max_workers, len(active_symbols)) if active_symbols else 1

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._fetch_price_safe,
                    symbol,
                    cb_results[symbol],
                    stale_symbols,
                    stale_lock,
                ): symbol
                for symbol in active_symbols
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result_tuple = future.result()
                    with price_lock:
                        price_results[symbol] = result_tuple
                except Exception as exc:
                    # future.result() içinden kaçan beklenmedik hata
                    logger.error(
                        "concurrent_future_unexpected_error",
                        symbol=symbol,
                        error=str(exc),
                    )
                    with stale_lock:
                        stale_symbols.append(symbol)

        # ── Adım 3: PositionDTO'ları oluştur ─────────────────────────────────
        positions: list[PositionDTO] = []
        for symbol, cbr in cb_results.items():
            if cbr.total_quantity <= ZERO:
                continue  # Kapatılmış pozisyon

            price_tuple = price_results.get(symbol, (None, None, None, None))
            current_price, current_value, unrealized_pnl, pnl_pct = price_tuple

            positions.append(PositionDTO(
                symbol=symbol,
                total_quantity=cbr.total_quantity,
                average_cost=cbr.average_cost,
                total_cost_basis=cbr.total_cost_basis,
                realized_pnl=cbr.realized_pnl,
                current_price=current_price,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
                pnl_percentage=pnl_pct,
            ))

        summary = self._aggregate(portfolio_id, positions, stale_symbols)

        logger.info(
            "portfolio_status_computed",
            portfolio_id=portfolio_id,
            position_count=len(positions),
            stale_count=len(stale_symbols),
            total_cost_basis=str(summary.total_cost_basis),
            total_current_value=str(summary.total_current_value),
        )
        return summary

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_price_safe(
        self,
        symbol: str,
        cb_result: Any,
        stale_symbols: list[str],
        stale_lock: threading.Lock,
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None]:
        """
        Thread-safe price fetch. Thread başına çağrılır.

        stale_lock: stale_symbols.append() birden fazla thread'den
        güvenle çağrılabilir.
        """
        try:
            analysis = self._market_svc.get_market_analysis(
                symbol=symbol,
                timeframe="1d",
            )
            latest_close = analysis.latest_close
            if latest_close is None:
                raise ValueError(f"latest_close None: {symbol}")

            price = Decimal(str(latest_close))
            value = cb_result.current_value(price)
            upnl = cb_result.unrealized_pnl(price)
            cost = cb_result.total_cost_basis
            pnl_pct = (
                Decimal(str(round(float(upnl / cost) * 100, 2)))
                if cost > ZERO else ZERO
            )
            return price, value, upnl, pnl_pct

        except Exception as exc:
            logger.warning(
                "market_price_fetch_failed",
                symbol=symbol,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            with stale_lock:
                stale_symbols.append(symbol)
            return None, None, None, None

    @staticmethod
    def _aggregate(
        portfolio_id: str,
        positions: list[PositionDTO],
        stale_symbols: list[str],
    ) -> PortfolioSummaryDTO:
        total_cost = sum((p.total_cost_basis for p in positions), ZERO)
        total_realized = sum((p.realized_pnl for p in positions), ZERO)
        priced = [p for p in positions if p.current_value is not None]
        total_current = sum((p.current_value for p in priced), ZERO)      # type: ignore[arg-type]
        total_unrealized = sum((p.unrealized_pnl for p in priced), ZERO)  # type: ignore[arg-type]

        return PortfolioSummaryDTO(
            portfolio_id=portfolio_id,
            positions=positions,
            position_count=len(positions),
            total_cost_basis=total_cost,
            total_realized_pnl=total_realized,
            total_current_value=total_current,
            total_unrealized_pnl=total_unrealized,
            stale_symbols=stale_symbols,
        )
