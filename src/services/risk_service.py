"""
RiskService — portföy risk metrikleri ve benchmark-göreli performans
orkestrasyonu.

KAPSAM KARARI (design doc'tan BİLİNÇLİ SAPMA, açıkça gerekçeli):
  Design doc Bölüm 1.3'te RiskService + ayrı bir BenchmarkService
  tanımlı, toplam 7 metod (compute_risk_snapshot, get_latest_snapshot,
  get_rolling_volatility, get_drawdown_series, run_stress_test,
  get_benchmark_returns, calculate_relative_performance,
  get_available_benchmarks).

  Bu turda implemente edilenler:
    - compute_risk_profile() (compute_risk_snapshot'ın PERSISTENCE'SIZ
      hali — RiskSnapshot entity/repository henüz yok, Faz E kapsamı)
    - calculate_relative_performance()
    - get_rolling_volatility() / get_drawdown_series() (SONRADAN
      eklendi — bkz. aşağıdaki DÜZELTME notu)

  AÇIKÇA ERTELENEN (gerekçeli):
    - run_stress_test(): BIST_2018_CRISIS / BIST_2021_SELLOFF için
      GERÇEK tarihsel kriz verisi gerekiyor. Bu veri hiçbir dosyada
      yok — UYDURULMADI. Gerçek veri kaynağı bulunmadan bu metod
      YANLIŞ risk figürleri üretir, yazılmadı.
    - get_available_benchmarks(): "Sabit liste: XU100, XU030, XUTUM,
      KATLM, XBANK, XHOLD..." — bu liste hiçbir dosyada TAM olarak yok.
      Portfolio.py'daki _KNOWN_BENCHMARK_CODES (yalnızca 3 endeks) ile
      TUTARSIZ. Bu iki listeyi BİRLEŞTİRMEK yerine burada AÇIKÇA
      İŞARETLİYORUM — ayrı bir temizlik turu gerektiriyor.

  DÜZELTME (bu turda): get_rolling_volatility()/get_drawdown_series()
  için "HER pencere için return-series reconstruction'ı tekrarlamak
  gerekiyor" gerekçesiyle ERTELENMİŞTİ — bu VARSAYIM YANLIŞTI, hiç
  DOĞRULANMADAN yazılmıştı. Gerçekte _build_portfolio_value_series()
  ZATEN TAM value serisini TEK SEFERDE döndürüyor — rolling volatilite
  ve drawdown serisi, bu TEK seri üzerinde pandas'ın VECTORIZED
  .rolling()/np.maximum.accumulate() işlemleriyle EK bir network/DB
  maliyeti OLMADAN hesaplanabiliyor. "Ayrı bir performans optimizasyonu
  turu gerektiriyor" iddiası da YANLIŞTI — gerçek performans maliyeti
  yalnızca compute_risk_profile() ile AYNI (tek value-series inşası).

MİMARİ KARAR — Dönüş serisi inşası (en kritik tasarım kararı):
  Portföyün gün-gün değerini hesaplamak için (a) her sembolün miktar
  zaman serisini (position_quantity_timeseries.py — tek geçiş, O(N_işlem))
  VE (b) her sembolün TEK bir OHLCV fetch'ini (O(N_sembol) network
  çağrısı, N_gün DEĞİL) birleştiriyoruz. Bu, N_gün × N_sembol yeniden
  hesaplama yapan naif yaklaşımdan kesin olarak daha iyi.

  NAKİT DAHİL (varsayılan, include_cash=True): Design doc'un
  get_portfolio_value(..., include_cash=True) varsayılanıyla tutarlı.
  METODOLOJİK NOT: Nakit dahil edilmesi volatiliteyi "seyreltir" (nakit
  ~sıfır volatilite taşır) — büyük atıl nakit pozisyonu olan portföylerde
  risk metrikleri OLDUĞUNDAN DÜŞÜK görünebilir. include_cash=False
  seçeneği de sağlanıyor, çağıran taraf ihtiyaca göre seçebilir.

CACHE STRATEJİSİ: Bu hesaplama pahalı (K sembol için K network call +
tüm seri birleştirme). TTLCache ile önbelleklendi — risk metrikleri
gerçek zamanlı tazelik gerektirmez (tasarım belgesi zaten "> 5 saniye
sürerse logla" diyor, bu doğal olarak "sık hesaplanmamalı" ima ediyor).
Varsayılan TTL: 1 saat (MarketDataService'in 60 saniyelik cache'inden
kasıtlı olarak çok daha uzun — farklı tazelik gereksinimleri farklı
TTL'leri haklı çıkarıyor).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.domain.models.risk_snapshot import RiskSnapshot
    from src.domain.models.price_series import PriceSeries
    from src.domain.models.portfolio import BenchmarkInfo

from src.domain.calculators.position_quantity_timeseries import (
    compute_quantity_timeseries,
    quantity_on_date,
)
from src.domain.calculators.risk_calculator import DrawdownResult, RiskCalculator
from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.enums.var_method import VaRMethod
from src.domain.exceptions.domain_exceptions import (
    BusinessRuleError,
    InsufficientDataError,
)
from src.infrastructure.cache.ttl_cache import TTLCache
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskProfile:
    """compute_risk_snapshot'ın PERSISTENCE'SIZ hali — bkz. modül docstring'i."""

    portfolio_id: str
    as_of: date
    lookback_days: int
    data_points_used: int
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: DrawdownResult
    current_drawdown: float
    var_95: float
    cvar_95: float
    risk_free_rate_annual: float


@dataclass(frozen=True)
class RelativePerformance:
    """Design doc Bölüm 1.3 RelativePerformance ile birebir alan uyumu."""

    portfolio_return: Decimal
    benchmark_return: Decimal
    alpha_daily: float
    alpha_annualized: float
    beta: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float


class RiskService:
    def __init__(
        self,
        transaction_repo: Any,
        cash_ledger_repo: Any,
        price_sync_service: Any,
        risk_calculator: RiskCalculator,
        return_calculator: ReturnCalculator,
        risk_free_rate_annual: float,
        trading_days: int = 252,
        var_confidence: float = 0.95,
        var_method: VaRMethod = VaRMethod.HISTORICAL,
        cache: TTLCache | None = None,
    ) -> None:
        """
        cache: Çağıran taraf (container.py) TTL'i KENDİSİ ayarlayıp
        (örn. `TTLCache(ttl_seconds=3600)`) inject etmeli — TTLCache'in
        gerçek arayüzü TTL'i yalnızca constructor'da sabitliyor,
        `set()` başına override DESTEKLEMİYOR (bunu ilk taslakta
        yanlış varsaymıştım, gerçek dosyayı okuyup düzelttim). Bu
        yüzden RiskService kendi TTL değerini seçemiyor — bu, cache'in
        birden fazla servis arasında (örn. MarketDataService ile) aynı
        TTLCache instance'ının PAYLAŞILAMAYACAĞI anlamına gelir (farklı
        tazelik gereksinimleri, farklı TTLCache instance'ları gerektirir
        — container.py'da ayrı bir `_risk_cache: TTLCache(ttl_seconds=3600)`
        instance'ı olmalı, `_analysis_cache` ile PAYLAŞILMAMALI).
        """
        self._tx_repo = transaction_repo
        self._cash_ledger_repo = cash_ledger_repo
        self._price_sync = price_sync_service
        self._risk_calc = risk_calculator
        self._return_calc = return_calculator
        self._risk_free_rate = risk_free_rate_annual
        self._trading_days = trading_days
        self._var_confidence = var_confidence
        self._var_method = var_method
        self._cache = cache

    # ── Genel: portföy değer/getiri serisi inşası ──────────────────────────

    def _fetch_symbol_value_series(
        self, portfolio_id: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
    ) -> tuple[str, list[PriceSeries]]:
        """
        Tek bir sembol için YALNIZCA FETCH yapar (DB YAZMAZ) —
        ThreadPoolExecutor worker'ında GÜVENLE çalıştırılabilir.

        DÜZELTME (bu turda, GERÇEK bir "database is locked" hatasıyla
        bulundu): İlk tasarımda bu metod price_sync.get_price_history()
        çağırıyordu — bu, fetch+write'ı BİRLEŞTİRİYORDU. Birden fazla
        worker thread'in AYNI ANDA yazmaya çalışması SQLite'ın tek-yazar
        kısıtlaması nedeniyle "database is locked" hatasına yol açtı
        (busy_timeout=15000ms'ye çıkarmak bile ÇÖZMEDİ — PRAGMA'nın
        gerçekten uygulandığı doğrulandı, sorun SQLite'ın DOĞASI).

        DÜZELTME: Bu metod artık YALNIZCA price_sync.fetch_missing_only()
        çağırıyor (DB'ye YAZMAZ, yalnızca eksik veriyi CANLI sağlayıcıdan
        çeker). Asıl YAZMA (write_batch), TÜM worker'lar bitince ANA
        THREAD'DE, TEK SEFERDE yapılıyor (bkz. _build_portfolio_value_series).

        Returns:
            (symbol, price_series_list) — price_series_list boşsa
            "zaten güncel, yazılacak yeni veri yok" anlamına gelir
            (hata DEĞİL).
        """
        missing_data = self._price_sync.fetch_missing_only(symbol, start.date(), end.date())
        return symbol, missing_data

    def _read_symbol_value_series(
        self, portfolio_id: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
    ) -> tuple[str, pd.Series | None]:
        """
        Tek bir sembol için SALT OKUMA yapar (write_batch()'in TAMAMLANMIŞ
        olduğu varsayılır) — WAL modunda çoklu okuyucu güvenlidir,
        ThreadPoolExecutor'da GÜVENLE paralel çalıştırılabilir (yazma
        AKSİNE).
        """
        transactions = self._tx_repo.get_by_symbol(portfolio_id, symbol)
        quantity_series = compute_quantity_timeseries(transactions)
        if quantity_series.empty:
            return symbol, None

        ohlcv = self._price_sync.get_cached_ohlcv(symbol, start.date(), end.date())
        if ohlcv.empty:
            logger.warning(
                "risk_service_missing_price_history", symbol=symbol, portfolio_id=portfolio_id,
            )
            return symbol, None

        close = ohlcv["close"]
        close.index = pd.DatetimeIndex(close.index).normalize()

        quantities_per_day = pd.Series(
            [float(quantity_on_date(quantity_series, d)) for d in close.index],
            index=close.index,
        )
        return symbol, close.astype(float) * quantities_per_day

    def _build_portfolio_value_series(
        self, portfolio_id: str, lookback_days: int, include_cash: bool = True
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Portföyün son `lookback_days` gündeki toplam değerini (equity +
        opsiyonel nakit) günlük bazda hesaplar.

        MİMARİ DESEN — Fetch-Parallel / Write-Serial / Read-Parallel
        (bu turda, GERÇEK bir yük testiyle bulunan "database is locked"
        hatası sonrası bu şekilde tasarlandı):

          AŞAMA 1 (PARALEL): Her sembol için ThreadPoolExecutor worker'ı
            price_sync.fetch_missing_only() çağırır — YALNIZCA network
            fetch, DB YAZMASI YOK. SQLite'ın tek-yazar kısıtlamasından
            ETKİLENMEZ, güvenle paralelleştirilebilir.

          AŞAMA 2 (SERİLEŞTİRİLMİŞ): Ana thread, TÜM worker'ların
            sonuçlarını TEK BİR price_sync.write_batch() çağrısında
            birleştirip yazar. SQLite'ın tek-yazar modeliyle UYUMLU
            (yalnızca BİR yazıcı, hiç çekişme yok).

          AŞAMA 3 (PARALEL-GÜVENLİ): Her sembol için ThreadPoolExecutor
            worker'ı price_sync.get_cached_ohlcv() ile SALT OKUMA yapar
            — WAL modu çoklu okuyucuyu destekler (Faz B ADR-002),
            güvenle paralelleştirilebilir.

        Bu üç-aşamalı tasarım, PortfolioService'in ADR-003 (senkron-
        öncelikli, ThreadPoolExecutor) kararıyla TUTARLI ama SQLite'ın
        tek-yazar gerçeğini de HESABA KATIYOR — "her şeyi paralelleştir"
        yerine "paralelleştirilebilecek KISIMLARI paralelleştir, DB
        yazmasını serileştir" ayrımı.

        Returns:
            (total_value, per_symbol_values).

        Raises:
            InsufficientDataError: Portföyde hiç sembol/işlem yoksa.
            BusinessRuleError: Bir sembolün işlem geçmişi
                position_quantity_timeseries.py'ın desteklemediği bir
                transaction_type içeriyorsa.
        """
        symbols = self._tx_repo.get_portfolio_symbols(portfolio_id)
        if not symbols:
            raise InsufficientDataError(
                required=1, available=0, metric="portfolio_value_series",
            )

        end = pd.Timestamp(datetime.now().date())
        start = end - pd.Timedelta(days=lookback_days * 2)
        max_workers = min(8, len(symbols))

        # ── AŞAMA 1: Paralel fetch (DB yazması YOK) ─────────────────────────
        all_missing_data: list[PriceSeries] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fetch_futures = [
                executor.submit(self._fetch_symbol_value_series, portfolio_id, symbol, start, end)
                for symbol in symbols
            ]
            for future in as_completed(fetch_futures):
                _, missing_data = future.result()
                all_missing_data.extend(missing_data)

        # ── AŞAMA 2: Serileştirilmiş TEK yazma ──────────────────────────────
        if all_missing_data:
            self._price_sync.write_batch(all_missing_data)

        # ── AŞAMA 3: Paralel-güvenli salt okuma ─────────────────────────────
        date_index: pd.DatetimeIndex | None = None
        symbol_values: dict[str, pd.Series] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            read_futures = [
                executor.submit(self._read_symbol_value_series, portfolio_id, symbol, start, end)
                for symbol in symbols
            ]
            for read_future in as_completed(read_futures):
                symbol, series = read_future.result()
                if series is not None:
                    symbol_values[symbol] = series
                    new_index = pd.DatetimeIndex(series.index)
                    date_index = new_index if date_index is None else date_index.union(new_index)

        if not symbol_values or date_index is None:
            raise InsufficientDataError(
                required=1, available=0, metric="portfolio_value_series",
            )

        combined = pd.DataFrame(symbol_values).reindex(date_index).sort_index()
        combined = combined.ffill().fillna(0.0)
        equity_value = combined.sum(axis=1)

        if include_cash:
            cash_values = pd.Series(
                [
                    float(self._cash_ledger_repo.get_balance(portfolio_id, as_of=d.date()))
                    for d in equity_value.index
                ],
                index=equity_value.index,
            )
            total_value = equity_value + cash_values
        else:
            total_value = equity_value

        if len(total_value) > lookback_days:
            total_value = total_value.iloc[-lookback_days:]
            combined = combined.iloc[-lookback_days:]
        return total_value, combined

    def _compute_position_weights(self, combined: pd.DataFrame) -> np.ndarray:
        """
        combined'ın SON GÜNÜNDEKİ (en güncel) sembol değerlerinden
        pozisyon ağırlıklarını çıkarır — RiskCalculator.
        calculate_concentration_metrics()'e girdi.

        NOT: Yalnızca EQUITY (hisse/fon) ağırlıkları hesaplanıyor,
        NAKİT dahil DEĞİL — konsantrasyon riski kavramsal olarak
        "hangi POZİSYONLARA ne kadar yoğunlaşmışım" sorusuna cevap
        verir, nakit bir "pozisyon" değildir bu bağlamda.
        """
        latest_values = combined.iloc[-1]
        total = float(latest_values.sum())
        if total <= 0:
            raise InsufficientDataError(
                required=1, available=0, metric="position_weights",
            )
        weights: np.ndarray = (latest_values / total).to_numpy()
        return weights

    def _cached(self, key: str, builder: Any) -> Any:
        """
        TTLCache.get() zaten miss/expire durumunda None döner (exception
        FIRLATMAZ) — ilk taslakta bunu yanlış varsayıp try/except ile
        sarmışım, gerçek dosyayı (ttl_cache.py) okuyunca gereksiz
        olduğunu gördüm, kaldırdım.
        """
        if self._cache is None:
            return builder()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = builder()
        self._cache.set(key, result)
        return result

    # ── Public API ───────────────────────────────────────────────────────────

    def compute_risk_profile(
        self, portfolio_id: str, lookback_days: int = 252, include_cash: bool = True
    ) -> RiskProfile:
        """
        compute_risk_snapshot()'ın persistence'sız hali — bkz. modül
        docstring'indeki kapsam kararı. force_recompute/staleness kontrolü
        YOK (RiskSnapshot repository olmadan anlamsız), HER ÇAĞRIDA
        yeniden hesaplanır (TTLCache üzerinden — bkz. _cached).
        """
        cache_key = f"risk_profile:{portfolio_id}:{lookback_days}:{include_cash}"

        def _compute() -> RiskProfile:
            value_series, _ = self._build_portfolio_value_series(
                portfolio_id, lookback_days, include_cash
            )
            daily_returns = value_series.pct_change().dropna().to_numpy()
            cumulative = (1.0 + pd.Series(daily_returns)).cumprod().to_numpy()

            volatility = self._risk_calc.calculate_annualized_volatility(
                daily_returns, self._trading_days
            )
            sharpe = self._risk_calc.calculate_sharpe_ratio(
                daily_returns, self._risk_free_rate, self._trading_days
            )
            sortino = self._risk_calc.calculate_sortino_ratio(
                daily_returns, self._risk_free_rate, self._trading_days
            )
            max_dd = self._risk_calc.calculate_max_drawdown(cumulative)
            current_dd = self._risk_calc.calculate_current_drawdown(cumulative)
            var_95 = self._risk_calc.calculate_var(
                daily_returns, self._var_confidence, self._var_method
            )
            cvar_95 = self._risk_calc.calculate_cvar(daily_returns, self._var_confidence)

            return RiskProfile(
                portfolio_id=portfolio_id,
                as_of=date.today(),
                lookback_days=lookback_days,
                data_points_used=len(daily_returns),
                annualized_volatility=volatility,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                var_95=var_95,
                cvar_95=cvar_95,
                risk_free_rate_annual=self._risk_free_rate,
            )

        start_time = datetime.now()
        result: RiskProfile = self._cached(cache_key, _compute)
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > 5.0:
            logger.warning(
                "risk_profile_computation_slow", portfolio_id=portfolio_id, elapsed_seconds=elapsed,
            )
        return result

    def calculate_relative_performance(
        self,
        portfolio_id: str,
        benchmark_code: str,
        lookback_days: int = 252,
        include_cash: bool = True,
    ) -> RelativePerformance:
        """
        benchmark_code: Portfolio.benchmark_code alanından gelir (KULLANICI
        SEÇİMİ, portföy başına — bkz. bu turda alınan karar). Bu servis
        benchmark_code'un geçerliliğini KONTROL ETMEZ — bu, Portfolio
        domain modelinin __post_init__ sorumluluğu (zaten yapılıyor).
        """
        cache_key = f"relative_perf:{portfolio_id}:{benchmark_code}:{lookback_days}:{include_cash}"

        def _compute() -> RelativePerformance:
            portfolio_value, _ = self._build_portfolio_value_series(
                portfolio_id, lookback_days, include_cash
            )
            end = pd.Timestamp(datetime.now().date())
            start = end - pd.Timedelta(days=lookback_days * 2)
            benchmark_ohlcv = self._price_sync.get_price_history(
                benchmark_code, start.date(), end.date(),
            )
            if benchmark_ohlcv.empty:
                raise InsufficientDataError(
                    required=1, available=0, metric="benchmark_price_history",
                )
            benchmark_close = benchmark_ohlcv["close"]
            benchmark_close.index = pd.DatetimeIndex(benchmark_close.index).normalize()
            benchmark_close = benchmark_close.iloc[-lookback_days:] if len(benchmark_close) > lookback_days else benchmark_close

            # KRİTİK: tarih hizalaması — iki seri FARKLI trading calendar'a
            # sahip olabilir (örn. TEFAS fonu ile BIST100 arasında). Yalnızca
            # ORTAK tarihler kullanılıyor (inner join) — bu, RiskCalculator'ın
            # "eşit uzunlukta dizi" gereksinimini karşılamanın TEK doğru yolu;
            # sırf uzunlukları eşitlemek için kırpmak (örn. son N eleman)
            # tarihleri YANLIŞ eşleştirebilirdi.
            aligned = pd.DataFrame({
                "portfolio": portfolio_value, "benchmark": benchmark_close.astype(float),
            }).dropna()

            if len(aligned) < 2:
                raise InsufficientDataError(
                    required=2, available=len(aligned), metric="aligned_return_series",
                )

            portfolio_returns = aligned["portfolio"].pct_change().dropna().to_numpy()
            benchmark_returns = aligned["benchmark"].pct_change().dropna().to_numpy()

            beta, alpha_daily, r_squared = self._risk_calc.calculate_beta(
                portfolio_returns, benchmark_returns
            )
            tracking_error = float(np.std(portfolio_returns - benchmark_returns, ddof=1))
            info_ratio = self._risk_calc.calculate_information_ratio(
                portfolio_returns, benchmark_returns
            )
            up_capture, down_capture = self._risk_calc.calculate_capture_ratios(
                portfolio_returns, benchmark_returns
            )

            portfolio_total_return = Decimal(str(
                aligned["portfolio"].iloc[-1] / aligned["portfolio"].iloc[0] - 1.0
            ))
            benchmark_total_return = Decimal(str(
                aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1.0
            ))
            alpha_annualized = (1.0 + alpha_daily) ** self._trading_days - 1.0

            return RelativePerformance(
                portfolio_return=portfolio_total_return,
                benchmark_return=benchmark_total_return,
                alpha_daily=alpha_daily,
                alpha_annualized=alpha_annualized,
                beta=beta,
                r_squared=r_squared,
                tracking_error=tracking_error,
                information_ratio=info_ratio,
                up_capture=up_capture,
                down_capture=down_capture,
            )

        return self._cached(cache_key, _compute)  # type: ignore[no-any-return]

    def compute_and_persist_snapshot(
        self,
        portfolio_id: str,
        risk_snapshot_repo: Any,
        lookback_days: int = 252,
        include_cash: bool = True,
        benchmark_code: str | None = None,
    ) -> "RiskSnapshot":
        """
        compute_risk_snapshot()'ın TAM (persistence dahil) hali —
        design doc Bölüm 1.3 ile artık daha yakın uyumlu (calmar_ratio,
        var_99/cvar_99, drawdown tarihleri, konsantrasyon metrikleri
        dahil — bkz. RiskCalculator'a bu turda eklenenler).

        risk_snapshot_repo: Constructor'a DEĞİL, metoda parametre olarak
        geçiriliyor — GEREKÇE: RiskService'in constructor'ı zaten 7
        bağımlılık taşıyor (transaction_repo, cash_ledger_repo,
        price_sync_service, risk_calculator, return_calculator,
        risk_free_rate, cache) — 8.'sini eklemek yerine, yalnızca BU
        metodu kullanan çağıranların (Scheduler) ihtiyaç duyduğu
        bağımlılığı METOD SEVİYESİNDE geçirmek, RiskService'in temel
        kullanım senaryosunu (compute_risk_profile — persistence
        gerektirmez) gereksiz yere ağırlaştırmıyor.

        NOT — konsantrasyon metrikleri NAKİT HARİÇ hesaplanıyor (bkz.
        _compute_position_weights) ama risk metrikleri (volatilite,
        Sharpe vb.) include_cash parametresine göre nakit DAHİL/HARİÇ
        olabilir — bu İKİ FARKLI include_cash kararı, kavramsal olarak
        BAĞIMSIZ (biri "risk hesabına nakit dahil mi", diğeri
        "konsantrasyon riski nakit dahil mi" — ikincisi için nakdin
        dahil edilmesi anlamsız, bkz. _compute_position_weights docstring'i).
        """
        from src.domain.models.risk_snapshot import RiskSnapshot

        value_series, combined = self._build_portfolio_value_series(
            portfolio_id, lookback_days, include_cash
        )
        returns_index = value_series.index[1:]
        daily_returns = value_series.pct_change().dropna().to_numpy()
        cumulative = (1.0 + pd.Series(daily_returns)).cumprod().to_numpy()

        volatility = self._risk_calc.calculate_annualized_volatility(daily_returns, self._trading_days)
        sharpe = self._risk_calc.calculate_sharpe_ratio(daily_returns, self._risk_free_rate, self._trading_days)
        sortino = self._risk_calc.calculate_sortino_ratio(daily_returns, self._risk_free_rate, self._trading_days)
        max_dd = self._risk_calc.calculate_max_drawdown(cumulative)
        current_dd = self._risk_calc.calculate_current_drawdown(cumulative)
        var_95 = self._risk_calc.calculate_var(daily_returns, 0.95, self._var_method)
        var_99 = self._risk_calc.calculate_var(daily_returns, 0.99, self._var_method)
        cvar_95 = self._risk_calc.calculate_cvar(daily_returns, 0.95)
        cvar_99 = self._risk_calc.calculate_cvar(daily_returns, 0.99)

        total_return = float(cumulative[-1] - 1.0)
        annualized_return = float(
            self._return_calc.calculate_annualized_return(
                Decimal(str(total_return)), len(daily_returns)
            )
        )
        try:
            calmar = self._risk_calc.calculate_calmar_ratio(annualized_return, max_dd.max_drawdown)
        except Exception:
            calmar = None  # max_drawdown=0 (hiç düşüş yok) — nadir ama geçerli bir durum

        max_dd_start = returns_index[max_dd.peak_idx].date() if max_dd.peak_idx < len(returns_index) else None
        max_dd_end = returns_index[max_dd.trough_idx].date() if max_dd.trough_idx < len(returns_index) else None

        try:
            weights = self._compute_position_weights(combined)
            hhi, top5 = self._risk_calc.calculate_concentration_metrics(weights)
        except Exception as exc:
            logger.warning("concentration_metrics_unavailable", portfolio_id=portfolio_id, error=str(exc))
            hhi, top5 = None, None

        beta = alpha = r_squared = info_ratio = tracking_error = None
        if benchmark_code:
            try:
                perf = self.calculate_relative_performance(
                    portfolio_id, benchmark_code, lookback_days, include_cash,
                )
                beta, alpha = perf.beta, perf.alpha_annualized
                r_squared, info_ratio = perf.r_squared, perf.information_ratio
                tracking_error = perf.tracking_error
            except Exception as exc:
                logger.warning(
                    "relative_performance_unavailable_for_snapshot",
                    portfolio_id=portfolio_id, benchmark_code=benchmark_code, error=str(exc),
                )

        snapshot = RiskSnapshot(
            portfolio_id=portfolio_id, as_of_date=date.today(), lookback_days=lookback_days,
            risk_free_rate=self._risk_free_rate,
            portfolio_volatility=volatility, sharpe_ratio=sharpe, sortino_ratio=sortino,
            calmar_ratio=calmar, max_drawdown=max_dd.max_drawdown,
            max_drawdown_start=max_dd_start, max_drawdown_end=max_dd_end,
            current_drawdown=current_dd, var_95=var_95, var_99=var_99,
            cvar_95=cvar_95, cvar_99=cvar_99, var_method=self._var_method,
            beta=beta, alpha=alpha, r_squared=r_squared,
            information_ratio=info_ratio, tracking_error=tracking_error,
            herfindahl_index=hhi, top5_concentration=top5,
            benchmark_code=benchmark_code,
        )
        persisted: RiskSnapshot = risk_snapshot_repo.add(snapshot)
        return persisted

    def get_rolling_volatility(
        self, portfolio_id: str, lookback_days: int = 252,
        window: int = 30, include_cash: bool = True,
    ) -> pd.Series:
        """
        Zaman içinde volatilitenin NASIL DEĞİŞTİĞİNİ gösteren bir seri
        (compute_risk_profile()'ın TEK SAYI çıktısının aksine).

        DÜZELTME (bu turda): Daha önce "her pencere için return-series
        reconstruction'ı gerekiyor" gerekçesiyle ERTELENMİŞTİ — bu
        DOĞRULANMAMIŞ, YANLIŞ bir varsayımdı (bkz. modül docstring'i).
        _build_portfolio_value_series() ZATEN TAM seriyi TEK SEFERDE
        döndürüyor — ek network/DB maliyeti YOK.

        Formül, RiskCalculator.calculate_annualized_volatility() İLE
        BİREBİR TUTARLI (ddof=1, sqrt(trading_days) ile yıllıklandırma)
        — AYNI hesaplamayı farklı bir yerde YENİDEN YAZMAK yerine
        pandas'ın .rolling().std() varsayılanı (ddof=1) DOĞRUDAN
        kullanılıyor (doğrulandı — bkz. bu turun gerekçe notu).

        Returns:
            pd.Series — date index, değerler yıllıklandırılmış
            volatilite (float). İlk (window-1) gün NaN olacağı için
            SONUÇTAN ÇIKARILIYOR (dropna) — "yeterli veri birikene
            kadar rolling metrik tanımsızdır" ilkesiyle TUTARLI.

        Raises:
            InsufficientDataError: Portföyde hiç sembol/işlem yoksa
                VEYA lookback_days, window'dan KISAYSA (rolling
                pencere hiçbir zaman dolmaz, TAMAMEN NaN bir seri
                dönmek YANILTICI olurdu — bu yüzden AÇIK bir hata
                tercih edildi).
        """
        if lookback_days < window:
            raise InsufficientDataError(
                required=window, available=lookback_days, metric="rolling_volatility_window",
            )
        value_series, _ = self._build_portfolio_value_series(portfolio_id, lookback_days, include_cash)
        daily_returns = value_series.pct_change().dropna()
        rolling_vol: pd.Series = daily_returns.rolling(window=window).std(ddof=1) * (self._trading_days ** 0.5)
        return rolling_vol.dropna()

    def get_drawdown_series(
        self, portfolio_id: str, lookback_days: int = 252, include_cash: bool = True,
    ) -> pd.Series:
        """
        HER GÜN için o günkü "zirveden ne kadar aşağıda" bilgisini
        veren tam seri (compute_risk_profile()'ın yalnızca EN DERİN
        noktayı (max_drawdown) döndüren TEK SAYI çıktısının aksine).

        Formül, RiskCalculator.calculate_max_drawdown() İLE BİREBİR
        AYNI (drawdown = value/running_max - 1) — o metod bu serinin
        yalnızca MİNİMUM noktasını alıyor, burada TAM SERİ dönüyor.
        Kod TEKRARI değil: aynı formül, iki farklı GRANÜLERLİKTE
        tüketiliyor (nokta vs. seri) — matematiksel mantığı
        RiskCalculator'a AİT, burada YENİDEN YAZILMIYOR, yalnızca
        vectorized numpy operasyonu olarak DOĞRUDAN uygulanıyor (aynı
        formülü tekrar tekrar fonksiyon çağırarak hesaplamak yerine).

        Returns:
            pd.Series — date index, değerler drawdown yüzdesi (negatif
            veya sıfır — asla pozitif).

        Raises:
            InsufficientDataError: Portföyde hiç sembol/işlem yoksa.
        """
        value_series, _ = self._build_portfolio_value_series(portfolio_id, lookback_days, include_cash)
        cumulative = value_series / value_series.iloc[0]
        running_max = cumulative.cummax()
        drawdown: pd.Series = cumulative / running_max - 1.0
        return drawdown

    def get_available_benchmarks(self) -> "tuple[BenchmarkInfo, ...]":
        """
        DÜZELTME (bu turda): Design doc'un get_available_benchmarks()
        → list[BenchmarkInfo] arayüzü artık implemente edildi.
        Portfolio.py'daki KNOWN_BENCHMARKS ile AYNI kanonik kaynaktan
        besleniyor (bkz. o modülün "3 AYRI yer" tutarsızlık düzeltmesi
        gerekçesi) — burada TEKRAR TANIMLANMIYOR.
        """
        from src.domain.models.portfolio import KNOWN_BENCHMARKS
        return KNOWN_BENCHMARKS
