"""
BacktestService — design doc Bölüm 4.7 "BacktestService.run(strategy,
symbols, start, end, params)" ile uyumlu, TEK SEMBOL alt kümesi (bkz.
backtest_engine.py kapsam sınırları).

KAPSAM: BacktestDataLoader ayrı bir sınıf olarak YAZILMADI — PriceSyncService
(Faz F'de inşa edilmiş, cache-aside + concurrency-safe) DOĞRUDAN
kullanılıyor. Design doc'un "L2 Cache'ten yükle, cache miss → Provider'dan
çek → Cache'e yaz" akışı, PriceSyncService.get_price_history() ile
ZATEN TAM olarak karşılanıyor — ayrı bir data loader katmanı DRY
ihlali olurdu.

run_with_benchmark() — bu turda eklendi: kullanıcının stratejisini VE
BuyAndHoldStrategy'yi AYNI fiyat verisi üzerinde çalıştırıp karşılaştırma
sunuyor (standart backtesting pratiği — bkz. buy_and_hold_strategy.py
modül docstring'i). Fiyat verisi YALNIZCA BİR KEZ çekiliyor (iki ayrı
PriceSyncService çağrısı YAPILMIYOR) — DRY + gereksiz network/cache
maliyetinden kaçınma.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, NamedTuple

import pandas as pd

from src.domain.calculators.backtest_engine import BacktestEngine, BacktestResult
from src.domain.exceptions.domain_exceptions import InsufficientDataError
from src.domain.strategies.buy_and_hold_strategy import BuyAndHoldStrategy


class BacktestComparison(NamedTuple):
    strategy_result: BacktestResult
    benchmark_result: BacktestResult  # Buy & Hold (AYNI sembol)
    # DÜZELTME (bu turda eklendi): Portfolio.KNOWN_BENCHMARKS'tan (BIST100
    # vb.) SEÇİLEN bir GERÇEK endeks karşılaştırması — Buy & Hold'un
    # AKSİNE, AYNI sembol değil FARKLI bir REFERANS (örn. "THYAO'da SMA
    # stratejim, THYAO'yu alıp tutmaktan iyi mi, PİYASANIN GENELİNDEN
    # (BIST100) iyi mi?" — iki FARKLI soru, ikisi de değerli).
    # None: kullanıcı bir endeks SEÇMEDİYSE VEYA o endeksin verisi
    # ÇEKİLEMEDİYSE (bkz. run_with_benchmark — bu İKİNCİL karşılaştırma,
    # BAŞARISIZ olması ANA karşılaştırmayı ÇÖKERTMEMELİ).
    index_benchmark_result: BacktestResult | None = None
    index_benchmark_symbol: str | None = None


class MultiSymbolBacktestResult(NamedTuple):
    """
    DÜZELTME (bu turda eklendi): Çoklu-sembol backtest'in TOPLU sonucu.

    symbol_results: Her sembolün BAĞIMSIZ BacktestResult'ı (mevcut,
      test edilmiş motor DEĞİŞTİRİLMEDEN kullanıldı).
    failed_symbols: Veri çekilemediği için ATLANAN semboller (hata
      izolasyonu — bkz. modül docstring'i).
    combined_equity_curve: TÜM başarılı sembollerin portfolio_value_series'inin
      TOPLAMI (tarih bazında hizalanmış).
    combined_total_return: (combined_final_value / combined_initial_capital) - 1.
    """
    symbol_results: dict[str, BacktestResult]
    failed_symbols: tuple[str, ...]
    combined_equity_curve: pd.Series
    combined_initial_capital: Decimal
    combined_final_value: Decimal
    combined_total_return: Decimal


class BacktestService:
    def __init__(self, price_sync_service: Any, backtest_engine: BacktestEngine) -> None:
        self._price_sync = price_sync_service
        self._engine = backtest_engine

    def _fetch_price_data(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """run() ve run_with_benchmark() arasında PAYLAŞILAN fetch mantığı — DRY."""
        ohlcv = self._price_sync.get_price_history(symbol, start, end)
        if ohlcv.empty:
            raise InsufficientDataError(required=1, available=0, metric="backtest_price_history")
        price_data = ohlcv[["close"]].copy()
        price_data["close"] = price_data["close"].astype(float)
        result: pd.DataFrame = price_data
        return result

    def run(
        self,
        symbol: str,
        strategy: Any,
        start: date,
        end: date,
        strategy_params: dict[str, Any] | None = None,
        initial_capital: Decimal = Decimal("10000"),
        commission_rate: Decimal = Decimal("0.001"),
        position_size_pct: Decimal = Decimal("1.0"),
    ) -> BacktestResult:
        """
        Raises:
            InsufficientDataError: symbol için fiyat verisi çekilemezse.
            BusinessRuleError: BacktestEngine.run()'ın girdi validasyonu
                başarısız olursa (bkz. o metodun docstring'i).
        """
        price_data = self._fetch_price_data(symbol, start, end)
        signals = strategy.generate_signals(price_data, strategy_params or {})

        return self._engine.run(
            symbol=symbol, price_data=price_data, signals=signals,
            initial_capital=initial_capital, commission_rate=commission_rate,
            position_size_pct=position_size_pct,
        )

    def run_with_benchmark(
        self,
        symbol: str,
        strategy: Any,
        start: date,
        end: date,
        strategy_params: dict[str, Any] | None = None,
        initial_capital: Decimal = Decimal("10000"),
        commission_rate: Decimal = Decimal("0.001"),
        position_size_pct: Decimal = Decimal("1.0"),
        index_benchmark_symbol: str | None = None,
    ) -> BacktestComparison:
        """
        Kullanıcının stratejisini VE Buy & Hold'u AYNI fiyat verisi
        üzerinde çalıştırıp İKİSİNİ birden döner.

        DÜZELTME (bu turda eklendi): index_benchmark_symbol verilirse
        (örn. "XU100.IS" — bkz. Portfolio.KNOWN_BENCHMARKS), o endeksin
        AYRI bir Buy & Hold sonucu da hesaplanır — "piyasanın genelini
        yendim mi" sorusuna cevap. Bu, AYRI bir fiyat verisi fetch'i
        gerektiriyor (farklı sembol) — GEREKSİZ risk almamak için bu
        fetch BAŞARISIZ olursa (endeks verisi çekilemezse) ANA
        karşılaştırma (strateji vs. aynı-sembol Buy&Hold) YİNE DE
        BAŞARIYLA döner, yalnızca index_benchmark_result None kalır
        (hata izolasyonu — bkz. run_multi_symbol()'daki AYNI ilke).

        NOT: Buy & Hold, KOMİSYON AÇISINDAN da ADİL karşılaştırılıyor
        (aynı commission_rate kullanılıyor) — yalnızca 1 işlem (baştaki
        AL) yapacağı için komisyon etkisi ZATEN minimal, ama "farklı
        maliyet varsayımıyla haksız karşılaştırma" riski YOK.

        Raises: run() ile AYNI (bkz. o metodun docstring'i).
        """
        price_data = self._fetch_price_data(symbol, start, end)

        strategy_signals = strategy.generate_signals(price_data, strategy_params or {})
        strategy_result = self._engine.run(
            symbol=symbol, price_data=price_data, signals=strategy_signals,
            initial_capital=initial_capital, commission_rate=commission_rate,
            position_size_pct=position_size_pct,
        )

        benchmark_signals = BuyAndHoldStrategy().generate_signals(price_data, {})
        benchmark_result = self._engine.run(
            symbol=symbol, price_data=price_data, signals=benchmark_signals,
            initial_capital=initial_capital, commission_rate=commission_rate,
            position_size_pct=position_size_pct,
        )

        index_benchmark_result: BacktestResult | None = None
        if index_benchmark_symbol is not None:
            try:
                index_price_data = self._fetch_price_data(index_benchmark_symbol, start, end)
                index_signals = BuyAndHoldStrategy().generate_signals(index_price_data, {})
                index_benchmark_result = self._engine.run(
                    symbol=index_benchmark_symbol, price_data=index_price_data, signals=index_signals,
                    initial_capital=initial_capital, commission_rate=commission_rate,
                    position_size_pct=position_size_pct,
                )
            except Exception:
                # DÜZELTME notu: endeks verisi çekilemezse (ör. yfinance'ta
                # bu endeks kodu çözümlenemiyorsa — bkz. Portfolio.py'daki
                # "GERÇEKTEN test edilemedi" belirsizlik notu), ANA
                # karşılaştırma ETKİLENMEMELİ — hata izolasyonu.
                index_benchmark_result = None

        return BacktestComparison(
            strategy_result=strategy_result, benchmark_result=benchmark_result,
            index_benchmark_result=index_benchmark_result,
            index_benchmark_symbol=index_benchmark_symbol if index_benchmark_result is not None else None,
        )

    def run_multi_symbol(
        self,
        symbols: list[str],
        strategy: Any,
        start: date,
        end: date,
        strategy_params: dict[str, Any] | None = None,
        initial_capital: Decimal = Decimal("10000"),
        commission_rate: Decimal = Decimal("0.001"),
        position_size_pct: Decimal = Decimal("1.0"),
    ) -> MultiSymbolBacktestResult:
        """
        DÜZELTME (bu turda eklendi — MİMARİ KARAR): N BAĞIMSIZ tek-sembol
        backtest'i (mevcut BacktestEngine.run() — DEĞİŞTİRİLMEDEN) EŞİT
        SERMAYE paylaşımıyla çalıştırıp sonuçları birleştirir.

        BİLİNÇLİ SINIRLAMALAR (MVP kapsamı, bkz. bu turun gerekçe notu):
          - Sermaye dağıtımı EŞİT AĞIRLIK (initial_capital / N sembol) —
            sinyal gücüne göre ağırlıklandırma YOK.
          - REBALANCING YOK — her sembol bağımsız alt-portföy gibi davranır.
          - Semboller arası PORTFÖY-SEVİYESİ mantık (pairs trading,
            korelasyon bazlı ağırlıklandırma) YOK — her sembol AYNI
            stratejiyi BAĞIMSIZ uygular.

        HATA İZOLASYONU: Bir sembol veri çekemezse (ağ hatası vb.), o
        sembol ATLANIR (failed_symbols'a eklenir), DİĞERLERİYLE devam
        edilir — TÜM backtest BAŞARISIZ OLMAZ (yalnızca TÜM semboller
        başarısız olursa InsufficientDataError fırlatılır).

        Raises:
            BusinessRuleError: symbols listesi boşsa.
            InsufficientDataError: HİÇBİR sembol için veri çekilemezse.
        """
        from src.domain.exceptions.domain_exceptions import BusinessRuleError

        if not symbols:
            raise BusinessRuleError("En az bir sembol gerekli.")

        per_symbol_capital = initial_capital / len(symbols)
        results: dict[str, BacktestResult] = {}
        failed: list[str] = []

        for symbol in symbols:
            try:
                results[symbol] = self.run(
                    symbol=symbol, strategy=strategy, start=start, end=end,
                    strategy_params=strategy_params, initial_capital=per_symbol_capital,
                    commission_rate=commission_rate, position_size_pct=position_size_pct,
                )
            except Exception:
                failed.append(symbol)

        if not results:
            raise InsufficientDataError(
                required=1, available=0, metric="multi_symbol_backtest_all_failed",
            )

        # Tarih bazında hizalayıp TÜM sembollerin equity serilerini TOPLA.
        combined = pd.Series(dtype=float)
        for result in results.values():
            combined = combined.add(result.portfolio_value_series, fill_value=0.0)

        combined_final_value = Decimal(str(combined.iloc[-1])) if len(combined) > 0 else Decimal("0")
        combined_total_return = (
            (combined_final_value / initial_capital) - Decimal("1")
            if initial_capital > Decimal("0") else Decimal("0")
        )

        return MultiSymbolBacktestResult(
            symbol_results=results, failed_symbols=tuple(failed),
            combined_equity_curve=combined, combined_initial_capital=initial_capital,
            combined_final_value=combined_final_value, combined_total_return=combined_total_return,
        )

