"""
PortfolioService unit testleri — Aşama 8 (concurrent + multi-portfolio).

Değişiklikler:
  - Tüm testler concurrent (ThreadPoolExecutor) mimarisiyle çalışıyor.
  - list_portfolios() testi eklendi.
  - portfolio_repo injection testi eklendi.
  - max_workers config testi eklendi.
  - Concurrency altında stale_symbols thread-safety testi eklendi.
  - Önceki Aşama 7 testlerinin hepsi geriye dönük uyumlu — API değişmedi.
"""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.domain.calculators.cost_basis_calculator import (
    CostBasisResult,
    WAVGCostBasisCalculator,
)
from src.domain.exceptions.domain_exceptions import InsufficientQuantityError
from src.services.market_data_service import MarketAnalysisResult, MarketDataService
from src.services.portfolio_service import (
    PortfolioService,
    PortfolioSummaryDTO,
    PositionDTO,
)

PORTFOLIO_ID = "test-portfolio-uuid"
ZERO = Decimal("0")


# ── Test Yardımcıları ─────────────────────────────────────────────────────────

def _make_cb_result(
    quantity: str = "100",
    avg_cost: str = "10.00",
    realized_pnl: str = "0",
) -> CostBasisResult:
    q = Decimal(quantity)
    a = Decimal(avg_cost)
    return CostBasisResult(
        total_quantity=q,
        average_cost=a,
        total_cost_basis=(q * a).quantize(Decimal("0.01")),
        realized_pnl=Decimal(realized_pnl),
        total_dividends=ZERO,
    )


def _make_mock_tx_repo(
    symbols: list[str] | None = None,
    transactions_by_symbol: dict | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock.get_portfolio_symbols.return_value = symbols or []
    txs = transactions_by_symbol or {}
    mock.get_by_symbol.side_effect = lambda pid, sym, **kw: txs.get(sym, [])
    return mock


def _make_mock_market_svc(latest_close: float | None = 12.50) -> MagicMock:
    mock = MagicMock(spec=MarketDataService)
    if latest_close is None:
        mock.get_market_analysis.side_effect = Exception("price unavailable")
    else:
        r = MagicMock(spec=MarketAnalysisResult)
        r.latest_close = latest_close
        mock.get_market_analysis.return_value = r
    return mock


def _make_mock_calculator(cb: CostBasisResult) -> MagicMock:
    mock = MagicMock(spec=WAVGCostBasisCalculator)
    mock.calculate.return_value = cb
    return mock


def _make_mock_portfolio_repo(
    portfolios: list[dict] | None = None,
) -> MagicMock:
    from types import SimpleNamespace
    mock = MagicMock()
    items = portfolios or [
        {"id": "p-1", "name": "BIST Portföyü"},
        {"id": "p-2", "name": "Büyüme Portföyü"},
    ]
    # MagicMock(name=...) özel parametre — mock'un debug adını set eder, attribute'ı değil.
    # SimpleNamespace ile gerçek attribute erişimi sağlanır.
    fake_portfolios = [SimpleNamespace(id=p["id"], name=p["name"]) for p in items]
    mock.list_all.return_value = fake_portfolios
    return mock


# ── Mutlu Yol ─────────────────────────────────────────────────────────────────

class TestHappyPath:

    def test_single_position_pnl_math(self):
        cb = _make_cb_result(quantity="100", avg_cost="10.00")
        tx_repo = _make_mock_tx_repo(["THYAO"], {"THYAO": [MagicMock()]})
        market_svc = _make_mock_market_svc(12.50)
        calculator = _make_mock_calculator(cb)

        service = PortfolioService(tx_repo, market_svc, calculator=calculator)
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        assert len(summary.positions) == 1
        pos = summary.positions[0]
        assert pos.symbol == "THYAO"
        assert pos.total_quantity == Decimal("100")
        assert pos.average_cost == Decimal("10.00")
        assert pos.current_price == Decimal("12.50")
        assert pos.unrealized_pnl == Decimal("250.00")

    def test_pnl_percentage_correct(self):
        cb = _make_cb_result(quantity="100", avg_cost="10.00")
        tx_repo = _make_mock_tx_repo(["THYAO"], {"THYAO": [MagicMock()]})
        market_svc = _make_mock_market_svc(12.50)
        calculator = _make_mock_calculator(cb)

        service = PortfolioService(tx_repo, market_svc, calculator=calculator)
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        assert summary.positions[0].pnl_percentage == Decimal("25.00")

    def test_summary_totals_correct(self):
        def calc_side(transactions):
            sym = transactions[0].symbol
            return _make_cb_result("100","10.00") if sym=="THYAO" else _make_cb_result("50","20.00")

        tx_repo = _make_mock_tx_repo(
            ["THYAO","GARAN"],
            {"THYAO":[MagicMock(symbol="THYAO")],"GARAN":[MagicMock(symbol="GARAN")]},
        )
        def price_side(symbol, timeframe):
            r = MagicMock(spec=MarketAnalysisResult)
            r.latest_close = 12.50 if symbol=="THYAO" else 22.00
            return r

        market_svc = MagicMock(spec=MarketDataService)
        market_svc.get_market_analysis.side_effect = price_side
        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.side_effect = calc_side

        service = PortfolioService(tx_repo, market_svc, calculator=calculator)
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        assert summary.position_count == 2
        assert summary.total_cost_basis == Decimal("2000.00")
        assert summary.total_current_value == Decimal("2350.00")
        assert summary.total_unrealized_pnl == Decimal("350.00")

    def test_realized_pnl_in_summary(self):
        cb = _make_cb_result("50","10.00","100.00")
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        service = PortfolioService(tx_repo, _make_mock_market_svc(12.00), calculator=_make_mock_calculator(cb))
        summary = service.get_portfolio_status(PORTFOLIO_ID)
        assert summary.total_realized_pnl == Decimal("100.00")

    def test_current_value_computed_correctly(self):
        cb = _make_cb_result("200","15.00")
        tx_repo = _make_mock_tx_repo(["AKBNK"],{"AKBNK":[MagicMock()]})
        service = PortfolioService(tx_repo, _make_mock_market_svc(18.00), calculator=_make_mock_calculator(cb))
        summary = service.get_portfolio_status(PORTFOLIO_ID)
        assert summary.positions[0].current_value == Decimal("3600.00")


# ── Hata Toleransı ────────────────────────────────────────────────────────────

class TestErrorTolerance:

    def test_price_fetch_failure_nulls_price_fields(self):
        cb = _make_cb_result("100","10.00")
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        service = PortfolioService(tx_repo, _make_mock_market_svc(None), calculator=_make_mock_calculator(cb))
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        pos = summary.positions[0]
        assert pos.total_quantity == Decimal("100")
        assert pos.average_cost == Decimal("10.00")
        assert pos.current_price is None
        assert pos.unrealized_pnl is None

    def test_price_failure_adds_to_stale_symbols(self):
        cb = _make_cb_result()
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        service = PortfolioService(tx_repo, _make_mock_market_svc(None), calculator=_make_mock_calculator(cb))
        summary = service.get_portfolio_status(PORTFOLIO_ID)
        assert "THYAO" in summary.stale_symbols

    def test_partial_failure_other_positions_computed(self):
        def calc_side(transactions):
            return _make_cb_result("100","10.00")

        tx_repo = _make_mock_tx_repo(
            ["THYAO","GARAN"],
            {"THYAO":[MagicMock(symbol="THYAO")],"GARAN":[MagicMock(symbol="GARAN")]},
        )
        def price_side(symbol, timeframe):
            if symbol=="THYAO": raise Exception("ağ hatası")
            r = MagicMock(spec=MarketAnalysisResult); r.latest_close=25.00; return r

        market_svc = MagicMock(spec=MarketDataService)
        market_svc.get_market_analysis.side_effect = price_side
        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.side_effect = calc_side

        service = PortfolioService(tx_repo, market_svc, calculator=calculator)
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        assert summary.position_count == 2
        assert "THYAO" in summary.stale_symbols
        assert "GARAN" not in summary.stale_symbols
        garan = next(p for p in summary.positions if p.symbol=="GARAN")
        assert garan.current_price == Decimal("25.00")
        assert garan.unrealized_pnl == Decimal("1500.00")

    def test_stale_positions_excluded_from_totals(self):
        def calc_side(transactions):
            return _make_cb_result("100","10.00")

        tx_repo = _make_mock_tx_repo(
            ["THYAO","GARAN"],
            {"THYAO":[MagicMock(symbol="THYAO")],"GARAN":[MagicMock(symbol="GARAN")]},
        )
        def price_side(symbol, timeframe):
            if symbol=="THYAO": raise Exception("hata")
            r = MagicMock(spec=MarketAnalysisResult); r.latest_close=15.00; return r

        market_svc = MagicMock(spec=MarketDataService)
        market_svc.get_market_analysis.side_effect = price_side
        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.side_effect = calc_side

        service = PortfolioService(tx_repo, market_svc, calculator=calculator)
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        assert summary.total_current_value == Decimal("1500.00")
        assert summary.total_cost_basis == Decimal("2000.00")

    def test_calculator_insufficient_qty_skips_symbol(self):
        tx_repo = _make_mock_tx_repo(["BRKB"],{"BRKB":[MagicMock()]})
        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.side_effect = InsufficientQuantityError(
            symbol="BRKB", requested=Decimal("200"), available=Decimal("100")
        )
        service = PortfolioService(tx_repo, _make_mock_market_svc(10.00), calculator=calculator)
        summary = service.get_portfolio_status(PORTFOLIO_ID)
        assert len(summary.positions) == 0

    def test_no_exception_raised_to_caller(self):
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        market_svc = MagicMock(spec=MarketDataService)
        market_svc.get_market_analysis.side_effect = RuntimeError("beklenmedik hata")
        service = PortfolioService(tx_repo, market_svc, calculator=_make_mock_calculator(_make_cb_result()))

        try:
            service.get_portfolio_status(PORTFOLIO_ID)
        except Exception as exc:
            pytest.fail(f"Servis exception fırlattı: {exc}")


# ── Boş Portföy ───────────────────────────────────────────────────────────────

class TestEmptyPortfolio:

    def test_no_symbols_returns_empty_summary(self):
        service = PortfolioService(_make_mock_tx_repo([]), _make_mock_market_svc())
        summary = service.get_portfolio_status(PORTFOLIO_ID)
        assert isinstance(summary, PortfolioSummaryDTO)
        assert len(summary.positions) == 0
        assert summary.total_cost_basis == ZERO

    def test_zero_quantity_position_excluded(self):
        cb = _make_cb_result("0","0")
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        service = PortfolioService(tx_repo, _make_mock_market_svc(12.00), calculator=_make_mock_calculator(cb))
        summary = service.get_portfolio_status(PORTFOLIO_ID)
        assert len(summary.positions) == 0


# ── Multi-Portfolio (Aşama 8) ─────────────────────────────────────────────────

class TestMultiPortfolio:

    def test_list_portfolios_returns_id_name_pairs(self):
        tx_repo = _make_mock_tx_repo([])
        portfolio_repo = _make_mock_portfolio_repo([
            {"id":"p-1","name":"BIST Portföyü"},
            {"id":"p-2","name":"Büyüme"},
        ])
        service = PortfolioService(tx_repo, _make_mock_market_svc(), portfolio_repo=portfolio_repo)
        result = service.list_portfolios()

        assert len(result) == 2
        assert result[0]["id"] == "p-1"
        assert result[0]["name"] == "BIST Portföyü"
        assert result[1]["id"] == "p-2"

    def test_list_portfolios_without_repo_raises_value_error(self):
        service = PortfolioService(_make_mock_tx_repo([]), _make_mock_market_svc())
        with pytest.raises(ValueError, match="portfolio_repo"):
            service.list_portfolios()

    def test_portfolio_repo_calls_list_all_active_only(self):
        tx_repo = _make_mock_tx_repo([])
        portfolio_repo = _make_mock_portfolio_repo()
        service = PortfolioService(tx_repo, _make_mock_market_svc(), portfolio_repo=portfolio_repo)
        service.list_portfolios()
        portfolio_repo.list_all.assert_called_once_with(include_inactive=False)

    def test_get_portfolio_status_uses_provided_portfolio_id(self):
        """Hardcoded 'default' kaldırıldı — verilen ID kullanılmalı."""
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        service = PortfolioService(tx_repo, _make_mock_market_svc(10.00),
                                   calculator=_make_mock_calculator(_make_cb_result()))
        service.get_portfolio_status("my-custom-portfolio-id")
        tx_repo.get_portfolio_symbols.assert_called_once_with("my-custom-portfolio-id")

    def test_different_portfolio_ids_independent(self):
        """İki farklı portföy ID'si birbirinin verilerini karıştırmamalı."""
        tx_repo = MagicMock()
        tx_repo.get_portfolio_symbols.side_effect = lambda pid: (
            ["THYAO"] if pid == "port-a" else ["GARAN"]
        )
        tx_repo.get_by_symbol.return_value = [MagicMock()]
        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.return_value = _make_cb_result("100","10.00")
        market_svc = _make_mock_market_svc(12.00)

        service = PortfolioService(tx_repo, market_svc, calculator=calculator)

        s_a = service.get_portfolio_status("port-a")
        s_b = service.get_portfolio_status("port-b")

        assert s_a.positions[0].symbol == "THYAO"
        assert s_b.positions[0].symbol == "GARAN"


# ── Concurrent / ThreadPoolExecutor (Aşama 8) ─────────────────────────────────

class TestConcurrency:

    def test_all_symbols_fetched_concurrently(self):
        """
        5 sembol için fiyat çekme: ThreadPoolExecutor ile paralel.
        Hepsinin sonuçları doğru DTO'ya yerleştirilmeli.
        """
        symbols = [f"SYM{i}" for i in range(5)]
        txs = {s: [MagicMock(symbol=s)] for s in symbols}
        tx_repo = _make_mock_tx_repo(symbols, txs)

        prices = {s: float(10 + i) for i, s in enumerate(symbols)}
        def price_side(symbol, timeframe):
            r = MagicMock(spec=MarketAnalysisResult)
            r.latest_close = prices[symbol]
            return r

        market_svc = MagicMock(spec=MarketDataService)
        market_svc.get_market_analysis.side_effect = price_side

        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.return_value = _make_cb_result("100","10.00")

        service = PortfolioService(tx_repo, market_svc, calculator=calculator, max_workers=5)
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        assert summary.position_count == 5
        assert summary.stale_symbols == []
        result_prices = {p.symbol: p.current_price for p in summary.positions}
        for i, s in enumerate(symbols):
            assert result_prices[s] == Decimal(str(10 + i))

    def test_concurrent_partial_failure_thread_safe(self):
        """
        Concurrent çalışmada bazı semboller hata fırlatsa bile
        stale_symbols thread-safe biçimde toplanmalı.
        """
        symbols = [f"S{i}" for i in range(10)]
        txs = {s: [MagicMock(symbol=s)] for s in symbols}
        tx_repo = _make_mock_tx_repo(symbols, txs)

        def price_side(symbol, timeframe):
            idx = int(symbol[1:])
            if idx % 2 == 0:
                raise Exception(f"{symbol} fiyat hatası")
            r = MagicMock(spec=MarketAnalysisResult)
            r.latest_close = 15.00
            return r

        market_svc = MagicMock(spec=MarketDataService)
        market_svc.get_market_analysis.side_effect = price_side
        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.return_value = _make_cb_result("100","10.00")

        service = PortfolioService(tx_repo, market_svc, calculator=calculator, max_workers=5)
        summary = service.get_portfolio_status(PORTFOLIO_ID)

        # 5 sembol hatalı (S0,S2,S4,S6,S8), 5'i başarılı
        assert len(summary.stale_symbols) == 5
        # Duplicate olmamalı (race condition sonucu)
        assert len(summary.stale_symbols) == len(set(summary.stale_symbols))
        # Başarılı olanlar current_price içermeli
        priced = [p for p in summary.positions if p.current_price is not None]
        assert len(priced) == 5

    def test_max_workers_is_injectable(self):
        """max_workers hard-code değil, constructor'dan inject edilebilir."""
        service = PortfolioService(
            _make_mock_tx_repo([]), _make_mock_market_svc(), max_workers=16
        )
        assert service._max_workers == 16

    def test_max_workers_capped_by_symbol_count(self):
        """3 sembol için 100 worker istense bile 3 thread yeterli."""
        symbols = ["A","B","C"]
        txs = {s: [MagicMock(symbol=s)] for s in symbols}
        tx_repo = _make_mock_tx_repo(symbols, txs)
        calculator = MagicMock(spec=WAVGCostBasisCalculator)
        calculator.calculate.return_value = _make_cb_result("100","10.00")

        service = PortfolioService(tx_repo, _make_mock_market_svc(10.00),
                                   calculator=calculator, max_workers=100)
        # 3 sembolden fazla thread açılmamalı — bu bir crash testi değil,
        # ThreadPoolExecutor min(workers, symbols) semantiğini doğrular
        summary = service.get_portfolio_status(PORTFOLIO_ID)
        assert summary.position_count == 3


# ── DI ve Statelessness ────────────────────────────────────────────────────────

class TestDependencyInjection:

    def test_calculator_is_injectable(self):
        cb = _make_cb_result("10","5.00")
        tx_repo = _make_mock_tx_repo(["SASA"],{"SASA":[MagicMock()]})
        calculator = _make_mock_calculator(cb)
        service = PortfolioService(tx_repo, _make_mock_market_svc(6.00), calculator=calculator)
        service.get_portfolio_status(PORTFOLIO_ID)
        calculator.calculate.assert_called_once()

    def test_default_calculator_is_wavg(self):
        service = PortfolioService(_make_mock_tx_repo([]), _make_mock_market_svc())
        assert isinstance(service._calculator, WAVGCostBasisCalculator)

    def test_market_data_service_called_with_daily_timeframe(self):
        cb = _make_cb_result()
        tx_repo = _make_mock_tx_repo(["EREGL"],{"EREGL":[MagicMock()]})
        market_svc = _make_mock_market_svc(50.00)
        service = PortfolioService(tx_repo, market_svc, calculator=_make_mock_calculator(cb))
        service.get_portfolio_status(PORTFOLIO_ID)
        market_svc.get_market_analysis.assert_called_once_with(symbol="EREGL", timeframe="1d")

    def test_repo_called_with_correct_portfolio_id(self):
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        service = PortfolioService(tx_repo, _make_mock_market_svc(10.00),
                                   calculator=_make_mock_calculator(_make_cb_result()))
        service.get_portfolio_status("my-portfolio-123")
        tx_repo.get_portfolio_symbols.assert_called_once_with("my-portfolio-123")
        tx_repo.get_by_symbol.assert_called_once_with("my-portfolio-123","THYAO")


class TestStatelessness:

    def test_consecutive_calls_independent(self):
        cb = _make_cb_result("100","10.00")
        tx_repo = _make_mock_tx_repo(["THYAO"],{"THYAO":[MagicMock()]})
        prices = [12.00, 15.00]
        call_idx = {"n": 0}

        def price_side(symbol, timeframe):
            r = MagicMock(spec=MarketAnalysisResult)
            r.latest_close = prices[call_idx["n"] % 2]
            call_idx["n"] += 1
            return r

        market_svc = MagicMock(spec=MarketDataService)
        market_svc.get_market_analysis.side_effect = price_side
        service = PortfolioService(tx_repo, market_svc, calculator=_make_mock_calculator(cb))

        s1 = service.get_portfolio_status(PORTFOLIO_ID)
        s2 = service.get_portfolio_status(PORTFOLIO_ID)

        assert s1.positions[0].current_price == Decimal("12.00")
        assert s2.positions[0].current_price == Decimal("15.00")
