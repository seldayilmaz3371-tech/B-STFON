"""
Portfolio domain modeli birim testleri.

DÜZELTME (bu turda bulundu — gerçek bir test boşluğu): Portfolio
domain modelinin __post_init__ validasyonu (benchmark_code, cost_method/
LIFO reddi, inception_date) hiçbir yerde DOĞRUDAN test edilmiyordu —
yalnızca test_portfolio_service.py'nin servis-katmanı entegrasyon
testleri DOLAYLI olarak bazı yolları kapsıyordu. Bu dosya o boşluğu
kapatıyor.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.domain.enums.cost_method import CostMethod
from src.domain.enums.currency_code import CurrencyCode
from src.domain.exceptions.domain_exceptions import ValidationError
from src.domain.models.portfolio import KNOWN_BENCHMARKS, BenchmarkInfo, Portfolio


def _make_portfolio(**overrides):
    defaults = dict(
        id="test-portfolio-id",
        name="Test Portföyü", currency=CurrencyCode.TRY, cost_method=CostMethod.WAVG,
        inception_date=date(2023, 1, 1),
    )
    defaults.update(overrides)
    return Portfolio(**defaults)


# ── benchmark_code validasyonu ───────────────────────────────────────────────

def test_portfolio_accepts_none_benchmark():
    """benchmark_code opsiyonel — None GEÇERLİ olmalı (henüz seçilmemiş)."""
    portfolio = _make_portfolio(benchmark_code=None)
    assert portfolio.benchmark_code is None


@pytest.mark.parametrize("benchmark", [b.code for b in KNOWN_BENCHMARKS])
def test_portfolio_accepts_all_known_benchmarks(benchmark):
    """
    KNOWN_BENCHMARKS listesindeki HER kod GEÇERLİ kabul edilmeli —
    bu, bu turda genişletilen 7-endeksli listenin GERÇEKTEN devrede
    olduğunu kanıtlıyor (yalnızca eski 3-endeksli liste değil).
    """
    portfolio = _make_portfolio(benchmark_code=benchmark)
    assert portfolio.benchmark_code == benchmark


def test_portfolio_rejects_unknown_benchmark():
    with pytest.raises(ValidationError, match="bilinen BIST endeks"):
        _make_portfolio(benchmark_code="TAMAMEN_UYDURMA_KOD")


def test_known_benchmarks_have_no_duplicate_codes():
    codes = [b.code for b in KNOWN_BENCHMARKS]
    assert len(codes) == len(set(codes))


def test_known_benchmarks_all_use_is_suffix():
    """Tüm bilinen benchmark kodları YFinanceAdapter'ın .IS konvansiyonunu takip etmeli."""
    for b in KNOWN_BENCHMARKS:
        assert b.code.endswith(".IS"), f"{b.code} .IS son eki taşımıyor"


# ── cost_method / LIFO reddi ─────────────────────────────────────────────────

def test_portfolio_accepts_wavg():
    portfolio = _make_portfolio(cost_method=CostMethod.WAVG)
    assert portfolio.cost_method == CostMethod.WAVG


def test_portfolio_accepts_fifo():
    portfolio = _make_portfolio(cost_method=CostMethod.FIFO)
    assert portfolio.cost_method == CostMethod.FIFO


def test_portfolio_rejects_lifo():
    """
    KRİTİK: DB CHECK constraint LIFO'ya izin veriyor ama HİÇBİR
    CostBasisCalculator onu implemente etmiyor. CostMethod enum'unun
    KENDİSİ LIFO üyesi taşımıyor (bilinçli — bkz. cost_method.py) —
    bu yüzden CostMethod.LIFO diye bir şey YAZILAMAZ bile.

    GERÇEK RİSK SENARYOSU: Bir repository, DB'den ham "LIFO" string'ini
    okuyup enum'a ÇEVİRMEDEN doğrudan Portfolio'ya geçirirse (Python
    dataclass'ları runtime'da tip ZORLAMAZ) — bu ham string senaryosu
    test ediliyor, CostMethod.LIFO (mevcut olmayan) DEĞİL.
    """
    with pytest.raises(ValidationError):
        _make_portfolio(cost_method="LIFO")  # ham string, enum DEĞİL — bkz. yukarıdaki gerekçe


# ── inception_date validasyonu ───────────────────────────────────────────────

def test_portfolio_rejects_future_inception_date():
    future_date = date.today() + timedelta(days=30)
    with pytest.raises(ValidationError):
        _make_portfolio(inception_date=future_date)


def test_portfolio_accepts_today_as_inception_date():
    portfolio = _make_portfolio(inception_date=date.today())
    assert portfolio.inception_date == date.today()


# ── isim validasyonu ──────────────────────────────────────────────────────────

def test_portfolio_rejects_empty_name():
    with pytest.raises(ValidationError):
        _make_portfolio(name="")


def test_portfolio_rejects_whitespace_only_name():
    with pytest.raises(ValidationError):
        _make_portfolio(name="   ")
