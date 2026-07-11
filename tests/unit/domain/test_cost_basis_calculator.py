"""
CostBasisCalculator golden dataset ve edge-case testleri.

Kaynak: BIST_TEFAS_Master_Design_Document.md Bölüm 4 (GD-001..GD-005).
Tolerans: tasarım belgesinde belirtilen ±0.01 TL (Decimal quantize ile
karşılaştırılıyor, float tolerans karşılaştırması KULLANILMIYOR).

GD-004 notu: DIVIDEND, pozisyonu (miktar/maliyet) ETKİLEMEZ ama
total_dividends alanına net tutar olarak eklenir — bu, önceden teslim
edilmiş test_portfolio_service.py'ı çalıştırarak tespit edilen gerçek
sözleşmedir (bkz. cost_basis_calculator.py revizyon notu); ilk taslakta
"tam no-op" varsayılmıştı, bu YANLIŞTI ve düzeltildi.

FIFO+SPLIT ve FIFO+BONUS_SHARE testleri GD referanslı DEĞİLDİR — bunlar
cost_basis_calculator.py'daki mühendislik uzantısının iç tutarlılığını
doğrular (bkz. o dosyadaki gerekçe notu). Test adlarında bu farkı
`test_engineering_extension_*` öneki ile ayırıyorum.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

import pytest

from src.domain.calculators.cost_basis_calculator import (
    FIFOCostBasisCalculator,
    WAVGCostBasisCalculator,
)
from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import (
    BusinessRuleError,
    InsufficientQuantityError,
)
from src.domain.models.transaction import Transaction

CENT = Decimal("0.01")


def q(value: Decimal) -> Decimal:
    """Karşılaştırma için 2 ondalığa yuvarla (±0.01 TL tolerans)."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def tx(
    symbol: str,
    ttype: TransactionType,
    day: str,
    quantity: str = "0",
    price: str = "0",
    split_ratio: str | None = None,
    net_amount: str | None = None,
) -> Transaction:
    return Transaction(
        symbol=symbol,
        transaction_type=ttype,
        timestamp=datetime.fromisoformat(day),
        quantity=Decimal(quantity),
        price=Decimal(price),
        split_ratio=Decimal(split_ratio) if split_ratio else None,
        net_amount=Decimal(net_amount) if net_amount is not None else None,
    )


# ── GD-001: Basit WAVG BUY/SELL ────────────────────────────────────────────────

def test_gd001_wavg_simple_buy_sell():
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("THYAO", TransactionType.BUY, "2024-02-01", "50", "14.00"),
        tx("THYAO", TransactionType.SELL, "2024-03-01", "80", "16.00"),
    ]
    result = WAVGCostBasisCalculator().calculate(transactions)

    assert result.total_quantity == Decimal("70")
    assert q(result.average_cost) == q(Decimal("11.333333"))
    assert q(result.total_cost_basis) == Decimal("793.33")
    assert q(result.realized_pnl) == Decimal("373.33")

    current_price = Decimal("18.00")
    assert q(result.current_value(current_price)) == Decimal("1260.00")
    assert q(result.unrealized_pnl(current_price)) == Decimal("466.67")


# ── GD-002: WAVG + Bedelsiz Hisse ──────────────────────────────────────────────

def test_gd002_wavg_bonus_share():
    transactions = [
        tx("GARAN", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("GARAN", TransactionType.BONUS_SHARE, "2024-03-15", "50", "0.00"),
        tx("GARAN", TransactionType.SELL, "2024-05-01", "75", "15.00"),
    ]
    result = WAVGCostBasisCalculator().calculate(transactions)

    assert result.total_quantity == Decimal("75")
    assert q(result.average_cost) == q(Decimal("6.666667"))
    assert q(result.total_cost_basis) == Decimal("500.00")
    assert q(result.realized_pnl) == Decimal("625.00")

    # Kritik doğrulama: realize edilen + kalan maliyet ≈ orijinal harcama
    assert q(result.total_cost_basis + Decimal("500.00")) == Decimal("1000.00")


# ── GD-003: FIFO + Çoklu Lot ────────────────────────────────────────────────────

def test_gd003_fifo_multi_lot():
    transactions = [
        tx("AKBNK", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("AKBNK", TransactionType.BUY, "2024-02-01", "50", "15.00"),
        tx("AKBNK", TransactionType.BUY, "2024-03-01", "30", "12.00"),
        tx("AKBNK", TransactionType.SELL, "2024-04-01", "120", "20.00"),
    ]
    result = FIFOCostBasisCalculator().calculate(transactions)

    assert result.total_quantity == Decimal("60")
    assert q(result.realized_pnl) == Decimal("1100.00")

    # Kalan lot kuyruğu doğrulaması — tasarım belgesiyle birebir
    assert len(result.lots) == 2
    assert result.lots[0].quantity == Decimal("30")
    assert result.lots[0].unit_cost == Decimal("15.00")
    assert result.lots[1].quantity == Decimal("30")
    assert result.lots[1].unit_cost == Decimal("12.00")

    assert q(result.total_cost_basis) == Decimal("810.00")

    current_price = Decimal("18.00")
    assert q(result.current_value(current_price)) == Decimal("1080.00")
    assert q(result.unrealized_pnl(current_price)) == Decimal("270.00")


def test_gd003_comparison_wavg_would_differ():
    """
    Tasarım belgesi GD-003'ün karşılaştırma notunu doğrular:
    aynı işlem seti WAVG ile FARKLI realized_pnl üretmeli (107 TL fark).
    Bu, iki stratejinin GERÇEKTEN farklı algoritmalar olduğunun kanıtı —
    yanlışlıkla birbirinin kopyası hâline gelmediklerini garanti eder.
    """
    transactions = [
        tx("AKBNK", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("AKBNK", TransactionType.BUY, "2024-02-01", "50", "15.00"),
        tx("AKBNK", TransactionType.BUY, "2024-03-01", "30", "12.00"),
        tx("AKBNK", TransactionType.SELL, "2024-04-01", "120", "20.00"),
    ]
    fifo = FIFOCostBasisCalculator().calculate(transactions)
    wavg = WAVGCostBasisCalculator().calculate(transactions)

    assert q(wavg.realized_pnl) == Decimal("993.33")
    assert q(fifo.realized_pnl - wavg.realized_pnl) == Decimal("106.67")


# ── GD-004: Temettü — pozisyonu ETKİLEMEMELİ, ama total_dividends'e eklenir ────

def test_gd004_dividend_does_not_affect_position_but_accumulates():
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "200", "12.00"),
        tx(
            "THYAO", TransactionType.DIVIDEND, "2024-05-15",
            quantity="200", price="1.50", net_amount="270.00",
        ),
    ]
    for calculator in (WAVGCostBasisCalculator(), FIFOCostBasisCalculator()):
        result = calculator.calculate(transactions)
        assert result.total_quantity == Decimal("200")
        assert q(result.average_cost) == Decimal("12.00")
        assert q(result.total_cost_basis) == Decimal("2400.00")
        assert result.realized_pnl == Decimal("0")
        assert q(result.total_dividends) == Decimal("270.00")

        # GD-004 "Total return (temettü dahil)" doğrulaması:
        # 600 (unrealized @ 15.00) + 0 (realized) + 270 (dividend) = 870
        current_price = Decimal("15.00")
        assert q(result.total_return(current_price)) == Decimal("870.00")
        expected_pct = (Decimal("870.00") / Decimal("2400.00") * 100)
        assert q(result.total_return(current_price) / result.total_cost_basis * 100) == q(expected_pct)


def test_dividend_without_net_amount_raises():
    with pytest.raises(ValueError):
        tx("THYAO", TransactionType.DIVIDEND, "2024-05-15", quantity="200", price="1.50")


# ── GD-005: Split — toplam maliyet KORUNUR (WAVG) ──────────────────────────────

def test_gd005_wavg_split():
    transactions = [
        tx("TUPRS", TransactionType.BUY, "2024-01-02", "1000", "100.00"),
        tx("TUPRS", TransactionType.SPLIT, "2024-06-01", split_ratio="10"),
    ]
    result = WAVGCostBasisCalculator().calculate(transactions)

    assert result.total_quantity == Decimal("10000")
    assert q(result.average_cost) == Decimal("10.00")
    assert q(result.total_cost_basis) == Decimal("100000.00")

    current_price = Decimal("11.00")
    assert q(result.unrealized_pnl(current_price)) == Decimal("10000.00")


# ── Mühendislik uzantısı: FIFO + SPLIT / BONUS_SHARE (GD referanslı DEĞİL) ─────

def test_engineering_extension_fifo_split_preserves_total_cost():
    transactions = [
        tx("TUPRS", TransactionType.BUY, "2024-01-02", "100", "50.00"),
        tx("TUPRS", TransactionType.BUY, "2024-02-01", "50", "60.00"),
        tx("TUPRS", TransactionType.SPLIT, "2024-06-01", split_ratio="2"),
    ]
    result = FIFOCostBasisCalculator().calculate(transactions)

    # Split öncesi toplam maliyet: 100*50 + 50*60 = 8000
    # Split sonrası da 8000 KORUNMALI (her lot ayrı ayrı ratio uygulanır).
    assert q(result.total_cost_basis) == Decimal("8000.00")
    assert result.total_quantity == Decimal("300")  # (100+50) * 2
    assert result.lots[0].quantity == Decimal("200")
    assert result.lots[0].unit_cost == Decimal("25.00")
    assert result.lots[1].quantity == Decimal("100")
    assert result.lots[1].unit_cost == Decimal("30.00")


def test_engineering_extension_fifo_bonus_share_adds_zero_cost_lot():
    transactions = [
        tx("GARAN", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("GARAN", TransactionType.BONUS_SHARE, "2024-03-15", "50", "0.00"),
    ]
    result = FIFOCostBasisCalculator().calculate(transactions)

    assert result.total_quantity == Decimal("150")
    assert q(result.total_cost_basis) == Decimal("1000.00")  # değişmedi
    assert len(result.lots) == 2
    assert result.lots[1].unit_cost == Decimal("0")


# ── Edge-case: yetersiz miktar ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_sell_exceeding_position_raises(calculator):
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "10", "10.00"),
        tx("THYAO", TransactionType.SELL, "2024-01-03", "20", "12.00"),
    ]
    with pytest.raises(InsufficientQuantityError) as exc_info:
        calculator.calculate(transactions)
    assert exc_info.value.symbol == "THYAO"
    assert exc_info.value.requested == Decimal("20")
    assert exc_info.value.available == Decimal("10")


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_bonus_share_without_position_raises(calculator):
    transactions = [
        tx("THYAO", TransactionType.BONUS_SHARE, "2024-01-02", "50", "0.00"),
    ]
    with pytest.raises(BusinessRuleError):
        calculator.calculate(transactions)


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_split_without_position_raises(calculator):
    transactions = [
        tx("THYAO", TransactionType.SPLIT, "2024-01-02", split_ratio="2"),
    ]
    with pytest.raises(BusinessRuleError):
        calculator.calculate(transactions)


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_empty_transaction_list_raises(calculator):
    with pytest.raises(BusinessRuleError):
        calculator.calculate([])


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_full_liquidation_then_rebuy_resets_cost_basis(calculator):
    """
    Pozisyon tamamen kapatılıp yeniden açıldığında eski maliyetin
    'hayalet' olarak sızmadığını doğrular — WAVG'daki total_cost=ZERO
    reset'inin ve FIFO'nun boş kuyruğunun doğru davrandığının kanıtı.
    """
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "10", "10.00"),
        tx("THYAO", TransactionType.SELL, "2024-01-10", "10", "12.00"),
        tx("THYAO", TransactionType.BUY, "2024-02-01", "5", "20.00"),
    ]
    result = calculator.calculate(transactions)
    assert result.total_quantity == Decimal("5")
    assert q(result.average_cost) == Decimal("20.00")
    assert q(result.total_cost_basis) == Decimal("100.00")
    assert q(result.realized_pnl) == Decimal("20.00")  # 10*(12-10)


# ── Transaction entity invariant testleri ──────────────────────────────────────

def test_transaction_rejects_non_positive_buy_quantity():
    with pytest.raises(ValueError):
        tx("THYAO", TransactionType.BUY, "2024-01-02", "0", "10.00")


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
@pytest.mark.parametrize(
    "ttype", [TransactionType.DEPOSIT, TransactionType.WITHDRAWAL, TransactionType.FEE]
)
def test_cash_only_types_are_noop_for_position(calculator, ttype):
    """
    DDL şemasında keşfedilen DEPOSIT/WITHDRAWAL/FEE — symbol_type='CASH'
    ile kullanılan, pozisyonu yapısal olarak etkilemeyen tipler.
    """
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("THYAO", ttype, "2024-01-05", "0", "0.00"),
    ]
    result = calculator.calculate(transactions)
    assert result.total_quantity == Decimal("100")
    assert q(result.total_cost_basis) == Decimal("1000.00")


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
@pytest.mark.parametrize(
    "ttype",
    [
        TransactionType.RIGHTS_SOLD,
        TransactionType.MERGER,
    ],
)
def test_unsupported_transaction_types_raise_explicit_error(calculator, ttype):
    """
    DDL'de tanımlı ama finansal mantığı golden dataset ile doğrulanmamış
    2 tip (DÜZELTME: REVERSE_SPLIT ve RIGHTS_USED bu turlarda listeden
    ÇIKARILDI — artık desteklenen tipler) — sessiz yanlış hesaplama
    yerine açık BusinessRuleError.
    """
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("THYAO", ttype, "2024-01-05", "10", "1.00"),
    ]
    with pytest.raises(BusinessRuleError):
        calculator.calculate(transactions)


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_rights_used_behaves_identically_to_buy(calculator):
    """
    DÜZELTME (bu turda eklendi): RIGHTS_USED artık destekleniyor —
    BUY ile MATEMATİKSEL OLARAK ÖZDEŞ. Bu test, RIGHTS_USED ile yapılan
    bir alımın, AYNI miktar/fiyatla yapılan bir BUY ile TAM OLARAK
    AYNI sonucu ürettiğini kanıtlıyor.
    """
    transactions_buy = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("THYAO", TransactionType.BUY, "2024-02-01", "50", "1.00"),  # rüçhan fiyatı SİMÜLE
    ]
    transactions_rights = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("THYAO", TransactionType.RIGHTS_USED, "2024-02-01", "50", "1.00"),
    ]
    result_buy = calculator.calculate(transactions_buy)
    result_rights = calculator.calculate(transactions_rights)

    assert result_buy.total_quantity == result_rights.total_quantity
    assert result_buy.total_cost_basis == result_rights.total_cost_basis
    assert result_buy.average_cost == result_rights.average_cost


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_reverse_split_halves_quantity_preserves_total_cost(calculator):
    """
    DÜZELTME (bu turda eklendi): REVERSE_SPLIT artık destekleniyor.
    1:10 ters bölünme — miktar 100'den 10'a düşmeli, TOPLAM maliyet
    (1000 TL) DEĞİŞMEMELİ (yalnızca birim maliyet 10x artmalı).
    """
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("THYAO", TransactionType.REVERSE_SPLIT, "2024-03-15", "0", "0", split_ratio="10"),
    ]
    result = calculator.calculate(transactions)
    assert result.total_quantity == Decimal("10")
    assert q(result.total_cost_basis) == Decimal("1000.00")  # KORUNUR


@pytest.mark.parametrize(
    "calculator", [WAVGCostBasisCalculator(), FIFOCostBasisCalculator()]
)
def test_reverse_split_without_position_raises(calculator):
    transactions = [
        tx("THYAO", TransactionType.REVERSE_SPLIT, "2024-01-02", "0", "0", split_ratio="10"),
    ]
    with pytest.raises(BusinessRuleError):
        calculator.calculate(transactions)


def test_split_then_reverse_split_returns_to_original_quantity():
    """
    MATEMATİKSEL DEĞİŞMEZ: 2:1 SPLIT sonrası 1:2 REVERSE_SPLIT (ratio=2),
    orijinal miktara GERİ DÖNMELİ — SPLIT ve REVERSE_SPLIT birbirinin
    TAM TERSİ olduğunu kanıtlayan en güçlü test.
    """
    calc = WAVGCostBasisCalculator()
    transactions = [
        tx("THYAO", TransactionType.BUY, "2024-01-02", "100", "10.00"),
        tx("THYAO", TransactionType.SPLIT, "2024-02-01", "0", "0", split_ratio="2"),
        tx("THYAO", TransactionType.REVERSE_SPLIT, "2024-03-01", "0", "0", split_ratio="2"),
    ]
    result = calc.calculate(transactions)
    assert result.total_quantity == Decimal("100")  # ORİJİNAL miktara DÖNDÜ
    assert q(result.total_cost_basis) == Decimal("1000.00")  # maliyet HİÇ değişmedi


@pytest.mark.parametrize(
    "ttype,expected",
    [
        (TransactionType.BUY, True),
        (TransactionType.SELL, True),
        (TransactionType.BONUS_SHARE, True),
        (TransactionType.SPLIT, True),
        (TransactionType.DIVIDEND, False),
        (TransactionType.TAX, False),
    ],
)
def test_affects_cost_basis(ttype, expected):
    assert ttype.affects_cost_basis() is expected


def test_transaction_rejects_negative_price():
    with pytest.raises(ValueError):
        tx("THYAO", TransactionType.BUY, "2024-01-02", "10", "-5.00")


def test_transaction_rejects_non_positive_bonus_quantity():
    with pytest.raises(ValueError):
        tx("THYAO", TransactionType.BONUS_SHARE, "2024-01-02", "0", "0.00")


def test_transaction_rejects_split_without_ratio():
    with pytest.raises(ValueError):
        Transaction(
            symbol="THYAO",
            transaction_type=TransactionType.SPLIT,
            timestamp=datetime.fromisoformat("2024-01-02"),
        )
