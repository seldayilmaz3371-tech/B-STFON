"""
CostBasisCalculator (WAVG/FIFO) için property-based testler (hypothesis).

GEREKÇE: Örnek-tabanlı testler (test_cost_basis_calculator.py, GD-001→005)
BELİRLİ senaryoları doğruluyor. Bu dosya FARKLI bir soru soruyor:
"HANGİ rastgele geçerli BUY/SELL dizisi verilirse verilsin, HER ZAMAN
doğru kalması gereken matematiksel değişmezler (invariant) NELERDİR,
ve GERÇEKTEN her zaman doğru mu?" Bu, örnek-tabanlı testlerin
KAÇIRABİLECEĞİ kenar durumları (özellikle belirli ondalık/yuvarlama
kombinasyonları) bulmak için tasarlandı.

KAPSAM: Yalnızca BUY/SELL dizileri (DIVIDEND/SPLIT/BONUS_SHARE gibi
diğer transaction_type'lar bu turun kapsamı dışında — ayrı bir
strateji/invariant seti gerektirir).

STRATEJİ TASARIMI: valid_buy_sell_sequence(), HİÇBİR SELL'in o ana
kadarki net pozisyonu AŞMAMASINI garanti ediyor — aksi halde
InsufficientQuantityError beklenen (ve zaten example-based testlerde
kapsanan) bir davranış olurdu, bu invariant testlerinin KONUSU DEĞİL.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from hypothesis import given, settings, strategies as st

from src.domain.calculators.cost_basis_calculator import (
    FIFOCostBasisCalculator,
    WAVGCostBasisCalculator,
)
from src.domain.enums.transaction_type import TransactionType
from src.domain.models.transaction import Transaction

_BASE_DATE = datetime(2024, 1, 1)


@st.composite
def valid_buy_sell_sequence(draw, max_transactions: int = 8):
    """
    Rastgele ama GEÇERLİ bir BUY/SELL dizisi üretir — hiçbir SELL,
    o ana kadarki net pozisyonu AŞMAZ.

    Returns:
        list[Transaction]
    """
    n = draw(st.integers(min_value=1, max_value=max_transactions))
    current_qty = Decimal("0")
    transactions: list[Transaction] = []

    for i in range(n):
        price = draw(st.decimals(
            min_value=Decimal("0.01"), max_value=Decimal("1000"),
            places=2, allow_nan=False, allow_infinity=False,
        ))
        action = draw(st.sampled_from(["BUY", "SELL"])) if current_qty > 0 else "BUY"

        if action == "BUY":
            qty = draw(st.decimals(
                min_value=Decimal("1"), max_value=Decimal("500"),
                places=4, allow_nan=False, allow_infinity=False,
            ))
            transactions.append(Transaction(
                symbol="TEST", transaction_type=TransactionType.BUY,
                timestamp=_BASE_DATE + timedelta(days=i), quantity=qty, price=price,
            ))
            current_qty += qty
        else:
            qty = draw(st.decimals(
                min_value=Decimal("0.0001"), max_value=current_qty,
                places=4, allow_nan=False, allow_infinity=False,
            ))
            transactions.append(Transaction(
                symbol="TEST", transaction_type=TransactionType.SELL,
                timestamp=_BASE_DATE + timedelta(days=i), quantity=qty, price=price,
            ))
            current_qty -= qty

    return transactions


def _buy_total_qty(transactions: list[Transaction]) -> Decimal:
    return sum((t.quantity for t in transactions if t.transaction_type is TransactionType.BUY), Decimal("0"))


def _sell_total_qty(transactions: list[Transaction]) -> Decimal:
    return sum((t.quantity for t in transactions if t.transaction_type is TransactionType.SELL), Decimal("0"))


def _buy_prices(transactions: list[Transaction]) -> list[Decimal]:
    return [t.price for t in transactions if t.transaction_type is TransactionType.BUY]


# ── Invariant 1: Miktar korunumu ─────────────────────────────────────────────

@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_wavg_quantity_conservation(transactions):
    """
    HER ZAMAN doğru olmalı: nihai total_quantity, TAM OLARAK
    (toplam BUY miktarı - toplam SELL miktarı)'na eşit olmalı —
    hiçbir 'kayıp' veya 'fazladan' miktar oluşmamalı.
    """
    result = WAVGCostBasisCalculator().calculate(transactions)
    expected = _buy_total_qty(transactions) - _sell_total_qty(transactions)
    assert result.total_quantity == expected


@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_fifo_quantity_conservation(transactions):
    result = FIFOCostBasisCalculator().calculate(transactions)
    expected = _buy_total_qty(transactions) - _sell_total_qty(transactions)
    assert result.total_quantity == expected


# ── Invariant 2: Negatif olmayan maliyet ────────────────────────────────────

@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_wavg_cost_basis_never_negative(transactions):
    result = WAVGCostBasisCalculator().calculate(transactions)
    assert result.total_cost_basis >= Decimal("0")
    assert result.average_cost >= Decimal("0")


@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_fifo_cost_basis_never_negative(transactions):
    result = FIFOCostBasisCalculator().calculate(transactions)
    assert result.total_cost_basis >= Decimal("0")
    assert result.average_cost >= Decimal("0")


# ── Invariant 3: WAVG ortalama maliyet, BUY fiyat aralığı İÇİNDE kalmalı ────

@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_wavg_average_cost_bounded_by_buy_price_range(transactions):
    """
    MATEMATİKSEL DEĞİŞMEZ: Ağırlıklı ortalama, DOĞASI GEREĞİ minimum
    ve maksimum girdi değerleri arasında kalmalıdır (bu, WAVG'ın
    TANIMINDAN kaynaklanan bir gerçek — herhangi bir ağırlıklı
    ortalama, ağırlıklandırılan değerlerin dışına ÇIKAMAZ). Pozisyon
    tamamen kapatılmadıysa (total_quantity > 0), average_cost, TÜM
    BUY işlemlerinin fiyat aralığı İÇİNDE olmalı.
    """
    result = WAVGCostBasisCalculator().calculate(transactions)
    if result.total_quantity > Decimal("0"):
        prices = _buy_prices(transactions)
        assert min(prices) <= result.average_cost <= max(prices), (
            f"average_cost ({result.average_cost}) BUY fiyat aralığının "
            f"[{min(prices)}, {max(prices)}] DIŞINDA — matematiksel olarak İMKANSIZ olmalıydı."
        )


# ── Invariant 4: total_cost_basis ≈ average_cost × total_quantity ──────────

@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_wavg_cost_basis_consistency(transactions):
    """
    total_cost_basis, average_cost × total_quantity'e YUVARLAMA
    TOLERANSI içinde eşit olmalı (tanım gereği: average_cost =
    total_cost_basis / total_quantity).
    """
    result = WAVGCostBasisCalculator().calculate(transactions)
    if result.total_quantity > Decimal("0"):
        implied_cost_basis = result.average_cost * result.total_quantity
        diff = abs(result.total_cost_basis - implied_cost_basis)
        # Tolerans: quantity'nin 4 ondalık hassasiyeti nedeniyle küçük
        # yuvarlama farkları BEKLENİYOR — mutlak sıfır eşitliği ARANMIYOR.
        assert diff < Decimal("0.01"), (
            f"total_cost_basis ({result.total_cost_basis}) ile "
            f"average_cost×total_quantity ({implied_cost_basis}) arasında "
            f"beklenenden BÜYÜK fark: {diff}"
        )


# ── Invariant 5: Pozisyon tam kapandığında maliyet SIFIR olmalı ────────────

@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_wavg_fully_closed_position_has_zero_cost_basis(transactions):
    result = WAVGCostBasisCalculator().calculate(transactions)
    if result.total_quantity == Decimal("0"):
        assert result.total_cost_basis == Decimal("0")


@given(valid_buy_sell_sequence())
@settings(max_examples=200, deadline=None)
def test_fifo_fully_closed_position_has_zero_cost_basis(transactions):
    result = FIFOCostBasisCalculator().calculate(transactions)
    if result.total_quantity == Decimal("0"):
        assert result.total_cost_basis == Decimal("0")


# ── Invariant 6: WAVG ve FIFO, TEK bir BUY + TEK bir TAM SELL için AYNI sonucu vermeli ──

@given(
    qty=st.decimals(min_value=Decimal("1"), max_value=Decimal("1000"), places=4, allow_nan=False, allow_infinity=False),
    buy_price=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("1000"), places=2, allow_nan=False, allow_infinity=False),
    sell_price=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("1000"), places=2, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=None)
def test_wavg_and_fifo_agree_on_single_lot(qty, buy_price, sell_price):
    """
    MATEMATİKSEL DEĞİŞMEZ: TEK bir BUY lot'u varsa (birden fazla farklı
    fiyatlı BUY YOKSA), WAVG ve FIFO'nun realized_pnl'i BİREBİR AYNI
    olmalı — "ağırlıklı ortalama" ile "ilk giren ilk çıkar" arasındaki
    fark, YALNIZCA BİRDEN FAZLA farklı-fiyatlı lot olduğunda ortaya
    çıkar; tek lot'ta iki yöntem de MATEMATİKSEL OLARAK ÖZDEŞTİR.
    """
    transactions = [
        Transaction(symbol="TEST", transaction_type=TransactionType.BUY,
                    timestamp=_BASE_DATE, quantity=qty, price=buy_price),
        Transaction(symbol="TEST", transaction_type=TransactionType.SELL,
                    timestamp=_BASE_DATE + timedelta(days=1), quantity=qty, price=sell_price),
    ]
    wavg_result = WAVGCostBasisCalculator().calculate(transactions)
    fifo_result = FIFOCostBasisCalculator().calculate(transactions)

    assert abs(wavg_result.realized_pnl - fifo_result.realized_pnl) < Decimal("0.01")
