"""ReturnCalculator testleri — GD-006 (TWRR/MWRR) ile doğrulanmış."""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import pytest

from src.domain.calculators.return_calculator import ReturnCalculator
from src.domain.exceptions.domain_exceptions import (
    BusinessRuleError,
    CalculationError,
    ConvergenceError,
)

PCT = Decimal("0.0001")


def q(value: Decimal) -> Decimal:
    return value.quantize(PCT, rounding=ROUND_HALF_UP)


# ── GD-006: TWRR ────────────────────────────────────────────────────────────
#
# KRİTİK BULGU — BU TESTLER YAZILIRKEN TESPİT EDİLDİ:
#   BIST_TEFAS_Master_Design_Document.md'deki GD-006 senaryosunun HPR
#   değerleri (0.10, 0.0625, 0.05882, 0.06667) DOĞRU, ama belgenin
#   kendi sonuç iddiası ("TWRR = 31.66%") YANLIŞ. Bağımsız doğrulama:
#
#     (1.10) × (1.0625) × (1.05882353) × (1.06666667) - 1 = 0.32 TAM OLARAK
#
#   Python Decimal ile kesir kesir hesapladım (bkz. aşağıdaki test) —
#   sonuç %32.00, %31.66 DEĞİL. Belgenin kendi ara adımları (1.16875,
#   sonraki çarpımlar) doğru HPR'lerle çarpıldığında %31.66'ya değil
#   %32.00'a ulaşıyor; belge muhtemelen elle yapılan bir ara yuvarlama
#   hatası içeriyor.
#
#   AYRICA MWRR (~%22.1 iddiası) DE YANLIŞ: Aynı nakit akışlarıyla
#   NPV(r)=0 denklemini scipy.optimize.brentq ile çözdüm VE r=%22.1'de
#   NPV'nin sıfırdan uzak (+9292) olduğunu, r=%30.89'da NPV'nin
#   makine hassasiyetinde sıfır (-3.6e-12) olduğunu bağımsız olarak
#   doğruladım (bkz. sohbet geçmişi — brentq + manuel NPV kontrolü).
#
#   KARAR: Testler DOKÜMANIN yanlış sayılarına değil, BAĞIMSIZ
#   DOĞRULANMIŞ matematiğe göre yazıldı (%32.00 TWRR, ~%30.89 MWRR).
#   GD-006'nın ANLATISI (TWRR > MWRR — yönetici iyi, yatırımcı
#   zamanlaması kötü) doğru sayılarla da GEÇERLİ KALIYOR (32.00 > 30.89),
#   yalnızca belgedeki iki sayısal değer yanlış. Bu belge hatası
#   ayrıca proje sahibine raporlanmalı.

def test_gd006_twrr_full_chain():
    calc = ReturnCalculator()

    hpr1 = calc.calculate_holding_period_return(
        Decimal("100000"), Decimal("110000"),
        date(2024, 1, 1), date(2024, 4, 1),
    )
    hpr2 = calc.calculate_holding_period_return(
        Decimal("160000"), Decimal("170000"),
        date(2024, 4, 1), date(2024, 7, 1),
    )
    hpr3 = calc.calculate_holding_period_return(
        Decimal("170000"), Decimal("180000"),
        date(2024, 7, 1), date(2024, 10, 1),
    )
    hpr4 = calc.calculate_holding_period_return(
        Decimal("150000"), Decimal("160000"),
        date(2024, 10, 1), date(2024, 12, 31),
    )

    # Bu 4 ara değer GD-006 ile TAM eşleşiyor (belgenin HPR hesapları doğru):
    assert q(hpr1) == Decimal("0.1000")
    assert q(hpr2) == Decimal("0.0625")
    assert q(hpr3) == Decimal("0.0588")
    assert q(hpr4) == Decimal("0.0667")

    twrr = calc.calculate_twrr([hpr1, hpr2, hpr3, hpr4])
    # GD-006 "%31.66" diyor — YUKARIDAKİ NOTA GÖRE bu YANLIŞ.
    # Doğru değer %32.00 (bağımsız doğrulandı, aşağıya bkz.).
    assert q(twrr) == Decimal("0.3200")


def test_gd006_arithmetic_independently_verified():
    """
    GD-006'nın kendi ara adımlarını (1.10, 1.0625, 1.05882, 1.06667)
    yuvarlanmamış tam kesirlerle çarparak belgenin %31.66 iddiasının
    yanlış olduğunu kanıtlar.
    """
    exact_product = (
        (Decimal("110000") / Decimal("100000"))
        * (Decimal("170000") / Decimal("160000"))
        * (Decimal("180000") / Decimal("170000"))
        * (Decimal("160000") / Decimal("150000"))
    )
    assert exact_product == Decimal("1.32")  # tam olarak, kesirler sadeleşiyor
    assert q(exact_product - 1) == Decimal("0.3200")


def test_gd006_mwrr_independently_verified():
    """
    GD-006 '~%22.1' diyor. brentq ile bulunan kökte NPV≈0 olduğunu,
    %22.1'de İSE NPV'nin sıfırdan uzak olduğunu doğrudan göstererek
    belgenin bu iddiasının da yanlış olduğunu kanıtlar.
    """
    calc = ReturnCalculator()
    cashflows = [
        (Decimal("-100000"), date(2024, 1, 1)),
        (Decimal("-50000"), date(2024, 4, 1)),
        (Decimal("30000"), date(2024, 10, 1)),
        (Decimal("160000"), date(2024, 12, 31)),
    ]
    mwrr = calc.calculate_mwrr(cashflows)

    # Doğru kökte NPV sıfıra çok yakın olmalı (brentq'un tanımı gereği zaten
    # öyle olacak, ama bunu burada da bağımsızca teyit ediyoruz):
    def npv(rate: float) -> float:
        t0 = date(2024, 1, 1)
        amounts = [-100000.0, -50000.0, 30000.0, 160000.0]
        days = [(d - t0).days for d in (date(2024,1,1), date(2024,4,1), date(2024,10,1), date(2024,12,31))]
        return sum(a / (1 + rate) ** (d / 365.0) for a, d in zip(amounts, days))

    assert abs(npv(float(mwrr))) < 0.01  # kökte NPV≈0
    assert abs(npv(0.221)) > 1000  # belgenin iddia ettiği %22.1'de NPV SIFIRDAN UZAK
    assert abs(mwrr - Decimal("0.3089")) < Decimal("0.001")  # doğru değer ~%30.89


def test_twrr_greater_than_mwrr_matches_gd006_narrative():
    """GD-006'nın anlatısı: TWRR > MWRR (yönetici iyi, yatırımcı zamanlaması kötü)."""
    calc = ReturnCalculator()
    hprs = [Decimal("0.10"), Decimal("0.0625"), Decimal("0.05882"), Decimal("0.06667")]
    twrr = calc.calculate_twrr(hprs)
    cashflows = [
        (Decimal("-100000"), date(2024, 1, 1)),
        (Decimal("-50000"), date(2024, 4, 1)),
        (Decimal("30000"), date(2024, 10, 1)),
        (Decimal("160000"), date(2024, 12, 31)),
    ]
    mwrr = calc.calculate_mwrr(cashflows)
    assert twrr > mwrr


# ── Simple HPR (cashflow'suz) ────────────────────────────────────────────────

def test_simple_hpr_no_cashflows():
    calc = ReturnCalculator()
    result = calc.calculate_holding_period_return(
        Decimal("100"), Decimal("110"), date(2024, 1, 1), date(2024, 2, 1)
    )
    assert result == Decimal("0.1")


def test_hpr_rejects_non_positive_start_value():
    calc = ReturnCalculator()
    with pytest.raises(CalculationError):
        calc.calculate_holding_period_return(
            Decimal("0"), Decimal("100"), date(2024, 1, 1), date(2024, 2, 1)
        )


def test_hpr_rejects_invalid_period():
    calc = ReturnCalculator()
    with pytest.raises(CalculationError):
        calc.calculate_holding_period_return(
            Decimal("100"), Decimal("110"), date(2024, 2, 1), date(2024, 1, 1)
        )


# ── Modified Dietz (GD referanslı DEĞİL — yalnızca ders kitabı doğrulaması) ────

def test_modified_dietz_textbook_example():
    """
    Standart ders kitabı örneği (GD-006'da EGZERSİZ EDİLMİYOR — bkz.
    return_calculator.py modül docstring'i):
      BMV=1000, EMV=1200, tek cashflow +100 tam periyodun ortasında (gün 50/100)
      Modified Dietz: (1200 - 1000 - 100) / (1000 + 100*0.5) = 100/1050 = 0.09524
    """
    calc = ReturnCalculator()
    result = calc.calculate_holding_period_return(
        Decimal("1000"), Decimal("1200"),
        date(2024, 1, 1), date(2024, 4, 10),  # 100 gün
        cashflows=[(Decimal("100"), date(2024, 2, 20))],  # ~50. gün
    )
    # Gün sayımı tam 50 olmayabilir (ay uzunlukları farklı) — geniş tolerans
    assert abs(result - Decimal("0.09524")) < Decimal("0.005")


def test_modified_dietz_rejects_cashflow_outside_period():
    calc = ReturnCalculator()
    with pytest.raises(BusinessRuleError):
        calc.calculate_holding_period_return(
            Decimal("1000"), Decimal("1200"),
            date(2024, 1, 1), date(2024, 2, 1),
            cashflows=[(Decimal("100"), date(2024, 3, 1))],  # period dışında
        )


# ── MWRR edge cases ──────────────────────────────────────────────────────────

def test_mwrr_requires_at_least_two_cashflows():
    calc = ReturnCalculator()
    with pytest.raises(BusinessRuleError):
        calc.calculate_mwrr([(Decimal("-100"), date(2024, 1, 1))])


def test_mwrr_same_sign_cashflows_raises_convergence_error():
    calc = ReturnCalculator()
    with pytest.raises(ConvergenceError):
        calc.calculate_mwrr([
            (Decimal("100"), date(2024, 1, 1)),
            (Decimal("100"), date(2024, 2, 1)),
        ])


# ── calculate_annualized_return ─────────────────────────────────────────────

def test_annualized_return_one_year():
    calc = ReturnCalculator()
    result = calc.calculate_annualized_return(Decimal("0.20"), 252)
    assert abs(result - Decimal("0.20")) < Decimal("0.0001")


def test_annualized_return_half_year_doubles_approximately():
    calc = ReturnCalculator()
    # 10% getiri 126 günde (yarım yıl) -> yıllıklandırılmış ~21%
    result = calc.calculate_annualized_return(Decimal("0.10"), 126)
    expected = (Decimal("1.10") ** 2) - 1  # yaklaşık kontrol
    assert abs(result - expected) < Decimal("0.001")


def test_annualized_return_total_wipeout_raises():
    calc = ReturnCalculator()
    with pytest.raises(CalculationError):
        calc.calculate_annualized_return(Decimal("-1.5"), 100)  # -150% getiri
