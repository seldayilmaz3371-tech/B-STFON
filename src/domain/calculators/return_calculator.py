"""
Getiri hesaplayıcı — Pure Domain (I/O yok, yalnızca matematik).

API sözleşmesi — BIST_TEFAS_Master_Design_Document.md Bölüm 1.4
"ReturnCalculator" interface contract'ından alındı, TEK BİR İSTİSNAYLA
(aşağıda gerekçeli):

TESPİT EDİLEN SPESİFİKASYON EKSİKLİĞİ:
  Dokümante edilen imza:
    calculate_holding_period_return(start_value, end_value, cashflows)
  Bu imzada nakit akışlı (Modified Dietz) durum için period BAŞLANGIÇ
  ve BİTİŞ tarihleri YOK. Modified Dietz formülü:
    R = (EMV - BMV - CF) / (BMV + Σ(CF_i × W_i))
    W_i = (period_end - CF_i_tarihi) / (period_end - period_start)
  Bu ağırlığı hesaplamak MATEMATIKSEL OLARAK period_start/period_end
  olmadan İMKANSIZ — cashflow'ların tarihleri period sınırlarına göre
  konumlandırılmalı. Bu, tercih meselesi değil, formülün doğası.

  KARAR: `period_start: date` ve `period_end: date` parametrelerini
  EKLEDİM (dokümante edilen imzadan sapma, ama zorunlu). Bu sapmayı
  burada açıkça belgeliyorum — sessizce "cashflows'un ilk/son tarihini
  kullan" gibi bir varsayımla ÖRTBAS ETMEDİM, çünkü bu yanlış sonuç
  üretirdi (period sınırları cashflow tarihleriyle örtüşmeyebilir).

GD-006 DOĞRULAMA KAPSAMI:
  calculate_twrr() ve basit (cashflow'suz) calculate_holding_period_return()
  GD-006 ile TAM doğrulandı. calculate_holding_period_return()'ün
  Modified Dietz dalı GD-006'da HİÇ EGZERSİZ EDİLMİYOR (GD-006, alt
  dönemlere bölüp her birinde basit HPR + calculate_twrr ile zincirleme
  kullanıyor — Modified Dietz'i atlıyor). Bu yüzden Modified Dietz
  dalı yalnızca standart ders kitabı formülüyle doğrulandı, golden
  dataset ile DEĞİL — bu ayrım testte açıkça işaretli.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, getcontext

import scipy.optimize

from src.domain.exceptions.domain_exceptions import (
    BusinessRuleError,
    CalculationError,
    ConvergenceError,
)

ZERO = Decimal("0")
ONE = Decimal("1")


class ReturnCalculator:
    """Stateless — tüm metodlar saf fonksiyon, instance state yok."""

    def calculate_holding_period_return(
        self,
        start_value: Decimal,
        end_value: Decimal,
        period_start: date,
        period_end: date,
        cashflows: list[tuple[Decimal, date]] | None = None,
    ) -> Decimal:
        """
        cashflows boşsa: Simple HPR = (end - start) / start.
        cashflows doluysa: Modified Dietz.

        Cashflow işareti sözleşmesi: POZİTİF = portföye giren para
        (ek yatırım), NEGATİF = çıkan para (çekim). Bu, GD-006'daki
        "+50.000 ek yatırım" gösterimiyle TUTARLI (calculate_mwrr'deki
        "negatif=çıkış" ile TERS yönde — bu iki metodun cashflow işaret
        sözleşmesi FARKLI, çünkü ikisi farklı matematiksel bağlamlardan
        geliyor: Modified Dietz'de CF portföyün büyümesine katkısı
        olarak pozitif; XIRR'de CF yatırımcının nakit akışı perspektifinden
        negatif (yatırımcı için çıkış). Bu ayrım docstring'de KASITLI
        olarak vurgulanıyor çünkü karıştırmak sessiz işaret hatasına
        yol açar.
        """
        if start_value <= ZERO:
            raise CalculationError(
                f"start_value pozitif olmalı, alınan: {start_value}"
            )
        if period_end <= period_start:
            raise CalculationError(
                f"period_end ({period_end}) period_start'tan ({period_start}) "
                "sonra olmalı."
            )

        cfs = cashflows or []
        if not cfs:
            return (end_value - start_value) / start_value

        total_days = (period_end - period_start).days
        net_cf = sum((cf for cf, _ in cfs), ZERO)
        weighted_cf = ZERO
        for cf, cf_date in cfs:
            if not (period_start <= cf_date <= period_end):
                raise BusinessRuleError(
                    f"Cashflow tarihi ({cf_date}) period sınırları "
                    f"[{period_start}, {period_end}] dışında.",
                )
            days_remaining = (period_end - cf_date).days
            weight = Decimal(days_remaining) / Decimal(total_days)
            weighted_cf += cf * weight

        denominator = start_value + weighted_cf
        if denominator == ZERO:
            raise CalculationError(
                "Modified Dietz paydası sıfır — start_value + ağırlıklı "
                "cashflow toplamı sıfıra eşit, getiri tanımsız."
            )
        return (end_value - start_value - net_cf) / denominator

    def calculate_twrr(self, sub_period_returns: list[Decimal]) -> Decimal:
        """Geometric linking: ∏(1 + r_i) - 1. GD-006 ile doğrulandı (±0.01%)."""
        if not sub_period_returns:
            raise BusinessRuleError(
                "calculate_twrr() en az bir alt dönem getirisi gerektirir."
            )
        product = ONE
        for r in sub_period_returns:
            product *= ONE + r
        return product - ONE

    def calculate_mwrr(self, cashflows: list[tuple[Decimal, date]]) -> Decimal:
        """
        XIRR (Extended Internal Rate of Return) — scipy.optimize.brentq.

        Cashflow işareti: NEGATİF = yatırımcı perspektifinden çıkış
        (yatırım yapıldı), POZİTİF = giriş (çekim/nihai değer) — GD-006
        "KARŞILAŞTIRMA (MWRR)" bölümündeki sözleşmeyle TUTARLI.

        float64'e düşüş BİLİNÇLİ: scipy.optimize.brentq Decimal ile
        çalışmaz (yalnızca float destekler). Bu, RiskCalculator'daki
        "risk hesaplamaları float64 kullanır" prensibiyle tutarlı bir
        istisna — parasal DEĞER değil, bir ORAN (rate) hesaplanıyor,
        bu yüzden precision kaybı burada realized_pnl gibi muhasebe
        kayıtlarındaki kadar kritik değil.
        """
        if len(cashflows) < 2:
            raise BusinessRuleError(
                "XIRR en az 2 nakit akışı (bir giriş, bir çıkış) gerektirir."
            )
        sorted_cf = sorted(cashflows, key=lambda x: x[1])
        t0 = sorted_cf[0][1]
        amounts = [float(cf) for cf, _ in sorted_cf]
        days = [(d - t0).days for _, d in sorted_cf]

        if all(a >= 0 for a in amounts) or all(a <= 0 for a in amounts):
            raise ConvergenceError(
                "XIRR",
                reason="Tüm nakit akışları aynı işarette — NPV(r) hiçbir "
                       "r için sıfırı kesmiyor, kök matematiksel olarak yok.",
            )

        def npv(rate: float) -> float:
            return float(sum(a / (1.0 + rate) ** (d / 365.0) for a, d in zip(amounts, days)))

        try:
            rate = scipy.optimize.brentq(npv, -0.999999, 100.0, maxiter=200)
        except ValueError as exc:
            raise ConvergenceError(
                "XIRR", reason=f"brentq kök bulamadı: {exc}"
            ) from exc

        return Decimal(str(rate))

    def calculate_annualized_return(self, total_return: Decimal, days: int) -> Decimal:
        """
        (1 + total_return)^(252/days) - 1.

        Decimal'de kesirli üs almak için ln/exp kullanılıyor (Decimal
        `**` operatörü yalnızca tam sayı üsleri güvenilir şekilde
        destekler) — bu, float'a düşmeden Decimal hassasiyetini korur.
        """
        if days <= 0:
            raise CalculationError(f"days pozitif olmalı, alınan: {days}")
        base = ONE + total_return
        if base <= ZERO:
            raise CalculationError(
                f"Toplam getiri ({total_return}) -%100 veya daha düşük — "
                "yıllıklandırma matematiksel olarak tanımsız (negatif "
                "tabanın kesirli kuvveti gerçek sayılarda tanımsız)."
            )
        exponent = Decimal(252) / Decimal(days)
        return (base.ln() * exponent).exp() - ONE
