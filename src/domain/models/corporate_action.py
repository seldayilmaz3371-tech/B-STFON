"""
CorporateAction domain modeli — design doc corporate_actions DDL'i ile
birebir.

KAPSAM SINIRI (bu proje boyunca defalarca gerekçelendirilen bir karar,
burada da GEÇERLİ): Bu model yalnızca kurumsal aksiyon OLAYINI (event)
taşır — "X hissesi Y tarihinde Z oranında bölündü" gibi bir GERÇEĞİ
kaydeder. Bu olayın PORTFÖYLERE NASIL UYGULANACAĞI (cost basis'in
yeniden hesaplanması, pozisyon miktarının güncellenmesi) BU TURUN
KAPSAMI DIŞINDA — CostBasisCalculator hâlâ RIGHTS_USED/RIGHTS_SOLD/
REVERSE_SPLIT/MERGER için BusinessRuleError fırlatıyor (bkz.
cost_basis_calculator.py) çünkü gerçek golden dataset YOK.

is_applied alanı DDL'de ZATEN bu ayrımı destekliyor (varsayılan 0,
"portföylere uygulandı mı" sorusuna cevap veren AYRI bir bayrak) —
bu model, is_applied=0 ile "olay kaydedildi ama henüz işlenmedi"
durumunu doğal olarak temsil ediyor.

action_data alanı JSON olarak saklanıyor (DDL'in kendi tasarımı) —
burada action_type'a göre değişen serbest-formlu bir dict olarak
modellendi, KATI bir tip şeması DAYATILMADI (design doc'un kendi
örnekleri farklı action_type'lar için farklı alan kümeleri gösteriyor
— DividendEvent'in dividend_per_share'i, Split'in ratio'su gibi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any


_VALID_ACTION_TYPES = {
    "DIVIDEND", "BONUS_SHARE", "RIGHTS_ISSUE",
    "SPLIT", "REVERSE_SPLIT", "MERGER",
    "SPIN_OFF", "DELISTING",
}


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    action_type: str
    ex_date: date
    action_data: dict[str, Any] = field(default_factory=dict)
    announcement_date: date | None = None
    record_date: date | None = None
    payment_date: date | None = None
    is_confirmed: bool = False
    is_applied: bool = False
    source: str = "manual"
    notes: str | None = None
    raw_data: str | None = None
    action_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol boş olamaz.")
        if self.action_type not in _VALID_ACTION_TYPES:
            raise ValueError(
                f"action_type geçersiz: {self.action_type}. Geçerli: {_VALID_ACTION_TYPES}"
            )
        # NOT: action_data'nın action_type'a göre DOĞRU alanları taşıyıp
        # taşımadığı BURADA DOĞRULANMIYOR (örn. SPLIT için 'ratio' var mı) —
        # bu, gerçek bir uygulama mantığı (Faz H) gerektirir, şimdilik
        # yalnızca "bir olay kaydedildi" garantisi veriliyor.
