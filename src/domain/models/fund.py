"""
Fund domain modeli — design doc funds DDL'i ile birebir.

KAPSAM: Yalnızca METADATA taşıyıcısı (fon adı, tipi, allocation, ücret
bilgisi). Fon fiyat geçmişi BURADA DEĞİL — PriceSeries/PriceRepository
zaten bunu ele alıyor (bkz. price_sync_service.py). Fund modeli,
"bu sembol hangi fon, kim yönetiyor, hangi tipte" sorularına cevap
veriyor; "bu fonun geçmiş NAV'ı ne" sorusuna PriceRepository cevap
veriyor — bu net ayrım BİLİNÇLİ (Single Responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal


_VALID_FUND_TYPES = {"YAT", "EMK", "BYF", "DIGER"}


@dataclass(frozen=True)
class Fund:
    fund_code: str
    fund_name: str
    fund_type: str
    currency: str = "TRY"
    umbrella_type: str | None = None
    founder: str | None = None
    stock_pct: Decimal | None = None
    bond_pct: Decimal | None = None
    repo_pct: Decimal | None = None
    foreign_stock_pct: Decimal | None = None
    gold_pct: Decimal | None = None
    other_pct: Decimal | None = None
    allocation_date: date | None = None
    last_nav: Decimal | None = None
    last_nav_date: date | None = None
    ytd_return: Decimal | None = None
    management_fee: Decimal | None = None
    is_active: bool = True
    last_updated: datetime | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.fund_code or not self.fund_code.strip():
            raise ValueError("fund_code boş olamaz.")
        if not self.fund_name or not self.fund_name.strip():
            raise ValueError("fund_name boş olamaz.")
        if self.fund_type not in _VALID_FUND_TYPES:
            raise ValueError(
                f"fund_type geçersiz: {self.fund_type}. Geçerli: {_VALID_FUND_TYPES}"
            )
        # NOT: allocation yüzdelerinin toplamının 100 olması ZORLANMIYOR
        # — TEFAS bu bilgiyi HER ZAMAN sağlamıyor (design doc'un kendi
        # notu: "opsiyonel — TEFAS her zaman sağlamıyor"), eksik/kısmi
        # allocation verisi GEÇERLİ bir durum, hata değil.
        for pct_field, value in [
            ("stock_pct", self.stock_pct), ("bond_pct", self.bond_pct),
            ("repo_pct", self.repo_pct), ("foreign_stock_pct", self.foreign_stock_pct),
            ("gold_pct", self.gold_pct), ("other_pct", self.other_pct),
        ]:
            if value is not None and (value < Decimal("0") or value > Decimal("100")):
                raise ValueError(f"{pct_field} 0-100 aralığında olmalı: {value}")
