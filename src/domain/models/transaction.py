"""
Transaction domain modeli — minimal, bloklayan alan seti.

Kapsam kararı (bilinçli daraltma):
  Tasarım belgesi Transaction için "immutability kuralları belgelenmiş"
  diyor ve reversal (iptal) mekanizması, portfolio_id, created_at gibi
  persistence-odaklı alanlardan bahsediyor. Bu alanlar BİLEREK bu turda
  eklenmedi çünkü:
    - reversal mekanizması bir repository/service kararı (Faz B/C),
      CostBasisCalculator'ın hesaplama mantığını etkilemiyor
    - portfolio_id, created_at gibi alanlar persistence sorumluluğu;
      hesaplama katmanı yalnızca "bu sembol için sıralı işlem listesi"
      görmeli (Single Responsibility)
  Bu yüzden burada yalnızca CostBasisCalculator'ın ihtiyaç duyduğu
  alan seti var. Faz B'de repository'ler yazılırken bu sınıf
  genişletilecek (yeni alan eklemek, mevcut alanları kullanan hiçbir
  kodu bozmaz — dataclass'a yeni optional alan eklemek geriye uyumlu).

Immutability kararı:
  frozen=True. Gerekçe: Transaction bir muhasebe kaydıdır, oluşturulduktan
  sonra değiştirilemez olmalı (audit trail bütünlüğü). Yanlış girilen
  bir işlem "reversal" (ters kayıt) ile telafi edilir, in-place
  düzenleme ile değil. Bu kural Faz B'de repository seviyesinde de
  (UPDATE yasak, yalnızca INSERT) uygulanacak.

Emin olunmayan nokta (varsayım yapılmadı):
  SPLIT ve BONUS_SHARE için kesirli hisse (fractional share) davranışı.
  Karar: quantity Decimal olarak tutuluyor (int değil) — hem TEFAS fon
  payları zaten kesirli olduğu için hem de BIST'te bazı kurumsal
  aksiyonlar (örn. oransal olmayan split sonucu) "kesirli pay hesabı"
  ile karşılanabildiği için. Kesirli payın nakde çevrilmesi (cash-in-lieu)
  mantığı bu modülün kapsamı DIŞINDA bırakıldı — gerçek BIST/TEFAS
  davranışı doğrulanmadan bu konuda bir kural koymak yanlış olur.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.domain.enums.transaction_type import TransactionType


@dataclass(frozen=True)
class Transaction:
    """
    Tek bir muhasebe olayı.

    Alan kullanımı transaction_type'a göre değişir:
      BUY / SELL      : quantity, price zorunlu
      BONUS_SHARE     : quantity zorunlu (bedelsiz adet), price yok sayılır
      SPLIT           : split_ratio zorunlu, quantity/price yok sayılır
      DIVIDEND        : gross_amount, withholding_tax, net_amount bilgi
                        amaçlı taşınır ama CostBasisCalculator bunları
                        okumaz (pozisyonu etkilemez, yalnızca CashLedger
                        ilgilenir — Faz B)
      TAX             : anlamı henüz netleşmedi (bkz. transaction_type.py
                        docstring), CostBasisCalculator no-op işler

    symbol normalizasyonu (örn. "THYAO" vs "THYAO.IS") bu katmanın
    sorumluluğu değil — MarketDataService/YFinanceAdapter sınırında
    zaten çözülüyor (bkz. yfinance_adapter.py _to_yfinance_symbol).
    Transaction her zaman "iç" (normalize edilmemiş, BIST-native) sembolü
    taşır.
    """

    symbol: str
    transaction_type: TransactionType
    timestamp: datetime
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    split_ratio: Decimal | None = None
    gross_amount: Decimal | None = None
    withholding_tax: Decimal | None = None
    net_amount: Decimal | None = None
    transaction_id: str | None = None
    # symbol_type: DDL'de NOT NULL ('BIST_STOCK', 'TEFAS_FUND', vb.) ama
    # burada opsiyonel — CostBasisCalculator bu alanı HİÇ KULLANMIYOR
    # (sembol bazlı çalışıyor, varlık sınıfından bağımsız). Yalnızca
    # persistence round-trip'ini kaybetmemek için taşınıyor. Repository
    # katmanı (transaction_repository.py) yazarken bu alanın None
    # olmamasını zorunlu kılar — domain seviyesinde zorunlu kılmak
    # YANLIŞ olur çünkü bir testte/hesaplamada Transaction'ı bu alan
    # olmadan constructe etmek tamamen geçerlidir.
    symbol_type: str | None = None
    # portfolio_id: DDL'de transactions.portfolio_id NOT NULL (FK) ama
    # burada opsiyonel — CostBasisCalculator ve position_quantity_timeseries
    # bu alanı KULLANMIYOR (sembol bazlı çalışıyorlar, portföyden
    # bağımsız). Bu turda eklendi: TransactionService.reverse_transaction()
    # reversal sırasında nakit ledger'ı güncelleyebilmek için (hangi
    # portföyün cash_ledger_entries'ine yazacağını bilmesi gerekiyor)
    # get_by_id() ile dönen Transaction'da bu bilgiye ihtiyaç duydu —
    # symbol_type ile AYNI gerekçe/desen (persistence round-trip'i
    # kaybetmemek, domain hesaplama mantığı bu alanı yok sayar).
    portfolio_id: str | None = None

    def __post_init__(self) -> None:
        # Fail-fast validasyon — CostBasisCalculator'a hatalı veri
        # ulaşmadan burada yakalanmalı (validation ADR-005 ile tutarlı:
        # "validasyon seviyeleri" — bu, en iç seviye: entity invariant'ı).
        if self.transaction_type in (TransactionType.BUY, TransactionType.SELL, TransactionType.RIGHTS_USED):
            # DÜZELTME (bu turda): RIGHTS_USED bu turda listeye eklendi
            # — BUY ile matematiksel olarak özdeş davranıyor (bkz.
            # cost_basis_calculator.py), bu yüzden AYNI quantity/price
            # validasyonunu paylaşmalı (önceden bu doğrulamadan
            # MUAFTI — tutarsızlık, bu turda düzeltildi).
            if self.quantity <= Decimal("0"):
                raise ValueError(
                    f"{self.transaction_type}: quantity pozitif olmalı, "
                    f"alınan: {self.quantity}"
                )
            if self.price < Decimal("0"):
                raise ValueError(
                    f"{self.transaction_type}: price negatif olamaz, "
                    f"alınan: {self.price}"
                )
        if self.transaction_type is TransactionType.BONUS_SHARE:
            if self.quantity <= Decimal("0"):
                raise ValueError(
                    f"BONUS_SHARE: quantity pozitif olmalı, "
                    f"alınan: {self.quantity}"
                )
        if self.transaction_type is TransactionType.SPLIT:
            if self.split_ratio is None or self.split_ratio <= Decimal("0"):
                raise ValueError(
                    "SPLIT: split_ratio pozitif bir Decimal olmalı "
                    f"(örn. 10:1 split için 10), alınan: {self.split_ratio}"
                )
        if self.transaction_type is TransactionType.REVERSE_SPLIT:
            # DÜZELTME (bu turda eklendi): UX kararı — kullanıcı
            # 'azaltma faktörünü' >1 bir sayı olarak girer (1:10 ters
            # bölünme için '10', SPLIT'in >0 kuralından FARKLI olarak
            # >1 zorunlu — 1:1 veya küçültme-olmayan bir "ters bölünme"
            # kavramsal olarak anlamsız). CostBasisCalculator bunu
            # quantity /= split_ratio olarak uygular (bkz. o dosyanın
            # REVERSE_SPLIT bloğu).
            if self.split_ratio is None or self.split_ratio <= Decimal("1"):
                raise ValueError(
                    "REVERSE_SPLIT: split_ratio 1'den BÜYÜK bir Decimal olmalı "
                    f"(örn. 1:10 ters bölünme için 10), alınan: {self.split_ratio}"
                )
        if self.transaction_type is TransactionType.DIVIDEND:
            if self.net_amount is None or self.net_amount < Decimal("0"):
                raise ValueError(
                    "DIVIDEND: net_amount negatif olmayan bir Decimal "
                    f"olmalı, alınan: {self.net_amount}"
                )
