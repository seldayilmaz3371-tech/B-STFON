"""
Maliyet bazı hesaplayıcıları — WAVG ve FIFO.

Clean Architecture sınır kuralı (bilinçli tercih):
  Bu dosya domain katmanındadır ve HİÇBİR infrastructure bağımlılığı
  taşımaz — logging_config.get_logger dahi import EDİLMEDİ. Domain,
  infrastructure'ın var olup olmadığından haberdar olmamalı (Dependency
  Rule). Hata durumları context bilgisiyle fırlatılıyor; loglama
  servis katmanının sorumluluğu (bkz. portfolio_service.py).

API sözleşmesi — VARSAYIM DEĞİL, mevcut kodun ÇALIŞTIRILARAK
tersine mühendisliğiyle çıkarıldı:
  portfolio_service.py VE test_portfolio_service.py zaten şu
  sözleşmeyle yazılmış (ikisi de önceden teslim edilmişti):
    - WAVGCostBasisCalculator() → parametresiz constructor
    - .calculate(transactions) → CostBasisResult (SOMUT sınıf,
      Protocol değil — test dosyası CostBasisResult(...) şeklinde
      doğrudan instantiate ediyor)
    - CostBasisResult alanları: total_quantity, average_cost,
      total_cost_basis, realized_pnl, total_dividends
    - CostBasisResult.current_value(price), .unrealized_pnl(price)

  DÜZELTME GEÇMİŞİ: İlk taslağımda total_dividends alanı YOKTU ve
  DIVIDEND işlemini tamamen no-op varsaymıştım. test_portfolio_service.py
  dosyasını GERÇEKTEN ÇALIŞTIRDIĞIMDA bu yanlış çıktı — test,
  CostBasisResult'ı total_dividends=ZERO ile construct ediyor, yani bu
  alan zorunlu bir parça. Bunu görünce tasarımı düzelttim: DIVIDEND
  işlemi artık `total_dividends`'e net_amount'u ekliyor (pozisyonu
  hâlâ etkilemiyor — GD-004'ün "temettü maliyeti etkilemez" kuralı
  korunuyor, yalnızca ayrı bir toplam olarak taşınıyor).

Golden Dataset doğrulama kapsamı (test_cost_basis_calculator.py):
  GD-001, GD-002, GD-003, GD-004, GD-005 — tümü ±0.01 TL tolerans ile
  doğrulandı (bkz. test dosyası, gerçekten çalıştırıldı — bkz. sohbet
  geçmişindeki pytest/mypy/coverage çıktıları).
  FIFO + SPLIT/BONUS_SHARE senaryoları golden dataset'te TANIMLI DEĞİL
  — bu dosyadaki FIFO split/bonus davranışı mühendislik uzantısı
  (aşağıda gerekçeli), tasarım belgesinden doğrudan alınmadı.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import (
    BusinessRuleError,
    InsufficientQuantityError,
)
from src.domain.interfaces.cost_basis_strategy import CostBasisResult
from src.domain.models.transaction import Transaction

# Geriye dönük uyumluluk / kanonik import yolu:
# test_portfolio_service.py ve portfolio_service.py CostBasisResult'ı
# BU modülden import ediyor (interfaces'ten değil). Tanım yeri
# interfaces (DTO, Strategy sözleşmesinin bir parçası) ama burada
# re-export ediliyor ki mevcut kod tek satır değişmeden çalışsın.
__all__ = [
    "CostBasisResult",
    "TaxLot",
    "WAVGCostBasisCalculator",
    "FIFOCostBasisCalculator",
]

ZERO = Decimal("0")


def _sorted_by_time(transactions: "list[Transaction]") -> list[Transaction]:
    """
    Çağıranın sıralı liste verdiğine GÜVENİLMEZ — savunmacı sıralama.

    Aynı timestamp'e sahip işlemler için Python'ın stable sort'u orijinal
    liste sırasını korur (aynı gün çoklu işlemde "girilme sırası =
    gerçekleşme sırası" varsayımı — gerçek veri ile ayrıca doğrulanmalı).
    """
    return sorted(transactions, key=lambda t: t.timestamp)


# ── WAVG ──────────────────────────────────────────────────────────────────────

class WAVGCostBasisCalculator:
    """
    Ağırlıklı Ortalama Maliyet (Weighted Average Cost) hesaplayıcı.

    ADR-001: Varsayılan yöntem. O(1) state (running total), lot geçmişi
    tutulmaz.
    """

    def calculate(self, transactions: "list[Transaction]") -> CostBasisResult:
        if not transactions:
            raise BusinessRuleError("calculate() boş işlem listesiyle çağrıldı.")

        symbol = transactions[0].symbol
        quantity = ZERO
        total_cost = ZERO
        realized_pnl = ZERO
        total_dividends = ZERO

        for tx in _sorted_by_time(transactions):
            if tx.transaction_type in (TransactionType.BUY, TransactionType.RIGHTS_USED):
                # DÜZELTME (bu turda eklendi): RIGHTS_USED (rüçhan hakkı
                # kullanımı), BUY ile MATEMATİKSEL OLARAK ÖZDEŞ — şirketin
                # DUYURDUĞU (bilinen, sabit) bir abonelik fiyatından yeni
                # hisse almak. Bu, RIGHTS_SOLD'un (piyasa arz-talebine göre
                # DEĞİŞKEN fiyatlı, hâlâ bloke) AKSİNE güvenle uygulanabilir
                # — web araştırmasıyla doğrulandı (bkz. bu turun gerekçe
                # notu: rüçhan kullanım fiyatı genelde 1 TL, şirketçe
                # ÖNCEDEN duyurulur, piyasa dalgalanmasına tabi DEĞİLDİR).
                quantity += tx.quantity
                total_cost += tx.quantity * tx.price

            elif tx.transaction_type is TransactionType.SELL:
                if tx.quantity > quantity:
                    raise InsufficientQuantityError(
                        symbol=symbol, requested=tx.quantity, available=quantity,
                    )
                avg_cost = total_cost / quantity if quantity > ZERO else ZERO
                realized_pnl += tx.quantity * (tx.price - avg_cost)
                total_cost -= tx.quantity * avg_cost
                quantity -= tx.quantity
                if quantity == ZERO:
                    # Kapanan pozisyonda Decimal yuvarlama artığını sıfırla
                    # — aksi halde sonraki BUY "hayalet maliyet" üzerine
                    # inşa eder.
                    total_cost = ZERO

            elif tx.transaction_type is TransactionType.BONUS_SHARE:
                if quantity <= ZERO:
                    raise BusinessRuleError(
                        f"{symbol}: Pozisyon yokken BONUS_SHARE alındı "
                        "(veri bütünlüğü hatası).",
                        symbol=symbol,
                    )
                quantity += tx.quantity  # total_cost DEĞİŞMEZ (GD-002)

            elif tx.transaction_type is TransactionType.SPLIT:
                if quantity <= ZERO:
                    raise BusinessRuleError(
                        f"{symbol}: Pozisyon yokken SPLIT alındı "
                        "(veri bütünlüğü hatası).",
                        symbol=symbol,
                    )
                assert tx.split_ratio is not None  # __post_init__ garanti eder
                quantity *= tx.split_ratio  # total_cost DEĞİŞMEZ (GD-005)

            elif tx.transaction_type is TransactionType.REVERSE_SPLIT:
                # DÜZELTME (bu turda eklendi): SPLIT'in TAM TERSİ —
                # quantity AZALIR, total_cost DEĞİŞMEZ (aynı
                # "toplam değer korunumu" ilkesi, yön ters). Golden
                # dataset'te TANIMLI DEĞİL (SPLIT/BONUS_SHARE gibi) ama
                # matematiği evrensel ve tartışmasız — mühendislik
                # uzantısı olarak GD-005'in simetriği.
                if quantity <= ZERO:
                    raise BusinessRuleError(
                        f"{symbol}: Pozisyon yokken REVERSE_SPLIT alındı "
                        "(veri bütünlüğü hatası).",
                        symbol=symbol,
                    )
                assert tx.split_ratio is not None  # __post_init__ garanti eder (>1)
                quantity /= tx.split_ratio  # total_cost DEĞİŞMEZ

            elif tx.transaction_type is TransactionType.DIVIDEND:
                # Pozisyonu ETKİLEMEZ (GD-004), yalnızca bilgi amaçlı
                # net temettü toplamına eklenir.
                assert tx.net_amount is not None  # __post_init__ garanti eder
                total_dividends += tx.net_amount

            elif tx.transaction_type in (
                TransactionType.TAX,
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL,
                TransactionType.FEE,
            ):
                # Cash-only — DDL'de symbol_type='CASH' ile kullanılırlar,
                # hisse pozisyonunu yapısal olarak etkilemezler.
                continue

            elif not tx.transaction_type.is_supported_by_calculator():
                # RIGHTS_USED, RIGHTS_SOLD, REVERSE_SPLIT, MERGER —
                # bilinçli olarak implemente edilmedi (bkz.
                # transaction_type.py modül docstring'i). Sessiz yanlış
                # hesaplama yerine açık hata.
                raise BusinessRuleError(
                    f"{symbol}: {tx.transaction_type} işlem tipi "
                    "CostBasisCalculator'da henüz desteklenmiyor "
                    "(golden dataset / finansal mantık doğrulanmadı).",
                    symbol=symbol,
                    transaction_type=str(tx.transaction_type),
                )

            else:  # pragma: no cover — enum kapalı küme, tüm dallar ele alındı
                raise BusinessRuleError(
                    f"Tanınmayan transaction_type: {tx.transaction_type}",
                )

        average_cost = total_cost / quantity if quantity > ZERO else ZERO

        return CostBasisResult(
            total_quantity=quantity,
            average_cost=average_cost,
            total_cost_basis=total_cost,
            realized_pnl=realized_pnl,
            total_dividends=total_dividends,
        )


# ── FIFO ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaxLot:
    """Tek bir alım lotu — FIFO kuyruğunun elemanı. Immutable."""

    quantity: Decimal
    unit_cost: Decimal

    @property
    def cost(self) -> Decimal:
        return self.quantity * self.unit_cost


class FIFOCostBasisCalculator:
    """
    İlk Giren İlk Çıkar (First In, First Out) hesaplayıcı.

    ADR-001'de config'den açılabilir opsiyon. O(n) lot kuyruğu tutar.

    FIFO + BONUS_SHARE/SPLIT davranışı — GOLDEN DATASET'TE TANIMLI
    DEĞİL, mühendislik uzantısı (gerekçe):
      - BONUS_SHARE: Sıfır maliyetli AYRI bir lot olarak kuyruğa eklenir
        (FIFO'nun doğal semantiğiyle WAVG'dan daha temiz örtüşüyor).
      - SPLIT: Kuyruktaki HER lot için quantity × ratio, unit_cost ÷ ratio
        — her lotun kendi toplam maliyetini koruması, GD-005'in
        lot-bazlı doğal genellemesi.
      Golden dataset ile doğrulanmadı; testte "iç tutarlılık" testi
      olarak ayrıca işaretlendi.
    """

    def calculate(self, transactions: "list[Transaction]") -> CostBasisResult:
        if not transactions:
            raise BusinessRuleError("calculate() boş işlem listesiyle çağrıldı.")

        symbol = transactions[0].symbol
        lots: deque[TaxLot] = deque()
        realized_pnl = ZERO
        total_dividends = ZERO

        for tx in _sorted_by_time(transactions):
            if tx.transaction_type in (TransactionType.BUY, TransactionType.RIGHTS_USED):
                # DÜZELTME (bu turda eklendi): bkz. WAVG bloğundaki
                # gerekçe — RIGHTS_USED, BUY ile matematiksel olarak özdeş.
                lots.append(TaxLot(quantity=tx.quantity, unit_cost=tx.price))

            elif tx.transaction_type is TransactionType.SELL:
                available = sum((lot.quantity for lot in lots), ZERO)
                if tx.quantity > available:
                    raise InsufficientQuantityError(
                        symbol=symbol, requested=tx.quantity, available=available,
                    )
                remaining = tx.quantity
                while remaining > ZERO:
                    oldest = lots[0]
                    if oldest.quantity <= remaining:
                        realized_pnl += oldest.quantity * (tx.price - oldest.unit_cost)
                        remaining -= oldest.quantity
                        lots.popleft()
                    else:
                        realized_pnl += remaining * (tx.price - oldest.unit_cost)
                        lots[0] = TaxLot(
                            quantity=oldest.quantity - remaining,
                            unit_cost=oldest.unit_cost,
                        )
                        remaining = ZERO

            elif tx.transaction_type is TransactionType.BONUS_SHARE:
                if not lots:
                    raise BusinessRuleError(
                        f"{symbol}: Pozisyon yokken BONUS_SHARE alındı "
                        "(veri bütünlüğü hatası).",
                        symbol=symbol,
                    )
                lots.append(TaxLot(quantity=tx.quantity, unit_cost=ZERO))

            elif tx.transaction_type is TransactionType.SPLIT:
                if not lots:
                    raise BusinessRuleError(
                        f"{symbol}: Pozisyon yokken SPLIT alındı "
                        "(veri bütünlüğü hatası).",
                        symbol=symbol,
                    )
                assert tx.split_ratio is not None
                ratio = tx.split_ratio
                lots = deque(
                    TaxLot(quantity=lot.quantity * ratio, unit_cost=lot.unit_cost / ratio)
                    for lot in lots
                )

            elif tx.transaction_type is TransactionType.REVERSE_SPLIT:
                # DÜZELTME (bu turda eklendi): SPLIT'in FIFO bloğunun
                # TAM TERSİ — her lot için quantity ÷ factor,
                # unit_cost × factor (lot'un TOPLAM maliyeti KORUNUR,
                # aynı GD-005 simetri ilkesi).
                if not lots:
                    raise BusinessRuleError(
                        f"{symbol}: Pozisyon yokken REVERSE_SPLIT alındı "
                        "(veri bütünlüğü hatası).",
                        symbol=symbol,
                    )
                assert tx.split_ratio is not None  # __post_init__ garanti eder (>1)
                factor = tx.split_ratio
                lots = deque(
                    TaxLot(quantity=lot.quantity / factor, unit_cost=lot.unit_cost * factor)
                    for lot in lots
                )

            elif tx.transaction_type is TransactionType.DIVIDEND:
                assert tx.net_amount is not None
                total_dividends += tx.net_amount

            elif tx.transaction_type in (
                TransactionType.TAX,
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL,
                TransactionType.FEE,
            ):
                continue

            elif not tx.transaction_type.is_supported_by_calculator():
                raise BusinessRuleError(
                    f"{symbol}: {tx.transaction_type} işlem tipi "
                    "CostBasisCalculator'da henüz desteklenmiyor "
                    "(golden dataset / finansal mantık doğrulanmadı).",
                    symbol=symbol,
                    transaction_type=str(tx.transaction_type),
                )

            else:  # pragma: no cover
                raise BusinessRuleError(
                    f"Tanınmayan transaction_type: {tx.transaction_type}",
                )

        total_quantity = sum((lot.quantity for lot in lots), ZERO)
        total_cost_basis = sum((lot.cost for lot in lots), ZERO)
        average_cost = total_cost_basis / total_quantity if total_quantity > ZERO else ZERO

        return CostBasisResult(
            total_quantity=total_quantity,
            average_cost=average_cost,
            total_cost_basis=total_cost_basis,
            realized_pnl=realized_pnl,
            total_dividends=total_dividends,
            lots=tuple(lots),
        )
