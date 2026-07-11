"""
Pozisyon miktarı zaman serisi hesaplayıcı — RiskService'in temel yapı taşı.

SORUMLULUK SINIRI (CostBasisCalculator'dan FARKI, bilinçli):
  CostBasisCalculator "bugün elimde ne var, maliyeti ne, gerçekleşen K/Z
  ne kadar" sorusuna cevap verir — TEK bir nihai anlık görüntü üretir.
  Bu modül FARKLI bir soruya cevap verir: "geçmişteki HERHANGİ bir T
  gününde elimde kaç adet vardı" — bir ZAMAN SERİSİ üretir.

  Bu, CostBasisCalculator'ın maliyet/PnL mantığının bir TEKRARI DEĞİL —
  o mantığı hiç içermiyor (avg_cost, realized_pnl burada YOK). Yalnızca
  "hangi transaction_type miktarı nasıl değiştirir" sınıflandırması
  ortak — ve bu ortak kısım zaten TransactionType enum'unun kendi
  metodlarında (affects_cost_basis, is_supported_by_calculator)
  kapsüllenmiş, burada TEKRAR YAZILMIYOR, yeniden KULLANILIYOR.

  Küçük bir miktar hesaplama mantığı (BUY→+, SELL→-, SPLIT→×ratio)
  kaçınılmaz olarak iki yerde de var — ama bu "aynı algoritmanın
  kopyası" değil, iki farklı sorunun (maliyet vs. zaman serisi) her
  birinin kendi minimal gereksinimidir. Tam birleştirme (tek bir
  "TransactionApplier" soyutlaması) değerlendirildi ama WAVG/FIFO'nun
  maliyet mutasyonuyla miktar mutasyonunun sıkı bağlı olması (SELL'de
  miktar VE maliyet aynı anda, avg_cost'a bağımlı şekilde değişiyor)
  nedeniyle temiz bir ortak arayüz çıkarmak, mevcut test edilmiş ve
  doğrulanmış cost_basis_calculator.py'ı riske atmadan mümkün olmadı.
  Bu bilinçli bir kapsam sınırı, gözden kaçırılmış bir DRY ihlali değil.

Kullanım: RiskService, bu zaman serisini symbol başına oluşturup
fiyat zaman serisiyle çarparak portföyün geçmiş toplam değerini
(equity value) hesaplıyor.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from typing import cast

import pandas as pd

from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import BusinessRuleError
from src.domain.models.transaction import Transaction


def compute_quantity_timeseries(transactions: list[Transaction]) -> pd.Series:
    """
    Bir sembol için, her işlemden HEMEN SONRAKİ miktarı gösteren bir
    step-function zaman serisi üretir.

    Returns:
        pd.Series — DatetimeIndex (işlem tarihleri, artan sıralı),
        değerler Decimal (float DEĞİL — miktar hassasiyeti korunuyor,
        RiskService'te fiyatla çarpılırken float'a dönüşüm sınırda,
        açıkça yapılacak).

        Boş liste verilirse boş Series döner (hata fırlatmaz — bir
        sembolün henüz hiç işlemi olmaması geçerli bir durum, RiskService
        bunu "bu sembol bu tarih aralığında portföyde yoktu" olarak
        yorumlayacak).

    Raises:
        BusinessRuleError: Pozisyon yokken BONUS_SHARE/SPLIT gelirse,
            veya desteklenmeyen bir transaction_type (RIGHTS_USED vb.)
            varsa — cost_basis_calculator.py ile TUTARLI davranış
            (aynı veri bütünlüğü kuralları, iki modülde de aynı
            sıkılıkta uygulanıyor).
    """
    if not transactions:
        return pd.Series(dtype=object)

    sorted_tx = sorted(transactions, key=lambda t: t.timestamp)
    symbol = sorted_tx[0].symbol
    quantity = Decimal("0")
    index: list[datetime] = []
    values: list[Decimal] = []

    for tx in sorted_tx:
        if tx.transaction_type in (TransactionType.BUY, TransactionType.RIGHTS_USED):
            # DÜZELTME (bu turda eklendi): RIGHTS_USED, BUY ile
            # matematiksel olarak özdeş — bkz. cost_basis_calculator.py'daki
            # tam gerekçe.
            quantity += tx.quantity
        elif tx.transaction_type is TransactionType.SELL:
            quantity -= tx.quantity
        elif tx.transaction_type is TransactionType.BONUS_SHARE:
            if quantity <= Decimal("0"):
                raise BusinessRuleError(
                    f"{symbol}: Pozisyon yokken BONUS_SHARE alındı.",
                    symbol=symbol,
                )
            quantity += tx.quantity
        elif tx.transaction_type is TransactionType.SPLIT:
            if quantity <= Decimal("0"):
                raise BusinessRuleError(
                    f"{symbol}: Pozisyon yokken SPLIT alındı.", symbol=symbol,
                )
            assert tx.split_ratio is not None
            quantity *= tx.split_ratio
        elif tx.transaction_type is TransactionType.REVERSE_SPLIT:
            # DÜZELTME (bu turda eklendi): cost_basis_calculator.py'daki
            # REVERSE_SPLIT desteğiyle SENKRON — bu dosya AYRI bir
            # miktar takip mantığı taşıdığı için (bkz. modül docstring'i
            # "iki farklı sorunun kendi minimal gereksinimi") burada da
            # AYRI ama SİMETRİK olarak eklenmesi gerekti.
            if quantity <= Decimal("0"):
                raise BusinessRuleError(
                    f"{symbol}: Pozisyon yokken REVERSE_SPLIT alındı.", symbol=symbol,
                )
            assert tx.split_ratio is not None
            quantity /= tx.split_ratio
        elif tx.transaction_type.is_cash_only():
            continue  # nakit-only işlemler miktarı etkilemez, kayda değmez
        elif not tx.transaction_type.is_supported_by_calculator():
            raise BusinessRuleError(
                f"{symbol}: {tx.transaction_type} işlem tipi pozisyon "
                "zaman serisinde henüz desteklenmiyor (aynı sınırlama "
                "cost_basis_calculator.py'da da geçerli).",
                symbol=symbol,
                transaction_type=str(tx.transaction_type),
            )
        else:  # pragma: no cover
            raise BusinessRuleError(f"Tanınmayan transaction_type: {tx.transaction_type}")

        index.append(tx.timestamp)
        values.append(quantity)

    return pd.Series(values, index=pd.DatetimeIndex(index), name=symbol)


def quantity_on_date(quantity_series: pd.Series, as_of: pd.Timestamp) -> Decimal:
    """
    Step-function serisinden belirli bir tarihteki miktarı okur
    (forward-fill mantığı: son işlemden bu yana miktar sabit kalır).

    `as_of`, serideki İLK işlem tarihinden ÖNCE ise Decimal('0') döner
    (henüz pozisyon açılmamıştı — bu bir hata değil, geçerli bir durum).
    """
    if quantity_series.empty or as_of < quantity_series.index[0]:
        return Decimal("0")
    # asof(): as_of'tan küçük/eşit en son index değerini bulur —
    # pandas'ın kendi step-function/forward-fill lookup'ı, elle
    # binary search yazmaktan daha az hataya açık.
    result = quantity_series.asof(as_of)
    if pd.isna(result):
        return Decimal("0")
    # DÜZELTME (bu turda, GERÇEK pandas-stubs kurulumu sonrası bulundu):
    # pandas'ın .asof() dönüş tipi, mypy için GENİŞ bir union (Series'in
    # GERÇEK içeriğinin Decimal olduğunu STATİK OLARAK bilemiyor —
    # dtype='object' bir Series'in içeriği çalışma zamanına kadar
    # belirsiz). Runtime'da result HER ZAMAN Decimal (values.append(quantity)
    # ile YALNIZCA Decimal ekleniyor, bkz. bu dosyanın üst kısmı) — bu
    # yüzden açık bir cast GÜVENLİ ve DOĞRU, tip hatası GİZLEMİYOR.
    return cast(Decimal, result)
