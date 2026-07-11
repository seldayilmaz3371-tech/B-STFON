"""
İşlem tipi enum'u — CANONICAL DDL ile senkronize.

DÜZELTME KAYDI (önemli):
  İlk taslağımda yalnızca 6 değer vardı (BUY, SELL, DIVIDEND,
  BONUS_SHARE, SPLIT, TAX). BIST_TEFAS_Master_Design_Document.md'deki
  `transactions` tablosunun CHECK constraint'ini okuduğumda gerçek
  kümenin 13 değer olduğunu gördüm:

    BUY, SELL, DIVIDEND, BONUS_SHARE, RIGHTS_USED, RIGHTS_SOLD,
    SPLIT, REVERSE_SPLIT, MERGER, DEPOSIT, WITHDRAWAL, FEE, TAX

  Bu eksikliği fark etmeseydim, repository katmanı DB'den bu tipleri
  okuduğunda enum coercion hatası (ValueError) alacaktı — sessiz veri
  kaybı değil ama sessiz bir "neden çalışmıyor" sorunu olurdu. Şimdi
  düzeltiliyor.

KAPSAM KARARI — hangi tipler CostBasisCalculator'da işleniyor:
  Tam finansal mantığı bilinen ve golden dataset'le doğrulanmış (5):
    BUY, SELL, BONUS_SHARE, SPLIT  → pozisyonu değiştirir
    DIVIDEND                        → pozisyonu değiştirmez, total_dividends'e eklenir

  Cash-only, pozisyonu etkilemediği YAPISAL OLARAK KESİN olan (3):
    DEPOSIT, WITHDRAWAL, FEE, TAX  → no-op (symbol_type='CASH' ile
    kullanılırlar — DDL'deki symbol_type CHECK'i 'CASH' içeriyor,
    yani bunlar bir hisseye değil portföyün nakit hesabına bağlı)

  FİNANSAL MANTIĞI HENÜZ DOĞRULANMAMIŞ, BİLİNÇLİ OLARAK
  IMPLEMENTE EDİLMEYEN (4):
    RIGHTS_USED (rüçhan hakkı kullanımı), RIGHTS_SOLD (rüçhan hakkı
    satışı), REVERSE_SPLIT (ters bölünme), MERGER (birleşme)

    Bu 4 tip CostBasisCalculator'a ulaştığında BİLİNÇLİ OLARAK
    NotSupportedTransactionTypeError fırlatılır (sessiz yanlış
    hesaplama değil, açık ve net hata). Gerekçe: bu işlemlerin doğru
    muhasebeleştirilmesi (örn. rüçhan hakkı kullanımının maliyet
    bazına etkisi, birleşmede oran/nakit karışık senaryolar) için
    golden dataset yok ve tasarım belgesinde bu senaryolar detaylı
    işlenmemiş. Varsayımla bir formül yazmak, yanlış vergi/PnL
    raporlamasına yol açabilir — finansal doğruluk önceliği gereği
    bunu YAPMIYORUM. Gerçek senaryo/golden dataset netleştiğinde
    eklenecek (muhtemelen Faz D veya ayrı bir "corporate actions"
    fazı).
"""

from __future__ import annotations

from enum import Enum


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    BONUS_SHARE = "BONUS_SHARE"
    RIGHTS_USED = "RIGHTS_USED"
    RIGHTS_SOLD = "RIGHTS_SOLD"
    SPLIT = "SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    MERGER = "MERGER"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    FEE = "FEE"
    TAX = "TAX"

    def affects_cost_basis(self) -> bool:
        """Miktar/maliyet bazını değiştirir mi (WAVG/FIFO hesabında)."""
        return self in (
            TransactionType.BUY,
            TransactionType.SELL,
            TransactionType.BONUS_SHARE,
            TransactionType.SPLIT,
        )

    def is_cash_only(self) -> bool:
        """
        Pozisyonu etkilemez, yalnızca nakit ledger'ı etkiler.

        Yapısal olarak kesin (DDL symbol_type='CASH' ile kullanılırlar).
        """
        return self in (
            TransactionType.DIVIDEND,
            TransactionType.DEPOSIT,
            TransactionType.WITHDRAWAL,
            TransactionType.FEE,
            TransactionType.TAX,
        )

    def is_supported_by_calculator(self) -> bool:
        """
        CostBasisCalculator bu tipi işleyebiliyor mu?

        DÜZELTME (bu turda): RIGHTS_USED artık DESTEKLENİYOR — BUY ile
        matematiksel olarak özdeş (şirketin duyurduğu SABİT abonelik
        fiyatından hisse almak). RIGHTS_SOLD HÂLÂ bloke — satış fiyatı
        piyasa arz-talebine göre DEĞİŞKEN, evrensel bir formülle
        çözülemez (web araştırmasıyla doğrulandı).

        False dönenler (HÂLÂ): RIGHTS_SOLD, MERGER — bunlar şirkete/
        anlaşmaya/piyasaya özgü fiyatlama içeriyor, evrensel bir
        formülle çözülemez, gerçek golden dataset gerektirir.
        """
        return self not in (
            TransactionType.RIGHTS_SOLD,
            TransactionType.MERGER,
        )
