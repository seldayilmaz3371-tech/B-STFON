"""
Portfolio domain modeli.

Kaynak: BIST_TEFAS_Master_Design_Document.md Bölüm 2.2 — alan alan
birebir uygulandı.

TESPİT EDİLEN ŞEMA/İMPLEMENTASYON ÇELİŞKİSİ (açıkça işaretleniyor):
  DDL (orm_models.py::portfolios_table) `cost_method` için CHECK
  constraint'te WAVG/FIFO/LIFO'ya izin veriyor. Ama
  cost_basis_calculator.py'da YALNIZCA WAVGCostBasisCalculator ve
  FIFOCostBasisCalculator var — LIFO hiç implemente edilmedi (bilinçli
  karar, ADR-001 yalnızca WAVG/FIFO'yu netleştiriyordu).

  Eğer bu domain modeli DB CHECK constraint'ine güvenip LIFO'yu
  geçerli kabul etseydi, bir kullanıcı LIFO seçebilir ve PortfolioService
  bu portföy için hesaplama yapmaya çalıştığında (WAVGCostBasisCalculator
  veya FIFOCostBasisCalculator arasında seçim yapan bir mekanizma
  olmadığı için) YA yanlış sessizce WAVG'a düşerdi YA DA
  NotImplementedError ile üretim ortamında çökerdi. Her ikisi de kabul
  edilemez.

  KARAR: Bu modelin __post_init__'i LIFO'yu DB CHECK'in izin verdiği
  ama uygulamanın gerçekte DESTEKLEMEDİĞİ bir değer olarak reddeder.
  DB şeması ileride LIFO implementasyonu eklenene kadar "gelecekte
  kullanılacak ama şu an aktif olmayan" bir seçenek olarak kalıyor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import NamedTuple

from src.domain.enums.cost_method import CostMethod
from src.domain.enums.currency_code import CurrencyCode
from src.domain.exceptions.domain_exceptions import ValidationError


class BenchmarkInfo(NamedTuple):
    code: str
    name: str
    description: str


# DÜZELTME (bu turda): Önceden ÜÇ AYRI yerde (Portfolio.py'nin
# validasyonu, RiskService.get_available_benchmarks() — henüz hiç
# yazılmamıştı, ve UI'ın create_portfolio formundaki hardcoded
# selectbox listesi) BAĞIMSIZ, TUTARSIZ olma riski taşıyan listeler
# olacaktı — bu, açıkça "ileride bir tutarsızlık kaynağı" olarak
# işaretlenmişti. Artık TEK bir kanonik kaynak: KNOWN_BENCHMARKS.
# Portfolio'nun validasyonu, RiskService.get_available_benchmarks(),
# ve UI'ın seçenekleri HEPSİ buradan türetiliyor (PortfolioService
# üzerinden dolaylı — katman izolasyonu, bkz. get_available_benchmark_codes()).
#
# LİSTE GEREKÇESİ (gerçekten web'den doğrulandı, icat EDİLMEDİ):
#   Design doc'un kendi örnek listesi ("XU100, XU030, XUTUM, KATLM,
#   XBANK, XHOLD...") "..." ile AÇIK UÇLU bırakılmıştı — kapalı bir
#   liste değil, "ileride tamamlanacak" bir taslaktı. Ayrıca "KATLM"
#   GERÇEK bir BIST ticker'ı DEĞİL — gerçek katılım endeksi kodu
#   XK100'dür (bu turda web araması ile doğrulandı, design doc'un
#   kendi örneği hatalıydı, düzeltilmeden kopyalanmadı).
#
#   BIST'in GERÇEKTE 90'dan fazla endeksi var (çoğu dar sektör alt-
#   endeksi, "benchmark" olarak anlamlı değil) — TAMAMINI listelemek
#   kapsam dışı. Burada YALNIZCA en yaygın kullanılan, GERÇEKTEN
#   "portföy benchmark'ı" olarak anlamlı 7 endeks kürasyonla seçildi.
#   BELİRSİZLİK (açıkça işaretleniyor, TEST EDİLEMEDİ): ".IS" son eki
#   YFinanceAdapter boyunca TÜM BIST sembolleri için tutarlı şekilde
#   kullanılıyor (XU100.IS, XU030.IS için bu proje boyunca GERÇEKTEN
#   doğrulandı) — ama XBANK.IS/XHOLD.IS/XUTUM.IS/XK100.IS'in yfinance'ta
#   GERÇEKTEN çözümlenip çözümlenmediği bu sandbox'ta (network bloke)
#   TEST EDİLEMEDİ. Formatın TUTARLI olması yüksek olasılıkla doğru
#   çalışacağını düşündürüyor ama bu bir GARANTİ değil.
KNOWN_BENCHMARKS: tuple[BenchmarkInfo, ...] = (
    BenchmarkInfo("XU100.IS", "BIST 100", "En yaygın kullanılan geniş piyasa endeksi"),
    BenchmarkInfo("XU030.IS", "BIST 30", "En büyük 30 şirket (büyük sermaye)"),
    BenchmarkInfo("XU050.IS", "BIST 50", "En büyük 50 şirket"),
    BenchmarkInfo("XUTUM.IS", "BIST TÜM", "Borsa İstanbul'daki tüm hisseler"),
    BenchmarkInfo("XBANK.IS", "BIST Banka", "Bankacılık sektörü endeksi"),
    BenchmarkInfo("XHOLD.IS", "BIST Holding", "Holding şirketleri endeksi"),
    BenchmarkInfo("XK100.IS", "BIST Katılım 100", "Katılım (faizsiz) finans uyumlu 100 şirket"),
)
_KNOWN_BENCHMARK_CODES = frozenset(b.code for b in KNOWN_BENCHMARKS)

# LIFO, DB şemasında izinli ama hiçbir CostBasisCalculator'da
# implemente değil — bkz. modül docstring'i.
_SUPPORTED_COST_METHODS = frozenset({CostMethod.WAVG, CostMethod.FIFO})


@dataclass(frozen=True)
class Portfolio:
    id: str
    name: str
    currency: CurrencyCode
    cost_method: CostMethod
    inception_date: date
    description: str | None = None
    benchmark_code: str | None = None
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValidationError("Portföy adı boş olamaz.", field="name")

        if self.inception_date > date.today():
            raise ValidationError(
                f"inception_date gelecekte olamaz: {self.inception_date}",
                field="inception_date",
            )

        if self.cost_method not in _SUPPORTED_COST_METHODS:
            raise ValidationError(
                f"cost_method '{self.cost_method}' DB şemasında izinli olabilir "
                "ama CostBasisCalculator'da HENÜZ İMPLEMENTE EDİLMEDİ. "
                f"Desteklenen: {sorted(m.value for m in _SUPPORTED_COST_METHODS)}",
                field="cost_method",
            )

        if self.benchmark_code is not None and self.benchmark_code not in _KNOWN_BENCHMARK_CODES:
            raise ValidationError(
                f"benchmark_code '{self.benchmark_code}' bilinen BIST endeks "
                f"listesinde değil. Bilinen (TAM OLMAYAN, bkz. modül "
                f"docstring'i) liste: {sorted(_KNOWN_BENCHMARK_CODES)}",
                field="benchmark_code",
            )
