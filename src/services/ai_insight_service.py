"""
AIInsightService — AI destekli doğal dil portföy özeti VE soru-cevap.

MİMARİ PRENSİP (kullanıcı onayıyla, bkz. AI_Entegrasyonu_Analiz.md):
  Bu servis YENİ bir hesaplama YAPMAZ. YALNIZCA PortfolioService ve
  RiskService'in ZATEN hesapladığı, ZATEN test edilmiş sayıları
  (toplam getiri, Sharpe oranı, pozisyon dağılımı vb.) alıp DOĞAL
  DİLE çevirir. Bu, "AI yanlış hesapladı" riskini YAPISAL olarak
  ortadan kaldırıyor — AI yalnızca YORUMLUYOR, HESAPLAMIYOR.

  DÜZELTME (bu turda eklendi — soru-cevap özelliği): Kullanıcının
  serbest metinli soruları da AYNI prensiple ele alınıyor — AI'a
  YALNIZCA zaten hesaplanmış veri + kullanıcının sorusu veriliyor,
  AI'ın kendi başına YENİ bir hesaplama yapması AÇIKÇA yasaklanıyor
  (prompt'ta talimat olarak). Soru, verilen verinin KAPSAMI DIŞINDAYSA
  (örn. "gelecek ay ne olur" gibi tahmin gerektiren bir soru), AI'ın
  bunu AÇIKÇA belirtmesi isteniyor — UYDURMA CEVAP vermemesi için.

API ANAHTARI (kullanıcı kararı): Kullanıcının KENDİ .env dosyasına
girdiği anahtar kullanılıyor (PORTFOLIO_OS__AI__API_KEY) — sistem
varsayılan bir anahtar İÇERMİYOR, maliyeti kullanıcı üstleniyor.

TEST EDİLEBİLİRLİK SINIRI (açıkça işaretleniyor): Bu sandbox'ta
GERÇEK bir Anthropic API anahtarı YOK — gerçek bir API çağrısı bu
oturumda TEST EDİLEMEDİ. Testler, Anthropic client'ı SAHTE (mock)
bir nesneyle değiştirerek: (a) prompt'un doğru inşa edildiğini,
(b) API anahtarı yokken zarif bir hata verdiğini, (c) API hatası
durumunda çökmediğini doğruluyor — GERÇEK bir API yanıtının
KALİTESİ (örn. "AI'ın ürettiği metin gerçekten anlamlı mı") bu
oturumda doğrulanamadı, kullanıcının KENDİ anahtarıyla test etmesi
gerekiyor.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any


class AIInsightService:
    def __init__(
        self,
        portfolio_service: Any,
        risk_service: Any,
        api_key: str | None,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
    ) -> None:
        self._portfolio_service = portfolio_service
        self._risk_service = risk_service
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens

    @property
    def is_configured(self) -> bool:
        """UI'ın 'API anahtarı ayarlanmamış' durumunu ÖNCEDEN kontrol edebilmesi için."""
        return bool(self._api_key)

    def generate_portfolio_summary(self, portfolio_id: str) -> str:
        """
        Raises:
            BusinessRuleError: API anahtarı ayarlanmamışsa.
            RuntimeError: Anthropic API çağrısı başarısız olursa.
        """
        context = self._gather_context(portfolio_id)
        instruction = (
            "\n\nYukarıdaki, ZATEN HESAPLANMIŞ portföy verilerini Türkçe, "
            "anlaşılır bir dille ÖZETLE. YENİ bir sayı HESAPLAMA veya TAHMİN "
            "ÜRETME — yalnızca verilen sayıları YORUMLA. Belirli bir hisse "
            "AL/SAT tavsiyesi VERME. Bu bir yatırım tavsiyesi değildir, "
            "genel bir performans özetidir. En fazla 3 kısa paragraf yaz."
        )
        return self._call_api(context + instruction)

    def answer_portfolio_question(self, portfolio_id: str, question: str) -> str:
        """
        DÜZELTME (bu turda eklendi): Serbest metinli soru-cevap.

        Raises:
            BusinessRuleError: API anahtarı ayarlanmamışsa VEYA soru boşsa.
            RuntimeError: Anthropic API çağrısı başarısız olursa.
        """
        from src.domain.exceptions.domain_exceptions import BusinessRuleError

        if not question or not question.strip():
            raise BusinessRuleError("Soru boş olamaz.")

        context = self._gather_context(portfolio_id)
        instruction = (
            f"\n\nKullanıcının sorusu: \"{question.strip()}\"\n\n"
            "Bu soruyu YALNIZCA yukarıda verilen, ZATEN HESAPLANMIŞ verilere "
            "dayanarak cevapla. YENİ bir sayı HESAPLAMA veya TAHMİN ÜRETME. "
            "Eğer soru verilen verinin KAPSAMI DIŞINDaysa (örn. gelecek "
            "tahmini, verilmemiş bir bilgi gerektiriyorsa), bunu AÇIKÇA "
            "belirt — UYDURMA CEVAP VERME. Belirli bir hisse AL/SAT "
            "tavsiyesi VERME. Bu bir yatırım tavsiyesi değildir. Kısa ve "
            "net cevap ver."
        )
        return self._call_api(context + instruction)

    def detect_anomalies(self, portfolio_id: str, lookback_days: int = 100) -> str:
        """
        DÜZELTME (bu turda eklendi — kullanıcı onayıyla, DÜŞÜK RİSK
        kategorisi): RiskService.get_rolling_volatility()/get_drawdown_series()
        (ZATEN hesaplanmış zaman serileri) AI'a verilip 'bu seride
        DİKKAT ÇEKİCİ bir nokta var mı' diye SORULUYOR. AI YENİ bir
        istatistik HESAPLAMIYOR — yalnızca ZATEN hesaplanmış serideki
        GÖRECELİ uç noktaları YORUMLUYOR (örn. 'son 5 günde oynaklık
        serinin geri kalanına göre belirgin şekilde yüksek').

        Raises:
            BusinessRuleError: API anahtarı ayarlanmamışsa.
            RuntimeError: Anthropic API çağrısı başarısız olursa.
        """
        lines = [f"Portföy: {portfolio_id}"]

        try:
            rolling_vol = self._risk_service.get_rolling_volatility(portfolio_id, lookback_days=lookback_days)
            recent_vol = rolling_vol.tail(10).tolist()
            lines.append(f"\nSon 10 günün yıllıklandırılmış kayan oynaklık değerleri: {[f'{v:.2%}' for v in recent_vol]}")
            lines.append(f"Serinin tamamının ortalaması: {rolling_vol.mean():.2%}, standart sapması: {rolling_vol.std():.2%}")
        except Exception:
            lines.append("\nOynaklık serisi: henüz yeterli veri yok.")

        try:
            drawdown = self._risk_service.get_drawdown_series(portfolio_id, lookback_days=lookback_days)
            recent_dd = drawdown.tail(10).tolist()
            lines.append(f"\nSon 10 günün drawdown değerleri: {[f'{v:.2%}' for v in recent_dd]}")
            lines.append(f"Serinin en derin noktası: {drawdown.min():.2%}")
        except Exception:
            lines.append("\nDrawdown serisi: henüz yeterli veri yok.")

        instruction = (
            "\n\nYukarıdaki, ZATEN HESAPLANMIŞ zaman serisi verilerine bakarak, "
            "SON GÜNLERDEKİ değerlerin serinin GERİ KALANINA göre DİKKAT "
            "ÇEKİCİ (istatistiksel olarak sıra dışı) olup olmadığını YORUMLA. "
            "YENİ bir istatistik HESAPLAMA — yalnızca verilen sayıları "
            "KARŞILAŞTIR. Eğer hiçbir şey dikkat çekici değilse, bunu AÇIKÇA "
            "söyle ('normal aralıkta görünüyor' gibi) — YAPAY BİR ENDİŞE "
            "UYDURMA. Bu bir yatırım tavsiyesi değildir, yalnızca istatistiksel "
            "bir gözlemdir. En fazla 2 kısa paragraf yaz."
        )
        return self._call_api("\n".join(lines) + instruction)

    def interpret_backtest_result(self, comparison: Any) -> str:
        """
        DÜZELTME (bu turda eklendi — kullanıcı onayıyla, KISITLI kapsam):
        AI, YALNIZCA verilen BacktestComparison sonucunu (strateji vs.
        Buy & Hold — ZATEN hesaplanmış) YORUMLAR. AI'ın YENİ bir strateji
        ÖNERMESİ, parametre DEĞİŞİKLİĞİ ÖNERMESİ AÇIKÇA YASAKLANIYOR —
        kullanıcının açık kararı buydu ("yalnızca yorumlasın, öneri
        üretmesin"). Bu, projenin "yatırım tavsiyesi değildir" ilkesini
        korumak için özellikle KATI bir sınır.

        Args:
            comparison: BacktestComparison (strategy_result + benchmark_result).

        Raises:
            BusinessRuleError: API anahtarı ayarlanmamışsa.
            RuntimeError: Anthropic API çağrısı başarısız olursa.
        """
        result = comparison.strategy_result
        benchmark = comparison.benchmark_result

        max_dd_str = f"{result.max_drawdown:.2%}" if result.max_drawdown is not None else "N/A"
        sharpe_str = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "N/A"
        lines = [
            f"Strateji sonucu — Toplam getiri: {float(result.total_return):.2%}, "
            f"Sharpe: {sharpe_str}, Maks. düşüş: {max_dd_str}, "
            f"Toplam işlem: {result.total_trades}",
            f"Buy & Hold sonucu — Toplam getiri: {float(benchmark.total_return):.2%}",
        ]

        instruction = (
            "\n\nYukarıdaki, ZATEN HESAPLANMIŞ backtest sonuçlarını (strateji "
            "vs. Buy & Hold karşılaştırması) Türkçe, anlaşılır bir dille "
            "YORUMLA. Stratejinin Buy & Hold'u yenip yenmediğini, ne kadar "
            "fark ettiğini AÇIKLA. KESİNLİKLE YENİ bir strateji ÖNERME, "
            "parametre DEĞİŞİKLİĞİ ÖNERME, 'şunu dene' gibi bir tavsiye "
            "VERME — YALNIZCA verilen sonucu YORUMLA. Bu bir yatırım "
            "tavsiyesi değildir, yalnızca geçmiş bir backtest'in açıklamasıdır. "
            "En fazla 2 kısa paragraf yaz."
        )
        return self._call_api("\n".join(lines) + instruction)

    def generate_portfolio_report(self, portfolio_id: str) -> str:
        """
        DÜZELTME (bu turda eklendi): generate_portfolio_summary()'nin
        GENİŞLETİLMİŞ hali — daha UZUN, BAŞLIKLI bir rapor metni üretir
        (docx/pdf'e dönüştürülmek üzere, bkz. app.py'daki kullanım).
        AYNI mimari prensip (yorumla, hesaplama yapma) — yalnızca ÇIKTI
        FORMATI farklı (rapor bölümleri, .docx için).

        Raises:
            BusinessRuleError: API anahtarı ayarlanmamışsa.
            RuntimeError: Anthropic API çağrısı başarısız olursa.
        """
        context = self._gather_context(portfolio_id)
        instruction = (
            "\n\nYukarıdaki, ZATEN HESAPLANMIŞ portföy verilerini kullanarak "
            "YAPILANDIRILMIŞ bir performans raporu yaz. Şu başlıkları kullan: "
            "'## Genel Bakış', '## Pozisyon Analizi', '## Risk Değerlendirmesi', "
            "'## Sonuç'. YENİ bir sayı HESAPLAMA veya TAHMİN ÜRETME — yalnızca "
            "verilen sayıları YORUMLA VE YAPILANDIR. Belirli bir hisse AL/SAT "
            "tavsiyesi VERME. Raporun başına 'Bu bir yatırım tavsiyesi değildir' "
            "notunu EKLE. Markdown başlıkları kullan (##)."
        )
        return self._call_api(context + instruction)

    def _call_api(self, prompt: str) -> str:
        from src.domain.exceptions.domain_exceptions import BusinessRuleError

        if not self.is_configured:
            raise BusinessRuleError(
                "AI özelliği için API anahtarı ayarlanmamış. "
                ".env dosyanıza PORTFOLIO_OS__AI__API_KEY=<kendi-anahtarınız> ekleyin."
            )

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)
            message = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # DÜZELTME (bu turda, GERÇEK mypy --strict taramasıyla
            # bulundu): message.content[0] HER ZAMAN metin BLOĞU
            # OLMAYABİLİR — Sonnet 5'in adaptive thinking özelliği
            # etkinse, YANIT ThinkingBlock ile BAŞLAYABİLİR (TextBlock
            # DEĞİL). content[0].text varsayımı bu durumda
            # AttributeError ile ÇÖKERDİ. Artık TÜM bloklar taranıp
            # yalnızca GERÇEK metin bloklarının içeriği birleştiriliyor.
            text_parts = [
                block.text for block in message.content
                if hasattr(block, "text")
            ]
            if not text_parts:
                raise RuntimeError("API yanıtında metin bloğu bulunamadı.")
            return "\n".join(text_parts)
        except Exception as exc:
            raise RuntimeError(f"AI isteği başarısız: {exc}") from exc

    def _gather_context(self, portfolio_id: str) -> str:
        """
        YALNIZCA ZATEN HESAPLANMIŞ verileri toplar — burada HİÇBİR
        yeni finansal hesaplama YAPILMIYOR (bkz. modül docstring'i).

        DÜZELTME (bu turda genişletildi): Artık yalnızca risk
        metrikleri değil, PER-POZİSYON dökümü de (PortfolioService.
        get_portfolio_status) dahil ediliyor — soru-cevap özelliğinin
        "hangi pozisyonum en çok zarar ediyor" gibi soruları
        cevaplayabilmesi için GEREKLİ (yalnızca AGREGE risk verisiyle
        bu tür sorular cevaplanamazdı).
        """
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)
        portfolio_name = portfolio.name if portfolio is not None else portfolio_id

        lines = [f"Portföy adı: {portfolio_name}"]

        try:
            status = self._portfolio_service.get_portfolio_status(portfolio_id)
            lines.append(f"\nToplam maliyet: {status.total_cost_basis:.2f} TL")
            lines.append(f"Toplam güncel değer: {status.total_current_value:.2f} TL")
            lines.append(f"Toplam gerçekleşmemiş K/Z: {status.total_unrealized_pnl:.2f} TL")
            lines.append(f"Toplam gerçekleşmiş K/Z: {status.total_realized_pnl:.2f} TL")
            if status.positions:
                lines.append("\nPozisyonlar:")
                for pos in status.positions:
                    pnl_str = f"{pos.unrealized_pnl:.2f} TL ({pos.pnl_percentage:.2%})" if pos.unrealized_pnl is not None else "fiyat verisi yok"
                    lines.append(
                        f"  - {pos.symbol}: {pos.total_quantity} adet, "
                        f"ortalama maliyet {pos.average_cost:.2f} TL, K/Z: {pnl_str}"
                    )
            if status.stale_symbols:
                lines.append(f"\nGüncel fiyatı alınamayan semboller: {', '.join(status.stale_symbols)}")
        except Exception:
            lines.append("\nPozisyon detayı: henüz yeterli veri yok (yeni portföy veya boş portföy).")

        try:
            risk_profile = self._risk_service.compute_risk_profile(portfolio_id)
            lines.extend([
                f"\nYıllıklandırılmış oynaklık: {risk_profile.annualized_volatility:.2%}",
                f"Sharpe oranı: {risk_profile.sharpe_ratio:.2f}",
                f"Sortino oranı: {risk_profile.sortino_ratio:.2f}",
                f"Maksimum düşüş: {risk_profile.max_drawdown.max_drawdown:.2%}",
                f"VaR (%95): {risk_profile.var_95:.2%}",
            ])
        except Exception:
            # Yetersiz veri (yeni portföy) — AI'a "risk verisi henüz yok"
            # bilgisini GERÇEĞE UYGUN şekilde iletiyoruz, UYDURMUYORUZ.
            lines.append("\nRisk metrikleri: henüz yeterli veri yok (yeni portföy).")

        return "\n".join(lines)
