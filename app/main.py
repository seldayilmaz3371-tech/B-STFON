"""
Portföy Takip Sistemi — BIST Analiz ve Portföy Yönetim Paneli.

Streamlit ana giriş noktası. Bu dosyanın TEK sorumluluğu orkestrasyondur:
  1. Sayfa ayarlarını yapılandırmak.
  2. DI container'ını Streamlit oturumu boyunca tek sefer kurmak.
  3. İki sekmeyi (tekil hisse / portföy özeti) yönetmek.
  4. Servis katmanından gelen hataları kullanıcıya temiz mesajlar olarak iletmek.

Katman İzolasyonu (kesinlikle ihlal edilmez):
  Bu dosya src.infrastructure veya src.domain altından HİÇBİR ŞEY import etmez.
  Infrastructure bağımlılıkları (YFinanceAdapter, TechnicalCalculator vb.)
  tamamen Container arkasına gizlenmiştir; app.py onları asla görmez.

Sekme tasarımı ve session state stratejisi:
  Her sekmenin kendi state anahtarları vardır (_STATE_* sabitleri).
  Sekme değişikliği diğer sekmenin state'ini bozmaz.
  Portföy sekmesi her gösterimde yeniden hesaplanır (şimdilik):
  gerçek zamanlı PnL'de cache'lemenin maliyeti-faydası, veri tazeliği
  gereksinimi ile dengelenmek zorundadır. Gelecekte APScheduler ile
  periyodik cache invalidation uygulanabilir.

Çalıştırma:
    streamlit run app/main.py
"""

from __future__ import annotations

import streamlit as st

from app.container import Container, get_container
from src.presentation.dashboard.components.metrics_display import render_metrics
from src.presentation.dashboard.components.portfolio_view import render_portfolio
from src.presentation.dashboard.components.search_bar import (
    SearchQuery,
    render_search_bar,
    validate_symbol_format,
)
from src.presentation.dashboard.components.technical_chart import render_technical_chart
from src.services.market_data_service import MarketAnalysisResult
from src.services.portfolio_service import PortfolioSummaryDTO

_PAGE_TITLE = "Portfolio OS — BIST Analiz Paneli"

# Session state anahtarları — sabit string'ler tek yerde, yazım hatası önler
_STATE_LAST_SYMBOL    = "last_searched_symbol"
_STATE_LAST_TIMEFRAME = "last_searched_timeframe"
_STATE_LAST_RESULT    = "last_analysis_result"
_STATE_PORTFOLIO_ID   = "active_portfolio_id"

_DEFAULT_SYMBOL       = "THYAO"
_DEFAULT_PORTFOLIO_ID = "default"  # Geliştirme aşamasında sabit; ileride kullanıcı seçiminden gelecek


def configure_page() -> None:
    st.set_page_config(
        page_title=_PAGE_TITLE,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


@st.cache_resource(show_spinner=False)
def _get_cached_container() -> Container:
    """
    DI container'ı: Streamlit oturumu boyunca tek instance.

    @st.cache_resource: her rerun'da Container, DB engine ve adapter
    sıfırdan oluşturulmaz — aynı instance paylaşılır.
    Bu, bağlantı havuzu (connection pool) stabilite ve performans için kritik.
    """
    return get_container()


def _init_session_state() -> None:
    defaults = {
        _STATE_LAST_SYMBOL:    _DEFAULT_SYMBOL,
        _STATE_LAST_TIMEFRAME: "1d",
        _STATE_LAST_RESULT:    None,
        _STATE_PORTFOLIO_ID:   _DEFAULT_PORTFOLIO_ID,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Sekme 1: Tekil Hisse Analizi ─────────────────────────────────────────────

def _fetch_analysis(container: Container, query: SearchQuery) -> MarketAnalysisResult | None:
    """
    market_data_service'ten veri çeker; tüm hataları UI mesajına dönüştürür.

    Katman notu: Exception by-name import edilmez — presentation, servis
    hiyerarşisini bilmez; str(exc) yeterli bilgiyi taşır.
    """
    try:
        with st.spinner(f"{query.symbol} için veri çekiliyor…"):
            return container.market_data_service.get_market_analysis(
                symbol=query.symbol,
                timeframe=query.timeframe,
            )
    except Exception as exc:
        # DÜZELTME (bu turda): SymbolNotFoundError artık bazı durumlarda
        # context'inde bir 'note' alanı taşıyor (ağ hatası/geçersiz sembol
        # ayrımının GÜVENİLİR yapılamadığı durumlarda dürüst bir uyarı —
        # bkz. yfinance_adapter.py). getattr ile DUCK-TYPING kullanılıyor,
        # import YOK (katman izolasyonu korunuyor — app.py'ın kendi kuralı:
        # "hiyerarşisini bilmez, str(exc) yeterli").
        note = getattr(exc, "context", {}).get("note") if hasattr(exc, "context") else None
        message = f"**{query.symbol}** analizi başarısız.\n\nSebep: {exc}"
        if note:
            message += f"\n\n_{note}_"
        st.error(message, icon="⚠️")
        return None


def render_analysis_tab(container: Container) -> None:
    """Tekil Hisse Analizi sekmesinin tüm içeriğini çizer."""
    query = render_search_bar(default_symbol=st.session_state[_STATE_LAST_SYMBOL])

    if query.submitted:
        err = validate_symbol_format(query.symbol)
        if err:
            st.warning(err, icon="⚠️")
        else:
            result = _fetch_analysis(container, query)
            st.session_state[_STATE_LAST_RESULT]    = result
            st.session_state[_STATE_LAST_SYMBOL]    = query.symbol
            st.session_state[_STATE_LAST_TIMEFRAME] = query.timeframe

    cached: MarketAnalysisResult | None = st.session_state[_STATE_LAST_RESULT]

    if cached is None:
        st.info(
            "Bir hisse sembolü girin ve **Analiz Et**'e tıklayın.",
            icon="👆",
        )
        return

    render_metrics(cached)
    st.divider()
    render_technical_chart(cached)


# ── Sekme 2: Portföyüm ────────────────────────────────────────────────────────

def _fetch_portfolio(container: Container, portfolio_id: str) -> PortfolioSummaryDTO | None:
    """
    portfolio_service'ten portföy özeti çeker; hataları UI mesajına dönüştürür.

    PortfolioService kendi içinde hata toleransı uygular (per-sembol düzeyinde),
    bu yüzden buradaki hata yalnızca servisin kendisinin çökmesi anlamına gelir
    (çok nadir, örn. DB erişilemez).
    """
    try:
        with st.spinner("Portföy durumu hesaplanıyor…"):
            return container.portfolio_service.get_portfolio_status(portfolio_id)
    except Exception as exc:
        st.error(
            f"Portföy durumu alınamadı.\n\nSebep: {exc}",
            icon="🛑",
        )
        return None


def _render_create_portfolio_form(container: Container) -> None:
    """
    Yeni portföy oluşturma formu.

    DÜZELTME (bu turda bulundu): benchmark_code alanı bu formda HİÇ
    YOKTU — Portfolio.benchmark_code (Faz E'de, "portföy başına
    kullanıcı seçimi" kararıyla eklenmişti) ilk kez UI'da görünüyor.
    Bu olmadan RiskService.calculate_relative_performance() hiçbir
    portföy için ÇAĞRILAMAZDI (benchmark_code=None kalırdı).

    Katman izolasyonu notu: Bu form yalnızca PLAIN STRING/date
    gönderir (CurrencyCode/CostMethod enum'larını İMPORT ETMEZ).
    Enum dönüşümü PortfolioService.create_portfolio() içinde yapılır.
    Buradaki seçenekler (currency_options, cost_method_options,
    benchmark_options) DB CHECK constraint'leri VE Portfolio.py'daki
    _KNOWN_BENCHMARK_CODES ile SENKRON TUTULMALI ama bu senkronizasyon
    otomatik DEĞİL (üç ayrı yerde hardcoded) — ileride bir tutarsızlık
    kaynağı olabilir, bu bilinçli bir teknik borç.
    """
    with st.expander("➕ Yeni Portföy Oluştur", expanded=False):
        with st.form(key="create_portfolio_form"):
            name = st.text_input("Portföy Adı", placeholder="Örn. Ana Portföy")
            col1, col2 = st.columns(2)
            with col1:
                currency = st.selectbox("Para Birimi", options=["TRY", "USD", "EUR"])
            with col2:
                cost_method = st.selectbox(
                    "Maliyet Yöntemi", options=["WAVG", "FIFO"],
                    help="LIFO şu an desteklenmiyor.",
                )
            # DÜZELTME (bu turda): Bu liste ÖNCEDEN hardcoded 3 endeksti
            # ve Portfolio.py'daki doğrulama listesiyle bile SENKRON
            # DEĞİLDİ (3 ayrı yerde bağımsız liste riski GERÇEKLEŞMİŞTİ).
            # Artık container.portfolio_service.get_available_benchmark_codes()
            # ile TEK kanonik kaynaktan (Portfolio.KNOWN_BENCHMARKS) besleniyor.
            benchmark_options = container.portfolio_service.get_available_benchmark_codes()
            benchmark_labels = ["Yok"] + [f"{b['code']} ({b['name']})" for b in benchmark_options]
            selected_benchmark_label = st.selectbox(
                "Benchmark (opsiyonel)",
                options=benchmark_labels,
                help="Risk analizinde bu portföyü karşılaştıracağınız endeks. "
                     "Kürasyonla seçilmiş 7 endeks — BIST'in TAMAMI değil "
                     "(90'dan fazla endeks var, çoğu dar sektör alt-endeksi).",
            )
            benchmark_code = (
                None if selected_benchmark_label == "Yok"
                else benchmark_options[benchmark_labels.index(selected_benchmark_label) - 1]["code"]
            )
            description = st.text_area("Açıklama (opsiyonel)", placeholder="")
            submitted = st.form_submit_button("Oluştur", width="stretch")

        if submitted:
            if not name.strip():
                st.warning("Portföy adı boş olamaz.", icon="⚠️")
                return
            try:
                created = container.portfolio_service.create_portfolio(
                    name=name.strip(),
                    currency=currency,
                    cost_method=cost_method,
                    benchmark_code=benchmark_code,  # zaten yukarıda çözümlendi (None ya da gerçek kod)
                    description=description.strip() or None,
                )
                st.success(f"**{created.name}** portföyü oluşturuldu.", icon="✅")
                st.rerun()
            except Exception as exc:
                st.error(f"Portföy oluşturulamadı.\n\nSebep: {exc}", icon="🛑")


def _render_add_transaction_form(container: Container, portfolio_id: str) -> None:
    """
    İşlem ekleme formu.

    Katman izolasyonu notu: TransactionType enum'u İMPORT EDİLMİYOR —
    yalnızca plain string gönderiliyor (bkz. TransactionService.
    add_transaction()'ın string-kabul-eden imzası, create_portfolio()
    ile aynı mimari gerekçeyle).

    Kapsam kararı (DÜZELTME — bu turda genişletildi): Önceden yalnızca
    BUY/SELL/DIVIDEND destekleniyordu. SPLIT/BONUS_SHARE'in backend'i
    (TransactionService, CostBasisCalculator — GD-002/GD-005 golden
    dataset ile doğrulanmış) HER ZAMAN hazırdı, yalnızca UI eksikti —
    bu turda kapatıldı. RIGHTS_USED/RIGHTS_SOLD/REVERSE_SPLIT/MERGER
    HÂLÂ eklenmedi — bunlar CostBasisCalculator'da hâlâ BusinessRuleError
    fırlatıyor (gerçek golden dataset yok, bilinçli olarak reddediliyor).
    """
    with st.expander("➕ İşlem Ekle", expanded=False):
        with st.form(key="add_transaction_form"):
            col1, col2 = st.columns(2)
            with col1:
                symbol = st.text_input("Sembol", placeholder="Örn. THYAO")
                transaction_type = st.selectbox(
                    "İşlem Tipi",
                    options=["BUY", "SELL", "DIVIDEND", "SPLIT", "REVERSE_SPLIT", "BONUS_SHARE", "RIGHTS_USED"],
                    format_func=lambda x: {
                        "BUY": "Alış", "SELL": "Satış", "DIVIDEND": "Temettü",
                        "SPLIT": "Bölünme (Split)", "REVERSE_SPLIT": "Ters Bölünme",
                        "BONUS_SHARE": "Bedelsiz Hisse", "RIGHTS_USED": "Rüçhan Hakkı Kullanımı",
                    }[x],
                )
            with col2:
                # SPLIT'te quantity/price ANLAMSIZ (yok sayılır, bkz.
                # Transaction.__post_init__) — GİZLENİYOR, yanıltmamak için.
                if transaction_type not in ("SPLIT", "REVERSE_SPLIT"):
                    quantity = st.number_input(
                        "Bedelsiz Adet" if transaction_type == "BONUS_SHARE" else "Miktar",
                        min_value=0.0, step=1.0, format="%.4f",
                    )
                    price = st.number_input("Fiyat (₺)", min_value=0.0, step=0.01, format="%.4f")
                else:
                    quantity = 0.0
                    price = 0.0
                    st.caption("Bu işlem tipinde miktar/fiyat gerekmez — mevcut pozisyon otomatik ayarlanır.")

            trade_date = st.date_input("İşlem Tarihi")
            net_amount = None
            split_ratio = None
            if transaction_type == "DIVIDEND":
                net_amount = st.number_input(
                    "Net Temettü Tutarı (₺)", min_value=0.0, step=0.01, format="%.2f",
                    help="Stopaj sonrası net tutar — pozisyonu etkilemez, yalnızca bilgi amaçlıdır.",
                )
            elif transaction_type == "SPLIT":
                split_ratio = st.number_input(
                    "Split Oranı", min_value=1.01, step=0.5, format="%.2f", value=2.0,
                    help="Örn. 2:1 split için 2 — mevcut pozisyonunuz bu oranla ÇARPILIR, "
                         "toplam maliyetiniz DEĞİŞMEZ (yalnızca birim maliyet düşer).",
                )
            elif transaction_type == "REVERSE_SPLIT":
                split_ratio = st.number_input(
                    "Ters Bölünme Oranı (Azaltma Faktörü)", min_value=1.01, step=0.5,
                    format="%.2f", value=10.0,
                    help="Örn. 1:10 ters bölünme için 10 — mevcut pozisyonunuz bu orana "
                         "BÖLÜNÜR, toplam maliyetiniz DEĞİŞMEZ (yalnızca birim maliyet artar).",
                )
            elif transaction_type == "BONUS_SHARE":
                st.caption(
                    "Bedelsiz hisseler, toplam maliyetinizi DEĞİŞTİRMEZ — "
                    "yalnızca sahip olduğunuz adet artar (birim maliyetiniz düşer)."
                )

            submitted = st.form_submit_button("Kaydet", width="stretch")

        if submitted:
            try:
                from decimal import Decimal
                container.transaction_service.add_transaction(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    transaction_type=transaction_type,
                    quantity=Decimal(str(quantity)),
                    price=Decimal(str(price)),
                    trade_date=trade_date,
                    net_amount=Decimal(str(net_amount)) if net_amount is not None else None,
                    split_ratio=Decimal(str(split_ratio)) if split_ratio is not None else None,
                )
                st.success(f"İşlem kaydedildi: {symbol}", icon="✅")
                st.rerun()
            except Exception as exc:
                st.error(f"İşlem kaydedilemedi.\n\nSebep: {exc}", icon="🛑")


def _render_transaction_history(container: Container, portfolio_id: str) -> None:
    """
    İşlem geçmişi + reversal (iptal) arayüzü.

    Immutability kuralı UI'a da yansıtıldı: DÜZENLEME butonu YOK,
    yalnızca "İptal Et" (reversal) var — bu, Transaction'ın frozen
    dataclass olmasıyla ve TransactionService.reverse_transaction()'ın
    UPDATE değil INSERT+is_active=0 deseniyle TUTARLI.
    """
    with st.expander("📜 İşlem Geçmişi", expanded=False):
        try:
            transactions = container.transaction_service.list_transactions(portfolio_id)
        except Exception as exc:
            st.error(f"İşlem geçmişi alınamadı: {exc}", icon="🛑")
            return

        if not transactions:
            st.caption("Henüz işlem yok.")
            return

        for tx in transactions:
            cols = st.columns([2, 1, 1, 1, 1, 1])
            cols[0].write(f"**{tx.symbol}**")
            cols[1].write(tx.transaction_type.value if hasattr(tx.transaction_type, "value") else tx.transaction_type)
            cols[2].write(f"{tx.quantity}")
            cols[3].write(f"{tx.price:.2f} ₺")
            cols[4].write(tx.timestamp.strftime("%d.%m.%Y"))
            if cols[5].button("İptal Et", key=f"reverse_{tx.transaction_id}"):
                st.session_state[f"confirm_reverse_{tx.transaction_id}"] = True

            if st.session_state.get(f"confirm_reverse_{tx.transaction_id}"):
                reason = st.text_input(
                    "İptal sebebi (zorunlu)", key=f"reason_{tx.transaction_id}",
                )
                if st.button("Onayla", key=f"confirm_{tx.transaction_id}"):
                    try:
                        container.transaction_service.reverse_transaction(tx.transaction_id, reason)
                        st.success("İşlem iptal edildi.", icon="✅")
                        del st.session_state[f"confirm_reverse_{tx.transaction_id}"]
                        st.rerun()
                    except Exception as exc:
                        st.error(f"İptal başarısız: {exc}", icon="🛑")


def _render_ai_summary(container: Container, portfolio_id: str) -> None:
    """
    AI destekli doğal dil portföy özeti — bu turda eklendi (kullanıcı
    onayıyla: kendi API anahtarını .env'e giriyor, maliyeti üstleniyor).

    MİMARİ PRENSİP: AI, YENİ bir hesaplama YAPMIYOR — YALNIZCA ZATEN
    hesaplanmış (RiskService/PortfolioService) sayıları yorumluyor.
    """
    with st.expander("🤖 AI Portföy Özeti", expanded=False):
        if not container.ai_insight_service.is_configured:
            st.caption(
                "AI özeti için API anahtarı ayarlanmamış. .env dosyanıza "
                "`PORTFOLIO_OS__AI__API_KEY=<kendi-anahtarınız>` ekleyin "
                "(kendi Anthropic API anahtarınız gerekir, maliyeti size aittir)."
            )
            return

        st.info(
            "⚠️ Bu bir yatırım tavsiyesi DEĞİLDİR — yalnızca ZATEN hesaplanmış "
            "risk/performans verilerinizin doğal dile çevrilmiş bir özetidir.",
            icon="ℹ️",
        )
        if st.button("AI Özeti Oluştur", key=f"ai_summary_{portfolio_id}"):
            with st.spinner("AI özeti oluşturuluyor…"):
                try:
                    summary = container.ai_insight_service.generate_portfolio_summary(portfolio_id)
                    st.markdown(summary)
                except Exception as exc:
                    st.error(f"AI özeti oluşturulamadı.\n\nSebep: {exc}", icon="🛑")

        st.divider()
        st.caption("📄 Daha ayrıntılı, indirilebilir bir performans raporu oluşturun:")
        if st.button("Rapor Oluştur", key=f"ai_report_{portfolio_id}"):
            with st.spinner("Rapor oluşturuluyor…"):
                try:
                    report = container.ai_insight_service.generate_portfolio_report(portfolio_id)
                    st.markdown(report)

                    col_md, col_docx = st.columns(2)
                    with col_md:
                        st.download_button(
                            "📥 Raporu İndir (.md)", data=report,
                            file_name=f"portfoy_raporu_{portfolio_id[:8]}.md",
                            mime="text/markdown", key=f"ai_report_download_md_{portfolio_id}",
                        )
                    with col_docx:
                        # DÜZELTME (bu turda eklendi): python-docx ile
                        # DAĞITILAN uygulamanın KENDİ Python ortamında
                        # üretilir — Node.js/docx-js GEREKTİRMEZ (bkz.
                        # document_export.py modül docstring'i).
                        try:
                            from src.infrastructure.export.document_export import (
                                markdown_report_to_docx_bytes,
                            )
                            report_portfolio = container.portfolio_service.get_portfolio(portfolio_id)
                            report_title = report_portfolio.name if report_portfolio is not None else "Portföy Raporu"
                            docx_bytes = markdown_report_to_docx_bytes(report, report_title)
                            st.download_button(
                                "📥 Raporu İndir (.docx)", data=docx_bytes,
                                file_name=f"portfoy_raporu_{portfolio_id[:8]}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"ai_report_download_docx_{portfolio_id}",
                            )
                        except Exception as exc:
                            st.caption(f".docx oluşturulamadı: {exc}")
                except Exception as exc:
                    st.error(f"Rapor oluşturulamadı.\n\nSebep: {exc}", icon="🛑")

        st.divider()
        st.caption("💬 Portföyünüz hakkında serbest metinli bir soru sorun (yalnızca zaten hesaplanmış verilere dayanarak cevaplanır):")
        question = st.text_input("Soru", placeholder="Örn. Hangi pozisyonum en çok zarar ediyor?", key=f"ai_question_{portfolio_id}")
        if st.button("Sor", key=f"ai_ask_{portfolio_id}") and question:
            with st.spinner("AI cevap oluşturuyor…"):
                try:
                    answer = container.ai_insight_service.answer_portfolio_question(portfolio_id, question)
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"Cevap oluşturulamadı.\n\nSebep: {exc}", icon="🛑")


def _render_risk_analysis(container: Container, portfolio_id: str) -> None:
    """
    Risk analizi bölümü — RiskService.compute_risk_profile() ve
    calculate_relative_performance() için UI.

    MİMARİ KARAR — otomatik DEĞİL, buton ile tetiklenen hesaplama:
    RiskService'in hesaplaması PAHALI (sembol başına network çağrısı +
    pozisyon zaman serisi inşası). Streamlit'in her rerun'da TÜM
    script'i yeniden çalıştırdığı göz önüne alınırsa, bunu otomatik
    (sayfa her açıldığında) çalıştırmak GEREKSİZ maliyet ve YAVAŞ bir
    UI demektir. Kullanıcı "Hesapla" butonuna basana kadar HİÇBİR
    network çağrısı yapılmaz — bu, Streamlit performans best-practice'i
    ile TUTARLI (pahalı işlemleri render path'inden çıkar).

    MİMARİ KARAR — InsufficientDataError'ı ÖZEL OLARAK yakalama:
    Yeni oluşturulmuş bir portföyün (bugün açılan bir pozisyon)
    lookback penceresinde 30 iş günü OLAMAZ — bu, GERÇEK ve SIK
    karşılaşılacak bir durum, "hata" değil "henüz yeterli veri yok"
    anlamına geliyor. Bu ayrım kullanıcıya net şekilde gösteriliyor
    (ham exception mesajı DEĞİL).
    """
    with st.expander("📊 Risk Analizi", expanded=False):
        if st.button("Risk Metriklerini Hesapla", key=f"compute_risk_{portfolio_id}"):
            try:
                profile = container.risk_service.compute_risk_profile(portfolio_id)
            except Exception as exc:
                # type(exc).__name__ ile string karşılaştırma — app.py'ın
                # katman izolasyonu kuralı (domain exception'ları import
                # ETMEZ) gereği isinstance() KULLANILAMIYOR. RİSK: bu bir
                # "magic string" bağımlılığı — InsufficientDataError adı
                # değişirse bu kontrol SESSİZCE devre dışı kalır (ama
                # ÇÖKMEZ, yalnızca daha genel bir hata mesajına düşer —
                # bu kabul edilebilir bir bozulma modu, kritik değil).
                if type(exc).__name__ == "InsufficientDataError":
                    st.info(
                        f"Risk metrikleri için en az {getattr(exc, 'required', '?')} "
                        f"veri noktası gerekiyor, mevcut: {getattr(exc, 'available', '?')}. "
                        "Portföyünüz için daha uzun bir fiyat geçmişi biriktikçe "
                        "bu analiz kullanılabilir olacak.",
                        icon="ℹ️",
                    )
                else:
                    st.error(f"Risk metrikleri hesaplanamadı: {exc}", icon="🛑")
                return

            col1, col2, col3 = st.columns(3)
            col1.metric("Yıllıklandırılmış Volatilite", f"{profile.annualized_volatility:.2%}")
            col2.metric("Sharpe Oranı", f"{profile.sharpe_ratio:.2f}")
            col3.metric("Sortino Oranı", f"{profile.sortino_ratio:.2f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("VaR (%95)", f"{profile.var_95:.2%}")
            col5.metric("CVaR (%95)", f"{profile.cvar_95:.2%}")
            col6.metric("Maks. Düşüş", f"{profile.max_drawdown.max_drawdown:.2%}")

            # DÜZELTME (bu turda eklendi): compute_risk_profile() yalnızca
            # TEK SAYI (nokta tahmini) veriyordu — "volatilite zamanla
            # nasıl değişti", "en derin düşüşler ne zaman yaşandı"
            # sorularına cevap YOKTU. get_rolling_volatility()/
            # get_drawdown_series() bu boşluğu kapatıyor.
            try:
                rolling_vol = container.risk_service.get_rolling_volatility(
                    portfolio_id, window=30, include_cash=False,
                )
                drawdown_series = container.risk_service.get_drawdown_series(
                    portfolio_id, include_cash=False,
                )
                st.caption("30 günlük kayan yıllıklandırılmış volatilite")
                st.line_chart(rolling_vol)
                st.caption("Zaman içinde düşüş (drawdown) serisi")
                st.area_chart(drawdown_series)

                if container.ai_insight_service.is_configured:
                    if st.button("🤖 AI ile Anormallik Kontrolü", key=f"ai_anomaly_{portfolio_id}"):
                        with st.spinner("Analiz ediliyor…"):
                            try:
                                anomaly_report = container.ai_insight_service.detect_anomalies(portfolio_id)
                                st.markdown(anomaly_report)
                            except Exception as exc:
                                st.error(f"Analiz oluşturulamadı.\n\nSebep: {exc}", icon="🛑")
            except Exception as exc:
                if type(exc).__name__ != "InsufficientDataError":
                    st.caption(f"Zaman serisi grafikleri hesaplanamadı: {exc}")
                # InsufficientDataError sessizce atlanıyor — ana risk
                # metrikleri ZATEN gösterildi, bu yalnızca EK bir görselleştirme.

            portfolio = container.portfolio_service.get_portfolio(portfolio_id)
            if portfolio is not None and portfolio.benchmark_code:
                st.divider()
                st.caption(f"Benchmark: {portfolio.benchmark_code}")
                try:
                    perf = container.risk_service.calculate_relative_performance(
                        portfolio_id, portfolio.benchmark_code,
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Beta", f"{perf.beta:.2f}")
                    c2.metric("Alpha (yıllık)", f"{perf.alpha_annualized:.2%}")
                    c3.metric("R²", f"{perf.r_squared:.2f}")
                except Exception as exc:
                    if type(exc).__name__ == "InsufficientDataError":
                        st.caption("Benchmark karşılaştırması için yeterli veri yok.")
                    else:
                        st.warning(f"Benchmark karşılaştırması hesaplanamadı: {exc}")
            elif portfolio is not None:
                st.caption(
                    "Bu portföy için benchmark tanımlanmamış — "
                    "karşılaştırmalı performans (beta, alpha) gösterilemiyor."
                )


def _render_cash_and_integrity(container: Container, portfolio_id: str) -> None:
    """
    DÜZELTME (bu turda bulundu — İKİNCİ unutulmuş entegrasyon): Hem
    nakit bakiyesi HEM DE `verify_balance()` (Faz B'de inşa edilmiş,
    hiç çağrılmayan bir tutarlılık kontrolü) daha önce UI'da HİÇ
    görünmüyordu. TransactionService artık her yazımdan sonra otomatik
    kontrol ediyor (bkz. _check_ledger_integrity) — burası, kullanıcının
    isteği zaman MANUEL olarak da tetikleyebildiği yer.
    """
    try:
        balance = container.transaction_service.get_cash_balance(portfolio_id)
        st.metric("💰 Nakit Bakiyesi", f"{balance:,.2f} ₺")
    except Exception as exc:
        st.caption(f"Nakit bakiyesi alınamadı: {exc}")

    with st.expander("🔒 Nakit Ledger Bütünlüğü", expanded=False):
        st.caption(
            "Her işlem eklendiğinde/iptal edildiğinde OTOMATİK olarak "
            "kontrol edilir. Bu buton, isteğe bağlı manuel doğrulama sağlar."
        )
        if st.button("Bütünlüğü Doğrula", key=f"verify_ledger_{portfolio_id}"):
            try:
                verification = container.transaction_service.check_ledger_integrity(portfolio_id)
                if verification.is_consistent:
                    st.success("Nakit ledger tutarlı. ✅", icon="✅")
                else:
                    st.error(
                        f"TUTARSIZLIK TESPİT EDİLDİ — beklenen: {verification.expected:,.2f} ₺, "
                        f"gerçek: {verification.actual:,.2f} ₺, fark: {verification.discrepancy:,.2f} ₺",
                        icon="🛑",
                    )
            except Exception as exc:
                st.error(f"Doğrulama yapılamadı: {exc}", icon="🛑")


def render_portfolio_tab(container: Container) -> None:
    """
    Portföyüm sekmesinin tüm içeriğini çizer.

    Aşama 8: "default" hardcoded ID kaldırıldı.
    PortfolioService.list_portfolios() ile aktif portföyler listelenir;
    kullanıcı st.selectbox ile seçim yapar.

    Bu turda eklenen: Portföy oluşturma formu + İşlem ekleme/iptal
    arayüzü — daha önce sistemde HİÇBİR kullanıcı arayüzünden veri
    girme yolu yoktu.
    """
    _render_create_portfolio_form(container)
    st.divider()

    # ── Portföy listesini çek ────────────────────────────────────────────────
    try:
        portfolios = container.portfolio_service.list_portfolios()
    except Exception as exc:
        st.error(f"Portföy listesi alınamadı: {exc}", icon="🛑")
        return

    if not portfolios:
        st.info(
            "Henüz portföy oluşturulmamış. "
            "Yukarıdaki formu kullanarak ilk portföyünüzü oluşturun.",
            icon="📭",
        )
        return

    # ── Portföy seçimi ────────────────────────────────────────────────────────
    col_select, col_refresh = st.columns([5, 1], vertical_alignment="bottom")

    with col_select:
        options = {p["name"]: p["id"] for p in portfolios}
        selected_name = st.selectbox(
            label="Portföy",
            options=list(options.keys()),
            help="Görüntülemek istediğiniz portföyü seçin.",
        )

    with col_refresh:
        st.button("🔄 Yenile", width="stretch", key="portfolio_refresh")

    selected_id = options[selected_name]
    st.session_state[_STATE_PORTFOLIO_ID] = selected_id

    _render_add_transaction_form(container, selected_id)
    _render_cash_and_integrity(container, selected_id)
    _render_transaction_history(container, selected_id)
    _render_ai_summary(container, selected_id)
    _render_risk_analysis(container, selected_id)
    st.divider()

    # ── Portföy durumu hesapla ────────────────────────────────────────────────
    summary = _fetch_portfolio(container, selected_id)
    if summary is not None:
        render_portfolio(summary)


# ── Ana Orkestratör ───────────────────────────────────────────────────────────

def _render_scheduler_sidebar(container: Container) -> None:
    """
    DÜZELTME (bu turda — Faz F): APScheduler entegrasyonu daha önce
    hiç yoktu. Scheduler, portföyden BAĞIMSIZ (tüm aktif portföyleri
    işliyor) olduğu için sidebar'a, herhangi bir sekmeye bağlı olmadan
    yerleştirildi.

    start() BURADA, HER render_app() çağrısında (= her Streamlit
    rerun'da) çağrılıyor — ama SnapshotScheduler.start() idempotent
    (zaten çalışıyorsa no-op), bu yüzden güvenli. Container zaten
    @st.cache_resource ile tekil olduğu için scheduler nesnesi de
    tekil — bu iki katmanlı idempotency (Container tekilliği +
    start()'ın kendi idempotency kontrolü) rerun döngüsünde çift
    güvence sağlıyor.
    """
    with st.sidebar:
        st.subheader("⏱️ Arka Plan Senkronizasyonu")
        try:
            scheduler = container.scheduler
            if container.scheduler_enabled:
                scheduler.start()
                st.caption(
                    f"Otomatik: her {container.scheduler_interval_hours} "
                    f"saatte bir risk snapshot'ı yenilenir."
                )
                st.caption(f"Durum: {'🟢 Çalışıyor' if scheduler.is_running else '🔴 Durdu'}")
            else:
                st.caption("Otomatik senkronizasyon devre dışı (settings.yaml).")

            if st.button("Şimdi Senkronize Et", key="manual_sync_button"):
                with st.spinner("Tüm portföyler için risk snapshot'ları hesaplanıyor..."):
                    result = scheduler.run_now()
                st.success(
                    f"Tamamlandı: {result['succeeded']} başarılı, {result['failed']} başarısız.",
                    icon="✅",
                )
        except Exception as exc:
            st.caption(f"Scheduler durumu alınamadı: {exc}")


def render_watchlist_tab(container: Container) -> None:
    """
    İzleme listesi sekmesi.

    KAPSAM (bilinçli sınırlama): Yalnızca CRUD (liste oluştur, sembol
    ekle/çıkar). Fiyat alarmı ALANLARI (alert_price_low/high) formda
    var ama GERÇEK alarm tetikleme/bildirim YOK — bkz.
    watchlist.py modül docstring'i.
    """
    with st.expander("➕ Yeni İzleme Listesi Oluştur", expanded=False):
        with st.form(key="create_watchlist_form"):
            name = st.text_input("Liste Adı", placeholder="Örn. BIST Favorilerim")
            description = st.text_area("Açıklama (opsiyonel)", placeholder="")
            submitted = st.form_submit_button("Oluştur", width="stretch")

        if submitted:
            if not name.strip():
                st.warning("Liste adı boş olamaz.", icon="⚠️")
            else:
                try:
                    created = container.watchlist_service.create_watchlist(
                        name=name.strip(), description=description.strip() or None,
                    )
                    st.success(f"**{created.name}** listesi oluşturuldu.", icon="✅")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Liste oluşturulamadı.\n\nSebep: {exc}", icon="🛑")

    st.divider()

    try:
        watchlists = container.watchlist_service.list_watchlists()
    except Exception as exc:
        st.error(f"İzleme listeleri alınamadı: {exc}", icon="🛑")
        return

    if not watchlists:
        st.info("Henüz izleme listesi oluşturulmamış.", icon="📭")
        return

    options = {w.name: w.watchlist_id for w in watchlists}
    selected_name = st.selectbox("İzleme Listesi", options=list(options.keys()))
    selected_id = options[selected_name]

    with st.expander("➕ Sembol Ekle", expanded=False):
        with st.form(key="add_watchlist_symbol_form"):
            symbol = st.text_input("Sembol", placeholder="Örn. THYAO")
            col1, col2 = st.columns(2)
            with col1:
                alert_low = st.number_input("Alt Alarm Fiyatı (opsiyonel)", min_value=0.0, step=0.01, format="%.2f")
            with col2:
                alert_high = st.number_input("Üst Alarm Fiyatı (opsiyonel)", min_value=0.0, step=0.01, format="%.2f")
            add_submitted = st.form_submit_button("Ekle", width="stretch")

        if add_submitted:
            try:
                from decimal import Decimal
                container.watchlist_service.add_symbol(
                    selected_id, symbol,
                    alert_price_low=Decimal(str(alert_low)) if alert_low > 0 else None,
                    alert_price_high=Decimal(str(alert_high)) if alert_high > 0 else None,
                )
                st.success(f"**{symbol}** eklendi.", icon="✅")
                st.rerun()
            except Exception as exc:
                st.error(f"Sembol eklenemedi.\n\nSebep: {exc}", icon="🛑")

    try:
        items = container.watchlist_service.list_symbols(selected_id)
    except Exception as exc:
        st.error(f"Semboller alınamadı: {exc}", icon="🛑")
        return

    if not items:
        st.caption("Bu listede henüz sembol yok.")
        return

    price_status_key = f"watchlist_price_statuses_{selected_id}"
    if st.button("Fiyatları Güncelle", key=f"refresh_watchlist_prices_{selected_id}"):
        # DÜZELTME (bu turda): Kullanıcının girdiği alarm eşikleri
        # (alert_price_low/high) daha önce HİÇBİR YERDE kontrol
        # edilmiyordu. Bu buton, Risk Analizi'nin "Hesapla" butonuyla
        # AYNI UX deseniyle (pahalı/network-bağımlı işlem, otomatik
        # değil açık tetiklemeli) güncel fiyatları çekip alarm
        # eşikleriyle karşılaştırıyor.
        with st.spinner("Güncel fiyatlar alınıyor…"):
            st.session_state[price_status_key] = {
                s.item.symbol: s
                for s in container.watchlist_service.get_items_with_current_price(selected_id)
            }

    # DÜZELTME: fiyat durumu session_state'te TUTULUYOR (erken return
    # YOK) — böylece "Fiyatları Güncelle" butonuna basmak "Kaldır"
    # listesini GİZLEMİYOR, ikisi BİRLİKTE gösteriliyor.
    price_statuses = st.session_state.get(price_status_key, {})

    for item in items:
        cols = st.columns([2, 2, 2, 1])
        cols[0].write(f"**{item.symbol}**")

        status = price_statuses.get(item.symbol)
        if status is None:
            cols[1].caption("Fiyat için 'Fiyatları Güncelle'ye basın")
        elif status.fetch_error is not None:
            cols[1].caption("⚠️ Fiyat alınamadı")
        elif status.current_price is not None:
            if status.alarm_triggered:
                direction_text = "🔴 Alt eşik" if status.alarm_direction == "low" else "🔴 Üst eşik"
                cols[1].write(f"{status.current_price:.2f} ₺")
                cols[1].caption(direction_text)
            else:
                cols[1].write(f"🟢 {status.current_price:.2f} ₺")

        alert_text = ""
        if item.alert_price_low is not None:
            alert_text += f"Alt: {item.alert_price_low:.2f} ₺ "
        if item.alert_price_high is not None:
            alert_text += f"Üst: {item.alert_price_high:.2f} ₺"
        cols[2].caption(alert_text or "Alarm yok")

        if cols[3].button("Kaldır", key=f"remove_watchlist_item_{item.item_id}"):
            try:
                container.watchlist_service.remove_symbol(item.item_id)
                st.rerun()
            except Exception as exc:
                st.error(f"Kaldırılamadı: {exc}", icon="🛑")

    if price_statuses:
        st.caption(
            "Not: Fiyatlar isteğe bağlı görüntülemedir — arka planda otomatik "
            "alarm/bildirim YOK, yalnızca 'Fiyatları Güncelle'ye her bastığınızda tazelenir."
        )


def render_backtest_tab(container: Container) -> None:
    """
    Backtest sekmesi — YALNIZCA örnek SMA Crossover stratejisi.

    KAPSAM (bilinçli sınırlama, bkz. backtest_engine.py modül
    docstring'i): tek strateji. Bu bir yatırım tavsiyesi
    değil — yalnızca backtest motorunun doğruluğunu göstermek için
    referans bir implementasyon.

    DÜZELTME (bu turda eklendi): Artık VİRGÜLLE AYRILMIŞ birden fazla
    sembol desteğiyle — bkz. BacktestService.run_multi_symbol()
    mimari kararı (eşit sermaye paylaşımı, rebalancing YOK, hata
    izolasyonu).
    """
    st.info(
        "⚠️ Bu bir yatırım tavsiyesi DEĞİLDİR — yalnızca backtest motorunun "
        "referans bir strateji (SMA Crossover) ile nasıl çalıştığını gösterir.",
        icon="ℹ️",
    )
    with st.form(key="backtest_form"):
        col1, col2 = st.columns(2)
        with col1:
            symbols_input = st.text_input(
                "Sembol(ler)", placeholder="Örn. THYAO veya THYAO, GARAN, AKBNK",
                help="Birden fazla sembol için virgülle ayırın — sermaye EŞİT olarak paylaştırılır.",
            )
            fast_window = st.number_input("Hızlı SMA Penceresi", min_value=2, value=10, step=1)
        with col2:
            initial_capital = st.number_input("Başlangıç Sermayesi (₺)", min_value=100.0, value=10000.0, step=100.0)
            slow_window = st.number_input("Yavaş SMA Penceresi", min_value=3, value=30, step=1)
        lookback_days = st.slider("Geriye Dönük Gün Sayısı", min_value=60, max_value=500, value=200)
        # DÜZELTME (bu turda eklendi): get_available_benchmark_codes()
        # ARTIK backtest'e de bağlandı — önceden yalnızca portföy
        # oluşturma formunda kullanılıyordu (bkz. Portfolio.KNOWN_BENCHMARKS
        # "3 AYRI yer" tutarsızlık düzeltmesi — bu 4. tüketici).
        benchmark_options = container.portfolio_service.get_available_benchmark_codes()
        index_labels = ["Yok"] + [f"{b['code']} ({b['name']})" for b in benchmark_options]
        selected_index_label = st.selectbox(
            "Endeks Karşılaştırması (opsiyonel)", options=index_labels,
            help="Stratejinizi, aynı sembolü alıp tutmanın YANI SIRA bir "
                 "endeksle (örn. BIST100) de karşılaştırın.",
        )
        submitted = st.form_submit_button("Backtest Çalıştır", width="stretch")

    if not submitted:
        return

    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    if not symbols:
        st.error("En az bir sembol girin.", icon="🛑")
        return

    if fast_window >= slow_window:
        st.error("Hızlı SMA penceresi, yavaş pencereden küçük olmalı.", icon="🛑")
        return

    index_benchmark_symbol = None
    if selected_index_label != "Yok":
        idx = index_labels.index(selected_index_label) - 1
        index_benchmark_symbol = benchmark_options[idx]["code"]

    if len(symbols) == 1:
        _run_single_symbol_backtest(
            container, symbols[0], fast_window, slow_window, lookback_days,
            initial_capital, index_benchmark_symbol,
        )
    else:
        _run_multi_symbol_backtest(container, symbols, fast_window, slow_window, lookback_days, initial_capital)


def _run_single_symbol_backtest(
    container: Container, symbol: str, fast_window: int, slow_window: int,
    lookback_days: int, initial_capital: float, index_benchmark_symbol: str | None = None,
) -> None:
    try:
        from datetime import date, timedelta
        from decimal import Decimal

        import pandas as pd

        from src.domain.strategies.sma_crossover_strategy import SMACrossoverStrategy

        with st.spinner("Backtest çalıştırılıyor (strateji + Buy & Hold karşılaştırması)…"):
            comparison = container.backtest_service.run_with_benchmark(
                symbol=symbol.strip().upper(),
                strategy=SMACrossoverStrategy(),
                start=date.today() - timedelta(days=lookback_days),
                end=date.today(),
                strategy_params={"fast_window": int(fast_window), "slow_window": int(slow_window)},
                initial_capital=Decimal(str(initial_capital)),
                index_benchmark_symbol=index_benchmark_symbol,
            )
    except Exception as exc:
        st.error(f"Backtest başarısız.\n\nSebep: {exc}", icon="🛑")
        return

    result = comparison.strategy_result
    benchmark = comparison.benchmark_result

    st.subheader("📊 Strateji Sonuçları (SMA Crossover)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Getiri", f"{float(result.total_return):.2%}")
    col2.metric("Sharpe Oranı", f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "N/A")
    col3.metric("Maks. Düşüş", f"{result.max_drawdown:.2%}" if result.max_drawdown is not None else "N/A")

    col4, col5 = st.columns(2)
    col4.metric("Toplam İşlem", result.total_trades)
    col5.metric("Nihai Değer", f"{float(result.final_value):,.2f} ₺")

    st.divider()
    st.subheader("⚖️ Buy & Hold ile Karşılaştırma")
    st.caption(
        "Standart backtesting pratiği: bir stratejinin GERÇEK değeri, "
        "'basitçe alıp hiç satmasaydım ne olurdu' referansını komisyon "
        "maliyetini karşılayacak kadar YENİP YENEMEDİĞİNDE ölçülür."
    )
    diff = float(result.total_return) - float(benchmark.total_return)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Strateji Getirisi", f"{float(result.total_return):.2%}")
    col_b.metric("Buy & Hold Getirisi", f"{float(benchmark.total_return):.2%}")
    col_c.metric("Fark", f"{diff:+.2%}", delta=f"{diff:+.2%}")

    if diff <= 0:
        st.warning(
            "Bu strateji, komisyon dahil basit Buy & Hold'u YENEMEDİ — "
            "bu, aktif stratejinin bu dönemde/parametrelerle GERÇEK bir "
            "değer katmadığının bir işareti olabilir.",
            icon="⚠️",
        )

    chart_df = pd.DataFrame({
        "Strateji": result.portfolio_value_series,
        "Buy & Hold": benchmark.portfolio_value_series,
    })

    if comparison.index_benchmark_result is not None:
        index_result = comparison.index_benchmark_result
        chart_df[f"Endeks ({comparison.index_benchmark_symbol})"] = index_result.portfolio_value_series

        st.divider()
        st.subheader(f"📈 Endeks Karşılaştırması ({comparison.index_benchmark_symbol})")
        index_diff = float(result.total_return) - float(index_result.total_return)
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Strateji Getirisi", f"{float(result.total_return):.2%}")
        col_y.metric(f"{comparison.index_benchmark_symbol} Getirisi", f"{float(index_result.total_return):.2%}")
        col_z.metric("Fark", f"{index_diff:+.2%}", delta=f"{index_diff:+.2%}")

    st.line_chart(chart_df)

    if container.ai_insight_service.is_configured:
        if st.button("🤖 AI ile Sonucu Yorumla", key="ai_interpret_backtest"):
            with st.spinner("Yorumlanıyor…"):
                try:
                    interpretation = container.ai_insight_service.interpret_backtest_result(comparison)
                    st.markdown(interpretation)
                except Exception as exc:
                    st.error(f"Yorum oluşturulamadı.\n\nSebep: {exc}", icon="🛑")


def _run_multi_symbol_backtest(
    container: Container, symbols: list[str], fast_window: int, slow_window: int,
    lookback_days: int, initial_capital: float,
) -> None:
    """
    DÜZELTME (bu turda eklendi): BacktestService.run_multi_symbol()'ın
    UI'ı — bkz. o metodun mimari karar gerekçesi (eşit sermaye
    paylaşımı, rebalancing YOK, hata izolasyonu).
    """
    try:
        from datetime import date, timedelta
        from decimal import Decimal

        from src.domain.strategies.sma_crossover_strategy import SMACrossoverStrategy

        with st.spinner(f"{len(symbols)} sembol için backtest çalıştırılıyor…"):
            result = container.backtest_service.run_multi_symbol(
                symbols=symbols, strategy=SMACrossoverStrategy(),
                start=date.today() - timedelta(days=lookback_days), end=date.today(),
                strategy_params={"fast_window": int(fast_window), "slow_window": int(slow_window)},
                initial_capital=Decimal(str(initial_capital)),
            )
    except Exception as exc:
        st.error(f"Backtest başarısız.\n\nSebep: {exc}", icon="🛑")
        return

    if result.failed_symbols:
        st.warning(
            f"Şu semboller için veri çekilemedi, atlandı: {', '.join(result.failed_symbols)}",
            icon="⚠️",
        )

    st.subheader(f"📊 Çoklu-Sembol Sonuçları ({len(result.symbol_results)} sembol, eşit sermaye paylaşımı)")
    col1, col2 = st.columns(2)
    col1.metric("Toplam Getiri", f"{float(result.combined_total_return):.2%}")
    col2.metric("Nihai Toplam Değer", f"{float(result.combined_final_value):,.2f} ₺")

    st.caption("Sembol bazında dağılım:")
    for symbol, symbol_result in result.symbol_results.items():
        st.write(
            f"**{symbol}**: {float(symbol_result.total_return):.2%} getiri, "
            f"{symbol_result.total_trades} işlem, "
            f"başlangıç sermayesi {float(symbol_result.initial_capital):,.2f} ₺"
        )

    st.caption("Birleştirilmiş toplam değer serisi:")
    st.line_chart(result.combined_equity_curve)


def render_app() -> None:
    st.title(f"📈 {_PAGE_TITLE}")
    st.caption("Borsa İstanbul hisseleri — gerçek zamanlı teknik analiz ve portföy yönetimi.")

    container = _get_cached_container()
    _init_session_state()
    _render_scheduler_sidebar(container)

    tab_analysis, tab_portfolio, tab_watchlist, tab_backtest = st.tabs(
        ["🔍 Tekil Hisse Analizi", "💼 Portföyüm", "⭐ İzleme Listem", "📊 Backtest"]
    )

    with tab_analysis:
        render_analysis_tab(container)

    with tab_portfolio:
        render_portfolio_tab(container)

    with tab_watchlist:
        render_watchlist_tab(container)

    with tab_backtest:
        render_backtest_tab(container)


def main() -> None:
    configure_page()
    render_app()


if __name__ == "__main__":
    main()
