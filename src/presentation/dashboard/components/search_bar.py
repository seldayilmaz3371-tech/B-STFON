"""
Sembol ve zaman aralığı arama bileşeni.

Katman izolasyonu: Bu modül yalnızca Streamlit'e ve stdlib'e bağımlıdır.
src/infrastructure veya src/domain altından hiçbir şey import etmez.
Desteklenen timeframe listesi aşağıda sabit olarak tanımlıdır; bu liste
src/infrastructure/data_providers/yfinance_adapter.py içindeki
_TIMEFRAME_TO_YF_INTERVAL sözlüğünün anahtarlarıyla birebir eşleşecek
şekilde elle senkronize tutulur (presentation katmanı o dosyayı import
edemeyeceği için tek pratik seçenek budur). Adapter'a yeni bir timeframe
eklenirse buradaki SUPPORTED_TIMEFRAMES listesi de güncellenmelidir.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

# Adapter'ın desteklediği timeframe'lerle senkron sabit liste.
# (label, value) çiftleri: label kullanıcıya gösterilir, value servise gönderilir.
SUPPORTED_TIMEFRAMES: list[tuple[str, str]] = [
    ("1 Dakika", "1m"),
    ("5 Dakika", "5m"),
    ("15 Dakika", "15m"),
    ("30 Dakika", "30m"),
    ("1 Saat", "1h"),
    ("4 Saat", "4h"),
    ("Günlük", "1d"),
    ("Haftalık", "1wk"),
    ("Aylık", "1mo"),
]

DEFAULT_TIMEFRAME_INDEX = 6  # "Günlük" (1d) — en yaygın kullanım

# Sembol giriş kısıtları (yalnızca UI seviyesinde temel format kontrolü;
# nihai geçerlilik denetimi Infrastructure/Provider katmanındadır).
_MIN_SYMBOL_LENGTH = 2
_MAX_SYMBOL_LENGTH = 10


@dataclass(frozen=True)
class SearchQuery:
    """Kullanıcının arama formundan girdiği değerlerin sabit (immutable) taşıyıcısı."""

    symbol: str
    timeframe: str
    submitted: bool


def render_search_bar(default_symbol: str = "") -> SearchQuery:
    """
    Sembol ve zaman aralığı seçim formunu çizer.

    Args:
        default_symbol: Formun sembol alanına önceden doldurulacak değer
                         (örn. session_state'teki son aranan sembol).

    Returns:
        SearchQuery: Kullanıcının girdiği (henüz doğrulanmamış) ham değerler
                     ve formun bu çalıştırmada gönderilip gönderilmediği
                     bilgisi (submitted).

    Not:
        Sembol normalizasyonu (büyük harfe çevirme, boşluk temizleme) burada
        yapılır — ancak sembolün gerçekten BIST'te var olup olmadığının
        doğrulanması bu bileşenin sorumluluğunda DEĞİLDİR; bu, servis
        katmanından dönen hatayla (MarketDataServiceError) anlaşılır.
    """
    with st.form(key="search_form", border=True):
        col_symbol, col_timeframe, col_button = st.columns([3, 2, 1], vertical_alignment="bottom")

        with col_symbol:
            raw_symbol = st.text_input(
                label="Hisse Sembolü",
                value=default_symbol,
                placeholder="Örn. THYAO, GARAN, ASELS",
                max_chars=_MAX_SYMBOL_LENGTH,
                help="BIST hisse senedi sembolünü girin (.IS son eki otomatik eklenir).",
            )

        with col_timeframe:
            labels = [label for label, _ in SUPPORTED_TIMEFRAMES]
            selected_label = st.selectbox(
                label="Zaman Aralığı",
                options=labels,
                index=DEFAULT_TIMEFRAME_INDEX,
            )

        with col_button:
            submitted = st.form_submit_button(label="Analiz Et", width="stretch")

    timeframe_value = dict(SUPPORTED_TIMEFRAMES)[selected_label]
    normalized_symbol = raw_symbol.strip().upper()

    return SearchQuery(
        symbol=normalized_symbol,
        timeframe=timeframe_value,
        submitted=submitted,
    )


def validate_symbol_format(symbol: str) -> str | None:
    """
    Sembolün temel format kurallarına uyup uymadığını kontrol eder.

    Bu yalnızca bariz kullanıcı hatalarını (boş alan, çok kısa/uzun girdi,
    rakam/özel karakter içeren sembol) erken yakalamak içindir — gerçek
    sembol geçerliliği servis katmanından gelen hata ile belirlenir.

    Args:
        symbol: render_search_bar()'dan dönen normalize edilmiş sembol.

    Returns:
        str | None: Geçersizse kullanıcıya gösterilecek hata mesajı,
                     geçerliyse None.
    """
    if not symbol:
        return "Lütfen bir hisse sembolü girin."
    if len(symbol) < _MIN_SYMBOL_LENGTH:
        return f"Sembol en az {_MIN_SYMBOL_LENGTH} karakter olmalıdır."
    if not symbol.replace(".", "").isalnum():
        return "Sembol yalnızca harf ve rakam içerebilir."
    return None
