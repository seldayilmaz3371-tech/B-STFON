"""
Teknik analiz metrik kartları bileşeni.

Katman izolasyonu: Bu modül yalnızca Streamlit'e bağımlıdır ve girdi
olarak src/services/market_data_service.py içindeki MarketAnalysisResult
DTO'sunu alır. Hiçbir infrastructure veya domain modülünü import etmez.

Gösterilen metrikler (DTO'da mevcut alanlara sadık):
  - latest_close  : Son kapanış fiyatı (₺)
  - latest_rsi    : RSI (14) — aşırı alım/satım yorumuyla
  - latest_rvol   : Bağıl Hacim — ortalamaya kıyasla kaç kat
  - ATR (14)      : result.atr serisinin son değeri — DTO'da hazır
                    latest_atr property'si olmadığından seri ucundan okunur

Tasarım kararı (kullanıcı onayı ile, 2026-06-21):
  DTO'da latest_obv alanı YOKTUR (servis yalnızca RSI/RVOL/ATR hesaplar).
  Bu bileşen DTO'nun sunduğu gerçeğe adapte olur; backend'i UI'a
  uydurmak Clean Architecture ihlali sayılır.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services.market_data_service import MarketAnalysisResult

# RSI yorumlama eşikleri (klasik teknik analiz konvansiyonu — Wilder RSI).
_RSI_OVERBOUGHT_THRESHOLD = 70.0
_RSI_OVERSOLD_THRESHOLD = 30.0

# RVOL yorumlama eşiği: ortalamanın belirgin üzerinde hacim.
_RVOL_HIGH_THRESHOLD = 1.5


def render_metrics(result: MarketAnalysisResult) -> None:
    """
    Son kapanış, RSI, RVOL ve ATR değerlerini dört kolonlu metrik kartı
    olarak çizer.

    Her metrik None (NaN/eksik veri) olabilir — bu durumda st.metric
    "—" gösterir ve uygulama çökmez (DTO'nun NaN-güvenli property'leri
    zaten None döndürür, bkz. MarketAnalysisResult.latest_rsi vb.).

    Args:
        result: MarketDataService.get_market_analysis()'ten dönen DTO.
    """
    col_close, col_rsi, col_rvol, col_atr = st.columns(4)

    with col_close:
        _render_close_metric(result)

    with col_rsi:
        _render_rsi_metric(result)

    with col_rvol:
        _render_rvol_metric(result)

    with col_atr:
        _render_atr_metric(result)

    st.caption(
        f"📊 {result.symbol} · {result.bar_count} bar · "
        f"Hesaplama zamanı: {result.generated_at:%Y-%m-%d %H:%M:%S} UTC"
    )


def _render_close_metric(result: MarketAnalysisResult) -> None:
    value = result.latest_close
    st.metric(
        label="Son Kapanış",
        value=f"{value:,.2f} ₺" if value is not None else "—",
    )


def _render_rsi_metric(result: MarketAnalysisResult) -> None:
    value = result.latest_rsi
    label_suffix = ""
    if value is not None:
        if value >= _RSI_OVERBOUGHT_THRESHOLD:
            label_suffix = " (Aşırı Alım)"
        elif value <= _RSI_OVERSOLD_THRESHOLD:
            label_suffix = " (Aşırı Satım)"

    st.metric(
        label=f"RSI (14){label_suffix}",
        value=f"{value:.1f}" if value is not None else "—",
    )


def _render_rvol_metric(result: MarketAnalysisResult) -> None:
    value = result.latest_rvol
    label_suffix = " (Yüksek Hacim)" if value is not None and value >= _RVOL_HIGH_THRESHOLD else ""

    st.metric(
        label=f"RVOL (Bağıl Hacim){label_suffix}",
        value=f"{value:.2f}x" if value is not None else "—",
        help="Hacmin son 20 barlık ortalamaya oranı. 1.0 = ortalama hacim.",
    )


def _render_atr_metric(result: MarketAnalysisResult) -> None:
    """
    Son ATR (Average True Range) değerini gösterir.

    DTO'da latest_atr adında bir hazır property bulunmadığından
    (yalnızca latest_close/latest_rsi/latest_rvol property'leri tanımlı),
    değer doğrudan result.atr serisinin son elemanından, aynı NaN-güvenli
    desenle (boş seri veya NaN ise None) okunur.
    """
    atr_series = result.atr
    value: float | None = None
    if not atr_series.empty and not pd.isna(atr_series.iloc[-1]):
        value = float(atr_series.iloc[-1])

    st.metric(
        label="ATR (14)",
        value=f"{value:,.2f} ₺" if value is not None else "—",
        help="Average True Range — son 14 barlık ortalama volatilite (fiyat birimi).",
    )
