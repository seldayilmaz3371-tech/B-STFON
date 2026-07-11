"""
İnteraktif mum grafiği (candlestick) bileşeni.

Katman izolasyonu: Bu modül yalnızca Streamlit, Plotly ve pandas'a
bağımlıdır. Girdi olarak src/services/market_data_service.py içindeki
MarketAnalysisResult DTO'sunu alır; hiçbir infrastructure veya domain
modülünü import etmez ve hiçbir hesaplama yapmaz — yalnızca DTO'da
zaten hesaplanmış olan ohlcv/rsi serilerini görselleştirir.

Görsel kimlik notu:
  Renk paleti, jenerik Plotly varsayılanları (kırmızı/yeşil parlak ton)
  yerine BIST yatırımcısının alışık olduğu, koyu zemin üzerinde net
  okunan bir "işlem terminali" paletine göre seçildi: yükseliş barları
  için zümrüt yeşili (#1FBF75), düşüş barları için kiremit kırmızısı
  (#E8553F) — ikisi de eşit görsel ağırlıkta, biri diğerini baskılamıyor.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.services.market_data_service import MarketAnalysisResult

# ── Görsel Kimlik Sabitleri ───────────────────────────────────────────────────
_COLOR_BULLISH = "#1FBF75"   # Zümrüt yeşili — yükseliş barı
_COLOR_BEARISH = "#E8553F"   # Kiremit kırmızısı — düşüş barı
_COLOR_VOLUME = "#5B6472"    # Nötr gri-mavi — hacim çubukları
_COLOR_RSI_LINE = "#6E8EFB"  # Yumuşak indigo — RSI çizgisi
_COLOR_RSI_OVERBOUGHT = "rgba(232, 85, 63, 0.12)"
_COLOR_RSI_OVERSOLD = "rgba(31, 191, 117, 0.12)"
_COLOR_GRID = "rgba(148, 158, 174, 0.15)"

_RSI_OVERBOUGHT_LEVEL = 70.0
_RSI_OVERSOLD_LEVEL = 30.0

# Alt panel yükseklik oranları: Fiyat %60, Hacim %15, RSI %25
_ROW_HEIGHTS = [0.60, 0.15, 0.25]


def render_technical_chart(result: MarketAnalysisResult) -> None:
    """
    OHLCV mum grafiğini, hacim çubuklarını ve RSI alt panelini tek bir
    interaktif Plotly figüründe çizer.

    Veri eksikliğine karşı savunma:
      ohlcv boşsa veya gerekli sütunlar (Open/High/Low/Close) eksikse,
      grafik çizilmez ve kullanıcıya bilgilendirici bir st.info gösterilir
      — bu bileşen hiçbir koşulda exception fırlatmaz (servis katmanı
      zaten boş veriyi MarketDataServiceError ile elemiş olmalı; bu
      kontrol yalnızca son bir savunma katmanıdır).

    Args:
        result: MarketDataService.get_market_analysis()'ten dönen DTO.
    """
    ohlcv = result.ohlcv
    required_columns = {"Open", "High", "Low", "Close"}

    if ohlcv.empty or not required_columns.issubset(ohlcv.columns):
        st.info("Grafik için yeterli fiyat verisi bulunamadı.")
        return

    figure = _build_figure(ohlcv=ohlcv, rsi=result.rsi, symbol=result.symbol, timeframe=result.timeframe)
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def _build_figure(
    ohlcv: pd.DataFrame,
    rsi: pd.Series,
    symbol: str,
    timeframe: str,
) -> go.Figure:
    """Üç panelli (fiyat / hacim / RSI) birleşik Plotly figürünü oluşturur."""
    has_volume = "Volume" in ohlcv.columns and ohlcv["Volume"].notna().any()
    has_rsi = not rsi.empty and rsi.notna().any()

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=_ROW_HEIGHTS,
    )

    _add_candlestick_trace(figure, ohlcv, row=1)

    if has_volume:
        _add_volume_trace(figure, ohlcv, row=2)

    if has_rsi:
        _add_rsi_trace(figure, rsi, row=3)

    _apply_layout(figure, symbol=symbol, timeframe=timeframe)
    return figure


def _add_candlestick_trace(figure: go.Figure, ohlcv: pd.DataFrame, row: int) -> None:
    figure.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            increasing_line_color=_COLOR_BULLISH,
            increasing_fillcolor=_COLOR_BULLISH,
            decreasing_line_color=_COLOR_BEARISH,
            decreasing_fillcolor=_COLOR_BEARISH,
            name="Fiyat",
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def _add_volume_trace(figure: go.Figure, ohlcv: pd.DataFrame, row: int) -> None:
    figure.add_trace(
        go.Bar(
            x=ohlcv.index,
            y=ohlcv["Volume"],
            marker_color=_COLOR_VOLUME,
            marker_line_width=0,
            name="Hacim",
            showlegend=False,
            opacity=0.7,
        ),
        row=row,
        col=1,
    )
    figure.update_yaxes(title_text="Hacim", row=row, col=1, showgrid=False)


def _add_rsi_trace(figure: go.Figure, rsi: pd.Series, row: int) -> None:
    figure.add_trace(
        go.Scatter(
            x=rsi.index,
            y=rsi,
            mode="lines",
            line=dict(color=_COLOR_RSI_LINE, width=1.5),
            name="RSI (14)",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    # Aşırı alım / aşırı satım bölgelerini yatay referans çizgileriyle işaretle
    figure.add_hline(
        y=_RSI_OVERBOUGHT_LEVEL,
        line=dict(color=_COLOR_BEARISH, width=1, dash="dot"),
        row=row,
        col=1,
    )
    figure.add_hline(
        y=_RSI_OVERSOLD_LEVEL,
        line=dict(color=_COLOR_BULLISH, width=1, dash="dot"),
        row=row,
        col=1,
    )
    figure.update_yaxes(
        title_text="RSI",
        row=row,
        col=1,
        range=[0, 100],
        showgrid=False,
    )


def _apply_layout(figure: go.Figure, symbol: str, timeframe: str) -> None:
    figure.update_layout(
        title=dict(
            text=f"{symbol} · {timeframe}",
            x=0.0,
            xanchor="left",
            font=dict(size=16),
        ),
        height=640,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    figure.update_xaxes(showgrid=True, gridcolor=_COLOR_GRID)
    figure.update_yaxes(showgrid=True, gridcolor=_COLOR_GRID)

    # Hafta sonu/tatil boşluklarını gizle (yalnızca günlük ve üzeri
    # timeframe'lerde anlamlı; dakika bazlı barlarda rangebreaks
    # intraday boşlukları da gizleyebileceğinden uygulanmaz).
    if timeframe in {"1d", "1wk", "1mo"}:
        figure.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
        )
