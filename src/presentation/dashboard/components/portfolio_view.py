"""
Portföy durum görünümü bileşeni.

Katman izolasyonu: Yalnızca Streamlit, Pandas ve stdlib'e bağımlıdır.
Girdi olarak src/services/portfolio_service.py'deki PortfolioSummaryDTO'yu
tüketir; hiçbir infrastructure veya domain modülü import etmez.

Görsel tasarım kararları:
  Renk kodlaması: kâr = yeşil (#1FBF75), zarar = kırmızı (#E8553F) —
  technical_chart.py'deki palette ile tutarlı. PnL sütununa Pandas Styler
  ile satır bazlı renk uygulanır; bu Streamlit'in st.dataframe'indeki
  native styling mekanizmasıdır (ek JS/CSS gerektirmez).

  "—" gösterimi: None değerler kullanıcıya boş/hatalı rakam yerine
  "Fiyat alınamadı" anlamına gelen "—" olarak gösterilir. Bu, ağ
  kesintisinde kullanıcının yanlış bir sıfır görmesini önler.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import streamlit as st

from src.services.portfolio_service import PortfolioSummaryDTO

_COLOR_PROFIT = "#1FBF75"
_COLOR_LOSS   = "#E8553F"
_COLOR_NEUTRAL = "color: inherit"
# DÜZELTME (bu turda, gerçek bir AppTest senaryosuyla bulundu): Önceki
# değer yalnızca "inherit" idi — bu bir CSS DEĞERİ, DEKLARASYONU DEĞİL.
# pandas Styler.map(), dönen string'i "attr: val" formatında bekliyor;
# bare "inherit" verilince ValueError fırlatıyordu. Bu, yalnızca en az
# bir hücre None/NaN olduğunda (örn. YENİ eklenmiş, henüz fiyatı
# gelmemiş bir pozisyon) tetikleniyordu — GD-001 senaryosunda (tüm
# hücreler dolu) hiç görünmüyordu, bu yüzden daha önce fark edilmemişti.


def render_portfolio(summary: PortfolioSummaryDTO) -> None:
    """
    Portföy özetini üç bölümde çizer:
      1. Üst özet metrikleri (toplam maliyet, güncel değer, K/Z)
      2. Pozisyon tablosu (renk kodlu K/Z ile)
      3. Stale sembol uyarısı (varsa)

    Args:
        summary: PortfolioService.get_portfolio_status() çıktısı.
    """
    if summary.position_count == 0:
        st.info(
            "Portföyde henüz açık pozisyon bulunmuyor. "
            "İşlem eklemek için 'İşlem Girişi' sekmesini kullanın.",
            icon="📭",
        )
        return

    _render_summary_metrics(summary)
    st.divider()
    _render_position_table(summary)

    if summary.stale_symbols:
        _render_stale_warning(summary.stale_symbols)


def _render_summary_metrics(summary: PortfolioSummaryDTO) -> None:
    """Üst satır: toplam maliyet / güncel değer / unrealized PnL / realized PnL."""
    col1, col2, col3, col4 = st.columns(4)

    total_cost = float(summary.total_cost_basis)
    total_value = float(summary.total_current_value)
    total_upnl = float(summary.total_unrealized_pnl)
    total_rpnl = float(summary.total_realized_pnl)

    # Toplam değer değişim delta'sı (stale semboller hariç pozisyonlar üzerinden)
    value_delta = total_value - total_cost if total_cost > 0 else None

    with col1:
        st.metric(
            label="Toplam Maliyet",
            value=f"{total_cost:,.2f} ₺",
            help="Açık pozisyonların toplam alış maliyeti.",
        )
    with col2:
        st.metric(
            label="Güncel Değer",
            value=f"{total_value:,.2f} ₺" if total_value > 0 else "—",
            delta=f"{value_delta:+,.2f} ₺" if value_delta is not None else None,
            help="Fiyatı alınabilen pozisyonların güncel piyasa değeri.",
        )
    with col3:
        st.metric(
            label="Gerçekleşmemiş K/Z",
            value=_fmt_pnl(summary.total_unrealized_pnl),
            delta=f"{total_upnl:+,.2f} ₺" if total_value > 0 else None,
            help="Açık pozisyonların mevcut K/Z (fiyatı alınan pozisyonlar).",
        )
    with col4:
        st.metric(
            label="Gerçekleşen K/Z",
            value=_fmt_pnl(summary.total_realized_pnl),
            help="Kapatılmış pozisyonların toplam gerçekleşen K/Z.",
        )


def _render_position_table(summary: PortfolioSummaryDTO) -> None:
    """Pozisyon tablosunu K/Z renk kodlamasıyla çizer."""
    st.subheader(f"Pozisyonlar ({summary.position_count})")

    rows = []
    for p in summary.positions:
        rows.append({
            "Sembol":          p.symbol,
            "Miktar":          float(p.total_quantity),
            "Ort. Maliyet ₺":  float(p.average_cost),
            "Maliyet Bazı ₺":  float(p.total_cost_basis),
            "Güncel Fiyat ₺":  float(p.current_price) if p.current_price is not None else None,
            "Güncel Değer ₺":  float(p.current_value) if p.current_value is not None else None,
            "G.Gelen K/Z ₺":   float(p.unrealized_pnl) if p.unrealized_pnl is not None else None,
            "K/Z %":           float(p.pnl_percentage) if p.pnl_percentage is not None else None,
            "Gerç. K/Z ₺":     float(p.realized_pnl),
        })

    df = pd.DataFrame(rows)

    styled = (
        df.style
        # pandas 3.x'te Styler.applymap() TAMAMEN KALDIRILDI (2.1'den beri
        # deprecated idi). Bu, GERÇEK bir çökme olarak AppTest ile tespit
        # edildi — portföy sekmesi, en az 1 pozisyon olduğunda %100 çöküyordu
        # (AttributeError: 'Styler' object has no attribute 'applymap').
        # .map() birebir aynı elementwise semantiğe sahip drop-in yerine
        # geçen metod (Styler.apply() DEĞİL — o axis-wise çalışır, farklı
        # bir semantik taşır ve burada YANLIŞ olurdu).
        .map(_color_pnl, subset=["G.Gelen K/Z ₺", "K/Z %", "Gerç. K/Z ₺"])
        .format({
            "Miktar":          "{:,.0f}",
            "Ort. Maliyet ₺":  "{:,.2f}",
            "Maliyet Bazı ₺":  "{:,.2f}",
            "Güncel Fiyat ₺":  lambda v: f"{v:,.2f}" if v is not None and not pd.isna(v) else "—",
            "Güncel Değer ₺":  lambda v: f"{v:,.2f}" if v is not None and not pd.isna(v) else "—",
            "G.Gelen K/Z ₺":   lambda v: f"{v:+,.2f}" if v is not None and not pd.isna(v) else "—",
            "K/Z %":           lambda v: f"{v:+.2f}%" if v is not None and not pd.isna(v) else "—",
            "Gerç. K/Z ₺":     "{:+,.2f}",
        })
        .set_properties(**{"text-align": "right"}, subset=df.columns[1:])
    )

    st.dataframe(styled, width="stretch", hide_index=True)


def _render_stale_warning(stale_symbols: list[str]) -> None:
    """Fiyatı güncellenemeyen semboller için uyarı bandı."""
    symbol_list = ", ".join(stale_symbols)
    st.warning(
        f"⚠️ Aşağıdaki semboller için anlık fiyat alınamadı; "
        f"K/Z değerleri gösterilemiyor: **{symbol_list}**\n\n"
        f"Olası sebepler: ağ bağlantısı, geçersiz sembol veya piyasa kapalı.",
        icon="⚠️",
    )


def _fmt_pnl(value: Decimal) -> str:
    """Decimal K/Z değerini işaretli, virgüllü string'e çevirir."""
    v = float(value)
    return f"{v:+,.2f} ₺"


def _color_pnl(val: float | None) -> str:
    """Pandas Styler için hücre rengi: pozitif=yeşil, negatif=kırmızı, None=normal."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return _COLOR_NEUTRAL
    if val > 0:
        return f"color: {_COLOR_PROFIT}; font-weight: bold"
    if val < 0:
        return f"color: {_COLOR_LOSS}; font-weight: bold"
    return _COLOR_NEUTRAL
