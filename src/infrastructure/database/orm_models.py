"""
SQLAlchemy Core Table tanımları (2.0-style, ORM declarative DEĞİL).

Mimari karar hatırlatması (bu proje boyunca gerekçelendirildi):
  ORM'in lazy-loading/ilişki karmaşıklığı finansal doğrulukta
  öngörülemezlik riski taşıyor. Core ile domain modelleri
  SQLAlchemy'den tamamen bağımsız kalır — bu dosya yalnızca şema
  tanımını taşır, hiçbir domain sınıfı buraya bağımlı değil (bağımlılık
  tersine akış: repository bu Table'ları kullanır, domain hiçbir
  şekilde bunları görmez).

KAPSAM KARARI — MVP alt kümesi (BİLİNÇLİ EKSİLTME, unutkanlık DEĞİL):
  BIST_TEFAS_Master_Design_Document.md Bölüm'ünde tanımlı canonical
  `transactions` DDL'i 27 kolon taşıyor (commission, commission_vat,
  bsmv, stamp_duty, isin, checksum, fx_rate, settlement_status,
  is_reversal, reversal_of, corporate_action_id, created_by, notes,
  source, transaction_currency, portfolio_currency, settlement_date...).

  Bu tabloyu şu an yalnızca CostBasisCalculator'ın tükettiği alt
  kümeyle oluşturuyorum:
    id, portfolio_id, symbol, symbol_type, transaction_type,
    quantity, price, net_amount, split_ratio, trade_date,
    is_active, created_at

  Gerekçe: Henüz hiçbir kod komisyon/BSMV/damga vergisi/reversal
  mantığını TÜKETMİYOR (portfolio_service.py bunları okumuyor).
  Şimdiden 27 kolonluk bir tablo + ORM mapping yazmak, kullanılmayan
  alanlarla teknik yüzey alanını büyütür ve YANLIŞ VARSAYıMLA
  (örn. commission'ın nasıl hesaplanacağı netleşmeden bir DEFAULT
  değer/format seçmek) dolu bir şema üretme riski taşır.

  DÜZELTME (bu turda): is_reversal/reversal_of/reversal_reason alanları
  YUKARIDA "ertelendi" denen listedeydi — TransactionService.
  reverse_transaction() için gerçek bir tüketici ortaya çıkınca eklendi.
  Geri kalan alanlar (commission, bsmv, stamp_duty, isin, checksum,
  fx_rate, settlement_status, corporate_action_id, created_by, notes,
  source, transaction_currency, portfolio_currency, settlement_date)
  HÂLÂ ERTELENMİŞ DURUMDA — henüz tüketicileri yok.

  RİSK (açıkça işaretleniyor): İleride commission/reversal alanları
  eklendiğinde Alembic migration gerekecek — bu, "yeniden yazım"
  değil "ekleme" (ALTER TABLE ADD COLUMN) olacağı için düşük maliyetli,
  ama sıfır değildir. Kabul edilmiş risk.

Decimal storage stratejisi: TEXT (tasarım belgesiyle birebir — "Decimal
saklama stratejisi (TEXT) kararlaştırılmış"). SQLite'ta REAL/FLOAT
kullanmak hassasiyet kaybına yol açar; TEXT + Decimal(str(x)) sınırda
dönüşüm, hassasiyeti korur. Repository katmanı bu dönüşümü yapar.
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
)

metadata = MetaData()

# ── portfolios ──────────────────────────────────────────────────────────────

portfolios_table = Table(
    "portfolios",
    metadata,
    Column("id", String, primary_key=True),  # UUID, str
    Column("name", String, nullable=False),
    Column("description", String, nullable=True),
    Column("currency", String, nullable=False, server_default="TRY"),
    Column("cost_method", String, nullable=False, server_default="WAVG"),
    Column("inception_date", String, nullable=False),  # ISO 8601 date
    Column("benchmark_code", String, nullable=True),
    Column("is_active", Integer, nullable=False, server_default="1"),
    Column("created_at", String, nullable=False),  # ISO 8601 datetime
    Column("updated_at", String, nullable=False),
    Column("tags", String, nullable=True),        # JSON array (TEXT), örn. '["emeklilik"]'
    Column("metadata", String, nullable=True),     # JSON object (TEXT)
    CheckConstraint("currency IN ('TRY', 'USD', 'EUR')", name="chk_portfolio_currency"),
    CheckConstraint("cost_method IN ('WAVG', 'FIFO', 'LIFO')", name="chk_cost_method"),
    CheckConstraint("is_active IN (0, 1)", name="chk_portfolio_is_active"),
    UniqueConstraint("name", name="uq_portfolio_name"),
)

# ── transactions (MVP alt kümesi — yukarıdaki modül docstring'ine bkz.) ────────

transactions_table = Table(
    "transactions",
    metadata,
    Column("id", String, primary_key=True),  # UUID, str
    Column("portfolio_id", String, ForeignKey("portfolios.id"), nullable=False),
    Column("symbol", String, nullable=False),
    Column("symbol_type", String, nullable=False),
    Column("transaction_type", String, nullable=False),
    Column("quantity", String, nullable=False),      # Decimal as TEXT
    Column("price", String, nullable=False),         # Decimal as TEXT
    Column("net_amount", String, nullable=True),      # DIVIDEND için zorunlu
    Column("split_ratio", String, nullable=True),      # SPLIT için zorunlu
    Column("trade_date", String, nullable=False),      # YYYY-MM-DD
    Column("is_active", Integer, nullable=False, server_default="1"),
    Column("created_at", String, nullable=False),
    Column("is_reversal", Integer, nullable=False, server_default="0"),
    Column("reversal_of", String, ForeignKey("transactions.id"), nullable=True),
    Column("reversal_reason", String, nullable=True),
    CheckConstraint(
        "symbol_type IN ('BIST_STOCK', 'TEFAS_FUND', 'BIST_ETF', 'BOND', 'CASH', 'OTHER')",
        name="chk_symbol_type",
    ),
    CheckConstraint(
        "transaction_type IN ("
        "'BUY','SELL','DIVIDEND','BONUS_SHARE','RIGHTS_USED','RIGHTS_SOLD',"
        "'SPLIT','REVERSE_SPLIT','MERGER','DEPOSIT','WITHDRAWAL','FEE','TAX'"
        ")",
        name="chk_transaction_type",
    ),
    CheckConstraint("is_active IN (0, 1)", name="chk_tx_is_active"),
)

# ── price_series ────────────────────────────────────────────────────────────

price_series_table = Table(
    "price_series",
    metadata,
    Column("id", String, primary_key=True),
    Column("symbol", String, nullable=False),
    Column("symbol_type", String, nullable=False),
    Column("date", String, nullable=False),  # YYYY-MM-DD
    Column("open_price", String, nullable=True),
    Column("high_price", String, nullable=True),
    Column("low_price", String, nullable=True),
    Column("close_price", String, nullable=False),
    Column("adjusted_close", String, nullable=True),
    Column("volume", String, nullable=True),
    Column("source", String, nullable=False),
    Column("is_holiday", Integer, nullable=False, server_default="0"),
    Column("created_at", String, nullable=False),
    UniqueConstraint("symbol", "date", name="uq_price_symbol_date"),
    CheckConstraint(
        "symbol_type IN ('BIST_STOCK', 'TEFAS_FUND', 'BIST_ETF', 'BENCHMARK')",
        name="chk_price_symbol_type",
    ),
    CheckConstraint(
        "source IN ('yfinance', 'tefas', 'isyatirim', 'bist', 'manual', 'mock')",
        name="chk_price_source",
    ),
)

# ── cash_ledger_entries ─────────────────────────────────────────────────────

cash_ledger_entries_table = Table(
    "cash_ledger_entries",
    metadata,
    Column("id", String, primary_key=True),
    Column("portfolio_id", String, ForeignKey("portfolios.id"), nullable=False),
    Column("transaction_id", String, ForeignKey("transactions.id"), nullable=True),
    Column("entry_type", String, nullable=False),
    Column("amount", String, nullable=False),  # Her zaman pozitif (DDL CHECK)
    Column("currency", String, nullable=False, server_default="TRY"),
    Column("entry_date", String, nullable=False),
    Column("description", String, nullable=False),
    Column("balance_after", String, nullable=False),
    Column("created_at", String, nullable=False),
    CheckConstraint("entry_type IN ('CREDIT', 'DEBIT')", name="chk_entry_type"),
    CheckConstraint("CAST(amount AS REAL) > 0", name="chk_amount_positive"),
)

# ── risk_snapshots ──────────────────────────────────────────────────────────
#
# Float kolonlar (Decimal-as-TEXT DEĞİL) — BİLİNÇLİ bir sapma. Bu
# proje boyunca "finansal muhasebe = Decimal, istatistiksel tahmin =
# float64 yeterli" ayrımı defalarca gerekçelendirildi (RiskCalculator/
# ReturnCalculator modül docstring'lerinde). Risk metrikleri (Sharpe,
# VaR, beta) BİRER TAHMİNDİR, muhasebe kaydı değil — Decimal hassasiyeti
# burada anlamsız bir maliyet olurdu.

risk_snapshots_table = Table(
    "risk_snapshots",
    metadata,
    Column("id", String, primary_key=True),
    Column("portfolio_id", String, ForeignKey("portfolios.id"), nullable=False),
    Column("computed_at", String, nullable=False),
    Column("as_of_date", String, nullable=False),
    Column("lookback_days", Integer, nullable=False),
    Column("portfolio_volatility", Float, nullable=True),
    Column("sharpe_ratio", Float, nullable=True),
    Column("sortino_ratio", Float, nullable=True),
    Column("calmar_ratio", Float, nullable=True),
    Column("max_drawdown", Float, nullable=True),
    Column("max_drawdown_start", String, nullable=True),
    Column("max_drawdown_end", String, nullable=True),
    Column("current_drawdown", Float, nullable=True),
    Column("var_95", Float, nullable=True),
    Column("var_99", Float, nullable=True),
    Column("cvar_95", Float, nullable=True),
    Column("cvar_99", Float, nullable=True),
    Column("var_method", String, nullable=False, server_default="HISTORICAL"),
    Column("beta", Float, nullable=True),
    Column("alpha", Float, nullable=True),
    Column("r_squared", Float, nullable=True),
    Column("information_ratio", Float, nullable=True),
    Column("tracking_error", Float, nullable=True),
    Column("herfindahl_index", Float, nullable=True),
    Column("top5_concentration", Float, nullable=True),
    Column("risk_free_rate", Float, nullable=False),
    Column("benchmark_code", String, nullable=True),
    Column("is_stale", Integer, nullable=False, server_default="0"),
    CheckConstraint(
        "var_method IN ('HISTORICAL', 'PARAMETRIC', 'MONTECARLO')",
        name="chk_snapshot_var_method",
    ),
    CheckConstraint("is_stale IN (0, 1)", name="chk_snapshot_is_stale"),
)

# ── watchlists / watchlist_items ─────────────────────────────────────────────

watchlists_table = Table(
    "watchlists",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("portfolio_id", String, ForeignKey("portfolios.id"), nullable=True),
    Column("description", String, nullable=True),
    Column("is_active", Integer, nullable=False, server_default="1"),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

watchlist_items_table = Table(
    "watchlist_items",
    metadata,
    Column("id", String, primary_key=True),
    Column("watchlist_id", String, ForeignKey("watchlists.id"), nullable=False),
    Column("symbol", String, nullable=False),
    Column("symbol_type", String, nullable=False),
    Column("alert_price_low", String, nullable=True),
    Column("alert_price_high", String, nullable=True),
    Column("alert_pct_change", String, nullable=True),
    Column("notes", String, nullable=True),
    Column("added_at", String, nullable=False),
    UniqueConstraint("watchlist_id", "symbol", name="uq_watchlist_symbol"),
)

# ── funds ─────────────────────────────────────────────────────────────────

funds_table = Table(
    "funds",
    metadata,
    Column("fund_code", String, primary_key=True),
    Column("fund_name", String, nullable=False),
    Column("fund_type", String, nullable=False),
    Column("umbrella_type", String, nullable=True),
    Column("founder", String, nullable=True),
    Column("currency", String, nullable=False, server_default="TRY"),
    Column("stock_pct", String, nullable=True),
    Column("bond_pct", String, nullable=True),
    Column("repo_pct", String, nullable=True),
    Column("foreign_stock_pct", String, nullable=True),
    Column("gold_pct", String, nullable=True),
    Column("other_pct", String, nullable=True),
    Column("allocation_date", String, nullable=True),
    Column("last_nav", String, nullable=True),
    Column("last_nav_date", String, nullable=True),
    Column("ytd_return", String, nullable=True),
    Column("management_fee", String, nullable=True),
    Column("is_active", Integer, nullable=False, server_default="1"),
    Column("last_updated", String, nullable=False),
    Column("created_at", String, nullable=False),
    CheckConstraint(
        "fund_type IN ('YAT', 'EMK', 'BYF', 'DIGER')", name="chk_fund_type",
    ),
)

# ── corporate_actions ────────────────────────────────────────────────────

corporate_actions_table = Table(
    "corporate_actions",
    metadata,
    Column("id", String, primary_key=True),
    Column("symbol", String, nullable=False),
    Column("action_type", String, nullable=False),
    Column("announcement_date", String, nullable=True),
    Column("ex_date", String, nullable=False),
    Column("record_date", String, nullable=True),
    Column("payment_date", String, nullable=True),
    Column("action_data", String, nullable=False),  # JSON, TEXT olarak saklanıyor
    Column("is_confirmed", Integer, nullable=False, server_default="0"),
    Column("is_applied", Integer, nullable=False, server_default="0"),
    Column("source", String, nullable=False, server_default="manual"),
    Column("notes", String, nullable=True),
    Column("raw_data", String, nullable=True),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    CheckConstraint(
        "action_type IN ('DIVIDEND', 'BONUS_SHARE', 'RIGHTS_ISSUE', "
        "'SPLIT', 'REVERSE_SPLIT', 'MERGER', 'SPIN_OFF', 'DELISTING')",
        name="chk_action_type",
    ),
)
