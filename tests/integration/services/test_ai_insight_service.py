"""
AIInsightService testleri — Anthropic client SAHTE (mock) ile.

TEST EDİLEBİLİRLİK SINIRI (bkz. ai_insight_service.py modül docstring'i):
Bu sandbox'ta GERÇEK bir Anthropic API anahtarı YOK. Bu testler
GERÇEK bir API çağrısı YAPMIYOR — yalnızca (a) prompt inşasının doğru
olduğunu, (b) API anahtarı yokken zarif hata verdiğini, (c) API hatası
durumunda çökmediğini doğruluyor. GERÇEK API yanıt kalitesi kullanıcının
KENDİ anahtarıyla doğrulanmalı.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.domain.exceptions.domain_exceptions import BusinessRuleError
from src.infrastructure.database.connection import (
    create_db_engine, create_session_factory, initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.repositories.sqlite.cash_ledger_repository import SQLiteCashLedgerRepository
from src.infrastructure.repositories.sqlite.portfolio_repository import SQLitePortfolioRepository
from src.infrastructure.repositories.sqlite.transaction_repository import SQLiteTransactionRepository
from src.services.ai_insight_service import AIInsightService
from src.services.portfolio_service import PortfolioService

pytestmark = pytest.mark.integration


@pytest.fixture()
def env(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'ai_insight_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)

    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sf() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="AI Test Portföyü", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()

    portfolio_repo = SQLitePortfolioRepository(sf)
    tx_repo = SQLiteTransactionRepository(sf)
    portfolio_service = PortfolioService(
        transaction_repo=tx_repo, market_data_service=None, portfolio_repo=portfolio_repo,
    )
    yield pid, portfolio_service
    engine.dispose()


class FakeRiskService:
    """Gerçek RiskService'i taklit eden sahte — bu testler AI entegrasyonuna odaklanıyor, risk hesaplama DOĞRULUĞUNA değil (o zaten AYRI test edildi)."""

    def compute_risk_profile(self, portfolio_id):
        from src.domain.calculators.risk_calculator import DrawdownResult
        profile = MagicMock()
        profile.annualized_volatility = 0.25
        profile.sharpe_ratio = 1.2
        profile.sortino_ratio = 1.5
        profile.max_drawdown = MagicMock(max_drawdown=-0.15)
        profile.var_95 = -0.02
        return profile


class FailingRiskService:
    def compute_risk_profile(self, portfolio_id):
        from src.domain.exceptions.domain_exceptions import InsufficientDataError
        raise InsufficientDataError(required=30, available=5, metric="test")


# ── is_configured ─────────────────────────────────────────────────────────────

def test_is_configured_false_without_api_key(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key=None)
    assert service.is_configured is False


def test_is_configured_true_with_api_key(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")
    assert service.is_configured is True


# ── generate_portfolio_summary — API anahtarı yok ───────────────────────────

def test_generate_summary_without_api_key_raises_business_rule_error(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key=None)
    with pytest.raises(BusinessRuleError, match="API anahtarı"):
        service.generate_portfolio_summary(pid)


# ── generate_portfolio_summary — API çağrısı (MOCK) ─────────────────────────

def test_generate_summary_calls_anthropic_with_correct_model(env):
    """API'ye GERÇEKTEN gidilmiyor — client.messages.create çağrısının DOĞRU parametrelerle yapıldığı doğrulanıyor."""
    pid, portfolio_service = env
    service = AIInsightService(
        portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key",
        model="claude-sonnet-5", max_tokens=500,
    )

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Test özeti.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.generate_portfolio_summary(pid)

        assert result == "Test özeti."
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-5"
        assert call_kwargs["max_tokens"] == 500
        assert "AI Test Portföyü" in call_kwargs["messages"][0]["content"]


def test_generate_summary_prompt_contains_only_precalculated_numbers(env):
    """
    KRİTİK doğrulama: prompt'un İÇERİĞİ, ZATEN hesaplanmış sayıları
    (FakeRiskService'in döndürdüğü) içermeli — AI'ın YENİ bir sayı
    hesaplamasına gerek KALMAMALI.
    """
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Özet.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        service.generate_portfolio_summary(pid)

        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "1.2" in prompt or "%25.00" in prompt or "-15.00%" in prompt or "Sharpe" in prompt
        assert "YENİ bir sayı HESAPLAMA" in prompt  # AI'a AÇIK talimat VAR mı


def test_generate_summary_handles_missing_risk_data_gracefully(env):
    """Yeni portföy (risk verisi yok) — prompt İNŞA EDİLEBİLMELİ, çökmemeli."""
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FailingRiskService(), api_key="sk-ant-fake-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Özet.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.generate_portfolio_summary(pid)
        assert result == "Özet."
        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "henüz yeterli veri yok" in prompt


def test_generate_summary_api_error_raises_runtime_error_not_silent(env):
    """API çağrısı BAŞARISIZ olursa (ağ, geçersiz anahtar) — SESSİZCE yutulmamalı, açık RuntimeError."""
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("Ağ hatası simülasyonu")
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="AI isteği başarısız"):
            service.generate_portfolio_summary(pid)


# ── answer_portfolio_question (bu turda eklendi) ────────────────────────────

def test_answer_question_without_api_key_raises(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key=None)
    with pytest.raises(BusinessRuleError, match="API anahtarı"):
        service.answer_portfolio_question(pid, "Portföyüm nasıl gidiyor?")


def test_answer_question_empty_question_raises(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")
    with pytest.raises(BusinessRuleError, match="boş olamaz"):
        service.answer_portfolio_question(pid, "   ")


def test_answer_question_includes_question_and_position_data_in_prompt(env):
    """
    KRİTİK doğrulama: prompt hem KULLANICININ SORUSUNU hem de per-pozisyon
    verisini içermeli — yalnızca agrege risk verisiyle 'hangi pozisyonum
    en çok zarar ediyor' gibi sorular CEVAPLANAMAZDI.
    """
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Cevap.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.answer_portfolio_question(pid, "Toplam getirim ne kadar?")

        assert result == "Cevap."
        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "Toplam getirim ne kadar?" in prompt
        assert "UYDURMA CEVAP VERME" in prompt  # AI'a AÇIK talimat VAR mı


def test_answer_question_api_error_raises_runtime_error(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = ConnectionError("Ağ hatası simülasyonu")
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="AI isteği başarısız"):
            service.answer_portfolio_question(pid, "Bir soru")


# ── detect_anomalies (bu turda eklendi) ─────────────────────────────────────

class FakeRiskServiceWithSeries(FakeRiskService):
    def get_rolling_volatility(self, portfolio_id, lookback_days=100):
        import pandas as pd
        return pd.Series([0.20, 0.21, 0.19, 0.22, 0.45], index=pd.date_range("2024-01-01", periods=5))

    def get_drawdown_series(self, portfolio_id, lookback_days=100):
        import pandas as pd
        return pd.Series([0.0, -0.01, -0.02, -0.03, -0.15], index=pd.date_range("2024-01-01", periods=5))


def test_detect_anomalies_without_api_key_raises(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskServiceWithSeries(), api_key=None)
    with pytest.raises(BusinessRuleError):
        service.detect_anomalies(pid)


def test_detect_anomalies_includes_series_data_in_prompt(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskServiceWithSeries(), api_key="sk-ant-fake-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Analiz.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.detect_anomalies(pid)

        assert result == "Analiz."
        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "oynaklık" in prompt.lower()
        assert "YAPAY BİR ENDİŞE" in prompt  # AI'a "uydurma" YASAĞI VAR mı


def test_detect_anomalies_handles_missing_series_gracefully(env):
    """Yeni portföy (seri verisi yok) — çökmemeli, prompt yine İNŞA edilebilmeli."""
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FailingRiskService(), api_key="sk-ant-fake-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Analiz.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.detect_anomalies(pid)
        assert result == "Analiz."


# ── interpret_backtest_result (bu turda eklendi) ────────────────────────────

def test_interpret_backtest_result_forbids_new_suggestions_in_prompt(env):
    """
    KRİTİK doğrulama (kullanıcı kararı): prompt AÇIKÇA 'yeni strateji
    ÖNERME' yasağı İÇERMELİ — bu, yatırım tavsiyesi sınırını koruyan
    özellikle KATI bir kısıtlama.
    """
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")

    comparison = MagicMock()
    comparison.strategy_result.total_return = Decimal("0.15")
    comparison.strategy_result.sharpe_ratio = 1.1
    comparison.strategy_result.max_drawdown = -0.10
    comparison.strategy_result.total_trades = 5
    comparison.benchmark_result.total_return = Decimal("0.10")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Yorum.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.interpret_backtest_result(comparison)

        assert result == "Yorum."
        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "YENİ bir strateji ÖNERME" in prompt
        assert "15.00%" in prompt  # strateji getirisi doğru aktarılmış mı
        assert "10.00%" in prompt  # benchmark getirisi doğru aktarılmış mı


def test_interpret_backtest_result_handles_none_drawdown_and_sharpe(env):
    """
    DÜZELTME KAYDI: max_drawdown/sharpe_ratio None OLABİLİR (yetersiz
    veri) — bu, gerçek çalıştırmadan ÖNCE fark edilip düzeltilen bir
    None-safe formatlama hatasıydı.
    """
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")

    comparison = MagicMock()
    comparison.strategy_result.total_return = Decimal("0.05")
    comparison.strategy_result.sharpe_ratio = None
    comparison.strategy_result.max_drawdown = None
    comparison.strategy_result.total_trades = 1
    comparison.benchmark_result.total_return = Decimal("0.03")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Yorum.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.interpret_backtest_result(comparison)  # ÇÖKMEMELİ
        assert result == "Yorum."


# ── generate_portfolio_report (bu turda eklendi) ────────────────────────────

def test_generate_report_requests_structured_headings(env):
    pid, portfolio_service = env
    service = AIInsightService(portfolio_service, FakeRiskService(), api_key="sk-ant-fake-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="## Genel Bakış\n\nRapor içeriği.")]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client

        result = service.generate_portfolio_report(pid)

        assert "## Genel Bakış" in result
        prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "## Genel Bakış" in prompt  # başlık talimatı VAR mı
        assert "yatırım tavsiyesi değildir" in prompt
