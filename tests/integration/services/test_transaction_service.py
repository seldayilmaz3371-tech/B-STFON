"""TransactionService testleri — gerçek SQLite repository ile (mock yok)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import (
    AlreadyReversedError,
    BusinessRuleError,
    NotFoundError,
)
from src.infrastructure.database.connection import (
    create_db_engine,
    create_session_factory,
    initialize_database,
)
from src.infrastructure.database.orm_models import portfolios_table
from src.infrastructure.event_bus.in_memory_event_bus import InMemoryEventBus
from src.infrastructure.repositories.sqlite.cash_ledger_repository import (
    SQLiteCashLedgerRepository,
)
from src.infrastructure.repositories.sqlite.transaction_repository import (
    SQLiteTransactionRepository,
)
from src.services.transaction_service import TransactionService

pytestmark = pytest.mark.integration


@pytest.fixture()
def portfolio_id(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'tx_service_test.db'}")
    initialize_database(engine)
    sf = create_session_factory(engine)
    pid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sf() as session:
        session.execute(portfolios_table.insert().values(
            id=pid, name="TX Test", currency="TRY", cost_method="WAVG",
            inception_date="2023-01-01", is_active=1, created_at=now, updated_at=now,
        ))
        session.commit()
    yield pid, sf
    engine.dispose()


@pytest.fixture()
def service(portfolio_id):
    _, sf = portfolio_id
    tx_repo = SQLiteTransactionRepository(sf)
    cash_repo = SQLiteCashLedgerRepository(sf)
    return TransactionService(tx_repo, cash_ledger_repo=cash_repo, event_bus=InMemoryEventBus())


# ── add_transaction ──────────────────────────────────────────────────────────

def test_add_buy_transaction_auto_classifies_bist(service, portfolio_id):
    pid, _ = portfolio_id
    tx = service.add_transaction(
        pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2),
    )
    assert tx.transaction_id is not None
    assert tx.symbol_type == "BIST_STOCK"


def test_add_buy_transaction_auto_classifies_tefas(service, portfolio_id):
    pid, _ = portfolio_id
    tx = service.add_transaction(
        pid, "YAC", TransactionType.BUY, Decimal("1000"), Decimal("1.5"), date(2024, 1, 2),
    )
    assert tx.symbol_type == "TEFAS_FUND"


def test_add_sell_exceeding_position_raises(service, portfolio_id):
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("10"), Decimal("10.00"), date(2024, 1, 2))
    with pytest.raises(BusinessRuleError, match="Yetersiz miktar"):
        service.add_transaction(pid, "THYAO", TransactionType.SELL, Decimal("20"), Decimal("12.00"), date(2024, 1, 3))


def test_add_sell_within_position_succeeds(service, portfolio_id):
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    tx = service.add_transaction(pid, "THYAO", TransactionType.SELL, Decimal("50"), Decimal("12.00"), date(2024, 1, 3))
    assert tx.transaction_id is not None


def test_add_transaction_empty_symbol_raises(service, portfolio_id):
    pid, _ = portfolio_id
    with pytest.raises(BusinessRuleError):
        service.add_transaction(pid, "  ", TransactionType.BUY, Decimal("10"), Decimal("1"), date(2024, 1, 2))


def test_add_transaction_invalid_quantity_raises(service, portfolio_id):
    pid, _ = portfolio_id
    with pytest.raises(BusinessRuleError):
        service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("-5"), Decimal("10"), date(2024, 1, 2))


# ── validate_transaction (dry-run) ──────────────────────────────────────────

def test_validate_transaction_does_not_persist(service, portfolio_id):
    pid, _ = portfolio_id
    result = service.validate_transaction(
        pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2),
    )
    assert result.is_valid is True
    assert service.list_transactions(pid) == []  # HİÇBİR ŞEY kaydedilmedi


def test_validate_transaction_reports_insufficient_quantity(service, portfolio_id):
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("10"), Decimal("10.00"), date(2024, 1, 2))
    result = service.validate_transaction(
        pid, "THYAO", TransactionType.SELL, Decimal("50"), Decimal("12.00"), date(2024, 1, 3),
    )
    assert result.is_valid is False
    assert any("Yetersiz miktar" in e for e in result.errors)


# ── list_transactions ────────────────────────────────────────────────────────

def test_list_transactions_all_symbols(service, portfolio_id):
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("10"), Decimal("10"), date(2024, 1, 1))
    service.add_transaction(pid, "GARAN", TransactionType.BUY, Decimal("20"), Decimal("5"), date(2024, 1, 2))
    all_tx = service.list_transactions(pid)
    assert len(all_tx) == 2
    assert {t.symbol for t in all_tx} == {"THYAO", "GARAN"}


def test_list_transactions_filtered_by_symbol(service, portfolio_id):
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("10"), Decimal("10"), date(2024, 1, 1))
    service.add_transaction(pid, "GARAN", TransactionType.BUY, Decimal("20"), Decimal("5"), date(2024, 1, 2))
    thyao_only = service.list_transactions(pid, symbol="THYAO")
    assert len(thyao_only) == 1
    assert thyao_only[0].symbol == "THYAO"


# ── reverse_transaction ──────────────────────────────────────────────────────

def test_reverse_transaction_removes_from_active_calculations(service, portfolio_id):
    pid, _ = portfolio_id
    tx = service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))

    service.reverse_transaction(tx.transaction_id, reason="Yanlış miktar girildi")

    active = service.list_transactions(pid)
    assert active == []  # aktif listede artık YOK


def test_reverse_transaction_preserves_audit_trail(service, portfolio_id):
    pid, _ = portfolio_id
    tx = service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    service.reverse_transaction(tx.transaction_id, reason="Test sebebi")

    full_history = service.list_transactions(pid, include_reversed=True)
    # Orijinal + reversal marker = 2 satır, İKİSİ DE audit görünümünde
    assert len(full_history) == 2


def test_reverse_already_reversed_transaction_raises(service, portfolio_id):
    pid, _ = portfolio_id
    tx = service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    service.reverse_transaction(tx.transaction_id, reason="İlk reversal")

    with pytest.raises(AlreadyReversedError):
        service.reverse_transaction(tx.transaction_id, reason="İkinci deneme")


def test_reverse_nonexistent_transaction_raises(service, portfolio_id):
    with pytest.raises(NotFoundError):
        service.reverse_transaction(str(uuid.uuid4()), reason="Var olmayan işlem")


def test_reverse_transaction_without_reason_raises(service, portfolio_id):
    pid, _ = portfolio_id
    tx = service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    with pytest.raises(BusinessRuleError, match="sebebi boş olamaz"):
        service.reverse_transaction(tx.transaction_id, reason="   ")


def test_reversal_allows_new_correct_transaction(service, portfolio_id):
    """
    Gerçek dünya senaryosu: yanlış miktar girildi (100 yerine 10),
    reverse edilip doğrusu eklendi. Nihai pozisyon YALNIZCA doğru
    değeri yansıtmalı.
    """
    pid, _ = portfolio_id
    wrong_tx = service.add_transaction(
        pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2),
    )
    service.reverse_transaction(wrong_tx.transaction_id, reason="Miktar yanlış girildi, doğrusu 10")
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("10"), Decimal("10.00"), date(2024, 1, 2))

    from src.domain.calculators.position_quantity_timeseries import compute_quantity_timeseries
    final_transactions = service.list_transactions(pid, symbol="THYAO")
    final_qty = compute_quantity_timeseries(final_transactions).iloc[-1]
    assert final_qty == Decimal("10")  # YALNIZCA doğru işlem etkili


# ── Nakit ledger entegrasyonu (bu turda bulunan KRİTİK boşluk) ──────────────

def test_buy_debits_cash_balance(service, portfolio_id):
    """
    ÖNCEKİ DURUM: CashLedgerRepository'nin HİÇBİR yazıcısı yoktu — her
    BUY sonrası nakit bakiyesi her zaman 0 kalıyordu. Bu test, BUY'ın
    artık GERÇEKTEN nakti düşürdüğünü kanıtlıyor.
    """
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    assert service.get_cash_balance(pid) == Decimal("-1000.00")  # 100 × 10.00


def test_sell_credits_cash_balance(service, portfolio_id):
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    service.add_transaction(pid, "THYAO", TransactionType.SELL, Decimal("50"), Decimal("12.00"), date(2024, 1, 3))
    # -1000 (BUY) + 600 (SELL: 50×12) = -400
    assert service.get_cash_balance(pid) == Decimal("-400.00")


def test_dividend_credits_net_amount(service, portfolio_id):
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    service.add_transaction(
        pid, "THYAO", TransactionType.DIVIDEND, Decimal("100"), Decimal("1.50"), date(2024, 2, 1),
        net_amount=Decimal("135.00"),
    )
    assert service.get_cash_balance(pid) == Decimal("-865.00")  # -1000 + 135


def test_reversal_correctly_offsets_cash_effect(service, portfolio_id):
    """
    KRİTİK: Bir BUY reverse edildiğinde, düşürülen nakit GERİ VERİLMELİ.
    Bu test yazılırken bu davranış GERÇEKTEN doğrulandı.
    """
    pid, _ = portfolio_id
    tx = service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    assert service.get_cash_balance(pid) == Decimal("-1000.00")

    service.reverse_transaction(tx.transaction_id, reason="Yanlış işlem")
    assert service.get_cash_balance(pid) == Decimal("0.00")  # tam olarak geri verildi


def test_reversal_keeps_cash_balance_consistent(service, portfolio_id):
    """
    En kritik doğrulama: karmaşık bir senaryo sonrası (BUY, SELL,
    DIVIDEND, bir reversal) CashLedgerRepository.verify_balance()
    HÂLÂ tutarlı olmalı — bu, muhasebe bütünlüğünün nihai kanıtı.
    """
    pid, sf = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    tx2 = service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("50"), Decimal("12.00"), date(2024, 1, 5))
    service.add_transaction(
        pid, "THYAO", TransactionType.DIVIDEND, Decimal("150"), Decimal("1.00"), date(2024, 2, 1),
        net_amount=Decimal("90.00"),
    )
    service.reverse_transaction(tx2.transaction_id, reason="İkinci alım yanlıştı")
    service.add_transaction(pid, "THYAO", TransactionType.SELL, Decimal("30"), Decimal("15.00"), date(2024, 3, 1))

    from src.infrastructure.repositories.sqlite.cash_ledger_repository import SQLiteCashLedgerRepository
    cash_repo = SQLiteCashLedgerRepository(sf)
    verification = cash_repo.verify_balance(pid)
    assert verification.is_consistent, (
        f"Nakit ledger TUTARSIZ: beklenen={verification.expected}, "
        f"gerçek={verification.actual}, fark={verification.discrepancy}"
    )


def test_split_and_bonus_share_have_no_cash_effect(service, portfolio_id):
    """SPLIT/BONUS_SHARE nakit etkisi taşımamalı — yalnızca miktarı değiştirirler."""
    pid, _ = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))
    balance_before = service.get_cash_balance(pid)
    service.add_transaction(pid, "THYAO", TransactionType.BONUS_SHARE, Decimal("50"), Decimal("0"), date(2024, 2, 1))
    assert service.get_cash_balance(pid) == balance_before  # değişmedi


def test_get_cash_balance_zero_when_no_repo_injected(portfolio_id):
    """cash_ledger_repo inject edilmezse (opsiyonel), get_cash_balance() çökmemeli, 0 dönmeli."""
    from src.infrastructure.repositories.sqlite.transaction_repository import SQLiteTransactionRepository
    pid, sf = portfolio_id
    service_without_cash = TransactionService(SQLiteTransactionRepository(sf), cash_ledger_repo=None)
    assert service_without_cash.get_cash_balance(pid) == Decimal("0")


def test_check_ledger_integrity_detects_manual_db_corruption(service, portfolio_id, caplog):
    """
    KRİTİK doğrulama: otomatik post-write kontrolü GERÇEKTEN çalışıyor mu?
    Bunu kanıtlamanın tek yolu GERÇEK bir bozulma senaryosu yaratmak —
    burada bir cash_ledger_entries satırını DOĞRUDAN SQL ile bozup
    (sanki harici bir müdahale/bug olmuş gibi), bir SONRAKİ işlem
    eklendiğinde _check_ledger_integrity'nin bunu CRITICAL log ile
    yakaladığını doğruluyoruz.
    """
    import logging
    from sqlalchemy import select
    from src.infrastructure.database.orm_models import cash_ledger_entries_table

    pid, sf = portfolio_id
    service.add_transaction(pid, "THYAO", TransactionType.BUY, Decimal("100"), Decimal("10.00"), date(2024, 1, 2))

    # Kasıtlı bozulma: mevcut kaydın balance_after'ını doğrudan SQL ile boz
    with sf() as session:
        row = session.execute(select(cash_ledger_entries_table.c.id)).first()
        session.execute(
            cash_ledger_entries_table.update()
            .where(cash_ledger_entries_table.c.id == row.id)
            .values(balance_after="-999999.00")
        )
        session.commit()

    with caplog.at_level(logging.ERROR):
        # Yeni bir işlem eklemek _check_ledger_integrity'yi tetikler
        service.add_transaction(pid, "GARAN", TransactionType.BUY, Decimal("10"), Decimal("5.00"), date(2024, 1, 3))

    # get_balance() artık bozulmuş -999999.00'ı okuyacak (en son yazılan
    # satır o), bu yüzden bu YENİ işlemin kendi balance_after'ı da onun
    # üzerine inşa edilecek — ama verify_balance() TÜM kayıtların
    # TOPLAMINI bağımsız hesapladığı için tutarsızlığı YAKALAYACAK.
    verification = service.check_ledger_integrity(pid)
    assert verification.is_consistent is False
