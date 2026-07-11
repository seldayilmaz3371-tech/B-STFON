"""
TransactionService — design doc Bölüm 3.3 "TransactionService Ana
Methodları" ile büyük ölçüde uyumlu, kapsamı bilinçli daraltılmış:

  UYGULANAN:  add_transaction, reverse_transaction, list_transactions,
              validate_transaction
  ERTELENEN:  import_transactions (CSV toplu içe aktarma) — bu turun
              kapsamı dışında, gerçek bir format/şema kararı gerektiriyor.

DÜZELTME (bu turda bulundu — KRİTİK): CashLedgerRepository'nin daha
önce HİÇBİR gerçek yazıcısı yoktu (yalnızca RiskService salt-okunur
get_balance() çağırıyordu). Her BUY/SELL/DIVIDEND artık karşılık gelen
bir CashLedgerEntry üretiyor.

MİMARİ KARAR — Nakit ledger yazımı EVENT BUS ÜZERİNDEN DEĞİL, DOĞRUDAN:
  İki alternatif değerlendirildi:
    A) TransactionService doğrudan yazar (SEÇİLEN)
    B) Event bus subscriber'ı (transaction.added -> cash ledger yaz)
  (B) mimari olarak "Event-Driven" prensibiyle daha uyumlu görünüyor
  ama KRİTİK bir risk taşıyor: InMemoryEventBus.publish() BEST-EFFORT
  (handler hatası loglanır, yukarı fırlatılmaz — bkz. in_memory_event_bus.py
  gerekçesi: "cache tutarlılığı, transaction atomicity'sinden daha az
  değerli"). O gerekçe CACHE invalidation gibi ikincil etkiler için
  doğruydu — ama CashLedgerEntry bir cache DEĞİL, Transaction ile AYNI
  SEVİYEDE birincil bir muhasebe kaydı. Best-effort event ile yazılırsa
  nakit etkisi SESSİZCE kaybolabilir. Bu yüzden (A) seçildi: nakit
  ledger yazımı add_transaction()'ın GARANTİLİ bir parçası, event bus'a
  bağımlı değil. Event yine YAYINLANIYOR (ikincil/best-effort tüketiciler
  için — örn. gelecekte cache invalidation) ama nakit etkisi ondan
  BAĞIMSIZ olarak zaten yazılmış oluyor.

BİLİNEN ATOMICITY SINIRLAMASI (açıkça işaretleniyor, çözülmedi):
  transaction_repo.add_transaction() ve cash_ledger_repo.add_entry()
  AYRI session'lar açıyor (her repository kendi session'ını yönetiyor)
  — bu, GERÇEK bir DB-seviyesi atomicity SAĞLAMIYOR. İki yazım arasında
  process çökerse, Transaction var ama karşılık gelen CashLedgerEntry
  YOK olabilir. Doğru çözüm (Unit of Work pattern, repository'ler arası
  paylaşılan session) TÜM repository'lerin session yönetim desenini
  değiştirmeyi gerektiren büyük bir refactor — bu turun kapsamı dışında.
  KABUL EDİLMİŞ RİSK: Tek kullanıcılı masaüstü uygulamasında iki yazım
  arası çökme olasılığı çok düşük; CashLedgerRepository.verify_balance()
  (Faz B'de tam olarak bu tür bir tutarsızlığı SONRADAN tespit etmek
  için inşa edilmişti) bir güvenlik ağı olarak zaten mevcut.

BİLİNÇLİ OLARAK ERTELENEN — DEPOSIT/WITHDRAWAL/FEE nakit etkisi:
  Bu 3 transaction_type UI formunda HENÜZ YOK (yalnızca BUY/SELL/
  DIVIDEND destekleniyor) ve Transaction modelinde bu tipler için net
  bir "tutar" alanı ayrımı yok (quantity/price alanları BUY/SELL
  semantiğine göre tasarlandı). Bu yüzden _cash_effect() bu tipler için
  None döner (nakit etkisi hesaplanmaz) — bu SESSİZCE yanlış değil,
  çünkü bu tipler zaten hiçbir yoldan girilemiyor. UI'a eklendiklerinde
  bu fonksiyon genişletilmeli.

BİLİNÇLİ OLARAK DOĞRULANMAYAN — negatif nakit bakiyesi:
  DEPOSIT (nakit yatırma) UI'da henüz YOK, yani bir kullanıcı hiç nakit
  yatırmadan doğrudan BUY yapabilir — bakiye negatife düşer. Bu
  BİLİNÇLİ olarak ENGELLENMEDİ çünkü DEPOSIT desteklenene kadar
  engellemek BUY işlemini tamamen kullanılamaz hale getirirdi. Bu,
  DEPOSIT/WITHDRAWAL UI'a eklenene kadar geçerli, açıkça işaretlenmiş
  bir ara-dönem durumu.

MİMARİ KARAR — Proaktif "yetersiz miktar" kontrolü:
  add_transaction() bir SELL işlemi eklemeden ÖNCE mevcut pozisyonu
  hesaplayıp kontrol eder (compute_quantity_timeseries ile — DRY,
  cost_basis_calculator'ın mantığını TEKRARLAMAZ, mevcut pure fonksiyonu
  YENİDEN KULLANIR).

MİMARİ KARAR — symbol_type otomatik sınıflandırma:
  provider_router.py::classify_symbol() YENİDEN KULLANILIYOR. SINIRLAMA:
  BIST_ETF, BIST_STOCK'tan ayırt EDİLEMEZ (pratik etkisi yok, ikisi de
  aynı maliyet hesaplama mantığını kullanıyor).

MİMARİ KARAR — Sembolün gerçekten var olup olmadığının doğrulanması:
  BİLİNÇLİ OLARAK YAPILMIYOR — bir muhasebe kaydı network kesintisinde
  bile girilebilmeli.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.domain.calculators.position_quantity_timeseries import (
    compute_quantity_timeseries,
)
from src.domain.enums.ledger_entry_type import LedgerEntryType
from src.domain.enums.transaction_type import TransactionType
from src.domain.exceptions.domain_exceptions import (
    BusinessRuleError,
    InsufficientQuantityError,
    NotFoundError,
)
from src.domain.models.cash_ledger_entry import CashLedgerEntry
from src.domain.models.transaction import Transaction
from src.infrastructure.data_providers.provider_router import classify_symbol
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


def _map_asset_class(symbol: str) -> str:
    """
    classify_symbol() 'BIST'/'TEFAS' döner — DB'nin symbol_type CHECK
    constraint'i ile eşlemesi gerekiyor. SINIRLAMA: BIST_ETF ayırt
    edilemez (bkz. modül docstring'i).
    """
    return "BIST_STOCK" if classify_symbol(symbol) == "BIST" else "TEFAS_FUND"


def _cash_effect(
    transaction_type: TransactionType, quantity: Decimal, price: Decimal, net_amount: Decimal | None,
) -> tuple[LedgerEntryType, Decimal] | None:
    """
    Bir işlemin nakit etkisini hesaplar.

    Returns:
        (entry_type, amount) — amount HER ZAMAN pozitif (CashLedgerEntry
        DDL CHECK: amount > 0), yön entry_type ile belirleniyor.
        None — bu transaction_type için nakit etkisi HENÜZ hesaplanmıyor
        (bkz. modül docstring'indeki "bilinçli olarak ertelenen" notu).

    NOT: Komisyon/BSMV/damga vergisi HENÜZ hesaba katılmıyor (transactions
    DDL'in MVP alt kümesinde bu alanlar yok — bkz. orm_models.py modül
    docstring'i). BUY'ın nakit etkisi yalnızca quantity×price; gerçek
    aracı kurum komisyonu bu tutara dahil DEĞİL. Bu, gerçek nakit
    bakiyesinden HAFİF SAPMAYA yol açar (komisyon kadar) — bilinen,
    kabul edilmiş bir MVP sınırlaması.
    """
    if transaction_type is TransactionType.BUY:
        return LedgerEntryType.DEBIT, quantity * price
    if transaction_type is TransactionType.SELL:
        return LedgerEntryType.CREDIT, quantity * price
    if transaction_type is TransactionType.DIVIDEND:
        assert net_amount is not None  # Transaction.__post_init__ garanti eder
        return LedgerEntryType.CREDIT, net_amount
    return None


class TransactionService:
    def __init__(
        self, transaction_repo: Any, cash_ledger_repo: Any | None = None, event_bus: Any | None = None,
    ) -> None:
        self._tx_repo = transaction_repo
        self._cash_ledger_repo = cash_ledger_repo
        self._event_bus = event_bus

    # ── Doğrulama (add_transaction ile PAYLAŞILAN mantık) ──────────────────

    def _current_quantity(self, portfolio_id: str, symbol: str) -> Decimal:
        transactions = self._tx_repo.get_by_symbol(portfolio_id, symbol)
        series = compute_quantity_timeseries(transactions)
        return series.iloc[-1] if not series.empty else Decimal("0")

    def _validate(
        self,
        portfolio_id: str,
        symbol: str,
        transaction_type: TransactionType,
        quantity: Decimal,
        price: Decimal,
        trade_date: date,
        split_ratio: Decimal | None,
        net_amount: Decimal | None,
    ) -> tuple[ValidationResult, Transaction | None]:
        errors: list[str] = []

        if not symbol or not symbol.strip():
            errors.append("Sembol boş olamaz.")

        transaction: Transaction | None = None
        try:
            transaction = Transaction(
                symbol=symbol.strip().upper(),
                transaction_type=transaction_type,
                timestamp=datetime.combine(trade_date, datetime.min.time()),
                quantity=quantity,
                price=price,
                split_ratio=split_ratio,
                net_amount=net_amount,
                symbol_type=_map_asset_class(symbol.strip().upper()) if symbol.strip() else None,
            )
        except ValueError as exc:
            errors.append(str(exc))

        if transaction is not None and transaction_type is TransactionType.SELL:
            current_qty = self._current_quantity(portfolio_id, transaction.symbol)
            if quantity > current_qty:
                errors.append(
                    f"Yetersiz miktar: {current_qty} adet mevcut, "
                    f"{quantity} adet satılmaya çalışılıyor."
                )

        return ValidationResult(is_valid=not errors, errors=tuple(errors)), (
            transaction if not errors else None
        )

    def _record_cash_effect(
        self, portfolio_id: str, transaction_id: str, transaction_type: TransactionType,
        quantity: Decimal, price: Decimal, net_amount: Decimal | None, trade_date: date,
        description: str,
    ) -> None:
        """Nakit etkisi varsa (bkz. _cash_effect) CashLedgerEntry yazar; yoksa no-op."""
        if self._cash_ledger_repo is None:
            return
        effect = _cash_effect(transaction_type, quantity, price, net_amount)
        if effect is None:
            return
        entry_type, amount = effect
        current_balance = self._cash_ledger_repo.get_balance(portfolio_id)
        new_balance = (
            current_balance + amount if entry_type is LedgerEntryType.CREDIT
            else current_balance - amount
        )
        self._cash_ledger_repo.add_entry(CashLedgerEntry(
            portfolio_id=portfolio_id, entry_type=entry_type, amount=amount,
            entry_date=trade_date, description=description,
            balance_after=new_balance, transaction_id=transaction_id,
        ))
        self._check_ledger_integrity(portfolio_id)

    def _check_ledger_integrity(self, portfolio_id: str) -> None:
        """
        DÜZELTME (bu turda bulundu — İKİNCİ unutulmuş entegrasyon):
        CashLedgerRepository.verify_balance() Faz B'de TAM OLARAK bu
        senaryo için ("tutarsızlığı sonradan tespit etmek") inşa
        edilmişti ama hiçbir zaman GERÇEKTEN ÇAĞRILMIYORDU — bir
        bataryası olmayan duman dedektörü gibiydi.

        NEDEN post-write (yazma SONRASI), pre-write DEĞİL: Tutarsızlık,
        BU işlemin kendisinden değil, GEÇMİŞTEKİ bir veri bütünlüğü
        sorunundan (örn. manuel DB müdahalesi, bir önceki reversal'daki
        gizli bir hata) kaynaklanabilir — bu kontrol "bu işlem doğru
        mu" değil "defter GENEL OLARAK hâlâ tutarlı mı" sorusuna cevap
        veriyor. Her yazımdan sonra çalıştırmak, sorunu OLUŞTUĞU ANA
        en yakın noktada yakalar (50 işlem sonra fark etmek yerine).

        NEDEN exception DEĞİL, CRITICAL log: Yazma zaten commit edildi
        — bu noktada exception fırlatmak yanlış bir sinyal verir
        ("işlem kaydedilemedi" izlenimi, oysa kaydedildi). Bunun yerine
        yüksek önemde loglanıyor + UI'da health-check bölümünde görünür
        oluyor (bkz. app.py::_render_ledger_health_check).
        """
        if self._cash_ledger_repo is None:
            return
        verification = self._cash_ledger_repo.verify_balance(portfolio_id)
        if not verification.is_consistent:
            logger.error(
                "cash_ledger_integrity_violation",
                portfolio_id=portfolio_id,
                expected=str(verification.expected),
                actual=str(verification.actual),
                discrepancy=str(verification.discrepancy),
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def validate_transaction(
        self,
        portfolio_id: str,
        symbol: str,
        transaction_type: str,
        quantity: Decimal,
        price: Decimal,
        trade_date: date,
        split_ratio: Decimal | None = None,
        net_amount: Decimal | None = None,
    ) -> ValidationResult:
        """Persist ETMEDEN doğrulama — UI'da submit öncesi canlı geri bildirim için."""
        try:
            ttype = TransactionType(transaction_type)
        except ValueError:
            valid = [t.value for t in TransactionType]
            return ValidationResult(
                is_valid=False,
                errors=(f"Geçersiz transaction_type: '{transaction_type}'. Geçerli: {valid}",),
            )
        result, _ = self._validate(
            portfolio_id, symbol, ttype, quantity, price,
            trade_date, split_ratio, net_amount,
        )
        return result

    def add_transaction(
        self,
        portfolio_id: str,
        symbol: str,
        transaction_type: str,
        quantity: Decimal,
        price: Decimal,
        trade_date: date,
        split_ratio: Decimal | None = None,
        net_amount: Decimal | None = None,
    ) -> Transaction:
        """
        Raises:
            BusinessRuleError: Validasyon hatası varsa VEYA transaction_type
                geçersiz bir string ise.
        """
        try:
            ttype = TransactionType(transaction_type)
        except ValueError:
            valid = [t.value for t in TransactionType]
            raise BusinessRuleError(
                f"Geçersiz transaction_type: '{transaction_type}'. Geçerli: {valid}"
            ) from None

        result, transaction = self._validate(
            portfolio_id, symbol, ttype, quantity, price,
            trade_date, split_ratio, net_amount,
        )
        if not result.is_valid or transaction is None:
            raise BusinessRuleError("; ".join(result.errors))

        symbol_type = _map_asset_class(transaction.symbol)
        new_id = self._tx_repo.add_transaction(portfolio_id, symbol_type, transaction)

        self._record_cash_effect(
            portfolio_id, new_id, ttype, quantity, price, net_amount, trade_date,
            description=f"{ttype.value} {transaction.symbol}",
        )

        if self._event_bus is not None:
            from src.infrastructure.event_bus.in_memory_event_bus import Event
            self._event_bus.publish(Event(
                name="transaction.added",
                payload={
                    "portfolio_id": portfolio_id, "transaction_id": new_id,
                    "symbol": transaction.symbol, "transaction_type": ttype.value,
                },
            ))

        return Transaction(
            symbol=transaction.symbol, transaction_type=transaction.transaction_type,
            timestamp=transaction.timestamp, quantity=transaction.quantity,
            price=transaction.price, split_ratio=transaction.split_ratio,
            net_amount=transaction.net_amount, transaction_id=new_id,
            symbol_type=transaction.symbol_type, portfolio_id=portfolio_id,
        )

    def list_transactions(
        self, portfolio_id: str, symbol: str | None = None, include_reversed: bool = False,
    ) -> list[Transaction]:
        if symbol is not None:
            result: list[Transaction] = self._tx_repo.get_by_symbol(portfolio_id, symbol)
            return result
        result = self._tx_repo.list_by_portfolio(portfolio_id, include_inactive=include_reversed)
        return result

    def reverse_transaction(self, transaction_id: str, reason: str) -> str:
        """
        Raises:
            NotFoundError, AlreadyReversedError: bkz.
                SQLiteTransactionRepository.reverse_transaction.

        Nakit etkisi tersine çevrilir: orijinal işlemin nakit etkisi
        varsa (bkz. _cash_effect), TERS yönde bir CashLedgerEntry
        eklenir (örn. BUY'ın DEBIT'i, reversal'da CREDIT olarak geri
        verilir). Bu, verify_balance()'ın reversal SONRASI da tutarlı
        kalmasını garanti eder — GERÇEKTEN test edildi (bkz.
        test_reversal_keeps_cash_balance_consistent).
        """
        if not reason or not reason.strip():
            raise BusinessRuleError("Reversal sebebi boş olamaz (audit trail zorunluluğu).")

        original = self._tx_repo.get_by_id(transaction_id)
        if original is None:
            raise NotFoundError("Transaction", transaction_id)

        marker_id: str = self._tx_repo.reverse_transaction(transaction_id, reason.strip())

        if self._cash_ledger_repo is not None and original.portfolio_id is not None:
            effect = _cash_effect(
                original.transaction_type, original.quantity, original.price, original.net_amount,
            )
            if effect is not None:
                entry_type, amount = effect
                # TERS yön: orijinal DEBIT ise reversal CREDIT, ve tersi.
                reversed_entry_type = (
                    LedgerEntryType.CREDIT if entry_type is LedgerEntryType.DEBIT
                    else LedgerEntryType.DEBIT
                )
                current_balance = self._cash_ledger_repo.get_balance(original.portfolio_id)
                new_balance = (
                    current_balance + amount if reversed_entry_type is LedgerEntryType.CREDIT
                    else current_balance - amount
                )
                self._cash_ledger_repo.add_entry(CashLedgerEntry(
                    portfolio_id=original.portfolio_id, entry_type=reversed_entry_type,
                    amount=amount,
                    # KRİTİK DÜZELTME (bu turda, verify_balance() testiyle
                    # BULUNDU): entry_date BURADA orijinal işlemin TARİHİ
                    # DEĞİL, BUGÜNÜN tarihi olmalı — reversal, muhasebe
                    # olarak BUGÜN gerçekleşen bir olay (geçmişi düzeltiyor
                    # ama kendisi geçmişte OLMADI). Orijinal tarihi
                    # kullanmak, get_balance()'ın "entry_date DESC" sıralamasını
                    # bozuyordu: reversal, aradaki (orijinal işlemden SONRA
                    # ama BUGÜNDEN ÖNCE tarihli) başka bir işlemin gerisinde
                    # kalıyor, bu da o işlemin current_balance() sorgusunun
                    # reversal'ın etkisini GÖRMEMESİNE yol açıyordu — somut
                    # olarak 600 TL'lik sistematik bir tutarsızlık üretti
                    # (test_reversal_keeps_cash_balance_consistent ile
                    # yakalandı).
                    entry_date=date.today(),
                    description=f"REVERSAL: {reason.strip()}",
                    balance_after=new_balance, transaction_id=marker_id,
                ))
                self._check_ledger_integrity(original.portfolio_id)

        if self._event_bus is not None:
            from src.infrastructure.event_bus.in_memory_event_bus import Event
            self._event_bus.publish(Event(
                name="transaction.reversed",
                payload={"original_transaction_id": transaction_id, "reversal_marker_id": marker_id},
            ))
        return marker_id

    def get_cash_balance(self, portfolio_id: str) -> Decimal:
        """UI'da 'nakit bakiyesi' göstermek için — daha önce hiçbir yerde yoktu."""
        if self._cash_ledger_repo is None:
            return Decimal("0")
        result: Decimal = self._cash_ledger_repo.get_balance(portfolio_id)
        return result

    def check_ledger_integrity(self, portfolio_id: str) -> Any:
        """
        UI'da isteğe bağlı, açık bir 'Bütünlüğü Doğrula' aksiyonu için
        (RiskService'in 'Risk Metriklerini Hesapla' butonuyla AYNI
        UX deseni — pahalı/kritik kontroller otomatik değil, açık
        tetiklemeli). Otomatik kontrol zaten her yazımdan sonra
        çalışıyor (bkz. _check_ledger_integrity) — bu, kullanıcının
        istediği an MANUEL olarak da tetikleyebilmesi için.

        Returns:
            BalanceVerification (cash_ledger_repository.py) — Any
            olarak tip verildi çünkü TransactionService bu DTO'yu
            import etmiyor (repository katmanına ait), yalnızca
            geçirdiği nesneyi olduğu gibi döndürüyor.
        """
        if self._cash_ledger_repo is None:
            raise ValueError("cash_ledger_repo inject edilmemiş.")
        return self._cash_ledger_repo.verify_balance(portfolio_id)
