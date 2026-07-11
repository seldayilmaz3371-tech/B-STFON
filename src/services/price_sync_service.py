"""
PriceSyncService — fiyat geçmişi için Cache-Aside (Read-Through Cache)
katmanı.

DÜZELTME (bu turda bulundu — ÜÇÜNCÜ "unutulmuş entegrasyon" örneği):
  PriceRepository (Faz B'de upsert/upsert_batch/get_missing_dates/
  get_latest_price ile inşa edildi) HİÇBİR ZAMAN gerçek bir yazıcıya
  sahip olmadı. RiskService, her risk hesaplamasında TÜM lookback
  penceresini (örn. 500 gün) canlı sağlayıcıdan (yfinance/TEFAS)
  YENİDEN ÇEKİYORDU — ne performans (gereksiz network trafiği) ne de
  güvenilirlik (sağlayıcı erişilemezse TÜM hesaplama başarısız oluyor)
  açısından sürdürülebilir.

MİMARİ DESEN — Cache-Aside (Read-Through Cache):
  1. price_repo'da bu sembol için hangi güncel veri var, öğren
     (get_latest_price).
  2. Eksik günleri (get_missing_dates — hafta sonu hariç, bkz. o
     metodun kendi BİLİNEN SINIRLAMASI: gerçek BIST tatil takvimi yok)
     hesapla.
  3. Eksik günler VARSA, YALNIZCA O ARALIĞI canlı sağlayıcıdan çek
     (500 gün değil, tipik olarak 1-5 gün) ve upsert_batch ile yaz.
  4. price_repo'dan TAM (şimdi güncel) veriyi oku ve döndür.

  Bu desen RiskService'DEN TAMAMEN GİZLENİYOR — RiskService yalnızca
  get_price_history(symbol, start, end) çağırır, verinin cache'ten mi
  canlı sağlayıcıdan mı geldiğini BİLMEZ (Adapter Pattern'in doğal
  bir uzantısı — "veri kaynağı soyutlama katmanı").

NEDEN AYRI BİR SERVİS (RiskService'e gömülü değil):
  Bu, yalnızca RiskService'in değil, gelecekteki TechnicalChart/
  Backtest Engine'in de faydalanacağı PAYLAŞILAN bir katman. RiskService
  içine gömülseydi, backtest engine yazıldığında AYNI mantık TEKRAR
  yazılırdı (DRY ihlali).

TEST EDİLEBİLİRLİK (bu servisin ÜÇ senaryosu AYRI AYRI doğrulanmalı,
gizlenmiş karmaşıklık kabul edilemez):
  1. Cache TAM (hiç eksik gün yok) → canlı sağlayıcıya HİÇ dokunmaz.
  2. Cache KISMEN eksik (yalnızca son N gün) → yalnızca O ARALIĞI çeker.
  3. Cache BOŞ (ilk kullanım) → TAM lookback penceresini çeker.

BİLİNEN SINIRLAMA (miras alınıyor, açıkça işaretleniyor):
  get_missing_dates()'in "yalnızca hafta sonu hariç, gerçek BIST tatil
  takvimi yok" sınırlaması burada da geçerli — resmi tatil günleri
  yanlışlıkla "eksik" sayılıp HER SENKRONİZASYONDA gereksiz yere
  yeniden denenecek (zararsız ama israf; yfinance zaten o günler için
  boş sonuç dönecek, sonsuz döngü YARATMAZ çünkü upsert_batch başarısız
  olsa bile get_missing_dates bir SONRAKİ çağrıda AYNI günü yine eksik
  görür — bu, gerçek bir tatil takvimi entegre edilene kadar kabul
  edilmiş bir verimlilik kaybı, doğruluk sorunu DEĞİL).

  İKİNCİ BİR BENZER SINIRLAMA (test yazılırken bulundu): Bir sembolün
  sağlayıcıdaki (yfinance/TEFAS) GERÇEK veri geçmişi, istenen
  lookback_days'ten KISA ise (örn. yakın zamanda halka arz olmuş bir
  hisse), get_missing_dates bu "sağlayıcıda hiç var olmayan" tarihleri
  de KALICI OLARAK eksik görür ve HER senkronizasyonda yeniden dener
  — provider boş yanıt döner, hiçbir zarar vermez ama kaynak israfı
  oluşturur. Gerçek bir çözüm ("bu sembol için veri şu tarihten önce
  yok" bilgisini kalıcı olarak işaretlemek) bu turun kapsamı dışında,
  ayrı bir iyileştirme olarak not ediliyor.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from src.domain.enums.asset_type import AssetType
from src.domain.models.price_series import PriceSeries
from src.infrastructure.data_providers.provider_router import classify_symbol
from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


def _map_asset_type(symbol: str) -> AssetType:
    """
    classify_symbol() 'BIST'/'TEFAS' döner — AssetType'a eşleniyor.
    SINIRLAMA: BIST_ETF/BENCHMARK ayrımı YAPILMIYOR (TransactionService.
    _map_asset_class ile AYNI, bilinen basitleştirme — bkz. o dosyanın
    docstring'i).
    """
    return AssetType.BIST_STOCK if classify_symbol(symbol) == "BIST" else AssetType.TEFAS_FUND


def _ohlcv_row_to_price_series(symbol: str, symbol_type: AssetType, row_date: date, row: Any, source: str) -> PriceSeries:
    def _dec(value: Any) -> Decimal | None:
        return Decimal(str(value)) if value is not None and not pd.isna(value) else None

    return PriceSeries(
        symbol=symbol, symbol_type=symbol_type, date=row_date,
        close_price=Decimal(str(row["Close"])),
        open_price=_dec(row.get("Open")), high_price=_dec(row.get("High")),
        low_price=_dec(row.get("Low")), volume=_dec(row.get("Volume")),
        source=source,
    )


class PriceSyncService:
    def __init__(self, price_repo: Any, market_data_provider: Any) -> None:
        self._price_repo = price_repo
        self._provider = market_data_provider

    def get_price_history(
        self, symbol: str, start_date: date, end_date: date,
    ) -> pd.DataFrame:
        """
        Cache-aside — bkz. modül docstring'i. Her zaman price_repo'dan
        OKUYARAK döner (canlı fetch yalnızca EKSİK günleri doldurmak
        için, dönüş değeri her zaman tutarlı tek bir kaynaktan gelir).

        Returns:
            pd.DataFrame — PriceRepository.get_ohlcv() ile AYNI format
            (date index, open/high/low/close/volume/adjusted_close
            kolonları).

        DİKKAT — ÇOKLU THREAD'DEN ÇAĞIRMAYIN: Bu metod fetch+write'ı
        TEK ÇAĞRIDA birleştiriyor. Birden fazla sembolü PARALEL
        işlemek istiyorsanız (örn. RiskService'in ThreadPoolExecutor
        kullanımı), bunun yerine fetch_missing_only() (worker thread'de
        güvenli) + write_batch() (ANA thread'de, TEK seferde) + 
        get_cached_ohlcv() (salt okuma, paralel güvenli) desenini
        kullanın — bkz. o metodların docstring'lerindeki GERÇEK
        ÖLÇÜLMÜŞ "database is locked" bulgusu.
        """
        missing = self._price_repo.get_missing_dates(
            symbol, start_date, end_date, trading_days_only=True,
        )
        if missing:
            self._fetch_and_cache_range(symbol, min(missing), max(missing))

        result: pd.DataFrame = self._price_repo.get_ohlcv(symbol, start_date, end_date)
        return result

    def fetch_missing_only(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[PriceSeries]:
        """
        DÜZELTME (bu turda, GERÇEK bir yük testiyle bulundu — KRİTİK
        eşzamanlılık kısıtlaması): RiskService'e ThreadPoolExecutor
        eklenirken, birden fazla worker thread'in AYNI ANDA
        get_price_history() (fetch+write birleşik) çağırması "database
        is locked" hatalarına yol açtı — SQLite'ın TEK-YAZAR kısıtlaması
        nedeniyle. busy_timeout'u 15000ms'ye çıkarmak bile ÇÖZMEDİ
        (PRAGMA'nın gerçekten uygulandığı doğrulandı — sorun timeout
        SÜRESİ değil, SQLite'ın DOĞASI: birden fazla UZUN transaction
        aynı anda yazmaya çalışınca ciddi çekişme oluşuyor).

        DOĞRU ÇÖZÜM: Fetch (network-bound, paralelleştirilebilir) ile
        write (DB-bound, SQLite'ta SERİLEŞTİRİLMELİ) sorumluluklarını
        AYIRMAK. Bu metod YALNIZCA fetch yapar, DB'YE HİÇ YAZMAZ —
        ThreadPoolExecutor worker'larında GÜVENLE çağrılabilir (yalnızca
        okuma [get_missing_dates] + network I/O, DB YAZMA YOK).

        Çağıran taraf (RiskService), TÜM worker'ların sonuçlarını
        topladıktan SONRA, ANA THREAD'DEN write_batch() ile TEK BİR
        yazma işlemi yapmalı.

        Returns:
            Yazılması gereken PriceSeries listesi (boş liste = zaten
            günceldi, yazılacak bir şey yok).
        """
        missing = self._price_repo.get_missing_dates(
            symbol, start_date, end_date, trading_days_only=True,
        )
        if not missing:
            return []
        return self._fetch_range_without_writing(symbol, min(missing), max(missing))

    def write_batch(self, price_list: list[PriceSeries]) -> Any:
        """
        fetch_missing_only()'den toplanan SONUÇLARI TEK BİR
        upsert_batch() çağrısıyla yazar — ANA THREAD'DEN çağrılmalı
        (SQLite'ın tek-yazar kısıtlaması nedeniyle, bkz.
        fetch_missing_only() docstring'i).
        """
        return self._price_repo.upsert_batch(price_list)

    def get_cached_ohlcv(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Salt okuma — write_batch() ile senkronize edilmiş veriyi okur.
        WAL modunda ÇOKLU OKUYUCU güvenlidir (bkz. Faz B ADR-002) —
        ThreadPoolExecutor worker'larından PARALEL çağrılabilir
        (write_batch()'in AKSİNE).
        """
        result: pd.DataFrame = self._price_repo.get_ohlcv(symbol, start_date, end_date)
        return result

    def _fetch_range_without_writing(self, symbol: str, start: date, end: date) -> list[PriceSeries]:
        """fetch_missing_only()'ün DB'siz versiyonu — _fetch_and_cache_range ile PAYLAŞILAN mantık."""
        try:
            ohlcv = self._provider.fetch_ohlcv(
                symbol=symbol, timeframe="1d",
                start_date=datetime.combine(start, datetime.min.time()),
                end_date=datetime.combine(end, datetime.min.time()),
            )
        except Exception as exc:
            logger.warning("price_sync_fetch_failed", symbol=symbol, error=str(exc))
            return []

        if ohlcv.empty:
            logger.warning("price_sync_empty_response", symbol=symbol, start=str(start), end=str(end))
            return []

        symbol_type = _map_asset_type(symbol)
        source_name = self._provider.get_provider_name()
        price_list = []
        for idx, row in ohlcv.iterrows():
            row_date = pd.Timestamp(idx).date()
            try:
                price_list.append(
                    _ohlcv_row_to_price_series(symbol, symbol_type, row_date, row, source=source_name)
                )
            except ValueError as exc:
                logger.warning("price_sync_invalid_row", symbol=symbol, date=str(row_date), error=str(exc))
                continue
        return price_list

    def sync_symbol(self, symbol: str, lookback_days: int = 252) -> dict[str, int]:
        """
        Bir sembol için INCREMENTAL senkronizasyon — Scheduler job'ından
        çağrılır. get_price_history()'den FARKI: dönüş değeri yok
        (yalnızca cache'i günceller), ve "eksik gün yoksa hiçbir şey
        yapma" davranışı AYNI ama giriş noktası farklı (proaktif
        senkronizasyon vs. reaktif okuma-sırasında-doldurma).
        """
        end = date.today()
        start = end - timedelta(days=lookback_days * 2)  # takvim günü, iş günü değil (hafta sonu payı)
        missing = self._price_repo.get_missing_dates(symbol, start, end, trading_days_only=True)
        if not missing:
            return {"fetched": 0, "cached": 0}
        return self._fetch_and_cache_range(symbol, min(missing), max(missing))

    def _fetch_and_cache_range(self, symbol: str, start: date, end: date) -> dict[str, int]:
        """
        get_price_history()/sync_symbol() (TEK THREAD'den çağrılan,
        fetch+write birleşik) yolu için — bkz. fetch_missing_only()
        docstring'i: bu metod ÇOKLU THREAD'DEN GÜVENLE ÇAĞRILAMAZ
        (write_batch DB yazması içeriyor). DRY: gerçek fetch mantığı
        _fetch_range_without_writing() ile PAYLAŞILIYOR, burada yalnızca
        onun üzerine write_batch() ekleniyor.
        """
        price_list = self._fetch_range_without_writing(symbol, start, end)
        if not price_list:
            return {"fetched": 0, "cached": 0}

        result = self.write_batch(price_list)
        logger.info(
            "price_sync_completed", symbol=symbol,
            inserted=result.inserted, updated=result.updated, failed=len(result.failed),
        )
        return {"fetched": len(price_list), "cached": result.inserted + result.updated}
