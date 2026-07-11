"""
SnapshotScheduler — periyodik risk snapshot hesaplama job'ı.

MİMARİ KARAR (Streamlit + APScheduler süreç izolasyonu — daha önce
gerekçelendirildi, burada özetleniyor):
  BackgroundScheduler (thread-based, AsyncIOScheduler DEĞİL) — Streamlit
  doğası gereği sync çalışıyor, PortfolioService'in ADR-003 (senkron-
  öncelikli, ThreadPoolExecutor) kararıyla TUTARLI. container.py'da
  @st.cache_resource ile TEK instance garantisi (Container'ın kendisiyle
  AYNI desen) — her Streamlit rerun'da yeni scheduler/çakışan job
  OLUŞMASINI önlüyor.

  Ayrı bir process (Seçenek B) DEĞİL: tek-kullanıcılı masaüstü ölçeğinde
  operasyonel karmaşıklığı (ayrı process başlatma/izleme) haklı
  çıkarmıyor — bu projede tutarlı olarak izlenen "ölçeğe uygun
  basitlik" prensibiyle örtüşüyor (SQLAlchemy Core>ORM, in-process
  EventBus>Kafka ile AYNI karar sınıfı).

HATA İZOLASYONU: Bir portföyün snapshot hesaplaması başarısız olursa
(örn. yetersiz veri, network kesintisi), DİĞER portföylerin
hesaplanmasını ENGELLEMEZ — upsert_batch()'teki per-item error
isolation deseniyle TUTARLI.

TEST EDİLEBİLİRLİK: run_now() metodu, zamanlayıcıyı beklemeden job'ı
HEMEN çalıştırır — testlerde ve UI'da "Şimdi Senkronize Et" gibi bir
manuel tetikleme için kullanılabilir.

BİLİNMEYEN/TEST EDİLEMEYEN (açıkça işaretleniyor): Bu sandbox'ta gerçek
network erişimi yok, bu yüzden GERÇEK bir arka plan job'ının SAATLERCE
çalışıp gerçek yfinance/TEFAS verisiyle beslendiği bir senaryo TEST
EDİLEMEDİ — yalnızca job'ın DOĞRU service metodlarını DOĞRU parametrelerle
çağırdığı, hata izolasyonunun çalıştığı, ve start()/shutdown()'ın
idempotent olduğu doğrulandı (sahte provider ile).

TESTABİLİTE İÇİN EK TASARIM — sınıf-seviyesi instance registry (bu
turda GERÇEK bir sızıntı ölçülerek bulundu — AppTest ile Streamlit
script'inin özel bir yükleme mekanizmasıyla çalıştırılması nedeniyle
normal `import` sonrasında "hangi Container/scheduler instance'ı
aktif" sorgusu GÜVENİLİR ÇALIŞMADI — 15 sahipsiz APScheduler thread'i
birikti). SnapshotScheduler artık oluşturulan HER instance'ı zayıf
referans (weakref) ile bir sınıf-seviyesi registry'e kaydediyor.
`shutdown_all_for_testing()`, Container/import yoluna BAĞIMLI OLMADAN,
process'te o ana kadar oluşturulmuş TÜM scheduler'ları kapatabiliyor.

weakref KULLANIMI bilinçli: registry, scheduler nesnelerinin garbage
collection'ını ENGELLEMEMELİ (güçlü referans tutsaydı, kendisi bir
bellek sızıntısı kaynağı olurdu).
"""

from __future__ import annotations

import weakref
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler

from src.infrastructure.logging_config import get_logger

logger = get_logger(__name__)

_JOB_ID = "risk_snapshot_sync"


class SnapshotScheduler:
    _instances: "weakref.WeakSet[SnapshotScheduler]" = weakref.WeakSet()

    def __init__(
        self,
        portfolio_service: Any,
        risk_service: Any,
        risk_snapshot_repo: Any,
        interval_hours: int = 6,
        lookback_days: int = 252,
        max_portfolios_per_run: int = 100,
    ) -> None:
        self._portfolio_service = portfolio_service
        self._risk_service = risk_service
        self._risk_snapshot_repo = risk_snapshot_repo
        self._interval_hours = interval_hours
        self._lookback_days = lookback_days
        self._max_portfolios = max_portfolios_per_run
        self._scheduler: BackgroundScheduler | None = None
        SnapshotScheduler._instances.add(self)

    @classmethod
    def shutdown_all_for_testing(cls) -> int:
        """
        YALNIZCA TESTLERDE kullanılır — process'te o ana kadar
        oluşturulmuş TÜM SnapshotScheduler instance'larını kapatır.

        NEDEN GEREKLİ (bkz. modül docstring'i): AppTest'in script'i
        çalıştırma mekanizması, normal `import` sonrasında "hangi
        Container/scheduler aktif" sorgusunu güvenilir kılmıyor. Bu
        classmethod, Container/import yoluna HİÇ bağımlı olmadan
        (yalnızca sınıf-seviyesi weakref registry'e bakarak) TÜM
        scheduler'ları bulup kapatabiliyor.

        Returns:
            Kapatılan (çalışıyor durumda bulunan) scheduler sayısı.
        """
        count = 0
        for instance in list(cls._instances):
            if instance.is_running:
                instance.shutdown()
                count += 1
        return count

    def start(self) -> None:
        """
        Idempotent — zaten çalışıyorsa NO-OP (Container.scheduler
        property'sinin @st.cache_resource ile tekil olması ZATEN bunu
        garanti ediyor, ama bu metod KENDİ BAŞINA da güvenli — birden
        fazla çağrı job çoğaltmaz, `replace_existing=True` + sabit
        `id` bunu garanti ediyor).
        """
        if self._scheduler is not None and self._scheduler.running:
            return

        self._scheduler = BackgroundScheduler(daemon=True)
        self._scheduler.add_job(
            self._run_snapshot_job,
            trigger="interval",
            hours=self._interval_hours,
            id=_JOB_ID,
            replace_existing=True,
            coalesce=True,  # kaçırılan çalışmalar BİRİKTİRİLMEZ, tek sefer çalışır
            max_instances=1,  # önceki çalışma bitmemişse YENİSİ BAŞLAMAZ (üst üste binme yok)
        )
        self._scheduler.start()
        logger.info(
            "snapshot_scheduler_started", interval_hours=self._interval_hours,
        )

    def shutdown(self, wait: bool = False) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=wait)
            self._scheduler = None
            logger.info("snapshot_scheduler_shutdown")

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def run_now(self) -> dict[str, int]:
        """
        Zamanlayıcıyı BEKLEMEDEN job'ı hemen çalıştırır — test/manuel
        tetikleme için. Zamanlanmış job'dan BAĞIMSIZ (onu etkilemez).

        Returns:
            {"succeeded": N, "failed": M} — UI'da özet göstermek için.
        """
        return self._run_snapshot_job()

    def _run_snapshot_job(self) -> dict[str, int]:
        succeeded = 0
        failed = 0
        try:
            portfolios = self._portfolio_service.list_portfolios()
        except Exception as exc:
            logger.error("scheduler_list_portfolios_failed", error=str(exc))
            return {"succeeded": 0, "failed": 0}

        for p in portfolios[: self._max_portfolios]:
            portfolio_id = p["id"]
            try:
                full_portfolio = self._portfolio_service.get_portfolio(portfolio_id)
                benchmark_code = full_portfolio.benchmark_code if full_portfolio else None
                self._risk_service.compute_and_persist_snapshot(
                    portfolio_id, self._risk_snapshot_repo,
                    lookback_days=self._lookback_days, benchmark_code=benchmark_code,
                )
                succeeded += 1
                logger.info("scheduler_snapshot_computed", portfolio_id=portfolio_id)
            except Exception as exc:
                # BİR portföyün hatası DİĞERLERİNİ durdurmamalı — bkz.
                # modül docstring'i "hata izolasyonu".
                failed += 1
                logger.warning(
                    "scheduler_snapshot_failed", portfolio_id=portfolio_id, error=str(exc),
                )
                continue

        logger.info("scheduler_run_completed", succeeded=succeeded, failed=failed)
        return {"succeeded": succeeded, "failed": failed}
