"""
Yapılandırılmış (structured) loglama — structlog tabanlı.

DÜZELTME KAYDI (bu turda, önceden var olan bir test çalıştırılarak bulundu):
  İlk taslak `structlog.PrintLoggerFactory()` kullanıyordu — bu,
  Python'ın stdlib `logging` modülünü TAMAMEN BY-PASS EDİP doğrudan
  stdout'a `print()` yapıyor. test_yfinance_adapter.py'daki ÖNCEDEN
  VAR OLAN `test_fetch_ohlcv_produces_no_stdout_noise` testi (docstring:
  "yalnızca logging kullanılmalı") bunu YAKALADI — bu testi bu turda
  ilk kez çalıştırana kadar fark edilmemişti (çünkü bu dosya daha önce
  hiç projenin tam test paketine dahil edilmemişti).

  KÖK NEDEN: structlog, `structlog.configure()` hiç çağrılmamışsa
  KENDİ VARSAYILANINI (PrintLoggerFactory) kullanır — bu, Python
  logging ekosisteminin "kütüphane kodu SESSİZ olmalı, uygulama
  açıkça handler eklemeden hiçbir şey konsola yazılmamalı" prensibiyle
  ÇELİŞİYOR (bkz. Python docs: "Logging in a library").

  DÜZELTME: get_logger() artık LAZY olarak, henüz configure_from_settings()
  çağrılmamışsa structlog'u stdlib `logging` modülüne yönlendiren VE
  kök logger'a yalnızca bir `NullHandler` ekleyen bir "sessiz varsayılan"
  kuruyor. Bu sayede:
    - YFinanceAdapter gibi bileşenler TEK BAŞINA (container/uygulama
      olmadan) kullanıldığında SESSİZ kalır (bu testin beklediği davranış).
    - configure_from_settings() çağrıldığında (container.py başlangıcında)
      GERÇEK handler'lar (console/JSON) eklenir ve loglar görünür olur.

Neden structlog (plain logging değil): portfolio_service.py ve
container.py `logger.info("event_adı", key=value, ...)` şeklinde
structlog'un kwargs-as-structured-fields idiomunu kullanıyor.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured: bool = False
_default_silent_setup_done: bool = False


def _ensure_silent_default() -> None:
    """
    configure_from_settings() HENÜZ ÇAĞRILMAMIŞSA, structlog'u stdlib
    logging'e yönlendirir ve kök logger'a yalnızca NullHandler ekler —
    bu, "kütüphane kodu sessiz olmalı" prensibini garanti eder.

    Idempotent: birden fazla çağrılsa da yalnızca ilk çağrıda etki eder.
    """
    global _default_silent_setup_done
    if _configured or _default_silent_setup_done:
        return

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # configure_from_settings() sonradan EZEBİLMELİ
    )
    _default_silent_setup_done = True


def configure_from_settings(settings: Any | None = None) -> None:
    """
    structlog + stdlib logging'i settings.logging'e göre GÖRÜNÜR şekilde
    yapılandırır — _ensure_silent_default()'ın sessiz varsayılanını
    BİLİNÇLİ OLARAK EZER (uygulama başlangıcında bir kez çağrılmalı,
    container.py bunu zaten yapıyor).

    Idempotent: Container/Streamlit rerun döngüsünde birden fazla
    çağrılsa bile yeniden yapılandırma yapmaz (ilk çağrı kalıcıdır) —
    bu, @st.cache_resource ile singleton container'ın her rerun'da
    logging handler'larını çoğaltmasını engeller.
    """
    global _configured
    if _configured:
        return

    level_name = "INFO"
    fmt = "console"
    overrides: dict[str, str] = {
        "sqlalchemy.engine": "WARNING",
        "apscheduler": "WARNING",
        "yfinance": "WARNING",
    }

    if settings is not None:
        try:
            level_name = settings.logging.level
            fmt = settings.logging.format
            overrides = settings.logging.overrides or overrides
        except AttributeError:
            pass

    level = getattr(logging, level_name.upper(), logging.INFO)

    # _ensure_silent_default()'ın eklediği NullHandler'ı kaldır — artık
    # GERÇEK bir handler ekleniyor.
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, logging.NullHandler):
            root_logger.removeHandler(h)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,  # _ensure_silent_default()'ın basicConfig'ini EZ
    )
    for logger_name, override_level in overrides.items():
        logging.getLogger(logger_name).setLevel(
            getattr(logging, override_level.upper(), logging.WARNING)
        )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        # DÖRDÜNCÜ VE KÖK NEDEN OLAN HATA (bağımsız bir mikro-deneyle
        # KANITLANDI): cache_logger_on_first_use=True, modül seviyesinde
        # BİR KEZ oluşturulan logger proxy'lerini (örn. yfinance_adapter.py
        # ::`logger = get_logger(__name__)`) İLK KULLANIMLARINDAKİ
        # yapılandırmaya KALICI OLARAK bağlıyor — structlog.reset_defaults()
        # + yeniden configure() BİLE bu ÖNCEDEN OLUŞTURULMUŞ proxy'nin
        # davranışını DEĞİŞTİREMİYOR (yalnızca YENİ get_logger() çağrıları
        # etkileniyor). Modüllerin Python'da yalnızca BİR KEZ import
        # edildiği ve logger'ların modül seviyesinde tanımlandığı bu
        # kod tabanında, bu, test-izolasyonunu YAPISAL OLARAK imkansız
        # kılıyordu. Performans kazancı (processor zincirini her log
        # çağrısında yeniden bağlamama) burada güvenilir test
        # izolasyonundan DAHA DÜŞÜK öncelikli — bu yüzden False'a
        # çekildi (maliyeti ölçülebilir değil, kişisel portföy
        # ölçeğinde log hacmi düşük).
        cache_logger_on_first_use=False,
    )

    _configured = True


def get_logger(name: str) -> Any:
    """
    Modül seviyesinde çağrılabilir logger factory.

    configure_from_settings() ÇAĞRILMAMIŞSA (örn. bir adapter tek
    başına, container olmadan kullanılıyorsa), _ensure_silent_default()
    ile SESSİZ bir stdlib-logging temeli kurulur — hiçbir stdout
    gürültüsü ÜRETİLMEZ. configure_from_settings() çağrıldıktan SONRA
    (gerçek uygulama akışı) loglar normal şekilde görünür olur.

    Dönüş tipi bilinçli olarak `Any`: structlog.get_logger() runtime'da
    configure() çağrısındaki wrapper_class'a göre değişen dinamik bir
    proxy döndürür — kesin bir tip iddia etmek yanlış olur.
    """
    _ensure_silent_default()
    return structlog.get_logger(name)


def reset_for_testing() -> None:
    """
    YALNIZCA TESTLERDE kullanılır — _configured/_default_silent_setup_done
    modül-seviyesi global'lerini sıfırlar.

    NEDEN GEREKLİ (bu turda GERÇEKTEN bulunan bir sorunla kanıtlandı):
    configure_from_settings()'in "idempotent, uygulama ömrü boyunca bir
    kez çağrılır" tasarımı (Streamlit rerun'larında handler çoğalmasını
    önlemek için BİLİNÇLİ bir karar) — PRODUCTION'da doğru, ama TEST
    SÜİTİNDE yanlış: bir test dosyası (örn. container.py kuran bir
    AppTest testi) configure_from_settings()'i çağırıp `_configured=True`
    yaptığında, bu durum pytest sürecinin SONUNA kadar kalıcı oluyor —
    sonraki test dosyaları (örn. test_yfinance_adapter.py'nin "sessiz
    varsayılan" bekleyen testi) YANLIŞLIKLA "configure edilmiş" (gürültülü)
    durumu miras alıyor. Bu, tam proje test paketi (`pytest tests/`)
    çalıştırılarak GERÇEKTEN tespit edildi — dosya tek başına
    çalıştırıldığında bu sorun görünmüyordu (sinsi bir test-order
    bağımlılığı).

    Kullanım (conftest.py'da autouse fixture):
        @pytest.fixture(autouse=True)
        def _reset_logging():
            from src.infrastructure.logging_config import reset_for_testing
            reset_for_testing()
            yield
            reset_for_testing()
    """
    global _configured, _default_silent_setup_done
    _configured = False
    _default_silent_setup_done = False
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # KRİTİK — İKİNCİ BİR HATA burada bulundu: structlog.configure()
    # PROCESS GENELİNDE global bir singleton state tutuyor, stdlib
    # `logging` handler'larından TAMAMEN BAĞIMSIZ. `PrintLoggerFactory`
    # doğrudan sys.stdout'a yazıyor — logging.getLogger() üzerinden HİÇ
    # geçmiyor. Bu yüzden yukarıdaki "root_logger handler'larını temizle"
    # adımı BU DURUMU ETKİLEMİYORDU — configure_from_settings()'in
    # kurduğu ConsoleRenderer/PrintLoggerFactory yapılandırması sonraki
    # testlere SIZMAYA devam ediyordu (tam proje test paketi
    # çalıştırılarak, ikinci bir gerçek hata olarak tespit edildi).
    # structlog.reset_defaults() bu global singleton'ı GERÇEKTEN sıfırlar.
    # structlog.reset_defaults() PROCESS-GLOBAL singleton'ı sıfırlar —
    # AMA "sıfırlamak", structlog'un KENDİ fabrika varsayılanına (ki bu
    # da PrintLoggerFactory!) dönmek anlamına geliyor, BENİM sessiz
    # varsayılanıma değil. Bu ÜÇÜNCÜ bir hataydı (debug ile doğrulandı:
    # reset_defaults() sonrası structlog.get_config()['logger_factory']
    # HÂLÂ PrintLoggerFactory çıktı). Bu yüzden reset'ten HEMEN SONRA
    # sessiz yapılandırmayı YENİDEN kurmak ZORUNLU — yalnızca flag'leri
    # sıfırlayıp structlog'u kendi haline bırakmak yetersiz.
    structlog.reset_defaults()
    _ensure_silent_default()
