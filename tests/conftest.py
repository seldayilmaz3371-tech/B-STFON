"""
Kök conftest.py — tüm test paketi için ortak fixture'lar.

_reset_logging_state: logging_config.py'daki modül-seviyesi global'ler
  (_configured, _default_silent_setup_done) GERÇEK bir test-izolasyonu
  hatasına yol açtığı için eklendi (bkz. logging_config.py::
  reset_for_testing() docstring'i — tam proje test paketi çalıştırılarak
  keşfedildi: bir test dosyasının configure_from_settings() çağırması,
  SONRAKİ test dosyalarının "sessiz varsayılan" beklentisini bozuyordu).

  autouse=True: HER testten önce/sonra otomatik çalışır — hiçbir test
  dosyasının bunu elle import edip çağırmasına gerek yok, bu da yeni
  yazılacak testlerin bu tuzağa YENİDEN düşmesini yapısal olarak önler.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_logging_state():
    from src.infrastructure.logging_config import reset_for_testing

    reset_for_testing()
    yield
    reset_for_testing()


@pytest.fixture(autouse=True)
def _shutdown_all_schedulers():
    """
    Genel güvenlik ağı — test_app_integration.py'daki DAHA SPESİFİK
    cache-clearing fixture'ına ek olarak (Streamlit cache_resource'a
    dokunmayan test dosyalarında da SnapshotScheduler kullanılırsa
    diye). SnapshotScheduler.shutdown_all_for_testing() Container/
    import yoluna bağımlı DEĞİL (sınıf-seviyesi weakref registry) —
    bu yüzden HER test dosyasında güvenle çalışır, ekstra maliyeti
    yoktur (registry boşsa no-op).
    """
    from src.infrastructure.scheduler.snapshot_scheduler import SnapshotScheduler

    yield
    SnapshotScheduler.shutdown_all_for_testing()
