"""
Piyasa verisi sağlayıcı soyutlama katmanı.

Bu modül, sistemdeki TÜM dış veri sağlayıcılarının (yfinance, TEFAS,
IS Yatirim, gelecekteki başka kaynaklar) uyması gereken sözleşmeyi tanımlar.

Adapter Pattern Gerekçesi:
  yfinance bugün çalışıyor, yarın API'sini değiştirebilir veya kırılabilir
  (scraping-based bir kütüphane olarak güvenilirlik riski taşır). Domain ve
  Service katmanları asla yfinance'a doğrudan bağımlı olmamalı — yalnızca
  bu interface'e bağımlı olmalılar. Provider değişikliği yalnızca yeni bir
  Adapter sınıfı yazmayı gerektirir; business logic dokunulmaz kalır.

Bu dosya src/domain/interfaces/data_providers.py içindeki
MarketDataProvider Protocol'ünden farklıdır:
  - domain/interfaces/data_providers.py → Protocol (duck typing, structural)
  - Bu dosya (infrastructure)           → ABC (nominal, explicit subclassing)
  Bu iki seviyeli soyutlama bilinçlidir: Protocol domain'in dış dünyaya
  bakan sözleşmesidir; ABC infrastructure içi somut implementasyonların
  ortak iskeletini zorunlu kılar (timeframe/period gibi infrastructure'a
  özgü parametreler Protocol'de yer almaz, yalnızca burada bulunur).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd


class MarketDataProvider(ABC):
    """
    Tüm piyasa verisi sağlayıcı adapter'larının uyması gereken soyut taban sınıf.

    Implementasyonlar: YFinanceAdapter, TEFASAdapter (gelecek),
                        ISYatirimAdapter (gelecek), MockProvider (test).

    Sorumluluk sınırı (kritik):
      Bu sınıfın implementasyonları yalnızca "temiz veri" sağlamaktan
      sorumludur. Hesaplama (RSI, ATR, RVOL vb.) bu katmanda YAPILMAZ —
      o sorumluluk src/domain/calculators/ katmanına aittir. Provider
      katmanı ile Domain katmanı arasında hiçbir çağrı yönü yoktur;
      Service katmanı ikisini orchestrate eder.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: date | datetime | None = None,
        end_date: date | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Belirtilen sembol için OHLCV (Open/High/Low/Close/Volume) verisi çek.

        Dönüş garantisi:
          - DataFrame her zaman Aşama 1'deki normalize_and_validate()
            pipeline'ından geçirilmiş, temizlenmiş veridir.
          - Hatalı barlar silinmez — NaN'a çevrilir (index bütünlüğü korunur).
          - Sütunlar: Open, High, Low, Close, Volume (en az).

        Args:
            symbol: Normalize edilmiş sembol (ör: "THYAO", "GARAN").
                    Borsa suffix'i (.IS gibi) adapter içinde eklenir,
                    çağıran taraf yalnızca çıplak sembolü verir.
            timeframe: Zaman dilimi (ör: "1d", "1h", "15m", "4h").
            start_date: Başlangıç tarihi (dahil). None ise provider'ın
                        varsayılan lookback'i kullanılır.
            end_date: Bitiş tarihi (dahil). None ise bugüne kadar.

        Returns:
            pd.DataFrame: DatetimeIndex'li, validate edilmiş OHLCV verisi.

        Raises:
            SymbolNotFoundError: Sembol provider'da bulunamadı.
            NoDataError: İstenen aralıkta veri yok.
            ProviderUnavailableError: Provider'a ulaşılamadı (tüm retry'lar tükendi).
            DataValidationError: Çekilen veri Aşama 1 doğrulamasını geçemedi.
        """
        raise NotImplementedError

    @abstractmethod
    def get_provider_name(self) -> str:
        """Provider tanımlayıcı adı (loglama ve hata mesajları için)."""
        raise NotImplementedError
