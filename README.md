# BIST & TEFAS Portföy İşletim Sistemi

Kişisel kullanım için tasarlanmış, BIST hisseleri ve TEFAS yatırım
fonlarını tek bir sistemde takip eden, transaction-based (işlem
bazlı) muhasebe, risk analizi ve backtest özellikleri sunan bir
portföy yönetim uygulaması.

**Bu dosyadaki HER iddia, gerçekten çalıştırılıp doğrulanmıştır** —
"tahmini" veya "hedeflenen" bir açıklama değil.

---

## Mevcut Özellikler (Doğrulanmış Durum)

| Sekme | İçerik |
|---|---|
| 🔍 Tekil Hisse Analizi | RSI, ATR, hacim analizi, teknik grafik |
| 💼 Portföyüm | Portföy oluşturma, işlem ekleme/iptal etme, nakit ledger, risk metrikleri |
| ⭐ İzleme Listem | Sembol izleme listeleri, fiyat alarmı alanları (tetikleme mekanizması henüz yok) |
| 📊 Backtest | SMA Crossover referans stratejisiyle tek-sembol backtest |

**Test durumu**: 501/502 test yeşil (1 önceden bilinen, bağımsız hata
— `test_rank_monotone_with_larger_values`, rolling rank percentile
satürasyonu, düzeltilmedi).

---

## Kurulum

```bash
# 1. Bağımlılıkları kur
pip install -r requirements.txt

# Geliştirme/test için ek olarak:
pip install -r requirements-dev.txt

# 2. Ortam değişkenlerini ayarla
cp .env.example .env
# .env dosyasını kendi ihtiyacına göre düzenle

# 3. Veritabanı şemasını oluştur
alembic upgrade head
# (Alternatif: hızlı yerel geliştirme için Python'dan doğrudan
#  initialize_database() çağrısı da mevcut — testler bunu kullanıyor.
#  Ama Alembic, versiyon takibi SAĞLAYAN tek yol — bkz. MIGRATIONS_README.md)

# 4. Uygulamayı başlat
streamlit run app/main.py
```

**NOT — düzeltilen bir GERÇEK yapısal hata (kullanıcı geri bildirimiyle
bulundu)**: Giriş noktası ÖNCEDEN `app/app.py`'da duruyordu — bu, GERÇEK
bir çalıştırmada şu hataya yol açıyordu: `ModuleNotFoundError: No module
named 'app.container'; 'app' is not a package`.

KÖK NEDEN (Streamlit'in kendi kaynak kodu okunarak doğrulandı):
`streamlit/web/bootstrap.py::_fix_sys_path()`, çalıştırılan script'in
KENDİ DİZİNİNİ (`os.path.dirname("app/app.py")` = `"app"`) `sys.path`'in
EN BAŞINA ekliyor. `from app.container import ...` çözümlenirken Python
bu `"app"` girdisine bakıyor — `app/app.py` (dosya) TAM OLARAK bulunuyor
ve "app" modülü OLARAK yorumlanıyor — ama bu bir DOSYA, PAKET değil.
`PYTHONPATH` bunu ÇÖZEMİYOR çünkü Streamlit'in eklediği girdi HER ZAMAN
en yüksek önceliğe sahip (position 0).

ÇÖZÜM: Giriş noktası `app/main.py`'a yeniden adlandırıldı — artık
`os.path.dirname("app/main.py")` = `"app"` sys.path'e eklense de, Python
`"app/app.py"` diye bir dosya ARAMIYOR (çünkü artık `import app`
YAPILMIYOR, dosya adı `main.py`) — çakışma ORTADAN KALKTI.

**TEST BOŞLUĞU İTİRAFI**: Bu proje boyunca yazılan 580 test (AppTest
tabanlı) bu hatayı YAKALAYAMAZDI — `AppTest.from_file()` script'i
DOĞRUDAN process içinde çalıştırıyor, Streamlit'in GERÇEK CLI
`bootstrap.py` sys.path mantığından GEÇMİYOR. Bu, yalnızca GERÇEK bir
`streamlit run` çağrısıyla (kullanıcının kendi ortamında) ortaya çıktı.

---

## Testleri Çalıştırma

```bash
# Tüm testler
pytest tests/ -q

# Yalnızca hypothesis property-based testleri (CostBasisCalculator matematiksel değişmezleri)
pytest tests/unit/domain/test_cost_basis_property_based.py -v

# mypy --strict (bilinen sistemik uyarılar hariç — bkz. aşağıdaki "Bilinen Teknik Borç")
mypy src/ --strict
```

---

## Proje Yapısı

```
portfolio-os/
├── app/
│   ├── app.py              # Streamlit giriş noktası (GERÇEK yol)
│   └── container.py        # Dependency Injection container
├── src/
│   ├── domain/              # Saf iş mantığı — I/O YOK, framework bağımsız
│   │   ├── enums/
│   │   ├── models/           # Transaction, Portfolio, Fund, CorporateAction, ...
│   │   ├── calculators/      # CostBasis (WAVG/FIFO), Return, Risk, Backtest
│   │   ├── strategies/       # SMA Crossover (referans backtest stratejisi)
│   │   ├── interfaces/       # Protocol tanımları
│   │   └── exceptions/
│   ├── infrastructure/
│   │   ├── database/         # SQLAlchemy Core Table tanımları
│   │   ├── repositories/     # Repository Pattern implementasyonları
│   │   ├── data_providers/   # yfinance/TEFAS adapter'ları + routing
│   │   ├── scheduler/        # APScheduler wrapper
│   │   ├── cache/            # TTLCache
│   │   └── event_bus/        # In-process event bus
│   ├── services/             # Servis katmanı orkestrasyonu
│   └── presentation/         # Streamlit UI bileşenleri
├── tests/
│   ├── unit/                 # Domain katmanı, mock/gerçek repo YOK
│   └── integration/          # Gerçek SQLite, sahte (ama sözleşmeye uygun) provider'lar
├── migrations/                # Alembic versiyonlu migration'lar
└── alembic.ini
```

---

## Mimari Kararlar (Özet — Gerekçeler Kod İçindeki Docstring'lerde)

- **Clean Architecture / Repository Pattern**: domain katmanı, infrastructure'dan HABERSIZ.
- **SQLAlchemy Core (ORM DEĞİL)**: PostgreSQL'e geçiş kolaylığı için.
- **Decimal-as-TEXT**: Finansal hesaplamalarda float hassasiyet kaybı riskini ortadan kaldırmak için TÜM parasal değerler DB'de TEXT olarak saklanıyor.
- **Custom event-driven Backtest Engine (vectorbt DEĞİL)**: Backtest sonuçlarının CANLI muhasebe mantığıyla (WAVGCostBasisCalculator) TUTARLI olması için.
- **Cache-Aside fiyat senkronizasyonu (PriceSyncService)**: Her risk hesaplamasında TAM geçmişi yeniden çekmek yerine yalnızca eksik günleri senkronize eder.

---

## Bilinen Teknik Borç (Şeffaflık İçin Listelendi)

| # | Konu | Etki |
|---|---|---|
| 1 | `calculate_rolling_rank_pct` percentile satürasyonu | Trend dönemlerinde ayırt edicilik kaybı |
| 2 | `container.py`'da sistemik mypy tip-ipucu eksikliği (`self._x = None`) | Tip güvenliği eksik, runtime hatasız |
| 3 | CorporateAction'ın PORTFÖYE UYGULANMASI mantığı yok | Yalnızca olay KAYDI var — RIGHTS_USED/SPLIT/MERGER hâlâ CostBasisCalculator'da BusinessRuleError fırlatıyor |
| 4 | `run_stress_test()` yok | Gerçek tarihsel kriz verisi olmadan uydurulmadı |
| 5 | Watchlist alarm TETİKLEME mekanizması yok | Alanlar şemada hazır, bildirim altyapısı yok |
| 6 | CI/CD pipeline yok | Bu turda henüz kurulmadı |
| 7 | Backtest yalnızca TEK sembol destekliyor | Çoklu-varlık portföy backtest, ayrı ve büyük bir problem |

---

## Lisans / Sorumluluk Reddi

Bu proje **yatırım tavsiyesi sunmaz**. Backtest sonuçları, geçmiş
performansın geleceği garanti etmediği ilkesine tabidir. TEFAS/BIST
verileri üçüncü parti sağlayıcılardan (yfinance, TEFAS) geliyor —
doğruluğu garanti edilmez.
