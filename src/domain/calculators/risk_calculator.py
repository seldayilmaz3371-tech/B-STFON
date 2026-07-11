"""
Risk metrikleri hesaplayıcı — Domain katmanında ama numpy/scipy bağımlılığı
BİLİNÇLİ OLARAK İZİN VERİLİYOR (design doc'un açık notu: "Risk hesaplamaları
float64 kullanır, istatistiksel kesinlik yeterli").

Bu, cost_basis_calculator.py'daki KATI Decimal disiplininden BİLİNÇLİ bir
sapmadır — finansal muhasebe (realized PnL, cost basis) yasal/vergisel
doğruluk gerektirir ve Decimal zorunludur; risk metrikleri (Sharpe, VaR)
istatistiksel tahminlerdir, zaten ölçüm belirsizliği taşırlar — float64'ün
~15-17 anlamlı hane hassasiyeti bu bağlamda hiçbir zaman darboğaz değildir.
numpy/scipy'nin vectorized operasyonları da yalnızca float64 ile çalışır.

API sözleşmesi — BIST_TEFAS_Master_Design_Document.md Bölüm 1.4
"RiskCalculator" interface contract'ından birebir alındı.
GD-007 ile doğrulanan metodlar: calculate_annualized_volatility,
calculate_sharpe_ratio, calculate_var (HISTORICAL).
GD-007'de YER ALMAYAN (yalnızca formül-seviyesi/ders kitabı doğrulaması
yapılan, golden dataset İLE DOĞRULANMAYAN): Sortino, drawdown, CVaR,
PARAMETRIC VaR, beta, information ratio — bu ayrım testte açıkça işaretli.

AÇIKÇA İŞARETLENEN BELİRSİZLİK — `mar` parametresi (calculate_sortino_ratio):
  Design doc `mar: float = 0.0` diyor ama GÜNLÜK mü YILLIK mı olduğunu
  belirtmiyor. `risk_free_rate_annual` parametresinin adında "_annual"
  açıkça var, `mar`'da YOK. Bu isimlendirme farkına dayanarak mar'ı
  GÜNLÜK eşik olarak yorumluyorum — ama bu bir VARSAYIM, kesin bir
  doküman referansı değil. Yanlışsa (mar aslında yıllık olmalıydıysa)
  çağıran taraf mar=0.10/252 gibi kendi dönüştürmesini yapmalı; bu
  belirsizlik netleşene kadar docstring'de vurgulanıyor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats

from src.domain.enums.var_method import VaRMethod
from src.domain.exceptions.domain_exceptions import (
    CalculationError,
    InsufficientDataError,
)

# Float eşitlik karşılaştırması (`== 0.0`) GÜVENİLMEZ: matematiksel olarak
# sabit bir seri bile floating-point çıkarma/toplama sırasında ~1e-18
# mertebesinde "gürültü" biriktirebilir (IEEE 754 yuvarlama hatası).
# Bu, GERÇEKTEN test edilerek bulundu: np.std([0.01]*10 - sabit_değer, ddof=1)
# tam olarak 0.0 DEĞİL, 1.8e-18 döndü — `== 0.0` kontrolü bu durumu
# YAKALAYAMAZ ve sıfıra bölme koruması sessizce devre dışı kalırdı.
# 1e-12 eşiği: gerçek (anlamlı) volatilite değerlerinden (tipik olarak
# 1e-4 - 1e-1 aralığı) kesin olarak ayrışırken, float gürültüsünü güvenle yakalar.
_EPSILON = 1e-12


@dataclass(frozen=True)
class DrawdownResult:
    max_drawdown: float  # 0 ile -1 arasında
    peak_idx: int
    trough_idx: int


class RiskCalculator:
    """
    Constructor sözleşmesi — container.py'den ÇIKARILDI (varsayım değil):
        RiskCalculator(trading_days=..., min_data_points=...)
    """

    def __init__(self, trading_days: int = 252, min_data_points: int = 30) -> None:
        self._trading_days = trading_days
        self._min_data_points = min_data_points

    def _check_min_data(self, arr: np.ndarray, metric: str) -> None:
        n = len(arr)
        if n < self._min_data_points:
            raise InsufficientDataError(
                required=self._min_data_points, available=n, metric=metric
            )

    def calculate_annualized_volatility(
        self, daily_returns: np.ndarray, trading_days: int | None = None
    ) -> float:
        td = trading_days or self._trading_days
        self._check_min_data(daily_returns, "annualized_volatility")
        daily_std = float(np.std(daily_returns, ddof=1))  # sample std (GD-007: n-1)
        return daily_std * float(np.sqrt(td))

    def calculate_sharpe_ratio(
        self,
        daily_returns: np.ndarray,
        risk_free_rate_annual: float,
        trading_days: int | None = None,
    ) -> float:
        """
        Günlük risk-free dönüşümü GEOMETRİK (bileşik) — GD-007'de
        doğrulandı: daily_rfr = (1+annual)^(1/252) - 1, basit bölme
        (annual/252) DEĞİL. Bu ayrım GD-007'nin kendisinde açıkça
        gösteriliyor (0.001330, 0.40/252=0.001587 DEĞİL).
        """
        td = trading_days or self._trading_days
        self._check_min_data(daily_returns, "sharpe_ratio")

        daily_rfr = (1.0 + risk_free_rate_annual) ** (1.0 / td) - 1.0
        excess = daily_returns - daily_rfr
        excess_std = float(np.std(excess, ddof=1))

        if excess_std < _EPSILON:
            raise CalculationError(
                "Excess return standart sapması sıfır — Sharpe oranı "
                "tanımsız (sabit getiri serisi, sıfıra bölme riski)."
            )
        return float(np.mean(excess)) / excess_std * float(np.sqrt(td))

    def calculate_sortino_ratio(
        self,
        daily_returns: np.ndarray,
        risk_free_rate_annual: float,
        trading_days: int | None = None,
        mar: float = 0.0,
    ) -> float:
        """mar GÜNLÜK eşik olarak yorumlanıyor — bkz. modül docstring'i."""
        td = trading_days or self._trading_days
        self._check_min_data(daily_returns, "sortino_ratio")

        daily_rfr = (1.0 + risk_free_rate_annual) ** (1.0 / td) - 1.0
        downside_diff = np.minimum(daily_returns - mar, 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside_diff ** 2)))

        if downside_deviation < _EPSILON:
            raise CalculationError(
                "Downside deviation sıfır — hiçbir getiri MAR'ın altında "
                "değil, Sortino oranı tanımsız (sıfıra bölme riski)."
            )
        numerator = float(np.mean(daily_returns)) - daily_rfr
        return float(numerator / downside_deviation * float(np.sqrt(td)))

    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> DrawdownResult:
        """
        VARSAYIM (açıkça işaretleniyor): cumulative_returns bir "wealth
        index" (1.0'dan başlayan çarpımsal büyüme serisi) olarak
        yorumlanıyor — design doc bunu 0'dan mı 1'den mi başladığını
        belirtmiyor. 1.0 tabanlı yorum, drawdown = value/peak - 1
        formülünün doğrudan uygulanabilmesini sağlıyor ve endüstri
        standardı (equity curve) ile örtüşüyor.
        """
        if len(cumulative_returns) == 0:
            raise InsufficientDataError(required=1, available=0, metric="max_drawdown")

        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns / running_max - 1.0

        trough_idx = int(np.argmin(drawdown))
        peak_idx = int(np.argmax(cumulative_returns[: trough_idx + 1]))

        return DrawdownResult(
            max_drawdown=float(drawdown[trough_idx]),
            peak_idx=peak_idx,
            trough_idx=trough_idx,
        )

    def calculate_current_drawdown(self, cumulative_returns: np.ndarray) -> float:
        if len(cumulative_returns) == 0:
            raise InsufficientDataError(required=1, available=0, metric="current_drawdown")
        running_max = np.maximum.accumulate(cumulative_returns)
        return float(cumulative_returns[-1] / running_max[-1] - 1.0)

    def calculate_var(
        self,
        daily_returns: np.ndarray,
        confidence: float = 0.95,
        method: VaRMethod = VaRMethod.HISTORICAL,
    ) -> float:
        """
        Dönüş DAİMA pozitif (kayıp büyüklüğü) — design doc: "Returns:
        pozitif değer (kayıp miktarı, portföy değerinin fraksiyonu)".
        GD-007 ile doğrulandı (HISTORICAL, %95 güven, 10 günlük seri).
        """
        self._check_min_data(daily_returns, "var")
        if not (0.0 < confidence < 1.0):
            raise CalculationError(f"confidence (0,1) aralığında olmalı: {confidence}")

        if method is VaRMethod.HISTORICAL:
            percentile = (1.0 - confidence) * 100.0
            var_value = float(np.percentile(daily_returns, percentile))
        elif method is VaRMethod.PARAMETRIC:
            mean = float(np.mean(daily_returns))
            std = float(np.std(daily_returns, ddof=1))
            z = float(scipy.stats.norm.ppf(1.0 - confidence))
            var_value = mean + z * std
        elif method is VaRMethod.MONTECARLO:
            var_value = self._monte_carlo_var(daily_returns, confidence)
        else:  # pragma: no cover — enum kapalı küme
            raise CalculationError(f"Tanınmayan VaR yöntemi: {method}")

        return -var_value  # kayıp = pozitif sayı

    def _monte_carlo_var(
        self, daily_returns: np.ndarray, confidence: float, n_simulations: int = 10_000,
    ) -> float:
        """
        Monte Carlo VaR — BOOTSTRAP RESAMPLING yöntemiyle (parametrik
        dağılım varsayımı YOK).

        KARAR GEREKÇESİ (parametrik/normal MC yerine bootstrap):
        BIST hisse getirileri normal dağılıma UYMAZ (fat-tail, skewness
        — bu proje boyunca defalarca vurgulandı, RiskCalculator'ın
        PARAMETRIC yönteminin de aynı sınırlaması var, ayrı bir
        seçenek olarak zaten sunuluyor). Getirilerin bir normal/t
        dağılımına "fit" edilip o dağılımdan örneklenmesi (klasik
        parametrik Monte Carlo), gerçek BIST getiri dağılımının
        kuyruk davranışını YANLIŞ modelleme riski taşır. Bootstrap
        (gözlemlenen GERÇEK getirilerden, yerine koyarak rastgele
        örnekleme) hiçbir dağılım varsayımı yapmıyor — yalnızca
        "geçmişte gözlemlenen getiri dağılımının gelecekte de
        geçerli olacağı" varsayımını taşıyor (bu, HISTORICAL yöntemle
        PAYLAŞILAN, kaçınılmaz bir varsayım).

        Rastgelelik: np.random.default_rng ile SABİT SEED YOK —
        her çağrıda farklı sonuç üretir (gerçek Monte Carlo davranışı).
        Bu, test edilebilirliği ZORLAŞTIRIR (bkz. test dosyasında
        toleranslı/istatistiksel testler, tam sayı eşitliği DEĞİL).

        n_simulations=10.000: Hız/hassasiyet dengesi — standart hata
        1/sqrt(n) ile azalır, 10k simülasyon günlük VaR için makul bir
        güven aralığı sağlar (kişisel portföy ölçeğinde <1sn hesaplama).
        """
        rng = np.random.default_rng()
        simulated_returns = rng.choice(daily_returns, size=n_simulations, replace=True)
        percentile = (1.0 - confidence) * 100.0
        return float(np.percentile(simulated_returns, percentile))

    def calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """
        Calmar Ratio = Yıllıklandırılmış Getiri / |Maks. Düşüş|.

        max_drawdown NEGATİF bir değer olarak bekleniyor (RiskCalculator.
        calculate_max_drawdown().max_drawdown ile TUTARLI — 0 ile -1
        arasında). Mutlak değeri alınıyor çünkü Calmar oranı geleneksel
        olarak pozitif bir sayı olarak raporlanır (yüksek Calmar = iyi
        risk-ayarlı getiri).
        """
        if max_drawdown == 0.0:
            raise CalculationError(
                "max_drawdown sıfır — Calmar oranı tanımsız (hiç düşüş "
                "yaşanmamış bir seri, sıfıra bölme riski)."
            )
        return annualized_return / abs(max_drawdown)

    def calculate_concentration_metrics(
        self, position_weights: np.ndarray,
    ) -> tuple[float, float]:
        """
        Portföy konsantrasyon (çeşitlendirme) metrikleri.

        Returns:
            (herfindahl_index, top5_concentration)

        herfindahl_index: Σ(w_i²) — 0 ile 1 arasında. 1/N'e ne kadar
          yakınsa o kadar EŞİT DAĞILMIŞ (N pozisyona eşit ağırlıkla
          dağılmış bir portföyün HHI'ı = 1/N). 1.0'a yakınsa TEK
          pozisyona yoğunlaşmış demektir.
        top5_concentration: En büyük 5 pozisyonun toplam ağırlığı
          (0 ile 1 arasında). Portföyde 5'ten az pozisyon varsa TÜM
          pozisyonların toplamı (=1.0) döner — bu bir hata DEĞİL,
          "portföyün tamamı ilk 5'te" anlamına geliyor.

        position_weights: Her pozisyonun (current_value / total_value)
          oranı — TOPLAMLARI 1.0'A YAKIN olmalı (küçük yuvarlama
          farkları kabul edilir, ama %5'ten fazla sapma BusinessRuleError
          fırlatır — çağıranın ağırlıkları YANLIŞ hesapladığının işareti).
        """
        if len(position_weights) == 0:
            raise InsufficientDataError(
                required=1, available=0, metric="concentration_metrics",
            )
        total_weight = float(np.sum(position_weights))
        if abs(total_weight - 1.0) > 0.05:
            raise CalculationError(
                f"position_weights toplamı 1.0'a yakın olmalı, alınan: "
                f"{total_weight:.4f} — ağırlıklar yanlış hesaplanmış olabilir."
            )

        hhi = float(np.sum(position_weights ** 2))
        top5 = float(np.sum(np.sort(position_weights)[::-1][:5]))
        return hhi, min(top5, 1.0)  # yuvarlama nedeniyle 1.0'ı hafif aşarsa kırp

    def calculate_cvar(self, daily_returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Expected Shortfall — yalnızca HISTORICAL yöntem (design doc
        calculate_cvar için method parametresi tanımlamıyor).
        """
        self._check_min_data(daily_returns, "cvar")
        percentile = (1.0 - confidence) * 100.0
        threshold = np.percentile(daily_returns, percentile)
        tail_losses = daily_returns[daily_returns <= threshold]

        if len(tail_losses) == 0:
            raise CalculationError(
                "VaR eşiğinin altında/eşit hiçbir gözlem yok — CVaR "
                "hesaplanamıyor (aşırı küçük örneklem veya confidence değeri)."
            )
        return -float(np.mean(tail_losses))

    def calculate_beta(
        self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> tuple[float, float, float]:
        if len(portfolio_returns) != len(benchmark_returns):
            raise CalculationError(
                f"portfolio_returns ({len(portfolio_returns)}) ve "
                f"benchmark_returns ({len(benchmark_returns)}) uzunlukları eşit olmalı."
            )
        self._check_min_data(portfolio_returns, "beta")

        covariance = float(np.cov(portfolio_returns, benchmark_returns, ddof=1)[0, 1])
        benchmark_variance = float(np.var(benchmark_returns, ddof=1))

        if abs(benchmark_variance) < _EPSILON:
            raise CalculationError(
                "Benchmark varyansı sıfır — beta tanımsız (benchmark "
                "sabit getiri serisi, sıfıra bölme riski)."
            )
        beta = covariance / benchmark_variance
        alpha_daily = float(np.mean(portfolio_returns)) - beta * float(np.mean(benchmark_returns))
        correlation = float(np.corrcoef(portfolio_returns, benchmark_returns)[0, 1])
        r_squared = correlation ** 2
        return beta, alpha_daily, r_squared

    def calculate_capture_ratios(
        self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> tuple[float, float]:
        """
        Up/Down Capture Ratio — standart bir performans metriği (design
        doc'ta yalnızca isim/tanım var, formül YOK — bu, endüstri
        standardı tanımıyla implemente edildi, icat EDİLMEDİ):

          Up Capture   = Π(1+r_p) için benchmark>0 günler / Π(1+r_b) aynı günler
          Down Capture = Π(1+r_p) için benchmark<0 günler / Π(1+r_b) aynı günler

        Geometrik (bileşik) linking kullanılıyor, basit ortalama DEĞİL —
        bu, ReturnCalculator.calculate_twrr() ile TUTARLI bir metodoloji
        (getiri serilerini birleştirirken bu projede zaten kurulan
        standart, bkz. GD-006).

        Returns:
            (up_capture, down_capture) — genellikle yüzde olarak
            yorumlanır (1.0 = %100, benchmark ile birebir aynı performans).
            Benchmark'ın hiç pozitif (veya hiç negatif) gün içermediği
            durumda ilgili oran `float('nan')` döner (tanımsız, sıfıra
            bölme yerine NaN — çağıran taraf bunu açıkça ele almalı).
        """
        if len(portfolio_returns) != len(benchmark_returns):
            raise CalculationError(
                "portfolio_returns ve benchmark_returns uzunlukları eşit olmalı."
            )
        self._check_min_data(portfolio_returns, "capture_ratios")

        up_mask = benchmark_returns > 0
        down_mask = benchmark_returns < 0

        def _geometric_link(returns: np.ndarray) -> float:
            return float(np.prod(1.0 + returns) - 1.0)

        if up_mask.any():
            up_capture = _geometric_link(portfolio_returns[up_mask]) / _geometric_link(
                benchmark_returns[up_mask]
            )
        else:
            up_capture = float("nan")

        if down_mask.any():
            down_capture = _geometric_link(portfolio_returns[down_mask]) / _geometric_link(
                benchmark_returns[down_mask]
            )
        else:
            down_capture = float("nan")

        return up_capture, down_capture

    def calculate_information_ratio(
        self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray
    ) -> float:
        """
        Design doc'ta trading_days parametresi YOK — bu yüzden çıktı
        girdi frekansında (günlükse günlük) bırakılıyor, yıllıklandırma
        UYGULANMIYOR. Çağıran taraf istiyorsa kendi × sqrt(trading_days)
        yapmalı. Bu, sessizce bir varsayım eklemektense dokümante
        edilen imzaya sadık kalmayı tercih etme kararı.
        """
        if len(portfolio_returns) != len(benchmark_returns):
            raise CalculationError(
                "portfolio_returns ve benchmark_returns uzunlukları eşit olmalı."
            )
        self._check_min_data(portfolio_returns, "information_ratio")

        excess = portfolio_returns - benchmark_returns
        tracking_error = float(np.std(excess, ddof=1))
        if tracking_error < _EPSILON:
            raise CalculationError(
                "Tracking error sıfır — information ratio tanımsız."
            )
        return float(np.mean(excess)) / tracking_error
