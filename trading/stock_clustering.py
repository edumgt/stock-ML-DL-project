"""
주식 군집화 모듈 (Stock Clustering)
=====================================
여러 종목의 주가 특성(수익률, 변동성, 기술적 지표 등)을 기반으로
K-Means / 계층적 군집화를 수행하고, 각 군집의 특성을 해석합니다.

주요 클래스:
    StockClusterer   – 군집화 실행 및 결과 반환
    ClusterAnalyzer  – 군집별 특성 해석 및 레이블링

사용 예시::

    from trading.naver_crawler import NaverFinanceCrawler
    from trading.stock_clustering import StockClusterer

    crawler  = NaverFinanceCrawler()
    tickers  = ["005930", "000660", "035420", "051910", "068270"]
    dfs      = {t: crawler.get_daily_ohlcv(t, pages=5) for t in tickers}

    clusterer = StockClusterer(n_clusters=3)
    result    = clusterer.fit(dfs)
    print(result.summary_df)

필요 패키지:
    pip install scikit-learn pandas numpy scipy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ---------------------------------------------------------------------------
# 결과 데이터 모델
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """군집화 결과"""
    n_clusters: int
    method: str
    labels: dict[str, int]           # {ticker: cluster_id}
    feature_df: pd.DataFrame         # 종목별 특성 행렬
    summary_df: pd.DataFrame         # 군집별 통계 요약
    silhouette: float
    cluster_names: dict[int, str]    # {cluster_id: 설명 레이블}
    pca_coords: Optional[pd.DataFrame] = None  # 2D PCA 좌표


# ---------------------------------------------------------------------------
# 특성 추출
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "ann_return",     # 연환산 수익률
    "ann_vol",        # 연환산 변동성
    "sharpe",         # 샤프 비율
    "max_dd",         # 최대 낙폭
    "rsi_mean",       # 평균 RSI
    "momentum_20",    # 20일 모멘텀
    "vol_ratio",      # 거래량 변화율
    "skewness",       # 수익률 왜도
]


def _extract_features(ticker: str, df: pd.DataFrame) -> dict:
    """단일 종목에서 군집화 특성 추출"""
    if df.empty or "Close" not in df.columns or len(df) < 20:
        return {}

    close  = df["Close"].astype(float)
    volume = df.get("Volume", pd.Series(dtype=float)).astype(float)

    ret = close.pct_change().dropna()

    ann_return = ret.mean() * 252
    ann_vol    = ret.std() * (252 ** 0.5)
    sharpe     = ann_return / (ann_vol + 1e-9)

    # 최대 낙폭
    cum    = (1 + ret).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / (peak + 1e-9)
    max_dd = float(dd.min())

    # 평균 RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    rsi   = (100 - 100 / (1 + gain / (loss + 1e-9))).dropna()
    rsi_mean = float(rsi.mean())

    # 20일 모멘텀
    lookback = min(21, len(close))
    momentum_20 = float(close.iloc[-1] / close.iloc[-lookback] - 1) if len(close) > 20 else 0.0

    # 거래량 변화율
    if len(volume) > 20 and volume.sum() > 0:
        vol_ratio = float(volume.iloc[-5:].mean() / (volume.iloc[-20:].mean() + 1e-9))
    else:
        vol_ratio = 1.0

    skewness = float(ret.skew())

    return {
        "ticker":      ticker,
        "ann_return":  round(float(ann_return), 4),
        "ann_vol":     round(float(ann_vol), 4),
        "sharpe":      round(float(sharpe), 4),
        "max_dd":      round(max_dd, 4),
        "rsi_mean":    round(rsi_mean, 4),
        "momentum_20": round(momentum_20, 4),
        "vol_ratio":   round(vol_ratio, 4),
        "skewness":    round(skewness, 4),
    }


# ---------------------------------------------------------------------------
# 군집 레이블 해석
# ---------------------------------------------------------------------------

def _name_cluster(stats: pd.Series) -> str:
    """군집 통계치를 보고 직관적 레이블 생성"""
    ret = stats.get("ann_return", 0)
    vol = stats.get("ann_vol", 0)
    dd  = stats.get("max_dd", 0)
    rsi = stats.get("rsi_mean", 50)
    mom = stats.get("momentum_20", 0)

    if ret > 0.15 and vol < 0.25:
        return "성장형 안정주 📈"
    if ret > 0.10 and mom > 0.05:
        return "모멘텀 강세주 🚀"
    if ret < -0.05 and dd < -0.20:
        return "하락 위험주 ⚠️"
    if vol > 0.35:
        return "고변동성 투기주 🎲"
    if abs(ret) < 0.05 and vol < 0.20:
        return "저변동 방어주 🛡️"
    if rsi > 65:
        return "과매수 구간 🔴"
    if rsi < 40:
        return "과매도 반등 후보 🟢"
    return "중립·관망 ➖"


# ---------------------------------------------------------------------------
# StockClusterer
# ---------------------------------------------------------------------------

class StockClusterer:
    """
    주식 종목 군집화 클래스.

    Parameters
    ----------
    n_clusters : 군집 수 (기본 4)
    method     : "kmeans" | "hierarchical"
    """

    def __init__(
        self,
        n_clusters: int = 4,
        method: str = "kmeans",
    ) -> None:
        if not _SKLEARN_OK:
            raise ImportError("pip install scikit-learn scipy 를 먼저 실행하세요.")
        self.n_clusters = n_clusters
        self.method     = method.lower()

    def fit(self, ticker_dfs: dict[str, pd.DataFrame]) -> ClusterResult:
        """
        여러 종목 데이터를 기반으로 군집화를 수행합니다.

        Parameters
        ----------
        ticker_dfs : {ticker: OHLCV DataFrame}

        Returns
        -------
        ClusterResult
        """
        feats = []
        for ticker, df in ticker_dfs.items():
            f = _extract_features(ticker, df)
            if f:
                feats.append(f)

        if len(feats) < self.n_clusters:
            raise ValueError(
                f"유효 종목 수({len(feats)})가 군집 수({self.n_clusters})보다 적습니다."
            )

        feat_df = pd.DataFrame(feats).set_index("ticker")

        # 결측·무한값 처리
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(feat_df[_FEATURE_COLS])

        # 군집화
        if self.method == "hierarchical":
            model  = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
            labels_arr = model.fit_predict(X_sc)
        else:
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
            labels_arr = model.fit_predict(X_sc)

        labels = {ticker: int(lbl) for ticker, lbl in zip(feat_df.index, labels_arr)}

        # 실루엣 계수
        sil = float(silhouette_score(X_sc, labels_arr)) if len(set(labels_arr)) > 1 else 0.0

        # PCA 2D
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_sc)
        pca_df = pd.DataFrame(
            coords, index=feat_df.index, columns=["PC1", "PC2"]
        )
        pca_df["cluster"] = labels_arr

        # 군집별 통계 요약
        feat_df["cluster"] = labels_arr
        summary_df = feat_df.groupby("cluster")[_FEATURE_COLS].mean().round(4)

        # 군집 레이블
        cluster_names = {
            cid: _name_cluster(row)
            for cid, row in summary_df.iterrows()
        }

        feat_df = feat_df.reset_index()

        logger.info(
            "군집화 완료 – 종목 %d개 / %d 군집 / 실루엣: %.3f",
            len(feats), self.n_clusters, sil,
        )
        return ClusterResult(
            n_clusters=self.n_clusters,
            method=self.method,
            labels=labels,
            feature_df=feat_df,
            summary_df=summary_df.reset_index(),
            silhouette=sil,
            cluster_names=cluster_names,
            pca_coords=pca_df.reset_index(),
        )

    @staticmethod
    def optimal_k(
        ticker_dfs: dict[str, pd.DataFrame],
        k_min: int = 2,
        k_max: int = 8,
    ) -> dict:
        """
        엘보우·실루엣 기법으로 최적 군집 수 탐색.

        Returns
        -------
        dict  {"k": int, "silhouettes": {k: score}, "inertias": {k: value}}
        """
        if not _SKLEARN_OK:
            raise ImportError("pip install scikit-learn 을 먼저 실행하세요.")

        feats = [f for t, df in ticker_dfs.items() if (f := _extract_features(t, df))]
        if len(feats) < k_max:
            k_max = len(feats)
        if k_max <= k_min:
            return {"k": k_min, "silhouettes": {}, "inertias": {}}

        feat_df = pd.DataFrame(feats).set_index("ticker")
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_sc    = StandardScaler().fit_transform(feat_df[_FEATURE_COLS])

        sils    = {}
        inertias = {}
        for k in range(k_min, k_max + 1):
            km  = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_sc)
            sils[k]     = round(float(silhouette_score(X_sc, lbl)), 4)
            inertias[k] = round(float(km.inertia_), 2)

        best_k = max(sils, key=sils.get)
        return {"k": best_k, "silhouettes": sils, "inertias": inertias}
