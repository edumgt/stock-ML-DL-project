"""주식 군집화 API 라우터"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/cluster", tags=["Stock Clustering"])


# ---------------------------------------------------------------------------
# 요청 모델
# ---------------------------------------------------------------------------

class ClusterReq(BaseModel):
    tickers: List[str] = Field(
        default=["005930", "000660", "035420", "051910", "068270",
                 "005380", "035720", "207940", "006400", "003670"],
        description="분석할 종목 코드 목록",
    )
    pages: int = Field(default=5, ge=1, le=30, description="종목별 수집 페이지 수")
    n_clusters: int = Field(default=4, ge=2, le=10, description="군집 수")
    method: str = Field(default="kmeans", description="kmeans | hierarchical")


class OptimalKReq(BaseModel):
    tickers: List[str] = Field(
        default=["005930", "000660", "035420", "051910", "068270",
                 "005380", "035720", "207940", "006400", "003670"],
    )
    pages: int = Field(default=5, ge=1, le=20)
    k_min: int = Field(default=2, ge=2)
    k_max: int = Field(default=8, le=15)


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@router.post("/run")
def run_clustering(req: ClusterReq):
    """
    여러 종목을 군집화하여 주식 항목을 그룹으로 분류합니다.

    - 네이버 금융에서 OHLCV 데이터 수집
    - 수익률·변동성·모멘텀 등 특성 추출
    - K-Means 또는 계층적 군집화 수행
    - 군집별 특성 요약 및 레이블 반환
    """
    try:
        from trading.naver_crawler import NaverFinanceCrawler
        from trading.stock_clustering import StockClusterer
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    crawler = NaverFinanceCrawler()

    ticker_dfs: dict = {}
    failed_tickers: list = []
    for t in req.tickers:
        try:
            df = crawler.get_daily_ohlcv(t, pages=req.pages)
            if not df.empty:
                ticker_dfs[t] = df
            else:
                failed_tickers.append(f"{t}: 데이터 없음")
        except Exception:
            failed_tickers.append(f"{t}: 수집 실패")

    if len(ticker_dfs) < req.n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"유효 종목 수({len(ticker_dfs)})가 군집 수({req.n_clusters})보다 적습니다.",
        )

    try:
        clusterer = StockClusterer(n_clusters=req.n_clusters, method=req.method)
        result    = clusterer.fit(ticker_dfs)
    except (ImportError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    # JSON 직렬화
    cluster_list = []
    for ticker, cid in result.labels.items():
        cluster_list.append({
            "ticker":       ticker,
            "cluster_id":   cid,
            "cluster_name": result.cluster_names.get(cid, ""),
        })

    pca_data = []
    if result.pca_coords is not None:
        for _, row in result.pca_coords.iterrows():
            pca_data.append({
                "ticker":  row.get("ticker", ""),
                "pc1":     round(float(row["PC1"]), 4),
                "pc2":     round(float(row["PC2"]), 4),
                "cluster": int(row["cluster"]),
            })

    feature_records = result.feature_df.to_dict(orient="records")
    summary_records = result.summary_df.to_dict(orient="records")

    return {
        "n_clusters":    result.n_clusters,
        "method":        result.method,
        "silhouette":    round(result.silhouette, 4),
        "cluster_names": {str(k): v for k, v in result.cluster_names.items()},
        "assignments":   cluster_list,
        "features":      feature_records,
        "summary":       summary_records,
        "pca":           pca_data,
        "errors":        failed_tickers,
    }


@router.post("/optimal-k")
def find_optimal_k(req: OptimalKReq):
    """
    엘보우·실루엣 기법으로 최적 군집 수(k)를 탐색합니다.
    """
    try:
        from trading.naver_crawler import NaverFinanceCrawler
        from trading.stock_clustering import StockClusterer
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    crawler    = NaverFinanceCrawler()
    ticker_dfs = {}
    for t in req.tickers:
        try:
            df = crawler.get_daily_ohlcv(t, pages=req.pages)
            if not df.empty:
                ticker_dfs[t] = df
        except Exception:
            pass

    if len(ticker_dfs) < 3:
        raise HTTPException(status_code=400, detail="유효 종목이 3개 미만입니다.")

    try:
        res = StockClusterer.optimal_k(
            ticker_dfs, k_min=req.k_min, k_max=req.k_max
        )
    except (ImportError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    return res
