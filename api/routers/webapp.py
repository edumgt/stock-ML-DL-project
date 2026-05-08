"""웹앱 대시보드용 4개 액션 API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/webapp", tags=["WebApp Dashboard"])


class CrawlReq(BaseModel):
    ticker: str = Field(default="005930")
    market: str = Field(default="kospi")
    pages: int = Field(default=5, ge=1, le=20)


class ClusterReq(BaseModel):
    tickers: list[str] = Field(default=["005930", "000660", "035420", "051910", "068270"])
    pages: int = Field(default=5, ge=1, le=20)
    n_clusters: int = Field(default=3, ge=2, le=8)
    method: str = Field(default="kmeans")


class MLPredictReq(BaseModel):
    ticker: str = Field(default="005930")
    source: str = Field(default="naver", description="naver | yfinance")
    pages: int = Field(default=30, ge=5, le=80)
    period: str = Field(default="3y")
    model_type: str = Field(default="rf", description="rf | gb | xgb")
    forward_days: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.01, ge=0.001, le=0.1)


class DLPredictReq(BaseModel):
    ticker: str = Field(default="005930")
    source: str = Field(default="naver", description="naver | yfinance")
    pages: int = Field(default=30, ge=5, le=80)
    period: str = Field(default="3y")
    model_type: str = Field(default="auto", description="auto | lstm | mlp")
    seq_len: int = Field(default=20, ge=5, le=60)
    forward_days: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.01, ge=0.001, le=0.1)
    epochs: int = Field(default=20, ge=5, le=200)


def _load_ohlcv(ticker: str, source: str, pages: int, period: str):
    if source == "naver":
        from trading.naver_crawler import NaverFinanceCrawler

        df = NaverFinanceCrawler().get_daily_ohlcv(ticker, pages=pages)
    else:
        import yfinance as yf

        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"데이터 없음: {ticker}")
    return df


@router.post("/crawl")
def crawl(req: CrawlReq):
    try:
        from trading.naver_crawler import NaverFinanceCrawler, get_market_stocks
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    crawler = NaverFinanceCrawler()
    df = crawler.get_daily_ohlcv(req.ticker, pages=req.pages)
    market_df = get_market_stocks(req.market, pages=min(2, req.pages))
    info = crawler.get_stock_info(req.ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"데이터 없음: {req.ticker}")

    latest = df.tail(1).copy()
    latest["Date"] = latest["Date"].dt.strftime("%Y-%m-%d")
    return {
        "ticker": req.ticker,
        "ohlcv_rows": len(df),
        "latest_ohlcv": latest.to_dict(orient="records")[0],
        "stock_info": info,
        "market": req.market.upper(),
        "market_sample": market_df.head(10).to_dict(orient="records"),
    }


@router.post("/cluster")
def cluster(req: ClusterReq):
    try:
        from trading.naver_crawler import NaverFinanceCrawler
        from trading.stock_clustering import StockClusterer
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    crawler = NaverFinanceCrawler()
    ticker_dfs: dict[str, object] = {}
    errors: list[str] = []
    for t in req.tickers:
        try:
            df = crawler.get_daily_ohlcv(t, pages=req.pages)
            if not df.empty:
                ticker_dfs[t] = df
            else:
                errors.append(f"{t}: 데이터 없음")
        except Exception:
            errors.append(f"{t}: 수집 실패")

    if len(ticker_dfs) < req.n_clusters:
        raise HTTPException(status_code=400, detail="유효 종목 수가 군집 수보다 적습니다.")

    result = StockClusterer(n_clusters=req.n_clusters, method=req.method).fit(ticker_dfs)
    return {
        "n_clusters": result.n_clusters,
        "method": result.method,
        "silhouette": round(result.silhouette, 4),
        "assignments": [{"ticker": k, "cluster": v} for k, v in result.labels.items()],
        "cluster_names": result.cluster_names,
        "summary": result.summary_df.to_dict(orient="records"),
        "errors": errors,
    }


@router.post("/ml-predict")
def ml_predict(req: MLPredictReq):
    try:
        from trading.ml_strategy import MLStrategy
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        df = _load_ohlcv(req.ticker, req.source, req.pages, req.period)
        strategy = MLStrategy(
            model_type=req.model_type,
            forward_days=req.forward_days,
            threshold=req.threshold,
        )
        train_result = strategy.train(df)
        signal = strategy.predict(df)
        proba = strategy.predict_proba(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ticker": req.ticker,
        "model_type": train_result.model_type,
        "accuracy": round(train_result.accuracy, 4),
        "signal": signal,
        "probabilities": proba,
    }


@router.post("/dl-predict")
def dl_predict(req: DLPredictReq):
    try:
        from trading.dl_strategy import DLStrategy
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        df = _load_ohlcv(req.ticker, req.source, req.pages, req.period)
        strategy = DLStrategy(
            model_type=req.model_type,
            seq_len=req.seq_len,
            forward_days=req.forward_days,
            threshold=req.threshold,
        )
        kwargs = {"epochs": req.epochs, "batch_size": 32} if strategy.model_type == "lstm" else {}
        train_result = strategy.train(df, **kwargs)
        signal = strategy.predict(df)
        proba = strategy.predict_proba(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ticker": req.ticker,
        "model_type": train_result.model_type,
        "accuracy": round(train_result.accuracy, 4),
        "signal": signal,
        "probabilities": proba,
    }
