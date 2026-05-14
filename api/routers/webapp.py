"""웹앱 대시보드용 액션 API."""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from fastapi import APIRouter, HTTPException
from pymongo.errors import PyMongoError
from pydantic import BaseModel, Field

from api.mongodb_store import repo

router = APIRouter(prefix="/api/webapp", tags=["WebApp Dashboard"])
logger = logging.getLogger(__name__)


class CrawlReq(BaseModel):
    ticker: str = Field(default="005930")
    market: str = Field(default="kospi")
    pages: int = Field(default=5, ge=1, le=20)


class ClusterReq(BaseModel):
    tickers: list[str] = Field(default=["005930", "000660", "035420", "051910", "068270"])
    source: str = Field(default="naver", description="naver | yfinance")
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


class AnalysisReq(BaseModel):
    ticker: str = Field(default="005930")
    source: str = Field(default="naver", description="naver | yfinance")
    pages: int = Field(default=80, ge=5, le=120)
    period: str = Field(default="5y")
    seq_len: int = Field(default=20, ge=5, le=60)


class StockForecastReq(BaseModel):
    ticker: str = Field(default="428560", description="종목코드 (예: 428560)")
    source: str = Field(default="naver", description="naver | yfinance")
    pages: int = Field(default=15, ge=1, le=50)
    period: str = Field(default="1y")


class MultiHeadReq(BaseModel):
    tickers: list[str] = Field(default=["005930", "000660", "035420", "051910"])
    source: str = Field(default="naver", description="naver | yfinance")
    pages: int = Field(default=80, ge=5, le=120)
    period: str = Field(default="5y")


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


def _save_crawl(payload: dict):
    try:
        repo.ping()
        return repo.insert_one(repo.crawls, payload)
    except PyMongoError as e:
        logger.warning("MongoDB crawl 저장 실패: %s", e)
        return None


def _save_analysis(payload: dict):
    try:
        repo.ping()
        return repo.insert_one(repo.analyses, payload)
    except PyMongoError as e:
        logger.warning("MongoDB analysis 저장 실패: %s", e)
        return None


@router.post("/crawl")
def crawl(req: CrawlReq):
    try:
        from trading.naver_crawler import NaverFinanceCrawler, get_market_stocks
    except ImportError:
        raise HTTPException(status_code=503, detail="크롤링 모듈을 불러올 수 없습니다.")

    crawler = NaverFinanceCrawler()
    df = crawler.get_daily_ohlcv(req.ticker, pages=req.pages)
    market_df = get_market_stocks(req.market, pages=min(2, req.pages))
    info = crawler.get_stock_info(req.ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"데이터 없음: {req.ticker}")

    latest = df.tail(1).copy()
    latest["Date"] = latest["Date"].dt.strftime("%Y-%m-%d")
    stock_info = {
        "ticker": req.ticker,
        "name": info.get("name", ""),
        "price": info.get("price", ""),
        "change": info.get("change", ""),
        "rate": info.get("rate", ""),
        "fetched_at": info.get("fetched_at", ""),
    }
    if "error" in info:
        stock_info["error"] = "종목 정보 조회 실패(내부 로그 확인)"
    response = {
        "ticker": req.ticker,
        "ohlcv_rows": len(df),
        "latest_ohlcv": latest.to_dict(orient="records")[0],
        "stock_info": stock_info,
        "market": req.market.upper(),
        "market_sample": market_df.head(10).to_dict(orient="records"),
    }
    mongo_id = _save_crawl(
        {
            "ticker": req.ticker,
            "market": req.market,
            "pages": req.pages,
            "ohlcv_rows": len(df),
            "latest_ohlcv": response["latest_ohlcv"],
            "stock_info": stock_info,
            "market_sample": response["market_sample"],
        }
    )
    if mongo_id:
        response["mongo_id"] = mongo_id
    return response


def _fetch_cluster_df(ticker: str, source: str, pages: int) -> "pd.DataFrame | None":
    """ticker 하나의 OHLCV를 source에 따라 수집. 실패 시 None 반환."""
    import pandas as pd
    if source == "yfinance":
        try:
            import yfinance as yf
            # pages → 대략적인 기간 매핑 (1 page ≈ 25 거래일)
            _period_map = {1: "3mo", 2: "3mo", 5: "6mo", 10: "1y", 20: "2y"}
            period = _period_map.get(pages, "1y") if pages <= 20 else "2y"
            t_yf = ticker if not ticker.isdigit() else ticker + ".KS"
            df = yf.download(t_yf, period=period, auto_adjust=True, progress=False)
            if df.empty and not ticker.endswith(".KS"):
                df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return df if not df.empty else None
        except Exception:
            return None
    else:
        try:
            from trading.naver_crawler import NaverFinanceCrawler
            df = NaverFinanceCrawler().get_daily_ohlcv(ticker, pages=pages)
            return df if not df.empty else None
        except Exception:
            return None


@router.post("/cluster")
def cluster(req: ClusterReq):
    try:
        from trading.stock_clustering import StockClusterer
    except ImportError:
        raise HTTPException(status_code=503, detail="군집화 모듈을 불러올 수 없습니다.")

    ticker_dfs: dict[str, object] = {}
    errors: list[str] = []
    for t in req.tickers:
        df = _fetch_cluster_df(t, req.source, req.pages)
        if df is not None:
            ticker_dfs[t] = df
        else:
            errors.append(f"{t}: 데이터 수집 실패 (source={req.source})")

    if len(ticker_dfs) < req.n_clusters:
        raise HTTPException(status_code=400, detail="유효 종목 수가 군집 수보다 적습니다.")

    result = StockClusterer(n_clusters=req.n_clusters, method=req.method).fit(ticker_dfs)
    response = {
        "n_clusters": result.n_clusters,
        "method": result.method,
        "silhouette": round(result.silhouette, 4),
        "assignments": [{"ticker": k, "cluster": v} for k, v in result.labels.items()],
        "cluster_names": result.cluster_names,
        "summary": result.summary_df.to_dict(orient="records"),
        "errors": errors,
    }
    mongo_id = _save_analysis(
        {
            "analysis_type": "cluster",
            "tickers": req.tickers,
            "params": req.model_dump(),
            "result": response,
        }
    )
    if mongo_id:
        response["mongo_id"] = mongo_id
    return response


@router.post("/ml-predict")
def ml_predict(req: MLPredictReq):
    try:
        from trading.ml_strategy import MLStrategy
    except ImportError:
        raise HTTPException(status_code=503, detail="ML 모듈을 불러올 수 없습니다.")

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
    except Exception:
        raise HTTPException(status_code=400, detail="ML 예측 처리 실패")

    response = {
        "ticker": req.ticker,
        "model_type": train_result.model_type,
        "accuracy": round(train_result.accuracy, 4),
        "cv_mean": round(train_result.cv_mean, 4),
        "cv_std": round(train_result.cv_std, 4),
        "cv_scores": train_result.cv_scores,
        "signal": signal,
        "probabilities": proba,
        "feature_importance": dict(list(train_result.feature_importance.items())[:10]),
    }
    mongo_id = _save_analysis(
        {
            "analysis_type": "ml",
            "ticker": req.ticker,
            "params": req.model_dump(),
            "result": response,
        }
    )
    if mongo_id:
        response["mongo_id"] = mongo_id
    return response


@router.post("/dl-predict")
def dl_predict(req: DLPredictReq):
    try:
        from trading.dl_strategy import DLStrategy
    except ImportError:
        raise HTTPException(status_code=503, detail="DL 모듈을 불러올 수 없습니다.")

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
    except Exception:
        raise HTTPException(status_code=400, detail="DL 예측 처리 실패")

    response = {
        "ticker": req.ticker,
        "model_type": train_result.model_type,
        "accuracy": round(train_result.accuracy, 4),
        "signal": signal,
        "probabilities": proba,
    }
    mongo_id = _save_analysis(
        {
            "analysis_type": "dl",
            "ticker": req.ticker,
            "params": req.model_dump(),
            "result": response,
        }
    )
    if mongo_id:
        response["mongo_id"] = mongo_id
    return response


@router.post("/timeseries")
def timeseries(req: AnalysisReq):
    try:
        from trading.webapp_analytics import timeseries_report
    except ImportError:
        raise HTTPException(status_code=503, detail="timeseries 모듈을 불러올 수 없습니다.")

    try:
        result = timeseries_report(req.ticker, req.source, req.pages, req.period)
        mongo_id = _save_analysis(
            {
                "analysis_type": "timeseries",
                "ticker": req.ticker,
                "params": req.model_dump(),
                "result": result,
            }
        )
        if mongo_id:
            result["mongo_id"] = mongo_id
        return result
    except Exception:
        raise HTTPException(status_code=400, detail="timeseries 분석 처리 실패")


@router.post("/sequence-lstm")
def sequence_lstm(req: AnalysisReq):
    try:
        from trading.webapp_analytics import sequence_report
    except ImportError:
        raise HTTPException(status_code=503, detail="sequence-lstm 모듈을 불러올 수 없습니다.")

    try:
        result = sequence_report(req.ticker, req.source, req.pages, req.period)
        mongo_id = _save_analysis(
            {
                "analysis_type": "sequence-lstm",
                "ticker": req.ticker,
                "params": req.model_dump(),
                "result": result,
            }
        )
        if mongo_id:
            result["mongo_id"] = mongo_id
        return result
    except Exception:
        raise HTTPException(status_code=400, detail="sequence-lstm 분석 처리 실패")


@router.post("/attention-core")
def attention_core(req: AnalysisReq):
    try:
        from trading.webapp_analytics import attention_report
    except ImportError:
        raise HTTPException(status_code=503, detail="attention-core 모듈을 불러올 수 없습니다.")

    try:
        result = attention_report(req.ticker, req.source, req.pages, req.period, req.seq_len)
        mongo_id = _save_analysis(
            {
                "analysis_type": "attention-core",
                "ticker": req.ticker,
                "params": req.model_dump(),
                "result": result,
            }
        )
        if mongo_id:
            result["mongo_id"] = mongo_id
        return result
    except Exception:
        raise HTTPException(status_code=400, detail="attention-core 분석 처리 실패")


@router.post("/transformer")
def transformer(req: AnalysisReq):
    try:
        from trading.webapp_analytics import transformer_report
    except ImportError:
        raise HTTPException(status_code=503, detail="transformer 모듈을 불러올 수 없습니다.")

    try:
        result = transformer_report(req.ticker, req.source, req.pages, req.period, req.seq_len)
        mongo_id = _save_analysis(
            {
                "analysis_type": "transformer",
                "ticker": req.ticker,
                "params": req.model_dump(),
                "result": result,
            }
        )
        if mongo_id:
            result["mongo_id"] = mongo_id
        return result
    except Exception:
        raise HTTPException(status_code=400, detail="transformer 분석 처리 실패")


@router.post("/patchtst")
def patchtst(req: AnalysisReq):
    try:
        from trading.webapp_analytics import patchtst_report
    except ImportError:
        raise HTTPException(status_code=503, detail="patchtst 모듈을 불러올 수 없습니다.")

    try:
        result = patchtst_report(req.ticker, req.source, req.pages, req.period)
        mongo_id = _save_analysis(
            {
                "analysis_type": "patchtst",
                "ticker": req.ticker,
                "params": req.model_dump(),
                "result": result,
            }
        )
        if mongo_id:
            result["mongo_id"] = mongo_id
        return result
    except Exception:
        raise HTTPException(status_code=400, detail="patchtst 분석 처리 실패")


@router.post("/multihead")
def multihead(req: MultiHeadReq):
    try:
        from trading.webapp_analytics import multihead_report
    except ImportError:
        raise HTTPException(status_code=503, detail="multihead 모듈을 불러올 수 없습니다.")

    try:
        result = multihead_report(req.tickers, req.source, req.pages, req.period)
        mongo_id = _save_analysis(
            {
                "analysis_type": "multihead",
                "tickers": req.tickers,
                "params": req.model_dump(),
                "result": result,
            }
        )
        if mongo_id:
            result["mongo_id"] = mongo_id
        return result
    except Exception:
        raise HTTPException(status_code=400, detail="multihead 분석 처리 실패")


@router.post("/backtest")
def backtest(req: AnalysisReq):
    try:
        from trading.webapp_analytics import backtest_report
    except ImportError:
        raise HTTPException(status_code=503, detail="backtest 모듈을 불러올 수 없습니다.")

    try:
        result = backtest_report(req.ticker, req.source, req.pages, req.period)
        mongo_id = _save_analysis(
            {
                "analysis_type": "backtest",
                "ticker": req.ticker,
                "params": req.model_dump(),
                "result": result,
            }
        )
        if mongo_id:
            result["mongo_id"] = mongo_id
        return result
    except Exception:
        raise HTTPException(status_code=400, detail="backtest 분석 처리 실패")


@router.post("/stock-forecast")
def stock_forecast(req: StockForecastReq):
    """
    종목의 내일 주가 예측 리포트를 생성합니다.
    네이버 금융(또는 yfinance)에서 데이터를 수집하고,
    최근 변동성 기반 예측 범위와 시나리오 분석을 반환합니다.
    """
    # 예측 파라미터 상수
    _TREND_WEIGHT = 0.3        # MA 차이에서 중심 예측값에 반영하는 가중치
    _VOL_MULTIPLIER = 1.5      # 변동성에 곱해 ±예측 폭을 결정하는 배율
    _CENTER_BAND = 0.005       # 중심값 표시용 ±0.5% 밴드
    _TREND_THRESHOLD = 0.005   # 상승/하락 시나리오 판단 기준(0.5%)

    # ── 1. 데이터 수집 ─────────────────────────────────────────────────────
    df = _load_ohlcv(req.ticker, req.source, req.pages, req.period)

    # ── 2. 종목명 조회 ─────────────────────────────────────────────────────
    stock_name = req.ticker
    try:
        from trading.naver_crawler import NaverFinanceCrawler
        info = NaverFinanceCrawler().get_stock_info(req.ticker)
        if info.get("name"):
            stock_name = info["name"]
    except Exception:
        pass

    # ── 3. 기본 통계 계산 ──────────────────────────────────────────────────
    close = df["Close"].astype(float)
    latest_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
    change = latest_close - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0.0

    tail_252 = close.tail(252)
    high_52w = float(tail_252.max())
    low_52w = float(tail_252.min())
    pos_52w = (
        (latest_close - low_52w) / (high_52w - low_52w) * 100
        if (high_52w - low_52w) > 0
        else 50.0
    )

    latest_volume = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0

    # ── 4. 변동성 기반 예측 범위 계산 ─────────────────────────────────────
    returns = close.pct_change().dropna()
    n_ret = len(returns)
    if n_ret >= 20:
        recent_vol = float(returns.tail(20).std())
    elif n_ret > 1:
        recent_vol = float(returns.std())
    else:
        recent_vol = 0.01

    ma5 = float(close.tail(5).mean())
    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
    trend_factor = (ma5 - ma20) / ma20 if ma20 else 0.0

    center_pred = latest_close * (1.0 + trend_factor * _TREND_WEIGHT)
    range_delta = latest_close * recent_vol * _VOL_MULTIPLIER
    lower_pred = center_pred - range_delta
    upper_pred = center_pred + range_delta
    center_low = center_pred * (1.0 - _CENTER_BAND)
    center_high = center_pred * (1.0 + _CENTER_BAND)

    # ── 5. 날짜 계산 ───────────────────────────────────────────────────────
    if "Date" in df.columns:
        latest_ts = pd.Timestamp(df["Date"].iloc[-1])
    else:
        latest_ts = pd.Timestamp(df.index[-1])

    tomorrow_ts = latest_ts + timedelta(days=1)
    while tomorrow_ts.weekday() >= 5:
        tomorrow_ts += timedelta(days=1)

    today_str = latest_ts.strftime("%Y년 %m월 %d일")
    tomorrow_str = tomorrow_ts.strftime("%Y년 %m월 %d일")

    # ── 6. 시나리오 판단 ───────────────────────────────────────────────────
    if trend_factor > _TREND_THRESHOLD:
        up_prob_label = "높음"
    elif trend_factor < -_TREND_THRESHOLD:
        up_prob_label = "낮음"
    else:
        up_prob_label = "중간"

    # ── 7. 응답 조립 ───────────────────────────────────────────────────────
    summary = [
        {"label": "종목명", "value": stock_name},
        {"label": "종목코드", "value": req.ticker},
        {"label": "기준일 종가", "value": f"{latest_close:,.0f}원"},
        {"label": "전일 대비", "value": f"{'↑' if change >= 0 else '↓'} {abs(change):,.0f}원 ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)"},
        {"label": "52주 최고 / 최저", "value": f"{high_52w:,.0f}원 / {low_52w:,.0f}원"},
        {"label": "거래량", "value": f"{latest_volume:,}주"},
        {"label": "내일 예상 범위", "value": f"{lower_pred:,.0f} ~ {upper_pred:,.0f}원"},
        {"label": "예측 대상일", "value": tomorrow_str},
    ]

    header_note = (
        f"{stock_name} (종목코드: {req.ticker})의 내일({tomorrow_str}) 주가 예측은 "
        "불확실성이 높습니다. ETF 주가는 기초지수 성과, 미국 증시 움직임, "
        "환율(원/달러), 시장 심리 등에 따라 크게 변동할 수 있습니다."
    )
    disclaimer_note = (
        "중요 주의사항: 주가 예측은 본질적으로 불확실하며, 과거 패턴이나 "
        "현재 추세가 미래를 보장하지 않습니다. 이는 투자 조언이 아니며, "
        "실제 투자 결정은 본인 책임 하에 전문가 상담이나 최신 시장 정보를 "
        "바탕으로 하시기 바랍니다."
    )
    notes = [
        header_note,
        f"최근 주가 정보 ({today_str} 장 마감 기준)",
        f"  종가: 약 {latest_close:,.0f}원 ({'+' if change >= 0 else ''}{change:,.0f}원, {'+' if change_pct >= 0 else ''}{change_pct:.2f}%)",
        f"  최근 범위: 52주 최고 {high_52w:,.0f}원 / 최저 {low_52w:,.0f}원 (현재 위치: 52주 범위의 {pos_52w:.0f}%)",
        f"  거래량: {latest_volume:,}주",
        "예측 범위 (단기 참고용)",
        f"  내일 예상 범위: {lower_pred:,.0f} ~ {upper_pred:,.0f}원 정도 (중심값 약 {center_low:,.0f} ~ {center_high:,.0f}원)",
        f"  상승 시나리오 (확률 {up_prob_label}): 시장 강세, 기초지수 및 관련 섹터가 긍정적 움직임을 보이면 +1% 내외 상승 가능.",
        "  하락 시나리오: 시장 조정, 위험회피 심리 확대 또는 원화 강세 시 -1% 정도 하락 가능.",
        "  중립: 보합권에서 마감할 가능성도 높음 (변동성 낮은 ETF 특성).",
        "영향 요인: 기초지수 성과, 미국 증시(S&P 500·Nasdaq) 및 금융주 동향, 원/달러 환율 변동, 글로벌 리스크(지정학·매크로 데이터).",
        disclaimer_note,
    ]

    # ── 8. 차트 데이터 ─────────────────────────────────────────────────────
    tail_n = min(30, len(df))
    recent = df.tail(tail_n)
    if "Date" in recent.columns:
        chart_labels = pd.to_datetime(recent["Date"]).dt.strftime("%m-%d").tolist()
    else:
        chart_labels = [str(i) for i in range(tail_n)]

    close_values = recent["Close"].tolist()
    pred_band_lower = [round(lower_pred)] * tail_n
    pred_band_upper = [round(upper_pred)] * tail_n

    charts = [
        {
            "title": f"{stock_name} 최근 종가 추이 및 예측 범위",
            "type": "line",
            "labels": chart_labels,
            "datasets": [
                {
                    "label": "종가",
                    "data": close_values,
                    "borderColor": "#06b6d4",
                    "backgroundColor": "rgba(6,182,212,0.10)",
                },
                {
                    "label": f"예측 상단 ({round(upper_pred):,}원)",
                    "data": pred_band_upper,
                    "borderColor": "#22c55e",
                    "backgroundColor": "rgba(34,197,94,0.06)",
                },
                {
                    "label": f"예측 하단 ({round(lower_pred):,}원)",
                    "data": pred_band_lower,
                    "borderColor": "#f97316",
                    "backgroundColor": "rgba(249,115,22,0.06)",
                },
            ],
        }
    ]

    response = {
        "ticker": req.ticker,
        "name": stock_name,
        "target_date": tomorrow_str,
        "base_date": today_str,
        "latest_close": round(latest_close),
        "change": round(change),
        "change_pct": round(change_pct, 2),
        "high_52w": round(high_52w),
        "low_52w": round(low_52w),
        "predicted_lower": round(lower_pred),
        "predicted_upper": round(upper_pred),
        "predicted_center": round(center_pred),
        "recent_volatility_pct": round(recent_vol * 100, 2),
        "summary": summary,
        "notes": notes,
        "charts": charts,
    }

    mongo_id = _save_analysis(
        {
            "analysis_type": "stock_forecast",
            "ticker": req.ticker,
            "params": req.model_dump(),
            "result": {k: v for k, v in response.items() if k not in ("summary", "notes", "charts")},
        }
    )
    if mongo_id:
        response["mongo_id"] = mongo_id

    return response
