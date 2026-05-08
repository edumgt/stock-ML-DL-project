"""네이버 금융 크롤러 API 라우터"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/naver", tags=["Naver Crawler"])


# ---------------------------------------------------------------------------
# 요청 모델
# ---------------------------------------------------------------------------

class StockInfoReq(BaseModel):
    ticker: str


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@router.get("/ohlcv/{ticker}")
def get_ohlcv(
    ticker: str,
    pages: int = Query(default=10, ge=1, le=50, description="수집 페이지 수 (1페이지≈25거래일)"),
    start_date: Optional[str] = Query(default=None, description="시작일 YYYY.MM.DD"),
    end_date:   Optional[str] = Query(default=None, description="종료일 YYYY.MM.DD"),
):
    """
    네이버 금융에서 특정 종목의 일별 OHLCV 데이터를 수집합니다.

    - **ticker**: 종목 코드 (예: 005930 – 삼성전자)
    - **pages**: 수집 페이지 수 (1페이지 ≈ 25거래일)
    """
    try:
        from trading.naver_crawler import NaverFinanceCrawler
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    crawler = NaverFinanceCrawler()
    try:
        df = crawler.get_daily_ohlcv(
            ticker, pages=pages, start_date=start_date, end_date=end_date
        )
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail=f"데이터 없음: {ticker}")

    records = df.copy()
    records["Date"] = records["Date"].dt.strftime("%Y-%m-%d")
    return {
        "ticker": ticker,
        "count":  len(records),
        "data":   records.to_dict(orient="records"),
    }


@router.get("/info/{ticker}")
def get_stock_info(ticker: str):
    """
    네이버 금융에서 종목 기본 정보(현재가, 종목명 등)를 조회합니다.
    """
    try:
        from trading.naver_crawler import NaverFinanceCrawler
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    crawler = NaverFinanceCrawler()
    try:
        info = crawler.get_stock_info(ticker)
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if "error" in info:
        raise HTTPException(status_code=500, detail="종목 정보 조회에 실패했습니다.")
    return info


@router.get("/market")
def get_market_stocks(
    market: str = Query(default="kospi", description="kospi 또는 kosdaq"),
    pages:  int = Query(default=2, ge=1, le=10, description="수집 페이지 수"),
):
    """
    KOSPI 또는 KOSDAQ 시가총액 상위 종목 목록을 반환합니다.
    """
    if market.lower() not in ("kospi", "kosdaq"):
        raise HTTPException(status_code=400, detail="market 은 kospi 또는 kosdaq 이어야 합니다.")

    try:
        from trading.naver_crawler import get_market_stocks as _fetch
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        df = _fetch(market=market, pages=pages)
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "market": market.upper(),
        "count":  len(df),
        "stocks": df.to_dict(orient="records"),
    }
