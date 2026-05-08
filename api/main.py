"""
Python Trading Sample – FastAPI 백엔드
========================================
실행 방법:
    uvicorn api.main:app --reload --port 8000

브라우저에서 http://localhost:8000 접속
API 문서: http://localhost:8000/docs
"""

from pathlib import Path
import logging
from importlib import import_module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Python Trading API",
    description="자동매매 모듈 FastAPI 테스트 서버",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROUTER_MODULES = [
    "api.routers.risk_manager",
    "api.routers.trade_logger",
    "api.routers.telegram_notifier",
    "api.routers.kiwoom",
    "api.routers.alpaca",
    "api.routers.ml_strategy",
    "api.routers.auto_trader",
    "api.routers.naver_crawler",
    "api.routers.stock_clustering",
    "api.routers.dl_strategy",
    "api.routers.webapp",
]

for module_path in _ROUTER_MODULES:
    try:
        module = import_module(module_path)
        app.include_router(module.router)
    except Exception as exc:
        logger.warning("라우터 로드 건너뜀 (%s): %s", module_path, exc)

# 정적 파일 (프론트엔드)
_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
