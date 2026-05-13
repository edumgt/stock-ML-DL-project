"""크롤러/군집화 마이크로서비스 진입점.

실행:
    uvicorn services.crawler_service:app --host 0.0.0.0 --port 8000

K8s Deployment args:
    ["uvicorn", "services.crawler_service:app", "--host", "0.0.0.0", "--port", "8000"]
"""

from __future__ import annotations

import logging
from importlib import import_module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crawler Service",
    description="네이버 금융 데이터 수집 및 주식 군집화 마이크로서비스",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROUTERS = [
    "api.routers.naver_crawler",
    "api.routers.stock_clustering",
]

for _path in _ROUTERS:
    try:
        _mod = import_module(_path)
        app.include_router(_mod.router)
    except Exception as exc:
        logger.warning("라우터 로드 건너뜀 (%s): %s", _path, exc)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "crawler-service"}
