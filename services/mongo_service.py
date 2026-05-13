"""MongoDB CRUD 마이크로서비스 진입점.

실행:
    uvicorn services.mongo_service:app --host 0.0.0.0 --port 8000

K8s Deployment args:
    ["uvicorn", "services.mongo_service:app", "--host", "0.0.0.0", "--port", "8000"]

환경 변수:
    MONGODB_URI      MongoDB 연결 URI (기본값: mongodb://localhost:27017)
    MONGODB_DB_NAME  데이터베이스 이름 (기본값: stock_mldl)
"""

from __future__ import annotations

import logging
from importlib import import_module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mongo Service",
    description="MongoDB CRUD 마이크로서비스 (사용자·크롤링·분석 데이터)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROUTERS = [
    "api.routers.mongodb_crud",
]

for _path in _ROUTERS:
    try:
        _mod = import_module(_path)
        app.include_router(_mod.router)
    except Exception as exc:
        logger.warning("라우터 로드 건너뜀 (%s): %s", _path, exc)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "mongo-service"}
