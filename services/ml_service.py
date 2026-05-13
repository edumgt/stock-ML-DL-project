"""ML/DL 마이크로서비스 진입점.

실행:
    uvicorn services.ml_service:app --host 0.0.0.0 --port 8000

K8s Deployment args:
    ["uvicorn", "services.ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
"""

from __future__ import annotations

import logging
from importlib import import_module

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML/DL Service",
    description="머신러닝 및 딥러닝 주가 예측 마이크로서비스",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROUTERS = [
    "api.routers.ml_strategy",
    "api.routers.dl_strategy",
]

for _path in _ROUTERS:
    try:
        _mod = import_module(_path)
        app.include_router(_mod.router)
    except Exception as exc:
        logger.warning("라우터 로드 건너뜀 (%s): %s", _path, exc)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "ml-service"}
