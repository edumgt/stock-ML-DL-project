# ── 베이스 스테이지 ──────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 시스템 의존성 (lxml 빌드 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libxml2-dev \
        libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 의존성 설치 스테이지 ─────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 런타임 스테이지 ───────────────────────────────────────────────
FROM base AS runtime

# 의존성 복사
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# 애플리케이션 코드 복사
COPY . .

EXPOSE 8000

# 기본 CMD: api-gateway (K8s Deployment에서 args로 오버라이드 가능)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
