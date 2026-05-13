# 주가 지수 데이터 활용 머신러닝 · 딥러닝 웹앱

### 선수 - https://github.com/edumgt/investment-analysis

### 선수 - https://github.com/edumgt/python-ai-basic-lab

### 선수 - https://github.com/edumgt/python-crawling-lab


국내 증시 데이터를 활용해 아래 4단계를 웹앱 버튼으로 실행하는 FastAPI + Vanilla JS 솔루션입니다.

1. **데이터 수집**: 네이버 금융 크롤링으로 종목 OHLCV/시장 데이터 수집
2. **종목 군집화**: 수익률·변동성·모멘텀 기반 주식 군집화
3. **ML 방향 예측**: 특징 기반 머신러닝 모델 학습 및 방향성 예측
4. **DL 방향 예측**: LSTM/MLP 기반 딥러닝 학습 및 방향성 예측

---

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

브라우저 접속:
- 웹앱: http://127.0.0.1:8000/
- API 문서: http://127.0.0.1:8000/docs

---

## 웹앱 프론트엔드

- `api/static/index.html`
- **Tailwind CSS + Pretendard 폰트 + Vanilla JavaScript**
- 메인 화면에 큰 4개 버튼으로 구성

---

## 백엔드 엔드포인트 (웹앱 전용)

- `POST /api/webapp/crawl`
- `POST /api/webapp/cluster`
- `POST /api/webapp/ml-predict`
- `POST /api/webapp/dl-predict`

웹앱의 각 버튼은 위 엔드포인트를 호출하여 결과를 화면에 표시합니다.

### MongoDB CRUD 엔드포인트

- 헬스체크: `GET /api/mongo/health`
- 로그인 사용자 CRUD: `POST/GET/PUT/DELETE /api/mongo/users`
- 사용자 로그인: `POST /api/mongo/auth/login`
- 크롤링 데이터 CRUD: `POST/GET/PUT/DELETE /api/mongo/crawls`
- 분석 데이터 CRUD: `POST/GET/PUT/DELETE /api/mongo/analyses`

기본 연결 정보:
- `MONGODB_URI` (기본값: `mongodb://localhost:27017`)
- `MONGODB_DB_NAME` (기본값: `stock_mldl`)

---

## 웹앱 화면 스크린샷

> 로컬에서 `uvicorn api.main:app --port 8000` 실행 후 Playwright로 재캡처한 주요 화면 10개입니다.
> 캡처 재현성을 위해 액션 결과 화면(5~8)은 Playwright에서 `fetch` 응답을 고정(mock)해 촬영했습니다.

### 1. 메인 화면 초기 상태
![메인 화면 초기](docs/screenshots/01_main_home.png)

### 2. 전체 페이지 스크롤
![전체 페이지](docs/screenshots/02_main_fullpage.png)

### 3. 좌측 입력 패널 (공통 입력 + 액션 버튼)
![입력 패널](docs/screenshots/03_input_panel.png)

### 4. 우측 결과 패널 (초기 상태)
![결과 패널 초기](docs/screenshots/04_result_panel_initial.png)

### 5. 데이터 수집 버튼 클릭 → 로딩/결과
![데이터 수집](docs/screenshots/05_crawl_loading.png)

### 6. ML 방향 예측 결과
![ML 예측](docs/screenshots/06_ml_result.png)

### 7. DL 방향 예측 결과
![DL 예측](docs/screenshots/07_dl_result.png)

### 8. 종목 군집화 결과
![군집화](docs/screenshots/08_cluster_result.png)

### 9. MongoDB CRUD 섹션
![MongoDB 섹션](docs/screenshots/09_mongodb_section.png)

### 10. 모바일 뷰 (375px)
![모바일](docs/screenshots/10_mobile_view.png)

---

## 프로젝트 구조 (핵심)

```text
api/
  main.py
  routers/
    webapp.py
    naver_crawler.py
    stock_clustering.py
    ml_strategy.py
    dl_strategy.py
  static/
    index.html

trading/
  naver_crawler.py
  stock_clustering.py
  ml_strategy.py
  dl_strategy.py
```

---

## Kubernetes MSA 배포

이 프로젝트는 단일 FastAPI 앱을 **4개의 독립 마이크로서비스**로 분리 배포할 수 있습니다.

### MSA 아키텍처

```
외부 트래픽
    │
    ▼
[Nginx Ingress]
    │
    ├── /api/ml/*   /api/dl/*       → ml-service      (ML/DL 예측)
    ├── /api/naver/* /api/cluster/* → crawler-service  (데이터 수집·군집화)
    ├── /api/mongo/*                → mongo-service    (MongoDB CRUD)
    └── /*                          → api-gateway      (프론트엔드 + 웹앱 대시보드)
                                           │
                                    [mongodb StatefulSet]
```

### 서비스별 역할

| 서비스 | 진입점 | 담당 라우트 | 리소스 |
|--------|--------|-------------|--------|
| `api-gateway` | `api.main:app` | `/`, `/api/webapp/*` 등 | 200m CPU / 512Mi |
| `ml-service` | `services.ml_service:app` | `/api/ml/*`, `/api/dl/*` | 500m CPU / 1Gi |
| `crawler-service` | `services.crawler_service:app` | `/api/naver/*`, `/api/cluster/*` | 300m CPU / 512Mi |
| `mongo-service` | `services.mongo_service:app` | `/api/mongo/*` | 100m CPU / 256Mi |
| `mongodb` | `mongo:7.0` StatefulSet | DB | 250m CPU / 512Mi |

### 로컬 개발 (Docker Compose)

```bash
docker compose up --build
```

| 서비스 | 로컬 포트 |
|--------|-----------|
| api-gateway | http://localhost:8000 |
| ml-service | http://localhost:8001 |
| crawler-service | http://localhost:8002 |
| mongo-service | http://localhost:8003 |

### Kubernetes 배포

```bash
# 1. 네임스페이스 및 공통 설정 생성
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# 2. MongoDB Secret 수정 후 적용 (비밀번호 반드시 변경)
#    vi k8s/mongodb/secret.yaml
kubectl apply -f k8s/mongodb/secret.yaml

# 3. MongoDB 배포
kubectl apply -f k8s/mongodb/

# 4. 애플리케이션 서비스 배포
kubectl apply -f k8s/api-gateway/
kubectl apply -f k8s/ml-service/
kubectl apply -f k8s/crawler-service/
kubectl apply -f k8s/mongo-service/

# 5. Ingress 적용 (Nginx Ingress Controller 사전 설치 필요)
kubectl apply -f k8s/ingress.yaml

# 전체 한 번에 적용
kubectl apply -R -f k8s/
```

### Docker 이미지 빌드

모든 서비스가 동일한 Docker 이미지를 사용하며, K8s Deployment의 `command`로 진입점을 구분합니다.

```bash
docker build -t stock-mldl:latest .
# 또는 레지스트리에 푸시
docker build -t ghcr.io/edumgt/stock-mldl:latest .
docker push ghcr.io/edumgt/stock-mldl:latest
```

> 빌드 후 `k8s/*/deployment.yaml` 의 `image:` 필드를 실제 레지스트리 경로로 업데이트하세요.
