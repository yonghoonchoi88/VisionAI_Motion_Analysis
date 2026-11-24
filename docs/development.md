# 개발 환경 실행과 테스트

README의 Docker Compose 실행 외에, 코드를 고치며 빠르게 확인하는 방법입니다.

## Docker 없이 로컬 실행 (WSL)

```bash
# ① DB·브로커만 Docker로
docker compose up -d postgres redis

# ② AI 추론 서버 (터미널 1) — 로컬 GPU 사용
cd ai_service
uv venv --python 3.12.14 --seed .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001

# ③ Django (터미널 2)
cd backend
uv venv --python 3.12.14 --seed .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python manage.py migrate && python manage.py createsuperuser
python manage.py runserver 0.0.0.0:8000

# ④ Celery Worker (터미널 3)
cd backend && source .venv/bin/activate
celery -A config worker --loglevel=INFO --concurrency=1
```

`.env`의 `POSTGRES_HOST`·`CELERY_BROKER_URL`·`FASTAPI_URL`이 `localhost`를 가리키는지 확인하세요. Redis·Worker 없이 화면만 확인하려면 `CELERY_TASK_ALWAYS_EAGER=1`(요청 안에서 즉시 실행), `DB_ENGINE=sqlite`를 지정합니다.

## 테스트

```bash
cd ai_service && python -m pytest                          # 45개 — GPU·YOLO 불필요 (가짜 Pose 모델·가짜 RF)
cd backend && DB_ENGINE=sqlite python manage.py test       # 35개 — FastAPI·카메라는 가짜로 대체 (PostgreSQL에서도 통과)
```

| 영역 | 무엇을 검증하나 |
|---|---|
| `ai_service/tests/test_model_and_inference.py` | 모델 3종 교차 검증(T01), Feature 순서가 바뀌면 거부, 사람별 집계 |
| `ai_service/tests/test_pose_pipeline.py` | 두 사람 분리(T04), 짧은 영상, 스쳐 간 사람 제외, 손상·초과 영상, 영상마다 Tracker 초기화, Window 전략 |
| `ai_service/tests/test_api.py` | `/health`, `/predict/video` 응답 계약과 오류 코드 |
| `backend/accounts/tests.py` | 회원가입·로그인·비로그인 차단, 내 정보·비밀번호 변경 |
| `backend/analyses/tests.py` | 업로드→저장(T02·T05·T06), 실패 처리(T09), 권한(T07), 관리자(T08), 카메라 수집, 입력 방식 비교, 시계 조정에도 정확한 처리 시간 |

## 환경 버전

학습 환경(6일차)과 맞춘 버전입니다. 특히 scikit-learn·numpy·joblib이 다르면 모델 로드가 거부됩니다(`model_loader.py`).

| | 버전 |
|---|---|
| Python | 3.12.14 |
| scikit-learn · numpy · joblib | 1.9.1 · 2.5.4 · 1.6.0 |
| ultralytics · torch · opencv-python · lap | 8.4.203 · 2.15.0 · 5.0.0.93 · 0.5.14 |
| Django · DRF · Celery | 6.1.3 · 3.18.3 · 5.7.0 |
