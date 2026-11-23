# VisionAI Motion Analysis

6개 동작 인식 모델(6일차 Random Forest)을 재사용한 Vision AI 웹 서비스
(Django·DRF · Celery · Redis · FastAPI · PostgreSQL)

## 모델 준비
`ai_service/models/`에 다음 파일을 둡니다. (`*.joblib`, `*.pt`는 Git 제외)
- `rf_action_baseline.joblib` — 원본 프로젝트 `artifacts/day06/`에서 복사
- `yolo11n-pose.pt` — 원본 프로젝트 루트에서 복사
