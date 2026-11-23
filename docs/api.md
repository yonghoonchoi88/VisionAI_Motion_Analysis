# API 계약

두 서버 사이, 그리고 브라우저와 서버 사이에 오가는 요청·응답 규칙입니다.

- **DRF API** (`backend/analyses/views.py`) — 브라우저 ↔ Django. 세션 로그인 + CSRF 토큰
- **FastAPI** (`ai_service/routers/predictions.py`) — Celery Worker ↔ AI 서버. DB 접근 없음

---

## 1. DRF API — `/api/analyses/`

| 기능 | 메서드 · 경로 | 응답 | 권한 |
|---|---|---|---|
| 분석 등록 | `POST /api/analyses/` | **202** + 작업 정보(`id`, `celery_task_id`, `status_url`) | 로그인 |
| 내 분석 목록 | `GET /api/analyses/` | 200, 페이지당 20개 | 본인 작업만 (관리자는 전체) |
| 분석 상세 | `GET /api/analyses/{id}/` | 200 + 사람별 예측 | 본인 또는 관리자, 아니면 **404** |
| 제목·메모 수정 | `PATCH /api/analyses/{id}/` | 200 | 본인 또는 관리자. `title`·`memo` 외 필드는 무시 |
| 분석 삭제 | `DELETE /api/analyses/{id}/` | 204 · 진행 중이면 **409** | 본인 또는 관리자 |
| 작업 상태 | `GET /api/analyses/{id}/status/` | 200 (폴링용 가벼운 응답) | 본인 또는 관리자 |

전체 교체(`PUT`)는 허용하지 않습니다(405). 다른 회원의 작업은 403이 아니라 404로 응답해 **존재 여부도 숨깁니다**.

### 분석 등록 요청 (`multipart/form-data`)

| 필드 | 영상 업로드 | IP Webcam | 검사 |
|---|---|---|---|
| `input_type` | `UPLOAD` | `CAMERA` | 필수 |
| `video_file` | 필수 | — | `.mp4 .avi .mov .mkv .webm`, 빈 파일 금지, 최대 200MB |
| `camera_url` | — | 필수 | http(s), **`ALLOWED_CAMERA_HOSTS`에 있는 host:port만**, 계정 정보 포함 금지 |
| `capture_seconds` | — | 선택 (기본 8) | 5~10초 |
| `title`, `memo` | 선택 | 선택 | 제목 100자 |
| `expected_action` | 선택 | 선택 | 6개 Action 중 하나 (정답이 아니라 비교용 사용자 선택) |

검사에 실패하면 **400** + 필드별 메시지(예: `{"camera_url": ["허용되지 않은 카메라입니다: …"]}`). 작업은 만들어지지 않습니다.

### 작업 상태 응답 (`GET /api/analyses/{id}/status/`)

```json
{
  "id": 9,
  "status": "SUCCESS",            "status_display": "완료",
  "stage": "DONE",                "stage_display": "완료",
  "is_finished": true,
  "error_message": "",
  "result_status": "OK",          "result_message": "1명의 동작을 예측했습니다.",
  "video_duration_sec": 8.0,
  "inference_time_sec": 4.08,     "total_time_sec": 18.7,
  "predictions": [
    {"track_id": 1, "action_id": "A006", "action_label": "wave", "action_name_ko": "손 흔들기",
     "confidence": 0.3, "margin": 0.11, "runner_up_label": "leg_raise_lower", "runner_up_name_ko": "한쪽 다리를 들었다 내리기",
     "is_uncertain": true, "window_count": 1, "window_mode": "whole_track",
     "valid_frame_ratio": 1.0, "duration_sec": 8.0, "probabilities": {"wave": 0.3, "...": 0.0}}
  ]
}
```

| 필드 | 값 |
|---|---|
| `status` | `PENDING` → `STARTED` → `SUCCESS` / `FAILURE` (Celery 표준 표기) |
| `stage` | `QUEUED`(대기) → `CAPTURING`(영상 수집, IP Webcam만) → `INFERRING`(AI 분석) → `DONE`(완료) |
| `total_time_sec` | 대기(생성→시작) + Worker 처리 시간(단조 시계) |
| `margin` · `runner_up_label` | 1등 − 2등 확률과 2등 동작. 판단 불확실 기준 ② (예전 기록은 `null`) |

---

## 2. FastAPI — AI 추론 서버

### `GET /health`

```json
{"status": "ok", "model_ready": true, "error": null, "device": "cpu", "window_strategy": "auto",
 "decision_rule": {"min_confidence": 0.55, "min_margin": 0.0},
 "model": {"model_type": "RandomForestClassifier", "n_features": 1140, "sequence_steps": 30,
           "checksum_verified": true, "classes": ["..."]}}
```

모델 로드에 실패해도 서버는 뜨고 `status: degraded`, `model_ready: false`, `error`에 원인을 담습니다. Docker healthcheck가 `model_ready`를 확인합니다.

### `POST /predict/video` (multipart `file`)

예시는 원본 day04 `walk_turn_walk.mp4`를 Docker(CPU)에서 분석한 실제 결과입니다.

```json
{
  "status": "OK",
  "message": "1명의 동작을 예측했습니다.",
  "person_count": 1,
  "predictions": [
    {"track_id": 1, "action_id": "A003", "action_label": "walk_turn_walk", "confidence": 0.66,
     "margin": 0.57, "runner_up_label": "wave", "is_uncertain": false, "probabilities": {"walk_turn_walk": 0.66, "...": 0.0},
     "window_count": 1, "window_mode": "whole_track",
     "seen_frames": 265, "valid_frames": 265, "valid_frame_ratio": 1.0, "duration_sec": 10.6}
  ],
  "skipped_tracks": [],
  "video": {"filename": "walk_turn_walk.mp4", "fps": 25.0, "fps_assumed": false, "frame_count": 265,
            "processed_frames": 265, "duration_sec": 10.6, "width": "…", "height": "…"},
  "uncertain_threshold": 0.55,
  "decision_rule": {"min_confidence": 0.55, "min_margin": 0.0},
  "processing_time_sec": 7.2
}
```

### 응답 해석 규칙 (Celery Worker 기준)

| FastAPI 응답 | 의미 | Worker 처리 (`backend/analyses/services.py` · `tasks.py`) |
|---|---|---|
| 200 `status: OK` | 사람별 예측 1건 이상 | **SUCCESS** + `PersonPrediction` 저장 |
| 200 `status: NO_PERSON` | 영상 정상, 사람 미검출 | **SUCCESS** (예측 0건, 메시지 표시) |
| 200 `status: NO_VALID_TRACK` | 사람이 판단에 필요한 만큼 오래 보이지 않음 | **SUCCESS** (예측 0건, `skipped_tracks` 사유) |
| 400 `UNSUPPORTED_FORMAT` · `EMPTY_FILE` | 형식 · 빈 파일 | **FAILURE** + 사유 (재시도 안 함) |
| 413 `FILE_TOO_LARGE` | 200MB 초과 | FAILURE + 사유 |
| 422 `VIDEO_UNREADABLE` · `VIDEO_TOO_LONG` | 손상 · 60초 초과 | FAILURE + 사유 |
| 503 `MODEL_NOT_READY`, 연결 실패, 타임아웃 | AI 서버 일시 장애 | **10초 뒤 재시도, 최대 3회** → 그래도 실패면 FAILURE |
| 500 `INFERENCE_ERROR` | 예상하지 못한 오류 | FAILURE + 사유 |

오류 응답 형식: `{"detail": {"code": "VIDEO_UNREADABLE", "message": "영상을 열 수 없습니다. …"}}`

**핵심 설계**: HTTP 상태 코드는 "분석을 할 수 있었는가", `status` 필드는 "무엇을 찾았는가"를 나타냅니다. 그래서 사람이 없는 영상(정상 처리)과 손상된 영상(처리 불가)이 구분됩니다.
