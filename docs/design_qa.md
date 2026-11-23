# 설계 질문 답변 (Design Q&A)

과제 문서의 **"답해야 할 질문 · 핵심 설계 질문 · 구현 전에 정할 문제"** 에 대한 답변입니다.
각 답은 **결론 → 이유 → 근거(코드 위치·증거)** 순서로 정리했습니다.

| 장 | 질문 | 바로가기 |
|---|---|---|
| 0 | IP Webcam: 누가 읽고, 언제 끝내고, 결과는 어떻게 받나 | [Q0](#q0-ip-webcam으로-짧은-구간을-분석하는-방법) |
| 1-1 | joblib만으로 MP4를 분류할 수 없는 이유 / 30×38 / Feature 순서 / `classes_` | [Q1~Q4](#1-모델-산출물-문제-1-1) |
| 1-2 | 모델 외에 필요한 처리 기능이 같은 결과를 내는가 | [Q5](#q5-원본-처리-기능을-그대로-재사용했는가-문제-1-2) |
| 2-2 | FastAPI 반환과 Django ORM 저장을 왜 나누나 | [Q6](#q6-fastapi가-json을-반환하는-것과-django가-orm으로-저장하는-것을-왜-구분하나) |
| 3-2 | 사용자가 Action Label·SUCCESS를 수정하면 안 되는 이유 / 삭제 정책 | [Q7~Q8](#3-권한과-crud-문제-3-2) |
| 4 | ERD 설계 원칙 | [Q9](#q9-erd에서-지킨-설계-원칙) |
| 5 | 두 사람 / 사람 없음 vs 손상 / 낮은 점수 / 반복 요청 | [Q10~Q13](#5-fastapi-구현-전에-정할-문제) |
| 7 | 동기 대기 vs 작업 ID 즉시 반환 / 커밋 시점 | [Q14~Q15](#7-비동기-처리) |
| 8 | Docker 서비스 구성의 결정 | [Q16](#q16-docker-compose에서-내린-결정) |
| 9-2 | 차트는 ORM 집계인가, FastAPI의 DB 조회인가 | [Q17](#q17-대시보드-차트는-django-orm-집계로-그려야-하나-fastapi가-db를-조회해-그려야-하나) |
| 추가 | 남의 작업 접근 시 403이 아니라 404인 이유 | [Q18](#q18-다른-회원의-작업에-접근하면-왜-403이-아니라-404를-주나) |

---

## Q0. IP Webcam으로 짧은 구간을 분석하는 방법

> 구현 전에 누가 카메라 영상을 읽는지, 언제 수집을 끝내는지, 브라우저가 결과를 어떻게 받는지 설명해 보세요. (0장)

| 질문 | 답 | 코드 |
|---|---|---|
| **누가 읽나** | 브라우저가 아니라 **Celery Worker**가 카메라 HTTP 주소(`/video`)에 직접 연결해 읽습니다. 브라우저 미리보기는 사용자 확인용일 뿐 분석에 쓰이지 않습니다. | `analyses/camera_capture.py` `capture_clip()` |
| **언제 끝내나** | 연결 후 **설정한 시간(5~10초, 기본 8초)이 지나면** 즉시 연결을 끊습니다. 도중에 30회 연속 읽기에 실패하면 "연결 끊김"으로 실패 처리합니다. 8초는 *입력 영상의 길이*이고, 연결·저장·추론 시간은 별도입니다. | 같은 파일, `seconds`·`MAX_CONSECUTIVE_FAILS` |
| **결과는 어떻게 받나** | 분석 시작 요청에 대해 Django가 **작업 ID를 즉시(202)** 돌려주고, 브라우저는 그 ID로 `GET /api/analyses/{id}/status/`를 **1초마다 조회**합니다. 완료되면 같은 응답에 사람별 결과가 들어 있습니다. WebSocket은 쓰지 않았습니다(선택 확장). | `static/js/analyses.js` `poll()` |

수집한 구간은 30fps·640px의 **WebM**으로 저장되어 이후 처리는 업로드 영상과 완전히 같습니다. 허용 목록(`ALLOWED_CAMERA_HOSTS`)에 없는 주소는 서버가 호출하지 않습니다 — 사용자가 임의 URL을 넣어 서버가 외부·내부망으로 요청을 보내게 만드는 공격(SSRF)을 막기 위해서입니다.

---

## 1. 모델 산출물 (문제 1-1)

### Q1. 왜 `rf_action_baseline.joblib` 하나만으로는 원본 MP4를 바로 분류할 수 없나?

**결론**: joblib 안에는 **"숫자 1,140개를 받아 6개 중 하나를 고르는 분류기"만** 들어 있고, 영상을 그 숫자로 바꾸는 과정은 들어 있지 않기 때문입니다.

- 로드해 보면 `type(model)`이 `RandomForestClassifier` **단독**입니다. 전처리를 포함한 `Pipeline`이 아닙니다.
- 영상이 모델 입력이 되기까지 필요한 단계:
  `MP4 → Frame → YOLO Pose(관절 17개) → 골반 중심·어깨폭 정규화 + 관절 각도 4개(38개) → 30 Step 리샘플 → 1,140개로 펼치기`
- 이 단계는 원본 `src/`의 코드에 있으므로 **수정 없이 복사**해 재사용했습니다.

**근거**: `ai_service/services/pose_pipeline.py`, `reports/evidence/T01_model_load.txt`

### Q2. `30 Step × 38 Feature = 1,140 Feature`는 무엇을 뜻하나?

- **38** = 한 순간(Frame)의 자세
  - 관절 17개의 x·y 좌표 = 34개 (골반 중심을 원점으로 옮기고 어깨폭으로 나눠 **사람 크기·위치와 무관**하게 만든 값)
  - 관절 각도 4개 = 좌·우 팔꿈치, 좌·우 무릎
- **30** = 시간 축. 영상 길이가 3초든 10초든 **30개 시점으로 균등하게 다시 샘플링**해 길이를 고정합니다.
- **1,140** = 30개 시점의 38개 값을 순서대로 이어 붙인 한 줄. 이름은 `t00_nose_x, t00_nose_y, … t00_right_knee_angle, t01_nose_x, … t29_right_knee_angle`입니다.

### Q3. 모델이 기대하는 Feature의 순서를 바꾸면 어떤 문제가 생기나?

**결론**: **오류 없이 틀린 답을 냅니다.** 가장 위험한 형태의 버그입니다.

- 이 모델은 numpy 배열로 학습되어 열 **이름**을 모릅니다(`feature_names_in_` 없음, `expects DataFrame: False`). 오직 **위치**로 "37번째 값이 크면 …"처럼 판단합니다.
- 순서가 바뀌어도 개수가 1,140개면 `predict()`는 정상 실행되고, 엉뚱한 관절 값으로 판단한 결과를 그럴듯하게 돌려줍니다.

**대응**: 서버 시작 시 `model_loader.py`가 세 파일을 교차 검증하고, 하나라도 틀리면 **서버를 추론 불가 상태로 두고 `/health`에 원인을 보고**합니다.
- i번째 컬럼이 `t{i // 38:02d}_`로 시작하는지 (Step 순서)
- 모든 Step이 같은 38개 이름을 같은 순서로 반복하는지
- 테스트: 두 컬럼의 순서만 바꾼 JSON으로 로드하면 `ModelSpecError` (`test_broken_feature_columns_detected`)

### Q4. `model.classes_`는 어떤 역할을 하나?

**결론**: `predict_proba()`가 돌려주는 **확률 6개가 각각 어느 동작인지 알려 주는 열 이름표**입니다.

- scikit-learn은 학습할 때 라벨을 **알파벳순으로 정렬**해 저장합니다. 실제 값:
  ```
  [0] bend_return  [1] drink_return  [2] leg_raise_lower  [3] sit_stand  [4] walk_turn_walk  [5] wave
  ```
- 과제 문서의 A001~A006 순서와 다릅니다. 예를 들어 `[1]`은 A002가 아니라 **A004 drink_return**입니다.
- 그래서 결과를 만들 때 **위치가 아니라 이름으로** 연결합니다: `label = classes_[argmax]` → `ACTION_IDS[label]`.

**근거**: `ai_service/services/action_inference.py`, `model_loader.py`의 `ACTION_IDS`, `T01_model_load.txt`의 `[1] A004 drink_return` 출력

---

## Q5. 원본 처리 기능을 그대로 재사용했는가? (문제 1-2)

| 확인 과제 | 결과 | 근거 |
|---|---|---|
| 새 환경에서도 COCO 17 Keypoint가 나오는가 | 같은 `yolo11n-pose.pt`, 같은 ultralytics 8.4.203 | `requirements.txt` |
| 학습 당시와 같은 좌표·각도 계산인가 | `feature_utils.py` 등 5개를 **한 글자도 수정하지 않고** 복사, `feature_from_keypoints()` 그대로 호출 | `ai_service/src/` |
| 30 Step·1,140 Feature 조건이 맞는가 | 시작 시 교차 검증 + 추론 직전 shape 검사 | T01 |
| 7일차 기능을 재사용할 수 있는가 | `TimedFeatureBuffer`(5초 Window), `is_ready()`, `resample_sequence()` 재사용 | `pose_pipeline.py` |
| 두 사람이 나타나면 ID별 예측이 분리되는가 | SBU 2인 영상 → Track 1·2 각각 예측 | T04 |
| 같은 Pose 모델을 정상 로드하는가 | `/health` → `model_ready: true` | Docker 실행 로그 |

라이브러리 버전도 학습 환경과 맞췄습니다(Python 3.12.14, scikit-learn 1.9.1, numpy 2.5.4 등). joblib은 내부적으로 pickle이라 **본인이 만든 파일에만** 사용하고, 사용자 업로드 파일에는 절대 쓰지 않습니다.

**이관하며 발견한 차이**: 6일차 학습은 *Clip 전체*를 30 Step으로 만들었는데, 7일차 실시간 방식(5초 Window)을 그대로 쓰니 10초 영상이 오분류됐습니다. 학습과 추론의 입력 단위를 맞추는 `window.strategy`를 추가해 해결했습니다 → [`ab_window_strategy.md`](../reports/evidence/ab_window_strategy.md)

---

## Q6. FastAPI가 JSON을 반환하는 것과 Django가 ORM으로 저장하는 것을 왜 구분하나?

> 핵심 설계 질문 (2-2)

**결론**: DB의 **주인을 하나(Django)로** 정하고, AI 서버는 **입력을 받아 답만 내는 계산기**로 두기 위해서입니다.

| 이유 | 설명 |
|---|---|
| 스키마 주인이 하나 | 테이블 구조는 `models.py`와 마이그레이션이 관리합니다. FastAPI가 SQLAlchemy로 같은 테이블을 따로 정의하면 두 정의가 어긋나는 순간 데이터가 깨집니다. |
| 장애 격리 | 추론 중 YOLO가 죽어도 DB에는 아무 일도 일어나지 않고, Worker가 `FAILURE`와 사유를 기록합니다. 오늘 torchvision 오류 때 실제로 fastapi만 멈추고 DB는 무사했습니다. |
| 교체·확장 | FastAPI가 상태를 갖지 않으므로 GPU 서버로 옮기거나 여러 대로 늘려도 DB 설정을 건드릴 필요가 없습니다. |
| 보안 | DB 비밀번호는 django·celery 컨테이너에만 있습니다. `compose.yaml`의 fastapi에는 DB 관련 환경변수가 없습니다. |
| 테스트 | FastAPI는 DB 없이 가짜 Pose 모델로 24개 테스트를 돌릴 수 있습니다. |

---

## 3. 권한과 CRUD (문제 3-2)

### Q7. 일반 사용자가 예측 Action Label이나 `SUCCESS` 상태를 수정하게 두면 안 되는 이유는?

**결론**: 그 값들은 **사용자의 입력이 아니라 시스템이 측정한 결과**이기 때문입니다.

- 사용자가 고칠 수 있으면 대시보드의 동작별 건수·성공률·불확실 비율이 의미를 잃고, 모델이 실제로 무엇을 예측했는지 기록(감사 추적)도 사라집니다.
- 사용자의 판단을 남기고 싶다면 별도 필드를 씁니다: **`expected_action`**(수행한 동작, 선택 입력). 예측과 섞지 않으므로 "선택 동작과 일치/다름"을 비교할 수 있습니다.

**구현**: `AnalysisJobUpdateSerializer`는 `title`·`memo`만 받습니다. `status`·`result_status`를 보내도 무시되고, 전체 교체(PUT)는 405입니다.
**근거**: `test_patch_only_changes_title_and_memo`

### Q8. 삭제 정책과 관련 예측 데이터 처리 규칙

| 규칙 | 이유 |
|---|---|
| **완료·실패한 작업만 삭제**, 진행 중이면 409 | Worker가 처리 중인 작업이 사라지면 결과를 저장할 곳이 없어집니다. (그래도 사라진 경우 Worker는 `MISSING`으로 조용히 종료) |
| `PersonPrediction`은 **CASCADE**로 함께 삭제 | 작업 없는 예측은 의미가 없습니다. |
| 영상 파일도 삭제 (**DB 커밋 후**) | 파일만 남아 저장소가 차는 것을 막고, DB 삭제가 취소되면 파일을 지우지 않도록 커밋 이후에 지웁니다. |

**근거**: `test_delete_removes_predictions_and_blocks_running`

---

## Q9. ERD에서 지킨 설계 원칙

| 원칙 (과제 4장) | 구현 |
|---|---|
| 한 사용자 → 여러 작업, 한 작업 → 여러 사람 예측 | `auth_user 1:N AnalysisJob 1:N PersonPrediction` |
| `track_id`는 작업 안의 번호이지 영구 ID가 아님 | `UNIQUE(job, track_id)` — 작업이 다르면 같은 번호가 다시 나올 수 있음 |
| 영상 파일은 DB 바이너리로 저장하지 않음 | `FileField` → MEDIA 저장소에 파일, DB에는 경로만 |
| `expected_action`을 검증된 정답과 혼동하지 않음 | 필드 설명에 "사용자 선택값, 정답 아님" 명시, 예측과 별도 저장 |
| 상태 표기 통일 | Celery 표준과 같은 `PENDING / STARTED / SUCCESS / FAILURE`, 화면 진행 단계는 별도 `stage` |
| FastAPI에 ORM 모델을 만들지 않음 | `ai_service`에는 DB 코드가 없음 |

---

## 5. FastAPI 구현 전에 정할 문제

### Q10. 영상에 두 사람이 있다면?

`predictions` 목록에 **사람마다 한 항목**씩 넣습니다. ByteTrack이 붙인 `track_id`로 사람을 구분하고, Track마다 Feature Buffer를 따로 둡니다(원본 `MultiPersonStateManager` 재사용). Worker는 항목 수만큼 `PersonPrediction`을 만듭니다. → T04

### Q11. 사람이 없는 영상과 손상된 영상을 Worker가 어떻게 구분하나?

| 상황 | FastAPI 응답 | Worker 처리 |
|---|---|---|
| 영상 정상, 사람 없음 | **200** `status: NO_PERSON`, 빈 `predictions`, `"영상에서 사람을 찾지 못했습니다."` | **SUCCESS** (예측 0건 + 메시지) — 분석은 제대로 끝났으므로 |
| 사람이 너무 짧게 보임 | **200** `status: NO_VALID_TRACK` + `skipped_tracks` 사유 | SUCCESS (예측 0건 + 메시지) |
| 파일을 읽을 수 없음 | **422** `detail.code: VIDEO_UNREADABLE` | **FAILURE** + 사유 — 다시 해도 같으므로 재시도 안 함 |
| AI 서버 준비 안 됨·연결 실패 | **503** / 연결 오류 | 최대 3회 재시도 후 FAILURE |

핵심은 **HTTP 상태 코드로 "분석을 했는가"를, `status` 필드로 "무엇을 찾았는가"를** 나눈 것입니다. 이 계약은 `routers/predictions.py` 맨 위 주석과 [`api.md`](api.md)에 정리했습니다. → T09 (작업 #2)

### Q12. 예측 점수가 낮다면? 어느 기준에서 `판단 불확실`을 표시하나?

**기준: Confidence < 0.55** — 7일차 실시간 실습의 `confidence_threshold`와 같은 값입니다.

데이터로도 확인했습니다. 오늘 결과를 높은 순으로 늘어놓으면:
```
0.63 ~ 0.67  동작 전체를 본 정답                → 확신
──────────── 0.55 ────────────
0.26 ~ 0.32  새 사람·새 장소, ID가 쪼개진 조각   → 불확실 (1순위는 대부분 정답)
0.23 ~ 0.25  학습하지 않은 동작 (SBU 악수)      → 불확실
```
기준을 0.3으로 낮추면 **학습하지 않은 동작까지 '확신'으로 표시**하게 되므로 낮추지 않았습니다. 화면에는 Confidence와 함께 Class별 확률, 유효 Frame 비율(입력 품질)을 같이 보여 줍니다. → [`ab_capture_condition.md`](../reports/evidence/ab_capture_condition.md)

### Q13. 영상이 여러 번 들어온다면?

모델은 **서버 시작 시 한 번** 로드해 `app.state`에 두고 모든 요청이 재사용합니다(`main.py`의 `lifespan`). 시작할 때 빈 화면으로 한 번 추론(Warm-up)해 첫 요청이 느려지지 않게 했습니다.
YOLO의 추적 상태는 모델 객체에 붙어 있으므로 **영상마다 Tracker를 초기화**하고, 동시에 두 영상이 섞이지 않도록 **Lock으로 한 번에 한 영상**만 처리합니다. (`test_tracker_reset_every_video`)

---

## 7. 비동기 처리

### Q14. 요청 함수에서 분석이 끝날 때까지 기다리는 방식과 작업 ID를 즉시 반환하는 방식의 사용자 경험 차이

| | 끝날 때까지 기다림 (동기) | 작업 ID 즉시 반환 (비동기, 채택) |
|---|---|---|
| 응답 시간 | 업로드 7초, IP Webcam 18초 동안 브라우저가 멈춤 | 파일 전송이 끝나는 즉시 202 + 작업 ID |
| 진행 상황 | 알 수 없음 | `대기 → 영상 수집 → AI 분석 → 완료` 단계 표시 |
| 실패 시 | 중간에 끊기면 결과도, 기록도 없음 | 실패도 사유와 함께 DB에 남아 나중에 확인 가능 |
| 서버 | 요청마다 웹 서버 작업자가 묶여 다른 사용자도 느려짐 | 웹 서버는 즉시 풀려나고, 무거운 일은 Worker가 순서대로 처리 |
| 새로고침·이탈 | 결과를 잃음 | 이력에서 다시 확인 |

### Q15. DB 저장이 끝나기 전에 Celery 작업이 실행되지 않게 하려면?

`transaction.on_commit()`으로 **DB 커밋이 확정된 뒤에만** Celery에 등록합니다.

커밋 전에 등록하면, 빠른 Worker가 아직 DB에 보이지 않는 작업을 찾다가 `DoesNotExist`로 실패할 수 있습니다. Task ID는 먼저 만들어 저장해 두므로 등록 전부터 화면과 Admin에서 추적할 수 있습니다. (`views.py` `enqueue_analysis()`)

---

## Q16. Docker Compose에서 내린 결정

| 결정 | 이유 |
|---|---|
| 서비스끼리 `localhost`가 아니라 **서비스 이름**으로 통신 | 컨테이너마다 `localhost`는 자기 자신입니다. `FASTAPI_URL=http://fastapi:8001` |
| 업로드 영상은 `media_data` 볼륨을 django·celery가 공유 | Worker가 Django가 저장한 파일을 같은 경로로 읽음. FastAPI에는 **파일 자체를 HTTP로** 전송 |
| PostgreSQL은 `postgres_data` 볼륨 | 재시작 후에도 데이터 유지 → T10 |
| `django`만 사용자에게 공개, `fastapi` 포트는 **개발용으로만** 공개 | `/docs`로 API를 직접 확인하기 위함. 운영에서는 닫아도 django·celery는 내부망으로 접근 |
| 바깥 포트를 `*_HOST_PORT` 변수로 | 다른 프로젝트와 포트가 겹쳐도 `.env` 한 줄로 해결 |
| healthcheck + `depends_on: service_healthy` | 모델이 준비되기 전에 Worker가 요청을 보내지 않도록 |
| 모델 파일은 이미지에 넣지 않고 읽기 전용 볼륨 | Git·이미지에 대용량 파일을 넣지 않고, 모델만 바꿔 끼울 수 있게 |
| IP Webcam 주소는 허용 목록만 | 사용자가 임의 URL을 넣어 서버가 외부로 요청하는 것을 차단 |

---

## Q17. 대시보드 차트는 Django ORM 집계로 그려야 하나, FastAPI가 DB를 조회해 그려야 하나?

> 질문 (9-2)

**Django ORM 집계**입니다. Q6의 원칙 그대로 FastAPI는 DB에 접근하지 않습니다. 관리자 화면은 Django의 책임이고 데이터도 Django가 관리합니다.

`page_views.py`에서 `values().annotate(Count())`로 동작별·상태별 건수를 집계하고, `json_script`로 템플릿에 넘겨 `dashboard.js`가 Chart.js로 그립니다.

> 구현 중 주의한 점: 모델의 기본 정렬(`Meta.ordering`)이 있으면 Django가 정렬 필드까지 `GROUP BY`에 넣어 집계가 잘게 쪼개집니다. 그래서 집계 전에 `.order_by()`로 정렬을 지웠습니다.

---

## Q18. 다른 회원의 작업에 접근하면 왜 403이 아니라 404를 주나?

403("권한 없음")은 **그 번호의 작업이 존재한다**는 사실을 알려 줍니다. 번호를 1, 2, 3…으로 바꿔 가며 남의 기록이 몇 개인지 알아낼 수 있습니다.

일반 회원에게는 조회 범위 자체를 **자기 작업으로 제한**(`get_queryset()`)하므로, 남의 작업은 "없는 것"과 똑같이 404가 됩니다. 관리자(`is_staff`)는 전체를 봅니다. → T07 `test_user_cannot_touch_others_jobs`
