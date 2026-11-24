# 단계별 수동 검증 기록 (9~10일차 개발 가이드 1~11단계 · 과제 T01~T10)

개발 가이드의 원칙은 **"한 단계씩 만들고 → 수동 검증 → 다음 단계"**입니다. 이 문서는 단계마다 **무엇을 실행하고, 무엇이 나오면 통과이며, 증거를 어디에 남겼는지**를 정리합니다.
자동 테스트(80개)는 회귀 방지용이고, 아래 수동 검증은 실제 5개 컨테이너에서 브라우저 · Swagger UI · curl · Django Admin · Celery 로그 · PostgreSQL로 확인한 기록입니다.

## 0. 준비

```bash
cd ~/ai_vision/VisionAI_Motion_Analysis
cp -n .env.example .env                                   # 이미 있으면 그대로 둠
docker compose up --build -d && docker compose ps          # 5개 모두 Up (fastapi는 healthy까지 약 1분)
docker compose exec django python manage.py createsuperuser
set -a; . ./.env; set +a                                   # 아래 명령에서 포트·DB 이름을 .env 값으로 사용
DJ=http://localhost:${DJANGO_HOST_PORT:-8000}; AI=http://localhost:${FASTAPI_HOST_PORT:-8001}
VID=~/ai_vision/subject10_pose_action/data/day05/videos     # 학습에 쓰지 않은 S02·S03 영상
(cd ai_service && python -m scripts.make_test_videos --from $VID/S02/S02_A001_T01.mp4)   # 비정상 입력 4종
psql_q() { docker compose exec -T postgres psql -U "${POSTGRES_USER:-vision}" -d "${POSTGRES_DB:-vision_action}" -c "$1"; }
```

증거 파일은 `reports/evidence/` 아래에 둡니다. 화면 캡처는 `screenshots/Txx_설명.png`, 터미널 출력은 `Txx_설명.txt`.

## 1~11단계 검증표

| 단계 | 검증 (실행) | 통과 기준 | 증거 |
|---|---|---|---|
| **1. 모델 준비** | `cd ai_service && python -m services.model_loader`<br>`(cd ai_service/models && sha256sum -c SHA256SUMS.txt)` | `[T01 PASS]` · 1140 = 30×38 · Class 6개 · `SHA256 verified: True` · 해시 모두 `OK` | [T01_model_load.txt](../reports/evidence/T01_model_load.txt) |
| **2. FastAPI 단독** | `curl -s $AI/api/v1/health \| python3 -m json.tool`<br>`curl -s -F "file=@$VID/S03/S03_A006_T01.mp4" $AI/api/v1/predict/ \| python3 -m json.tool`<br>브라우저 `$AI/docs` → `POST /api/v1/predict/` → Try it out | `model_ready: true` · 정상 영상 → `status: OK` + `track_id` · `action_label` · `confidence` · `margin` | [T01_swagger_predict.png](../reports/evidence/screenshots/T01_swagger_predict.png) |
| 2. 비정상 입력 | `for f in no_person.mp4 too_short.mp4 broken.mp4 notes.txt; do echo "== $f"; curl -s -w ' [HTTP %{http_code}]\n' -F "file=@ai_service/samples/$f" $AI/api/v1/predict/; done` | no_person → 200 `NO_PERSON` · too_short → 200 `NO_VALID_TRACK` · broken → **422** `VIDEO_UNREADABLE` · notes.txt → **400** `UNSUPPORTED_FORMAT` | [T09_fastapi_bad_inputs.txt](../reports/evidence/T09_fastapi_bad_inputs.txt) |
| **3. Django·PostgreSQL·Admin** | `docker compose exec django python manage.py showmigrations analyses`<br>`psql_q '\dt analyses_*'`<br>브라우저 `$DJ/admin/` | 마이그레이션 0001~0004 `[X]` · 테이블 2개 · Admin에 분석 작업·사람별 예측, 상단 '분석 대시보드' 링크 | [T08_admin.png](../reports/evidence/screenshots/T08_admin.png) |
| **4. 회원·권한** | 브라우저: 가입 → 로그인 → 로그아웃 · 같은 아이디로 가입 · 비밀번호 불일치<br>`curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' $DJ/analyses/` | 중복·불일치는 폼 오류 · 비로그인 화면 → **302 /accounts/login/** · 일반 회원 `/dashboard/` → **403** | [T07_signup_errors.png](../reports/evidence/screenshots/T07_signup_errors.png) |
| **5. DRF 생성·조회** | `docker compose stop celery` → 브라우저 `$DJ/api/v1/analyses/` (Browsable API)에서 업로드 POST → `psql_q 'SELECT id,status,stage FROM analyses_analysisjob ORDER BY id DESC LIMIT 3;'`<br>회원 B로 로그인해 A의 `$DJ/api/v1/analyses/<id>/` 열기 | 202 + 작업 ID · DB에 **PENDING** · 비로그인 403 · **다른 회원 404** · 확장자 틀림·빈 파일 400 | [T07_other_user_404.png](../reports/evidence/screenshots/T07_other_user_404.png) |
| **6. Redis·Celery** | `docker compose logs -f celery` (다른 창) → `docker compose start celery` | 멈춰 있던 PENDING 작업을 Worker가 받음(`Task analyses.tasks.run_analysis[...] received`) → STARTED → SUCCESS | [T05_celery_worker_resume.txt](../reports/evidence/T05_celery_worker_resume.txt) |
| **7. 업로드 통합** | `bash scripts/smoke_e2e.sh $VID/S03/S03_A006_T01.mp4`<br>`docker compose stop fastapi` → 같은 명령 → `docker compose start fastapi` | 202가 **1초 안에** 오고 상태가 PENDING → STARTED → SUCCESS · DB 조회 값 = API 값 · FastAPI 중단 시 재시도 3회 뒤 **FAILURE + "AI 서버에 연결할 수 없습니다"** | [smoke 로그](../reports/evidence/smoke/) · [T09_ai_server_down.png](../reports/evidence/screenshots/T09_ai_server_down.png) |
| **8. IP Webcam** | Celery 컨테이너에서 카메라 확인:<br>`CAM=${ALLOWED_CAMERA_HOSTS%%,*}; docker compose exec celery python -c "import cv2; c=cv2.VideoCapture('http://$CAM/video'); print('opened', c.isOpened())"`<br>브라우저 IP Webcam 탭 → 8초 수집 → 도중에 앱 종료로 끊김 시험 | `opened True` · 8초 수집 → 결과 저장 · 앱 종료 시 **FAILURE "수집 중 카메라 연결이 끊겼습니다"** · 허용 목록 밖 주소는 400 | [T03_webcam.png](../reports/evidence/screenshots/T03_webcam.png) · [T09_camera_lost.png](../reports/evidence/screenshots/T09_camera_lost.png) |
| **9. 사용자 화면** | 브라우저 `$DJ/analyses/`: 업로드 → 단계 표시 → 결과 → **새로고침** | 대기 → AI 분석 → 완료 · 사람별 Track·동작·신뢰도·2등과 차이 · 새로고침 후에도 이력·결과 유지 | [T02_upload_result.png](../reports/evidence/screenshots/T02_upload_result.png) · [T06_after_refresh.png](../reports/evidence/screenshots/T06_after_refresh.png) |
| **10. 관리자 대시보드** | 관리자로 `$DJ/dashboard/` → 새 분석 1건 → 다시 열기<br>`psql_q 'SELECT action_label, count(*) FROM analyses_personprediction GROUP BY 1 ORDER BY 1;'`<br>`psql_q 'SELECT status, count(*) FROM analyses_analysisjob GROUP BY 1;'` | 차트 막대·도넛의 숫자 = SQL 결과 · 새 분석 후 숫자 +1 · 일반 회원 403 | [T08_dashboard.png](../reports/evidence/screenshots/T08_dashboard.png) · [T08_dashboard_vs_db.txt](../reports/evidence/T08_dashboard_vs_db.txt) |
| **11. Docker 통합** | `docker compose restart && sleep 60 && docker compose ps`<br>`docker compose down && docker compose up -d` (`-v` 금지 — 볼륨 삭제)<br>`psql_q 'SELECT count(*) FROM analyses_analysisjob;'` | 5개 Up · 재시작 전후 작업 수 동일 · 화면 이력 유지 · 재시작 후 새 분석도 성공 | [T10_after_restart.txt](../reports/evidence/T10_after_restart.txt) |

## 과제 T01~T10 ↔ 증거

| 과제 테스트 | 위 단계 | 증거 (캡처 + 자동 테스트) |
|---|---|---|
| T01 모델 불러오기 | 1 · 2 | [T01_model_load.txt](../reports/evidence/T01_model_load.txt) · [T01_swagger_predict.png](../reports/evidence/screenshots/T01_swagger_predict.png) · `test_t01_real_model_loads` |
| T02 영상 업로드 | 7 · 9 | [T02_upload_result.png](../reports/evidence/screenshots/T02_upload_result.png) · [smoke 로그](../reports/evidence/smoke/) · [holdout_eval.md](../reports/evidence/holdout/holdout_eval.md) |
| T03 IP Webcam | 8 | [T03_webcam.png](../reports/evidence/screenshots/T03_webcam.png) |
| T04 사람별 결과 | 7 (2인 영상) | [T04_two_tracks.png](../reports/evidence/screenshots/T04_two_tracks.png) · `test_two_people_get_separate_tracks` |
| T05 비동기 처리 | 5 · 6 · 7 | [smoke 로그](../reports/evidence/smoke/)의 "202 · 응답 0.0x초" · [T05_celery_worker_resume.txt](../reports/evidence/T05_celery_worker_resume.txt) |
| T06 영구 저장 | 9 · 10 | [T06_after_refresh.png](../reports/evidence/screenshots/T06_after_refresh.png) · smoke 로그의 PostgreSQL 조회 |
| T07 CRUD 권한 | 4 · 5 | [T07_other_user_404.png](../reports/evidence/screenshots/T07_other_user_404.png) · [T07_signup_errors.png](../reports/evidence/screenshots/T07_signup_errors.png) · `test_user_cannot_touch_others_jobs` |
| T08 관리자 | 3 · 10 | [T08_admin.png](../reports/evidence/screenshots/T08_admin.png) · [T08_dashboard.png](../reports/evidence/screenshots/T08_dashboard.png) · [T08_dashboard_vs_db.txt](../reports/evidence/T08_dashboard_vs_db.txt) · [T08_dashboard_forbidden.png](../reports/evidence/screenshots/T08_dashboard_forbidden.png) |
| T09 예외 처리 | 2 · 7 · 8 | [T09_fastapi_bad_inputs.txt](../reports/evidence/T09_fastapi_bad_inputs.txt) · [T09_broken_file.png](../reports/evidence/screenshots/T09_broken_file.png) · [T09_camera_lost.png](../reports/evidence/screenshots/T09_camera_lost.png) · [T09_ai_server_down.png](../reports/evidence/screenshots/T09_ai_server_down.png) |
| T10 Docker | 11 | [T10_restart.png](../reports/evidence/screenshots/T10_restart.png) · [T10_after_restart.txt](../reports/evidence/T10_after_restart.txt) |

## 작업 상태 표기

과제 문서 4장이 제시한 후보(`PENDING · STARTED · SUCCESS · FAILURE`)이자 Celery 표준 표기를 그대로 쓰고, 프로젝트 전체에서 통일했습니다. 개발 가이드 6단계의 권장 표기와는 다음과 같이 대응합니다.

| 개발 가이드 권장 | 이 프로젝트 `status` | 진행 단계 `stage` (화면 안내) | 화면 표시 |
|---|---|---|---|
| `PENDING` | `PENDING` | `QUEUED` | 대기 |
| `PROCESSING` | `STARTED` | `CAPTURING`(IP Webcam만) → `INFERRING` | 처리 중 (영상 수집 / AI 분석) |
| `COMPLETED` | `SUCCESS` | `DONE` | 완료 |
| `FAILED` | `FAILURE` | `DONE` | 실패 + 사유 |

## 캡처 규칙

- 브라우저는 100% 배율, 창 너비 약 1280px. 주소창이 보이게 찍어 **어느 URL인지** 드러나게 합니다.
- Windows `Win + Shift + S` → 창 캡처, 또는 Chrome 개발자 도구(F12) → `Ctrl + Shift + P` → *Capture full size screenshot*.
- 카메라 IP · 비밀번호 · `.env` 내용이 보이면 가리거나 다시 찍습니다 (과제 12장 제출 전 확인).
