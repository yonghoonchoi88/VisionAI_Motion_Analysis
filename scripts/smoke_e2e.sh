#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 전체 흐름 수동 검증을 curl로 한 번에 (개발 가이드 5·7단계 + 과제 T02·T05·T06)
#
#   FastAPI /api/v1/health → 로그인(세션·CSRF) → POST /api/v1/analyses/ (영상 업로드)
#   → 202 + 작업 ID 즉시 반환 확인 → status 폴링 → SUCCESS/FAILURE → PostgreSQL 직접 조회
#
# 사용법 (저장소 최상위 폴더, docker compose up 이후)
#   bash scripts/smoke_e2e.sh <영상 파일>
#   예) bash scripts/smoke_e2e.sh ~/ai_vision/subject10_pose_action/data/day05/videos/S03/S03_A006_T01.mp4
#
# 환경변수 (선택)
#   SMOKE_USER / SMOKE_PASSWORD  검증용 계정 (없으면 자동 생성, 기본 smoke_tester)
#   NO_DOCKER=1                  docker 없이 실행 중인 서버를 검증 (계정 자동 생성·DB 조회 생략)
#   TIMEOUT=300                  결과를 기다릴 최대 초
# 결과 로그: reports/evidence/smoke/smoke_<시각>.log  → README·발표 증거로 사용
# ─────────────────────────────────────────────────────────────
set -euo pipefail

VIDEO="${1:-}"
if [[ -z "$VIDEO" || ! -f "$VIDEO" ]]; then
  echo "사용법: bash scripts/smoke_e2e.sh <영상 파일 경로>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

BASE="${BASE_URL:-http://localhost:${DJANGO_HOST_PORT:-8000}}"
AI="${AI_URL:-http://localhost:${FASTAPI_HOST_PORT:-8001}}"
USER_NAME="${SMOKE_USER:-smoke_tester}"
USER_PASS="${SMOKE_PASSWORD:-Smoke!$(date +%Y)check}"
TIMEOUT="${TIMEOUT:-300}"
JAR="$(mktemp)"
trap 'rm -f "$JAR"' EXIT

LOG_DIR="reports/evidence/smoke"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {}, {'d': d}))" "$1"; }
csrf() { awk '$6=="csrftoken"{print $7}' "$JAR" | tail -1; }
pass() { echo "  ✅ PASS  $*"; }
fail() { echo "  ❌ FAIL  $*"; exit 1; }

echo "== Vision AI 전체 흐름 검증 $(date '+%Y-%m-%d %H:%M:%S') =="
echo "   Django=$BASE  FastAPI=$AI  영상=$(basename "$VIDEO")"

echo "[1/6] FastAPI 단독 상태 (GET /api/v1/health)"
HEALTH="$(curl -fsS "$AI/api/v1/health")" || fail "FastAPI에 연결할 수 없습니다 → docker compose ps / logs fastapi 확인"
echo "  $HEALTH"
[[ "$(echo "$HEALTH" | json 'd["model_ready"]')" == "True" ]] && pass "모델 준비 완료" || fail "model_ready=false"

echo "[2/6] 검증용 계정 준비"
if [[ "${NO_DOCKER:-0}" != "1" ]]; then
  docker compose exec -T django python manage.py shell -c "
from django.contrib.auth import get_user_model
U = get_user_model(); u, created = U.objects.get_or_create(username='$USER_NAME')
u.set_password('$USER_PASS'); u.save(); print('  계정', 'created' if created else 'exists', u.username)"
fi

echo "[3/6] 로그인 (세션 + CSRF)"
curl -fsS -c "$JAR" -b "$JAR" "$BASE/accounts/login/" -o /dev/null
CODE="$(curl -sS -c "$JAR" -b "$JAR" -o /dev/null -w '%{http_code}' -X POST "$BASE/accounts/login/" \
  -H "Referer: $BASE/accounts/login/" \
  --data-urlencode "csrfmiddlewaretoken=$(csrf)" --data-urlencode "username=$USER_NAME" --data-urlencode "password=$USER_PASS")"
[[ "$CODE" == "302" ]] && pass "로그인 → 302 리다이렉트" || fail "로그인 실패 (HTTP $CODE) — SMOKE_USER/SMOKE_PASSWORD 확인"

echo "[4/6] 비로그인 차단 확인 (GET /api/v1/analyses/ without cookie)"
ANON="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/v1/analyses/")"
[[ "$ANON" == "403" || "$ANON" == "401" ]] && pass "비로그인 요청 → $ANON" || fail "비로그인 요청이 $ANON"

echo "[5/6] 분석 요청 (POST /api/v1/analyses/) — 작업 ID가 즉시 와야 함"
START=$(date +%s.%N)
RESP="$(curl -sS -c "$JAR" -b "$JAR" -w '\n%{http_code}' -X POST "$BASE/api/v1/analyses/" \
  -H "X-CSRFToken: $(csrf)" -H "Referer: $BASE/analyses/" \
  -F input_type=UPLOAD -F "title=smoke $(date +%H:%M)" -F "video_file=@$VIDEO")"
CODE="$(echo "$RESP" | tail -1)"; BODY="$(echo "$RESP" | sed '$d')"
ELAPSED=$(python3 -c "print(round($(date +%s.%N) - $START, 3))")
[[ "$CODE" == "202" ]] || { echo "$BODY"; fail "HTTP $CODE (202여야 함)"; }
JOB_ID="$(echo "$BODY" | json 'd["id"]')"
pass "202 Accepted · 작업 #$JOB_ID · 응답 ${ELAPSED}초 · 상태 $(echo "$BODY" | json 'd["status"]') · task $(echo "$BODY" | json 'd["celery_task_id"]')"

echo "[6/6] 상태 폴링 (GET /api/v1/analyses/$JOB_ID/status/)"
DEADLINE=$(( $(date +%s) + TIMEOUT )); LAST=""
while :; do
  S="$(curl -fsS -b "$JAR" "$BASE/api/v1/analyses/$JOB_ID/status/")"
  NOW="$(echo "$S" | json 'd["status"] + " / " + d["stage"]')"
  [[ "$NOW" != "$LAST" ]] && echo "  $(date +%H:%M:%S)  $NOW" && LAST="$NOW"
  [[ "$(echo "$S" | json 'd["is_finished"]')" == "True" ]] && break
  (( $(date +%s) > DEADLINE )) && fail "${TIMEOUT}초 안에 끝나지 않음 → docker compose logs celery fastapi"
  sleep 1
done
echo "$S" | python3 -c "
import json,sys; d=json.load(sys.stdin)
print(f\"  결과: {d['status']} · {d.get('result_message') or d.get('error_message')} · AI {d.get('inference_time_sec')}초 · 전체 {d.get('total_time_sec')}초\")
for p in d['predictions']:
    flag = '판단 불확실' if p['is_uncertain'] else '확정'
    m = p.get('margin'); m = f'{m:.3f}' if m is not None else '-'
    print(f\"   Track {p['track_id']}: {p['action_id']} {p['action_label']} conf={p['confidence']:.3f} margin={m} → {flag}\")"
[[ "$(echo "$S" | json 'd["status"]')" == "SUCCESS" ]] && pass "Worker가 결과를 저장함" || echo "  ⚠️  FAILURE — 사유가 저장되었는지 위 메시지 확인 (T09)"

if [[ "${NO_DOCKER:-0}" != "1" ]]; then
  echo "[DB] PostgreSQL 직접 조회 — 화면·API와 같은 값인지 (T06)"
  docker compose exec -T postgres psql -U "${POSTGRES_USER:-vision}" -d "${POSTGRES_DB:-vision_action}" -c \
    "SELECT j.id, u.username, j.input_type, j.status, j.stage, round(j.inference_time_sec::numeric,2) AS ai_sec,
            p.track_id, p.action_label, round(p.confidence::numeric,3) AS conf, round(p.margin::numeric,3) AS margin, p.is_uncertain
       FROM analyses_analysisjob j JOIN auth_user u ON u.id = j.user_id
       LEFT JOIN analyses_personprediction p ON p.job_id = j.id
      WHERE j.id = $JOB_ID ORDER BY p.track_id;"
fi
echo "== 완료: 로그 $LOG =="
