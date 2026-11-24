"""
FastAPI 추론 서버 호출 (services.py)

FastAPI의 API 계약(ai_service/routers/predictions.py)을 Worker 관점의 3가지 결과로 바꾼다.
    - 성공(InferenceResult)                         → Job SUCCESS
    - InferenceInputError  (4xx: 입력 문제)          → Job FAILURE (재시도해도 같으므로 재시도 안 함)
    - InferenceUnavailable (503·연결 실패·타임아웃)   → Celery 재시도 → 그래도 실패면 FAILURE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """사용자에게 보여 줄 메시지를 담는 기본 예외."""


class InferenceInputError(InferenceError):
    """영상 자체의 문제 (손상, 형식, 길이) — 재시도 무의미."""


class InferenceUnavailable(InferenceError):
    """AI 서버가 일시적으로 응답할 수 없음 — 재시도 대상."""


@dataclass(frozen=True)
class InferenceResult:
    status: str                       # OK / NO_PERSON / NO_VALID_TRACK
    message: str
    predictions: list[dict[str, Any]]
    video: dict[str, Any]
    processing_time_sec: float
    raw: dict[str, Any]


def predict_video(video_path: Path | str) -> InferenceResult:
    """영상 파일을 multipart로 FastAPI /predict/video에 보내고 결과를 해석한다."""
    url = f"{settings.FASTAPI_URL}/predict/video"
    video_path = Path(video_path)
    try:
        with video_path.open("rb") as f:
            resp = requests.post(
                url,
                files={"file": (video_path.name, f, "application/octet-stream")},
                timeout=(5, settings.FASTAPI_TIMEOUT_SECONDS),     # (연결, 응답) 타임아웃
            )
    except requests.Timeout as e:
        raise InferenceUnavailable(f"AI 서버 응답 시간 초과 ({settings.FASTAPI_TIMEOUT_SECONDS:.0f}초)") from e
    except requests.ConnectionError as e:
        raise InferenceUnavailable(f"AI 서버에 연결할 수 없습니다 ({settings.FASTAPI_URL})") from e

    body = _json_or_none(resp)
    if resp.status_code == 200 and body is not None:
        return InferenceResult(
            status=body["status"],
            message=body.get("message", ""),
            predictions=body.get("predictions", []),
            video=body.get("video", {}),
            processing_time_sec=float(body.get("processing_time_sec", 0.0)),
            raw=body,
        )

    detail = (body or {}).get("detail") if isinstance(body, dict) else None
    message = detail.get("message") if isinstance(detail, dict) else str(detail or resp.text[:200])
    code = detail.get("code") if isinstance(detail, dict) else f"HTTP_{resp.status_code}"
    logger.warning("FastAPI %s %s: %s", resp.status_code, code, message)

    if resp.status_code == 503 or resp.status_code >= 502:
        raise InferenceUnavailable(f"AI 서버 준비 안 됨: {message}")
    if 400 <= resp.status_code < 500:
        raise InferenceInputError(f"[{code}] {message}")
    raise InferenceError(f"AI 서버 오류 [{code}] {message}")


def decision_rule() -> dict[str, float] | None:
    """화면 안내 문구용: FastAPI가 지금 적용 중인 판단 불확실 기준 (60초 캐시, 연결 실패 시 None).

    기준값은 ai_service/project.json 한 곳에서만 관리하고, 화면은 그 값을 받아 보여 준다 (값 중복 방지).
    """
    from django.core.cache import cache

    rule = cache.get("ai_decision_rule")
    if rule is None:
        rule = (check_health() or {}).get("decision_rule")
        if rule:
            cache.set("ai_decision_rule", rule, 60)
    return rule


def check_health() -> dict[str, Any] | None:
    """대시보드에 AI 서버 상태를 표시하기 위한 가벼운 확인 (실패하면 None)."""
    try:
        resp = requests.get(f"{settings.FASTAPI_URL}/health", timeout=2)
        return resp.json()
    except Exception:  # noqa: BLE001 - 상태 표시용이므로 어떤 오류든 '연결 안 됨'으로 처리
        return None


def _json_or_none(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None
