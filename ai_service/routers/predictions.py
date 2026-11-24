"""
추론 API (routers/predictions.py)

    GET  /health          서버·모델 준비 상태
    POST /predict/video   Celery Worker가 보낸 영상 1건 → 사람별 예측 JSON (과제 문서의 API 계약)
    GET  /api/v1/health · POST /api/v1/predict/   위와 같은 함수 — 9~10일차 개발 가이드의 권장 경로 규칙

API 계약 (Celery Worker가 결과를 해석하는 기준)
    200 + status="OK"              사람별 예측 1건 이상 → AnalysisJob SUCCESS, PersonPrediction 저장
    200 + status="NO_PERSON"       영상은 정상, 사람 미검출 → SUCCESS (예측 0건, message 표시)
    200 + status="NO_VALID_TRACK"  사람은 보였지만 너무 짧게 보임 → SUCCESS (예측 0건, message 표시)
    4xx + detail.code              입력 문제 (재시도해도 같음) → FAILURE, detail.message 저장
    503 MODEL_NOT_READY            서버 문제 (잠시 후 재시도 가능) → Worker 재시도 대상
    500 INFERENCE_ERROR            예상하지 못한 오류 → FAILURE
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from services.action_inference import DecisionRule, predict_tracks
from services.pose_pipeline import VideoReadError

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
CHUNK_SIZE = 1024 * 1024  # 1 MB씩 디스크로 복사 (큰 영상을 메모리에 통째로 올리지 않음)


# ─────────────────────────────────────────────────────────────
# 응답 스키마 (/docs에 그대로 표시되는 API 계약)
# ─────────────────────────────────────────────────────────────
class PersonResult(BaseModel):
    track_id: int = Field(description="이 영상 안에서의 추적 번호 (영구 사람 ID 아님)")
    action_id: str = Field(examples=["A006"])
    action_label: str = Field(examples=["wave"])
    confidence: float = Field(ge=0, le=1)
    margin: float = Field(ge=0, le=1, description="1등 확률 - 2등 확률. 작을수록 두 동작 사이에서 헷갈린 것")
    runner_up_label: str = Field(examples=["drink_return"], description="2등 동작")
    is_uncertain: bool = Field(description="confidence < min_confidence 또는 margin < min_margin 이면 True → 화면에 '판단 불확실'")
    probabilities: dict[str, float]
    window_count: int
    window_mode: Literal["sliding", "whole_track", "hybrid"]
    seen_frames: int
    valid_frames: int
    valid_frame_ratio: float = Field(description="입력 품질 지표: 검출 Frame 중 Feature 계산 성공 비율")
    duration_sec: float


class SkippedTrackResult(BaseModel):
    track_id: int
    seen_frames: int
    valid_frames: int
    reason: str


class VideoMeta(BaseModel):
    filename: str
    fps: float
    fps_assumed: bool
    frame_count: int
    processed_frames: int
    duration_sec: float
    width: int
    height: int


class DecisionRuleResult(BaseModel):
    min_confidence: float
    min_margin: float


class PredictVideoResponse(BaseModel):
    status: Literal["OK", "NO_PERSON", "NO_VALID_TRACK"]
    message: str
    person_count: int
    predictions: list[PersonResult]
    skipped_tracks: list[SkippedTrackResult]
    video: VideoMeta
    uncertain_threshold: float = Field(description="하위 호환용 = decision_rule.min_confidence")
    decision_rule: DecisionRuleResult
    processing_time_sec: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_ready: bool
    error: str | None = None
    device: str | None = None
    window_strategy: str | None = None
    decision_rule: DecisionRuleResult | None = None
    model: dict | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


ERROR_RESPONSES = {
    400: {"model": ErrorDetail, "description": "UNSUPPORTED_FORMAT / EMPTY_FILE"},
    413: {"model": ErrorDetail, "description": "FILE_TOO_LARGE"},
    422: {"model": ErrorDetail, "description": "VIDEO_UNREADABLE / VIDEO_TOO_LONG"},
    503: {"model": ErrorDetail, "description": "MODEL_NOT_READY"},
}


# ─────────────────────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    state = request.app.state
    ready = getattr(state, "bundle", None) is not None and getattr(state, "pipeline", None) is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_ready=ready,
        error=getattr(state, "load_error", None),
        device=str(state.pipeline.device) if ready else None,
        window_strategy=state.pipeline.config.window_strategy if ready else None,
        decision_rule=DecisionRuleResult(**_decision_rule(state.pipeline).to_dict()) if ready else None,
        model=state.bundle.summary() if ready else None,
    )


# async가 아닌 일반 def → FastAPI가 Thread Pool에서 실행 (무거운 YOLO 추론이 서버 전체를 막지 않음)
@router.post(
    "/predict/video",
    response_model=PredictVideoResponse,
    responses=ERROR_RESPONSES,
    tags=["inference"],
)
def predict_video(request: Request, file: UploadFile = File(..., description="분석할 영상 (mp4 등)")):
    started = time.perf_counter()
    state = request.app.state
    bundle, pipeline = getattr(state, "bundle", None), getattr(state, "pipeline", None)
    if bundle is None or pipeline is None:
        raise _error(503, "MODEL_NOT_READY", f"추론 모델이 준비되지 않았습니다: {getattr(state, 'load_error', None)}")

    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise _error(400, "UNSUPPORTED_FORMAT", f"지원하지 않는 형식입니다: '{suffix or '없음'}' (허용: {sorted(ALLOWED_SUFFIXES)})")

    tmp_path = _save_upload(file, suffix)
    try:
        try:
            output = pipeline.run(tmp_path)
        except VideoReadError as e:
            raise _error(422, e.code, str(e)) from e

        rule = _decision_rule(pipeline)
        preds = predict_tracks(
            bundle,
            {t.track_id: t.windows for t in output.tracks},
            rule,
            modes_by_track={t.track_id: t.window_mode for t in output.tracks},
        )
        track_by_id = {t.track_id: t for t in output.tracks}

        predictions = [
            PersonResult(
                **p.to_dict(),
                window_mode=track_by_id[p.track_id].window_mode,
                seen_frames=track_by_id[p.track_id].seen_frames,
                valid_frames=track_by_id[p.track_id].valid_frames,
                valid_frame_ratio=track_by_id[p.track_id].valid_frame_ratio,
                duration_sec=track_by_id[p.track_id].duration_sec,
            )
            for p in preds
        ]

        if predictions:
            status_, message = "OK", f"{len(predictions)}명의 동작을 예측했습니다."
        elif output.skipped_tracks:
            status_, message = "NO_VALID_TRACK", "사람이 검출되었지만 판단에 필요한 만큼 오래 보이지 않았습니다."
        else:
            status_, message = "NO_PERSON", "영상에서 사람을 찾지 못했습니다."

        v = output.video
        return PredictVideoResponse(
            status=status_,
            message=message,
            person_count=len(predictions),
            predictions=predictions,
            skipped_tracks=[SkippedTrackResult(**asdict(s)) for s in output.skipped_tracks],
            video=VideoMeta(filename=filename, **asdict(v)),
            uncertain_threshold=rule.min_confidence,
            decision_rule=DecisionRuleResult(**rule.to_dict()),
            processing_time_sec=round(time.perf_counter() - started, 3),
        )
    except HTTPException:
        raise
    except Exception as e:  # 예상하지 못한 오류도 Worker가 해석할 수 있는 형태로 반환
        logger.exception("inference failed: %s", filename)
        raise _error(500, "INFERENCE_ERROR", f"추론 중 오류가 발생했습니다: {type(e).__name__}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


# 개발 가이드 권장 경로(/api/v1/...)도 같은 함수로 연결 — 경로만 다르고 계약은 완전히 같다
router.add_api_route("/api/v1/health", health, methods=["GET"], response_model=HealthResponse, tags=["v1"])
router.add_api_route(
    "/api/v1/predict/", predict_video, methods=["POST"], response_model=PredictVideoResponse,
    responses=ERROR_RESPONSES, tags=["v1"],
)


def _decision_rule(pipeline) -> DecisionRule:
    cfg = pipeline.config
    return DecisionRule(min_confidence=cfg.min_confidence, min_margin=cfg.min_margin)


def _save_upload(file: UploadFile, suffix: str) -> Path:
    """업로드를 임시 파일로 저장 (OpenCV는 파일 경로가 필요). 크기 제한을 넘으면 즉시 중단."""
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="predict_") as tmp:
        tmp_path = Path(tmp.name)
        try:
            while chunk := file.file.read(CHUNK_SIZE):
                written += len(chunk)
                if written > limit:
                    raise _error(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "FILE_TOO_LARGE", f"파일이 {MAX_UPLOAD_MB}MB를 넘습니다.")
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise _error(400, "EMPTY_FILE", "빈 파일입니다.")
    return tmp_path
