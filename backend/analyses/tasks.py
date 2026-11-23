"""
Celery 작업 (tasks.py)

run_analysis(job_id)
    1) Job을 STARTED로 표시
    2) 영상 확보: 업로드 파일 경로 사용 / IP Webcam이면 camera_capture로 N초 수집해 MEDIA에 저장
    3) FastAPI /predict/video 호출 (services.py)
    4) Django ORM으로 결과 저장: PersonPrediction 생성 + Job SUCCESS  (하나의 트랜잭션)
       실패하면 Job FAILURE + 사용자용 오류 메시지 — 어떤 경우에도 STARTED로 남겨 두지 않는다.
"""

from __future__ import annotations

import logging

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .camera_capture import CameraCaptureError, capture_clip
from .constants import ACTION_ID
from .models import AnalysisJob, PersonPrediction
from .services import InferenceError, InferenceUnavailable, predict_video

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10


@shared_task(bind=True, max_retries=MAX_RETRIES)
def run_analysis(self, job_id: int) -> str:
    try:
        job = AnalysisJob.objects.get(pk=job_id)
    except AnalysisJob.DoesNotExist:
        logger.warning("job %s not found (삭제됨)", job_id)
        return "MISSING"

    if job.is_finished:                         # 중복 실행 방지 (acks_late로 재전달된 경우 등)
        return job.status

    _update(job, status=AnalysisJob.Status.STARTED, started_at=job.started_at or timezone.now(), error_message="")

    try:
        video_path = _acquire_video(job)
        _update(job, stage=AnalysisJob.Stage.INFERRING)
        result = predict_video(video_path)

    except InferenceUnavailable as e:           # AI 서버 일시 장애 → 재시도
        if self.request.retries < MAX_RETRIES and not getattr(self.request, "is_eager", False):
            _update(job, stage=AnalysisJob.Stage.QUEUED, error_message=f"재시도 대기 중: {e}")
            raise self.retry(exc=e, countdown=RETRY_DELAY_SECONDS)
        return _fail(job, str(e))
    except (CameraCaptureError, InferenceError) as e:
        return _fail(job, str(e))
    except Exception as e:  # noqa: BLE001 - 예상 못한 오류도 반드시 FAILURE로 기록
        logger.exception("job %s failed unexpectedly", job_id)
        return _fail(job, f"처리 중 알 수 없는 오류가 발생했습니다: {type(e).__name__}")

    with transaction.atomic():
        job.predictions.all().delete()          # 재실행 시 중복 방지
        PersonPrediction.objects.bulk_create([_to_prediction(job, p) for p in result.predictions])
        _update(
            job,
            status=AnalysisJob.Status.SUCCESS,
            stage=AnalysisJob.Stage.DONE,
            result_status=result.status,
            result_message=result.message,
            video_duration_sec=result.video.get("duration_sec"),
            video_fps=result.video.get("fps"),
            inference_time_sec=result.processing_time_sec,
            error_message="",
            finished_at=timezone.now(),
        )
    logger.info("job %s SUCCESS: %s (%d명)", job_id, result.status, len(result.predictions))
    return AnalysisJob.Status.SUCCESS


def _acquire_video(job: AnalysisJob):
    if job.input_type == AnalysisJob.InputType.UPLOAD:
        if not job.video_file:
            raise InferenceError("업로드된 영상 파일이 없습니다.")
        return job.video_file.path

    # IP Webcam: 수집한 구간을 MEDIA에 저장하고 Job에 연결 (나중에 다시 확인 가능)
    _update(job, stage=AnalysisJob.Stage.CAPTURING)
    rel_path = f"captures/{timezone.localdate():%Y/%m/%d}/job_{job.pk}.mp4"
    out_path = settings.MEDIA_ROOT / rel_path
    capture_clip(job.camera_url, job.capture_seconds or settings.CAMERA_CAPTURE_SECONDS_DEFAULT, out_path)
    job.video_file.name = rel_path
    job.save(update_fields=["video_file"])
    return out_path


def _to_prediction(job: AnalysisJob, p: dict) -> PersonPrediction:
    return PersonPrediction(
        job=job,
        track_id=p["track_id"],
        action_label=p["action_label"],
        action_id=p.get("action_id") or ACTION_ID.get(p["action_label"], ""),
        confidence=p["confidence"],
        is_uncertain=p.get("is_uncertain", False),
        window_count=p.get("window_count", 0),
        window_mode=p.get("window_mode", ""),
        valid_frame_ratio=p.get("valid_frame_ratio"),
        duration_sec=p.get("duration_sec"),
        probabilities=p.get("probabilities", {}),
    )


def _fail(job: AnalysisJob, message: str) -> str:
    _update(
        job,
        status=AnalysisJob.Status.FAILURE,
        stage=AnalysisJob.Stage.DONE,
        error_message=message[:1000],
        finished_at=timezone.now(),
    )
    logger.warning("job %s FAILURE: %s", job.pk, message)
    return AnalysisJob.Status.FAILURE


def _update(job: AnalysisJob, **fields) -> None:
    """지정한 필드만 저장 (다른 프로세스가 바꾼 필드를 덮어쓰지 않기 위해)."""
    for k, v in fields.items():
        setattr(job, k, v)
    job.save(update_fields=list(fields))
