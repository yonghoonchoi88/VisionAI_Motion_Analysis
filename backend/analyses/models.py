"""
업무 테이블 2개 (ERD)

    auth_user (Django 기본) 1 ──< AnalysisJob 1 ──< PersonPrediction

- AnalysisJob      : 분석 요청 1건과 진행 상태. 영상 파일 자체는 MEDIA에 두고 DB에는 경로만 저장.
- PersonPrediction : 그 작업에서 추적된 사람(track_id)별 예측 1건. track_id는 '작업 안의 번호'일 뿐 영구 ID가 아니다.
"""

from __future__ import annotations

from django.conf import settings
from django.db import models

from .constants import ACTION_CHOICES, ACTION_ID


class AnalysisJob(models.Model):
    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        CAMERA = "CAMERA", "IP Webcam"

    class Status(models.TextChoices):           # Celery 표준 상태명과 동일하게 통일
        PENDING = "PENDING", "대기"
        STARTED = "STARTED", "처리 중"
        SUCCESS = "SUCCESS", "완료"
        FAILURE = "FAILURE", "실패"

    class Stage(models.TextChoices):            # 화면의 진행 단계 표시용 (대기 → 영상 수집 → AI 분석 → 완료)
        QUEUED = "QUEUED", "대기"
        CAPTURING = "CAPTURING", "영상 수집"
        INFERRING = "INFERRING", "AI 분석"
        DONE = "DONE", "완료"

    # ── 요청 정보 ─────────────────────────────────────────
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, verbose_name="요청 사용자", on_delete=models.CASCADE, related_name="analysis_jobs"
    )
    title = models.CharField("제목", max_length=100, blank=True)
    memo = models.TextField("메모", blank=True)
    input_type = models.CharField("입력 방식", max_length=10, choices=InputType.choices)
    video_file = models.FileField("영상 파일", upload_to="videos/%Y/%m/%d/", blank=True)
    camera_url = models.CharField("카메라 주소", max_length=255, blank=True)
    capture_seconds = models.PositiveSmallIntegerField("수집 시간(초)", null=True, blank=True)
    expected_action = models.CharField(
        "수행하려는 동작(사용자 선택)", max_length=30, choices=ACTION_CHOICES, blank=True,
        help_text="사용자가 직접 고른 값일 뿐 검증된 정답 라벨이 아님",
    )

    # ── 처리 상태 (Worker만 변경, 사용자 수정 불가) ───────────
    status = models.CharField("상태", max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True)
    stage = models.CharField("진행 단계", max_length=10, choices=Stage.choices, default=Stage.QUEUED)
    celery_task_id = models.CharField("Celery task ID", max_length=64, blank=True)
    error_message = models.TextField("오류 메시지", blank=True)

    # ── AI 응답 요약 ──────────────────────────────────────
    result_status = models.CharField("추론 결과 상태", max_length=20, blank=True)   # OK / NO_PERSON / NO_VALID_TRACK
    result_message = models.CharField("추론 결과 메시지", max_length=255, blank=True)
    video_duration_sec = models.FloatField("영상 길이(초)", null=True, blank=True)
    video_fps = models.FloatField("영상 FPS", null=True, blank=True)
    inference_time_sec = models.FloatField("AI 추론 시간(초)", null=True, blank=True)
    processing_time_sec = models.FloatField(
        "Worker 처리 시간(초)", null=True, blank=True,
        help_text="영상 확보·추론·저장에 걸린 시간. 벽시계 조정의 영향을 받지 않는 단조 시계로 측정",
    )

    # ── 시간 ──────────────────────────────────────────────
    created_at = models.DateTimeField("요청 시각", auto_now_add=True, db_index=True)
    started_at = models.DateTimeField("시작 시각", null=True, blank=True)
    finished_at = models.DateTimeField("완료 시각", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "-created_at"])]
        verbose_name = "분석 작업"
        verbose_name_plural = "분석 작업"

    def __str__(self) -> str:
        return f"#{self.pk} {self.get_input_type_display()} · {self.get_status_display()}"

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.SUCCESS, self.Status.FAILURE)

    @property
    def total_time_sec(self) -> float | None:
        """요청부터 완료까지 걸린 전체 시간 = 대기(생성→시작) + Worker 처리 시간.

        경과 시간은 벽시계 두 시각의 차로 재면 안 된다. WSL2·Docker는 호스트 시계와 동기화하며
        시각을 순간적으로 조정할 수 있어, finished_at - created_at이 AI 추론 시간보다 짧게 나온 사례가 있었다.
        그래서 Worker 구간은 단조 시계로 잰 processing_time_sec을 쓴다.
        """
        if not (self.finished_at and self.created_at):
            return None
        if self.processing_time_sec is not None and self.started_at:
            queued = max(0.0, (self.started_at - self.created_at).total_seconds())
            return round(queued + self.processing_time_sec, 2)
        # 이전 기록(processing_time_sec 없음): 벽시계 차를 쓰되 AI 추론 시간보다 짧을 수는 없으므로 하한 적용
        wall = (self.finished_at - self.created_at).total_seconds()
        return round(max(wall, self.inference_time_sec or 0.0), 2)

    @property
    def expected_action_id(self) -> str:
        """수행 동작(사용자 선택)의 과제 ID — 예측과 같은 형식(A006 wave)으로 비교하기 위함."""
        return ACTION_ID.get(self.expected_action, "")

    @property
    def display_title(self) -> str:
        return self.title or (self.video_file.name.rsplit("/", 1)[-1] if self.video_file else self.get_input_type_display())


class PersonPrediction(models.Model):
    job = models.ForeignKey(AnalysisJob, verbose_name="분석 작업", on_delete=models.CASCADE, related_name="predictions")
    track_id = models.PositiveIntegerField("Track ID")
    action_label = models.CharField("예측 동작", max_length=30, db_index=True)
    action_id = models.CharField("Action ID", max_length=8)
    confidence = models.FloatField("Confidence")
    margin = models.FloatField(
        "1·2등 확률 차", null=True, blank=True,
        help_text="1등 확률 - 2등 확률. 작으면 두 동작 사이에서 헷갈린 것 (판단 불확실 기준 ②)",
    )
    runner_up_label = models.CharField("2등 동작", max_length=30, blank=True)
    is_uncertain = models.BooleanField("판단 불확실", default=False)
    window_count = models.PositiveIntegerField("처리한 Window 수", default=0)
    window_mode = models.CharField("Window 방식", max_length=20, blank=True)
    valid_frame_ratio = models.FloatField("유효 Frame 비율", null=True, blank=True)
    duration_sec = models.FloatField("등장 구간(초)", null=True, blank=True)
    probabilities = models.JSONField("Class별 확률", default=dict, blank=True)
    created_at = models.DateTimeField("예측 시각", auto_now_add=True)

    class Meta:
        ordering = ["job", "track_id"]
        constraints = [models.UniqueConstraint(fields=["job", "track_id"], name="uniq_job_track")]
        verbose_name = "사람별 예측"
        verbose_name_plural = "사람별 예측"

    def __str__(self) -> str:
        return f"job {self.job_id} · track {self.track_id} · {self.action_label} ({self.confidence:.2f})"

    @property
    def sorted_probabilities(self) -> list[tuple[str, float]]:
        """화면 막대그래프용: 확률이 큰 동작부터."""
        return sorted((self.probabilities or {}).items(), key=lambda kv: kv[1], reverse=True)
