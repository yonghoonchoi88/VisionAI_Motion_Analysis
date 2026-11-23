"""
입력 검사와 JSON 응답 형태

- AnalysisJobCreateSerializer : 분석 요청(업로드 영상 또는 허용 카메라) 검사
- AnalysisJobSerializer       : 목록·상세 응답 (상태·AI 결과는 읽기 전용)
- AnalysisJobUpdateSerializer : PATCH로 제목·메모만 수정 허용
- AnalysisJobStatusSerializer : 브라우저 폴링용 가벼운 상태 응답
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from django.conf import settings
from rest_framework import serializers

from .constants import ACTION_KO
from .models import AnalysisJob, PersonPrediction


class PersonPredictionSerializer(serializers.ModelSerializer):
    action_name_ko = serializers.SerializerMethodField()
    runner_up_name_ko = serializers.SerializerMethodField()

    class Meta:
        model = PersonPrediction
        fields = [
            "track_id", "action_id", "action_label", "action_name_ko", "confidence", "margin",
            "runner_up_label", "runner_up_name_ko", "is_uncertain",
            "window_count", "window_mode", "valid_frame_ratio", "duration_sec", "probabilities", "created_at",
        ]
        read_only_fields = fields

    def get_action_name_ko(self, obj) -> str:
        return ACTION_KO.get(obj.action_label, obj.action_label)

    def get_runner_up_name_ko(self, obj) -> str:
        return ACTION_KO.get(obj.runner_up_label, obj.runner_up_label)


def validate_camera_url(value: str) -> str:
    """허용 목록(ALLOWED_CAMERA_HOSTS)에 있는 host:port만 받는다. 임의 외부 URL 호출(SSRF) 방지."""
    parts = urlsplit(value.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise serializers.ValidationError("http:// 또는 https:// 로 시작하는 카메라 주소를 입력하세요.")
    host = f"{parts.hostname}:{parts.port}" if parts.port else parts.hostname
    allowed = settings.ALLOWED_CAMERA_HOSTS
    if not allowed:
        raise serializers.ValidationError("서버에 허용된 카메라가 없습니다. 관리자에게 ALLOWED_CAMERA_HOSTS 설정을 요청하세요.")
    if host not in allowed:
        raise serializers.ValidationError(f"허용되지 않은 카메라입니다: {host} (허용: {', '.join(allowed)})")
    if parts.username or parts.password:
        raise serializers.ValidationError("주소에 계정 정보를 넣지 마세요.")
    return value.strip()


class AnalysisJobCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisJob
        fields = ["input_type", "title", "memo", "expected_action", "video_file", "camera_url", "capture_seconds"]
        extra_kwargs = {"video_file": {"required": False}, "camera_url": {"required": False}}

    def validate_video_file(self, f):
        if f is None:
            return f
        ext = Path(f.name).suffix.lower()
        if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise serializers.ValidationError(f"지원하지 않는 형식입니다: {ext or '없음'} (허용: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)})")
        if f.size == 0:
            raise serializers.ValidationError("빈 파일입니다.")
        if f.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise serializers.ValidationError(f"파일이 {settings.MAX_UPLOAD_MB}MB를 넘습니다.")
        return f

    def validate(self, attrs):
        input_type = attrs.get("input_type")
        if input_type == AnalysisJob.InputType.UPLOAD:
            if not attrs.get("video_file"):
                raise serializers.ValidationError({"video_file": "분석할 영상 파일을 선택하세요."})
            attrs["camera_url"] = ""
            attrs["capture_seconds"] = None
        elif input_type == AnalysisJob.InputType.CAMERA:
            if not attrs.get("camera_url"):
                raise serializers.ValidationError({"camera_url": "카메라 주소를 입력하세요."})
            attrs["camera_url"] = validate_camera_url(attrs["camera_url"])
            seconds = attrs.get("capture_seconds") or settings.CAMERA_CAPTURE_SECONDS_DEFAULT
            lo, hi = settings.CAMERA_CAPTURE_SECONDS_MIN, settings.CAMERA_CAPTURE_SECONDS_MAX
            if not lo <= seconds <= hi:
                raise serializers.ValidationError({"capture_seconds": f"수집 시간은 {lo}~{hi}초 사이여야 합니다."})
            attrs["capture_seconds"] = seconds
            attrs.pop("video_file", None)
        return attrs


class AnalysisJobSerializer(serializers.ModelSerializer):
    predictions = PersonPredictionSerializer(many=True, read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)
    input_type_display = serializers.CharField(source="get_input_type_display", read_only=True)
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    stage_display = serializers.CharField(source="get_stage_display", read_only=True)
    is_finished = serializers.BooleanField(read_only=True)
    total_time_sec = serializers.FloatField(read_only=True)
    display_title = serializers.CharField(read_only=True)
    video_url = serializers.SerializerMethodField()
    status_url = serializers.HyperlinkedIdentityField(view_name="analysisjob-status")

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "username", "display_title", "title", "memo",
            "input_type", "input_type_display", "video_url", "camera_url", "capture_seconds", "expected_action",
            "status", "status_display", "stage", "stage_display", "is_finished", "celery_task_id", "error_message",
            "result_status", "result_message", "video_duration_sec", "video_fps", "inference_time_sec",
            "processing_time_sec", "total_time_sec", "created_at", "started_at", "finished_at", "status_url", "predictions",
        ]
        read_only_fields = fields

    def get_video_url(self, obj) -> str | None:
        return obj.video_file.url if obj.video_file else None


class AnalysisJobUpdateSerializer(serializers.ModelSerializer):
    """사용자는 제목·메모만 바꿀 수 있다. 상태·AI 결과를 보내도 무시된다."""

    class Meta:
        model = AnalysisJob
        fields = ["title", "memo"]


class AnalysisJobStatusSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    stage_display = serializers.CharField(source="get_stage_display", read_only=True)
    is_finished = serializers.BooleanField(read_only=True)
    total_time_sec = serializers.FloatField(read_only=True)
    predictions = PersonPredictionSerializer(many=True, read_only=True)

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "status", "status_display", "stage", "stage_display", "is_finished", "error_message",
            "result_status", "result_message", "video_duration_sec", "inference_time_sec", "total_time_sec",
            "predictions",
        ]
        read_only_fields = fields
