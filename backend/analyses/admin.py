"""Django Admin: 분석 작업·사람별 예측 목록/검색/필터."""

from django.contrib import admin
from django.db.models import Count

from .models import AnalysisJob, PersonPrediction


class PersonPredictionInline(admin.TabularInline):
    model = PersonPrediction
    extra = 0
    can_delete = False
    fields = ("track_id", "action_id", "action_label", "confidence", "is_uncertain", "window_mode", "valid_frame_ratio")
    readonly_fields = fields


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title_or_file", "input_type", "status", "person_count", "result_status", "created_at", "finished_at")
    list_display_links = ("id", "title_or_file")
    list_filter = ("status", "input_type", "result_status", "created_at")
    search_fields = ("user__username", "title", "memo", "celery_task_id")
    date_hierarchy = "created_at"
    list_select_related = ("user",)
    readonly_fields = (
        "celery_task_id", "status", "stage", "error_message", "result_status", "result_message",
        "video_duration_sec", "video_fps", "inference_time_sec", "processing_time_sec", "created_at", "started_at", "finished_at",
    )
    inlines = [PersonPredictionInline]

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_person_count=Count("predictions"))

    @admin.display(description="제목·파일")
    def title_or_file(self, obj):
        return obj.display_title

    @admin.display(description="예측 인원", ordering="_person_count")
    def person_count(self, obj):
        return obj._person_count


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "track_id", "action_id", "action_label", "confidence", "is_uncertain", "window_mode", "created_at")
    list_filter = ("action_label", "is_uncertain", "window_mode")
    search_fields = ("job__user__username", "action_label", "job__id")
    list_select_related = ("job", "job__user")
