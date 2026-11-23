"""
HTML 화면 (page_views.py)

    /analyses/        분석 화면: 입력 방식 선택 → AI 분석 시작 → 상태·사람별 결과 → 최근 이력
    /analyses/<id>/   분석 상세 (본인 또는 관리자)
    /dashboard/       관리자 전용 대시보드 — 차트 데이터는 Django ORM 집계로 만든다 (FastAPI는 DB에 접근하지 않음)
"""

from __future__ import annotations

from functools import wraps

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.db.models import Avg, Count, Q
from django.shortcuts import get_object_or_404, render

from .constants import ACTIONS
from .models import AnalysisJob, PersonPrediction
from .services import check_health


def staff_required(view):
    """로그인 안 했으면 로그인 화면으로, 로그인했지만 관리자가 아니면 403."""

    @login_required
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_staff:
            raise PermissionDenied("관리자만 접근할 수 있습니다.")
        return view(request, *args, **kwargs)

    return wrapper


def input_type_comparison(jobs) -> list[dict]:
    """입력 방식별 비교 요약 (과제 9-1: 입력 방식·수행 동작·예측·Confidence·처리 시간).

    성공한 작업만 대상으로 하며, 사람이 여러 명이면 사람별 예측을 각각 센다.
    '선택 동작 일치율'은 사용자가 수행 동작을 고른 예측만 분모로 쓴다.
    """
    rows = []
    for input_type in AnalysisJob.InputType:
        done = [j for j in jobs if j.input_type == input_type.value and j.status == AnalysisJob.Status.SUCCESS]
        preds = [(j, p) for j in done for p in j.predictions.all()]
        confs = [p.confidence for _, p in preds]
        labeled = [(j, p) for j, p in preds if j.expected_action]
        matched = sum(1 for j, p in labeled if p.action_label == j.expected_action)
        totals = [j.total_time_sec for j in done if j.total_time_sec is not None]
        infers = [j.inference_time_sec for j in done if j.inference_time_sec is not None]
        rows.append({
            "key": input_type.value,
            "label": input_type.label,
            "jobs": len(done),
            "predictions": len(preds),
            "avg_confidence": sum(confs) / len(confs) if confs else None,
            "uncertain": sum(1 for _, p in preds if p.is_uncertain),
            "match": matched,
            "labeled": len(labeled),
            "match_rate": matched / len(labeled) if labeled else None,
            "avg_total": sum(totals) / len(totals) if totals else None,
            "avg_infer": sum(infers) / len(infers) if infers else None,
        })
    return rows


@login_required
def index(request):
    jobs = list(
        AnalysisJob.objects.filter(user=request.user)
        .prefetch_related("predictions")
        .order_by("-created_at")
    )
    context = {
        "recent_jobs": jobs[:20],
        "comparison": input_type_comparison(jobs),
        "actions": ACTIONS,
        "allowed_cameras": settings.ALLOWED_CAMERA_HOSTS,
        "capture_default": settings.CAMERA_CAPTURE_SECONDS_DEFAULT,
        "capture_min": settings.CAMERA_CAPTURE_SECONDS_MIN,
        "capture_max": settings.CAMERA_CAPTURE_SECONDS_MAX,
        "max_upload_mb": settings.MAX_UPLOAD_MB,
        "allowed_ext": ",".join(settings.ALLOWED_VIDEO_EXTENSIONS),
    }
    return render(request, "analyses/index.html", context)


@login_required
def detail(request, pk: int):
    qs = AnalysisJob.objects.select_related("user").prefetch_related("predictions")
    job = get_object_or_404(qs if request.user.is_staff else qs.filter(user=request.user), pk=pk)
    return render(request, "analyses/detail.html", {"job": job})


@staff_required
def dashboard(request):
    # ① 6개 Action별 예측 건수 (예측이 0건인 동작도 막대에 표시)
    # .order_by()로 Meta 기본 정렬을 지워야 GROUP BY가 정렬 필드까지 묶지 않는다
    preds = PersonPrediction.objects.order_by()
    counts = dict(preds.values("action_label").annotate(n=Count("id")).values_list("action_label", "n"))
    uncertain = dict(
        preds.filter(is_uncertain=True).values("action_label").annotate(n=Count("id")).values_list("action_label", "n")
    )
    action_chart = {
        "labels": [f"{aid} {ko}" for _, aid, ko in ACTIONS],
        "confident": [counts.get(label, 0) - uncertain.get(label, 0) for label, _, _ in ACTIONS],
        "uncertain": [uncertain.get(label, 0) for label, _, _ in ACTIONS],
    }

    # ② 작업 상태별 건수
    jobs = AnalysisJob.objects.order_by()
    status_counts = dict(jobs.values("status").annotate(n=Count("id")).values_list("status", "n"))
    status_chart = {
        "labels": [s.label for s in AnalysisJob.Status],
        "values": [status_counts.get(s.value, 0) for s in AnalysisJob.Status],
        "keys": [s.value for s in AnalysisJob.Status],
    }

    # ③ (선택) 입력 방식별 비율 + 평균 AI 추론 시간
    by_input = {
        row["input_type"]: row
        for row in jobs.values("input_type").annotate(
            n=Count("id"),
            ok=Count("id", filter=Q(status=AnalysisJob.Status.SUCCESS)),
            avg_infer=Avg("inference_time_sec", filter=Q(status=AnalysisJob.Status.SUCCESS)),
        )
    }
    input_chart = {
        "labels": [t.label for t in AnalysisJob.InputType],
        "values": [by_input.get(t.value, {}).get("n", 0) for t in AnalysisJob.InputType],
        "avg_infer": [round(by_input.get(t.value, {}).get("avg_infer") or 0, 2) for t in AnalysisJob.InputType],
    }

    total_jobs = sum(status_counts.values())
    finished = status_counts.get("SUCCESS", 0) + status_counts.get("FAILURE", 0)
    total_preds = sum(counts.values())
    summary = {
        "total_jobs": total_jobs,
        "success_rate": round(100 * status_counts.get("SUCCESS", 0) / finished, 1) if finished else None,
        "total_predictions": total_preds,
        "uncertain_rate": round(100 * sum(uncertain.values()) / total_preds, 1) if total_preds else None,
        "users": get_user_model().objects.filter(analysis_jobs__isnull=False).distinct().count(),
    }

    recent = AnalysisJob.objects.select_related("user").annotate(person_count=Count("predictions")).order_by("-created_at")[:10]

    return render(
        request,
        "analyses/dashboard.html",
        {
            "chart_data": {"actions": action_chart, "status": status_chart, "inputs": input_chart},
            "summary": summary,
            "recent_jobs": recent,
            "ai_health": check_health(),
            "fastapi_url": settings.FASTAPI_URL,
        },
    )
