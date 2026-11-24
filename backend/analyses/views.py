"""
DRF 분석 API (views.py)

    POST   /api/v1/analyses/               분석 요청 → DB 저장 → (커밋 후) Celery 등록 → 202 + 작업 ID
    GET    /api/v1/analyses/               내 분석 목록 (관리자는 전체)
    GET    /api/v1/analyses/{id}/          상세 + 사람별 결과
    PATCH  /api/v1/analyses/{id}/          제목·메모만 수정
    DELETE /api/v1/analyses/{id}/          삭제 (진행 중이면 409, 사람별 예측·영상 파일도 함께 삭제)
    GET    /api/v1/analyses/{id}/status/   폴링용 상태
"""

from __future__ import annotations

import uuid

from django.db import transaction
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import AnalysisJob
from .serializers import (
    AnalysisJobCreateSerializer,
    AnalysisJobSerializer,
    AnalysisJobStatusSerializer,
    AnalysisJobUpdateSerializer,
)
from .tasks import run_analysis


class IsOwnerOrStaff(permissions.BasePermission):
    def has_object_permission(self, request, view, obj) -> bool:
        return request.user.is_staff or obj.user_id == request.user.id


def enqueue_analysis(job: AnalysisJob) -> None:
    """Task ID를 먼저 정해 저장하고, DB 커밋이 끝난 '뒤'에 Celery에 등록한다.

    커밋 전에 등록하면 Worker가 아직 저장되지 않은 Job을 찾다가 실패할 수 있다.
    """
    task_id = str(uuid.uuid4())
    job.celery_task_id = task_id
    job.save(update_fields=["celery_task_id"])
    transaction.on_commit(lambda: run_analysis.apply_async(args=[job.pk], task_id=task_id))


class AnalysisJobViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrStaff]
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]   # PUT(전체 교체) 금지

    def get_queryset(self):
        qs = AnalysisJob.objects.select_related("user").prefetch_related("predictions")
        # 일반 회원은 자기 것만 '존재'한다 → 남의 작업 ID로 접근하면 404 (존재 여부도 숨김)
        return qs if self.request.user.is_staff else qs.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == "create":
            return AnalysisJobCreateSerializer
        if self.action == "partial_update":
            return AnalysisJobUpdateSerializer
        if self.action == "status":
            return AnalysisJobStatusSerializer
        return AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        with transaction.atomic():
            job = serializer.save(user=request.user)
            enqueue_analysis(job)
        job.refresh_from_db()            # Eager 모드에서는 이미 처리됐을 수 있음
        body = AnalysisJobSerializer(job, context=self.get_serializer_context()).data
        return Response(body, status=status.HTTP_202_ACCEPTED)

    def partial_update(self, request, *args, **kwargs):
        job = self.get_object()
        serializer = self.get_serializer(job, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(AnalysisJobSerializer(job, context=self.get_serializer_context()).data)

    def destroy(self, request, *args, **kwargs):
        job = self.get_object()
        if not job.is_finished:
            return Response(
                {"detail": "진행 중인 작업은 삭제할 수 없습니다. 완료 또는 실패 후 삭제하세요."},
                status=status.HTTP_409_CONFLICT,
            )
        video = job.video_file.name if job.video_file else None
        storage = job.video_file.storage if job.video_file else None
        job.delete()                                         # PersonPrediction은 CASCADE로 함께 삭제
        if video and storage:
            transaction.on_commit(lambda: storage.delete(video))   # 영상 파일도 정리
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        return Response(self.get_serializer(self.get_object()).data)
