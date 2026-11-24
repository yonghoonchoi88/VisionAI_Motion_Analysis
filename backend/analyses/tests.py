"""
분석 기능 테스트 — 과제 기능 테스트 T02·T05~T09에 대응

FastAPI 호출(services.predict_video)과 카메라는 가짜로 바꿔, DB·권한·상태 전이·오류 처리를 검증한다.
Celery는 CELERY_TASK_ALWAYS_EAGER로 요청 안에서 즉시 실행한다.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .camera_capture import CameraCaptureError, capture_clip
from .models import AnalysisJob, PersonPrediction
from .services import InferenceInputError, InferenceResult, InferenceUnavailable
from .tasks import run_analysis

TMP_MEDIA = Path(tempfile.mkdtemp(prefix="test_media_"))


def fake_result(n_people: int = 2) -> InferenceResult:
    preds = [
        {
            "track_id": i + 1, "action_id": "A006", "action_label": "wave", "confidence": 0.82 - 0.4 * i,
            "margin": round(0.64 - 0.4 * i, 4), "runner_up_label": "sit_stand",
            "is_uncertain": i > 0, "window_count": 1, "window_mode": "whole_track",
            "valid_frame_ratio": 1.0, "duration_sec": 4.9, "probabilities": {"wave": 0.82, "sit_stand": 0.18},
        }
        for i in range(n_people)
    ]
    return InferenceResult(
        status="OK" if n_people else "NO_PERSON",
        message=f"{n_people}명의 동작을 예측했습니다." if n_people else "영상에서 사람을 찾지 못했습니다.",
        predictions=preds,
        video={"duration_sec": 5.0, "fps": 30.0},
        processing_time_sec=1.23,
        raw={},
    )


def video_upload(name: str = "wave.mp4") -> SimpleUploadedFile:
    return SimpleUploadedFile(name, b"\x00\x00\x00\x18ftypmp42" + b"0" * 2048, content_type="video/mp4")


@override_settings(MEDIA_ROOT=TMP_MEDIA, CELERY_TASK_ALWAYS_EAGER=True, ALLOWED_CAMERA_HOSTS=["192.168.0.10:8080"])
class AnalysisApiTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def setUp(self):
        User = get_user_model()
        self.a = User.objects.create_user("student_a", password="Vision!2026pw")
        self.b = User.objects.create_user("student_b", password="Vision!2026pw")
        self.admin = User.objects.create_user("teacher", password="Vision!2026pw", is_staff=True)
        self.client.force_login(self.a)
        self.list_url = reverse("analysisjob-list")

    def create_upload(self, result=None, side_effect=None, **extra):
        with mock.patch("analyses.tasks.predict_video", return_value=result or fake_result(), side_effect=side_effect):
            with self.captureOnCommitCallbacks(execute=True):        # 커밋 후 Celery 등록 → Eager 실행
                return self.client.post(self.list_url, {"input_type": "UPLOAD", "video_file": video_upload(), **extra})

    # ── T02 · T05 · T06: 업로드 → 작업 ID 즉시 반환 → 결과 DB 저장 ──
    def test_upload_creates_job_and_saves_predictions(self):
        resp = self.create_upload(title="손 흔들기", expected_action="wave")
        self.assertEqual(resp.status_code, 202)
        job_id = resp.json()["id"]
        self.assertTrue(resp.json()["celery_task_id"])

        job = AnalysisJob.objects.get(pk=job_id)
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)
        self.assertEqual(job.stage, AnalysisJob.Stage.DONE)
        self.assertEqual(job.result_status, "OK")
        self.assertEqual(job.inference_time_sec, 1.23)
        self.assertIsNotNone(job.processing_time_sec)               # Worker 구간은 단조 시계로 측정
        self.assertEqual(job.predictions.count(), 2)              # T04: 한 작업에 사람 2명
        self.assertTrue(Path(job.video_file.path).is_file())      # 영상은 MEDIA, DB에는 경로

        status = self.client.get(reverse("analysisjob-status", args=[job_id])).json()
        self.assertTrue(status["is_finished"])
        self.assertEqual([p["track_id"] for p in status["predictions"]], [1, 2])
        self.assertTrue(status["predictions"][1]["is_uncertain"])
        first = status["predictions"][0]                          # 판단 근거(1·2등 차이)도 저장·응답
        self.assertEqual((first["margin"], first["runner_up_label"]), (0.64, "sit_stand"))
        self.assertEqual(first["runner_up_name_ko"], "앉았다 일어나기")

    def test_old_ai_response_without_margin_still_saves(self):
        """margin이 없는 예전 AI 서버 응답도 저장된다 (배포 순서가 달라도 깨지지 않게)."""
        result = fake_result(1)
        for p in result.predictions:
            p.pop("margin"), p.pop("runner_up_label")
        job = AnalysisJob.objects.get(pk=self.create_upload(result=result).json()["id"])
        pred = job.predictions.get()
        self.assertEqual((job.status, pred.margin, pred.runner_up_label), ("SUCCESS", None, ""))

    def test_detail_page_shows_probabilities_sorted(self):
        job_id = self.create_upload().json()["id"]
        pred = AnalysisJob.objects.get(pk=job_id).predictions.first()
        self.assertEqual([k for k, _ in pred.sorted_probabilities], ["wave", "sit_stand"])
        html = self.client.get(reverse("analyses:detail", args=[job_id])).content.decode()
        self.assertIn("차이", html)

    def test_api_uses_v1_prefix(self):
        """개발 가이드 권장 경로: /api/v1/analyses/ · /api/v1/analyses/{id}/status/"""
        self.assertEqual(self.list_url, "/api/v1/analyses/")
        self.assertEqual(reverse("analysisjob-status", args=[7]), "/api/v1/analyses/7/status/")

    def test_admin_links_to_dashboard(self):
        """Django Admin 모든 화면 상단에서 관리자 대시보드로 이동할 수 있다 (개발 가이드 10단계)."""
        self.client.force_login(self.admin)
        self.admin.is_superuser = True
        self.admin.save()
        html = self.client.get("/admin/").content.decode()
        self.assertIn('href="/dashboard/"', html)
        self.assertIn("Vision AI 관리자", html)

    def test_index_shows_current_decision_rule(self):
        """화면 안내 문구는 AI 서버가 실제로 쓰는 기준을 보여 준다 (예전 '55%' 고정 문구 방지)."""
        from django.core.cache import cache

        cache.clear()
        with mock.patch("analyses.services.check_health", return_value={"decision_rule": {"min_confidence": 0.2, "min_margin": 0.11}}):
            html = self.client.get(reverse("analyses:index")).content.decode()
        self.assertIn("20% 미만이거나 2등과 차이가 11%p 미만", html)
        self.assertNotIn("55%", html)
        cache.clear()

    def test_dashboard_forbidden_page_is_friendly(self):
        """일반 회원의 /dashboard/ 접근 → 403 + 안내 화면 (T08·T09 사용자 친화적 오류)."""
        resp = self.client.get("/dashboard/")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("접근 권한이 없습니다", resp.content.decode())

    def test_no_person_is_success_with_zero_predictions(self):
        job_id = self.create_upload(result=fake_result(0)).json()["id"]
        job = AnalysisJob.objects.get(pk=job_id)
        self.assertEqual((job.status, job.result_status, job.predictions.count()), ("SUCCESS", "NO_PERSON", 0))

    # ── T09: 잘못된 파일·AI 서버 장애 → FAILURE + 사용자용 메시지 ──
    def test_corrupt_video_marks_failure(self):
        err = InferenceInputError("[VIDEO_UNREADABLE] 영상을 열 수 없습니다.")
        job_id = self.create_upload(side_effect=err).json()["id"]
        job = AnalysisJob.objects.get(pk=job_id)
        self.assertEqual(job.status, AnalysisJob.Status.FAILURE)
        self.assertIn("영상을 열 수 없습니다", job.error_message)
        self.assertIsNotNone(job.finished_at)

    def test_ai_server_down_marks_failure(self):
        job_id = self.create_upload(side_effect=InferenceUnavailable("AI 서버에 연결할 수 없습니다")).json()["id"]
        self.assertEqual(AnalysisJob.objects.get(pk=job_id).status, AnalysisJob.Status.FAILURE)

    def test_rejects_wrong_extension_and_missing_file(self):
        bad = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")
        r1 = self.client.post(self.list_url, {"input_type": "UPLOAD", "video_file": bad})
        r2 = self.client.post(self.list_url, {"input_type": "UPLOAD"})
        self.assertEqual((r1.status_code, r2.status_code), (400, 400))
        self.assertIn("video_file", r1.json())
        self.assertEqual(AnalysisJob.objects.count(), 0)

    # ── IP Webcam: 허용 목록만 ──
    def test_camera_url_must_be_allowed(self):
        r = self.client.post(self.list_url, {"input_type": "CAMERA", "camera_url": "http://evil.example.com/video"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("허용되지 않은 카메라", str(r.json()))
        r = self.client.post(self.list_url, {"input_type": "CAMERA", "camera_url": "http://192.168.0.10:8080/video", "capture_seconds": 30})
        self.assertEqual(r.status_code, 400)

    def test_camera_job_captures_then_infers(self):
        def fake_capture(url, seconds, out):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"fake")
            return mock.Mock(path=Path(out))
        with mock.patch("analyses.tasks.capture_clip", side_effect=fake_capture) as cap, \
             mock.patch("analyses.tasks.predict_video", return_value=fake_result(1)):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.client.post(self.list_url, {"input_type": "CAMERA", "camera_url": "http://192.168.0.10:8080/video", "capture_seconds": 8})
        self.assertEqual(r.status_code, 202)
        cap.assert_called_once()
        self.assertEqual(cap.call_args.args[1], 8)
        job = AnalysisJob.objects.get(pk=r.json()["id"])
        self.assertEqual((job.status, job.predictions.count()), ("SUCCESS", 1))
        self.assertTrue(job.video_file.name.startswith("captures/"))
        self.assertTrue(job.video_file.name.endswith(".webm"))

    def test_camera_disconnect_marks_failure(self):
        with mock.patch("analyses.tasks.capture_clip", side_effect=CameraCaptureError("수집 중 카메라 연결이 끊겼습니다")):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.client.post(self.list_url, {"input_type": "CAMERA", "camera_url": "http://192.168.0.10:8080/video"})
        job = AnalysisJob.objects.get(pk=r.json()["id"])
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("연결이 끊겼습니다", job.error_message)

    # ── T07: 다른 회원의 작업은 조회·수정·삭제 불가 ──
    def test_user_cannot_touch_others_jobs(self):
        job_id = self.create_upload().json()["id"]
        self.client.force_login(self.b)
        detail = reverse("analysisjob-detail", args=[job_id])
        self.assertEqual(self.client.get(detail).status_code, 404)
        self.assertEqual(self.client.patch(detail, {"title": "hack"}, content_type="application/json").status_code, 404)
        self.assertEqual(self.client.delete(detail).status_code, 404)
        self.assertEqual(self.client.get(reverse("analysisjob-status", args=[job_id])).status_code, 404)
        self.assertEqual(self.client.get(reverse("analyses:detail", args=[job_id])).status_code, 404)
        self.assertEqual(self.client.get(self.list_url).json()["count"], 0)
        self.assertTrue(AnalysisJob.objects.filter(pk=job_id).exists())

    def test_patch_only_changes_title_and_memo(self):
        job_id = self.create_upload().json()["id"]
        url = reverse("analysisjob-detail", args=[job_id])
        r = self.client.patch(url, {"title": "새 제목", "memo": "메모", "status": "FAILURE", "result_status": "X"}, content_type="application/json")
        self.assertEqual(r.status_code, 200)
        job = AnalysisJob.objects.get(pk=job_id)
        self.assertEqual((job.title, job.memo, job.status, job.result_status), ("새 제목", "메모", "SUCCESS", "OK"))
        self.assertEqual(self.client.put(url, {}, content_type="application/json").status_code, 405)

    def test_delete_removes_predictions_and_blocks_running(self):
        job_id = self.create_upload().json()["id"]
        path = Path(AnalysisJob.objects.get(pk=job_id).video_file.path)
        with self.captureOnCommitCallbacks(execute=True):
            self.assertEqual(self.client.delete(reverse("analysisjob-detail", args=[job_id])).status_code, 204)
        self.assertFalse(PersonPrediction.objects.filter(job_id=job_id).exists())
        self.assertFalse(path.exists())

        running = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status="STARTED")
        self.assertEqual(self.client.delete(reverse("analysisjob-detail", args=[running.pk])).status_code, 409)

    def test_task_is_idempotent_for_finished_job(self):
        job_id = self.create_upload().json()["id"]
        with mock.patch("analyses.tasks.predict_video") as pv:
            self.assertEqual(run_analysis.apply(args=[job_id]).get(), "SUCCESS")
            pv.assert_not_called()

    # ── T08: 관리자 화면 ──
    def test_dashboard_staff_only(self):
        self.create_upload()
        self.assertEqual(self.client.get(reverse("dashboard")).status_code, 403)
        self.client.force_login(self.admin)
        with mock.patch("analyses.page_views.check_health", return_value={"model_ready": True, "device": "cpu", "window_strategy": "auto"}):
            r = self.client.get(reverse("dashboard"))
        self.assertEqual(r.status_code, 200)
        data = r.context["chart_data"]
        self.assertEqual(len(data["actions"]["labels"]), 6)
        wave = data["actions"]["labels"].index("A006 손 흔들기")
        self.assertEqual(data["actions"]["confident"][wave] + data["actions"]["uncertain"][wave], 2)
        self.assertEqual(dict(zip(data["status"]["keys"], data["status"]["values"]))["SUCCESS"], 1)
        self.assertEqual(r.context["summary"]["total_jobs"], 1)

    def test_dashboard_compares_all_users(self):
        """관리자 대시보드의 입력 방식 비교는 전체 사용자 기준 (분석 화면은 본인 기록만)."""
        self.create_upload(expected_action="wave")                # 회원 A: 업로드 1건 (예측 2명)
        self.client.force_login(self.b)
        self.create_upload()                                       # 회원 B: 업로드 1건 (예측 2명)
        mine = {r["key"]: r for r in self.client.get(reverse("analyses:index")).context["comparison"]}
        self.assertEqual(mine["UPLOAD"]["jobs"], 1)                # B 화면에는 B 기록만

        self.client.force_login(self.admin)
        with mock.patch("analyses.page_views.check_health", return_value=None):
            r = self.client.get(reverse("dashboard"))
        rows = {c["key"]: c for c in r.context["comparison"]}
        self.assertEqual((rows["UPLOAD"]["jobs"], rows["UPLOAD"]["predictions"]), (2, 4))
        self.assertEqual((rows["UPLOAD"]["match"], rows["UPLOAD"]["labeled"]), (2, 2))   # 수행 동작을 고른 건 A만
        self.assertContains(r, "요청 사용자")
        self.assertContains(r, "전체 사용자")
        self.assertContains(r, "student_b")

    def test_staff_can_see_all_jobs_in_api(self):
        self.create_upload()
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(self.list_url).json()["count"], 1)

    def test_clock_adjustment_does_not_shrink_total_time(self):
        """벽시계가 뒤로 조정돼도 전체 처리 시간이 실제 처리 시간보다 짧게 나오지 않는다 (#3 사례 재현)."""
        from datetime import timedelta
        from django.utils import timezone as dj_tz

        def slow_predict(path):
            time.sleep(0.3)                                       # 실제로 0.3초 걸리는 추론
            return fake_result(1)

        base = dj_tz.now()
        fake_tz = mock.Mock()                                     # tasks 모듈이 보는 시계만 바꿈 (created_at은 실제 시계)
        fake_tz.now.side_effect = [base + timedelta(seconds=0.01), base + timedelta(seconds=0.05)]  # 시작·완료 차이 0.04초
        with mock.patch("analyses.tasks.timezone", fake_tz), \
             mock.patch("analyses.tasks.predict_video", side_effect=slow_predict):
            with self.captureOnCommitCallbacks(execute=True):
                job_id = self.client.post(self.list_url, {"input_type": "UPLOAD", "video_file": video_upload()}).json()["id"]
        job = AnalysisJob.objects.get(pk=job_id)
        self.assertLess((job.finished_at - job.started_at).total_seconds(), 0.1)   # 벽시계로는 0.04초
        self.assertGreaterEqual(job.processing_time_sec, 0.3)                       # 단조 시계로는 0.3초 이상
        self.assertGreaterEqual(job.total_time_sec, 0.3)

    def test_legacy_total_time_never_below_inference(self):
        """processing_time_sec이 없는 이전 기록도 AI 추론 시간보다 짧게 표시하지 않는다."""
        from datetime import timedelta
        job = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status="SUCCESS", inference_time_sec=2.361)
        AnalysisJob.objects.filter(pk=job.pk).update(
            started_at=job.created_at + timedelta(seconds=0.016), finished_at=job.created_at + timedelta(seconds=0.14))
        job.refresh_from_db()
        self.assertEqual(job.total_time_sec, 2.36)

    def test_index_compares_input_types(self):
        """과제 9-1 비교 화면: 입력 방식별 평균 신뢰도·선택 동작 일치율·처리 시간."""
        self.create_upload(expected_action="wave")                       # wave 0.82 · 0.42(불확실), 둘 다 일치
        def fake_capture(url, seconds, out):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"fake")
            return mock.Mock(path=Path(out))
        with mock.patch("analyses.tasks.capture_clip", side_effect=fake_capture), \
             mock.patch("analyses.tasks.predict_video", return_value=fake_result(1)):
            with self.captureOnCommitCallbacks(execute=True):
                self.client.post(self.list_url, {"input_type": "CAMERA", "camera_url": "http://192.168.0.10:8080/video",
                                                 "expected_action": "sit_stand"})    # 예측 wave → 불일치

        resp = self.client.get(reverse("analyses:index"))
        rows = {r["key"]: r for r in resp.context["comparison"]}
        up, cam = rows["UPLOAD"], rows["CAMERA"]
        self.assertEqual((up["jobs"], up["predictions"], up["uncertain"]), (1, 2, 1))
        self.assertAlmostEqual(up["avg_confidence"], 0.62)
        self.assertEqual((up["match"], up["labeled"], up["match_rate"]), (2, 2, 1.0))
        self.assertEqual((cam["match"], cam["labeled"], cam["match_rate"]), (0, 1, 0.0))
        self.assertIsNotNone(up["avg_total"])
        self.assertContains(resp, "수행 동작")
        self.assertContains(resp, "일치")

    def test_pages_render(self):
        job_id = self.create_upload().json()["id"]
        self.assertContains(self.client.get(reverse("analyses:index")), "AI 분석 시작")
        self.assertContains(self.client.get(reverse("analyses:detail", args=[job_id])), "Track 1")


class FakeStream:
    """IP Webcam 대역: 실제 카메라처럼 '실시간 속도'로 Frame을 내보낸다 (fps·해상도·끊김 시점 조절)."""

    def __init__(self, fps=60, size=(1280, 720), fail_after=None, opened=True):
        self.interval, self.size, self.fail_after, self.opened, self.n = 1 / fps, size, fail_after, opened, 0

    def __call__(self, *args, **kwargs):          # cv2.VideoCapture(...) 호출을 흉내
        return self

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(self.interval)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        return True, np.full((self.size[1], self.size[0], 3), self.n % 255, np.uint8)

    def release(self):
        pass


class CameraCaptureTests(TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def capture(self, stream, seconds=1.0):
        with mock.patch("analyses.camera_capture.cv2.VideoCapture", stream):
            return capture_clip("http://192.168.0.10:8080/video", seconds, self.tmp / "clip.webm")

    def test_capture_limits_fps_and_size(self):
        """60fps·1280x720 스트림 → 약 30fps·640px로 저장, 브라우저 재생용 WebM.

        수집 시간을 2초로 둔다: 1초면 CPU가 바쁠 때 Frame 생성·인코딩이 밀려 MIN_FRAMES(15)에 못 미쳐 가끔 실패했다.
        """
        res = self.capture(FakeStream(fps=60), seconds=2.0)
        self.assertTrue(res.path.is_file())
        self.assertEqual(res.path.suffix, ".webm")
        self.assertLessEqual(res.fps, 31)
        self.assertGreaterEqual(res.frames, 20)
        self.assertEqual((res.width, res.height), (640, 360))
        cap = cv2.VideoCapture(str(res.path))
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        self.assertEqual(n, res.frames)             # 저장한 Frame이 모두 다시 읽힘

    def test_unreachable_camera_raises(self):
        with self.assertRaisesRegex(CameraCaptureError, "연결할 수 없습니다"):
            self.capture(FakeStream(opened=False))

    def test_disconnect_mid_capture_raises(self):
        with self.assertRaisesRegex(CameraCaptureError, "끊겼습니다"):
            self.capture(FakeStream(fps=100, fail_after=5), seconds=2.0)
