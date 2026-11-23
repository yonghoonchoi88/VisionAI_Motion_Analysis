"""회원가입·로그인·비로그인 접근 제한."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class AccountTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        resp = self.client.post(
            reverse("signup"),
            {"username": "student01", "password1": "Vision!2026pw", "password2": "Vision!2026pw"},
        )
        self.assertRedirects(resp, reverse("analyses:index"))
        self.assertTrue(get_user_model().objects.filter(username="student01").exists())
        self.assertEqual(self.client.get(reverse("analyses:index")).status_code, 200)

    def test_signup_rejects_mismatched_password(self):
        resp = self.client.post(
            reverse("signup"), {"username": "x", "password1": "Vision!2026pw", "password2": "other!2026pw"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(get_user_model().objects.filter(username="x").exists())

    def test_login_logout(self):
        get_user_model().objects.create_user("student02", password="Vision!2026pw")
        resp = self.client.post(reverse("login"), {"username": "student02", "password": "Vision!2026pw"})
        self.assertRedirects(resp, reverse("analyses:index"))
        self.client.post(reverse("logout"))
        self.assertRedirects(self.client.get(reverse("analyses:index")), f"{reverse('login')}?next={reverse('analyses:index')}")

    def test_anonymous_is_blocked(self):
        self.assertEqual(self.client.get(reverse("analyses:index")).status_code, 302)
        self.assertEqual(self.client.get("/api/analyses/").status_code, 403)


class ProfileTests(TestCase):
    def setUp(self):
        from analyses.models import AnalysisJob, PersonPrediction
        User = get_user_model()
        self.me = User.objects.create_user("student01", password="Vision!2026pw")
        other = User.objects.create_user("student02", password="Vision!2026pw")
        job = AnalysisJob.objects.create(user=self.me, input_type="UPLOAD", status="SUCCESS")
        PersonPrediction.objects.create(job=job, track_id=1, action_label="wave", action_id="A006", confidence=0.7)
        AnalysisJob.objects.create(user=self.me, input_type="CAMERA", status="FAILURE")
        AnalysisJob.objects.create(user=other, input_type="UPLOAD", status="SUCCESS")   # 남의 기록은 집계 제외

    def test_profile_shows_only_my_summary(self):
        self.client.force_login(self.me)
        resp = self.client.get(reverse("profile"))
        self.assertEqual(resp.status_code, 200)
        s = resp.context["stats"]
        self.assertEqual((s["total"], s["success"], s["failure"], s["upload"], s["camera"], s["predictions"]), (2, 1, 1, 1, 1, 1))
        self.assertEqual(resp.context["top_actions"][0]["action_id"], "A006")
        self.assertContains(resp, "student01")

    def test_profile_requires_login(self):
        self.assertEqual(self.client.get(reverse("profile")).status_code, 302)

    def test_password_change(self):
        self.client.force_login(self.me)
        resp = self.client.post(reverse("password_change"), {
            "old_password": "Vision!2026pw", "new_password1": "NewVision!2027", "new_password2": "NewVision!2027"})
        self.assertRedirects(resp, reverse("password_change_done"))
        self.me.refresh_from_db()
        self.assertTrue(self.me.check_password("NewVision!2027"))
