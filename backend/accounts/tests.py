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
