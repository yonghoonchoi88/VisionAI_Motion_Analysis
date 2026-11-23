from django.contrib.auth import views as auth_views
from django.urls import path, reverse_lazy

from .views import SignUpView, profile

urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),
    path("login/", auth_views.LoginView.as_view(redirect_authenticated_user=True), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),     # Django 5+: POST로만 로그아웃
    path("me/", profile, name="profile"),
    path("password/", auth_views.PasswordChangeView.as_view(success_url=reverse_lazy("password_change_done")), name="password_change"),
    path("password/done/", auth_views.PasswordChangeDoneView.as_view(), name="password_change_done"),
]
