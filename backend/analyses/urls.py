"""분석 HTML 화면 URL (API는 api_urls.py, 대시보드는 config/urls.py의 /dashboard/)."""

from django.urls import path

from . import page_views

app_name = "analyses"

urlpatterns = [
    path("", page_views.index, name="index"),
    path("<int:pk>/", page_views.detail, name="detail"),
]
