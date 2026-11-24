"""최상위 URL: Admin, 회원, 분석(화면+API), 관리자 대시보드."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

from analyses.page_views import dashboard

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="analyses:index", permanent=False)),
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("analyses/", include("analyses.urls")),
    path("api/v1/", include("analyses.api_urls")),          # 9~10일차 개발 가이드의 권장 경로 규칙 (/api/v1/...)
    path("dashboard/", dashboard, name="dashboard"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
