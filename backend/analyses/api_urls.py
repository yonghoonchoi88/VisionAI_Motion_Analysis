"""DRF Router: /api/v1/analyses/ (CRUD) + /api/v1/analyses/{id}/status/"""

from rest_framework.routers import DefaultRouter

from .views import AnalysisJobViewSet

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysisjob")

urlpatterns = router.urls
