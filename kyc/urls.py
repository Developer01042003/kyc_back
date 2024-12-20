from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import KYCViewSet

router = DefaultRouter()
router.register(r'', KYCViewSet)

urlpatterns = router.urls