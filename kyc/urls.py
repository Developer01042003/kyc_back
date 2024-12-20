# kyc/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import KYCViewSet

router = DefaultRouter()
router.register(r'kyc', KYCViewSet, basename='kyc')

urlpatterns = [
    path('', include(router.urls)),
]
