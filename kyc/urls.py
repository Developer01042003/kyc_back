# urls.py
from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import KYCViewSet

router = DefaultRouter()
router.register(r'', KYCViewSet, basename='kyc')

# The following URLs will be automatically generated:
# POST /start-liveness-session/ - Start new liveness session
# POST /check-liveness/ - Process liveness frames
# POST / - Submit KYC (existing endpoint)
# GET / - List KYCs (existing endpoint)
# GET /{id}/ - Retrieve KYC (existing endpoint)

urlpatterns = router.urls
