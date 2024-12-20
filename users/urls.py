from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import UserViewSet

router = DefaultRouter()
router.register(r'', UserViewSet)

urlpatterns = router.urls + [
    path('signup/', UserViewSet.as_view({'post': 'signup'}), name='user-signup'),
    path('login/', UserViewSet.as_view({'post': 'login'}), name='user-login'),
]