from django.db import models
from users.models import User

class KYC(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    selfie_url = models.URLField()
    face_id = models.CharField(max_length=255)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"KYC for {self.user.email}"

    class Meta:
        verbose_name = "KYC"
        verbose_name_plural = "KYCs"