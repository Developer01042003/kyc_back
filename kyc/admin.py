from django.contrib import admin
from .models import KYC

@admin.register(KYC)
class KYCAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_verified', 'created_at')
    list_filter = ('is_verified',)
    search_fields = ('user__email', 'user__username')
    readonly_fields = ('face_id', 'selfie_url')