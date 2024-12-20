from django.contrib import admin
from .models import Company, CompanyUser
import uuid

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'is_verified', 'api_id', 'api_key', 'created_at')
    list_filter = ('is_verified', 'country')
    search_fields = ('name', 'email')
    readonly_fields = ('api_id', 'api_key')
    actions = ['verify_companies']

    def verify_companies(self, request, queryset):
        for company in queryset:
            if not company.is_verified:
                company.is_verified = True
                company.api_id = uuid.uuid4()
                company.api_key = uuid.uuid4()
                company.save()
    verify_companies.short_description = "Verify selected companies and generate API credentials"

@admin.register(CompanyUser)
class CompanyUserAdmin(admin.ModelAdmin):
    list_display = ('company', 'user', 'created_at')
    list_filter = ('company', 'created_at')
    search_fields = ('company__name', 'user__email')