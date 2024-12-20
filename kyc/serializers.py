from rest_framework import serializers
from .models import KYC

class KYCSerializer(serializers.ModelSerializer):
    class Meta:
        model = KYC
        fields = ('id', 'user', 'selfie_url', 'is_verified', 'created_at')
        read_only_fields = ('is_verified', 'selfie_url')

class KYCVerificationSerializer(serializers.Serializer):
    selfie = serializers.ImageField()