from rest_framework import serializers
from .models import Company, CompanyUser

class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = ('id', 'name', 'email', 'address', 'country', 
                 'is_verified', 'api_id', 'api_key')
        read_only_fields = ('is_verified', 'api_id', 'api_key')

class CompanySignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    
    class Meta:
        model = Company
        fields = ('name', 'email', 'password', 'address', 'country')

class CompanyLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

class CompanyUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CompanyUser
        fields = ('id', 'company', 'user', 'created_at')