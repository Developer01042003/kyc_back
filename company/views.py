from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.hashers import check_password
from .models import Company, CompanyUser
from .serializers import (CompanySerializer, CompanySignupSerializer,
                        CompanyLoginSerializer, CompanyUserSerializer)
from users.models import User

class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    serializer_class = CompanySerializer

    @action(detail=False, methods=['post'])
    def signup(self, request):
        serializer = CompanySignupSerializer(data=request.data)
        if serializer.is_valid():
            company = serializer.save()
            return Response({
                'message': 'Company registered successfully. Please wait for admin verification.',
                'company': CompanySerializer(company).data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def login(self, request):
        serializer = CompanyLoginSerializer(data=request.data)
        if serializer.is_valid():
            try:
                company = Company.objects.get(email=serializer.validated_data['email'])
                if check_password(serializer.validated_data['password'], company.password):
                    refresh = RefreshToken.for_user(company)
                    return Response({
                        'refresh': str(refresh),
                        'access': str(refresh.access_token),
                        'company': CompanySerializer(company).data
                    })
                return Response({'error': 'Invalid credentials'}, 
                              status=status.HTTP_401_UNAUTHORIZED)
            except Company.DoesNotExist:
                return Response({'error': 'Company not found'}, 
                              status=status.HTTP_404_NOT_FOUND)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'])
    def check_user(self, request):
        if not request.user.is_authenticated or not hasattr(request.user, 'company') or not request.user.company.is_verified:
            return Response({'error': 'Unauthorized'}, 
                          status=status.HTTP_401_UNAUTHORIZED)
        
        email = request.data.get('email')
        user_exists = User.objects.filter(email=email, is_verified=True).exists()
        return Response({'exists': user_exists})

    @action(detail=False, methods=['post'])
    def add_user(self, request):
        if not request.user.is_authenticated or not hasattr(request.user, 'company') or not request.user.company.is_verified:
            return Response({'error': 'Unauthorized'}, 
                          status=status.HTTP_401_UNAUTHORIZED)

        try:
            user = User.objects.get(email=request.data['email'], is_verified=True)
            company_user = CompanyUser.objects.create(
                company=request.user.company,
                user=user
            )
            return Response(CompanyUserSerializer(company_user).data)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, 
                          status=status.HTTP_404_NOT_FOUND)