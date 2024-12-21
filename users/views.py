# views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework.permissions import AllowAny
from .models import User
from .serializers import UserSerializer, UserLoginSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def signup(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            # Changed to match frontend's expected token format
            token = RefreshToken.for_user(user)
            return Response({
                'token': str(token.access_token),  # Changed from 'access' to 'token'
                'user': serializer.data
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def login(self, request):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            
            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                return Response({
                    'error': 'Invalid credentials'  # Generic error message for security
                }, status=status.HTTP_401_UNAUTHORIZED)

            user = authenticate(username=user.username, password=password)
            if user:
                # Changed to match frontend's expected token format
                token = RefreshToken.for_user(user)
                return Response({
                    'token': str(token.access_token),  # Changed from 'access' to 'token'
                    'user': UserSerializer(user).data
                })
            return Response({
                'error': 'Invalid credentials'
            }, status=status.HTTP_401_UNAUTHORIZED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
