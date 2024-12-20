# views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from .models import KYC
from .serializers import KYCSerializer, KYCVerificationSerializer
from utils.aws_helper import AWSRekognition
import base64
import json

class KYCViewSet(viewsets.ModelViewSet):
    queryset = KYC.objects.all()
    serializer_class = KYCSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'])
    def start_liveness_session(self, request):
        try:
            aws = AWSRekognition()
            session_id = aws.create_face_liveness_session()
            return Response({
                'sessionId': session_id
            })
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['post'])
    def check_liveness(self, request):
        try:
            aws = AWSRekognition()
            session_id = request.data.get('sessionId')
            frames = request.data.get('frames')

            liveness_result = aws.check_face_liveness(session_id, frames)
            return Response(liveness_result)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def create(self, request):
        serializer = KYCVerificationSerializer(data=request.data)
        if serializer.is_valid():
            aws = AWSRekognition()
            
            # Get base64 image and convert to bytes
            image_data = serializer.validated_data['selfie']
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # Handle base64 string from frontend
                image_bytes = base64.b64decode(image_data.split(',')[1])
            else:
                # Handle file upload
                image_bytes = image_data.read()

            # Verify face
            if not aws.verify_face(image_bytes):
                return Response({'error': 'Invalid face image'}, 
                              status=status.HTTP_400_BAD_REQUEST)

            # Check for duplicates
            if aws.check_face_duplicate(image_bytes):
                return Response({'error': 'Face already exists'}, 
                              status=status.HTTP_400_BAD_REQUEST)

            # Generate image hash and upload to S3
            image_hash = aws.generate_image_hash(image_bytes)
            file_name = f'selfies/{image_hash}.jpg'
            selfie_url = aws.upload_to_s3(image_bytes, file_name)

            if not selfie_url:
                return Response({'error': 'Error uploading image'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Index face
            face_id = aws.index_face(image_bytes)
            if not face_id:
                return Response({'error': 'Error indexing face'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Create KYC record
            kyc = KYC.objects.create(
                user=request.user,
                selfie_url=selfie_url,
                face_id=face_id,
                is_verified=True
            )

            # Update user verification status
            request.user.is_verified = True
            request.user.save()

            return Response(KYCSerializer(kyc).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
