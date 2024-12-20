# views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from .models import KYC
from .serializers import KYCSerializer, KYCVerificationSerializer
from utils.aws_helper import AWSRekognition

class KYCViewSet(viewsets.ModelViewSet):
    queryset = KYC.objects.all()
    serializer_class = KYCSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'])
    def start_liveness_session(self, request):
        """Start a new liveness detection session"""
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
        """Process frames for liveness detection"""
        try:
            session_id = request.data.get('sessionId')
            frames = request.data.get('frames')

            if not session_id or not frames:
                return Response({
                    'error': 'Missing sessionId or frames'
                }, status=status.HTTP_400_BAD_REQUEST)

            aws = AWSRekognition()
            result = aws.process_liveness_frames(session_id, frames)

            if result['isLive']:
                # Get best frame for KYC
                best_frame = aws.get_best_frame(frames)
                if best_frame:
                    result['bestFrame'] = best_frame

            return Response(result)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Your existing create method remains the same
    def create(self, request):
        serializer = KYCVerificationSerializer(data=request.data)
        if serializer.is_valid():
            aws = AWSRekognition()
            
            # Read image file
            image_file = serializer.validated_data['selfie']
            image_bytes = image_file.read()

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
