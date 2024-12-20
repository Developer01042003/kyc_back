# kyc/views.py
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from .models import KYC
from .serializers import KYCSerializer
from utils.aws_helper import AWSRekognition
import logging
import json
import base64

logger = logging.getLogger(__name__)

class KYCViewSet(viewsets.ModelViewSet):
    queryset = KYC.objects.all()
    serializer_class = KYCSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'], url_path='start-liveness-session')
    def start_liveness_session(self, request):
        """Start a new liveness detection session"""
        try:
            logger.info(f"Starting liveness session for user: {request.user.id}")
            aws = AWSRekognition()
            session_id = aws.create_face_liveness_session()
            
            logger.info(f"Created liveness session: {session_id}")
            return Response({
                'status': 'success',
                'sessionId': session_id,
                'message': 'Liveness session created successfully'
            })
        except Exception as e:
            logger.error(f"Error creating liveness session: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'Failed to create liveness session: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['post'], url_path='process-liveness')
    def process_liveness(self, request):
        """Process liveness detection frames"""
        try:
            session_id = request.data.get('sessionId')
            frames = request.data.get('frames', [])

            if not session_id or not frames:
                return Response({
                    'status': 'error',
                    'message': 'Session ID and frames are required'
                }, status=status.HTTP_400_BAD_REQUEST)

            logger.info(f"Processing liveness frames for session: {session_id}")
            aws = AWSRekognition()
            
            # Process frames for liveness
            result = aws.process_liveness_frames(session_id, frames)
            
            if result['isLive']:
                # Get the best frame for face verification
                best_frame = aws.get_best_frame(frames)
                if best_frame:
                    # Store the best frame or use it for further verification
                    image_hash = aws.generate_image_hash(best_frame)
                    file_name = f'liveness/{image_hash}.jpg'
                    selfie_url = aws.upload_to_s3(best_frame, file_name)

                    if selfie_url:
                        # Create or update KYC record
                        kyc, created = KYC.objects.get_or_create(
                            user=request.user,
                            defaults={
                                'selfie_url': selfie_url,
                                'is_verified': True
                            }
                        )
                        if not created:
                            kyc.selfie_url = selfie_url
                            kyc.is_verified = True
                            kyc.save()

            return Response({
                'status': 'success',
                'data': result
            })

        except Exception as e:
            logger.error(f"Error processing liveness: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'Failed to process liveness: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['post'], url_path='liveness-status')
    def get_liveness_status(self, request):
        """Get the current liveness verification status"""
        try:
            kyc = KYC.objects.filter(user=request.user).first()
            return Response({
                'status': 'success',
                'isVerified': bool(kyc and kyc.is_verified),
                'selfieUrl': kyc.selfie_url if kyc else None
            })
        except Exception as e:
            logger.error(f"Error getting liveness status: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'Failed to get liveness status: {str(e)}'
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
