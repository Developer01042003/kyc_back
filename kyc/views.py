from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import KYC
from .serializers import KYCSerializer
from utils.aws_helper import AWSRekognition
import logging

logger = logging.getLogger(__name__)

class KYCViewSet(viewsets.ModelViewSet):
    queryset = KYC.objects.all()
    serializer_class = KYCSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'], url_path='start-liveness-session')
    def start_liveness_session(self, request):
        """Start a new liveness detection session"""
        try:
            aws = AWSRekognition()
            session_id = aws.create_face_liveness_session()
            
            logger.info(f"Created liveness session for user: {request.user.id}")
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

            aws = AWSRekognition()
            result = aws.process_liveness_frames(session_id, frames)
            
            return Response({
                'status': 'success',
                'isLive': result['isLive'],
                'confidence': result['confidence'],
                'sessionStatus': result['status']
            })

        except Exception as e:
            logger.error(f"Error processing liveness: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'Failed to process liveness: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'], url_path='session-results')
    def get_session_results(self, request):
        """Get results of a liveness session"""
        try:
            session_id = request.query_params.get('sessionId')
            if not session_id:
                return Response({
                    'status': 'error',
                    'message': 'Session ID is required'
                }, status=status.HTTP_400_BAD_REQUEST)

            aws = AWSRekognition()
            result = aws.get_session_results(session_id)
            
            return Response({
                'status': 'success',
                'isLive': result['isLive'],
                'confidence': result['confidence'],
                'sessionStatus': result['status']
            })

        except Exception as e:
            logger.error(f"Error getting session results: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'Failed to get session results: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def create(self, request):
        """Create a new KYC record with selfie verification"""
        try:
            serializer = KYCVerificationSerializer(data=request.data)
            if serializer.is_valid():
                aws = AWSRekognition()
                
                # Read image file
                image_file = serializer.validated_data['selfie']
                image_bytes = image_file.read()

                # Verify face
                if not aws.verify_face(image_bytes):
                    return Response({
                        'status': 'error',
                        'message': 'Invalid face image'
                    }, status=status.HTTP_400_BAD_REQUEST)

                # Check for duplicates
                if aws.check_face_duplicate(image_bytes):
                    return Response({
                        'status': 'error',
                        'message': 'Face already exists'
                    }, status=status.HTTP_400_BAD_REQUEST)

                # Generate image hash and upload to S3
                image_hash = aws.generate_image_hash(image_bytes)
                file_name = f'selfies/{image_hash}.jpg'
                selfie_url = aws.upload_to_s3(image_bytes, file_name)

                if not selfie_url:
                    return Response({
                        'status': 'error',
                        'message': 'Error uploading image'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Index face
                face_id = aws.index_face(image_bytes)
                if not face_id:
                    return Response({
                        'status': 'error',
                        'message': 'Error indexing face'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

                return Response(
                    KYCSerializer(kyc).data, 
                    status=status.HTTP_201_CREATED
                )
            
            # If serializer is not valid
            return Response(
                {
                    'status': 'error',
                    'message': 'Invalid data',
                    'errors': serializer.errors
                }, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        except Exception as e:
            logger.error(f"KYC creation error: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'KYC creation failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
