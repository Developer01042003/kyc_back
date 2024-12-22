import logging
import cv2
import numpy as np
from mtcnn import MTCNN
from django.core.files.uploadedfile import SimpleUploadedFile
from io import BytesIO
import os

from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import KYC
from .serializers import KYCSerializer, KYCVerificationSerializer
from utils.aws_helper import AWSRekognition

logger = logging.getLogger(__name__)

class KYCViewSet(viewsets.ModelViewSet):
    queryset = KYC.objects.all()
    serializer_class = KYCSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request):
        """Create a new KYC record with selfie verification"""
        try:
            # Extract video file from the request
            video_file = request.FILES.get("video")
            if not video_file:
                logger.error("No video file provided in the request.")
                return Response({
                    'status': 'error',
                    'message': 'No video provided'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Log video file details
            logger.info(f"Received video file: {video_file.name}, size: {video_file.size} bytes")

            # Save video locally for processing
            video_path = "temp_video.webm"
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            try:
                is_live, eye_open_image, _ = self.process_video(video_path)
            except Exception as processing_error:
                logger.error(f"Video processing error: {str(processing_error)}")
                os.remove(video_path)
                return Response({
                    'status': 'error',
                    'message': 'Error processing video'
                }, status=status.HTTP_400_BAD_REQUEST)

            if not is_live or eye_open_image is None:
                logger.warning("Liveness check failed.")
                os.remove(video_path)
                return Response({
                    'status': 'error',
                    'message': 'Liveness check failed. Please try again.'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Convert eye-open image to Django `UploadedFile`
            eye_open_image.seek(0)
            selfie_file = SimpleUploadedFile("eye_open_selfie.jpg", eye_open_image.read(), content_type="image/jpeg")

            serializer = KYCVerificationSerializer(data={'selfie': selfie_file})
            if serializer.is_valid():
                aws = AWSRekognition()
                image_file = serializer.validated_data['selfie']
                image_bytes = image_file.read()

                if not aws.verify_face(image_bytes):
                    logger.warning("AWS Rekognition failed to verify face.")
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Invalid face image'
                    }, status=status.HTTP_400_BAD_REQUEST)

                if aws.check_face_duplicate(image_bytes):
                    logger.warning("Duplicate face detected.")
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Face already exists'
                    }, status=status.HTTP_400_BAD_REQUEST)

                image_hash = aws.generate_image_hash(image_bytes)
                file_name = f'selfies/{image_hash}.jpg'
                selfie_url = aws.upload_to_s3(image_bytes, file_name)

                if not selfie_url:
                    logger.error("Error uploading image to S3.")
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Error uploading image'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                face_id = aws.index_face(image_bytes)
                if not face_id:
                    logger.error("Error indexing face with AWS Rekognition.")
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Error indexing face'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                kyc = KYC.objects.create(
                    user=request.user,
                    selfie_url=selfie_url,
                    face_id=face_id,
                    is_verified=True
                )
                request.user.is_verified = True
                request.user.save()

                os.remove(video_path)
                self.cleanup_image(image_bytes)

                return Response(
                    KYCSerializer(kyc).data,
                    status=status.HTTP_201_CREATED
                )

            logger.error(f"Invalid data: {serializer.errors}")
            os.remove(video_path)
            return Response({
                'status': 'error',
                'message': 'Invalid data',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"KYC creation error: {str(e)}")
            return Response({
                'status': 'error',
                'message': f'KYC creation failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def process_video(self, video_path):
        """Process the video to check for liveness through blink detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        # Initialize MTCNN for face detection
        detector = MTCNN()
        blink_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces and landmarks
            results = detector.detect_faces(frame)
            for result in results:
                keypoints = result['keypoints']
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']

                # Calculate EAR for both eyes
                left_ear = self.calculate_ear(left_eye, frame)
                right_ear = self.calculate_ear(right_eye, frame)

                # Check if EAR is below a threshold indicating a blink
                if left_ear < 0.2 or right_ear < 0.2:  # Adjust threshold as needed
                    blink_detected = True
                    break

            if blink_detected:
                break

        cap.release()

        # Define liveness based on detected actions
        is_live = blink_detected
        return is_live, None, False  # Simplifying by ignoring glare detection for now

    def calculate_ear(self, eye, frame):
        """Calculate Eye Aspect Ratio (EAR) for a given eye."""
        # This is a simplified example; you may need to adjust based on actual landmarks
        # EAR calculation typically involves multiple points around the eye
        # Here, we assume `eye` is a tuple of (x, y) coordinates for the eye center
        # You would need to calculate distances between specific points around the eye
        # For demonstration, we use a placeholder value
        return 0.3  # Placeholder value; implement actual EAR calculation

    def cleanup_image(self, image_bytes):
        """Clean up the uploaded image after processing"""
        try:
            image_bytes.close()
        except Exception as e:
            logger.error(f"Error cleaning up image: {str(e)}")
