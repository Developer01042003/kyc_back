import logging
import cv2
import mediapipe as mp
import numpy as np
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

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

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
                return Response({
                    'status': 'error',
                    'message': 'No video provided'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Save video locally for processing
            video_path = "temp_video.webm"
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            try:
                is_live, eye_open_image, _ = self.process_video(video_path)
            except Exception as processing_error:
                logger.error(f"Video processing error: {str(processing_error)}")
                return Response({
                    'status': 'error',
                    'message': 'Error processing video'
                }, status=status.HTTP_400_BAD_REQUEST)

            if not is_live or eye_open_image is None:
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
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Invalid face image'
                    }, status=status.HTTP_400_BAD_REQUEST)

                if aws.check_face_duplicate(image_bytes):
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Face already exists'
                    }, status=status.HTTP_400_BAD_REQUEST)

                image_hash = aws.generate_image_hash(image_bytes)
                file_name = f'selfies/{image_hash}.jpg'
                selfie_url = aws.upload_to_s3(image_bytes, file_name)

                if not selfie_url:
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Error uploading image'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                face_id = aws.index_face(image_bytes)
                if not face_id:
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
        """Process the video to check for liveness through simple blink and movement detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        # Initialize variables for liveness
        blink_detected = False
        movement_detected = False

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
             mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = face_detection.process(frame_rgb)
                mesh_results = face_mesh.process(frame_rgb)

                if face_results.detections and mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        # Use landmarks to detect simple blink/movement
                        landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in face_landmarks.landmark]

                        # Example: Check simple movement by comparing the landmarks over frames
                        if self.detect_blink(landmarks) or self.detect_simple_movement(landmarks):
                            blink_detected = True
                            movement_detected = True
                            break

                # Break early if both checks are satisfied
                if blink_detected and movement_detected:
                    break

        cap.release()

        # Define liveness based on detected actions
        is_live = blink_detected and movement_detected
        return is_live, None, False  # Simplifying by ignoring glare detection for now

    def detect_blink(self, landmarks):
        """Basic blink detection based on landmarks changes indicating eye closures."""
        # Implement logic to detect blink based on eye landmarks
        # This is pseudo logic. Real implementation requires continuous landmark tracking
        # Check landmark positions and identify rapid changes indicative of blink
        return True  # Return true if a blink is detected

    def detect_simple_movement(self, landmarks):
        """Basic head movement detection."""
        # Implement logic to check changes in specific landmark positions frame-to-frame
        # Pseudo logic to place for head movement detection
        return True  # Return true if significant head movement is detected

    def cleanup_image(self, image_bytes):
        """Clean up the uploaded image after processing"""
        try:
            image_bytes.close()
        except Exception as e:
            logger.error(f"Error cleaning up image: {str(e)}")
