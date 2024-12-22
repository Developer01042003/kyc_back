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

# Initialize MediaPipe Face Detection and Landmark Model
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
                is_live, eye_open_image, glare_detected = self.process_video(video_path)
            except Exception as processing_error:
                logger.error(f"Video processing error: {str(processing_error)}")
                return Response({
                    'status': 'error',
                    'message': 'Error processing video'
                }, status=status.HTTP_400_BAD_REQUEST)

            if glare_detected:
                os.remove(video_path)
                return Response({
                    'status': 'error',
                    'message': 'Glare detected in video, please try again.'
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
        """Process the video and return liveness status, eye-open frame, and glare detection."""
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
             mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Unable to open video file")

            EAR_THRESHOLD = 0.3
            BLINK_COUNT = 0
            eye_open_frame = None
            glare_detected = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                square_frame = self.crop_to_square(frame)
                rgb_frame = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
                face_results = face_detection.process(rgb_frame)

                if face_results.detections:
                    mesh_results = face_mesh.process(rgb_frame)
                    if mesh_results.multi_face_landmarks:
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            landmarks = [
                                (int(landmark.x * square_frame.shape[1]),
                                 int(landmark.y * square_frame.shape[0]))
                                for landmark in face_landmarks.landmark
                            ]

                            left_ear = self.calculate_ear(landmarks, [33, 133, 153, 154, 133, 33])
                            right_ear = self.calculate_ear(landmarks, [362, 263, 362, 467, 463, 362])

                            if left_ear > EAR_THRESHOLD or right_ear > EAR_THRESHOLD:
                                if eye_open_frame is None:
                                    eye_open_frame = frame

                            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                                BLINK_COUNT += 1

                if self.detect_glare(square_frame):
                    glare_detected = True

            cap.release()

            is_live = BLINK_COUNT >= 1
            if eye_open_frame is not None:
                _, buffer = cv2.imencode(".jpg", eye_open_frame)
                eye_open_image_bytes = BytesIO(buffer.tobytes())
                return is_live, eye_open_image_bytes, glare_detected

            return is_live, None, glare_detected

    def crop_to_square(self, frame):
        """Crop the frame to make it square."""
        height, width, _ = frame.shape
        size = min(height, width)
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        return frame[start_y:start_y + size, start_x:start_x + size]

    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR)"""
        eye_points = [landmarks[i] for i in eye_indices]
        A = self.euclidean_distance(eye_points[1], eye_points[5])
        B = self.euclidean_distance(eye_points[2], eye_points[4])
        C = self.euclidean_distance(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def detect_glare(self, frame):
        """Detect glare in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) > 220

    def cleanup_image(self, image_bytes):
        """Clean up the uploaded image after processing"""
        try:
            image_bytes.close()
        except Exception as e:
            logger.error(f"Error cleaning up image: {str(e)}")
