from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import KYC
from .serializers import KYCSerializer
from .serializers import KYCVerificationSerializer
from utils.aws_helper import AWSRekognition
import logging
import cv2
import dlib
import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from io import BytesIO
import os

logger = logging.getLogger(__name__)

# Initialize Dlib detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

            # Process the video to check liveness and glare
            video_path = "temp_video.webm"
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            is_live, eye_open_image, glare_detected = self.process_video(video_path)
            
            # Handle glare detection
            if glare_detected:
                os.remove(video_path)
                return Response({
                    'status': 'error',
                    'message': 'Glare detected in video, please try again.'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Handle liveness failure
            if not is_live or eye_open_image is None:
                os.remove(video_path)
                return Response({
                    'status': 'error',
                    'message': 'Liveness check failed. Please try again.'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Convert eye-open image to Django `UploadedFile`
            eye_open_image.seek(0)
            selfie_file = SimpleUploadedFile("eye_open_selfie.jpg", eye_open_image.read(), content_type="image/jpeg")

            # Now pass the selfie image to the serializer
            serializer = KYCVerificationSerializer(data={'selfie': selfie_file})
            if serializer.is_valid():
                aws = AWSRekognition()
                
                # Read image file
                image_file = serializer.validated_data['selfie']
                image_bytes = image_file.read()

                # Verify face
                if not aws.verify_face(image_bytes):
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Invalid face image'
                    }, status=status.HTTP_400_BAD_REQUEST)

                # Check for duplicates
                if aws.check_face_duplicate(image_bytes):
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Face already exists'
                    }, status=status.HTTP_400_BAD_REQUEST)

                # Generate image hash and upload to S3
                image_hash = aws.generate_image_hash(image_bytes)
                file_name = f'selfies/{image_hash}.jpg'
                selfie_url = aws.upload_to_s3(image_bytes, file_name)

                if not selfie_url:
                    os.remove(video_path)
                    return Response({
                        'status': 'error',
                        'message': 'Error uploading image'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Index face
                face_id = aws.index_face(image_bytes)
                if not face_id:
                    os.remove(video_path)
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

                # Clean up: delete video and image from the backend
                os.remove(video_path)  # Delete video after processing
                self.cleanup_image(image_bytes)  # Delete image after upload to S3

                return Response(
                    KYCSerializer(kyc).data, 
                    status=status.HTTP_201_CREATED
                )

            # If serializer is not valid
            os.remove(video_path)
            return Response({
                'status': 'error',
                'message': 'Invalid data',
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"KYC creation error: {str(e)}")
            os.remove(video_path)
            return Response({
                'status': 'error',
                'message': f'KYC creation failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def process_video(self, video_path):
        """Process the video and return liveness status, eye-open frame, and glare detection."""
        cap = cv2.VideoCapture(video_path)
        EAR_THRESHOLD = 0.2
        BLINK_COUNT = 0
        eye_open_frame = None
        glare_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

                left_eye_points = [36, 37, 38, 39, 40, 41]
                right_eye_points = [42, 43, 44, 45, 46, 47]
                left_ear = self.extract_eye_aspect_ratio(landmarks_points, left_eye_points)
                right_ear = self.extract_eye_aspect_ratio(landmarks_points, right_eye_points)

                if left_ear > EAR_THRESHOLD and right_ear > EAR_THRESHOLD:
                    if eye_open_frame is None:
                        eye_open_frame = frame

                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    BLINK_COUNT += 1

            # Check for glare: Calculate mean brightness of the image
            if self.check_for_glare(frame):
                glare_detected = True

        cap.release()

        is_live = BLINK_COUNT >= 2
        if is_live and eye_open_frame is not None:
            _, buffer = cv2.imencode(".jpg", eye_open_frame)
            eye_open_image_bytes = BytesIO(buffer.tobytes())
            return is_live, eye_open_image_bytes, glare_detected

        return is_live, None, glare_detected

    def extract_eye_aspect_ratio(self, landmarks, eye_points):
        """Calculate the Eye Aspect Ratio (EAR)."""
        def euclidean_distance(point1, point2):
            return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

        A = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
        B = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
        C = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
        return (A + B) / (2.0 * C)

    def check_for_glare(self, frame):
        """Detect glare in the frame based on brightness levels."""
        # Convert frame to grayscale and calculate its brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # If the mean brightness is above a threshold, glare is likely present
        return mean_brightness > 200  # Threshold for glare (adjust as needed)

    def cleanup_image(self, image_bytes):
        """Clean up the uploaded image after it's uploaded to S3."""
        try:
            image_bytes.close()  # Close the image file to free up resources
        except Exception as e:
            logger.error(f"Error cleaning up image: {str(e)}")
