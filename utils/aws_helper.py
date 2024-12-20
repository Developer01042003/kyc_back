import boto3
import hashlib
import time
import base64
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class AWSRekognition:
    def __init__(self):
        self.client = boto3.client('rekognition',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME
        )
        self.s3_client = boto3.client('s3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME
        )
        self.collection_id = 'unique_faces'
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure collection exists, create if it doesn't"""
        try:
            self.client.describe_collection(CollectionId=self.collection_id)
            logger.info(f"Collection {self.collection_id} exists")
        except self.client.exceptions.ResourceNotFoundException:
            try:
                self.client.create_collection(CollectionId=self.collection_id)
                logger.info(f"Created collection {self.collection_id}")
            except Exception as e:
                logger.error(f"Error creating collection: {str(e)}")
                raise

    def create_face_liveness_session(self):
        """Create a new face liveness session"""
        try:
            response = self.client.create_face_liveness_session()
            session_id = response.get("SessionId")
            logger.info(f'Created liveness session: {session_id}')
            return session_id
        except Exception as e:
            logger.error(f"Error creating liveness session: {str(e)}")
            raise

    def get_session_results(self, session_id):
        """Get results of a liveness session"""
        try:
            response = self.client.get_face_liveness_session_results(
                SessionId=session_id
            )
            confidence = response.get("Confidence")
            status = response.get("Status")
            
            logger.info(f'Session {session_id} - Confidence: {confidence}%, Status: {status}')
            return {
                'confidence': confidence,
                'status': status,
                'isLive': confidence > 90 if confidence else False
            }
        except Exception as e:
            logger.error(f"Error getting session results: {str(e)}")
            raise

    def process_liveness_frames(self, session_id, frames):
        """Process multiple frames for liveness detection"""
        try:
            # Process each frame
            for frame in frames:
                # Convert base64 frame to bytes if needed
                if isinstance(frame, str):
                    if frame.startswith('data:image'):
                        image_bytes = base64.b64decode(frame.split(',')[1])
                    else:
                        image_bytes = base64.b64decode(frame)
                else:
                    image_bytes = frame

                # Update the liveness session with the frame
                self.client.update_face_liveness_session(
                    SessionId=session_id,
                    Image={'Bytes': image_bytes}
                )

            # Get final results after processing all frames
            return self.get_session_results(session_id)
        except Exception as e:
            logger.error(f"Error processing liveness frames: {str(e)}")
            raise

    def verify_face(self, image_bytes):
        try:
            response = self.client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            return len(response['FaceDetails']) == 1
        except Exception as e:
            logger.error(f"Error verifying face: {str(e)}")
            return False

    def check_face_duplicate(self, image_bytes):
        try:
            response = self.client.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                MaxFaces=1,
                FaceMatchThreshold=95
            )
            return len(response['FaceMatches']) > 0
        except Exception as e:
            logger.error(f"Error checking face duplicate: {str(e)}")
            return False

    def index_face(self, image_bytes):
        try:
            response = self.client.index_faces(
                CollectionId=self.collection_id,
                Image={'Bytes': image_bytes},
                MaxFaces=1,
                QualityFilter="AUTO"
            )
            return response['FaceRecords'][0]['Face']['FaceId']
        except Exception as e:
            logger.error(f"Error indexing face: {str(e)}")
            return None

    def generate_image_hash(self, image_bytes):
        timestamp = str(time.time()).encode('utf-8')
        return hashlib.md5(image_bytes + timestamp).hexdigest()

    def upload_to_s3(self, file_bytes, file_name):
        try:
            self.s3_client.put_object(
                Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                Key=file_name,
                Body=file_bytes
            )
            return f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{file_name}"
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return None

    def get_best_frame(self, frames):
        """Select the best frame from the liveness sequence"""
        try:
            best_quality = 0
            best_frame = None

            for frame in frames:
                # Convert base64 to bytes if needed
                if isinstance(frame, str) and frame.startswith('data:image'):
                    image_bytes = base64.b64decode(frame.split(',')[1])
                else:
                    image_bytes = frame

                # Analyze face quality
                response = self.client.detect_faces(
                    Image={'Bytes': image_bytes},
                    Attributes=['QUALITY']
                )

                if response['FaceDetails']:
                    quality = response['FaceDetails'][0]['Quality']['Brightness'] + \
                             response['FaceDetails'][0]['Quality']['Sharpness']
                    
                    if quality > best_quality:
                        best_quality = quality
                        best_frame = image_bytes

            return best_frame
        except Exception as e:
            logger.error(f"Error selecting best frame: {str(e)}")
            return None
