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
            response = self.client.create_face_liveness_session(
                ClientRequestToken=str(int(time.time()))
            )
            logger.info("Created face liveness session")
            return response['SessionId']
        except Exception as e:
            logger.error(f"Error creating liveness session: {str(e)}")
            raise

    def check_liveness_frames(self, session_id, frames):
        """Check liveness for a given set of frames"""
        try:
            processed_frames = []
            for frame in frames:
                # Convert base64 to bytes if needed
                if isinstance(frame, str):
                    if frame.startswith('data:image'):
                        image_bytes = base64.b64decode(frame.split(',')[1])
                    else:
                        image_bytes = base64.b64decode(frame)
                else:
                    image_bytes = frame

                # Update liveness session with each frame
                try:
                    self.client.update_face_liveness_session(
                        SessionId=session_id,
                        Image={'Bytes': image_bytes}
                    )
                    processed_frames.append(image_bytes)
                except Exception as frame_error:
                    logger.warning(f"Error processing frame: {str(frame_error)}")

            # Get final liveness session results
            result = self.client.get_face_liveness_session_results(
                SessionId=session_id
            )

            # Determine liveness based on confidence
            is_live = result.get('Confidence', 0) > 90
            
            return {
                'status': 'success',
                'isLive': is_live,
                'confidence': result.get('Confidence', 0),
                'sessionId': session_id
            }

        except Exception as e:
            logger.error(f"Error checking liveness frames: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'isLive': False,
                'confidence': 0
            }

    def process_liveness_frames(self, session_id, frames):
        """Process liveness detection frames"""
        try:
            processed_frames = []
            for frame in frames:
                # Convert base64 to bytes if needed
                if isinstance(frame, str):
                    if frame.startswith('data:image'):
                        image_bytes = base64.b64decode(frame.split(',')[1])
                    else:
                        image_bytes = base64.b64decode(frame)
                else:
                    image_bytes = frame

                # Update liveness session
                try:
                    self.client.update_face_liveness_session(
                        SessionId=session_id,
                        Image={'Bytes': image_bytes}
                    )
                    processed_frames.append(image_bytes)
                except Exception as frame_error:
                    logger.warning(f"Error processing frame: {str(frame_error)}")

            # Get final liveness session results
            result = self.client.get_face_liveness_session_results(
                SessionId=session_id
            )

            # Determine liveness
            is_live = result.get('Confidence', 0) > 90

            # If liveness is confirmed, get the best frame
            selfie_url = None
            if is_live:
                best_frame = self.get_best_frame(processed_frames)
                
                # Upload best frame to S3 if available
                if best_frame:
                    image_hash = self.generate_image_hash(best_frame)
                    file_name = f'liveness/{image_hash}.jpg'
                    selfie_url = self.upload_to_s3(best_frame, file_name)

            return {
                'status': 'success',
                'isLive': is_live,
                'confidence': result.get('Confidence', 0),
                'sessionId': session_id,
                'selfieUrl': selfie_url
            }

        except Exception as e:
            logger.error(f"Error processing liveness frames: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'isLive': False,
                'confidence': 0
            }

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
