import boto3
import hashlib
import time
from django.conf import settings
import logging

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
        self._ensure_collection_exists()  # Check collection on initialization

    def _ensure_collection_exists(self):
        """Ensure collection exists, create if it doesn't"""
        try:
            # Try to describe the collection
            self.client.describe_collection(CollectionId=self.collection_id)
            logger.info(f"Collection {self.collection_id} exists")
        except self.client.exceptions.ResourceNotFoundException:
            # Collection doesn't exist, create it
            try:
                self.client.create_collection(CollectionId=self.collection_id)
                logger.info(f"Created collection {self.collection_id}")
            except Exception as e:
                logger.error(f"Error creating collection: {str(e)}")
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
