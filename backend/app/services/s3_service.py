from pathlib import Path
import logging

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


class S3Service:
    def __init__(self):
        self.client = boto3.client(   ## AWS S3 Client.
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        self.bucket = settings.s3_bucket_name

    def upload_file(
        self,
        file_path: Path,   ## path where the file is temporarily stored in disk.
        s3_key: str,   ## For storing key in S3.
        content_type: str,   ## Mime Type
    ) -> None:
        extra_args = {
            "ContentType": content_type,
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": settings.kms_key_arn,
        }

        try:
            self.client.upload_file(
                str(file_path),
                self.bucket,
                s3_key,
                ExtraArgs=extra_args,
            )
        except ClientError as exc:  ## For handling errors during upload.
            logger.error(
                "S3 upload failed for key=%s: %s",
                s3_key,
                exc,
            )
            raise
