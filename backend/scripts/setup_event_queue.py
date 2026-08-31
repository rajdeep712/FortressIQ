"""
Idempotently provision the S3 -> SQS event pipeline.

Steps:
  1. Create the (Standard) SQS queue S3 events will land on.
  2. Attach a queue policy allowing S3 to send messages for the bucket.
  3. Set the bucket's notification configuration for s3:ObjectCreated:*.

NOTE: S3 event notifications do NOT support FIFO queues, so a Standard
queue is used; consumers deduplicate (replace-by-doc + COMPLETED skip).

Usage (from backend/):
    .venv/Scripts/python.exe scripts/setup_event_queue.py

Outputs the AWS_SQS_QUEUE_URL value to paste into .env.
"""

import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402


def main() -> None:

    sqs = boto3.client(
        "sqs",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    s3 = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    bucket = settings.s3_bucket_name
    queue_name = f"{bucket}-doc-events"

    # ----------------------------------------------------------
    # 1. Queue
    # ----------------------------------------------------------

    try:
        created = sqs.create_queue(
            QueueName=queue_name,
            Attributes={"VisibilityTimeout": "60"},
        )
        queue_url = created["QueueUrl"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "QueueAlreadyExists":
            raise
        queue_url = sqs.get_queue_url(
            QueueName=queue_name
        )["QueueUrl"]

    queue_arn = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["QueueArn"],
    )["Attributes"]["QueueArn"]

    print(f"[1/3] Queue ready: {queue_url}")

    # ----------------------------------------------------------
    # 2. Queue policy (allow S3 to send to the queue)
    # ----------------------------------------------------------

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowS3ToSendToSQS",
                "Effect": "Allow",
                "Principal": {
                    "Service": "s3.amazonaws.com",
                },
                "Action": "sqs:SendMessage",
                "Resource": queue_arn,
                "Condition": {
                    "ArnEquals": {
                        "aws:SourceArn": f"arn:aws:s3:::{bucket}",
                    }
                },
            }
        ],
    }

    sqs.set_queue_attributes(
        QueueUrl=queue_url,
        Attributes={"Policy": json.dumps(policy)},
    )

    print("[2/3] Queue policy attached")

    # ----------------------------------------------------------
    # 3. Bucket notification configuration (merged, additive)
    # ----------------------------------------------------------

    existing = s3.get_bucket_notification_configuration(
        Bucket=bucket
    )

    queue_configs = [
        config
        for config in (
            existing.get("QueueConfigurations") or []
        )
        if config.get("Id") != "doc-upload-events"
    ]
    queue_configs.append(
        {
            "Id": "doc-upload-events",
            "QueueArn": queue_arn,
            "Events": ["s3:ObjectCreated:*"],
        }
    )

    merged = {
        key: value
        for key, value in existing.items()
        if key
        not in (
            "ResponseMetadata",
            "QueueConfigurations",
        )
    }
    merged["QueueConfigurations"] = queue_configs

    s3.put_bucket_notification_configuration(
        Bucket=bucket,
        NotificationConfiguration=merged,
    )

    print("[3/3] Bucket notification configured (s3:ObjectCreated:*)")
    print()
    print("Add to backend/.env:")
    print(f"AWS_SQS_QUEUE_URL={queue_url}")


if __name__ == "__main__":
    main()