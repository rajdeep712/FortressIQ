import asyncio
import json
import logging
import re

import boto3
from arq import cron
from arq.connections import RedisSettings
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal, init_db
from app.core.tracing import (
    flush,
    maybe_span,
    set_span_attributes,
    setup_tracing,
)
from app.ingestion.embedding import EmbeddingService
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

# Document ids are always `doc_` + 32 lowercase hex chars
# (see app/services/document_service.py generate_doc_id).
DOC_ID_PATTERN = re.compile(r"^doc_[0-9a-f]{32}$")


def _configure_logging() -> None:
    """Route app.* loggers (ingestion, embedding, worker, ...) to
    stdout at INFO so pipeline activity is visible in the worker
    console regardless of arq's own logging setup."""
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    if not any(
        isinstance(handler, logging.StreamHandler)
        for handler in app_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s "
                "%(name)s: %(message)s"
            )
        )
        app_logger.addHandler(handler)


_configure_logging()


def parse_doc_id_from_key(
    s3_key: str,
) -> tuple[str, str] | None:
    """Return (user_id, doc_id) for an S3 key of the form
    `{user_id}/{doc_id}_{safe_filename}`, else None."""
    parts = s3_key.split("/", 1)
    if len(parts) != 2:
        return None
    user_id, tail = parts
    doc_id = "_".join(tail.split("_")[:2])
    if "/" in doc_id or not DOC_ID_PATTERN.match(doc_id):
        return None
    return user_id, doc_id


def _build_sqs_client():
    return boto3.client(
        "sqs",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )


def _is_completed(doc_id: str) -> bool:
    db: Session = SessionLocal()
    try:
        document = DocumentRepository(db).get_by_id(doc_id)
        return (
            document is not None
            and document.status == "COMPLETED"
        )
    finally:
        db.close()


async def startup(ctx):
    # The worker must not depend on the API having booted to create
    # the SQLite schema (init_db is idempotent).
    init_db()
    setup_tracing("worker")
    ctx["sqs"] = _build_sqs_client()
    ctx["ingestion"] = IngestionService()
    ctx["embedder"] = EmbeddingService()
    logger.info("ARQ worker started")


async def shutdown(ctx):
    flush()
    logger.info("ARQ worker shutting down")


async def _handle_s3_event(
    ctx,
    sqs,
    queue_url: str,
    message: dict,
) -> bool:
    """Enqueue ingestion jobs for an S3 event message.

    Returns True when the message should be deleted from SQS
    (handled or irrelevant), False when it must be left for
    redelivery (enqueueing failed)."""
    body = json.loads(message["Body"])

    for record in body.get("Records") or []:
        event_name = record.get("eventName", "")
        if not event_name.startswith("ObjectCreated:"):
            continue
        s3 = record.get("s3") or {}
        key = (s3.get("object") or {}).get("key")
        if not key:
            continue

        parsed = parse_doc_id_from_key(key)
        if parsed is None:
            logger.warning(
                "S3 event key not recognised: %s",
                key,
            )
            continue

        user_id, doc_id = parsed

        if _is_completed(doc_id):
            logger.info("Doc %s already COMPLETED; dropping event", doc_id)
            continue

        await ctx["redis"].enqueue_job(
            "ingest_document",
            doc_id,
            user_id,
        )

    return True


async def poll_s3_events(ctx):
    queue_url = settings.aws_sqs_queue_url

    if not queue_url:
        logger.warning(
            "AWS_SQS_QUEUE_URL not configured; skipping poll."
        )
        return

    sqs = ctx["sqs"]

    response = await asyncio.to_thread(
        sqs.receive_message,
        QueueUrl=queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=1,
    )

    messages = response.get("Messages", [])

    with maybe_span(
        "poll_s3_events",
        kind="CHAIN",
        queue=queue_url,
        message_count=len(messages),
    ):
        for message in messages:

            delete_message = False

            try:
                delete_message = await _handle_s3_event(
                    ctx=ctx,
                    sqs=sqs,
                    queue_url=queue_url,
                    message=message,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to process SQS message; leaving it for redelivery."
                )

            if delete_message:
                await asyncio.to_thread(
                    sqs.delete_message,
                    QueueUrl=queue_url,
                    ReceiptHandle=message["ReceiptHandle"],
                )


async def ingest_document(
    ctx,
    doc_id: str,
    user_id: str,
) -> dict:
    with maybe_span("ingest_document", kind="CHAIN", doc_id=doc_id, user_id=user_id):
        result = await asyncio.to_thread(
            ctx["ingestion"].ingest_document,
            doc_id,
            user_id,
        )
    return result


async def embed_chat_message(
    ctx,
    message_id: str,
    content: str,
) -> dict:
    """Background job: embed a chat message and persist its vector.

    Runs off the request hot path so history similarity search reads
    precomputed embeddings instead of embedding on each query. Falls
    back gracefully if the embedding backend is unavailable.
    """
    if not content.strip():
        return {"message_id": message_id, "embedded": False}

    def _work() -> dict:
        db = SessionLocal()
        try:
            row = ConversationRepository(db).get_message(message_id)
            if row is None:
                return {"message_id": message_id, "embedded": False}
            with maybe_span(
                "embed_chat_message",
                kind="EMBEDDING",
                message_id=message_id,
                chat_id=getattr(row, "chat_id", "") or "",
            ):
                vectors = ctx["embedder"].embed([content])
                if not vectors:
                    return {"message_id": message_id, "embedded": False}
                ok = ConversationRepository(db).update_embedding(
                    message_id, vectors[0]
                )
                set_span_attributes(
                    provider=ctx["embedder"].last_provider or "",
                    model=ctx["embedder"].last_model or "",
                    embedded=ok,
                )
            return {
                "message_id": message_id,
                "embedded": ok,
                "provider": ctx["embedder"].last_provider,
            }
        finally:
            db.close()

    result = await asyncio.to_thread(_work)
    logger.info("Embedded chat message %s (embedded=%s)", message_id, result.get("embedded"))
    return result


class WorkerSettings:
    functions = [ingest_document, poll_s3_events, embed_chat_message]
    cron_jobs = [
        cron(
            poll_s3_events,
            run_at_startup=True,
            unique=True,
            second=set(
                range(
                    0,
                    60,
                    settings.worker_sqs_poll_seconds,
                )
            ),
        )
    ]
    max_tries = settings.worker_max_tries
    job_timeout = settings.worker_job_timeout_seconds
    # arq 0.28 forwards a fixed set of redis options from RedisSettings;
    # Upstash/cluster Redis is aggressive about closing idle sockets, so
    # enable retry-on-timeout and retry short-lived connection errors on
    # COMMANDS (finish_job, enqueue_job, ...) rather than dying mid-flight.
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    redis_settings.retry_on_timeout = True
    redis_settings.retry_on_error = [RedisConnectionError]
    redis_settings.conn_retries = settings.worker_max_tries
    on_startup = startup
    on_shutdown = shutdown