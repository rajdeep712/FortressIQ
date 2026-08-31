from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Production RAG Backend"
    environment: str = "development"

    # Auth
    jwt_secret_key: str
    access_token_expire_minutes: int = 1440
    verification_token_expire_hours: int = 24
    resend_api_key: str = ""
    resend_from_email: str = ""
    google_client_id: str = ""
    frontend_url: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000"

    # Mock authentication for now
    # Upload limits
    max_file_size_mb: int = 50

    # AWS / S3
    aws_region: str = "us-east-1"
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_bucket_name: str
    kms_key_arn: str

    # OpenDocumentLoader API
    odl_api_endpoint: str = ""
    odl_timeout_seconds: int = 300

    # Docling API
    docling_api_endpoint: str = ""
    docling_api_key: str = ""
    docling_timeout_seconds: int = 600

    # LibreOffice (DOCX -> PDF)
    libreoffice_path: str = "soffice"

    # Embeddings (local sentence-transformers; no rate limits)
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    # Load the embedding model purely from the local HF cache
    # (no hub pings/verification). Requires a one-time download.
    hf_hub_offline: bool = True

    # Chunking
    chunk_child_max_chars: int = 2000
    chunk_parent_soft_max_chars: int = 7500
    chunk_parent_max_chars: int = 20000

    # Database
    database_url: str = "sqlite:///./documents.db"

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = ""
    qdrant_timeout_seconds: int = 120
    qdrant_upsert_batch_size: int = 100

    # Redis / arq worker
    redis_url: str = ""

    # SQS (S3 event notification destination)
    aws_sqs_queue_url: str = ""

    # Ingestion worker behaviour
    worker_max_tries: int = 5
    worker_job_timeout_seconds: int = 3600
    worker_sqs_poll_seconds: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()