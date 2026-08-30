from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Production RAG Backend"
    environment: str = "development"

    # Mock authentication for now
    mock_user_id: str = "user_mock_001"

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

    # Gemini Embeddings
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_output_dimension: int = 768
    gemini_embedding_task_type: str = "RETRIEVAL_DOCUMENT"
    gemini_embedding_batch_size: int = 100

    # Chunking
    chunk_child_max_chars: int = 2000
    chunk_parent_soft_max_chars: int = 7500
    chunk_parent_max_chars: int = 20000

    # Database
    database_url: str = "sqlite:///./documents.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()