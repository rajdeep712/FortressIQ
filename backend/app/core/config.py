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

    # Database
    database_url: str = "sqlite:///./documents.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()