from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Production RAG Backend"
    environment: str = "development"

    # Auth
    jwt_secret_key: str
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 30
    verification_token_expire_hours: int = 24
    password_reset_token_expire_minutes: int = 30
    resend_api_key: str = ""
    resend_from_email: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = ""
    frontend_url: str = "http://localhost:3000"
    cors_origins: str = "http://localhost:3000"
    # Fallback avatar for accounts without their own (e.g. email/password).
    default_avatar_url: str = (
        "https://www.gravatar.com/avatar/"
        "00000000000000000000000000000000?d=mp&f=y"
    )

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

    # Embeddings. Primary provider is Gemini (gemini-embedding-2) with strict
    # free-tier rate limits; sentence-transformers (below) is the fallback.
    # |gemini|  |sentence-transformer|
    embedding_provider: str = "gemini"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    # Gemini Embedding 2 (free tier). Caps are hard: the client throttles
    # (sleeps) so these are never exceeded, then backs off on 429.
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-2"
    # Output dimension of the embedding vectors. Must equal
    # embedding_dimension so the local fallback stays dimension-compatible.
    # gemini-embedding-2 supports 128-3072 and auto-normalizes truncated dims.
    gemini_output_dimensionality: int = 768
    # Per-request batch size when embedding multiple texts.
    gemini_embedding_batch_size: int = 16
    # Free-tier caps (never surpassed; wait until budget frees up).
    gemini_rate_rpm: int = 100
    gemini_rate_tpm: int = 30000
    gemini_rate_rpd: int = 1000
    # Daily usage is persisted here so worker restarts don't reset RPD.
    gemini_rate_state_file: str = ".cache/gemini_usage.json"
    # 429 backoff: wait Retry-After (or 60s) then double up to the max,
    # giving up after max_attempts consecutive 429s.
    gemini_retry_base_seconds: int = 60
    gemini_retry_max_seconds: int = 600
    gemini_retry_max_attempts: int = 4
    # After a Gemini outage triggers a fallback, keep using fallback for this
    # long before probing Gemini again (seconds).
    gemini_fallback_cooldown_seconds: int = 300

    # Load the embedding model purely from the local HF cache
    # (no hub pings/verification). Requires a one-time download.
    hf_hub_offline: bool = True

    # Chunking
    chunk_child_max_chars: int = 2000
    chunk_parent_soft_max_chars: int = 7500
    chunk_parent_max_chars: int = 20000

    # Retrieval: conversation context search + sufficiency routing.
    # These are tunable knobs; validate against a labelled dataset before
    # relying on any single band threshold.
    conversation_search_window: int = 40
    conversation_search_top_k: int = 5
    conversation_min_hits: int = 1
    conversation_overlap_threshold: float = 0.6
    # Dependency bands: >= conversation_only_threshold -> conversation-only
    # (skip Qdrant entirely); >= hybrid_threshold -> use both conversation
    # and Qdrant (context fusion + reranking); else -> Qdrant only.
    conversation_only_threshold: float = 0.8
    hybrid_threshold: float = 0.55
    conversation_reference_boost: float = 0.15

    # Retrieval: document (Qdrant) hybrid dense + BM25-sparse search + rerank.
    # Sparse vectors come from fastembed's SparseTextEmbedding ("Qdrant/bm25").
    sparse_model_name: str = "Qdrant/bm25"
    # FlashRank cross-encoder used to rerank the fused hybrid candidates.
    rerank_model_name: str = "ms-marco-MiniLM-L-12-v2"
    # Local dir where FlashRank downloads/caches its ONNX model. Default
    # "/tmp" is unusable on Windows, so default to a project-relative dir.
    rerank_cache_dir: str = ".cache/flashrank"
    # How many candidates to pull per modality (dense + sparse) before fusing.
    retrieval_prefetch_dense: int = 20
    retrieval_prefetch_sparse: int = 20
    # Fusion strategy for combining dense + sparse: rrf | dbsf.
    retrieval_fusion: str = "rrf"
    # When True, retrieval fails if the rerank model cannot be loaded; when
    # False it gracefully falls back to the fusion order.
    rerank_required: bool = False

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