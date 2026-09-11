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

    # Embeddings: local SentenceTransformer (BAAI/bge-base-en-v1.5, 768-dim).
    # Remote providers (Gemini, OpenRouter) have been removed; all dense
    # embeddings run locally via sentence-transformers with batched encoding.
    embedding_provider: str = "sentence-transformer"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    # Load the embedding model purely from the local HF cache
    # (no hub pings/verification). Requires a one-time download.
    hf_hub_offline: bool = True

    # Chunking
    # Standardized (non-tabular) sizing is token-based:
    #   * a child chunk is at most `chunk_child_max_tokens` tokens
    #   * consecutive children within a parent overlap by 200-300 chars,
    #     aligned to sentence/element seams so the window is coherent
    #   * a parent chunk holds 4-8 children on average (`chunk_children_min`
    #     .. `chunk_children_max`) and never exceeds `chunk_parent_max_tokens`
    #   * chunk sizes are measured with tiktoken cl100k_base (see
    #     app/ingestion/chunker/tokenizer.py) with a ~4 chars/token fallback
    # Tabular formats (CSV/XLSX) keep the legacy character caps below.
    chunk_standard_windowing: bool = True
    chunk_child_max_tokens: int = 250
    chunk_parent_max_tokens: int = 1800
    chunk_overlap_min_chars: int = 200
    chunk_overlap_max_chars: int = 300
    chunk_children_min: int = 4
    chunk_children_max: int = 8
    # Legacy character caps (CSV/XLSX + explicit-char-override callers).
    chunk_child_max_chars: int = 2000
    chunk_parent_soft_max_chars: int = 7500
    chunk_parent_max_chars: int = 20000

    # PDF chunking: structure-driven (OpenDocumentLoader tree: major
    # items become parents, kids become 1:1 children) instead of the
    # window-packing SectionChunker.
    pdf_chunk_structured: bool = True
    # Adjacent PDF children shorter than this many chars are folded into
    # their neighbour (fragments, page-number stubs etc.).
    pdf_merge_tiny_child_chars: int = 60
    # Opt-in reading order for PDF siblings via (page, bbox y, bbox x).
    # Off by default: OpenDocumentLoader emits logical order, and bbox
    # order can be wrong on multi-column layouts.
    pdf_sort_siblings_by_bbox: bool = False

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

    # Chat / conversation memory
    conversation_window: int = 40
    conversation_top_k: int = 5

    # LLM generation (Groq-hosted, OpenAI-compatible)
    groq_api_key: str = ""
    groq_chat_model: str = "openai/gpt-oss-120b"
    groq_max_tokens: int = 4096
    groq_temperature: float = 0.1
    groq_request_timeout_seconds: int = 120

    # RRF fusion over conversation + document hits
    fusion_k: int = 60

    # Observability: Arize Phoenix (OpenTelemetry / OpenInference).
    # Enabled by default (per project decision). The whole telemetry path is
    # fail-open and runs on a background thread (BatchSpanProcessor), so it
    # never blocks or slows the RAG application; export errors are swallowed.
    phoenix_enabled: bool = True
    phoenix_project_name: str = "bring-any-doc-rag"
    # Phoenix Cloud OTLP traces endpoint. For a local Phoenix it would be
    # something like http://localhost:6006/v1/traces.
    phoenix_collector_endpoint: str = "https://app.phoenix.arize.com/v1/traces"
    # Phoenix Cloud auth: the exporter sends `authorization: Bearer <key>`.
    # Keep credentials in .env (PHOENIX_API_KEY).
    phoenix_api_key: str = ""
    # Legacy header mechanism, parsed into headers only when phoenix_api_key
    # is empty (e.g. "Authorization=<token>"). Left empty by default.
    phoenix_client_headers: str = ""
    # Mount the ARQ job-queue monitoring dashboard at /worq. Read-only by
    # default; job args (chat message content, user/doc ids) are visible, so
    # keep the API bound to localhost unless extra auth is added.
    monitor_enabled: bool = True
    # Probabilistic per-trace sample rate (1.0 = trace everything). Lower this
    # under load to drop traces probabilistically and reduce telemetry cost.
    phoenix_sample_rate: float = 1.0
    # BatchSpanProcessor safety knobs: short export timeout + bounded queue so
    # a slow/down Phoenix can never stall a request or balloon memory.
    phoenix_export_timeout_ms: int = 10000
    phoenix_max_queue_size: int = 2048
    phoenix_max_export_batch_size: int = 512

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()