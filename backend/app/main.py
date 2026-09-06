import asyncio
import sys
from contextlib import asynccontextmanager   ## manages startup/shutdown behavior.

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.tracing import flush, setup_tracing

# Note: setup_tracing is called in the lifespan (startup), not at import time,
# so that tests importing this module don't accidentally create a real Phoenix
# provider and export spans to the cloud.


# Psycopg3 async mode is incompatible with the default Windows event loop
# (Proactor). The LangGraph Postgres checkpointer runs on psycopg async, so on
# Windows we must select a SelectorEventLoop before the app starts serving.
if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()
    setup_tracing("api")
    yield
    flush()


app = FastAPI(
    title="Production RAG Backend",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Validation error handler: turn FastAPI/Pydantic's structured 422 into a
# readable 400 so request-body problems (e.g. a missing required parameter)
# surface as a plain-string `detail` that the frontend can display verbatim.
# ---------------------------------------------------------------------------


def _field_name(loc) -> str:
    # loc is a tuple like ("body", "email") or ("query", "code"); drop
    # integer indexes so list items don't clutter the message.
    parts = [str(part) for part in loc if not isinstance(part, int)]
    if not parts:
        return "request"
    return ".".join(parts[-2:])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()

    missing = [e for e in errors if e.get("type") == "missing"]
    if missing:
        fields = sorted(
            {_field_name(e.get("loc", ())) for e in missing}
        )
        named = ", ".join(fields)
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"Missing required parameter: {named}"
            },
        )

    if errors:
        first = errors[0]
        field = _field_name(first.get("loc", ()))
        message = str(first.get("msg", "Invalid value.")).split(".")[0]
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"Invalid value for {field}: {message}."
            },
        )

    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request."},
    )

origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)

# ---------------------------------------------------------------------------
# ARQ job-queue monitoring dashboard (Worq). Read-only, pure Redis reads, so
# it never touches load on the RAG path. Only mounted when MONITOR_ENABLED is
# true. Job args/results (chat content, user/doc ids) are visible in the UI,
# so keep the API bound to localhost unless extra auth is added on /worq.
# ---------------------------------------------------------------------------
if settings.monitor_enabled:
    try:
        from worq import create_dashboard

        from app.services.dashboard_adapter import ArqDashboardAdapter

        app.mount(
            "/worq",
            create_dashboard(
                ArqDashboardAdapter(
                    redis_url=settings.redis_url,
                    cron_jobs=[
                        {
                            "name": "poll_s3_events",
                            "schedule": (
                                f"every {settings.worker_sqs_poll_seconds} seconds"
                            ),
                        }
                    ],
                ),
                title="RAG Worker Queues",
            ),
        )
    except Exception:  # pragma: no cover - fail-open, never break the app
        import logging

        logging.getLogger(__name__).warning(
            "Worq dashboard unavailable; skipping /worq mount",
            exc_info=True,
        )


@app.get("/health")
async def health_check():
    return {"status": "ok"}
