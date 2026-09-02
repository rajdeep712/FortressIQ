from contextlib import asynccontextmanager   ## manages startup/shutdown behavior.

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()
    yield


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


@app.get("/health")
async def health_check():
    return {"status": "ok"}
