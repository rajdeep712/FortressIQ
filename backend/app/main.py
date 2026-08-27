from contextlib import asynccontextmanager   ## manages startup/shutdown behavior.

from fastapi import FastAPI

from app.api.documents import router as documents_router
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


app.include_router(documents_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
