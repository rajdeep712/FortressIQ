"""Postgres checkpointer for per-chat LangGraph memory.

Uses the *async* ``AsyncPostgresSaver`` (psycopg3 async).

On Windows the default asyncio event loop (``WindowsProactorEventLoopPolicy``)
is incompatible with psycopg async mode, so the application must select a
``SelectorEventLoop`` before running the graph (see ``app.main``).
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings


def _normalized_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


async def build_checkpointer(conn: Any | None = None) -> AsyncPostgresSaver:
    """Build (and initialize) an async PostgresSaver over the configured URL.

    Pass a pre-existing psycopg async connection for tests; otherwise open one
    from ``settings.database_url``. Caller is responsible for closing ``conn``.
    """
    if conn is None:
        import psycopg

        conn = await psycopg.AsyncConnection.connect(
            _normalized_url(settings.database_url),
            autocommit=True,
        )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    return checkpointer
