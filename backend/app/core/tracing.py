"""Arize Phoenix / OpenTelemetry tracing bootstrap.

Design goals (unchanged by anything in the RAG hot path):
  * Fail-open: if Phoenix is disabled or down (or the exporters are missing),
    *nothing* here raises or blocks the application.
  * Background export only: we use a bounded ``BatchSpanProcessor`` wrapped in
    a short export timeout, never a synchronous exporter, so a request/worker
    thread never waits on the network.
  * Lightweight attributes: callers attach small ids/counts/elapsed values and
    truncated snippets only (never full text or embeddings).
  * Zero overhead when disabled: every span site goes through ``maybe_span`` /
    ``turn_context`` which resolve to cheap no-ops when tracing is off.

There are two processes: the API (FastAPI, runs retrieval) and the arq worker
(runs all ingestion + background jobs). Both call ``setup_tracing`` so all
spans export to the same Phoenix Cloud collector (same project).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)

_ENABLED: bool = False
_PROVIDER = None
_TRACER = None
_setup_lock = threading.Lock()
_setup_done = False

# OpenInference span-kind attribute Phoenix uses to render spans nicely.
_OPENINFERENCE_SPAN_KIND = "openinference.span.kind"


def _parse_headers(raw: str) -> dict[str, str]:
    """Parse ``"api_key=xxx;foo=bar"`` into an OTel headers dict."""
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        headers[key.strip()] = value.strip()
    return headers


def _build_exporter_headers(
    api_key: str | None = None,
    client_headers: str = "",
) -> dict[str, str]:
    """Combine Phoenix auth into the OTLP exporter header dict.

    ``api_key`` wins (SDK-style ``authorization: Bearer <key>``); legacy
    ``client_headers`` are only parsed when no ``api_key`` is provided.
    """
    headers: dict[str, str] = {}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    elif client_headers:
        headers.update(
            {
                key.lower(): value
                for key, value in _parse_headers(client_headers).items()
            }
        )
    return headers


def _instrument(provider) -> None:
    """Auto-instrument supported libraries. A missing instrumentor never
    disables the rest — each is wrapped in its own try/except."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception:  # noqa: BLE001
        logger.warning("Phoenix: LangChain instrumentor unavailable", exc_info=True)

    try:
        from opentelemetry.instrumentation.qdrant import QdrantInstrumentor

        QdrantInstrumentor().instrument(tracer_provider=provider)
    except Exception:  # noqa: BLE001
        logger.warning("Phoenix: Qdrant instrumentor unavailable", exc_info=True)

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    except Exception:  # noqa: BLE001
        logger.warning("Phoenix: httpx instrumentor unavailable", exc_info=True)


def setup_tracing(service_name: str = "api"):
    """Register the global tracer provider + instrumentors (once).

    Returns the provider (or None if disabled/failed). Safe to call from both
    process entrypoints; idempotent via a lock. Never raises.
    """
    global _ENABLED, _PROVIDER, _TRACER, _setup_done
    if _setup_done:
        return _PROVIDER

    with _setup_lock:
        if _setup_done:
            return _PROVIDER
        _setup_done = True

        if not settings.phoenix_enabled:
            logger.info("Phoenix tracing disabled (phoenix_enabled=False)")
            return None

        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
            from phoenix.otel import HTTPSpanExporter, PROJECT_NAME, TracerProvider

            resource = Resource.create(
                {
                    PROJECT_NAME: settings.phoenix_project_name,
                    "service.name": f"{settings.phoenix_project_name}-{service_name}",
                }
            )

            sampler = TraceIdRatioBased(settings.phoenix_sample_rate)

            exporter = HTTPSpanExporter(
                endpoint=settings.phoenix_collector_endpoint,
                headers=_build_exporter_headers(
                    api_key=settings.phoenix_api_key,
                    client_headers=settings.phoenix_client_headers,
                )
                or None,
                timeout=settings.phoenix_export_timeout_ms / 1000.0,
            )
            # Bounded batch export (async background thread): a full queue drops
            # spans rather than blocking the request, and each export has a short
            # timeout so the worker/API never waits on the network. This is the
            # guarantee that keeps tracing off the RAG hot path.
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=settings.phoenix_max_queue_size,
                max_export_batch_size=settings.phoenix_max_export_batch_size,
                schedule_delay_millis=5000,
                export_timeout_millis=float(settings.phoenix_export_timeout_ms),
            )

            provider = TracerProvider(resource=resource, sampler=sampler)
            provider.add_span_processor(processor)
            otel_trace.set_tracer_provider(provider)

            _instrument(provider)

            _ENABLED = True
            _PROVIDER = provider
            _TRACER = otel_trace.get_tracer("rag")
            logger.info(
                "Phoenix tracing enabled (service=%s, project=%s, sample=%.2f)",
                service_name,
                settings.phoenix_project_name,
                settings.phoenix_sample_rate,
            )
            return provider
        except Exception:  # noqa: BLE001 - never let tracing break the app
            _ENABLED = False
            _PROVIDER = None
            logger.warning(
                "Phoenix tracing disabled (setup failed):",
                exc_info=True,
            )
            return None


def tracing_enabled() -> bool:
    return _ENABLED


def get_tracer():
    """Return the module tracer (no-op safe when disabled)."""
    if _TRACER is not None:
        return _TRACER
    import opentelemetry.trace as otel_trace

    return otel_trace.get_tracer("rag")


@contextmanager
def maybe_span(name: str, kind: str | None = None, **attributes):
    """Like ``tracer.start_as_current_span`` but a cheap no-op when disabled.

    ``kind`` is an OpenInference span kind (CHAIN, TOOL, RETRIEVER, RERANKER,
    EMBEDDING, ...) used by Phoenix to render the span. Attributes are applied
    best-effort; a failure to set one is never allowed to raise.
    """
    if not _ENABLED:
        yield None
        return
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if kind:
            try:
                span.set_attribute(_OPENINFERENCE_SPAN_KIND, kind)
            except Exception:  # noqa: BLE001
                pass
        for key, value in attributes.items():
            try:
                span.set_attribute(key, "" if value is None else value)
            except Exception:  # noqa: BLE001
                pass
        yield span


@contextmanager
def turn_context(chat_id: str | None = None, user_id: str | None = None):
    """Tag every span in a block with session + user context (no-op when off)."""
    if not _ENABLED:
        yield
        return
    try:
        from phoenix.otel import using_session, using_user
    except Exception:  # noqa: BLE001
        yield
        return
    with using_session(chat_id or ""), using_user(user_id or ""):
        yield


def set_span_attributes(**attributes) -> None:
    """Attach metadata to the currently-active span (best-effort, no-op off)."""
    if not _ENABLED:
        return
    try:
        import opentelemetry.trace as otel_trace

        span = otel_trace.get_current_span()
        for key, value in attributes.items():
            if value is None:
                continue
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001
        pass


def flush() -> None:
    """Force pending spans to export. Only safe off the chat request path
    (worker shutdown / after ingestion jobs). Never raises."""
    if not _ENABLED or _PROVIDER is None:
        return
    try:
        _PROVIDER.force_flush()
    except Exception:  # noqa: BLE001
        logger.warning("Phoenix flush failed", exc_info=True)
