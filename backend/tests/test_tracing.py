"""Safety tests for the Phoenix tracing bootstrap.

These guard the core invariant that tracing must *never* slow down or break
the RAG application:

  * When tracing is disabled (the default in tests), every helper is a cheap
    no-op that never raises and never touches the event loop / network.
  * ``setup_tracing`` is idempotent and fail-open even with a bogus endpoint.
  * Instrumentation package imports used by ``_instrument`` resolve to real
    classes so wiring the API/worker is reliable.
"""

import pytest

from app.core import tracing


@pytest.fixture(autouse=True)
def _reset_tracing_state(monkeypatch):
    """Force tracing helpers into the disabled no-op path for each test."""
    monkeypatch.setattr(tracing, "_ENABLED", False)
    monkeypatch.setattr(tracing, "_PROVIDER", None)
    monkeypatch.setattr(tracing, "_TRACER", None)
    monkeypatch.setattr(tracing, "_setup_done", False)
    yield


def test_maybe_span_is_noop_when_disabled():
    with tracing.maybe_span("anything", kind="CHAIN", doc_id="x") as span:
        assert span is None
    # Should not raise even with odd attribute values / types.
    with tracing.maybe_span(
        "weird", kind=None, bytes=b"\x00", nested={"a": 1}
    ) as span:
        assert span is None


def test_turn_context_is_noop_when_disabled():
    with tracing.turn_context(chat_id="c", user_id="u"):
        pass
    # Nested / empty variants must be harmless too.
    with tracing.turn_context():
        with tracing.turn_context(chat_id=None, user_id=None):
            pass


def test_set_span_attributes_is_noop_when_disabled():
    tracing.set_span_attributes(doc_id="d", score=1.0, flag=True, none_value=None)
    tracing.set_span_attributes()


def test_flush_is_noop_when_disabled():
    tracing.flush()


def test_setup_tracing_disabled_returns_none(monkeypatch):
    class _Settings:
        phoenix_enabled = False
        phoenix_project_name = "p"
        phoenix_sample_rate = 1.0
        phoenix_collector_endpoint = "http://127.0.0.1:9999/v1/traces"
        phoenix_client_headers = ""
        phoenix_api_key = ""
        phoenix_export_timeout_ms = 10
        phoenix_max_queue_size = 8
        phoenix_max_export_batch_size = 4

    monkeypatch.setattr(tracing, "settings", _Settings())
    assert tracing.setup_tracing("api") is None
    assert tracing.tracing_enabled() is False


def test_setup_tracing_is_idempotent(monkeypatch):
    """Once setup has run, subsequent calls are a no-op returning the same
    provider — no re-instrumentation, no repeated setup work."""
    sentinel = object()
    monkeypatch.setattr(tracing, "_setup_done", True)
    monkeypatch.setattr(tracing, "_PROVIDER", sentinel)
    monkeypatch.setattr(tracing, "_TRACER", None)

    first = tracing.setup_tracing("api")
    second = tracing.setup_tracing("worker")
    assert first is sentinel is second


def test_setup_tracing_fails_open_on_bad_enable(monkeypatch):
    class _Settings:
        phoenix_enabled = True
        phoenix_project_name = "p"
        phoenix_sample_rate = 1.0
        phoenix_collector_endpoint = "http://127.0.0.1:9999/v1/traces"
        phoenix_client_headers = ""
        phoenix_api_key = ""
        phoenix_export_timeout_ms = 10
        phoenix_max_queue_size = 8
        phoenix_max_export_batch_size = 4

    monkeypatch.setattr(tracing, "settings", _Settings())
    monkeypatch.setattr(tracing, "_setup_done", False)

    # Make the exporter construction raise so we can confirm fail-open behavior.
    import phoenix.otel as phoenix_otel

    monkeypatch.setattr(
        phoenix_otel,
        "HTTPSpanExporter",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = tracing.setup_tracing("api")
    assert result is None
    assert tracing.tracing_enabled() is False


def test_build_exporter_headers_prefers_api_key():
    headers = tracing._build_exporter_headers(
        api_key="K123",
        client_headers="Authorization=Bearer LegacyValue",
    )
    assert headers == {"authorization": "Bearer K123"}


def test_build_exporter_headers_falls_back_to_legacy():
    headers = tracing._build_exporter_headers(
        api_key=None,
        client_headers="Authorization=Bearer abc;x-api-key=zz",
    )
    assert headers == {"authorization": "Bearer abc", "x-api-key": "zz"}


def test_build_exporter_headers_empty():
    assert tracing._build_exporter_headers() == {}


def test_instrumentor_imports_resolve():
    """The instrumentors referenced by ``_instrument`` must exist so the API
    and worker can wire them without failing the whole setup."""
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.instrumentation.qdrant import QdrantInstrumentor
    from opentelemetry.instrumentation.httpx import (
        HTTPXClientInstrumentor,
    )

    assert LangChainInstrumentor is not None
    assert QdrantInstrumentor is not None
    assert HTTPXClientInstrumentor is not None
