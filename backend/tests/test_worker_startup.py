import asyncio

import app.ingestion.worker as worker_module


def test_worker_startup_initializes_db_schema(monkeypatch):
    init_calls = []

    monkeypatch.setattr(
        worker_module,
        "init_db",
        lambda: init_calls.append(True),
    )
    monkeypatch.setattr(
        worker_module,
        "_build_sqs_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        worker_module,
        "IngestionService",
        lambda: object(),
    )

    asyncio.run(worker_module.startup({"redis": None}))

    assert init_calls == [True]