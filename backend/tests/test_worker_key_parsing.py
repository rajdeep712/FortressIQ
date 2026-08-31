from app.ingestion.worker import (
    parse_doc_id_from_key,
    WorkerSettings,
)


def test_parses_standard_upload_key():
    key = (
        "user_mock_001/"
        "doc_0123456789abcdef0123456789abcdef_report_final.pdf"
    )
    assert parse_doc_id_from_key(key) == (
        "user_mock_001",
        "doc_0123456789abcdef0123456789abcdef",
    )


def test_rejects_key_without_user_folder():
    assert (
        parse_doc_id_from_key(
            "doc_0123456789abcdef0123456789abcdef_x.pdf"
        )
        is None
    )


def test_rejects_non_doc_id():
    assert (
        parse_doc_id_from_key(
            "user1/not_a_real_id_x.pdf"
        )
        is None
    )


def test_rejects_extra_path_depth():
    assert (
        parse_doc_id_from_key(
            "user1/doc_abc123/sub/x.pdf"
        )
        is None
    )


def test_worker_settings_declared_jobs():
    function_names = {
        func.__name__
        for func in WorkerSettings.functions
    }
    assert {"ingest_document", "poll_s3_events"} <= set(
        function_names
    )
    assert len(WorkerSettings.cron_jobs) == 1
    assert WorkerSettings.max_tries >= 1