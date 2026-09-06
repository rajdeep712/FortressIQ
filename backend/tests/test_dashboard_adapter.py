import asyncio
import time

from app.services.dashboard_adapter import (
    _MGET_KEYS_PER_CALL,
    ArqDashboardAdapter,
)


class _FakeRedis:
    """Minimal async Redis double supporting only mget."""

    def __init__(self, data: dict[str, bytes]):
        self._data = data
        self.mget_calls = 0
        self.retrieved_keys: list[list[str]] = []

    async def mget(self, keys: list[str]):
        self.mget_calls += 1
        self.retrieved_keys.append(list(keys))
        return [self._data.get(k) for k in keys]


def _adapter(fake: _FakeRedis) -> ArqDashboardAdapter:
    adapter = ArqDashboardAdapter(redis_url="redis://unused")

    async def _redis():  # pragma: no cover - called by get_metrics
        return fake

    adapter._redis = _redis  # type: ignore[method-assign]
    return adapter


def test_metrics_returns_empty_history_with_no_traffic():
    adapter = _adapter(_FakeRedis({}))
    metrics = asyncio.run(adapter.get_metrics(hours=1))
    assert metrics.buckets == []
    assert metrics.total_processed == 0
    assert metrics.total_failed == 0
    assert metrics.failure_rate == 0.0
    assert metrics.avg_duration_ms == 0.0


def test_metrics_aggregates_present_buckets():
    minute = (int(time.time()) // 60) * 60
    fake = _FakeRedis(
        {
            f"worq:metrics:processed:{minute}": b"3",
            f"worq:metrics:failed:{minute}": b"1",
            f"worq:metrics:duration:{minute}": b"2.5",
            f"worq:metrics:count:{minute}": b"2",
        }
    )
    metrics = asyncio.run(_adapter(fake).get_metrics(hours=1))
    assert len(metrics.buckets) == 1
    bucket = metrics.buckets[0]
    assert bucket.processed == 3
    assert bucket.failed == 1
    assert bucket.avg_duration_ms == 1.25
    assert metrics.total_processed == 3
    assert metrics.total_failed == 1
    assert metrics.failure_rate == 25.0
    assert metrics.avg_duration_ms == 1.25


def test_metrics_reads_all_bucket_keys_with_few_mget_calls():
    fake = _FakeRedis({})
    metrics = asyncio.run(_adapter(fake).get_metrics(hours=24))
    total_keys = sum(len(c) for c in fake.retrieved_keys)
    now_s = int(time.time())
    minutes = (now_s // 60) - ((now_s - 1440 * 60) // 60) + 1
    assert total_keys == minutes * 4
    assert fake.mget_calls * _MGET_KEYS_PER_CALL >= total_keys
    assert metrics.buckets == []