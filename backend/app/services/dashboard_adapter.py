"""ARQ dashboard adapter with Upstash-friendly metrics reads.

Worq's upstream ``ARQAdapter.get_metrics`` issues four sequential Redis GETs
for every minute bucket in the requested window (1440 buckets for the default
24h window = 5760 round trips). Against a cloud Redis like Upstash each round
trip is tens of milliseconds, so the dashboard overview page gets stuck for
minutes. This subclass reads all bucket keys with chunked MGETs (one round
trip per ~500 keys) and aggregates the exact same way.

Private helpers/constants are imported deliberately — the Worq keys this
reads are internal to worq 0.5.x, so this is pinned to the installed version.
"""

from __future__ import annotations

import time

from worq.adapters.arq import ARQAdapter, _METRICS_PREFIX, _ms_to_dt
from worq.types import MetricsBucket, MetricsHistory

_MGET_KEYS_PER_CALL = 500  # ~22KB/command, well under Upstash command limits


class ArqDashboardAdapter(ARQAdapter):
    """ARQAdapter with chunked-MGET metrics aggregation."""

    async def get_metrics(self, hours: int = 24) -> MetricsHistory:
        r = await self._redis()
        now_s = int(time.time())
        start_s = now_s - (hours * 3600)
        start_minute = (start_s // 60) * 60
        now_minute = (now_s // 60) * 60
        minutes = range(start_minute, now_minute + 1, 60)

        keys: list[str] = []
        for minute in minutes:
            keys.extend(
                (
                    f"{_METRICS_PREFIX}:processed:{minute}",
                    f"{_METRICS_PREFIX}:failed:{minute}",
                    f"{_METRICS_PREFIX}:duration:{minute}",
                    f"{_METRICS_PREFIX}:count:{minute}",
                )
            )

        values: list[bytes | None] = []
        for start in range(0, len(keys), _MGET_KEYS_PER_CALL):
            values.extend(await r.mget(keys[start : start + _MGET_KEYS_PER_CALL]))

        buckets: list[MetricsBucket] = []
        total_processed = 0
        total_failed = 0
        total_duration = 0.0
        total_duration_count = 0

        for i, minute in enumerate(minutes):
            base = i * 4
            processed_raw = values[base]
            failed_raw = values[base + 1]
            duration_raw = values[base + 2]
            count_raw = values[base + 3]

            processed = int(processed_raw) if processed_raw else 0
            failed = int(failed_raw) if failed_raw else 0
            duration_sum = float(duration_raw) if duration_raw else 0.0
            duration_count = int(count_raw) if count_raw else 0
            avg_dur = duration_sum / duration_count if duration_count > 0 else 0.0

            if processed > 0 or failed > 0:
                buckets.append(
                    MetricsBucket(
                        timestamp=_ms_to_dt(minute * 1000),
                        processed=processed,
                        failed=failed,
                        avg_duration_ms=avg_dur,
                    )
                )
                total_processed += processed
                total_failed += failed
                total_duration += duration_sum
                total_duration_count += duration_count

        total_jobs = total_processed + total_failed
        failure_rate = (total_failed / total_jobs * 100) if total_jobs > 0 else 0.0
        avg_duration = total_duration / total_duration_count if total_duration_count > 0 else 0.0

        return MetricsHistory(
            buckets=buckets,
            total_processed=total_processed,
            total_failed=total_failed,
            failure_rate=round(failure_rate, 2),
            avg_duration_ms=round(avg_duration, 2),
        )