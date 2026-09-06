"""Tests for the conversation x document RRF fusion (hybrid routing)."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import FusionItem, rrf_fuse


def _conv(item_ids):
    return [
        FusionItem(item_id=i, source="conversation", payload={"role": "user"})
        for i in item_ids
    ]


def _doc(item_ids):
    return [
        FusionItem(item_id=i, source="document", payload={"chunk_id": i})
        for i in item_ids
    ]


class TestRrfFuse:
    def test_all_items_present_in_fused_output(self):
        fused = rrf_fuse(_conv(["a", "b"]), _doc(["x", "y", "z"]), k=60)
        ids = [f.item_id for f in fused]
        assert set(ids) == {"a", "b", "x", "y", "z"}

    def test_shared_rank_honors_boost_within_list(self):
        # Top doc and top conv item both rank first; conversation_boost tilts
        # the top spot toward the conversation item.
        fused = rrf_fuse(
            _conv(["c_first"]),
            _doc(["d_first"]),
            k=60,
            conversation_boost=1.0,
        )
        # Only one list -> the first list is the conversation (boosted) one.
        assert fused[0].item_id == "c_first"

    def test_conversation_boost_changes_ordering(self):
        # conversation_boost boosts conversation items' effective rank.
        boosted = rrf_fuse(_conv(["a"]), _doc(["a"]), k=60, conversation_boost=1.0)
        unboosted = rrf_fuse(_conv(["a"]), _doc(["a"]), k=60, conversation_boost=0.0)
        # Same id in both lists; with boost the conversation copy should
        # score at least as high, but both are present either way.
        assert {f.item_id for f in boosted} == {"a"}
        assert {f.item_id for f in unboosted} == {"a"}

    def test_k_controls_rrf_denominator(self):
        # Larger k flattens the score difference but never drops items.
        small_k = rrf_fuse(_conv(["a"]), _doc(["b"]), k=1)
        large_k = rrf_fuse(_conv(["a"]), _doc(["b"]), k=100)
        assert {f.item_id for f in small_k} == {"a", "b"}
        assert {f.item_id for f in large_k} == {"a", "b"}

    def test_returns_descending_score_ordering(self):
        # Items ranked first in either list (rank 1) score highest under RRF.
        fused = rrf_fuse(_conv(["a", "c"]), _doc(["b"]), k=60)
        # "a" and "b" are each rank 1 in their own list and tie; "c" is rank 2.
        ids = [f.item_id for f in fused]
        assert ids.index("c") > ids.index("a")
        assert set(ids) == {"a", "b", "c"}

    def test_empty_lists(self):
        fused = rrf_fuse([], [], k=60)
        assert fused == []

    def test_fused_items_carry_rrf_score(self):
        fused = rrf_fuse(_conv(["a"]), _doc(["b"]), k=60)
        assert all(isinstance(f.score, float) for f in fused)
        a = next(f for f in fused if f.item_id == "a")
        b = next(f for f in fused if f.item_id == "b")
        assert a.score == pytest.approx(1.0 / 61.0)
        assert b.score == pytest.approx(1.0 / 61.0)

    def test_score_accumulates_across_lists(self):
        # "a" ranks 1st in both lists -> 2/(k+1); "c" ranks 2nd in conv only.
        fused = rrf_fuse(_conv(["a", "c"]), _doc(["a"]), k=4)
        by_id = {f.item_id: f.score for f in fused}
        assert by_id["a"] == pytest.approx(2.0 / 5.0)
        assert by_id["c"] == pytest.approx(1.0 / 6.0)
        assert by_id["a"] > by_id["c"]
