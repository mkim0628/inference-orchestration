"""Unit tests for InfoFlowChunkReorderCache (Activity B)."""

import time

import pytest
import torch

from src.cache.base import CacheStore
from src.cache.info_flow_reorder import InfoFlowChunkReorderCache


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

CAPACITY = 64 * 1024 * 1024  # 64 MiB


def make_kv(layers: int = 2, heads: int = 2, seq: int = 32, dim: int = 16) -> torch.Tensor:
    return torch.randn(layers, heads, seq, dim)


def make_cache() -> InfoFlowChunkReorderCache:
    return InfoFlowChunkReorderCache(capacity_bytes=CAPACITY)


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #

class TestInfoflowScoreComputation:
    def test_infoflow_score_computation(self) -> None:
        """_compute_infoflow_score returns a non-negative float."""
        cache = make_cache()
        kv = make_kv()
        score = cache._compute_infoflow_score(kv)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_score_stored_on_put(self) -> None:
        """put() stores the infoflow score alongside the KV tensor."""
        cache = make_cache()
        kv = make_kv()
        cache.put("seg1", kv)
        score = cache.get_infoflow_score("seg1")
        assert score is not None
        assert score >= 0.0

    def test_score_differs_by_content(self) -> None:
        """Segments with very different K norms produce different scores."""
        cache = make_cache()
        low_kv = torch.ones(1, 1, 8, 8) * 0.001
        high_kv = torch.ones(1, 1, 8, 8) * 100.0
        s_low = cache._compute_infoflow_score(low_kv)
        s_high = cache._compute_infoflow_score(high_kv)
        assert s_high > s_low


class TestReorderChunksDescending:
    def test_reorder_chunks_descending(self) -> None:
        """reorder_chunks() returns segments sorted by infoflow score descending."""
        cache = make_cache()
        # Insert 3 segments; drive different scores via norm magnitude
        for name, scale in [("low", 0.01), ("mid", 1.0), ("high", 100.0)]:
            kv = torch.ones(1, 1, 16, 8) * scale
            cache.put(name, kv)

        reordered = cache.reorder_chunks(["low", "mid", "high"])
        # Must be in descending score order
        scores = [cache.get_infoflow_score(k) for k in reordered]
        assert scores == sorted(scores, reverse=True)

    def test_reorder_single_chunk(self) -> None:
        """Single-chunk input returns the same key."""
        cache = make_cache()
        cache.put("only", make_kv())
        assert cache.reorder_chunks(["only"]) == ["only"]

    def test_reorder_missing_chunks_excluded(self) -> None:
        """Chunks not in cache are silently excluded from result."""
        cache = make_cache()
        cache.put("present", make_kv())
        result = cache.reorder_chunks(["present", "absent"])
        assert "absent" not in result
        assert "present" in result


class TestReorderWithExternalScores:
    def test_reorder_with_external_scores(self) -> None:
        """When external attention scores are provided they are blended 0.5/0.5."""
        cache = make_cache()
        # Both internal scores are equal (same content)
        kv = torch.ones(1, 1, 8, 4)
        cache.put("a", kv)
        cache.put("b", kv)

        # External score strongly favours "b"
        ext = {"a": 0.0, "b": 1000.0}
        reordered = cache.reorder_chunks(["a", "b"], attention_scores=ext)
        assert reordered[0] == "b"

    def test_external_scores_not_required(self) -> None:
        """reorder_chunks works correctly when attention_scores is None."""
        cache = make_cache()
        for name in ["x", "y", "z"]:
            cache.put(name, make_kv())
        result = cache.reorder_chunks(["x", "y", "z"])
        assert len(result) == 3


class TestRopeOverheadMeasurement:
    def test_rope_overhead_measurement(self) -> None:
        """last_timing() reports non-negative reorder and RoPE times."""
        cache = make_cache()
        for i in range(5):
            cache.put(f"s{i}", make_kv(seq=64))

        cache.reorder_chunks([f"s{i}" for i in range(5)])
        timing = cache.last_timing()

        assert "reorder_s" in timing
        assert "rope_recompute_s" in timing
        assert timing["reorder_s"] >= 0.0
        assert timing["rope_recompute_s"] >= 0.0

    def test_timing_resets_on_reset_stats(self) -> None:
        """reset_stats() zeroes the timing fields."""
        cache = make_cache()
        cache.put("t", make_kv())
        cache.reorder_chunks(["t"])
        cache.reset_stats()
        timing = cache.last_timing()
        assert timing["reorder_s"] == 0.0
        assert timing["rope_recompute_s"] == 0.0


class TestCacheStoreInterface:
    def test_cachestore_interface(self) -> None:
        """InfoFlowChunkReorderCache is a proper CacheStore subclass."""
        assert issubclass(InfoFlowChunkReorderCache, CacheStore)
        cache = make_cache()
        assert callable(cache.put)
        assert callable(cache.get)
        assert callable(cache.evict)
        assert callable(cache.hit_rate)
        assert callable(cache.memory_bytes)
        assert callable(cache.reset_stats)

    def test_put_get_round_trip(self) -> None:
        cache = make_cache()
        kv = make_kv()
        cache.put("key", kv)
        result = cache.get("key")
        assert result is not None
        assert result.shape == kv.shape

    def test_miss_returns_none(self) -> None:
        cache = make_cache()
        assert cache.get("missing") is None

    def test_hit_rate_correct(self) -> None:
        cache = make_cache()
        cache.put("h", make_kv())
        cache.get("h")   # hit
        cache.get("m")   # miss
        assert cache.hit_rate() == pytest.approx(0.5)

    def test_evict_returns_bytes(self) -> None:
        cache = make_cache()
        kv = make_kv()
        cache.put("e", kv)
        freed = cache.evict()
        assert freed == kv.nbytes

    def test_evict_empty_returns_zero(self) -> None:
        cache = make_cache()
        assert cache.evict() == 0

    def test_memory_bytes_tracks_puts(self) -> None:
        cache = make_cache()
        assert cache.memory_bytes() == 0
        cache.put("m", make_kv())
        assert cache.memory_bytes() > 0
