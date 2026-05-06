"""Unit tests for QueryCentricRecomputeCache (Activity B)."""

import pytest
import torch

from src.cache.base import CacheStore
from src.cache.query_centric_recompute import QueryCentricRecomputeCache


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def make_kv(layers: int = 2, heads: int = 2, seq: int = 64, dim: int = 32) -> torch.Tensor:
    return torch.randn(layers, heads, seq, dim)


def make_cache(capacity: int = 10 * 1024 * 1024) -> QueryCentricRecomputeCache:
    return QueryCentricRecomputeCache(capacity_bytes=capacity)


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #

class TestPutGetBasic:
    def test_put_get_basic(self) -> None:
        """put/get round-trip returns the stored tensor."""
        cache = make_cache()
        kv = make_kv()
        cache.put("seg1", kv)
        result = cache.get("seg1")
        assert result is not None
        assert result.shape == kv.shape

    def test_get_miss_returns_none(self) -> None:
        """get() returns None for unknown keys."""
        cache = make_cache()
        assert cache.get("nonexistent") is None

    def test_put_duplicate_key_does_not_grow(self) -> None:
        """Re-putting an existing key does not add a new entry."""
        cache = make_cache()
        kv = make_kv()
        cache.put("k1", kv)
        cache.put("k1", kv)
        assert len(cache._store) == 1


class TestStage1AttnNormFilter:
    def test_stage1_attn_norm_filter(self) -> None:
        """Stage 1 selects top-50% by attention norm."""
        cache = make_cache()
        # Create segments with deliberately different norms
        low_kv = torch.ones(2, 2, 32, 16) * 0.01    # low norm
        high_kv = torch.ones(2, 2, 32, 16) * 10.0   # high norm
        cache.put("low", low_kv)
        cache.put("high", high_kv)

        query = torch.randn(16)
        # With 2 segments and top-50% → 1 passes stage 1
        selected = cache.selective_recompute(query, ["low", "high"], budget=1.0)
        # "high" must be in selected (higher attn_norm)
        assert "high" in selected

    def test_stage1_selects_at_least_one(self) -> None:
        """Stage 1 always selects at least 1 candidate."""
        cache = make_cache()
        cache.put("only", make_kv())
        query = torch.randn(32)
        selected = cache.selective_recompute(query, ["only"], budget=1.0)
        assert len(selected) >= 1


class TestStage2CosineRelevance:
    def test_stage2_cosine_relevance(self) -> None:
        """Stage 2 ranks survivors by cosine similarity with query."""
        cache = make_cache()
        dim = 16
        # "relevant" segment: KV mean aligned with query
        relevant_kv = torch.ones(1, 1, 32, dim)
        query = torch.ones(dim)
        cache.put("relevant", relevant_kv)

        # "irrelevant" segment: KV mean anti-aligned with query
        irrelevant_kv = -torch.ones(1, 1, 32, dim)
        cache.put("irrelevant", irrelevant_kv)

        selected = cache.selective_recompute(query, ["relevant", "irrelevant"], budget=1.0)
        # "relevant" should rank first
        if len(selected) >= 2:
            assert selected[0] == "relevant"
        elif len(selected) == 1:
            assert selected[0] == "relevant"


class TestRecomputeBudgetLimit:
    def test_recompute_budget_limit(self) -> None:
        """Budget of 20% should not return more than 20% of total tokens."""
        cache = make_cache()
        # 5 segments, each 100 tokens
        for i in range(5):
            cache.put(f"seg{i}", make_kv(layers=1, heads=1, seq=100, dim=16))

        query = torch.randn(16)
        selected = cache.selective_recompute(
            query, [f"seg{i}" for i in range(5)], budget=0.20
        )
        total_kept = sum(cache._store[k]["kv"].shape[2] for k in selected if k in cache._store)
        assert total_kept <= 100  # 20% of 500 tokens = 100

    def test_empty_cache_returns_empty(self) -> None:
        """selective_recompute on empty candidate list returns []."""
        cache = make_cache()
        result = cache.selective_recompute(torch.randn(16), [], budget=0.20)
        assert result == []

    def test_missing_segments_are_skipped(self) -> None:
        """Segments not in store are silently ignored."""
        cache = make_cache()
        cache.put("exists", make_kv())
        result = cache.selective_recompute(
            torch.randn(32), ["exists", "ghost"], budget=1.0
        )
        assert "ghost" not in result
        assert "exists" in result


class TestBudgetRatioParameter:
    @pytest.mark.parametrize("ratio", [0.10, 0.15, 0.20, 0.25, 0.30])
    def test_budget_ratio_parameter(self, ratio: float) -> None:
        """Varying recompute_budget_ratio [0.10, 0.30] works without error."""
        cache = QueryCentricRecomputeCache(
            capacity_bytes=10 * 1024 * 1024,
            recompute_budget_ratio=ratio,
        )
        for i in range(4):
            cache.put(f"s{i}", make_kv(layers=1, heads=1, seq=64, dim=16))
        query = torch.randn(16)
        selected = cache.selective_recompute(
            query, [f"s{i}" for i in range(4)], budget=ratio
        )
        assert isinstance(selected, list)

    def test_invalid_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            QueryCentricRecomputeCache(capacity_bytes=1024, recompute_budget_ratio=0.0)

    def test_invalid_stage1_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            QueryCentricRecomputeCache(capacity_bytes=1024, stage1_top_k_ratio=1.5)


class TestHitRateTracking:
    def test_hit_rate_tracking(self) -> None:
        """hit_rate() accurately reflects get() outcomes."""
        cache = make_cache()
        cache.put("a", make_kv())

        cache.get("a")    # hit
        cache.get("b")    # miss

        assert cache.hit_rate() == pytest.approx(0.5)

    def test_hit_rate_zero_when_no_queries(self) -> None:
        cache = make_cache()
        assert cache.hit_rate() == 0.0

    def test_reset_stats_zeroes_counters(self) -> None:
        cache = make_cache()
        cache.put("x", make_kv())
        cache.get("x")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0


class TestCacheStoreInterface:
    def test_cachestore_interface(self) -> None:
        """QueryCentricRecomputeCache implements all CacheStore abstract methods."""
        assert issubclass(QueryCentricRecomputeCache, CacheStore)
        cache = make_cache()
        assert callable(cache.put)
        assert callable(cache.get)
        assert callable(cache.evict)
        assert callable(cache.hit_rate)
        assert callable(cache.memory_bytes)
        assert callable(cache.reset_stats)

    def test_evict_returns_bytes(self) -> None:
        cache = make_cache()
        kv = make_kv()
        cache.put("evictable", kv)
        freed = cache.evict()
        assert freed == kv.nbytes

    def test_evict_empty_cache_returns_zero(self) -> None:
        cache = make_cache()
        assert cache.evict() == 0

    def test_memory_bytes_grows_with_puts(self) -> None:
        cache = make_cache()
        assert cache.memory_bytes() == 0
        cache.put("m1", make_kv())
        assert cache.memory_bytes() > 0
