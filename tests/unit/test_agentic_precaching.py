"""Unit tests for AgenticChunkPreCachingPipeline (Activity A+B).

Covers:
  - CacheStore interface compliance
  - precache_predicted_chunks() execution and pre-folded prefix registration
  - get_with_precache() pre-folded prefix retrieval
  - prediction miss fallback to RadixAttention
  - precache_efficiency metric
"""

import pytest
import torch

from src.cache.agentic_chunk_precaching import (
    AgenticChunkPreCachingPipeline,
    AgenticPreCachingConfig,
)
from src.cache.kv_fold_accumulative import KVFoldConfig
from src.scheduler.pbkv_agent_segment_scheduler import PBKVConfig


@pytest.fixture
def small_config() -> AgenticPreCachingConfig:
    kvfold = KVFoldConfig(
        chunk_size=16,
        max_entries=200,
        d_head=32,
        n_heads=4,
        n_layers=2,
        enable_streaming_fallback=False,
        seed=42,
    )
    pbkv = PBKVConfig(
        segment_emb_dim=32,
        history_steps=3,
        chunk_size=16,
        fairness_max_wait=5,
        seed=42,
    )
    return AgenticPreCachingConfig(
        kvfold=kvfold,
        pbkv=pbkv,
        precache_top_k=5,
        precache_min_prob=0.0,  # accept all chunks in unit tests
    )


@pytest.fixture
def pipeline(small_config: AgenticPreCachingConfig) -> AgenticChunkPreCachingPipeline:
    return AgenticChunkPreCachingPipeline(small_config)


def make_chunk_tokens(n: int = 16, offset: int = 0) -> list:
    return list(range(offset, offset + n))


# ------------------------------------------------------------------ #
# CacheStore interface                                                 #
# ------------------------------------------------------------------ #

class TestCacheStoreInterface:
    def test_put_get_roundtrip(self, pipeline: AgenticChunkPreCachingPipeline) -> None:
        kv = torch.randn(4, 2, 4, 32)
        pipeline.put("k1", kv)
        result = pipeline.get("k1")
        assert result is not None
        assert result.shape == kv.shape

    def test_get_miss_returns_none(self, pipeline: AgenticChunkPreCachingPipeline) -> None:
        assert pipeline.get("nonexistent") is None

    def test_evict_returns_nonnegative(self, pipeline: AgenticChunkPreCachingPipeline) -> None:
        pipeline.put("e", torch.randn(4, 2, 4, 32))
        freed = pipeline.evict()
        assert freed >= 0

    def test_hit_rate_in_range(self, pipeline: AgenticChunkPreCachingPipeline) -> None:
        pipeline.put("k", torch.randn(4, 2, 4, 32))
        pipeline.get("k")
        pipeline.get("miss")
        hr = pipeline.hit_rate()
        assert 0.0 <= hr <= 1.0

    def test_memory_bytes_nonnegative(self, pipeline: AgenticChunkPreCachingPipeline) -> None:
        assert pipeline.memory_bytes() >= 0

    def test_reset_stats(self, pipeline: AgenticChunkPreCachingPipeline) -> None:
        pipeline.put("k", torch.randn(4, 2, 4, 32))
        pipeline.get("k")
        pipeline.reset_stats()
        assert pipeline.hit_rate() == 0.0


# ------------------------------------------------------------------ #
# precache_predicted_chunks()                                          #
# ------------------------------------------------------------------ #

class TestPrecachePredictedChunks:
    def test_returns_fold_key_for_valid_input(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        chunks = [make_chunk_tokens(16, i * 16) for i in range(3)]
        fold_key = pipeline.precache_predicted_chunks("agent_a", chunks)
        assert fold_key is not None
        assert isinstance(fold_key, str)

    def test_returns_none_for_empty_input(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        result = pipeline.precache_predicted_chunks("agent_b", [])
        assert result is None

    def test_fold_key_registered_in_registry(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        chunks = [make_chunk_tokens(16, i * 16) for i in range(3)]
        fold_key = pipeline.precache_predicted_chunks("agent_c", chunks)
        assert fold_key is not None
        assert fold_key in pipeline._prefolded_registry

    def test_registry_has_chunk_ids_and_prob(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        chunks = [make_chunk_tokens(16, i * 16) for i in range(3)]
        fold_key = pipeline.precache_predicted_chunks("agent_d", chunks)
        assert fold_key is not None
        entry = pipeline._prefolded_registry[fold_key]
        assert "chunk_ids" in entry
        assert "prob" in entry
        assert 0.0 <= entry["prob"] <= 1.0

    def test_precache_raises_hit_in_fold_cache(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        """After precaching, the fold cache should have the prefix registered."""
        chunks = [make_chunk_tokens(16, i * 16) for i in range(2)]
        fold_key = pipeline.precache_predicted_chunks("agent_e", chunks)
        assert fold_key is not None
        prefix = pipeline.fold_cache.get_folded_prefix(fold_key)
        assert prefix is not None

    def test_min_prob_filter_rejects_all_when_high(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        """precache_min_prob=1.0 (impossible) → None returned."""
        pipeline.config.precache_min_prob = 1.0
        chunks = [make_chunk_tokens(16, i * 16) for i in range(3)]
        result = pipeline.precache_predicted_chunks("agent_f", chunks)
        assert result is None


# ------------------------------------------------------------------ #
# get_with_precache()                                                  #
# ------------------------------------------------------------------ #

class TestGetWithPrecache:
    def test_returns_triple(self, pipeline: AgenticChunkPreCachingPipeline) -> None:
        tokens = make_chunk_tokens(32, 0)
        hits, misses, fold_prefix = pipeline.get_with_precache(tokens)
        assert isinstance(hits, list)
        assert isinstance(misses, list)
        assert fold_prefix is None  # no precache_fold_key

    def test_returns_fold_prefix_when_registered(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        chunks = [make_chunk_tokens(16, i * 16) for i in range(3)]
        fold_key = pipeline.precache_predicted_chunks("agent_g", chunks)
        assert fold_key is not None
        pipeline.fold_cache.reset_stats()  # reset counters before test
        token_ids = make_chunk_tokens(48, 0)
        hits, misses, fold_prefix = pipeline.get_with_precache(
            token_ids, layer_idx=0, precache_fold_key=fold_key
        )
        assert fold_prefix is not None

    def test_precache_hits_increments(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        chunks = [make_chunk_tokens(16, i * 16) for i in range(2)]
        fold_key = pipeline.precache_predicted_chunks("agent_h", chunks)
        assert fold_key is not None
        pipeline._precache_hits = 0
        pipeline.get_with_precache(make_chunk_tokens(32), precache_fold_key=fold_key)
        assert pipeline._precache_hits == 1


# ------------------------------------------------------------------ #
# fallback_to_radix_attention()                                        #
# ------------------------------------------------------------------ #

class TestFallbackToRadixAttention:
    def test_fallback_increments_counter(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        tokens = make_chunk_tokens(32)
        pipeline.fallback_to_radix_attention(tokens)
        assert pipeline._fallback_count == 1

    def test_fallback_returns_hits_misses_tuple(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        tokens = make_chunk_tokens(32)
        result = pipeline.fallback_to_radix_attention(tokens)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fallback_finds_stored_segments(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        tokens = make_chunk_tokens(32)
        kv = torch.randn(16, 2, 4, 32)
        pipeline.fold_cache._store.put_segment(tokens, 0, kv, 0)
        hits, misses = pipeline.fallback_to_radix_attention(tokens)
        assert len(hits) >= 1


# ------------------------------------------------------------------ #
# precache_efficiency()                                                #
# ------------------------------------------------------------------ #

class TestPrecacheEfficiency:
    def test_efficiency_zero_before_any_access(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        assert pipeline.precache_efficiency() == 0.0

    def test_efficiency_positive_after_precache_hit(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        chunks = [make_chunk_tokens(16, i * 16) for i in range(2)]
        fold_key = pipeline.precache_predicted_chunks("eff_agent", chunks)
        assert fold_key is not None
        pipeline.reset_stats()
        pipeline.get_with_precache(make_chunk_tokens(32), precache_fold_key=fold_key)
        # At least one hit or miss was recorded; precache_hits may have incremented
        eff = pipeline.precache_efficiency()
        assert 0.0 <= eff <= 1.0

    def test_noncontiguous_hit_rate_delegates_to_fold_cache(
        self, pipeline: AgenticChunkPreCachingPipeline
    ) -> None:
        rate = pipeline.noncontiguous_hit_rate()
        assert rate == pipeline.fold_cache.noncontiguous_hit_rate()
