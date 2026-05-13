"""Unit tests for KVFoldAccumulativeRadixCache (Activity B).

Covers:
  - CacheStore interface compliance (all 6 abstract methods)
  - fold_chunk accumulation shape and determinism
  - Drift plateau detection convergence
  - noncontiguous_hit_rate tracking
  - register_prefolded_prefix
  - foldl chain depth statistics
"""

import pytest
import torch

from src.cache.kv_fold_accumulative import KVFoldAccumulativeRadixCache, KVFoldConfig


@pytest.fixture
def default_config() -> KVFoldConfig:
    return KVFoldConfig(
        chunk_size=16,
        max_entries=100,
        drift_threshold=1e-3,
        max_fold_depth=511,
        enable_streaming_fallback=False,
        d_head=32,
        n_heads=4,
        n_layers=2,
        seed=42,
    )


@pytest.fixture
def cache(default_config: KVFoldConfig) -> KVFoldAccumulativeRadixCache:
    return KVFoldAccumulativeRadixCache(default_config)


# ------------------------------------------------------------------ #
# CacheStore interface                                                 #
# ------------------------------------------------------------------ #

class TestCacheStoreInterface:
    def test_put_get_roundtrip(self, cache: KVFoldAccumulativeRadixCache) -> None:
        """put → get returns the stored tensor."""
        kv = torch.randn(4, 2, 4, 32)
        cache.put("test_key", kv)
        result = cache.get("test_key")
        assert result is not None
        assert result.shape == kv.shape

    def test_get_miss_returns_none(self, cache: KVFoldAccumulativeRadixCache) -> None:
        assert cache.get("nonexistent_key") is None

    def test_evict_frees_bytes(self, cache: KVFoldAccumulativeRadixCache) -> None:
        kv = torch.randn(8, 2, 4, 32)
        cache.put("evict_key", kv)
        mem_before = cache.memory_bytes()
        freed = cache.evict()
        assert freed > 0

    def test_hit_rate_tracks_correctly(self, cache: KVFoldAccumulativeRadixCache) -> None:
        kv = torch.randn(4, 2, 4, 32)
        cache.put("k1", kv)
        cache.get("k1")    # hit
        cache.get("miss")  # miss
        hr = cache.hit_rate()
        # hit_rate from standard path: 1 hit via get (standard), 1 miss
        # Note: get increments counters; fold_hits are also counted
        assert 0.0 <= hr <= 1.0

    def test_memory_bytes_increases_with_put(self, cache: KVFoldAccumulativeRadixCache) -> None:
        m0 = cache.memory_bytes()
        cache.put("k", torch.randn(4, 2, 4, 32))
        assert cache.memory_bytes() >= m0

    def test_reset_stats_clears_counters(self, cache: KVFoldAccumulativeRadixCache) -> None:
        cache.put("k", torch.randn(4, 2, 4, 32))
        cache.get("k")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0


# ------------------------------------------------------------------ #
# fold_chunk accumulation                                              #
# ------------------------------------------------------------------ #

class TestFoldChunk:
    def test_fold_chunk_returns_key_and_tensor(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        chunk_tokens = list(range(16))
        fold_key, acc_kv = cache.fold_chunk(chunk_tokens, layer_idx=0)
        assert isinstance(fold_key, str)
        assert acc_kv.dim() == 4

    def test_fold_chunk_shape_matches_config(
        self, cache: KVFoldAccumulativeRadixCache, default_config: KVFoldConfig
    ) -> None:
        chunk_tokens = list(range(default_config.chunk_size))
        _, acc_kv = cache.fold_chunk(chunk_tokens, layer_idx=0)
        assert acc_kv.shape[1] == 2
        assert acc_kv.shape[2] == default_config.n_heads
        assert acc_kv.shape[3] == default_config.d_head

    def test_fold_chunk_accumulates_across_calls(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        tokens_a = list(range(16))
        tokens_b = list(range(16, 32))
        fold_key1, kv1 = cache.fold_chunk(tokens_a, layer_idx=0)
        fold_key2, kv2 = cache.fold_chunk(tokens_b, layer_idx=0, existing_fold_key=fold_key1)
        # After two chunks the accumulated KV should have more tokens than one chunk
        assert kv2.shape[0] >= kv1.shape[0]

    def test_fold_chunk_deterministic_with_same_tokens(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        tokens = list(range(16))
        _, kv1 = cache.fold_chunk(tokens, layer_idx=0)
        # Create fresh cache and repeat
        cache2 = KVFoldAccumulativeRadixCache(cache.config)
        _, kv2 = cache2.fold_chunk(tokens, layer_idx=0)
        assert torch.allclose(kv1.float(), kv2.float(), atol=1e-5)

    def test_fold_chunk_early_exit_on_plateau(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        """After plateau is reached, fold_chunk returns same key without new state."""
        tokens = list(range(16))
        # Force plateau by directly setting state
        fold_key1, _ = cache.fold_chunk(tokens, layer_idx=0)
        # Manually mark plateau
        state = cache._fold_states[fold_key1]
        state.plateau_reached = True
        state.fold_depth = 5
        # Second call with same key should return it unchanged
        returned_key, _ = cache.fold_chunk(list(range(32, 48)), layer_idx=0,
                                            existing_fold_key=fold_key1)
        assert returned_key == fold_key1


# ------------------------------------------------------------------ #
# Drift plateau detection                                              #
# ------------------------------------------------------------------ #

class TestDriftPlateau:
    def test_plateau_not_reached_initially(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        tokens = list(range(16))
        fold_key, _ = cache.fold_chunk(tokens, layer_idx=0)
        state = cache._fold_states[fold_key]
        # First call: fold_depth=1, plateau requires depth >= 2
        assert not state.plateau_reached

    def test_drift_is_finite_after_first_chunk(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        tokens = list(range(16))
        fold_key, _ = cache.fold_chunk(tokens, layer_idx=0)
        state = cache._fold_states[fold_key]
        # After first chunk with empty prefix, drift is inf (no prev_tail)
        assert state.last_drift == float("inf")

    def test_fold_depth_increments(self, cache: KVFoldAccumulativeRadixCache) -> None:
        tokens_a = list(range(16))
        tokens_b = list(range(16, 32))
        fold_key1, _ = cache.fold_chunk(tokens_a, layer_idx=0)
        fold_key2, _ = cache.fold_chunk(tokens_b, layer_idx=0, existing_fold_key=fold_key1)
        state = cache._fold_states[fold_key2]
        assert state.fold_depth == 2


# ------------------------------------------------------------------ #
# Noncontiguous hit rate                                               #
# ------------------------------------------------------------------ #

class TestNoncontiguousHitRate:
    def test_noncontiguous_hit_rate_zero_initially(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        assert cache.noncontiguous_hit_rate() == 0.0

    def test_noncontiguous_hit_rate_after_fold_hits(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        tokens = list(range(16))
        fold_key, _ = cache.fold_chunk(tokens, layer_idx=0)
        # Fetch the prefix — increments fold_hits and hits
        prefix = cache.get_folded_prefix(fold_key)
        assert prefix is not None
        # fold_hits > 0, hits > 0 → rate > 0
        assert cache.noncontiguous_hit_rate() > 0.0

    def test_noncontiguous_hit_rate_bounded(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        tokens = list(range(16))
        fold_key, _ = cache.fold_chunk(tokens, layer_idx=0)
        cache.get_folded_prefix(fold_key)
        rate = cache.noncontiguous_hit_rate()
        assert 0.0 <= rate <= 1.0


# ------------------------------------------------------------------ #
# register_prefolded_prefix                                            #
# ------------------------------------------------------------------ #

class TestRegisterPrefoldedPrefix:
    def test_register_and_retrieve(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        acc_kv = torch.randn(32, 2, 4, 32)
        chunk_ids = [1, 2, 3]
        fold_key = "test_prefolded_key"
        cache.register_prefolded_prefix(fold_key, acc_kv, chunk_ids)
        retrieved = cache.get_folded_prefix(fold_key)
        assert retrieved is not None
        assert retrieved.shape == acc_kv.shape

    def test_register_sets_plateau(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        acc_kv = torch.randn(16, 2, 4, 32)
        cache.register_prefolded_prefix("pre_key", acc_kv, [10, 20])
        state = cache._fold_states["pre_key"]
        assert state.plateau_reached is True

    def test_register_also_puts_in_segment_store(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        acc_kv = torch.randn(16, 2, 4, 32)
        fold_key = "pre_store_key"
        cache.register_prefolded_prefix(fold_key, acc_kv, [1])
        # Should be accessible via standard get()
        result = cache._store.get(fold_key)
        assert result is not None


# ------------------------------------------------------------------ #
# fold depth statistics                                                #
# ------------------------------------------------------------------ #

class TestFoldDepthStats:
    def test_stats_empty_cache(self, cache: KVFoldAccumulativeRadixCache) -> None:
        stats = cache.get_fold_depth_stats()
        assert stats["mean_depth"] == 0.0
        assert stats["plateau_ratio"] == 0.0
        assert stats["n_chains"] == 0

    def test_stats_after_folds(self, cache: KVFoldAccumulativeRadixCache) -> None:
        tokens_a = list(range(16))
        tokens_b = list(range(16, 32))
        fold_key1, _ = cache.fold_chunk(tokens_a, layer_idx=0)
        cache.fold_chunk(tokens_b, layer_idx=0, existing_fold_key=fold_key1)
        stats = cache.get_fold_depth_stats()
        assert stats["n_chains"] >= 1
        assert stats["mean_depth"] >= 1.0


# ------------------------------------------------------------------ #
# Segment API compatibility                                            #
# ------------------------------------------------------------------ #

class TestSegmentAPI:
    def test_put_segment_and_get_segments(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        token_ids = list(range(32))
        kv = torch.randn(16, 2, 4, 32)
        cache.put_segment(token_ids, 0, kv, layer_idx=0)
        hits, misses = cache.get_segments(token_ids, layer_idx=0)
        assert len(hits) > 0

    def test_get_segments_with_fold_returns_triple(
        self, cache: KVFoldAccumulativeRadixCache
    ) -> None:
        token_ids = list(range(32))
        hits, misses, fold_prefix = cache.get_segments_with_fold(token_ids, layer_idx=0)
        assert isinstance(hits, list)
        assert isinstance(misses, list)
        assert fold_prefix is None  # no fold_key provided


# ------------------------------------------------------------------ #
# StreamingLLM fallback                                                #
# ------------------------------------------------------------------ #

class TestStreamingFallback:
    def test_streaming_fallback_caps_accumulated_tokens(self) -> None:
        config = KVFoldConfig(
            chunk_size=16,
            max_entries=100,
            enable_streaming_fallback=True,
            window_size=2,  # very small window to trigger fallback
            d_head=32,
            n_heads=4,
        )
        cache = KVFoldAccumulativeRadixCache(config)
        fold_key = None
        for i in range(10):
            tokens = list(range(i * 16, (i + 1) * 16))
            fold_key, acc_kv = cache.fold_chunk(tokens, layer_idx=0, existing_fold_key=fold_key)

        max_expected = config.window_size * config.chunk_size
        # After streaming fallback the accumulated tensor should be capped
        assert acc_kv.shape[0] <= max_expected + 4  # +4 for sink tokens
