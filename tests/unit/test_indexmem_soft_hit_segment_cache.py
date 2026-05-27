"""Unit tests for IndexMemSoftHitSegmentCache (Activity B-1)."""

import pytest
import torch

from src.cache.indexmem_soft_hit_segment_cache import (
    HitResult,
    IndexMemSoftHitSegmentCache,
    SoftHitSegmentConfig,
)


def _make_cache(
    max_physical: int = 10,
    latent_pool_max: int = 100,
    chunk_size: int = 4,
    kv_dim: int = 32,
    latent_dim: int = 16,
    n_layers: int = 2,
    seed: int = 42,
) -> IndexMemSoftHitSegmentCache:
    cfg = SoftHitSegmentConfig(
        chunk_size=chunk_size,
        max_physical_entries=max_physical,
        latent_pool_max_segments=latent_pool_max,
        beta_soft=0.1,
        beta_weight=0.5,
        kv_dim=kv_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        seed=seed,
    )
    return IndexMemSoftHitSegmentCache(cfg)


def _rand_kv(n: int = 8, d: int = 32, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


# --------------------------------------------------------------------------- #


def test_cache_store_interface_all_methods():
    """All CacheStore abstract methods must work."""
    cache = _make_cache()
    kv = _rand_kv()

    cache.put("k1", kv)
    result = cache.get("k1")
    assert result is not None

    freed = cache.evict()
    assert freed >= 0

    hr = cache.hit_rate()
    assert 0.0 <= hr <= 1.0

    mb = cache.memory_bytes()
    assert isinstance(mb, int)

    cache.reset_stats()
    assert cache.hit_rate() == 0.0


def test_hard_hit_returns_kv_tensor():
    cache = _make_cache()
    kv = _rand_kv(seed=1)
    cache.put("hard_key", kv)
    result = cache.get_hit_result("hard_key")
    assert result.type == "hard"
    assert result.kv_tensor is not None
    assert result.latent_state is None
    assert torch.allclose(result.kv_tensor, kv)


def test_soft_hit_after_eviction():
    # Use a cache with only 1 physical entry so eviction happens on second put
    cache = _make_cache(max_physical=1, kv_dim=32, latent_dim=16)
    kv1 = _rand_kv(seed=10)
    cache.put("key_evict", kv1)

    # Force eviction by adding a second entry
    kv2 = _rand_kv(seed=11)
    cache.put("key_evict2", kv2)

    # key_evict should have been evicted (LRU) -> soft hit
    result = cache.get_hit_result("key_evict")
    assert result.type == "soft"
    assert result.latent_state is not None
    assert result.kv_tensor is None


def test_miss_no_physical_no_latent():
    cache = _make_cache()
    result = cache.get_hit_result("nonexistent_key")
    assert result.type == "miss"
    assert result.kv_tensor is None
    assert result.latent_state is None


def test_soft_hit_residual_output_shape():
    cache = _make_cache(kv_dim=32, latent_dim=16)
    torch.manual_seed(42)
    latent = torch.randn(16)  # latent_dim=16
    query = torch.randn(4, 32)  # [n_q, d_head]
    residual = cache.soft_hit_residual(query, latent)
    assert residual.shape == query.shape


def test_weighted_hit_rate_formula():
    cache = _make_cache(kv_dim=32, latent_dim=16, max_physical=1)
    torch.manual_seed(42)

    # Populate physical cache (1 entry max)
    kv1 = _rand_kv(seed=20)
    cache.put("k1", kv1)
    # Evict k1 by adding k2
    kv2 = _rand_kv(seed=21)
    cache.put("k2", kv2)

    # Reset stats and measure
    cache.reset_stats()

    # k1 -> soft hit, k2 -> hard hit, k3 -> miss
    cache.get_hit_result("k1")   # soft hit
    cache.get_hit_result("k2")   # hard hit
    cache.get_hit_result("k3")   # miss

    whr = cache.weighted_hit_rate()
    # (1 + 0.5*1) / 3 = 1.5/3 = 0.5
    expected = (1 + 0.5 * 1) / 3
    assert abs(whr - expected) < 1e-5, f"Expected {expected}, got {whr}"


def test_soft_hit_rate():
    cache = _make_cache(kv_dim=32, latent_dim=16, max_physical=1)
    torch.manual_seed(42)
    kv1 = _rand_kv(seed=30)
    cache.put("s1", kv1)
    kv2 = _rand_kv(seed=31)
    cache.put("s2", kv2)  # evicts s1

    cache.reset_stats()
    cache.get_hit_result("s1")  # soft hit
    cache.get_hit_result("s2")  # hard hit
    cache.get_hit_result("s3")  # miss

    shr = cache.soft_hit_rate()
    expected = 1.0 / 3.0
    assert abs(shr - expected) < 1e-5


def test_latent_pool_lru_eviction():
    cache = _make_cache(max_physical=1, latent_pool_max=2, kv_dim=32, latent_dim=16)
    # Force 3 evictions to fill + overflow latent pool

    # Put and evict 3 entries
    for i in range(4):
        kv = _rand_kv(seed=i + 100)
        cache.put(f"key_{i}", kv)

    # The first key should be LRU-evicted from latent pool if pool is full
    # With latent_pool_max=2, at most 2 latent states stored
    total_latent = len(cache._segment_latent_pool)
    assert total_latent <= 2


def test_put_segment_returns_key():
    cache = _make_cache(chunk_size=4, kv_dim=32, latent_dim=16)
    token_ids = [1, 2, 3, 4]
    kv = _rand_kv(n=4, d=32, seed=0)
    key = cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)
    assert isinstance(key, str)
    assert len(key) > 0


def test_get_segments_returns_hit_results():
    cache = _make_cache(chunk_size=4, kv_dim=32, latent_dim=16)
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 chunks
    kv0 = _rand_kv(n=4, d=32, seed=0)
    kv1 = _rand_kv(n=4, d=32, seed=1)

    cache.put_segment(token_ids, chunk_idx=0, kv=kv0, layer_idx=0)
    cache.put_segment(token_ids, chunk_idx=1, kv=kv1, layer_idx=0)

    hits, misses = cache.get_segments(token_ids, layer_idx=0)
    # Should have 2 hits (both chunks stored)
    assert len(hits) == 2
    assert len(misses) == 0

    for chunk_idx, hit_result in hits:
        assert isinstance(hit_result, HitResult)
        assert hit_result.type in ("hard", "soft", "miss")


def test_get_segments_partial_miss():
    cache = _make_cache(chunk_size=4, kv_dim=32, latent_dim=16)
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 chunks
    kv0 = _rand_kv(n=4, d=32, seed=0)

    # Only store chunk 0
    cache.put_segment(token_ids, chunk_idx=0, kv=kv0, layer_idx=0)

    hits, misses = cache.get_segments(token_ids, layer_idx=0)
    assert len(hits) == 1
    assert len(misses) == 1
    assert misses[0] == 1  # chunk 1 missed


def test_memory_bytes_increases_with_puts():
    cache = _make_cache(max_physical=100, kv_dim=32, latent_dim=16)
    assert cache.memory_bytes() == 0
    kv = _rand_kv(n=8, d=32)
    cache.put("mb_key", kv)
    assert cache.memory_bytes() > 0


def test_reset_stats_clears_counters():
    cache = _make_cache(kv_dim=32, latent_dim=16)
    kv = _rand_kv()
    cache.put("rk", kv)
    cache.get_hit_result("rk")
    cache.reset_stats()
    assert cache._n_hard_hits == 0
    assert cache._n_soft_hits == 0
    assert cache._n_misses == 0
