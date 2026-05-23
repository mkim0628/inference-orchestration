"""Unit tests for CLCPositionalBiasGatedSegmentCache (Activity B).

Covers: delta_pos computation, 3-stage policy gate, CacheStore interface,
LRU eviction, hit tracking, and non-contiguous direct hit rate.
"""

import pytest
import torch

from src.cache.clc_positional_bias_gated_segment_cache import (
    CLCBiasGateConfig,
    CLCPositionalBiasGatedSegmentCache,
    ReencodingPolicy,
    SegmentMeta,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def config() -> CLCBiasGateConfig:
    return CLCBiasGateConfig(
        max_context_length=4096,
        bias_threshold=0.15,
        rope_distortion_threshold=0.40,
        max_entries=100,
        seed=42,
    )


@pytest.fixture
def cache(config: CLCBiasGateConfig) -> CLCPositionalBiasGatedSegmentCache:
    return CLCPositionalBiasGatedSegmentCache(config)


def _make_kv(seq_len: int = 16, d_head: int = 64, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, d_head)


# --------------------------------------------------------------------------- #
# Delta position computation                                                   #
# --------------------------------------------------------------------------- #


def test_clc_delta_pos_zero_same_position(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """pos_orig_start=100, pos_target_start=100 -> ΔPos=0.0."""
    delta = cache.compute_delta_pos(100, 100)
    assert delta == 0.0


def test_clc_delta_pos_normalized(cache: CLCPositionalBiasGatedSegmentCache) -> None:
    """|pos_target - pos_orig| / max_context_length normalization check."""
    # |200 - 100| / 4096 ≈ 0.02441
    delta = cache.compute_delta_pos(100, 200)
    expected = abs(200 - 100) / 4096
    assert abs(delta - expected) < 1e-8, f"delta={delta} != expected={expected}"


# --------------------------------------------------------------------------- #
# 3-stage policy gate                                                          #
# --------------------------------------------------------------------------- #


def test_clc_bias_gate_direct_reuse_small_delta(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """ΔPos=0.10 <= 0.15 -> ReencodingPolicy.DIRECT_REUSE."""
    # ΔPos = 0.10 -> shift = 0.10 * 4096 = 409.6 ≈ 409 tokens
    meta = SegmentMeta(pos_orig_start=0, pos_orig_end=64, content_hash="h1")
    pos_target = int(0.10 * 4096)
    policy = cache.check_bias(meta, pos_target)
    assert policy == ReencodingPolicy.DIRECT_REUSE


def test_clc_bias_gate_partial_reencoding_mid_delta(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """ΔPos=0.25 (0.15 < 0.25 <= 0.40) -> ReencodingPolicy.PARTIAL_REENCODING."""
    meta = SegmentMeta(pos_orig_start=0, pos_orig_end=64, content_hash="h2")
    pos_target = int(0.25 * 4096)
    policy = cache.check_bias(meta, pos_target)
    assert policy == ReencodingPolicy.PARTIAL_REENCODING


def test_clc_bias_gate_full_reencoding_large_delta(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """ΔPos=0.60 > 0.40 -> ReencodingPolicy.FULL_REENCODING."""
    meta = SegmentMeta(pos_orig_start=0, pos_orig_end=64, content_hash="h3")
    pos_target = int(0.60 * 4096)
    policy = cache.check_bias(meta, pos_target)
    assert policy == ReencodingPolicy.FULL_REENCODING


# --------------------------------------------------------------------------- #
# put_segment / get_with_policy                                                #
# --------------------------------------------------------------------------- #


def test_clc_put_segment_stores_meta(cache: CLCPositionalBiasGatedSegmentCache) -> None:
    """put_segment stores _meta[key]."""
    kv = _make_kv()
    cache.put_segment("seg1", kv, pos_orig_start=0, pos_orig_end=16, content_hash="abc")
    assert "seg1" in cache._meta
    assert cache._meta["seg1"].pos_orig_start == 0
    assert cache._meta["seg1"].content_hash == "abc"


def test_clc_get_with_policy_returns_tuple(cache: CLCPositionalBiasGatedSegmentCache) -> None:
    """put_segment -> get_with_policy returns (tensor, policy) tuple."""
    kv = _make_kv()
    cache.put_segment("seg1", kv, pos_orig_start=0, pos_orig_end=16, content_hash="abc")
    result, policy = cache.get_with_policy("seg1", pos_target_start=0)
    assert result is not None
    assert isinstance(policy, ReencodingPolicy)


def test_clc_get_with_policy_miss_returns_none(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """Non-existent key -> (None, FULL_REENCODING)."""
    result, policy = cache.get_with_policy("nonexistent", pos_target_start=0)
    assert result is None
    assert policy == ReencodingPolicy.FULL_REENCODING


def test_clc_direct_reuse_hit_count_increments(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """DIRECT_REUSE policy hit increments _direct_reuse_hits."""
    kv = _make_kv()
    cache.put_segment("seg1", kv, pos_orig_start=0, pos_orig_end=16, content_hash="abc")
    # ΔPos = 0 -> DIRECT_REUSE
    cache.get_with_policy("seg1", pos_target_start=0)
    assert cache._direct_reuse_hits >= 1


def test_clc_partial_reencoding_hit_count_increments(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """PARTIAL_REENCODING policy hit increments _partial_reencoding_hits."""
    kv = _make_kv()
    cache.put_segment("seg1", kv, pos_orig_start=0, pos_orig_end=16, content_hash="abc")
    pos_target = int(0.25 * 4096)  # ΔPos ≈ 0.25 -> PARTIAL
    cache.get_with_policy("seg1", pos_target_start=pos_target)
    assert cache._partial_reencoding_hits >= 1


# --------------------------------------------------------------------------- #
# noncontiguous_direct_hit_rate                                                #
# --------------------------------------------------------------------------- #


def test_clc_noncontiguous_direct_hit_rate(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """direct_reuse_hits=3, partial=1, full=1 -> rate=3/5=0.60."""
    cache._direct_reuse_hits = 3
    cache._partial_reencoding_hits = 1
    cache._full_reencoding_hits = 1
    rate = cache.noncontiguous_direct_hit_rate()
    assert abs(rate - 0.60) < 1e-6, f"rate={rate} != 0.60"


# --------------------------------------------------------------------------- #
# CacheStore interface                                                         #
# --------------------------------------------------------------------------- #


def test_clc_cachestore_interface_full(cache: CLCPositionalBiasGatedSegmentCache) -> None:
    """put/get/evict/hit_rate/memory_bytes/reset_stats all work."""
    kv = _make_kv()
    cache.put("a", kv)
    assert cache.get("a") is not None
    assert cache.get("missing") is None
    assert cache.hit_rate() > 0.0
    assert cache.memory_bytes() > 0
    freed = cache.evict()
    assert freed > 0
    cache.reset_stats()
    assert cache._hits == 0
    assert cache._misses == 0


def test_clc_evict_lru_oldest_first() -> None:
    """max_entries=2, 3rd put evicts oldest entry."""
    cfg = CLCBiasGateConfig(max_entries=2, seed=42)
    cache = CLCPositionalBiasGatedSegmentCache(cfg)
    cache.put("a", _make_kv(seed=1))
    cache.put("b", _make_kv(seed=2))
    cache.put("c", _make_kv(seed=3))  # evicts "a"
    assert "a" not in cache._store
    assert "b" in cache._store
    assert "c" in cache._store


def test_clc_hit_rate_tracking(cache: CLCPositionalBiasGatedSegmentCache) -> None:
    """2 puts, 1 hit + 1 miss -> hit_rate() == 0.5."""
    cache.put("a", _make_kv(seed=1))
    cache.put("b", _make_kv(seed=2))
    cache.get("a")        # hit
    cache.get("missing")  # miss
    assert cache.hit_rate() == 0.5


# --------------------------------------------------------------------------- #
# reset_stats clears policy hit counters                                       #
# --------------------------------------------------------------------------- #


def test_clc_reset_stats_clears_policy_counters(
    cache: CLCPositionalBiasGatedSegmentCache,
) -> None:
    """reset_stats clears all hit counters including policy-specific ones."""
    cache._direct_reuse_hits = 5
    cache._partial_reencoding_hits = 3
    cache._full_reencoding_hits = 2
    cache.reset_stats()
    assert cache._direct_reuse_hits == 0
    assert cache._partial_reencoding_hits == 0
    assert cache._full_reencoding_hits == 0
    assert cache._hits == 0
    assert cache._misses == 0
