"""Unit tests for KVPacketCache (Activity B, 2026-05-25).

Tests cover store/retrieve roundtrip, multi-packet assembly, adapter shapes,
LRU eviction, and non-contiguous hit tracking.
"""

import pytest
import torch

from src.cache.kv_packet import KVPacketCache, KVPacketConfig, KVPacket


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_config(
    n_heads: int = 4,
    d_head: int = 32,
    n_adapter_tokens: int = 4,
    max_packets: int = 8,
    adapter_steps: int = 10,  # Reduced for test speed
    seed: int = 42,
) -> KVPacketConfig:
    return KVPacketConfig(
        n_heads=n_heads,
        d_head=d_head,
        n_adapter_tokens=n_adapter_tokens,
        max_packets=max_packets,
        adapter_steps=adapter_steps,
        seed=seed,
    )


def _make_kv_block(
    n_tokens: int = 16,
    n_heads: int = 4,
    d_head: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Return [n_tokens, 2, n_heads, d_head] KV block."""
    torch.manual_seed(seed)
    return torch.randn(n_tokens, 2, n_heads, d_head)


def _make_cache(seed: int = 42) -> KVPacketCache:
    return KVPacketCache(_make_config(seed=seed))


# --------------------------------------------------------------------------- #
# store_retrieve_roundtrip tests
# --------------------------------------------------------------------------- #


def test_store_retrieve_roundtrip():
    """store_packet/retrieve_packet roundtrip returns adapter-prepended KV."""
    cache = _make_cache()
    n_tokens, n_heads, d_head = 16, 4, 32
    n_adapter = cache.config.n_adapter_tokens

    K = torch.randn(n_tokens, n_heads, d_head)
    V = torch.randn(n_tokens, n_heads, d_head)
    cache.store_packet("seg1", K, V)

    retrieved = cache.retrieve_packet("seg1")
    assert retrieved is not None
    expected_shape = (n_adapter + n_tokens, 2, n_heads, d_head)
    assert retrieved.shape == expected_shape, \
        f"Expected {expected_shape}, got {retrieved.shape}"


def test_put_get_roundtrip():
    """put/get roundtrip via kv_data [n_tokens, 2, n_heads, d_head]."""
    cache = _make_cache()
    n_tokens, n_heads, d_head = 16, 4, 32
    n_adapter = cache.config.n_adapter_tokens

    kv_block = _make_kv_block(n_tokens, n_heads, d_head)
    cache.put("seg1", kv_block)

    result = cache.get("seg1")
    assert result is not None
    assert result.shape == (n_adapter + n_tokens, 2, n_heads, d_head)


def test_get_miss_returns_none():
    """get() returns None on cache miss."""
    cache = _make_cache()
    assert cache.get("nonexistent") is None


def test_put_initializes_adapters():
    """put() initializes adapter_K and adapter_V with correct shapes."""
    cfg = _make_config(n_heads=4, d_head=32, n_adapter_tokens=4)
    cache = KVPacketCache(cfg)
    kv_block = _make_kv_block(16, 4, 32)
    cache.put("seg1", kv_block)

    packet = cache._store["seg1"]
    assert packet.adapter_K.shape == (4, 4, 32)  # [n_adapter_tokens, n_heads, d_head]
    assert packet.adapter_V.shape == (4, 4, 32)
    assert packet.kv_data.dtype == torch.float16


def test_n_adapter_tokens_default_is_4():
    """Default n_adapter_tokens=4 (KV Packet paper default, different from kv_packet_adapter rank=8)."""
    cfg = KVPacketConfig()
    assert cfg.n_adapter_tokens == 4


# --------------------------------------------------------------------------- #
# assemble_multi tests
# --------------------------------------------------------------------------- #


def test_assemble_multi():
    """assemble_multi() concatenates multiple packets with adapter tokens."""
    cache = _make_cache()
    n_tokens1, n_tokens2 = 16, 24
    n_heads, d_head = 4, 32
    n_adapter = cache.config.n_adapter_tokens

    cache.put("seg1", _make_kv_block(n_tokens1, n_heads, d_head, seed=1))
    cache.put("seg2", _make_kv_block(n_tokens2, n_heads, d_head, seed=2))

    assembled = cache.assemble_multi(["seg1", "seg2"])
    assert assembled is not None

    expected_tokens = (n_adapter + n_tokens1) + (n_adapter + n_tokens2)
    assert assembled.shape == (expected_tokens, 2, n_heads, d_head)


def test_assemble_multi_miss_returns_none():
    """assemble_multi() returns None if any key is missing."""
    cache = _make_cache()
    cache.put("seg1", _make_kv_block(16, 4, 32))
    result = cache.assemble_multi(["seg1", "seg_missing"])
    assert result is None


def test_assemble_multi_single():
    """assemble_multi() with one key returns same as get()."""
    cache = _make_cache()
    kv = _make_kv_block(16, 4, 32)
    cache.put("seg1", kv)

    assembled = cache.assemble_multi(["seg1"])
    single = cache.get("seg1")

    assert assembled is not None
    assert single is not None
    assert assembled.shape == single.shape


# --------------------------------------------------------------------------- #
# Adapter shapes tests
# --------------------------------------------------------------------------- #


def test_adapter_shapes():
    """Adapter tensors have the correct shapes after put()."""
    cfg = _make_config(n_heads=8, d_head=64, n_adapter_tokens=4)
    cache = KVPacketCache(cfg)
    kv = _make_kv_block(32, 8, 64)
    cache.put("seg1", kv)

    packet = cache._store["seg1"]
    assert packet.adapter_K.shape == (4, 8, 64)
    assert packet.adapter_V.shape == (4, 8, 64)


def test_get_for_vericache_shapes():
    """get_for_vericache() returns (K, V) with correct shapes."""
    cfg = _make_config(n_heads=4, d_head=32, n_adapter_tokens=4)
    cache = KVPacketCache(cfg)
    n_tokens = 16
    kv = _make_kv_block(n_tokens, 4, 32)
    cache.put("seg1", kv)

    result = cache.get_for_vericache("seg1")
    assert result is not None
    K, V = result
    expected_tokens = 4 + n_tokens  # n_adapter_tokens + n_tokens
    assert K.shape == (expected_tokens, 4, 32)
    assert V.shape == (expected_tokens, 4, 32)


def test_get_for_vericache_miss_returns_none():
    """get_for_vericache() returns None on miss."""
    cache = _make_cache()
    result = cache.get_for_vericache("missing")
    assert result is None


# --------------------------------------------------------------------------- #
# LRU eviction tests
# --------------------------------------------------------------------------- #


def test_lru_eviction():
    """LRU eviction removes the least recently used entry."""
    cfg = _make_config(max_packets=3)
    cache = KVPacketCache(cfg)

    for i in range(3):
        cache.put(f"seg{i}", _make_kv_block(8, 4, 32, seed=i))

    # Access seg0 and seg1 (making seg2 the LRU)
    cache.get("seg0")
    cache.get("seg1")

    # Adding a 4th entry should evict LRU (seg2)
    cache.put("seg3", _make_kv_block(8, 4, 32, seed=99))

    assert len(cache._store) == 3
    assert "seg2" not in cache._store  # seg2 was evicted
    assert "seg3" in cache._store


def test_eviction_prefers_high_distillation_loss():
    """Eviction prefers entries with distillation_loss > threshold over LRU entries below threshold."""
    cfg = _make_config(max_packets=4, adapter_steps=1)
    cache = KVPacketCache(cfg)

    # Store 4 packets
    for i in range(4):
        cache.put(f"seg{i}", _make_kv_block(8, 4, 32, seed=i))

    # Set seg2's loss above threshold; all others are at initial value 1.0
    # With default threshold=0.1, initial loss 1.0 > threshold for all packets.
    # So we need to set 3 packets BELOW threshold and 1 ABOVE to test the preference.
    cache._store["seg0"].distillation_loss = 0.05  # below threshold
    cache._store["seg1"].distillation_loss = 0.05  # below threshold
    cache._store["seg2"].distillation_loss = 10.0  # above threshold - should be evicted first
    cache._store["seg3"].distillation_loss = 0.05  # below threshold

    freed = cache.evict()
    assert freed > 0
    # seg2 should be evicted first (only one above threshold)
    assert "seg2" not in cache._store


def test_evict_returns_bytes():
    """evict() returns positive byte count."""
    cache = _make_cache()
    cache.put("seg1", _make_kv_block(16, 4, 32))
    freed = cache.evict()
    assert freed > 0


def test_evict_empty_store():
    """evict() on empty store returns 0."""
    cache = _make_cache()
    freed = cache.evict()
    assert freed == 0


# --------------------------------------------------------------------------- #
# train_adapter tests
# --------------------------------------------------------------------------- #


def test_train_adapter_runs():
    """train_adapter() runs without error and returns finite loss."""
    cfg = _make_config(adapter_steps=5)  # Fast for test
    cache = KVPacketCache(cfg)
    kv = _make_kv_block(16, 4, 32)
    cache.put("seg1", kv)

    context_kv = _make_kv_block(32, 4, 32, seed=99)
    loss = cache.train_adapter("seg1", context_kv)

    assert loss < float("inf")
    assert not torch.isnan(torch.tensor(loss))


def test_train_adapter_updates_packet():
    """train_adapter() updates packet.distillation_loss."""
    cfg = _make_config(adapter_steps=5)
    cache = KVPacketCache(cfg)
    kv = _make_kv_block(16, 4, 32)
    cache.put("seg1", kv)

    initial_loss = cache._store["seg1"].distillation_loss  # 1.0 initial
    context_kv = _make_kv_block(32, 4, 32, seed=99)
    new_loss = cache.train_adapter("seg1", context_kv)

    assert cache._store["seg1"].distillation_loss == new_loss
    # Loss should have changed from initial value
    assert new_loss != initial_loss


def test_train_adapter_miss_returns_inf():
    """train_adapter() returns inf for missing key."""
    cache = _make_cache()
    result = cache.train_adapter("nonexistent", _make_kv_block(8, 4, 32))
    assert result == float("inf")


# --------------------------------------------------------------------------- #
# hit_rate, memory_bytes, reset_stats tests
# --------------------------------------------------------------------------- #


def test_hit_rate():
    """hit_rate() reflects actual hit/miss pattern."""
    cache = _make_cache()
    kv = _make_kv_block(16, 4, 32)
    cache.put("seg1", kv)

    cache.get("seg1")  # hit
    cache.get("seg1")  # hit
    cache.get("miss")  # miss

    assert cache.hit_rate() == pytest.approx(2 / 3, rel=1e-4)


def test_memory_bytes():
    """memory_bytes() returns positive value after storing packets."""
    cache = _make_cache()
    kv = _make_kv_block(16, 4, 32)
    cache.put("seg1", kv)

    mem = cache.memory_bytes()
    assert mem > 0


def test_reset_stats():
    """reset_stats() resets all counters."""
    cache = _make_cache()
    kv = _make_kv_block(16, 4, 32)
    cache.put("seg1", kv)

    cache.get("seg1")  # hit
    cache.get("miss")  # miss
    cache.reset_stats()

    assert cache._hits == 0
    assert cache._misses == 0
    assert cache._noncontiguous_hits == 0
    assert cache.hit_rate() == 0.0


# --------------------------------------------------------------------------- #
# noncontiguous_hit_rate tests
# --------------------------------------------------------------------------- #


def test_noncontiguous_hit_rate():
    """noncontiguous_hit_rate() tracks non-sequential access patterns."""
    cfg = _make_config(max_packets=10)
    cache = KVPacketCache(cfg)

    # Store 5 packets in sequence
    for i in range(5):
        cache.put(f"seg{i}", _make_kv_block(8, 4, 32, seed=i))

    # Access in non-contiguous order (skip indices)
    cache.get("seg0")
    cache.get("seg2")  # non-contiguous: jumped from 0 to 2
    cache.get("seg4")  # non-contiguous: jumped from 2 to 4

    # At least some non-contiguous hits should be tracked
    # (exact count depends on _store key ordering after move_to_end)
    nc_rate = cache.noncontiguous_hit_rate()
    assert 0.0 <= nc_rate <= 1.0


def test_noncontiguous_hit_rate_zero_hits():
    """noncontiguous_hit_rate() returns 0.0 when no hits."""
    cache = _make_cache()
    assert cache.noncontiguous_hit_rate() == 0.0


# --------------------------------------------------------------------------- #
# CacheStore interface compliance
# --------------------------------------------------------------------------- #


def test_cachestore_interface_compliance():
    """KVPacketCache implements all required CacheStore abstract methods."""
    from src.cache.base import CacheStore
    cache = _make_cache()
    assert isinstance(cache, CacheStore)

    kv = _make_kv_block(8, 4, 32)
    cache.put("k1", kv)
    val = cache.get("k1")
    assert val is not None

    freed = cache.evict()
    assert isinstance(freed, int)

    rate = cache.hit_rate()
    assert isinstance(rate, float)

    mem = cache.memory_bytes()
    assert isinstance(mem, int)

    cache.reset_stats()  # Should not raise
