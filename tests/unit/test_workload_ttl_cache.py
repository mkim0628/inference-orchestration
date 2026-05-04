"""Unit tests for WorkloadAwareTTLCache (Activity B)."""

import time
import torch
import pytest

from src.cache.workload_ttl_cache import WorkloadAwareTTLCache, TTLEntry


torch.manual_seed(42)


def _make_tensor(n: int = 4) -> torch.Tensor:
    return torch.randn(n, 8)


def _make_cache(**kwargs) -> WorkloadAwareTTLCache:
    defaults = {"max_entries": 50, "chunk_size": 4}
    defaults.update(kwargs)
    return WorkloadAwareTTLCache(**defaults)


# ------------------------------------------------------------------ #
# Basic put/get / TTL                                                  #
# ------------------------------------------------------------------ #

def test_put_get_exact_hit():
    cache = _make_cache()
    v = _make_tensor()
    cache.put_segment("k1", v, category="chat")
    result = cache.get("k1")
    assert result is not None
    assert result.shape == v.shape


def test_ttl_expiry_returns_miss():
    cache = _make_cache()
    v = _make_tensor()
    cache.put_segment("k1", v, category="chat", override_ttl_sec=0.001)
    time.sleep(0.005)
    assert cache.get("k1") is None


def test_miss_key_not_in_store():
    cache = _make_cache()
    assert cache.get("nonexistent") is None


# ------------------------------------------------------------------ #
# Category classification                                              #
# ------------------------------------------------------------------ #

def test_category_classification_code():
    cache = _make_cache()
    category = cache._classify_category("def foo(): pass")
    assert category == "code"


def test_category_classification_code_class():
    cache = _make_cache()
    assert cache._classify_category("class MyModel:") == "code"


def test_category_classification_rag():
    cache = _make_cache()
    assert cache._classify_category("retrieved document: blah") == "rag"


def test_category_classification_agentic():
    cache = _make_cache()
    assert cache._classify_category("tool_call: get_weather") == "agentic"


def test_category_default_chat():
    cache = _make_cache()
    assert cache._classify_category("hello world, how are you?") == "chat"


# ------------------------------------------------------------------ #
# TTL profiles                                                         #
# ------------------------------------------------------------------ #

def test_ttl_profiles_different_per_category():
    cache = _make_cache()
    code_ttl = cache._ttl_profiles["code"]["ttl_base_sec"]
    chat_ttl = cache._ttl_profiles["chat"]["ttl_base_sec"]
    agentic_ttl = cache._ttl_profiles["agentic"]["ttl_base_sec"]
    rag_ttl = cache._ttl_profiles["rag"]["ttl_base_sec"]
    # code > agentic > chat > rag
    assert code_ttl > agentic_ttl > chat_ttl > rag_ttl


# ------------------------------------------------------------------ #
# Pin / unpin                                                          #
# ------------------------------------------------------------------ #

def test_pin_prevents_eviction():
    cache = _make_cache(max_entries=2)
    v = _make_tensor()
    cache.put_segment("pinned", v, category="chat", override_ttl_sec=0.001)
    cache.pin("pinned")
    time.sleep(0.005)
    # Fill cache past capacity to trigger eviction
    cache.put_segment("extra1", _make_tensor(), category="chat")
    cache.put_segment("extra2", _make_tensor(), category="chat")
    # pinned should still be in _store
    assert "pinned" in cache._store


def test_unpin_allows_eviction():
    cache = _make_cache(max_entries=2)
    v = _make_tensor()
    cache.put_segment("k1", v, category="chat", override_ttl_sec=0.001)
    cache.pin("k1")
    cache.unpin("k1")
    time.sleep(0.005)
    # Now evict should remove k1 (TTL expired)
    cache.evict()
    assert "k1" not in cache._store


# ------------------------------------------------------------------ #
# adjust_ttl                                                           #
# ------------------------------------------------------------------ #

def test_adjust_ttl_extends():
    cache = _make_cache()
    v = _make_tensor()
    cache.put_segment("k1", v, category="chat", override_ttl_sec=0.001)
    cache.adjust_ttl("k1", 9999.0)
    time.sleep(0.005)
    # Should still be accessible
    assert cache.get("k1") is not None


def test_adjust_ttl_to_zero_immediate_eviction_candidate():
    cache = _make_cache()
    v = _make_tensor()
    cache.put_segment("k1", v, category="chat")
    cache.adjust_ttl("k1", 0.0)
    candidates = cache.evict_candidates()
    assert "k1" in candidates


# ------------------------------------------------------------------ #
# Non-contiguous hit rate proxy                                        #
# ------------------------------------------------------------------ #

def test_noncontiguous_hit_rate_ttl_based():
    """TTL-preserved hits contribute to noncontiguous_ratio ≥ 0.30."""
    cache = _make_cache()
    # Insert many segments with default TTL
    for i in range(10):
        v = _make_tensor()
        cache.put_segment(f"seg_{i}", v, category="chat")

    # Access all of them (all TTL-preserved hits)
    for i in range(10):
        cache.get(f"seg_{i}")

    stats = cache.ttl_hit_stats()
    assert stats["noncontiguous_ratio"] >= 0.30


# ------------------------------------------------------------------ #
# EMA TTL update                                                       #
# ------------------------------------------------------------------ #

def test_ema_ttl_update_on_hit():
    cache = _make_cache()
    v = _make_tensor()
    cache.put_segment("k1", v, category="chat")
    old_ttl = cache._ttl_profiles["chat"]["ttl_base_sec"]
    cache.record_hit("k1", is_ttl_preserved=True)
    new_ttl = cache._ttl_profiles["chat"]["ttl_base_sec"]
    # EMA update should have changed the profile TTL
    assert new_ttl != old_ttl


# ------------------------------------------------------------------ #
# Eviction ordering                                                    #
# ------------------------------------------------------------------ #

def test_evict_ttl_expired_first():
    cache = _make_cache(max_entries=10)
    v = _make_tensor()
    # Insert one expired and one live segment
    cache.put_segment("expired", v, category="chat", override_ttl_sec=0.001)
    cache.put_segment("live", v, category="chat", override_ttl_sec=9999.0)
    time.sleep(0.005)
    cache.evict()
    assert "expired" not in cache._store
    assert "live" in cache._store


def test_lru_fallback_when_no_expired():
    cache = _make_cache(max_entries=10)
    v = _make_tensor()
    cache.put_segment("oldest", v, category="chat", override_ttl_sec=9999.0)
    cache.put_segment("newer", v, category="chat", override_ttl_sec=9999.0)
    # Touch 'newer' to make it more recent
    cache.get("newer")
    cache.evict()
    # 'oldest' should be evicted (LRU fallback)
    assert "oldest" not in cache._store
    assert "newer" in cache._store


# ------------------------------------------------------------------ #
# CacheStore interface compliance                                       #
# ------------------------------------------------------------------ #

def test_cachestore_interface_compliance():
    cache = _make_cache()
    v = _make_tensor()
    cache.put("key1", v)
    result = cache.get("key1")
    assert result is not None
    freed = cache.evict()
    assert isinstance(freed, int)
    hr = cache.hit_rate()
    assert 0.0 <= hr <= 1.0
    mb = cache.memory_bytes()
    assert isinstance(mb, int)
    cache.reset_stats()


def test_reset_stats():
    cache = _make_cache()
    v = _make_tensor()
    cache.put_segment("k1", v, category="chat")
    cache.get("k1")
    cache.get("missing")
    cache.reset_stats()
    assert cache._exact_hits == 0
    assert cache._ttl_preserved_hits == 0
    assert cache._misses == 0


# ------------------------------------------------------------------ #
# chunk_key compatibility                                              #
# ------------------------------------------------------------------ #

def test_chunk_key_deterministic():
    from src.cache.segmented import SegmentedHashCache
    token_ids = list(range(16))
    cache_ttl = WorkloadAwareTTLCache(chunk_size=4)
    cache_seg = SegmentedHashCache(chunk_size=4)
    # Both caches should produce identical chunk keys
    for chunk_idx in range(4):
        key_ttl = cache_ttl.chunk_key(token_ids, chunk_idx, layer_idx=0)
        key_seg = cache_seg.chunk_key(token_ids, chunk_idx, layer_idx=0)
        assert key_ttl == key_seg, f"chunk_idx={chunk_idx}: {key_ttl} != {key_seg}"


def test_chunk_key_different_layers():
    cache = WorkloadAwareTTLCache(chunk_size=4)
    token_ids = list(range(8))
    key_l0 = cache.chunk_key(token_ids, 0, layer_idx=0)
    key_l1 = cache.chunk_key(token_ids, 0, layer_idx=1)
    assert key_l0 != key_l1
