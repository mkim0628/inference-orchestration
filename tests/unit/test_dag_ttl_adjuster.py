"""Unit tests for DAGAwareTTLAdjuster (Cross-1 adapter)."""

import time
import pytest
import torch

from src.cache.workload_ttl_cache import WorkloadAwareTTLCache
from src.scheduler.dag_ttl_adjuster import DAGAwareTTLAdjuster

torch.manual_seed(42)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_cache(category: str = "chat") -> WorkloadAwareTTLCache:
    return WorkloadAwareTTLCache(max_entries=50, chunk_size=4)


def _make_adjuster(cache=None, alpha: float = 2.0, measure_latency: bool = True) -> DAGAwareTTLAdjuster:
    if cache is None:
        cache = _make_cache()
    return DAGAwareTTLAdjuster(cache=cache, alpha=alpha, measure_latency=measure_latency)


def _put_segment(cache: WorkloadAwareTTLCache, key: str, category: str = "chat") -> None:
    v = torch.randn(4, 8)
    cache.put_segment(key, v, category=category)


# ------------------------------------------------------------------ #
# Test 1: on_kv_reuse_event extends TTL                                #
# ------------------------------------------------------------------ #

def test_on_kv_reuse_event_extends_ttl():
    """dag_reuse_probability=0.8, base_ttl=300, alpha=2.0 → adjusted_ttl=780."""
    cache = _make_cache()
    _put_segment(cache, "seg1", category="chat")
    # Verify initial TTL is the chat default (300s)
    assert cache._ttl_profiles["chat"]["ttl_base_sec"] == pytest.approx(300.0)

    adjuster = _make_adjuster(cache=cache, alpha=2.0)
    adjuster.on_kv_reuse_event("seg1", dag_reuse_probability=0.8)

    entry = cache._store["seg1"]
    expected_ttl = 300.0 * (1.0 + 0.8 * 2.0)  # = 780.0
    assert entry.ttl_sec == pytest.approx(expected_ttl, rel=1e-4)


# ------------------------------------------------------------------ #
# Test 2: on_node_complete sets TTL to 0                               #
# ------------------------------------------------------------------ #

def test_on_node_complete_sets_ttl_zero():
    """on_node_complete() calls adjust_ttl(key, 0.0)."""
    cache = _make_cache()
    _put_segment(cache, "seg2", category="agentic")

    adjuster = _make_adjuster(cache=cache)
    adjuster.on_node_complete("seg2")

    entry = cache._store["seg2"]
    assert entry.ttl_sec == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ #
# Test 3: on_node_complete calls unpin                                 #
# ------------------------------------------------------------------ #

def test_on_node_complete_calls_unpin():
    """on_node_complete() removes segment from _pinned set."""
    cache = _make_cache()
    _put_segment(cache, "seg3", category="agentic")
    cache.pin("seg3")
    assert "seg3" in cache._pinned

    adjuster = _make_adjuster(cache=cache)
    adjuster.on_node_complete("seg3")

    assert "seg3" not in cache._pinned


# ------------------------------------------------------------------ #
# Test 4: Latency is measured                                          #
# ------------------------------------------------------------------ #

def test_latency_measured():
    """With measure_latency=True, overhead_stats()['n_samples'] > 0 after events."""
    cache = _make_cache()
    _put_segment(cache, "seg4")
    adjuster = _make_adjuster(cache=cache, measure_latency=True)
    adjuster.on_kv_reuse_event("seg4", dag_reuse_probability=0.5)
    stats = adjuster.overhead_stats()
    assert stats["n_samples"] > 0


# ------------------------------------------------------------------ #
# Test 5: overhead p50 < 1ms                                           #
# ------------------------------------------------------------------ #

def test_overhead_p50_below_1ms():
    """100 on_kv_reuse_event() calls → p50 latency < 1.0ms."""
    cache = _make_cache()
    for i in range(100):
        _put_segment(cache, f"seg_{i}")

    adjuster = _make_adjuster(cache=cache, measure_latency=True)
    for i in range(100):
        adjuster.on_kv_reuse_event(f"seg_{i}", dag_reuse_probability=0.5)

    stats = adjuster.overhead_stats()
    assert stats["p50_ms"] < 1.0, f"p50={stats['p50_ms']:.3f}ms exceeded 1ms threshold"
    assert stats["n_samples"] == 100


# ------------------------------------------------------------------ #
# Test 6: Zero probability does not extend TTL                         #
# ------------------------------------------------------------------ #

def test_zero_probability_does_not_extend_ttl():
    """dag_reuse_probability=0.0 → adjusted_ttl = base_ttl × 1.0 (no change)."""
    cache = _make_cache()
    _put_segment(cache, "seg5", category="chat")
    entry = cache._store["seg5"]
    original_ttl = entry.ttl_sec

    adjuster = _make_adjuster(cache=cache, alpha=2.0)
    adjuster.on_kv_reuse_event("seg5", dag_reuse_probability=0.0)

    # base_ttl × (1 + 0.0 × 2.0) = base_ttl × 1.0
    # original_ttl was set from ttl_profiles["chat"]["ttl_base_sec"] = 300.0
    expected = 300.0 * 1.0
    assert entry.ttl_sec == pytest.approx(expected, rel=1e-4)


# ------------------------------------------------------------------ #
# Edge case: missing key                                               #
# ------------------------------------------------------------------ #

def test_on_kv_reuse_event_missing_key_no_error():
    """Event for a key not in cache should be a no-op without raising."""
    cache = _make_cache()
    adjuster = _make_adjuster(cache=cache)
    # Should not raise
    adjuster.on_kv_reuse_event("nonexistent_key", dag_reuse_probability=0.9)


def test_on_node_complete_missing_key_no_error():
    """Completing a node for a key not in cache should be a no-op without raising."""
    cache = _make_cache()
    adjuster = _make_adjuster(cache=cache)
    adjuster.on_node_complete("nonexistent_key")
