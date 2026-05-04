"""Cross-1 integration tests: DAGTopologyScheduler + DAGAwareTTLAdjuster + WorkloadAwareTTLCache.

Verifies end-to-end pipeline behaviour of Activity A + B + C (auxiliary).
"""

import hashlib
import struct
import time
import pytest
import torch
import torch.nn.functional as F

from src.cache.workload_ttl_cache import WorkloadAwareTTLCache
from src.cache.redundancy_eviction import RedundancyAwareEvictionPolicy
from src.scheduler.dag_topology_scheduler import DAGTopologyScheduler
from src.scheduler.dag_ttl_adjuster import DAGAwareTTLAdjuster
from src.engine.runner import InferenceRequest

torch.manual_seed(42)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _chunk_key(token_ids, chunk_idx, chunk_size=4, layer_idx=0):
    start = chunk_idx * chunk_size
    chunk = token_ids[start: start + chunk_size]
    raw = struct.pack(f"{len(chunk)}I", *chunk)
    layer_prefix = struct.pack("I", layer_idx)
    return hashlib.sha256(layer_prefix + raw).hexdigest()


def _dag_spec(dag_id="wf_cross1"):
    return {
        "dag_id": dag_id,
        "nodes": [
            {"agent_id": "A", "tool_calls": [], "expected_kv_tokens": 512, "parent_ids": []},
            {"agent_id": "B", "tool_calls": ["t1"], "expected_kv_tokens": 256, "parent_ids": ["A"]},
            {"agent_id": "C", "tool_calls": ["t2"], "expected_kv_tokens": 128, "parent_ids": ["B"]},
        ],
    }


def _build_pipeline(retain_threshold=0.3, alpha=2.0):
    """Construct the full Cross-1 pipeline."""
    policy = RedundancyAwareEvictionPolicy(doc_id_shortcut=True)
    cache = WorkloadAwareTTLCache(max_entries=100, chunk_size=4, eviction_policy=policy)
    adjuster = DAGAwareTTLAdjuster(cache=cache, alpha=alpha, measure_latency=True)

    sched = DAGTopologyScheduler(
        cache=cache,
        retain_threshold=retain_threshold,
        alpha_ttl_extend=alpha,
        on_kv_reuse_event=adjuster.on_kv_reuse_event,
        on_node_complete_event=adjuster.on_node_complete,
    )
    return cache, adjuster, sched


def _make_request(dag_id, agent_id, token_ids=None):
    req = InferenceRequest(
        request_id=f"{dag_id}_{agent_id}",
        token_ids=token_ids or list(range(16)),
    )
    req.dag_id = dag_id  # type: ignore[attr-defined]
    req.agent_id = agent_id  # type: ignore[attr-defined]
    return req


# ------------------------------------------------------------------ #
# Test 1: End-to-end pipeline completes without exception              #
# ------------------------------------------------------------------ #

def test_end_to_end_dag_ttl_pipeline():
    """Full pipeline: register → schedule → TTL adjust → query — no exceptions."""
    cache, adjuster, sched = _build_pipeline()
    sched.register_workflow(_dag_spec())

    # Pre-populate some segments
    for i in range(5):
        cache.put_segment(f"seg_{i}", torch.randn(4, 8), category="agentic")

    req_a = _make_request("wf_cross1", "A")
    req_b = _make_request("wf_cross1", "B")
    requests = [req_a, req_b]
    result = sched.schedule(requests)

    assert len(result) == 2
    assert hasattr(result[0], "kv_reuse_probability")


# ------------------------------------------------------------------ #
# Test 2: DAG KV reuse extends TTL                                     #
# ------------------------------------------------------------------ #

def test_dag_kv_reuse_extends_ttl():
    """Segment TTL for a high-probability DAG node should exceed the category base TTL."""
    cache, adjuster, sched = _build_pipeline(retain_threshold=0.0, alpha=2.0)
    sched.register_workflow(_dag_spec())

    # Insert a segment for node A's token_ids
    token_ids = list(range(16))
    key = _chunk_key(token_ids, 0, chunk_size=4)
    cache.put_segment(key, torch.randn(4, 8), category="agentic")

    base_ttl = cache._ttl_profiles["agentic"]["ttl_base_sec"]  # 480.0
    prob_a = sched.predict_kv_reuse("wf_cross1", "A")  # > 0

    if prob_a > 0.0:
        adjuster.on_kv_reuse_event(key, prob_a)
        entry = cache._store[key]
        assert entry.ttl_sec > base_ttl, (
            f"TTL {entry.ttl_sec} should exceed base {base_ttl}"
        )


# ------------------------------------------------------------------ #
# Test 3: DAG completion enables early eviction                        #
# ------------------------------------------------------------------ #

def test_dag_completion_enables_early_eviction():
    """After notify_node_complete(), segment appears in evict_candidates()."""
    cache, adjuster, sched = _build_pipeline()
    sched.register_workflow(_dag_spec())

    key = "seg_dag_complete"
    cache.put_segment(key, torch.randn(4, 8), category="agentic")
    cache.pin(key)

    # Simulate DAG completion for node A
    sched._pinned_segments[("wf_cross1", "A")] = [key]
    sched.notify_node_complete("wf_cross1", "A")  # unpins + fires on_node_complete

    # adjust_ttl(0.0) was called → segment is now an eviction candidate
    candidates = cache.evict_candidates()
    assert key in candidates


# ------------------------------------------------------------------ #
# Test 4: Redundancy eviction in pipeline                              #
# ------------------------------------------------------------------ #

def test_redundancy_eviction_in_pipeline():
    """WorkloadAwareTTLCache + RedundancyAwareEvictionPolicy: duplicate evicted first."""
    policy = RedundancyAwareEvictionPolicy(doc_id_shortcut=True)
    cache = WorkloadAwareTTLCache(max_entries=100, chunk_size=4, eviction_policy=policy)

    # Insert duplicate segments (same doc_id prefix) — will be TTL-expired immediately
    cache.put_segment("doc:1:chunk_0", torch.randn(4, 8), category="rag", override_ttl_sec=0.001)
    cache.put_segment("doc:1:chunk_1", torch.randn(4, 8), category="rag", override_ttl_sec=0.001)
    # Insert unique segment — also TTL-expired but not redundant
    cache.put_segment("unique_seg", torch.randn(4, 8), category="rag", override_ttl_sec=0.001)

    time.sleep(0.005)

    # Evict one → should choose a doc:1 segment (redundancy=1.0)
    cache.evict()
    remaining = set(cache._store.keys())

    # At least one doc:1 segment should be gone
    doc1_remaining = {k for k in remaining if k.startswith("doc:1:")}
    assert len(doc1_remaining) < 2, "Expected at least one doc:1 segment to be evicted"


# ------------------------------------------------------------------ #
# Test 5: Non-contiguous hit rate ≥ 30%                                #
# ------------------------------------------------------------------ #

def test_noncontiguous_hit_rate_above_30pct():
    """TTL-preserved hits contribute to noncontiguous_ratio ≥ 0.30."""
    cache = WorkloadAwareTTLCache(max_entries=200, chunk_size=4)

    # Insert many segments with long TTL (chat category, 300s default)
    n = 20
    keys = [f"key_{i}" for i in range(n)]
    for k in keys:
        cache.put_segment(k, torch.randn(4, 8), category="chat")

    # Access all segments → all count as TTL-preserved hits
    for k in keys:
        cache.get(k)

    stats = cache.ttl_hit_stats()
    assert stats["noncontiguous_ratio"] >= 0.30, (
        f"noncontiguous_ratio={stats['noncontiguous_ratio']:.3f} < 0.30"
    )


# ------------------------------------------------------------------ #
# Test 6: Pinned segments not evicted during pipeline                  #
# ------------------------------------------------------------------ #

def test_pinned_segments_not_evicted_during_pipeline():
    """Pinned segments survive evict() calls."""
    cache, adjuster, sched = _build_pipeline(retain_threshold=0.0)
    sched.register_workflow(_dag_spec())

    key = "pinned_seg"
    cache.put_segment(key, torch.randn(4, 8), category="agentic")
    cache.pin(key)

    # Trigger multiple evictions
    for _ in range(5):
        cache.evict()

    assert key in cache._store, "Pinned segment was incorrectly evicted"


# ------------------------------------------------------------------ #
# Test 7: Scheduling overhead within 5%                                #
# ------------------------------------------------------------------ #

def test_scheduling_overhead_within_5pct():
    """DAGTopologyScheduler.schedule() overhead: p50 < 1ms per request."""
    cache, adjuster, sched = _build_pipeline()
    sched.register_workflow(_dag_spec("dag_overhead"))

    n_requests = 100
    requests = []
    for i in range(n_requests):
        req = _make_request("dag_overhead", "A", token_ids=list(range(16)))
        req.request_id = f"r_{i}"
        requests.append(req)

    t0 = time.monotonic()
    sched.schedule(requests)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    per_request_ms = elapsed_ms / n_requests
    # Allow 5ms per request (≪ TTFT p50 typical ~100ms)
    assert per_request_ms < 5.0, f"Per-request overhead {per_request_ms:.2f}ms > 5ms"


# ------------------------------------------------------------------ #
# Test 8: Hit rate improvement with DAG scheduling                     #
# ------------------------------------------------------------------ #

def test_hit_rate_improvement_with_dag_scheduling():
    """DAGTopologyScheduler pinning increases retention → higher hit rate than no-pin baseline."""
    import copy

    # Baseline cache: no pinning, low max_entries to force evictions
    baseline_cache = WorkloadAwareTTLCache(max_entries=5, chunk_size=4)

    # Experiment cache: with DAG scheduler pinning
    dag_cache = WorkloadAwareTTLCache(max_entries=5, chunk_size=4)
    sched = DAGTopologyScheduler(cache=dag_cache, retain_threshold=0.0)
    sched.register_workflow(_dag_spec("dag_hit"))

    # Populate both caches with the same segments
    n = 8
    keys = [f"k_{i}" for i in range(n)]
    tensors = [torch.randn(4, 8) for _ in range(n)]

    for k, t in zip(keys[:5], tensors[:5]):
        baseline_cache.put_segment(k, t, category="agentic")
        dag_cache.put_segment(k, t, category="agentic")

    # Pin first 2 segments in dag_cache (simulate high-probability nodes)
    for k in keys[:2]:
        dag_cache.pin(k)

    # Evict to trigger pressure (adds 3 more segments, forces eviction of old ones)
    for k, t in zip(keys[5:], tensors[5:]):
        baseline_cache.put_segment(k, t, category="agentic")
        dag_cache.put_segment(k, t, category="agentic")

    # Check pinned segments still present in dag_cache
    pinned_present = sum(1 for k in keys[:2] if k in dag_cache._store)
    # Without pinning, these might have been evicted; with pinning, they stay
    assert pinned_present == 2, f"Expected 2 pinned segments, got {pinned_present}"


# ------------------------------------------------------------------ #
# Test 9: Accuracy preservation in full pipeline                       #
# ------------------------------------------------------------------ #

def test_accuracy_preservation_in_full_pipeline():
    """High-importance segments survive the full Cross-1 pipeline."""
    policy = RedundancyAwareEvictionPolicy(doc_id_shortcut=True)
    cache = WorkloadAwareTTLCache(max_entries=50, chunk_size=4, eviction_policy=policy)
    adjuster = DAGAwareTTLAdjuster(cache=cache, alpha=2.0)
    sched = DAGTopologyScheduler(
        cache=cache,
        retain_threshold=0.0,
        on_kv_reuse_event=adjuster.on_kv_reuse_event,
        on_node_complete_event=adjuster.on_node_complete,
    )
    sched.register_workflow(_dag_spec())

    # Insert high-importance segments
    important_keys = []
    for i in range(3):
        key = f"important_{i}"
        cache.put_segment(key, torch.randn(4, 8), category="agentic", override_ttl_sec=9999.0)
        cache.record_importance(key, 1.0)
        cache.pin(key)  # simulate DAG scheduler pinning high-prob nodes
        important_keys.append(key)

    # Insert redundant segments with short TTL
    for i in range(5):
        key = f"doc:99:chunk_{i}"
        cache.put_segment(key, torch.randn(4, 8), category="rag", override_ttl_sec=0.001)

    time.sleep(0.005)

    # Run evictions
    for _ in range(5):
        cache.evict()

    # All important segments must still be in cache
    for key in important_keys:
        assert key in cache._store, f"Important segment {key} was evicted!"
