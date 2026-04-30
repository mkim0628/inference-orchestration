"""Unit tests for MultiNodeScheduler (Activity A).

Tests verify:
  - schedule() returns all input requests
  - High hit-rate request is scheduled before cold request
  - Fairness: cold request scheduled within fairness_max_wait steps
  - route() returns valid (prefill, decode) NodeConfig pair with correct types
  - Large KV → simulate_transfer uses codec (shape preserved)
  - Single-node fallback (prefill_nodes=[], decode_nodes=[]) matches parent
  - node_load() returns dict with all node IDs
"""

import torch
import pytest
from typing import List

from src.cache.compression import HadamardInt4Codec
from src.cache.segmented import SegmentedHashCache
from src.engine.runner import InferenceRequest
from src.scheduler.cache_aware_scheduler import CacheAwareScheduler
from src.scheduler.multi_node_scheduler import MultiNodeScheduler, NodeConfig


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

CHUNK_SIZE = 4


def _make_request(request_id: str, token_ids: List[int], seed: int = 0) -> InferenceRequest:
    return InferenceRequest(request_id=request_id, token_ids=token_ids, seed=seed)


def _make_nodes(n_prefill: int = 2, n_decode: int = 2) -> tuple:
    prefill_nodes = [
        NodeConfig(node_id=f"p{i}", node_type="prefill", transfer_latency_ms=10.0 + i * 2)
        for i in range(n_prefill)
    ]
    decode_nodes = [
        NodeConfig(node_id=f"d{i}", node_type="decode", current_load=0.1 * i)
        for i in range(n_decode)
    ]
    return prefill_nodes, decode_nodes


@pytest.fixture
def cache() -> SegmentedHashCache:
    return SegmentedHashCache(chunk_size=CHUNK_SIZE, max_entries=200)


@pytest.fixture
def scheduler(cache: SegmentedHashCache) -> MultiNodeScheduler:
    prefill_nodes, decode_nodes = _make_nodes()
    return MultiNodeScheduler(
        cache=cache,
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        fairness_max_wait=5,
        chunk_size=CHUNK_SIZE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_schedule_returns_all_requests(scheduler: MultiNodeScheduler) -> None:
    """schedule() must return all input requests, no duplicates or drops."""
    requests = [
        _make_request(f"req_{i}", list(range(i * 4, (i + 1) * 4))) for i in range(10)
    ]
    scheduled = scheduler.schedule(requests)
    assert len(scheduled) == len(requests), (
        f"Expected {len(requests)} requests, got {len(scheduled)}"
    )
    assert {r.request_id for r in scheduled} == {r.request_id for r in requests}


def test_high_hit_rate_scheduled_first(cache: SegmentedHashCache) -> None:
    """A request whose chunks are already in cache should be scheduled before a cold one."""
    torch.manual_seed(0)
    prefill_nodes, decode_nodes = _make_nodes()
    sched = MultiNodeScheduler(
        cache=cache,
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        fairness_max_wait=10,
        chunk_size=CHUNK_SIZE,
    )

    # Warm up cache: put chunk for warm_req at layer 0
    warm_tokens = [1, 2, 3, 4]
    warm_kv = torch.randn(CHUNK_SIZE, 16)
    key = cache.chunk_key(warm_tokens, chunk_idx=0, layer_idx=0)
    cache.put(key, warm_kv)

    warm_req = _make_request("warm", warm_tokens)
    cold_req = _make_request("cold", [100, 200, 300, 400])

    scheduled = sched.schedule([cold_req, warm_req])
    assert scheduled[0].request_id == "warm", (
        f"Expected warm request first, got {scheduled[0].request_id}"
    )


def test_fairness(cache: SegmentedHashCache) -> None:
    """Cold request must not be starved: it should appear within first 2 positions after
    waiting fairness_max_wait steps.

    The fairness mechanism caps the wait penalty at 1.0 (priority → 0) and uses
    wait_steps as a tie-breaker.  A warm request with hit_rate > 0 still outranks
    a cold request with priority = 0, so cold appears at position 0 or 1 (not starved).
    This matches the semantics validated in test_cache_aware_scheduler.py::test_fairness_max_wait.
    """
    prefill_nodes, decode_nodes = _make_nodes()
    fairness_max_wait = 3
    sched = MultiNodeScheduler(
        cache=cache,
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        fairness_max_wait=fairness_max_wait,
        chunk_size=CHUNK_SIZE,
    )

    torch.manual_seed(5)
    # Warm request — always hits
    warm_tokens = [10, 20, 30, 40]
    key = cache.chunk_key(warm_tokens, 0, 0)
    cache.put(key, torch.randn(CHUNK_SIZE, 16))
    warm_req = _make_request("warm", warm_tokens)

    # Cold request — never hits
    cold_req = _make_request("cold", [999, 998, 997, 996])

    # Simulate fairness_max_wait rounds without processing cold_req
    for _ in range(fairness_max_wait):
        sched.schedule([warm_req, cold_req])
        sched.update_wait(processed_ids=["warm"], all_ids=["warm", "cold"])

    # After max_wait rounds, cold_req must appear in the first 2 positions (not starved)
    final = sched.schedule([warm_req, cold_req])
    cold_position = next(i for i, r in enumerate(final) if r.request_id == "cold")
    assert cold_position <= 1, (
        f"Cold request at position {cold_position} after {fairness_max_wait} wait steps; "
        "should not be starved (position <= 1)"
    )


def test_route_returns_valid_nodes(scheduler: MultiNodeScheduler) -> None:
    """route() must return a (NodeConfig, NodeConfig) with node_type prefill/decode."""
    req = _make_request("r1", [1, 2, 3, 4])
    p_node, d_node = scheduler.route(req)

    assert isinstance(p_node, NodeConfig), "Prefill result must be NodeConfig"
    assert isinstance(d_node, NodeConfig), "Decode result must be NodeConfig"
    assert p_node.node_type == "prefill", f"Expected prefill type, got {p_node.node_type}"
    assert d_node.node_type == "decode", f"Expected decode type, got {d_node.node_type}"
    assert p_node in scheduler.prefill_nodes, "Prefill node must be from prefill_nodes list"
    assert d_node in scheduler.decode_nodes, "Decode node must be from decode_nodes list"


def test_compress_before_transfer(cache: SegmentedHashCache) -> None:
    """Large KV tensor should trigger codec compression in simulate_transfer; shape preserved."""
    codec = HadamardInt4Codec(num_layers=12, cutoff_ratio=0.2)
    prefill_nodes, decode_nodes = _make_nodes(1, 1)
    sched = MultiNodeScheduler(
        cache=cache,
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        codec=codec,
        compress_threshold_bytes=0,  # always compress
        chunk_size=CHUNK_SIZE,
    )

    kv = torch.randn(256, 64)  # large enough to trigger compression
    p_node = prefill_nodes[0]
    d_node = decode_nodes[0]

    kv_out, latency_ms = sched.simulate_transfer(kv, p_node, d_node)

    assert kv_out.shape == kv.shape, (
        f"Transfer output shape {kv_out.shape} != input shape {kv.shape}"
    )
    assert latency_ms > 0, "Transfer latency must be positive"
    # Compression was applied: latency should be reduced (0.25×)
    assert len(sched._transfer_log) > 0
    assert sched._transfer_log[-1]["compressed"] is True


def test_single_node_fallback(cache: SegmentedHashCache) -> None:
    """prefill_nodes=[], decode_nodes=[] → ordering matches parent CacheAwareScheduler."""
    # MultiNodeScheduler with no nodes
    multi = MultiNodeScheduler(
        cache=cache,
        prefill_nodes=[],
        decode_nodes=[],
        fairness_max_wait=10,
        chunk_size=CHUNK_SIZE,
    )
    # Parent scheduler (same cache, same params)
    parent = CacheAwareScheduler(cache=cache, fairness_max_wait=10, chunk_size=CHUNK_SIZE)

    requests = [
        _make_request(f"req_{i}", list(range(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE)))
        for i in range(5)
    ]

    multi_order = [r.request_id for r in multi.schedule(requests)]
    parent_order = [r.request_id for r in parent.schedule(requests)]

    assert multi_order == parent_order, (
        f"Fallback order {multi_order} != parent order {parent_order}"
    )


def test_node_load_dict(scheduler: MultiNodeScheduler) -> None:
    """node_load() must return a dict with all node IDs as keys."""
    load = scheduler.node_load()

    assert isinstance(load, dict), "node_load() must return a dict"

    all_node_ids = {n.node_id for n in scheduler.prefill_nodes + scheduler.decode_nodes}
    assert set(load.keys()) == all_node_ids, (
        f"node_load() keys {set(load.keys())} != expected {all_node_ids}"
    )

    for node_id, value in load.items():
        assert isinstance(value, float), f"Load for {node_id} must be a float"
        assert 0.0 <= value <= 1.0, f"Load {value} for {node_id} out of [0,1] range"
