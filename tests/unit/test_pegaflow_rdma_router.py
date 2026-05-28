"""Unit tests for PegaFlowRDMACrossNodeRouter and PeerRegistry (Activity A-2)."""

import pytest
import torch

from src.cache.pegaflow_kv_connector import MockPegaFlowConnector, PegaFlowConnectorConfig
from src.scheduler.pegaflow_rdma_router import (
    PegaFlowRDMARouterConfig,
    PegaFlowRDMACrossNodeRouter,
    PeerNodeEntry,
    PeerRegistry,
)


def _router_config(**kwargs) -> PegaFlowRDMARouterConfig:
    defaults = dict(
        peer_nodes_config_path="configs/pegaflow_peer_nodes.yaml",
        bloom_filter_capacity=1000,
        bloom_filter_error_rate=0.01,
        bloom_sync_interval_ms=500.0,
        rdma_reuse_discount=0.8,
        seed=42,
    )
    defaults.update(kwargs)
    return PegaFlowRDMARouterConfig(**defaults)


def _make_local_connector() -> MockPegaFlowConnector:
    cfg = PegaFlowConnectorConfig(use_mock=True, seed=42)
    return MockPegaFlowConnector(cfg)


def _make_router(rdma_bw: float = 200.0) -> PegaFlowRDMACrossNodeRouter:
    config = _router_config()
    peer_registry = PeerRegistry(config)
    local = _make_local_connector()
    return PegaFlowRDMACrossNodeRouter(local, peer_registry, config, rdma_bandwidth_gbps=rdma_bw)


def test_peer_registry_has_segment_false_for_unknown():
    config = _router_config()
    registry = PeerRegistry(config)
    # Register a peer manually
    registry._peers["node-x"] = PeerNodeEntry("node-x", "127.0.0.1", 9000)
    from src.scheduler.pegaflow_rdma_router import _BloomFilter
    registry._filters["node-x"] = _BloomFilter(1000, 0.01)

    seg = b"some_segment_not_registered"
    result = registry.has_segment("node-x", seg)
    # May have false positives but usually False for unregistered content
    # For a fresh filter, it must be False
    assert result is False


def test_peer_registry_has_segment_true_after_register():
    config = _router_config()
    registry = PeerRegistry(config)
    registry._peers["node-y"] = PeerNodeEntry("node-y", "127.0.0.1", 9001)
    from src.scheduler.pegaflow_rdma_router import _BloomFilter
    registry._filters["node-y"] = _BloomFilter(1000, 0.01)

    seg = b"known_segment_abc123"
    registry._filters["node-y"].add(seg)

    assert registry.has_segment("node-y", seg) is True


def test_route_returns_pegaflow_local_on_local_hit():
    router = _make_router()
    t = torch.tensor([1.0, 2.0, 3.0])
    seg_id = b"seg_local_hit"
    # Put into local connector
    router.local_connector.put(seg_id.hex(), t)
    result, hit_type = router.route_segment_request(seg_id, "local")
    assert hit_type == "pegaflow_local"
    assert result is not None
    assert torch.allclose(result, t)


def test_route_returns_miss_when_no_peers():
    router = _make_router()
    seg_id = b"seg_no_peers"
    result, hit_type = router.route_segment_request(seg_id, "local")
    assert hit_type == "miss"
    assert result is None


def test_rdma_reuse_decision_below_discount():
    """RDMA latency < recompute × discount → RDMA selected."""
    config = _router_config(rdma_reuse_discount=0.8)
    peer_registry = PeerRegistry(config)
    local = _make_local_connector()
    # Use very high bandwidth to make RDMA cheap
    router = PegaFlowRDMACrossNodeRouter(
        local, peer_registry, config, rdma_bandwidth_gbps=1_000_000.0
    )

    t = torch.tensor([1.0, 2.0])
    seg_id = b"seg_rdma_test"
    router.register_mock_remote_segment("peer-1", seg_id, t)

    result, hit_type = router.route_segment_request_with_size(
        seg_id,
        local_node_id="local",
        segment_size_bytes=1,             # tiny → very low RDMA cost
        segment_token_count=1_000_000,    # large → very high recompute cost
        gpu_throughput_tokens_per_ms=1.0,
    )
    assert hit_type == "rdma_remote"
    assert result is not None


def test_rdma_reuse_decision_above_discount():
    """RDMA latency >= recompute × discount → no RDMA, miss."""
    config = _router_config(rdma_reuse_discount=0.8)
    peer_registry = PeerRegistry(config)
    local = _make_local_connector()
    # Very low bandwidth → expensive RDMA
    router = PegaFlowRDMACrossNodeRouter(
        local, peer_registry, config, rdma_bandwidth_gbps=0.000001
    )

    t = torch.tensor([1.0, 2.0])
    seg_id = b"seg_rdma_expensive"
    router.register_mock_remote_segment("peer-1", seg_id, t)

    result, hit_type = router.route_segment_request_with_size(
        seg_id,
        local_node_id="local",
        segment_size_bytes=1_000_000_000,  # 1 GB → very high cost at low BW
        segment_token_count=1,             # trivial recompute
        gpu_throughput_tokens_per_ms=1.0,
    )
    assert hit_type == "miss"


def test_estimate_rdma_latency_formula():
    router = _make_router(rdma_bw=200.0)
    # 200 GB/s = 200e9 bytes/s = 200e6 bytes/ms
    # 1 MB / (200e6 bytes/ms) = 1e6 / 200e6 = 0.005 ms
    size = 1_000_000  # 1 MB
    latency = router.estimate_rdma_latency_ms(size, "peer")
    expected = size / (200.0 * 1e9 / 1000.0)
    assert abs(latency - expected) < 1e-9


def test_estimate_recompute_latency_formula():
    router = _make_router()
    latency = router.estimate_recompute_latency_ms(256, gpu_throughput_tokens_per_ms=2.0)
    assert abs(latency - 128.0) < 1e-9


def test_schedule_returns_input_unchanged():
    router = _make_router()
    requests = ["req1", "req2", "req3"]
    result = router.schedule(requests)
    assert result == requests
