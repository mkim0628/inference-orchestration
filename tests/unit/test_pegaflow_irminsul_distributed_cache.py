"""Unit tests for PegaFlowIrminsulDistributedSegmentCache (Activity B-1)."""

import pytest
import torch

from src.cache.base import CacheStore
from src.cache.irminsul_mla_segment_cache import IrminsulMLAConfig, IrminsulMLASegmentCache
from src.cache.pegaflow_kv_connector import MockPegaFlowConnector, PegaFlowConnectorConfig
from src.cache.pegaflow_irminsul_distributed_cache import (
    DistributedSegmentCacheConfig,
    IrminsulKVEntry,
    PegaFlowIrminsulDistributedSegmentCache,
)
from src.scheduler.pegaflow_rdma_router import (
    PegaFlowRDMARouterConfig,
    PegaFlowRDMACrossNodeRouter,
    PeerRegistry,
)


def _make_distributed_cache(
    local_max_entries: int = 100,
) -> PegaFlowIrminsulDistributedSegmentCache:
    irminsul_config = IrminsulMLAConfig(avg_chunk_size=4, min_chunk_size=2, max_chunk_size=16, max_entries=50)
    local_irminsul = IrminsulMLASegmentCache(irminsul_config)

    pf_config = PegaFlowConnectorConfig(use_mock=True, seed=42)
    pegaflow_local = MockPegaFlowConnector(pf_config)

    router_config = PegaFlowRDMARouterConfig(
        peer_nodes_config_path="configs/pegaflow_peer_nodes.yaml",
        bloom_filter_capacity=1000,
        bloom_filter_error_rate=0.01,
        rdma_reuse_discount=0.8,
        seed=42,
    )
    peer_registry = PeerRegistry(router_config)
    rdma_router = PegaFlowRDMACrossNodeRouter(
        pegaflow_local, peer_registry, router_config, rdma_bandwidth_gbps=200.0
    )

    dist_config = DistributedSegmentCacheConfig(
        local_max_entries=local_max_entries, avg_chunk_size=4, seed=42
    )
    return PegaFlowIrminsulDistributedSegmentCache(
        local_irminsul, pegaflow_local, rdma_router, dist_config
    )


def _make_entry(n_tokens: int = 4, d_c: int = 8, d_r: int = 8, source_pos: int = 0) -> IrminsulKVEntry:
    return IrminsulKVEntry(
        segment_id=b"seg_test",
        c_kv_tensor=torch.randn(n_tokens, d_c),
        k_r_tensor=torch.randn(n_tokens, d_r),
        source_position=source_pos,
    )


def test_cache_store_interface_all_methods():
    """All CacheStore abstract methods must be implemented."""
    cache = _make_distributed_cache()
    assert isinstance(cache, CacheStore)
    assert callable(cache.put)
    assert callable(cache.get)
    assert callable(cache.evict)
    assert callable(cache.hit_rate)
    assert callable(cache.memory_bytes)
    assert callable(cache.reset_stats)


def test_get_distributed_local_hard_hit():
    cache = _make_distributed_cache()
    seg_id = b"local_seg_001"
    c_kv = torch.randn(4, 8)
    k_r = torch.randn(4, 8)
    entry = IrminsulKVEntry(
        segment_id=seg_id,
        c_kv_tensor=c_kv,
        k_r_tensor=k_r,
        source_position=0,
    )
    # Store directly in the local entry store
    cache._entry_store[seg_id] = entry

    result, hit_type = cache.get_distributed(seg_id, target_position=0)
    assert hit_type == "local_hard_hit"
    assert result is not None


def test_get_distributed_pegaflow_local_hit():
    cache = _make_distributed_cache()
    seg_id = b"pegaflow_local_seg"
    t = torch.randn(4, 8)
    # Store in PegaFlow local only (not in local entry store)
    cache._pegaflow_local.put(seg_id.hex(), t)

    result, hit_type = cache.get_distributed(seg_id, target_position=0)
    assert hit_type == "pegaflow_local_hit"
    assert result is not None


def test_get_distributed_miss_all_layers():
    cache = _make_distributed_cache()
    seg_id = b"never_stored_segment"
    result, hit_type = cache.get_distributed(seg_id, target_position=0)
    assert hit_type == "miss"
    assert result is None


def test_delta_rotation_c_kv_unchanged():
    """c_KV component must remain unchanged regardless of δ."""
    cache = _make_distributed_cache()
    n_tokens, d_c, d_r = 4, 8, 8
    c_kv = torch.randn(n_tokens, d_c)
    k_r = torch.randn(n_tokens, d_r)
    entry = _make_entry(n_tokens=n_tokens, d_c=d_c, d_r=d_r, source_pos=10)
    entry.c_kv_tensor = c_kv.clone()
    entry.k_r_tensor = k_r.clone()

    # δ = 0 → everything unchanged
    result_delta0 = cache.apply_delta_rotation(entry, target_position=10)
    c_kv_out = result_delta0[:, :d_c]
    assert torch.allclose(c_kv_out, c_kv, atol=1e-5)

    # δ ≠ 0 → c_KV portion still unchanged
    result_delta5 = cache.apply_delta_rotation(entry, target_position=15)
    c_kv_out_shifted = result_delta5[:, :d_c]
    assert torch.allclose(c_kv_out_shifted, c_kv, atol=1e-5)


def test_delta_rotation_k_r_changes_with_delta():
    """k_r component must change when δ ≠ 0."""
    cache = _make_distributed_cache()
    n_tokens, d_c, d_r = 4, 8, 8
    entry = _make_entry(n_tokens=n_tokens, d_c=d_c, d_r=d_r, source_pos=0)
    entry.k_r_tensor = torch.randn(n_tokens, d_r)

    result_no_delta = cache.apply_delta_rotation(entry, target_position=0)
    result_with_delta = cache.apply_delta_rotation(entry, target_position=50)

    k_r_no_delta = result_no_delta[:, d_c:]
    k_r_with_delta = result_with_delta[:, d_c:]

    # k_r should differ when δ ≠ 0
    assert not torch.allclose(k_r_no_delta, k_r_with_delta, atol=1e-4)


def test_hit_rate_breakdown_sums_to_one():
    cache = _make_distributed_cache()
    # Record various hit types
    cache._metrics.record("local_hard_hit")
    cache._metrics.record("pegaflow_local_hit")
    cache._metrics.record("rdma_remote_hit")
    cache._metrics.record("miss")

    breakdown = cache.distributed_hit_rate_breakdown()
    total = (
        breakdown["local_hard_hit_rate"]
        + breakdown["pegaflow_local_hit_rate"]
        + breakdown["rdma_remote_hit_rate"]
        + breakdown["miss_rate"]
    )
    assert abs(total - 1.0) < 1e-6


def test_distributed_hit_rate_above_local_only():
    """Combined hit rate must be >= local-only hit rate."""
    cache = _make_distributed_cache()
    cache._metrics.record("local_hard_hit")
    cache._metrics.record("pegaflow_local_hit")
    cache._metrics.record("miss")

    breakdown = cache.distributed_hit_rate_breakdown()
    assert breakdown["distributed_hit_rate"] >= breakdown["local_hard_hit_rate"]


def test_evict_offloads_to_pegaflow():
    """evict() must put the evicted entry into PegaFlow local."""
    cache = _make_distributed_cache()
    seg_id = b"evict_me"
    entry = IrminsulKVEntry(
        segment_id=seg_id,
        c_kv_tensor=torch.randn(4, 8),
        k_r_tensor=torch.randn(4, 8),
        source_position=0,
    )
    cache._entry_store[seg_id] = entry

    before_pf = cache._pegaflow_local.memory_bytes()
    freed = cache.evict()

    # Freed bytes must be positive
    assert freed > 0
    # PegaFlow local should now have the entry
    after_pf = cache._pegaflow_local.memory_bytes()
    assert after_pf > before_pf
