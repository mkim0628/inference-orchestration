"""Unit tests for AMPDAdapShotLazyLoadPipeline (Activity B).

Covers:
  - CacheStore interface: put/get/evict/hit_rate/memory_bytes/reset_stats
  - Stage 1 resolve_segments: returns SegmentMeta, no KV tensors
  - Segment put/get, non-contiguous hit detection
  - AdapShot RoPE reencoding: correctness, identity at same position, shape preservation
  - Async load_and_reencode_segment
"""

import asyncio

import pytest
import torch

from src.cache.ampd_adapshot_lazy_pipeline import AMPDAdapShotLazyLoadPipeline, LazyPipelineConfig
from src.scheduler.ampd_lazy_segment_fetch import SegmentMeta


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_pipeline(chunk_size: int = 4, max_entries: int = 100) -> AMPDAdapShotLazyLoadPipeline:
    cfg = LazyPipelineConfig(chunk_size=chunk_size, max_entries=max_entries, d_head=8, seed=42)
    return AMPDAdapShotLazyLoadPipeline(cfg)


def _token_ids(n: int = 12) -> list:
    return list(range(n))


def _kv(n_tokens: int = 4, d_head: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n_tokens, d_head)


# ------------------------------------------------------------------ #
# CacheStore interface                                                 #
# ------------------------------------------------------------------ #

def test_cachestore_interface() -> None:
    pipeline = _make_pipeline()
    key = "test_key"
    value = _kv()

    # put / get
    pipeline.put(key, value)
    result = pipeline.get(key)
    assert result is not None
    assert result.shape == value.shape

    # hit_rate
    assert pipeline.hit_rate() > 0.0

    # memory_bytes
    assert pipeline.memory_bytes() > 0

    # evict
    freed = pipeline.evict()
    assert isinstance(freed, int)
    assert freed >= 0

    # reset_stats
    pipeline.reset_stats()
    assert pipeline.hit_rate() == 0.0


def test_resolve_segments_returns_metas_no_kv() -> None:
    pipeline = _make_pipeline(chunk_size=4)
    token_ids = _token_ids(12)

    # First, store a segment so there's a hit
    pipeline.put_segment(token_ids, chunk_idx=0, kv=_kv(), layer_idx=0)

    hit_metas, miss_indices = pipeline.resolve_segments(token_ids, layer_idx=0)

    # hit_metas should be SegmentMeta instances (no kv_tensor attribute)
    for meta in hit_metas:
        assert isinstance(meta, SegmentMeta)
        assert isinstance(meta.segment_id, str)
        assert not hasattr(meta, "kv_tensor")


def test_put_segment_and_get_segments() -> None:
    pipeline = _make_pipeline(chunk_size=4)
    token_ids = _token_ids(12)
    kv = _kv()

    pipeline.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)
    hits, misses = pipeline.get_segments(token_ids, layer_idx=0)

    hit_indices = {idx for idx, _ in hits}
    assert 0 in hit_indices


def test_noncontiguous_hit_detection() -> None:
    """chunk 0 miss + chunk 2 hit → non-contiguous hit counted."""
    pipeline = _make_pipeline(chunk_size=4)
    token_ids = _token_ids(16)  # 4 chunks of 4

    # Store only chunk 2 (skip chunk 0 and 1 → non-contiguous)
    pipeline.put_segment(token_ids, chunk_idx=2, kv=_kv(), layer_idx=0)

    hits, misses = pipeline.get_segments(token_ids, layer_idx=0)
    hit_indices = {idx for idx, _ in hits}
    assert 2 in hit_indices

    # Non-contiguous hit must be positive: chunk 2 is hit but chunk 0 (earlier) is miss
    assert pipeline._noncontiguous_hits > 0


def test_noncontiguous_hit_rate_above_threshold() -> None:
    """Multiple non-contiguous patterns → noncontiguous_hit_rate() > 0."""
    pipeline = _make_pipeline(chunk_size=4)

    for run in range(10):
        token_ids = list(range(run, run + 16))
        # Always store chunk 2 but not chunk 0
        pipeline.put_segment(token_ids, chunk_idx=2, kv=_kv(), layer_idx=0)
        pipeline.get_segments(token_ids, layer_idx=0)

    assert pipeline.noncontiguous_hit_rate() > 0.0


def test_adapshot_reencode_changes_kv() -> None:
    pipeline = _make_pipeline()
    kv = torch.randn(4, 8)
    reencoded = pipeline._adapshot_rope_reencode(kv, source_pos=0, target_pos=10)
    # source_pos != target_pos → values should differ
    assert not torch.allclose(kv.half(), reencoded, atol=1e-3)


def test_adapshot_reencode_identity_same_pos() -> None:
    """Same source and target position → reencoding is identity (no-op path)."""
    pipeline = _make_pipeline()
    kv = torch.randn(4, 8)
    # _adapshot_rope_reencode is called with delta=0 → angle=0 → cos=1, sin=0
    reencoded = pipeline._adapshot_rope_reencode(kv, source_pos=5, target_pos=5)
    # cos(0)=1, sin(0)=0 → rotated = [x1, x2] unchanged (up to FP16 rounding)
    assert torch.allclose(kv.half(), reencoded, atol=1e-3)


def test_adapshot_reencode_preserves_shape() -> None:
    pipeline = _make_pipeline()
    kv = torch.randn(8, 16)
    reencoded = pipeline._adapshot_rope_reencode(kv, source_pos=0, target_pos=5)
    assert reencoded.shape == kv.shape


def test_load_and_reencode_segment_async() -> None:
    pipeline = _make_pipeline(chunk_size=4)
    token_ids = _token_ids(12)
    pipeline.put_segment(token_ids, chunk_idx=0, kv=_kv(), layer_idx=0)
    seg_id = pipeline._store.chunk_key(token_ids, 0, layer_idx=0)

    async def _run():
        return await pipeline.load_and_reencode_segment(seg_id, source_position=0, target_position=5)

    result = asyncio.get_event_loop().run_until_complete(_run())
    assert result is not None
    assert isinstance(result, torch.Tensor)


def test_load_and_reencode_returns_none_on_miss() -> None:
    pipeline = _make_pipeline()

    async def _run():
        return await pipeline.load_and_reencode_segment(
            "nonexistent_segment_id", source_position=0, target_position=0
        )

    result = asyncio.get_event_loop().run_until_complete(_run())
    assert result is None


def test_memory_bytes_increases_on_put() -> None:
    pipeline = _make_pipeline()
    before = pipeline.memory_bytes()
    pipeline.put("new_key", torch.randn(8, 8))
    after = pipeline.memory_bytes()
    assert after > before
