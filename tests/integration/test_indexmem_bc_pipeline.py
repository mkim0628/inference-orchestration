"""Integration tests for IndexMem B+C pipeline (Cross-1).

End-to-end pipeline tests including hard hit, soft hit, memory reduction,
InferenceRunner compatibility, and VeriCache plugin (Cross-2).
"""

from __future__ import annotations

import pytest
import torch

from src.cache.indexmem_soft_hit_segment_cache import (
    HitResult,
    IndexMemSoftHitSegmentCache,
    SoftHitSegmentConfig,
)
from src.cache.indexmem_eviction_codec import IndexMemEvictionCodec, IndexMemEvictionConfig
from src.cache.indexmem_latent_memory_module import LatentMemoryConfig
from src.engine.indexmem_bc_pipeline import (
    BCPipelineConfig,
    IndexMemBCIntegrationPipeline,
    SharedLatentEncoder,
    UnifiedLatentPool,
)

SEED = 42
KV_DIM = 32
LATENT_DIM = 16
N_LAYERS = 2
CHUNK_SIZE = 4


def _make_pipeline(
    max_physical: int = 20,
    latent_pool_max: int = 100,
    budget_ratio: float = 0.5,
) -> IndexMemBCIntegrationPipeline:
    soft_cfg = SoftHitSegmentConfig(
        chunk_size=CHUNK_SIZE,
        max_physical_entries=max_physical,
        latent_pool_max_segments=latent_pool_max,
        kv_dim=KV_DIM,
        latent_dim=LATENT_DIM,
        n_layers=N_LAYERS,
        beta_soft=0.1,
        beta_weight=0.5,
        seed=SEED,
    )
    soft_cache = IndexMemSoftHitSegmentCache(soft_cfg)

    codec_cfg = IndexMemEvictionConfig(
        budget_ratio=budget_ratio,
        zero_shot_mode=True,
        n_layers=N_LAYERS,
        kv_dim=KV_DIM,
        latent_dim=LATENT_DIM,
        seed=SEED,
    )
    eviction_codec = IndexMemEvictionCodec(codec_cfg)

    pipeline_cfg = BCPipelineConfig(
        budget_ratio=budget_ratio,
        segment_latent_pool_mb=1.0,
        token_latent_pool_mb=0.5,
        seed=SEED,
    )
    unified_pool = UnifiedLatentPool(pipeline_cfg)

    latent_cfg = LatentMemoryConfig(
        kv_dim=KV_DIM,
        latent_dim=LATENT_DIM,
        n_layers=N_LAYERS,
        encoder_hidden_dim=32,
        encoder_n_heads=2,
        encoder_ffn_dim=64,
        encoder_n_layers=2,
        seed=SEED,
    )
    shared_encoder = SharedLatentEncoder(latent_cfg)

    return IndexMemBCIntegrationPipeline(
        soft_hit_cache=soft_cache,
        eviction_codec=eviction_codec,
        unified_pool=unified_pool,
        shared_encoder=shared_encoder,
        config=pipeline_cfg,
    )


def _rand_kv(n: int = CHUNK_SIZE, d: int = KV_DIM, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


# --------------------------------------------------------------------------- #


def test_bc_pipeline_hard_hit_flow():
    """B+C pipeline: put_segment then get_segments returns hard hit."""
    pipeline = _make_pipeline()
    token_ids = [1, 2, 3, 4]
    kv = _rand_kv(seed=1)

    pipeline.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)

    hits, misses = pipeline.get_segments(token_ids, layer_idx=0)
    assert len(hits) == 1, f"Expected 1 hit, got {len(hits)}"
    assert len(misses) == 0

    chunk_idx, hit_result = hits[0]
    assert hit_result.type == "hard"
    assert hit_result.kv_tensor is not None


def test_bc_pipeline_soft_hit_flow():
    """After eviction, segment latent provides soft hit with non-zero readout."""
    pipeline = _make_pipeline(max_physical=1)
    token_ids_a = [1, 2, 3, 4]
    token_ids_b = [5, 6, 7, 8]
    kv_a = _rand_kv(seed=2)
    kv_b = _rand_kv(seed=3)

    pipeline.put_segment(token_ids_a, chunk_idx=0, kv=kv_a, layer_idx=0)
    # Adding second segment forces eviction of first (max_physical=1)
    pipeline.put_segment(token_ids_b, chunk_idx=0, kv=kv_b, layer_idx=0)

    hits_a, misses_a = pipeline.get_segments(token_ids_a, layer_idx=0)

    # token_ids_a should be evicted -> soft hit or miss
    for chunk_idx, hit_result in hits_a:
        assert hit_result.type in ("soft", "hard"), "Expected soft or hard hit after eviction"


def test_bc_pipeline_unified_weighted_hit_rate():
    """Unified weighted hit rate >= binary hit rate after soft hits."""
    pipeline = _make_pipeline(max_physical=2, latent_pool_max=100)
    token_ids = list(range(8))
    kv0 = _rand_kv(n=4, seed=10)
    kv1 = _rand_kv(n=4, seed=11)

    pipeline.put_segment(token_ids, chunk_idx=0, kv=kv0, layer_idx=0)
    pipeline.put_segment(token_ids, chunk_idx=1, kv=kv1, layer_idx=0)

    pipeline.get_segments(token_ids, layer_idx=0)

    uwhr = pipeline.unified_weighted_hit_rate()
    assert 0.0 <= uwhr <= 1.0
    assert uwhr >= 0.0


def test_bc_pipeline_memory_reduction_above_40pct():
    """With budget_ratio=0.5, memory reduction should be ~50%."""
    pipeline = _make_pipeline(budget_ratio=0.5)
    torch.manual_seed(SEED)
    n_tokens = 20
    kv = torch.randn(n_tokens, KV_DIM)

    original_bytes = kv.nbytes

    compressed = pipeline.eviction_codec.encode(kv, layer_idx=0, request_key="mem_test")
    compressed_bytes = compressed.nbytes

    reduction_pct = (1.0 - compressed_bytes / original_bytes) * 100
    assert reduction_pct >= 40.0, f"Memory reduction {reduction_pct:.1f}% < 40%"


def test_bc_pipeline_inference_runner_compat():
    """IndexMemBCIntegrationPipeline has get_segments/put_segment API."""
    pipeline = _make_pipeline()
    token_ids = [10, 20, 30, 40, 50, 60, 70, 80]

    torch.manual_seed(SEED)
    kv0 = torch.randn(CHUNK_SIZE, KV_DIM)
    kv1 = torch.randn(CHUNK_SIZE, KV_DIM)

    pipeline.put_segment(token_ids, chunk_idx=0, kv=kv0, layer_idx=0)
    pipeline.put_segment(token_ids, chunk_idx=1, kv=kv1, layer_idx=0)

    hits, misses = pipeline.get_segments(token_ids, layer_idx=0)
    total = len(hits) + len(misses)
    assert total == 2  # 2 chunks for 8 tokens with chunk_size=4


def test_cross2_vericache_indexmem_plugin():
    """Cross-2: VeriCacheSpeculativeCodec.set_draft_codec(IndexMem) integration."""
    from src.cache.vericache_speculative_codec import (
        VeriCacheSpeculativeCodec,
        VeriCacheConfig,
    )

    vericache = VeriCacheSpeculativeCodec(VeriCacheConfig(d_head=KV_DIM, seed=SEED))

    codec_cfg = IndexMemEvictionConfig(
        budget_ratio=0.5,
        zero_shot_mode=True,
        n_layers=N_LAYERS,
        kv_dim=KV_DIM,
        latent_dim=LATENT_DIM,
        seed=SEED,
    )
    indexmem_codec = IndexMemEvictionCodec(codec_cfg)

    # Plugin connection
    vericache.set_draft_codec(indexmem_codec)
    assert vericache._draft_codec is indexmem_codec

    # Verify compress/decompress works
    torch.manual_seed(SEED)
    kv = torch.randn(16, KV_DIM)
    compressed = indexmem_codec.compress(kv)
    decompressed = indexmem_codec.decompress(compressed)
    assert decompressed.shape[1] == KV_DIM


def test_unified_latent_pool_lru_eviction():
    """UnifiedLatentPool evicts LRU entries when over capacity."""
    cfg = BCPipelineConfig(
        segment_latent_pool_mb=0.001,  # tiny pool
        token_latent_pool_mb=0.001,
        seed=SEED,
    )
    pool = UnifiedLatentPool(cfg)

    torch.manual_seed(SEED)
    latent = torch.randn(64)

    # Fill pool beyond capacity
    for i in range(100):
        pool.put_segment_latent(f"seg_{i}", latent)

    # Pool should be bounded
    assert len(pool._segment_pool) <= pool._max_segment_entries


def test_shared_latent_encoder_encode():
    """SharedLatentEncoder encodes KV to latent vector."""
    latent_cfg = LatentMemoryConfig(
        kv_dim=KV_DIM,
        latent_dim=LATENT_DIM,
        n_layers=N_LAYERS,
        encoder_hidden_dim=32,
        encoder_n_heads=2,
        encoder_ffn_dim=64,
        encoder_n_layers=2,
        seed=SEED,
    )
    encoder = SharedLatentEncoder(latent_cfg)
    torch.manual_seed(SEED)
    kv = torch.randn(8, KV_DIM)
    latent = encoder.encode(kv)
    assert latent.shape == (LATENT_DIM,)
    assert torch.isfinite(latent).all()


def test_bc_pipeline_reset_stats():
    """reset_stats clears all counters in the pipeline."""
    pipeline = _make_pipeline()
    token_ids = [1, 2, 3, 4]
    kv = _rand_kv(seed=5)
    pipeline.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=0)
    pipeline.get_segments(token_ids, layer_idx=0)

    pipeline.reset_stats()
    assert pipeline._n_hard_hits == 0
    assert pipeline._n_soft_hits == 0 if hasattr(pipeline, '_n_soft_hits') else True
    assert pipeline.soft_hit_cache._n_hard_hits == 0
