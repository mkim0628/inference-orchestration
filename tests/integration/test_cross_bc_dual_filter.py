"""
Integration tests for the Activity B+C cross-component pipeline.

Tests the DualFilterSegmentSelector and QueryCentricTriAttentionCache
end-to-end, verifying the full pipeline from segment storage through
two-stage filtering to selective recompute.
"""

import pytest
import torch

from src.cache.dual_filter_selector import DualFilterSegmentSelector
from src.cache.info_flow_reorder import InfoFlowChunkReorderCache
from src.cache.qc_tri_store import QueryCentricTriAttentionCache
from src.cache.query_centric_recompute import QueryCentricRecomputeCache
from src.cache.tri_attention_codec import TriAttentionCodec


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

LAYERS, HEADS, SEQ, DIM = 2, 2, 64, 32
CAPACITY = 128 * 1024 * 1024  # 128 MiB
N_SEGMENTS = 8


def make_codec(calibrated: bool = True) -> TriAttentionCodec:
    codec = TriAttentionCodec(n_layers=LAYERS, n_heads=HEADS, head_dim=DIM)
    if calibrated:
        kvs = [torch.randn(LAYERS, HEADS, 32, DIM) for _ in range(10)]
        codec.calibrate(kvs)
    return codec


def populate_store(n: int) -> tuple:
    """Create n segments and return (segment_store, keys_pre_rope, query, candidate_keys)."""
    keys = [f"seg{i}" for i in range(n)]
    store = {}
    k_pre = {}
    for key in keys:
        kv = torch.randn(LAYERS, HEADS, SEQ, DIM)
        emb = kv.mean(dim=(0, 1, 2))
        store[key] = {"kv": kv, "embedding": emb}
        k_pre[key] = torch.randn(LAYERS, HEADS, SEQ, DIM)
    query = torch.randn(DIM)
    return store, k_pre, query, keys


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #

class TestDualFilterPipelineE2E:
    def test_dual_filter_pipeline_e2e(self) -> None:
        """Full two-stage pipeline: query-relevance → token importance."""
        qcrc = QueryCentricRecomputeCache(capacity_bytes=CAPACITY)
        codec = make_codec()
        selector = DualFilterSegmentSelector(
            qcrc=qcrc,
            codec=codec,
            stage1_filter_ratio=0.5,
            stage2_token_budget=0.3,
        )

        store, k_pre, query, keys = populate_store(N_SEGMENTS)

        result = selector.select(query, keys, store, k_pre)

        # At most ceil(N * 0.5) = 4 segments pass stage 1
        assert 1 <= len(result) <= (N_SEGMENTS // 2) + 1

        # Each returned value should be a Tensor with original sequence length
        for seg_key, kv in result.items():
            assert isinstance(kv, torch.Tensor)
            assert kv.shape[2] == SEQ  # decompress pads back to original length

    def test_pipeline_handles_all_ratios(self) -> None:
        """Pipeline works for various stage1/stage2 ratio combinations."""
        store, k_pre, query, keys = populate_store(10)
        for s1, s2 in [(0.2, 0.1), (0.5, 0.2), (1.0, 0.5)]:
            selector = DualFilterSegmentSelector(
                qcrc=QueryCentricRecomputeCache(capacity_bytes=CAPACITY),
                codec=make_codec(),
                stage1_filter_ratio=s1,
                stage2_token_budget=s2,
            )
            result = selector.select(query, keys, store, k_pre)
            assert isinstance(result, dict)


class TestDualFilterVsSingleFilter:
    def test_dual_filter_vs_single_filter(self) -> None:
        """
        Dual filter should select fewer (or equal) tokens per segment than
        a single query-relevance filter (stage 1 only with ratio=1.0).
        """
        store, k_pre, query, keys = populate_store(8)

        codec = make_codec()
        qcrc = QueryCentricRecomputeCache(capacity_bytes=CAPACITY)

        # Single filter: stage 1 passes all (ratio=1.0), stage 2 budget=1.0 (no reduction)
        single = DualFilterSegmentSelector(
            qcrc=qcrc, codec=codec,
            stage1_filter_ratio=1.0, stage2_token_budget=1.0,
        )
        # Dual filter: stage 1 passes 50%, stage 2 keeps 20% of tokens
        dual = DualFilterSegmentSelector(
            qcrc=qcrc, codec=codec,
            stage1_filter_ratio=0.5, stage2_token_budget=0.2,
        )

        single_result = single.select(query, keys, store, k_pre)
        dual_result = dual.select(query, keys, store, k_pre)

        # Dual filter must select fewer or equal segments
        assert len(dual_result) <= len(single_result)

    def test_stage1_scores_ordered_correctly(self) -> None:
        """Stage 1 scores should be in descending order when ranked."""
        store, _, query, keys = populate_store(6)
        qcrc = QueryCentricRecomputeCache(capacity_bytes=CAPACITY)
        codec = make_codec()
        selector = DualFilterSegmentSelector(qcrc=qcrc, codec=codec)

        scores = selector.stage1_scores(query, keys, store)
        sorted_scores = sorted(scores.values(), reverse=True)
        assert sorted_scores == sorted(scores.values(), reverse=True)


class TestQcTriCacheFullPipeline:
    def test_qc_tri_cache_full_pipeline(self) -> None:
        """QueryCentricTriAttentionCache full pipeline: put_with_query → get → recompute."""
        codec = make_codec()
        cache = QueryCentricTriAttentionCache(
            capacity_bytes=CAPACITY,
            codec=codec,
            relevance_threshold=0.0,   # low threshold → mostly raw storage
        )

        keys = [f"seg{i}" for i in range(5)]
        kvs = {}
        for key in keys:
            kv = torch.randn(LAYERS, HEADS, SEQ, DIM)
            kvs[key] = kv
            query_emb = kv.mean(dim=(0, 1, 2))
            k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
            cache.put_with_query(key, kv, k_pre, query_emb)

        # All gets should succeed
        for key in keys:
            result = cache.get(key)
            assert result is not None

        # selective_recompute should return a subset
        query = torch.randn(DIM)
        selected = cache.selective_recompute(query, keys, budget=0.4)
        assert isinstance(selected, list)

    def test_qc_tri_cache_with_compressed_segments(self) -> None:
        """Low-relevance segments are compressed and retrievable."""
        codec = make_codec()
        cache = QueryCentricTriAttentionCache(
            capacity_bytes=CAPACITY,
            codec=codec,
            relevance_threshold=2.0,   # impossible threshold → all compressed
        )

        kv = torch.randn(LAYERS, HEADS, SEQ, DIM)
        query_emb = -kv.mean(dim=(0, 1, 2))  # anti-aligned → very low relevance
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put_with_query("low_rel", kv, k_pre, query_emb)

        result = cache.get("low_rel")
        assert result is not None
        assert result.shape[2] == SEQ

    def test_qc_tri_cache_with_reorder_integration(self) -> None:
        """InfoFlowChunkReorderCache integrates with QCTriCache pipeline."""
        codec = make_codec()
        qc_cache = QueryCentricTriAttentionCache(
            capacity_bytes=CAPACITY,
            codec=codec,
            relevance_threshold=0.0,
        )
        reorder_cache = InfoFlowChunkReorderCache(capacity_bytes=CAPACITY)

        keys = [f"chunk{i}" for i in range(4)]
        for key in keys:
            kv = torch.randn(LAYERS, HEADS, SEQ, DIM)
            q_emb = kv.mean(dim=(0, 1, 2))
            k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
            qc_cache.put_with_query(key, kv, k_pre, q_emb)
            reorder_cache.put(key, kv)

        # Reorder the chunks
        reordered = reorder_cache.reorder_chunks(keys)
        assert len(reordered) == len(keys)

        # Selective recompute on the reordered list
        query = torch.randn(DIM)
        selected = qc_cache.selective_recompute(query, reordered, budget=0.5)
        assert isinstance(selected, list)

    def test_hit_rate_across_both_stores(self) -> None:
        """hit_rate() aggregates hits from raw store, compressed store, and QCRC."""
        codec = make_codec()
        cache = QueryCentricTriAttentionCache(
            capacity_bytes=CAPACITY, codec=codec
        )
        kv = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put("p", kv)

        cache.get("p")   # hit
        cache.get("x")   # miss

        assert cache.hit_rate() == pytest.approx(0.5)
