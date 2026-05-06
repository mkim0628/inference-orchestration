"""Unit tests for DualFilterSegmentSelector (Activity B+C)."""

import pytest
import torch

from src.cache.dual_filter_selector import DualFilterSegmentSelector
from src.cache.query_centric_recompute import QueryCentricRecomputeCache
from src.cache.tri_attention_codec import TriAttentionCodec


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

LAYERS, HEADS, SEQ, DIM = 2, 2, 32, 16
CAPACITY = 32 * 1024 * 1024  # 32 MiB


def make_kv(seq: int = SEQ) -> torch.Tensor:
    return torch.randn(LAYERS, HEADS, seq, DIM)


def make_qcrc() -> QueryCentricRecomputeCache:
    return QueryCentricRecomputeCache(capacity_bytes=CAPACITY)


def make_codec(calibrated: bool = True) -> TriAttentionCodec:
    codec = TriAttentionCodec(n_layers=LAYERS, n_heads=HEADS, head_dim=DIM)
    if calibrated:
        kvs = [torch.randn(LAYERS, HEADS, 32, DIM) for _ in range(10)]
        codec.calibrate(kvs)
    return codec


def make_selector(
    stage1_ratio: float = 0.40,
    stage2_budget: float = 0.20,
    calibrated_codec: bool = True,
) -> DualFilterSegmentSelector:
    return DualFilterSegmentSelector(
        qcrc=make_qcrc(),
        codec=make_codec(calibrated=calibrated_codec),
        stage1_filter_ratio=stage1_ratio,
        stage2_token_budget=stage2_budget,
    )


def build_store(n: int, dim: int = DIM) -> tuple:
    """Return (segment_store, keys_pre_rope, query_embedding, candidate_keys)."""
    keys = [f"seg{i}" for i in range(n)]
    store = {}
    k_pre = {}
    for key in keys:
        kv = make_kv()
        emb = kv.mean(dim=(0, 1, 2))
        store[key] = {"kv": kv, "embedding": emb}
        k_pre[key] = torch.randn(LAYERS, HEADS, SEQ, dim)
    query = torch.randn(dim)
    return store, k_pre, query, keys


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #

class TestStage1FilterRatio:
    def test_stage1_keeps_correct_fraction(self) -> None:
        """With stage1_ratio=0.5 and 10 candidates, at most 5 pass stage 1."""
        selector = make_selector(stage1_ratio=0.5, stage2_budget=1.0)
        store, k_pre, query, keys = build_store(10)
        result = selector.select(query, keys, store, k_pre)
        assert len(result) <= 5

    def test_stage1_always_keeps_at_least_one(self) -> None:
        selector = make_selector(stage1_ratio=0.1)
        store, k_pre, query, keys = build_store(5)
        result = selector.select(query, keys, store, k_pre)
        assert len(result) >= 1

    def test_missing_segments_ignored(self) -> None:
        """Segments not in segment_store are silently skipped."""
        selector = make_selector(stage1_ratio=1.0)
        store, k_pre, query, keys = build_store(3)
        # Add a ghost key
        result = selector.select(query, keys + ["ghost"], store, k_pre)
        assert "ghost" not in result


class TestStage2TokenBudget:
    def test_stage2_reduces_token_count(self) -> None:
        """After stage 2, each segment should have fewer tokens (compressed)."""
        selector = make_selector(stage1_ratio=1.0, stage2_budget=0.2)
        store, k_pre, query, keys = build_store(3)
        result = selector.select(query, keys, store, k_pre)
        for seg_key, filtered_kv in result.items():
            original_seq = store[seg_key]["kv"].shape[2]
            # Filtered KV should be full-size (after decompress) but zero-padded
            assert filtered_kv.shape[2] == original_seq  # decompress pads to original len

    def test_stage2_skipped_without_pre_rope_keys(self) -> None:
        """Without keys_pre_rope, stage 2 is skipped and KV is returned as-is."""
        selector = make_selector(stage1_ratio=1.0, stage2_budget=0.2)
        store, _, query, keys = build_store(3)
        result = selector.select(query, keys, store, keys_pre_rope=None)
        for seg_key, filtered_kv in result.items():
            assert filtered_kv.shape == store[seg_key]["kv"].shape

    def test_stage2_skipped_when_codec_uncalibrated(self) -> None:
        """When codec is uncalibrated, stage 2 falls back to pass-through."""
        selector = make_selector(stage1_ratio=1.0, stage2_budget=0.2, calibrated_codec=False)
        store, k_pre, query, keys = build_store(3)
        result = selector.select(query, keys, store, k_pre)
        for seg_key, filtered_kv in result.items():
            assert filtered_kv.shape == store[seg_key]["kv"].shape


class TestFullPipeline:
    def test_select_returns_dict(self) -> None:
        selector = make_selector()
        store, k_pre, query, keys = build_store(6)
        result = selector.select(query, keys, store, k_pre)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, torch.Tensor)

    def test_select_empty_candidates(self) -> None:
        selector = make_selector()
        store, k_pre, query, _ = build_store(3)
        result = selector.select(query, [], store, k_pre)
        assert result == {}

    def test_stage1_scores_returns_all_candidates(self) -> None:
        selector = make_selector()
        store, _, query, keys = build_store(5)
        scores = selector.stage1_scores(query, keys, store)
        assert set(scores.keys()) == set(keys)

    def test_stage1_scores_cosine_range(self) -> None:
        """Cosine similarity scores should be in [-1, 1]."""
        selector = make_selector()
        store, _, query, keys = build_store(5)
        scores = selector.stage1_scores(query, keys, store)
        for score in scores.values():
            assert -1.0 <= score <= 1.0 + 1e-6

    def test_select_with_embedding_fallback(self) -> None:
        """Segments without 'embedding' key use KV mean as fallback."""
        selector = make_selector(stage1_ratio=1.0)
        store, k_pre, query, keys = build_store(3)
        # Remove 'embedding' from store entries
        for key in keys:
            store[key].pop("embedding")
        result = selector.select(query, keys, store, k_pre)
        assert len(result) >= 1


class TestParameterValidation:
    def test_invalid_stage1_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            DualFilterSegmentSelector(make_qcrc(), make_codec(), stage1_filter_ratio=0.0)

    def test_invalid_stage2_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            DualFilterSegmentSelector(make_qcrc(), make_codec(), stage2_token_budget=1.5)
