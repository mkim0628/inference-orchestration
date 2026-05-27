"""Unit tests for IndexMemLearnableIndexer (Activity C-1)."""

import os
import tempfile

import pytest
import torch

from src.cache.indexmem_learnable_indexer import (
    IndexMemLearnableIndexer,
    LearnableIndexerConfig,
)


def _make_indexer(zero_shot: bool = True, seed: int = 42) -> IndexMemLearnableIndexer:
    cfg = LearnableIndexerConfig(zero_shot_mode=zero_shot, seed=seed)
    return IndexMemLearnableIndexer(cfg)


def _make_inputs(n_tokens: int = 16, d_head: int = 32, seed: int = 42):
    torch.manual_seed(seed)
    k = torch.randn(n_tokens, d_head)
    v = torch.randn(n_tokens, d_head)
    query = torch.randn(d_head)
    positions = torch.arange(n_tokens, dtype=torch.float32)
    return k, v, query, positions


# --------------------------------------------------------------------------- #


def test_predict_output_shape():
    indexer = _make_indexer(zero_shot=True)
    k, v, q, pos = _make_inputs(16, 32)
    probs = indexer.predict(k, v, q, pos, current_position=16)
    assert probs.shape == (16,), f"Expected (16,), got {probs.shape}"
    assert probs.dtype == torch.float32
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_predict_zero_shot_mode():
    indexer = _make_indexer(zero_shot=True)
    k, v, q, pos = _make_inputs(8, 16)
    probs = indexer.predict(k, v, q, pos, current_position=8, segment_key="seg0")
    assert probs.shape == (8,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_predict_learned_mode():
    indexer = _make_indexer(zero_shot=False)
    k, v, q, pos = _make_inputs(8, 16)
    probs = indexer.predict(k, v, q, pos, current_position=8)
    assert probs.shape == (8,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_select_tokens_by_budget():
    indexer = _make_indexer()
    torch.manual_seed(0)
    probs = torch.rand(20)
    kept, evicted = indexer.select_tokens_by_budget(probs, budget_ratio=0.5)
    assert kept.shape[0] == 10
    assert evicted.shape[0] == 10
    assert kept.shape[0] + evicted.shape[0] == 20
    # No overlap
    kept_set = set(kept.tolist())
    evict_set = set(evicted.tolist())
    assert len(kept_set & evict_set) == 0


def test_select_tokens_by_budget_minimum_one():
    indexer = _make_indexer()
    probs = torch.rand(4)
    kept, evicted = indexer.select_tokens_by_budget(probs, budget_ratio=0.0)
    assert kept.shape[0] >= 1


def test_update_cumul_attn_ema():
    cfg = LearnableIndexerConfig(zero_shot_mode=True, ema_alpha_cumul_attn=0.5, seed=42)
    indexer = IndexMemLearnableIndexer(cfg)
    n = 8
    seg = "seg_ema"
    # Initialize
    _ = indexer._get_or_init_cumul_attn(seg, n)
    initial = indexer._cumul_attn[seg].clone()

    attn = torch.ones(n) * 0.8
    indexer.update_cumul_attn(seg, attn)
    updated = indexer._cumul_attn[seg]

    # EMA: 0.5 * 0.8 + 0.5 * 0.5 = 0.65
    expected = 0.5 * 0.8 + 0.5 * initial
    assert torch.allclose(updated, expected, atol=1e-5)


def test_select_tokens_deterministic():
    indexer1 = _make_indexer(seed=42)
    indexer2 = _make_indexer(seed=42)
    k, v, q, pos = _make_inputs(16, 32, seed=7)

    probs1 = indexer1.predict(k, v, q, pos, 16, "seg")
    kept1, _ = indexer1.select_tokens_by_budget(probs1, 0.5)

    probs2 = indexer2.predict(k, v, q, pos, 16, "seg")
    kept2, _ = indexer2.select_tokens_by_budget(probs2, 0.5)

    assert torch.equal(kept1, kept2)


def test_save_load_weights():
    indexer = _make_indexer(zero_shot=False, seed=42)
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        indexer.save_weights(path)
        indexer2 = _make_indexer(zero_shot=False, seed=99)  # different seed
        indexer2.load_weights(path)
        assert torch.allclose(indexer._w1, indexer2._w1, atol=1e-5)
        assert torch.allclose(indexer._b1, indexer2._b1, atol=1e-5)
        assert torch.allclose(indexer._w2, indexer2._w2, atol=1e-5)
        assert torch.allclose(indexer._b2, indexer2._b2, atol=1e-5)
    finally:
        os.unlink(path)


def test_predict_with_2d_query():
    indexer = _make_indexer(zero_shot=True)
    k, v, _, pos = _make_inputs(8, 16)
    query_2d = torch.randn(4, 16)  # n_q × d_head
    probs = indexer.predict(k, v, query_2d, pos, current_position=8)
    assert probs.shape == (8,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
