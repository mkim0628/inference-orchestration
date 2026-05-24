"""Unit tests for TriAttentionPreRoPEKVSelectorCodec (Activity C, highest priority).

Tests:
  - compute_concentration() numerical correctness
  - compute_dist_pref_scores() output shape [N]
  - select_kv() shapes, sorted kept_indices
  - is_reasoning_task budget routing
  - kv_pool_pressure extra reduction
  - detect_reasoning_task() tag detection
  - CacheStore full interface: put/get/evict/hit_rate/memory_bytes/reset_stats
  - get_importance_mask() bool mask shape [original_seq_len]
  - concentration_stats() dict return
  - memory_reduction_ratio() at budget=0.093 >= 0.85
"""

import torch
import pytest

from src.cache.triattention_pre_rope_kv_selector_codec import (
    TriAttentionPreRoPEKVSelectorCodec,
    TriAttentionSelectorConfig,
    SelectorKVEntry,
)


# ---- fixtures ----------------------------------------------------------------

def _make_codec(
    d_head: int = 64,
    n_kv_heads: int = 4,
    rope_base: float = 10000.0,
    kv_budget_ratio_reasoning: float = 0.093,
    kv_budget_ratio_default: float = 0.20,
    max_entries: int = 100,
    seed: int = 42,
) -> TriAttentionPreRoPEKVSelectorCodec:
    cfg = TriAttentionSelectorConfig(
        d_head=d_head,
        n_kv_heads=n_kv_heads,
        rope_base=rope_base,
        kv_budget_ratio_reasoning=kv_budget_ratio_reasoning,
        kv_budget_ratio_default=kv_budget_ratio_default,
        max_entries=max_entries,
        seed=seed,
    )
    return TriAttentionPreRoPEKVSelectorCodec(cfg)


def _random_qkv(n_q: int, n_kv: int, d_head: int, seed: int = 42):
    torch.manual_seed(seed)
    Q = torch.randn(n_q, d_head)
    K = torch.randn(n_kv, d_head)
    V = torch.randn(n_kv, d_head)
    return Q, K, V


# ---- compute_concentration ---------------------------------------------------

def test_compute_concentration_aligned_vectors_near_one() -> None:
    """Identical-direction vectors: concentration ≈ 1.0."""
    d = 64
    direction = torch.randn(d)
    direction = direction / direction.norm()
    vecs = direction.unsqueeze(0).expand(16, -1) * 2.0  # same direction, same norm
    conc = TriAttentionPreRoPEKVSelectorCodec.compute_concentration(vecs)
    assert conc >= 0.90, f"aligned vectors: conc={conc:.4f} < 0.90"


def test_compute_concentration_uniform_sphere_near_zero() -> None:
    """Uniformly distributed unit vectors: concentration ≈ 0.0."""
    torch.manual_seed(0)
    d = 64
    # Large number of random unit vectors → mean ≈ 0 by central limit theorem
    vecs = torch.randn(1024, d)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    conc = TriAttentionPreRoPEKVSelectorCodec.compute_concentration(vecs)
    assert conc < 0.20, f"uniform sphere: conc={conc:.4f} >= 0.20"


def test_compute_concentration_returns_float_in_0_1() -> None:
    torch.manual_seed(7)
    vecs = torch.randn(10, 64)
    conc = TriAttentionPreRoPEKVSelectorCodec.compute_concentration(vecs)
    assert isinstance(conc, float)
    assert 0.0 <= conc <= 1.0


# ---- compute_dist_pref_scores ------------------------------------------------

def test_compute_dist_pref_scores_shape() -> None:
    """Output shape should be [N]."""
    codec = _make_codec(d_head=64)
    N = 128
    torch.manual_seed(1)
    mu_q = torch.randn(64)
    key_positions = torch.arange(N, dtype=torch.int64)
    scores = codec.compute_dist_pref_scores(mu_q, key_positions, pos_q=0)
    assert scores.shape == (N,), f"expected ({N},), got {scores.shape}"


def test_compute_dist_pref_scores_finite() -> None:
    """All scores should be finite."""
    codec = _make_codec(d_head=64)
    torch.manual_seed(2)
    mu_q = torch.randn(64)
    key_positions = torch.arange(256, dtype=torch.int64)
    scores = codec.compute_dist_pref_scores(mu_q, key_positions, pos_q=10)
    assert torch.isfinite(scores).all(), "dist_pref scores contain non-finite values"


# ---- select_kv ---------------------------------------------------------------

def test_select_kv_output_shapes() -> None:
    """K_sel, V_sel shapes match n_keep; kept_indices length equals n_keep."""
    codec = _make_codec(d_head=64, kv_budget_ratio_default=0.20)
    N, n_q, d = 100, 8, 64
    Q, K, V = _random_qkv(n_q, N, d)
    K_sel, V_sel, kept_idx, conc_q, conc_k, budget = codec.select_kv(
        Q, K, V, is_reasoning_task=False
    )
    n_keep = max(1, int(N * 0.20))
    assert K_sel.shape[0] == n_keep, f"K_sel shape[0]={K_sel.shape[0]} != {n_keep}"
    assert V_sel.shape[0] == n_keep
    assert kept_idx.shape[0] == n_keep


def test_select_kv_kept_indices_sorted() -> None:
    """kept_indices must be sorted in ascending order."""
    codec = _make_codec(d_head=64, kv_budget_ratio_default=0.30)
    Q, K, V = _random_qkv(4, 50, 64, seed=10)
    _, _, kept_idx, _, _, _ = codec.select_kv(Q, K, V)
    assert (kept_idx[1:] >= kept_idx[:-1]).all(), "kept_indices not sorted"


def test_select_kv_reasoning_task_budget() -> None:
    """is_reasoning_task=True should use kv_budget_ratio_reasoning=0.093."""
    codec = _make_codec(d_head=64, kv_budget_ratio_reasoning=0.093, kv_budget_ratio_default=0.20)
    N = 256
    Q, K, V = _random_qkv(8, N, 64, seed=3)
    K_sel, _, _, _, _, budget = codec.select_kv(Q, K, V, is_reasoning_task=True)
    assert abs(budget - 0.093) < 1e-9, f"reasoning budget={budget} != 0.093"
    n_keep_expected = max(1, int(N * 0.093))
    assert K_sel.shape[0] == n_keep_expected, (
        f"K_sel.shape[0]={K_sel.shape[0]} != {n_keep_expected}"
    )


def test_select_kv_default_task_budget() -> None:
    """is_reasoning_task=False should use kv_budget_ratio_default=0.20."""
    codec = _make_codec(d_head=64, kv_budget_ratio_reasoning=0.093, kv_budget_ratio_default=0.20)
    N = 256
    Q, K, V = _random_qkv(8, N, 64, seed=4)
    _, _, _, _, _, budget = codec.select_kv(Q, K, V, is_reasoning_task=False)
    assert abs(budget - 0.20) < 1e-9, f"default budget={budget} != 0.20"


def test_select_kv_high_pressure_extra_reduction() -> None:
    """kv_pool_pressure=0.85 should apply extra 10% reduction to budget."""
    codec = _make_codec(
        d_head=64,
        kv_budget_ratio_default=0.20,
        kv_budget_ratio_reasoning=0.093,
    )
    N = 100
    Q, K, V = _random_qkv(4, N, 64, seed=5)
    # Default budget: 0.20 * (1 - 0.10) = 0.18
    _, _, _, _, _, budget = codec.select_kv(
        Q, K, V, is_reasoning_task=False, kv_pool_pressure=0.85
    )
    expected = 0.20 * (1.0 - 0.10)
    assert abs(budget - expected) < 1e-9, f"high pressure budget={budget} != {expected}"


def test_select_kv_high_pressure_reasoning() -> None:
    """kv_pool_pressure=0.85 + reasoning: 0.093 * 0.9 applied."""
    codec = _make_codec(d_head=64)
    N = 200
    Q, K, V = _random_qkv(4, N, 64, seed=6)
    _, _, _, _, _, budget = codec.select_kv(
        Q, K, V, is_reasoning_task=True, kv_pool_pressure=0.85
    )
    expected = 0.093 * (1.0 - 0.10)
    assert abs(budget - expected) < 1e-9, f"reasoning+pressure budget={budget}"


def test_select_kv_indices_valid_range() -> None:
    """kept_indices must be in [0, N)."""
    codec = _make_codec(d_head=64)
    N = 80
    Q, K, V = _random_qkv(4, N, 64, seed=7)
    _, _, kept_idx, _, _, _ = codec.select_kv(Q, K, V)
    assert (kept_idx >= 0).all() and (kept_idx < N).all(), "kept_indices out of range"


# ---- detect_reasoning_task ---------------------------------------------------

def test_detect_reasoning_task_think_open_tag() -> None:
    assert TriAttentionPreRoPEKVSelectorCodec.detect_reasoning_task(
        "Let me think: <think>step 1</think>"
    ) is True


def test_detect_reasoning_task_think_close_tag_only() -> None:
    assert TriAttentionPreRoPEKVSelectorCodec.detect_reasoning_task(
        "done</think>"
    ) is True


def test_detect_reasoning_task_no_tag() -> None:
    assert TriAttentionPreRoPEKVSelectorCodec.detect_reasoning_task(
        "What is 2+2?"
    ) is False


def test_detect_reasoning_task_empty_string() -> None:
    assert TriAttentionPreRoPEKVSelectorCodec.detect_reasoning_task("") is False


# ---- CacheStore interface ----------------------------------------------------

def test_cachestore_put_and_get() -> None:
    codec = _make_codec(d_head=64, max_entries=10)
    torch.manual_seed(0)
    val = torch.randn(32, 64)
    codec.put("k1", val)
    result = codec.get("k1")
    assert result is not None, "get after put should not return None"
    assert result.shape == val.shape


def test_cachestore_get_miss_returns_none() -> None:
    codec = _make_codec(d_head=64)
    assert codec.get("nonexistent") is None


def test_cachestore_hit_rate_after_hit_and_miss() -> None:
    codec = _make_codec(d_head=64)
    val = torch.randn(16, 64)
    codec.put("x", val)
    codec.get("x")       # hit
    codec.get("y")       # miss
    hr = codec.hit_rate()
    assert abs(hr - 0.5) < 1e-9, f"hit_rate={hr} expected 0.5"


def test_cachestore_memory_bytes_positive_after_put() -> None:
    codec = _make_codec(d_head=64)
    codec.put("a", torch.randn(16, 64))
    assert codec.memory_bytes() > 0


def test_cachestore_evict_returns_positive_bytes() -> None:
    codec = _make_codec(d_head=64)
    codec.put("a", torch.randn(16, 64))
    freed = codec.evict()
    assert freed > 0
    assert codec.get("a") is None


def test_cachestore_evict_empty_store_returns_zero() -> None:
    codec = _make_codec(d_head=64)
    assert codec.evict() == 0


def test_cachestore_reset_stats() -> None:
    codec = _make_codec(d_head=64)
    val = torch.randn(16, 64)
    codec.put("k", val)
    codec.get("k")
    codec.get("miss")
    codec.reset_stats()
    assert codec._hits == 0
    assert codec._misses == 0
    assert codec.hit_rate() == 0.0


def test_cachestore_max_entries_triggers_eviction() -> None:
    codec = _make_codec(d_head=64, max_entries=3)
    for i in range(4):
        codec.put(f"k{i}", torch.randn(8, 64))
    assert len(codec._store) <= 3


def test_cachestore_duplicate_put_skipped() -> None:
    codec = _make_codec(d_head=64)
    v1 = torch.randn(8, 64)
    v2 = torch.randn(8, 64) * 10.0
    codec.put("key", v1)
    codec.put("key", v2)  # second put should be ignored
    result = codec.get("key")
    assert result is not None
    assert result.shape == v1.shape


# ---- put_compressed ----------------------------------------------------------

def test_put_compressed_returns_selector_entry() -> None:
    codec = _make_codec(d_head=64)
    Q, K, V = _random_qkv(8, 100, 64)
    entry = codec.put_compressed("kc", Q, K, V, is_reasoning_task=False)
    assert isinstance(entry, SelectorKVEntry)
    assert entry.original_seq_len == 100


def test_put_compressed_get_returns_selected_kv() -> None:
    codec = _make_codec(d_head=64, kv_budget_ratio_default=0.20)
    Q, K, V = _random_qkv(4, 50, 64, seed=99)
    codec.put_compressed("c1", Q, K, V, is_reasoning_task=False)
    result = codec.get("c1")
    assert result is not None
    expected_keep = max(1, int(50 * 0.20))
    assert result.shape[0] == expected_keep


# ---- get_importance_mask -----------------------------------------------------

def test_get_importance_mask_shape() -> None:
    codec = _make_codec(d_head=64, kv_budget_ratio_default=0.20)
    N = 80
    Q, K, V = _random_qkv(4, N, 64, seed=11)
    codec.put_compressed("m1", Q, K, V)
    mask = codec.get_importance_mask("m1")
    assert mask is not None
    assert mask.shape == (N,), f"mask shape={mask.shape} expected ({N},)"
    assert mask.dtype == torch.bool


def test_get_importance_mask_count_equals_n_keep() -> None:
    codec = _make_codec(d_head=64, kv_budget_ratio_default=0.20)
    N = 100
    Q, K, V = _random_qkv(4, N, 64, seed=12)
    codec.put_compressed("m2", Q, K, V)
    mask = codec.get_importance_mask("m2")
    n_keep = max(1, int(N * 0.20))
    assert mask.sum().item() == n_keep


def test_get_importance_mask_missing_key_returns_none() -> None:
    codec = _make_codec(d_head=64)
    assert codec.get_importance_mask("absent") is None


# ---- concentration_stats -----------------------------------------------------

def test_concentration_stats_returns_dict() -> None:
    codec = _make_codec(d_head=64)
    Q, K, V = _random_qkv(4, 50, 64, seed=20)
    codec.select_kv(Q, K, V)
    stats = codec.concentration_stats()
    assert isinstance(stats, dict)
    assert "conc_q_mean" in stats
    assert "conc_k_mean" in stats


def test_concentration_stats_empty_history() -> None:
    codec = _make_codec(d_head=64)
    stats = codec.concentration_stats()
    assert stats["conc_q_mean"] == 0.0
    assert stats["conc_k_mean"] == 0.0


def test_concentration_stats_values_in_0_1() -> None:
    codec = _make_codec(d_head=64)
    for seed in range(10):
        Q, K, V = _random_qkv(4, 40, 64, seed=seed)
        codec.select_kv(Q, K, V)
    stats = codec.concentration_stats()
    assert 0.0 <= stats["conc_q_mean"] <= 1.0
    assert 0.0 <= stats["conc_k_mean"] <= 1.0


# ---- memory_reduction_ratio --------------------------------------------------

def test_memory_reduction_ratio_reasoning_budget() -> None:
    """budget=0.093: memory_reduction_ratio >= 0.80.

    The memory_reduction_ratio compares actual stored bytes vs original_seq_len * d * 2 (FP16).
    When K is stored as float16 (2 bytes/element), ratio = 1 - 0.093 = 0.907 >= 0.85.
    When K is stored as float32 (4 bytes/element), ratio = 1 - 0.093*2 = 0.814 >= 0.80.
    We use FP16 tensors to achieve >= 0.85.
    """
    codec = _make_codec(d_head=64, kv_budget_ratio_reasoning=0.093)
    N = 256
    Q, K, V = _random_qkv(8, N, 64, seed=30)
    # Use FP16 storage so ratio = 1 - budget ≈ 0.907
    K_fp16 = K.half()
    V_fp16 = V.half()
    Q_fp32 = Q.float()
    codec.put_compressed(
        "reasoning", Q_fp32, K_fp16, V_fp16, is_reasoning_task=True
    )
    ratio = codec.memory_reduction_ratio()
    assert ratio >= 0.85, f"memory_reduction_ratio={ratio:.4f} < 0.85 (budget=0.093)"


def test_memory_reduction_ratio_positive_after_compression() -> None:
    codec = _make_codec(d_head=64, kv_budget_ratio_default=0.20)
    Q, K, V = _random_qkv(4, 100, 64)
    codec.put_compressed("c", Q, K, V, is_reasoning_task=False)
    ratio = codec.memory_reduction_ratio()
    assert ratio > 0.0


def test_memory_reduction_ratio_empty_store() -> None:
    codec = _make_codec(d_head=64)
    assert codec.memory_reduction_ratio() == 0.0


# ---- compression_hook --------------------------------------------------------

def test_compression_hook_reduces_sequence_length() -> None:
    codec = _make_codec(d_head=64, kv_budget_ratio_default=0.20)
    val = torch.randn(100, 64)
    result = codec.compression_hook("any_key", val)
    n_keep = max(1, int(100 * 0.20))
    assert result.shape[0] == n_keep
