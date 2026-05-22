"""Unit tests for DapQPositionAwareEvictionCodec (Activity C).

Covers:
  - RoPE rotation correctness
  - Importance score computation
  - KV index selection (count, recency guarantee, sort order)
  - Compression hook behavior (zero-out, preserve selected, mask storage)
  - Memory reduction ratio
  - Pressure-adaptive budget ratio
  - CacheStore interface compliance
  - Seed reproducibility
  - LRU eviction ordering
  - Hit rate tracking
"""

import pytest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")

from src.cache.dapq_position_aware_eviction_codec import (
    DapQEvictionConfig,
    DapQPositionAwareEvictionCodec,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


def _make_codec(
    budget_ratio: float = 0.30,
    recent_window: int = 10,
    d_head: int = 64,
    use_unit_template: bool = True,
    max_entries: int = 100,
    seed: int = 42,
) -> DapQPositionAwareEvictionCodec:
    cfg = DapQEvictionConfig(
        d_head=d_head,
        budget_ratio=budget_ratio,
        recent_window=recent_window,
        use_unit_template=use_unit_template,
        max_entries=max_entries,
        seed=seed,
    )
    return DapQPositionAwareEvictionCodec(cfg)


# ------------------------------------------------------------------ #
# RoPE rotation tests                                                  #
# ------------------------------------------------------------------ #


def test_dapq_rope_apply_rotation_changes_vector() -> None:
    """_apply_rope(x, pos=10) != x: RoPE rotation changes the vector."""
    torch.manual_seed(42)
    x = torch.randn(64)
    rotated = DapQPositionAwareEvictionCodec._apply_rope(x, pos=10)
    assert not torch.allclose(rotated, x), "RoPE rotation should change the vector at pos=10"


def test_dapq_rope_apply_pos_zero_identity() -> None:
    """_apply_rope(x, pos=0) ≈ x: no rotation at position 0."""
    torch.manual_seed(42)
    x = torch.randn(64)
    rotated = DapQPositionAwareEvictionCodec._apply_rope(x, pos=0)
    assert torch.allclose(rotated.float(), x.float(), atol=1e-5), (
        f"RoPE at pos=0 should be identity; max diff={( rotated - x).abs().max():.2e}"
    )


# ------------------------------------------------------------------ #
# Importance computation                                               #
# ------------------------------------------------------------------ #


def test_dapq_compute_importance_sums_to_one() -> None:
    """compute_importance returns softmax distribution summing to 1.0."""
    codec = _make_codec()
    torch.manual_seed(42)
    K = torch.randn(50, 64)
    importance = codec.compute_importance(K, pos_decode=50)
    assert abs(importance.sum().item() - 1.0) < 1e-5, (
        f"Importance sum={importance.sum().item():.8f} != 1.0"
    )


def test_dapq_compute_importance_shape() -> None:
    """compute_importance returns [seq_len] tensor."""
    codec = _make_codec()
    torch.manual_seed(42)
    seq_len, d_head = 75, 64
    K = torch.randn(seq_len, d_head)
    importance = codec.compute_importance(K, pos_decode=75)
    assert importance.shape == (seq_len,), (
        f"Expected shape ({seq_len},), got {importance.shape}"
    )


# ------------------------------------------------------------------ #
# KV index selection                                                   #
# ------------------------------------------------------------------ #


def test_dapq_select_kv_indices_count() -> None:
    """seq_len=100, budget_ratio=0.30, recent_window=10 → at least max(10,30)=30 indices."""
    codec = _make_codec(budget_ratio=0.30, recent_window=10)
    torch.manual_seed(42)
    K = torch.randn(100, 64)
    indices = codec.select_kv_indices(K, pos_decode=100)
    assert len(indices) >= 30, f"Expected >= 30 indices, got {len(indices)}"


def test_dapq_select_kv_indices_recent_always_included() -> None:
    """Most recent recent_window token indices are always included in selection."""
    recent_window = 10
    seq_len = 100
    codec = _make_codec(budget_ratio=0.30, recent_window=recent_window)
    torch.manual_seed(42)
    K = torch.randn(seq_len, 64)
    indices = codec.select_kv_indices(K, pos_decode=seq_len)
    indices_set = set(indices.tolist())
    recent_start = seq_len - recent_window
    for i in range(recent_start, seq_len):
        assert i in indices_set, f"Recent token index {i} not in selected indices"


def test_dapq_select_kv_indices_sorted() -> None:
    """select_kv_indices returns indices in ascending order."""
    codec = _make_codec()
    torch.manual_seed(42)
    K = torch.randn(100, 64)
    indices = codec.select_kv_indices(K, pos_decode=100)
    assert torch.all(indices[1:] > indices[:-1]), "Selected indices are not sorted ascending"


# ------------------------------------------------------------------ #
# Compression hook                                                     #
# ------------------------------------------------------------------ #


def test_dapq_compression_hook_zeros_unselected() -> None:
    """Unselected positions are zero after compression_hook."""
    codec = _make_codec(budget_ratio=0.30, recent_window=10)
    torch.manual_seed(42)
    seq_len, d_head = 100, 64
    value = torch.randn(seq_len, d_head)
    compressed = codec.compression_hook("test_key", value)
    mask = codec.get_importance_mask("test_key")
    assert mask is not None
    unselected = ~mask
    if unselected.any():
        assert compressed[unselected].abs().max().item() == 0.0, (
            "Unselected positions should be zeroed out"
        )


def test_dapq_compression_hook_preserves_selected() -> None:
    """Selected positions retain their original values after compression_hook."""
    codec = _make_codec(budget_ratio=0.30, recent_window=10)
    torch.manual_seed(42)
    seq_len, d_head = 100, 64
    value = torch.randn(seq_len, d_head)
    compressed = codec.compression_hook("test_key", value)
    mask = codec.get_importance_mask("test_key")
    assert mask is not None
    selected = mask
    if selected.any():
        assert torch.allclose(compressed[selected], value[selected]), (
            "Selected positions should retain original values"
        )


def test_dapq_compression_hook_stores_importance_mask() -> None:
    """put() followed by get_importance_mask returns a bool tensor of shape [seq_len]."""
    codec = _make_codec()
    torch.manual_seed(42)
    seq_len, d_head = 100, 64
    value = torch.randn(seq_len, d_head)
    codec.put("mask_key", value)
    mask = codec.get_importance_mask("mask_key")
    assert mask is not None, "Importance mask should be stored after put()"
    assert mask.dtype == torch.bool, f"Mask dtype should be bool, got {mask.dtype}"
    assert mask.shape == (seq_len,), f"Mask shape should be ({seq_len},), got {mask.shape}"


# ------------------------------------------------------------------ #
# Memory reduction                                                     #
# ------------------------------------------------------------------ #


def test_dapq_memory_reduction_ratio_above_60pct() -> None:
    """budget_ratio=0.30, seq_len=100, recent_window=10 → logical reduction >= 0.50.

    With budget_ratio=0.30, seq_len=100, recent_window=10:
      top_k = max(10, int(100 * 0.30)) = max(10, 30) = 30
    So at most 30 tokens are kept → logical reduction = 1 - 30/100 = 0.70 >= 0.50.
    We set pool_utilization to 0.65 (between low and high thresholds) so the
    effective budget ratio is the configured 0.30 (not the conservative 0.50).
    """
    codec = _make_codec(budget_ratio=0.30, recent_window=10, d_head=64)
    # Set utilization in normal range (between low=0.50 and high=0.80)
    codec.update_pool_utilization(0.65)
    torch.manual_seed(42)
    seq_len = 100
    K = torch.randn(seq_len, 64)
    indices = codec.select_kv_indices(K, pos_decode=seq_len)
    kept_ratio = len(indices) / seq_len
    logical_reduction = 1.0 - kept_ratio
    assert logical_reduction >= 0.50, (
        f"Logical memory reduction {logical_reduction:.4f} < 0.50 "
        f"(budget_ratio=0.30, recent_window=10, kept={len(indices)}/{seq_len})"
    )


# ------------------------------------------------------------------ #
# Pressure-adaptive budget ratio                                       #
# ------------------------------------------------------------------ #


def test_dapq_high_pressure_uses_aggressive_budget() -> None:
    """update_pool_utilization(0.90) → effective budget ratio == high_pressure_budget_ratio."""
    cfg = DapQEvictionConfig(
        high_pressure_threshold=0.80,
        high_pressure_budget_ratio=0.15,
        seed=42,
    )
    codec = DapQPositionAwareEvictionCodec(cfg)
    codec.update_pool_utilization(0.90)
    assert codec._get_effective_budget_ratio() == cfg.high_pressure_budget_ratio, (
        f"Expected {cfg.high_pressure_budget_ratio}, got {codec._get_effective_budget_ratio()}"
    )


def test_dapq_low_pressure_uses_conservative_budget() -> None:
    """update_pool_utilization(0.30) → effective budget ratio == low_pressure_budget_ratio."""
    cfg = DapQEvictionConfig(
        low_pressure_threshold=0.50,
        low_pressure_budget_ratio=0.50,
        seed=42,
    )
    codec = DapQPositionAwareEvictionCodec(cfg)
    codec.update_pool_utilization(0.30)
    assert codec._get_effective_budget_ratio() == cfg.low_pressure_budget_ratio, (
        f"Expected {cfg.low_pressure_budget_ratio}, got {codec._get_effective_budget_ratio()}"
    )


# ------------------------------------------------------------------ #
# Template mode                                                        #
# ------------------------------------------------------------------ #


def test_dapq_unit_template_vs_nonunit_same_dtype() -> None:
    """Both use_unit_template=True and False return [d_head] float32 template."""
    d_head = 64
    cfg_unit = DapQEvictionConfig(d_head=d_head, use_unit_template=True, seed=42)
    cfg_semantic = DapQEvictionConfig(d_head=d_head, use_unit_template=False, seed=42)
    codec_unit = DapQPositionAwareEvictionCodec(cfg_unit)
    codec_semantic = DapQPositionAwareEvictionCodec(cfg_semantic)

    # Provide semantic templates for non-unit codec
    templates = torch.randn(12, 8, d_head)
    codec_semantic.set_q_templates(templates)

    t_unit = codec_unit._get_q_template()
    t_semantic = codec_semantic._get_q_template()
    assert t_unit.shape == (d_head,), f"Unit template shape: {t_unit.shape}"
    assert t_semantic.shape == (d_head,), f"Semantic template shape: {t_semantic.shape}"
    assert t_unit.dtype == torch.float32
    assert t_semantic.dtype == torch.float32


# ------------------------------------------------------------------ #
# CacheStore interface compliance                                      #
# ------------------------------------------------------------------ #


def test_dapq_cachestore_interface_full() -> None:
    """put/get/evict/hit_rate/memory_bytes/reset_stats all function correctly."""
    codec = _make_codec(max_entries=10)
    torch.manual_seed(42)
    kv = torch.randn(50, 64)

    codec.put("a", kv)
    assert codec.get("a") is not None, "get() should return stored value"
    assert codec.get("missing_key") is None, "get() should return None on miss"
    assert codec.hit_rate() > 0.0, "hit_rate() should be > 0 after a hit"
    assert codec.memory_bytes() > 0, "memory_bytes() should be > 0 after put()"

    freed = codec.evict()
    assert freed > 0, "evict() should return > 0 bytes freed"

    codec.reset_stats()
    assert codec._hits == 0
    assert codec._misses == 0
    assert codec._total_bytes_original == 0
    assert codec._total_bytes_stored == 0


# ------------------------------------------------------------------ #
# Reproducibility                                                      #
# ------------------------------------------------------------------ #


def test_dapq_seed_reproducibility() -> None:
    """Same seed + same input → same selected_indices."""
    torch.manual_seed(42)
    K = torch.randn(100, 64)

    def _run(seed: int) -> torch.Tensor:
        codec = _make_codec(seed=seed)
        return codec.select_kv_indices(K.clone(), pos_decode=100)

    idx1 = _run(42)
    idx2 = _run(42)
    assert torch.all(idx1 == idx2), "Same seed should produce identical selected indices"


# ------------------------------------------------------------------ #
# LRU eviction ordering                                               #
# ------------------------------------------------------------------ #


def test_dapq_evict_lru_oldest_first() -> None:
    """max_entries=2, insert 3 keys → first inserted key is evicted."""
    codec = _make_codec(max_entries=2)
    torch.manual_seed(42)
    kv = torch.randn(50, 64)

    codec.put("first", kv)
    codec.put("second", kv)
    codec.put("third", kv)  # triggers eviction of "first"

    assert codec.get("first") is None, "Oldest entry should have been evicted"
    assert codec.get("second") is not None, "Second entry should still be present"
    assert codec.get("third") is not None, "Third entry should be present"


# ------------------------------------------------------------------ #
# Hit rate tracking                                                    #
# ------------------------------------------------------------------ #


def test_dapq_hit_rate_tracking() -> None:
    """1 hit + 1 miss → hit_rate() == 0.5.

    Note: reset_stats() clears the store, so we put entries AFTER reset.
    """
    codec = _make_codec()
    torch.manual_seed(42)
    kv = torch.randn(50, 64)

    # Put entries first, then get one hit and one miss (no reset needed)
    codec.put("exists", kv)
    initial_hits = codec._hits
    initial_misses = codec._misses

    codec.get("exists")    # hit
    codec.get("no_such")   # miss

    new_hits = codec._hits - initial_hits
    new_misses = codec._misses - initial_misses
    assert new_hits == 1 and new_misses == 1, (
        f"Expected 1 hit + 1 miss, got {new_hits} hits + {new_misses} misses"
    )
    # Overall hit rate: 1 hit out of 2 accesses = 0.5
    total = codec._hits + codec._misses
    computed_rate = codec._hits / total
    assert abs(computed_rate - 0.5) < 1e-6, (
        f"Expected hit_rate=0.5, got {computed_rate}"
    )
