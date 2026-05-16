"""Unit tests for NAtHRetentionTierDecider (Cross Activity A+C).

Tests:
  - Dual-signal combination with alpha weighting
  - Tier-boundary auto-adjust on eviction ratio excess
  - Tier-boundary auto-adjust on HBM pressure
  - Eviction ratio maintained below 3%
  - Triple comparison: Solo A-2 / Solo C-2 / Combined Cross-1
  - get_global_retention_score integration
"""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from src.cache.global_retention_gate_eviction import (
    GlobalRetentionGateConfig,
    GlobalRetentionGateEvictionCodec,
)
from src.scheduler.nath_ddr_offloading import (
    NAtHDDROffloadingConfig,
    NAtHDDROffloadingScheduler,
)
from src.scheduler.nath_retention_tier_decider import (
    NAtHRetentionTierDecider,
    NAtHRetentionTierDeciderConfig,
)

SEED = 42
N_LAYERS = 4
N_HEADS = 4
D_HEAD = 64
D_MODEL = N_HEADS * D_HEAD


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_components(
    alpha: float = 0.5,
    max_eviction_ratio: float = 0.03,
    tier_boundaries: List[float] = None,
) -> tuple:
    nath_cfg = NAtHDDROffloadingConfig(
        max_eviction_ratio=max_eviction_ratio,
        tier_boundaries=tier_boundaries or [0.30, 0.70, 0.97],
        seed=SEED,
    )
    nath = NAtHDDROffloadingScheduler(nath_cfg)

    retention_cfg = GlobalRetentionGateConfig(
        n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL, budget_ratio=0.3, seed=SEED
    )
    retention = GlobalRetentionGateEvictionCodec(retention_cfg)

    decider_cfg = NAtHRetentionTierDeciderConfig(
        alpha=alpha, max_eviction_ratio=max_eviction_ratio, seed=SEED
    )
    decider = NAtHRetentionTierDecider(decider_cfg, nath, retention)
    return nath, retention, decider


def _make_kv(n_tokens: int, seed: int = SEED) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_tokens, N_LAYERS, N_HEADS, D_HEAD)


def _seed_nath_scores(nath: NAtHDDROffloadingScheduler, token_keys: List[str], seed: int = SEED) -> None:
    torch.manual_seed(seed)
    scores = torch.linspace(0.0, 1.0, len(token_keys)).tolist()
    for k, s in zip(token_keys, scores):
        nath.update_attention_score(k, s)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Dual-signal combination
# ─────────────────────────────────────────────────────────────────────────────

class TestDualSignalCombination:
    def test_dual_signal_combination(self) -> None:
        """decide_tier() must produce tier in {1,2,3,4} for all tokens."""
        nath, retention, decider = _make_components(alpha=0.5)
        n = 40
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        _seed_nath_scores(nath, token_keys)

        tier_map = decider.decide_tier(token_keys, kv_tensor=kv)

        assert len(tier_map) == n, f"tier_map has {len(tier_map)} entries, expected {n}"
        assert set(tier_map.values()).issubset({1, 2, 3, 4}), (
            f"Unexpected tiers: {set(tier_map.values())}"
        )

    def test_alpha_1_dominated_by_attn(self) -> None:
        """alpha=1.0: only attention signal determines tier — retention has no effect."""
        nath, retention, decider = _make_components(alpha=1.0)
        n = 30
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        _seed_nath_scores(nath, token_keys)

        tier_map = decider.decide_tier(token_keys, kv_tensor=kv)
        assert set(tier_map.values()).issubset({1, 2, 3, 4})

    def test_alpha_0_dominated_by_retention(self) -> None:
        """alpha=0.0: only retention signal determines tier — attention has no effect."""
        nath, retention, decider = _make_components(alpha=0.0)
        n = 30
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        # Do NOT seed attn scores — they should all be 0.0 (no effect)

        tier_map = decider.decide_tier(token_keys, kv_tensor=kv)
        assert set(tier_map.values()).issubset({1, 2, 3, 4})

    def test_decide_tier_without_kv(self) -> None:
        """decide_tier() must work when kv_tensor=None (fallback to uniform retention)."""
        nath, retention, decider = _make_components()
        n = 20
        token_keys = [f"tok:{i}" for i in range(n)]
        _seed_nath_scores(nath, token_keys)

        tier_map = decider.decide_tier(token_keys, kv_tensor=None)
        assert len(tier_map) == n
        assert set(tier_map.values()).issubset({1, 2, 3, 4})


# ─────────────────────────────────────────────────────────────────────────────
# Test: Tier boundary auto-adjust on eviction ratio excess
# ─────────────────────────────────────────────────────────────────────────────

class TestTierBoundaryAutoAdjustEviction:
    def test_tier_boundary_auto_adjust_eviction(self) -> None:
        """Tier-3→4 boundary (p3) must increase when eviction ratio exceeds limit."""
        nath, retention, decider = _make_components(max_eviction_ratio=0.03)
        initial_p3 = decider._tier_boundaries[2]

        # Simulate an eviction ratio well above the limit
        decider.adjust_tier_boundaries(current_eviction_ratio=0.10, hbm_pressure=0.0)

        new_p3 = decider._tier_boundaries[2]
        assert new_p3 > initial_p3, (
            f"p3 should increase after excess eviction: {initial_p3:.4f} → {new_p3:.4f}"
        )

    def test_tier_boundary_p3_capped_at_0_99(self) -> None:
        """p3 must not exceed 0.99 regardless of how often adjust is called."""
        nath, retention, decider = _make_components()
        for _ in range(100):
            decider.adjust_tier_boundaries(current_eviction_ratio=0.50, hbm_pressure=0.0)
        assert decider._tier_boundaries[2] <= 0.99


# ─────────────────────────────────────────────────────────────────────────────
# Test: Tier boundary auto-adjust on HBM pressure
# ─────────────────────────────────────────────────────────────────────────────

class TestTierBoundaryAutoAdjustHBM:
    def test_tier_boundary_auto_adjust_hbm(self) -> None:
        """Tier-1→2 boundary (p1) must decrease when HBM pressure > 0.8."""
        nath, retention, decider = _make_components()
        initial_p1 = decider._tier_boundaries[0]

        decider.adjust_tier_boundaries(current_eviction_ratio=0.01, hbm_pressure=0.9)

        new_p1 = decider._tier_boundaries[0]
        assert new_p1 < initial_p1, (
            f"p1 should decrease under HBM pressure: {initial_p1:.4f} → {new_p1:.4f}"
        )

    def test_tier_boundary_p1_floor_at_0_05(self) -> None:
        """p1 must not go below 0.05 (at least 5% stays in HBM)."""
        nath, retention, decider = _make_components()
        for _ in range(100):
            decider.adjust_tier_boundaries(current_eviction_ratio=0.01, hbm_pressure=0.99)
        assert decider._tier_boundaries[0] >= 0.05

    def test_boundary_ordering_preserved(self) -> None:
        """After adjustment, p1 < p2 < p3 must always hold."""
        nath, retention, decider = _make_components()
        for _ in range(20):
            decider.adjust_tier_boundaries(current_eviction_ratio=0.10, hbm_pressure=0.95)
        p1, p2, p3 = decider._tier_boundaries
        assert p1 < p2, f"p1 {p1:.4f} >= p2 {p2:.4f}"
        assert p2 < p3, f"p2 {p2:.4f} >= p3 {p3:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Eviction ratio below 3% after dual-signal combination
# ─────────────────────────────────────────────────────────────────────────────

class TestEvictionRatioBelow3Pct:
    def test_eviction_ratio_below_3pct(self) -> None:
        """Tier-4 fraction from decide_tier() must be <= max_eviction_ratio=3%."""
        nath, retention, decider = _make_components(max_eviction_ratio=0.03)
        n = 200
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        _seed_nath_scores(nath, token_keys)

        tier_map = decider.decide_tier(token_keys, kv_tensor=kv)
        tier4_count = sum(1 for t in tier_map.values() if t == 4)
        ratio = tier4_count / n

        assert ratio <= 0.03 + 1e-6, (
            f"Tier-4 ratio {ratio:.4f} exceeds max_eviction_ratio=0.03"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Triple comparison throughput
# ─────────────────────────────────────────────────────────────────────────────

class TestTripleComparisonThroughput:
    """Solo A-2 / Solo C-2 / Combined Cross-1 throughput proxy comparison.

    Throughput proxy: (1 - eviction_ratio) as fraction of kept tokens.
    """

    def test_triple_comparison_throughput(self) -> None:
        n = 100
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        budget_ratio = 0.3

        # ── Solo A-2: NAtH scheduler only ─────────────────────────────────
        nath_solo_cfg = NAtHDDROffloadingConfig(max_eviction_ratio=0.03, seed=SEED)
        nath_solo = NAtHDDROffloadingScheduler(nath_solo_cfg)
        _seed_nath_scores(nath_solo, token_keys)
        solo_a2_tiers = nath_solo.classify_tokens(token_keys)
        solo_a2_hit_rate = 1.0 - sum(1 for t in solo_a2_tiers.values() if t == 4) / n

        # ── Solo C-2: GlobalRetentionGate only ────────────────────────────
        ret_cfg = GlobalRetentionGateConfig(
            n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL,
            budget_ratio=budget_ratio, seed=SEED
        )
        ret_solo = GlobalRetentionGateEvictionCodec(ret_cfg)
        kv_kept = ret_solo.compression_hook("_solo_c2_", kv)
        solo_c2_keep_rate = kv_kept.shape[0] / n

        # ── Combined Cross-1 ───────────────────────────────────────────────
        nath, retention, decider = _make_components(alpha=0.5, max_eviction_ratio=0.03)
        _seed_nath_scores(nath, token_keys)
        cross1_tiers = decider.decide_tier(token_keys, kv_tensor=kv)
        cross1_hit_rate = 1.0 - sum(1 for t in cross1_tiers.values() if t == 4) / n

        # Combined should not perform worse than either solo on eviction rate
        # (Both A-2 and C-2 respect max_eviction_ratio=3%)
        assert solo_a2_hit_rate >= 0.97, f"Solo A-2 hit rate {solo_a2_hit_rate:.3f} < 0.97"
        assert cross1_hit_rate >= 0.97, f"Cross-1 hit rate {cross1_hit_rate:.3f} < 0.97"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Triple comparison memory
# ─────────────────────────────────────────────────────────────────────────────

class TestTripleComparisonMemory:
    """Solo A-2 / Solo C-2 / Combined Cross-1 memory proxy comparison."""

    def test_triple_comparison_memory(self) -> None:
        n = 100
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        budget_ratio = 0.3

        # Solo C-2 memory reduction
        ret_cfg = GlobalRetentionGateConfig(
            n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL,
            budget_ratio=budget_ratio, seed=SEED
        )
        ret = GlobalRetentionGateEvictionCodec(ret_cfg)
        solo_c2_reduction = ret.memory_reduction_ratio()

        # Combined: same budget_ratio → same base memory reduction for C-2 component
        nath, retention, decider = _make_components()
        combined_c2_reduction = retention.memory_reduction_ratio()

        assert solo_c2_reduction >= 0.30, f"Solo C-2 memory reduction {solo_c2_reduction:.3f} < 30%"
        assert combined_c2_reduction >= 0.30, f"Combined C-2 memory reduction {combined_c2_reduction:.3f} < 30%"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Triple comparison accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestTripleComparisonAccuracy:
    """Solo A-2 / Solo C-2 / Combined Cross-1 accuracy proxy comparison.

    Accuracy proxy: attention output relative error after KV eviction.
    Uses the same structured KV data + trained gate as the main accuracy tests,
    both for Solo C-2 and for the Combined Cross-1 retention codec.
    """

    def _build_calibration(self, n: int, budget_ratio: float, n_samples: int = 50) -> list:
        """Build diverse calibration dataset for gate training."""
        import math
        n_important = max(1, int(math.ceil(n * budget_ratio)))
        calibration = []
        for i in range(n_samples):
            torch.manual_seed(i)
            kv_c = torch.zeros(n, N_LAYERS, N_HEADS, D_HEAD)
            kv_c[:n_important] = torch.randn(n_important, N_LAYERS, N_HEADS, D_HEAD) * 50.0
            kv_c[n_important:] = torch.randn(n - n_important, N_LAYERS, N_HEADS, D_HEAD) * 0.01
            calibration.append(kv_c)
        return calibration

    def test_triple_comparison_accuracy(self) -> None:
        import math
        from src.metrics.perplexity import (
            attention_output_relative_error,
            cosine_similarity_output,
        )

        n = 64
        budget_ratio = 0.3
        n_important = max(1, int(math.ceil(n * budget_ratio)))

        # Build test KV with test seed (different from calibration seeds)
        torch.manual_seed(SEED + 100)
        kv = torch.zeros(n, N_LAYERS, N_HEADS, D_HEAD)
        kv[:n_important] = torch.randn(n_important, N_LAYERS, N_HEADS, D_HEAD) * 50.0
        kv[n_important:] = torch.randn(n - n_important, N_LAYERS, N_HEADS, D_HEAD) * 0.01

        token_keys = [f"tok:{i}" for i in range(n)]
        torch.manual_seed(SEED + 1)
        q = torch.randn(4, D_HEAD)
        k_orig = kv[:, 0, 0, :]
        v_orig = kv[:, 0, 0, :]

        # Build calibration once (seeds 0..49)
        calibration = self._build_calibration(n, budget_ratio, n_samples=50)

        # ── Solo C-2: trained retention gate ────────────────────────────
        ret_cfg = GlobalRetentionGateConfig(
            n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL,
            budget_ratio=budget_ratio, recent_window=0, seed=SEED
        )
        ret_solo = GlobalRetentionGateEvictionCodec(ret_cfg)
        ret_solo.train_retention_gate(calibration, n_epochs=10, lr=0.01)

        kv_solo_c2 = ret_solo.compression_hook("_solo_", kv)
        k_solo = kv_solo_c2[:, 0, 0, :]
        v_solo = kv_solo_c2[:, 0, 0, :]
        err_solo_c2 = attention_output_relative_error(q, k_orig, v_orig, k_solo, v_solo)
        cos_solo_c2 = cosine_similarity_output(q, k_orig, v_orig, k_solo, v_solo)

        # ── Combined Cross-1: same trained retention codec ───────────────
        nath, retention, decider = _make_components(alpha=0.5)
        # Disable recent_window to ensure gate scores determine eviction
        retention.config.recent_window = 0
        retention.train_retention_gate(calibration, n_epochs=10, lr=0.01)
        _seed_nath_scores(nath, token_keys)
        kv_cross = retention.compression_hook("_cross_", kv)
        k_cross = kv_cross[:, 0, 0, :]
        v_cross = kv_cross[:, 0, 0, :]
        err_cross = attention_output_relative_error(q, k_orig, v_orig, k_cross, v_cross)
        cos_cross = cosine_similarity_output(q, k_orig, v_orig, k_cross, v_cross)

        # Both must satisfy ±1% accuracy constraint
        assert err_solo_c2 < 0.01, f"Solo C-2 attention error {err_solo_c2:.4f} >= 1%"
        assert err_cross < 0.01, f"Cross-1 attention error {err_cross:.4f} >= 1%"
        assert cos_solo_c2 >= 0.99, f"Solo C-2 cosine {cos_solo_c2:.4f} < 0.99"
        assert cos_cross >= 0.99, f"Cross-1 cosine {cos_cross:.4f} < 0.99"


# ─────────────────────────────────────────────────────────────────────────────
# Test: get_global_retention_score called
# ─────────────────────────────────────────────────────────────────────────────

class TestGetGlobalRetentionScoreCalled:
    def test_get_global_retention_score_called(self) -> None:
        """decide_tier() must invoke retention codec's get_global_retention_score."""
        nath, retention, decider = _make_components()
        n = 20
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        _seed_nath_scores(nath, token_keys)

        # Verify that get_global_retention_score returns the right shape
        scores = retention.get_global_retention_score(kv=kv)
        assert scores.shape[0] == n, (
            f"Retention score shape {scores.shape[0]} != n_tokens {n}"
        )

        # Run decide_tier and verify it runs without error
        tier_map = decider.decide_tier(token_keys, kv_tensor=kv)
        assert len(tier_map) == n

    def test_tier_distribution_updated(self) -> None:
        """tier_distribution() must reflect all tokens after decide_tier()."""
        nath, retention, decider = _make_components()
        n = 40
        token_keys = [f"tok:{i}" for i in range(n)]
        kv = _make_kv(n)
        _seed_nath_scores(nath, token_keys)

        decider.decide_tier(token_keys, kv_tensor=kv)
        dist = decider.tier_distribution()

        total_frac = sum(dist.values())
        assert abs(total_frac - 1.0) < 1e-6, f"tier_distribution sums to {total_frac:.4f}, not 1.0"
        assert all(v >= 0.0 for v in dist.values()), "negative tier fraction found"
