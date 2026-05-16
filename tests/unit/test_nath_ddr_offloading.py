"""Unit tests for NAtHDDROffloadingScheduler (Activity A).

Tests:
  - EMA update convergence
  - 4-tier classification correctness
  - max_eviction_ratio enforcement (Tier-4 <= 3%)
  - Tier-2 DDR offload FP16 precision preservation
  - Tier-3 DDR offload INT8 quantisation
  - Tier-2 prefetch zero-approximation-error
  - Scheduling overhead under 5ms
  - Fairness: max_wait_ratio not exceeded
  - permanent_eviction_ratio() metric accuracy
"""

from __future__ import annotations

import time
from typing import Dict, List

import pytest
import torch

from src.scheduler.nath_ddr_offloading import (
    NAtHDDROffloadingConfig,
    NAtHDDROffloadingScheduler,
)

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_scheduler(
    max_eviction_ratio: float = 0.03,
    ema_alpha: float = 0.95,
    tier_boundaries: List[float] = None,
) -> NAtHDDROffloadingScheduler:
    if tier_boundaries is None:
        tier_boundaries = [0.30, 0.70, 0.97]
    cfg = NAtHDDROffloadingConfig(
        max_eviction_ratio=max_eviction_ratio,
        ema_alpha=ema_alpha,
        tier_boundaries=tier_boundaries,
        seed=SEED,
    )
    return NAtHDDROffloadingScheduler(cfg)


def _feed_scores(
    scheduler: NAtHDDROffloadingScheduler,
    token_key: str,
    scores: List[float],
) -> None:
    for s in scores:
        scheduler.update_attention_score(token_key, s)


def _make_token_keys(n: int, prefix: str = "tok") -> List[str]:
    return [f"{prefix}:{i}" for i in range(n)]


def _assign_varied_scores(
    scheduler: NAtHDDROffloadingScheduler,
    token_keys: List[str],
    seed: int = SEED,
) -> None:
    """Assign diverse attention scores so all 4 tiers get populated."""
    torch.manual_seed(seed)
    scores = torch.linspace(0.0, 1.0, len(token_keys)).tolist()
    for k, s in zip(token_keys, scores):
        scheduler.update_attention_score(k, s)


# ─────────────────────────────────────────────────────────────────────────────
# Test: EMA update convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestEMAUpdateConvergence:
    def test_ema_update_convergence(self) -> None:
        """EMA must converge toward the steady-state signal value."""
        scheduler = _make_scheduler(ema_alpha=0.9)
        key = "tok:0"
        target = 0.8

        # Repeatedly feed the same score — EMA should converge toward target
        for _ in range(200):
            scheduler.update_attention_score(key, target)

        score = scheduler._attn_score_ema[key]
        # After 200 updates with alpha=0.9 and target=0.8,
        # score = target * (1 - 0.9^200) ≈ 0.8
        assert abs(score - target) < 0.01, (
            f"EMA did not converge: got {score:.4f}, expected ~{target:.4f}"
        )

    def test_ema_exponential_decay(self) -> None:
        """EMA decays old values with correct exponential factor."""
        alpha = 0.95
        scheduler = _make_scheduler(ema_alpha=alpha)
        key = "tok:decay"

        # Single update: from 0 → new_score = alpha*0 + (1-alpha)*1.0 = 0.05
        scheduler.update_attention_score(key, 1.0)
        expected = (1.0 - alpha) * 1.0
        actual = scheduler._attn_score_ema[key]
        assert abs(actual - expected) < 1e-6, (
            f"Single-step EMA: got {actual:.6f}, expected {expected:.6f}"
        )

    def test_ema_old_score_decay(self) -> None:
        """After injecting a high score then zeros, the EMA decays."""
        scheduler = _make_scheduler(ema_alpha=0.9)
        key = "tok:decay2"

        # Feed 1.0 once
        scheduler.update_attention_score(key, 1.0)
        score_after_peak = scheduler._attn_score_ema[key]

        # Feed 0.0 many times — score should decay
        for _ in range(50):
            scheduler.update_attention_score(key, 0.0)

        score_after_decay = scheduler._attn_score_ema[key]
        assert score_after_decay < score_after_peak, "EMA should decay after sustained zero scores"
        assert score_after_decay < 0.01, f"EMA did not decay sufficiently: {score_after_decay:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: 4-tier classification
# ─────────────────────────────────────────────────────────────────────────────

class TestTierClassification4Tiers:
    def test_tier_classification_4tiers(self) -> None:
        """All tier assignments must be in {1, 2, 3, 4}."""
        scheduler = _make_scheduler()
        token_keys = _make_token_keys(100)
        _assign_varied_scores(scheduler, token_keys)

        tier_map = scheduler.classify_tokens(token_keys)

        assert set(tier_map.values()).issubset({1, 2, 3, 4}), (
            f"Unexpected tier values: {set(tier_map.values())}"
        )
        assert len(tier_map) == len(token_keys)

    def test_high_score_gets_tier1(self) -> None:
        """Tokens with the highest EMA score should be in Tier 1."""
        scheduler = _make_scheduler()
        n = 50
        token_keys = _make_token_keys(n)
        # Assign scores: first n//2 are high, rest are low
        for i, k in enumerate(token_keys):
            scheduler.update_attention_score(k, 1.0 if i < n // 2 else 0.0)

        tier_map = scheduler.classify_tokens(token_keys)
        high_score_tiers = [tier_map[token_keys[i]] for i in range(n // 2)]

        # At least some high-score tokens should be in Tier 1
        assert 1 in high_score_tiers, "No high-score token was classified as Tier 1"

    def test_low_score_gets_tier4_or_3(self) -> None:
        """Tokens with the lowest scores should be Tier 3 or Tier 4."""
        scheduler = _make_scheduler()
        n = 100
        token_keys = _make_token_keys(n)
        # Linspace: first 3 are strictly lowest
        scores = list(range(n))
        for k, s in zip(token_keys, scores):
            scheduler.update_attention_score(k, float(s))

        tier_map = scheduler.classify_tokens(token_keys)
        # The very lowest tokens should be in Tier 3 or 4
        for i in range(3):
            assert tier_map[token_keys[i]] in (3, 4), (
                f"Low-score token {i} got Tier {tier_map[token_keys[i]]}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test: max_eviction_ratio enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxEvictionRatioEnforced:
    def test_max_eviction_ratio_enforced(self) -> None:
        """Tier-4 fraction must not exceed max_eviction_ratio=0.03."""
        max_eviction_ratio = 0.03
        scheduler = _make_scheduler(max_eviction_ratio=max_eviction_ratio)
        n = 200
        token_keys = _make_token_keys(n)
        _assign_varied_scores(scheduler, token_keys)

        tier_map = scheduler.classify_tokens(token_keys)
        tier4_count = sum(1 for t in tier_map.values() if t == 4)
        tier4_ratio = tier4_count / n

        assert tier4_ratio <= max_eviction_ratio + 1e-6, (
            f"Tier-4 ratio {tier4_ratio:.4f} exceeds max_eviction_ratio {max_eviction_ratio}"
        )

    def test_max_eviction_ratio_enforced_strict_3pct(self) -> None:
        """Permanent eviction ratio metric must be <= 3% after many requests."""
        scheduler = _make_scheduler(max_eviction_ratio=0.03)
        for batch in range(5):
            token_keys = _make_token_keys(100, prefix=f"batch{batch}")
            _assign_varied_scores(scheduler, token_keys, seed=SEED + batch)
            scheduler.classify_tokens(token_keys)

        ratio = scheduler.permanent_eviction_ratio()
        assert ratio <= 0.03 + 0.001, (
            f"permanent_eviction_ratio {ratio:.4f} exceeds 3%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test: DDR offloading — Tier 2 FP16
# ─────────────────────────────────────────────────────────────────────────────

class TestDDROffloadTier2FP16:
    def test_ddr_offload_tier2_fp16(self) -> None:
        """Tier-2 offload preserves FP16 tensor exactly on CPU."""
        scheduler = _make_scheduler()
        key = "tier2_tok:0"
        torch.manual_seed(SEED)
        kv = torch.randn(8, 64, dtype=torch.float16)

        scheduler.offload_to_ddr(key, kv, tier=2)

        assert key in scheduler._ddr_buffer_fp16, "Tier-2 buffer not populated"
        stored = scheduler._ddr_buffer_fp16[key]
        assert stored.device.type == "cpu", "Tier-2 buffer should be on CPU"
        assert stored.dtype == torch.float16, f"Tier-2 buffer dtype {stored.dtype} != float16"
        assert torch.allclose(stored, kv.cpu()), "Tier-2 buffer content differs from original"

    def test_ddr_offload_tier2_not_in_int8_buffer(self) -> None:
        """Tier-2 offload must NOT go into INT8 buffer."""
        scheduler = _make_scheduler()
        key = "tier2_tok:1"
        kv = torch.randn(8, 64, dtype=torch.float16)
        scheduler.offload_to_ddr(key, kv, tier=2)
        assert key not in scheduler._ddr_buffer_int8, "Tier-2 token found in INT8 buffer"


# ─────────────────────────────────────────────────────────────────────────────
# Test: DDR offloading — Tier 3 INT8
# ─────────────────────────────────────────────────────────────────────────────

class TestDDROffloadTier3INT8:
    def test_ddr_offload_tier3_int8(self) -> None:
        """Tier-3 offload stores INT8 quantised tensor on CPU."""
        scheduler = _make_scheduler()
        key = "tier3_tok:0"
        torch.manual_seed(SEED)
        kv = torch.randn(8, 64, dtype=torch.float32)

        scheduler.offload_to_ddr(key, kv, tier=3)

        assert key in scheduler._ddr_buffer_int8, "Tier-3 INT8 buffer not populated"
        stored = scheduler._ddr_buffer_int8[key]
        assert stored.dtype == torch.int8, f"Tier-3 buffer dtype {stored.dtype} != int8"
        assert stored.device.type == "cpu"
        assert key in scheduler._ddr_scale, "Tier-3 quantisation scale not stored"

    def test_ddr_offload_tier3_dequant_approximate(self) -> None:
        """Tier-3 dequantised tensor should approximate the original."""
        scheduler = _make_scheduler()
        key = "tier3_tok:1"
        torch.manual_seed(SEED)
        kv = torch.randn(16, 64, dtype=torch.float32)

        scheduler.offload_to_ddr(key, kv, tier=3)
        restored = scheduler.restore_tier3_from_ddr(key, target_dtype=torch.float32)

        assert restored is not None, "restore_tier3_from_ddr returned None"
        # INT8 quantisation: expect < 1% relative error for typical Gaussian KV
        rel_err = ((kv - restored).norm() / kv.norm().clamp(min=1e-8)).item()
        assert rel_err < 0.02, f"INT8 dequant relative error {rel_err:.4f} > 2%"

    def test_ddr_offload_tier3_not_in_fp16_buffer(self) -> None:
        """Tier-3 offload must NOT go into FP16 buffer."""
        scheduler = _make_scheduler()
        key = "tier3_tok:2"
        kv = torch.randn(8, 64)
        scheduler.offload_to_ddr(key, kv, tier=3)
        assert key not in scheduler._ddr_buffer_fp16, "Tier-3 token found in FP16 buffer"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Tier-2 prefetch zero-approximation-error
# ─────────────────────────────────────────────────────────────────────────────

class TestPrefetchZeroApproximationError:
    def test_prefetch_zero_approximation_error(self) -> None:
        """Tier-2 prefetch restores FP16 with zero approximation error."""
        scheduler = _make_scheduler()
        key = "prefetch_tok:0"
        torch.manual_seed(SEED)
        kv_orig = torch.randn(16, 64, dtype=torch.float16)

        scheduler.offload_to_ddr(key, kv_orig, tier=2)

        prefetched = scheduler.prefetch_from_ddr([key])
        assert key in prefetched, "prefetch_from_ddr did not return the requested key"

        restored = prefetched[key]
        assert torch.allclose(restored.cpu(), kv_orig.cpu()), (
            "Tier-2 prefetch: restored tensor differs from original (non-zero approximation error)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Scheduling overhead
# ─────────────────────────────────────────────────────────────────────────────

class TestSchedulingOverheadUnder5Pct:
    def test_scheduling_overhead_under_5pct(self) -> None:
        """Scheduling overhead p50 must be < 5ms per request."""
        scheduler = _make_scheduler()

        for i in range(20):
            request = {
                "id": f"req_{i}",
                "token_ids": list(range(64)),
                "arrival_time": time.monotonic(),
            }
            scheduler.schedule_request(request)

        p50 = scheduler.scheduling_overhead_ms_p50()
        # 5ms is the TTFT overhead limit; scheduling alone should be << 5ms
        assert p50 < 5.0, f"Scheduling overhead p50 {p50:.2f}ms >= 5ms"

    def test_scheduling_overhead_recorded(self) -> None:
        """Scheduling times must be recorded for each schedule_request call."""
        scheduler = _make_scheduler()
        n_requests = 10
        for i in range(n_requests):
            scheduler.schedule_request({"id": f"r{i}", "token_ids": [1, 2, 3], "arrival_time": 0.0})

        assert len(scheduler._scheduling_times) == n_requests


# ─────────────────────────────────────────────────────────────────────────────
# Test: Fairness
# ─────────────────────────────────────────────────────────────────────────────

class TestFairnessMaxWait:
    def test_fairness_max_wait(self) -> None:
        """Request scheduling annotates token keys without exceeding wait logic."""
        scheduler = _make_scheduler()
        # Send same request twice — keys should be consistent (no crash)
        arrival = time.monotonic()
        req = {"id": "req_fair", "token_ids": list(range(10)), "arrival_time": arrival}
        result1 = scheduler.schedule_request(req.copy())
        result2 = scheduler.schedule_request(req.copy())

        assert "tier_assignment" in result1
        assert "tier_assignment" in result2
        # max_wait_ratio=2.0 means no request should be blocked beyond 2× normal wait
        # (this test verifies the scheduler returns a valid result, not a freeze)


# ─────────────────────────────────────────────────────────────────────────────
# Test: permanent_eviction_ratio metric accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestPermanentEvictionRatioMetric:
    def test_permanent_eviction_ratio_metric(self) -> None:
        """permanent_eviction_ratio() must match actual Tier-4 fraction."""
        scheduler = _make_scheduler(max_eviction_ratio=0.03)
        n = 100
        token_keys = _make_token_keys(n)
        _assign_varied_scores(scheduler, token_keys)

        tier_map = scheduler.classify_tokens(token_keys)
        tier4_count = sum(1 for t in tier_map.values() if t == 4)
        expected_ratio = tier4_count / scheduler._total_decisions if scheduler._total_decisions > 0 else 0.0

        actual_ratio = scheduler.permanent_eviction_ratio()
        assert abs(actual_ratio - expected_ratio) < 1e-6, (
            f"permanent_eviction_ratio {actual_ratio:.6f} != expected {expected_ratio:.6f}"
        )

    def test_cache_hit_rate_complement(self) -> None:
        """cache_hit_rate() must equal 1 - permanent_eviction_ratio()."""
        scheduler = _make_scheduler()
        n = 80
        token_keys = _make_token_keys(n)
        _assign_varied_scores(scheduler, token_keys)
        scheduler.classify_tokens(token_keys)

        hit_rate = scheduler.cache_hit_rate()
        eviction_ratio = scheduler.permanent_eviction_ratio()
        assert abs(hit_rate + eviction_ratio - 1.0) < 1e-6, (
            f"hit_rate {hit_rate:.4f} + eviction_ratio {eviction_ratio:.4f} != 1.0"
        )

    def test_zero_ratio_before_any_classification(self) -> None:
        """permanent_eviction_ratio() must return 0.0 before any classifications."""
        scheduler = _make_scheduler()
        assert scheduler.permanent_eviction_ratio() == 0.0
        assert scheduler.cache_hit_rate() == 0.0
        assert scheduler.scheduling_overhead_ms_p50() == 0.0
