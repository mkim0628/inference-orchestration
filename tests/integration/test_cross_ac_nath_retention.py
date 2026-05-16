"""E2E integration test: Cross A+C NAtHRetentionTierDecider + GlobalRetentionGateEvictionCodec.

Tests the full pipeline:
  - Multiple requests scheduled by NAtHDDROffloadingScheduler
  - KV cache stored via GlobalRetentionGateEvictionCodec (with global retention gate eviction)
  - Tier decisions made by NAtHRetentionTierDecider (dual-signal)
  - DDR offloading and prefetch round-trip
  - Accuracy preserved (attention error < 1%) across the entire flow
  - Permanent eviction ratio <= 3% throughout

Integration scope (Activity A + C):
  NAtHDDROffloadingScheduler → GlobalRetentionGateEvictionCodec → NAtHRetentionTierDecider
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F

from src.cache.global_retention_gate_eviction import (
    GlobalRetentionGateConfig,
    GlobalRetentionGateEvictionCodec,
)
from src.metrics.perplexity import (
    attention_output_relative_error,
    attention_kl_divergence,
    cosine_similarity_output,
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
BUDGET_RATIO = 0.3
N_REQUESTS = 10
TOKENS_PER_REQUEST = 32


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _build_system() -> Tuple[
    NAtHDDROffloadingScheduler,
    GlobalRetentionGateEvictionCodec,
    NAtHRetentionTierDecider,
]:
    nath_cfg = NAtHDDROffloadingConfig(
        max_eviction_ratio=0.03,
        ema_alpha=0.95,
        tier_boundaries=[0.30, 0.70, 0.97],
        seed=SEED,
    )
    nath = NAtHDDROffloadingScheduler(nath_cfg)

    retention_cfg = GlobalRetentionGateConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        budget_ratio=BUDGET_RATIO,
        # recent_window=0: let gate score alone determine which tokens are kept;
        # this is required for accuracy tests since recent tokens are noise tokens
        # in the structured KV design (important tokens are at indices 0..n_important-1)
        recent_window=0,
        seed=SEED,
    )
    retention = GlobalRetentionGateEvictionCodec(retention_cfg)

    # Train gate on diverse calibration samples so it selects high-magnitude tokens
    calib = [_make_kv(TOKENS_PER_REQUEST, seed=i) for i in range(50)]
    retention.train_retention_gate(calib, n_epochs=10, lr=0.01)

    decider_cfg = NAtHRetentionTierDeciderConfig(
        alpha=0.5, max_eviction_ratio=0.03, seed=SEED
    )
    decider = NAtHRetentionTierDecider(decider_cfg, nath, retention)

    return nath, retention, decider


def _make_kv(n_tokens: int, seed: int = SEED) -> torch.Tensor:
    """Generate structured KV where important tokens (top BUDGET_RATIO) dominate attention.

    This mirrors the accuracy test design: high-magnitude important tokens are
    selected by the gate (via norm fallback or trained weights), achieving < 1% error.
    """
    torch.manual_seed(seed)
    n_important = max(1, int(BUDGET_RATIO * n_tokens))
    kv = torch.zeros(n_tokens, N_LAYERS, N_HEADS, D_HEAD)
    kv[:n_important] = torch.randn(n_important, N_LAYERS, N_HEADS, D_HEAD) * 50.0
    kv[n_important:] = torch.randn(n_tokens - n_important, N_LAYERS, N_HEADS, D_HEAD) * 0.01
    return kv


def _simulate_decode_scores(
    nath: NAtHDDROffloadingScheduler,
    token_keys: List[str],
    n_steps: int = 20,
    seed: int = SEED,
) -> None:
    """Simulate n_steps of attention score accumulation for given tokens."""
    torch.manual_seed(seed)
    for _ in range(n_steps):
        raw = torch.zeros(len(token_keys)).exponential_(lambd=1.0)
        raw = raw / raw.sum()
        for k, s in zip(token_keys, raw.tolist()):
            nath.update_attention_score(k, s)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test: Full multi-request E2E flow
# ─────────────────────────────────────────────────────────────────────────────

class TestE2EMultiRequestDDROffloadGlobalEviction:
    """Full E2E: schedule multiple requests, store KV with global eviction, verify accuracy."""

    def test_e2e_multi_request_flow(self) -> None:
        nath, retention, decider = _build_system()

        q = torch.randn(4, D_HEAD)
        all_attention_errors: List[float] = []

        for req_idx in range(N_REQUESTS):
            # Simulate arrival
            request = {
                "id": f"req_{req_idx}",
                "token_ids": list(range(req_idx * 10, req_idx * 10 + TOKENS_PER_REQUEST)),
                "arrival_time": time.monotonic(),
            }

            # 1. Schedule request (builds token_keys, classifies tiers)
            scheduled = nath.schedule_request(request)
            assert "tier_assignment" in scheduled

            # 2. Build actual KV tensor for this request
            kv_orig = _make_kv(TOKENS_PER_REQUEST, seed=SEED + req_idx)

            # 3. Simulate decoding: update attention scores for this request's tokens
            token_keys = list(scheduled["tier_assignment"].keys())
            _simulate_decode_scores(nath, token_keys, n_steps=15, seed=SEED + req_idx)

            # 4. Dual-signal tier decision
            tier_map = decider.decide_tier(token_keys, kv_tensor=kv_orig)
            assert set(tier_map.values()).issubset({1, 2, 3, 4})

            # 5. DDR offload Tier-2 tokens
            tier2_keys = [k for k, t in tier_map.items() if t == 2]
            if tier2_keys:
                nath.offload_to_ddr(tier2_keys[0], kv_orig[:4], tier=2)

            # 6. Store compressed KV via GlobalRetentionGate
            cache_key = f"req_{req_idx}_kv"
            retention.put(cache_key, kv_orig)
            retrieved = retention.get(cache_key)
            assert retrieved is not None, f"Cache miss after put for req {req_idx}"

            # 7. Measure attention accuracy
            k_orig = kv_orig[:, 0, 0, :]
            v_orig = kv_orig[:, 0, 0, :]
            k_kept = retrieved[:, 0, 0, :]
            v_kept = retrieved[:, 0, 0, :]

            err = attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)
            all_attention_errors.append(err)

        # All attention errors must be < 1%
        for i, err in enumerate(all_attention_errors):
            assert err < 0.01, (
                f"Request {i} attention error {err:.4f} >= 1% after E2E flow"
            )

    def test_e2e_permanent_eviction_ratio_below_3pct(self) -> None:
        """Permanent eviction ratio must stay <= 3% throughout the E2E flow."""
        nath, retention, decider = _build_system()

        for req_idx in range(N_REQUESTS):
            request = {
                "id": f"req_{req_idx}",
                "token_ids": list(range(TOKENS_PER_REQUEST)),
                "arrival_time": 0.0,
            }
            scheduled = nath.schedule_request(request)
            token_keys = list(scheduled["tier_assignment"].keys())
            _simulate_decode_scores(nath, token_keys, seed=SEED + req_idx)
            decider.decide_tier(token_keys, kv_tensor=_make_kv(TOKENS_PER_REQUEST, seed=req_idx))

        eviction_ratio = nath.permanent_eviction_ratio()
        assert eviction_ratio <= 0.03 + 1e-6, (
            f"E2E permanent_eviction_ratio {eviction_ratio:.4f} exceeds 3%"
        )

    def test_e2e_scheduling_overhead_low(self) -> None:
        """Scheduling overhead p50 must stay < 5ms during E2E flow."""
        nath, retention, decider = _build_system()

        for req_idx in range(20):
            request = {
                "id": f"req_{req_idx}",
                "token_ids": list(range(64)),
                "arrival_time": time.monotonic(),
            }
            nath.schedule_request(request)

        p50 = nath.scheduling_overhead_ms_p50()
        assert p50 < 5.0, f"E2E scheduling overhead p50 {p50:.2f}ms >= 5ms"


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test: DDR prefetch round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestDDROffloadPrefetchRoundTrip:
    """Tier-2 DDR offload → prefetch must restore FP16 exactly (zero-approx-error)."""

    def test_ddr_prefetch_zero_approx_error(self) -> None:
        nath, retention, decider = _build_system()
        token_key = "round_trip:tok0"
        torch.manual_seed(SEED)
        kv_orig = torch.randn(16, 64, dtype=torch.float16)

        nath.offload_to_ddr(token_key, kv_orig, tier=2)
        prefetched = nath.prefetch_from_ddr([token_key])

        assert token_key in prefetched
        restored = prefetched[token_key].cpu()
        assert torch.allclose(restored, kv_orig.cpu()), (
            "DDR prefetch round-trip: restored tensor differs from original"
        )

    def test_ddr_tier3_restore_approximate(self) -> None:
        """Tier-3 INT8 restore must be approximately correct (<2% error)."""
        nath, retention, decider = _build_system()
        token_key = "tier3_rt:tok0"
        torch.manual_seed(SEED)
        kv_orig = torch.randn(16, 64, dtype=torch.float32)

        nath.offload_to_ddr(token_key, kv_orig, tier=3)
        restored = nath.restore_tier3_from_ddr(token_key, target_dtype=torch.float32)

        assert restored is not None
        rel_err = ((kv_orig - restored).norm() / kv_orig.norm().clamp(min=1e-8)).item()
        assert rel_err < 0.02, f"Tier-3 restore relative error {rel_err:.4f} > 2%"


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test: CacheStore interface compliance in integrated system
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheStoreInterfaceInSystem:
    """GlobalRetentionGateEvictionCodec used as CacheStore in A+C integrated system."""

    def test_cachestore_full_interface(self) -> None:
        _, retention, _ = _build_system()
        kv = _make_kv(TOKENS_PER_REQUEST)

        # put / get
        retention.put("sys_k1", kv)
        r = retention.get("sys_k1")
        assert r is not None

        # hit_rate
        assert retention.hit_rate() > 0.0

        # memory_bytes
        assert retention.memory_bytes() > 0

        # evict
        retention.put("sys_k2", kv)
        freed = retention.evict()
        assert freed > 0

        # reset_stats
        retention.reset_stats()
        assert retention.hit_rate() == 0.0

    def test_cache_hit_after_put_get_cycle(self) -> None:
        """Repeated put/get cycles must maintain correct hit/miss counts."""
        _, retention, _ = _build_system()
        kv = _make_kv(TOKENS_PER_REQUEST)

        n_puts = 5
        for i in range(n_puts):
            retention.put(f"k{i}", kv)

        # Hit each key once
        for i in range(n_puts):
            r = retention.get(f"k{i}")
            assert r is not None, f"Cache miss on key k{i} after put"

        # Miss on unknown key
        r_miss = retention.get("nonexistent")
        assert r_miss is None

        hr = retention.hit_rate()
        expected = n_puts / (n_puts + 1)   # n_puts hits, 1 miss
        assert abs(hr - expected) < 0.01, (
            f"hit_rate {hr:.4f} != expected {expected:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test: LongBench proxy (KL + cosine) under integrated system
# ─────────────────────────────────────────────────────────────────────────────

class TestLongBenchProxyIntegrated:
    """KL < 0.015 and cosine >= 0.99 when using the integrated A+C system."""

    def test_longbench_kl_and_cosine(self) -> None:
        nath, retention, decider = _build_system()

        n_tokens = 64
        kv_orig = _make_kv(n_tokens)
        q = torch.randn(4, D_HEAD)

        # Schedule + compress
        request = {
            "id": "longbench_req",
            "token_ids": list(range(n_tokens)),
            "arrival_time": 0.0,
        }
        scheduled = nath.schedule_request(request)
        token_keys = list(scheduled["tier_assignment"].keys())
        _simulate_decode_scores(nath, token_keys, n_steps=30)
        decider.decide_tier(token_keys, kv_tensor=kv_orig)

        retention.put("longbench_kv", kv_orig)
        kv_kept = retention.get("longbench_kv")
        assert kv_kept is not None

        k_orig = kv_orig[:, 0, 0, :]
        v_orig = kv_orig[:, 0, 0, :]
        k_kept = kv_kept[:, 0, 0, :]
        v_kept = kv_kept[:, 0, 0, :]

        # Pad k_kept to same token count as k_orig for KL computation
        if k_kept.shape[0] < k_orig.shape[0]:
            pad_len = k_orig.shape[0] - k_kept.shape[0]
            k_kept_padded = F.pad(k_kept, (0, 0, 0, pad_len), "constant", 0.0)
        else:
            k_kept_padded = k_kept
        kl = attention_kl_divergence(q, k_orig, k_kept_padded)
        cos = cosine_similarity_output(q, k_orig, v_orig, k_kept, v_kept)

        assert kl < 0.015, f"LongBench proxy KL {kl:.4f} >= 0.015"
        assert cos >= 0.99, f"LongBench proxy cosine {cos:.4f} < 0.99"

    def test_memory_reduction_target_met(self) -> None:
        """Integrated system must achieve >= 30% memory reduction at budget_ratio=0.3."""
        _, retention, _ = _build_system()
        ratio = retention.memory_reduction_ratio()
        assert ratio >= 0.30, f"Integrated system memory_reduction_ratio {ratio:.3f} < 30%"
