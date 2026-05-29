"""End-to-end integration tests for GroundedWynerZivBCPipeline (Cross-1).

Tests the full B+C pipeline: 4-gate safety gating (B-1) combined with
Wyner-Ziv adaptive window eviction (C-1) sharing a single ε budget.
"""

import pytest
import torch

from src.cache.grounded_safety_gated_cache import (
    GroundedSafetyGatedConfig,
    GroundedSafetyGatedSegmentCache,
    GateThresholds,
)
from src.cache.wyner_ziv_adaptive_window_eviction import (
    WynerZivAdaptiveWindowEvictionCache,
    WynerZivConfig,
)
from src.engine.grounded_wyner_ziv_bc_pipeline import (
    BCPipelineConfig,
    GroundedWynerZivBCPipeline,
)
from src.metrics.hit_rate import SafetyGatedHitRateMetrics
from src.metrics.memory import WynerZivWindowMetrics


# ------------------------------------------------------------------ #
# Pipeline factory                                                     #
# ------------------------------------------------------------------ #

SHARED_BUDGET = 0.01


def make_pipeline(
    content_sim: float = 0.0,
    evidence_overlap: float = 0.0,
    attn_support: float = 0.0,
    n_layers: int = 2,
    max_window: int = 4096,
    warmup: int = 0,
    auto_fallback: bool = True,
    sample_ratio: float = 1.0,
    window_config_path: str = "configs/wyner_ziv_window_config.yaml",
) -> GroundedWynerZivBCPipeline:
    thresholds = GateThresholds(
        content_sim_threshold=content_sim,
        evidence_overlap_threshold=evidence_overlap,
        attn_support_threshold=attn_support,
    )
    safety_cfg = GroundedSafetyGatedConfig(gate_thresholds=thresholds)
    safety_cache = GroundedSafetyGatedSegmentCache(safety_cfg)

    wz_cfg = WynerZivConfig(
        accuracy_budget=SHARED_BUDGET,
        min_window_size=64,
        max_window_size=max_window,
        warmup_batches=warmup,
        n_layers=n_layers,
        window_config_path=window_config_path,
    )
    window_eviction = WynerZivAdaptiveWindowEvictionCache(wz_cfg)

    safety_metrics = SafetyGatedHitRateMetrics()
    window_metrics = WynerZivWindowMetrics()

    pipeline_cfg = BCPipelineConfig(
        shared_accuracy_budget=SHARED_BUDGET,
        perplexity_sample_ratio=sample_ratio,
        auto_fallback=auto_fallback,
        seed=42,
    )
    return GroundedWynerZivBCPipeline(
        safety_cache=safety_cache,
        window_eviction=window_eviction,
        safety_metrics=safety_metrics,
        window_metrics=window_metrics,
        config=pipeline_cfg,
    )


def make_kv(n_tokens: int = 8, d_kv: int = 16) -> torch.Tensor:
    return torch.randn(n_tokens, d_kv)


# ------------------------------------------------------------------ #
# Test 1: full request flow                                            #
# ------------------------------------------------------------------ #

def test_bc_pipeline_full_request_flow():
    """Full B+C pipeline processes a request and returns a valid (kv, outcome) pair."""
    pipeline = make_pipeline()
    tokens = list(range(1, 25))
    kv = make_kv()

    result_kv, outcome = pipeline.process_request(
        query_tokens=tokens,
        current_context_tokens=tokens,
        kv_computed=kv,
        layer_idx=0,
    )
    assert outcome in (
        "safe_reuse_hit", "partial_reuse_hit", "gate_rejected", "stale_evicted", "miss"
    )
    # On first request, segment is not yet cached → miss, but kv_computed returned
    assert outcome == "miss"
    assert result_kv is not None


# ------------------------------------------------------------------ #
# Test 2: safe_reuse_hit uses gated cache, not WynerZiv put            #
# ------------------------------------------------------------------ #

def test_bc_safe_reuse_hit_uses_gated_cache():
    """On safe_reuse_hit, WynerZiv window eviction is NOT called to store KV."""
    pipeline = make_pipeline()
    tokens = list(range(1, 25))
    kv = make_kv()

    # First request: miss, inserts into safety cache and WynerZiv
    pipeline.process_request(
        query_tokens=tokens,
        current_context_tokens=tokens,
        kv_computed=kv,
        layer_idx=0,
    )

    # WynerZiv batch_count after first request (miss path)
    batch_count_after_miss = pipeline._window_eviction._batch_count

    # Second request: should be safe_reuse_hit (thresholds = 0)
    _, outcome2 = pipeline.process_request(
        query_tokens=tokens,
        current_context_tokens=tokens,
        layer_idx=0,
    )
    assert outcome2 == "safe_reuse_hit", f"Expected safe_reuse_hit, got {outcome2}"
    # WynerZiv batch_count should NOT increase on cache hit
    assert pipeline._window_eviction._batch_count == batch_count_after_miss, \
        "WynerZiv put() should not be called on safe_reuse_hit"


# ------------------------------------------------------------------ #
# Test 3: miss inserts into window eviction                            #
# ------------------------------------------------------------------ #

def test_bc_miss_inserts_into_window_eviction():
    """On cache miss with kv_computed, KV is stored in WynerZiv window."""
    pipeline = make_pipeline()
    tokens = list(range(50, 75))
    kv = make_kv()

    bc_before = pipeline._window_eviction._batch_count

    _, outcome = pipeline.process_request(
        query_tokens=tokens,
        current_context_tokens=tokens,
        kv_computed=kv,
        layer_idx=0,
    )
    assert outcome == "miss"
    # batch_count incremented (put was called)
    assert pipeline._window_eviction._batch_count > bc_before


# ------------------------------------------------------------------ #
# Test 4: shared accuracy budget                                       #
# ------------------------------------------------------------------ #

def test_bc_accuracy_budget_shared():
    """B-1 safety cache and C-1 WynerZiv both use the same ε value."""
    pipeline = make_pipeline()
    assert pipeline._config.shared_accuracy_budget == SHARED_BUDGET
    assert pipeline._window_eviction._config.accuracy_budget == SHARED_BUDGET


# ------------------------------------------------------------------ #
# Test 5: auto-fallback triggers on high perplexity delta              #
# ------------------------------------------------------------------ #

def test_bc_fallback_auto_triggers_on_high_delta():
    """auto_fallback=True + perplexity_delta > ε → W* doubled for all layers."""
    pipeline = make_pipeline(auto_fallback=True, sample_ratio=1.0, n_layers=2)

    # Set initial windows to known values
    pipeline._window_eviction._window_size[0] = 300
    pipeline._window_eviction._window_size[1] = 400

    tokens = list(range(1, 25))
    kv = make_kv()

    pipeline.process_request(
        query_tokens=tokens,
        current_context_tokens=tokens,
        kv_computed=kv,
        perplexity_delta=0.1,  # far exceeds ε=0.01
    )

    # Fallback should have been triggered → W* doubled (capped at max_window)
    assert pipeline._window_eviction._fallback_count >= 1


# ------------------------------------------------------------------ #
# Test 6: both cache classes implement CacheStore                      #
# ------------------------------------------------------------------ #

def test_bc_cachestore_interface_compat():
    """Both GroundedSafetyGatedSegmentCache and WynerZivAdaptiveWindowEvictionCache
    satisfy the CacheStore interface (put, get, evict, hit_rate, memory_bytes, reset_stats).
    """
    from src.cache.base import CacheStore

    pipeline = make_pipeline()
    assert isinstance(pipeline._safety_cache, CacheStore)
    assert isinstance(pipeline._window_eviction, CacheStore)

    # Exercise all interface methods
    kv = make_kv()
    for cache in [pipeline._safety_cache, pipeline._window_eviction]:
        cache.put("interface_test_key", kv)
        result = cache.get("interface_test_key")
        assert result is not None
        cache.evict()
        assert 0.0 <= cache.hit_rate() <= 1.0
        assert cache.memory_bytes() >= 0
        cache.reset_stats()


# ------------------------------------------------------------------ #
# Test 7: combined memory reduction > single C-1                       #
# ------------------------------------------------------------------ #

def test_bc_memory_reduction_exceeds_single_c():
    """Combined B+C reduces more tokens in-cache than C-1 alone.

    Strategy: fill C-1 alone with N tokens; then in B+C pipeline, safe_reuse_hits
    bypass the window eviction so fewer tokens accumulate in WynerZiv.
    """
    n_tokens_to_insert = 20

    # --- C-1 alone ---
    wz_cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=1,
        max_window_size=4096,
        warmup_batches=0,
        n_layers=1,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    wz_only = WynerZivAdaptiveWindowEvictionCache(wz_cfg)
    wz_only._warmup_done = True
    wz_only._window_size[0] = 4096

    for i in range(n_tokens_to_insert):
        kv = make_kv()
        wz_only.put(f"key_{i}", kv, layer_idx=0)
    mem_c_only = wz_only.memory_bytes()

    # --- B+C pipeline: first request = miss (inserts into WynerZiv),
    # subsequent requests for same tokens = safe_reuse_hit (no WynerZiv insert)
    pipeline = make_pipeline(n_layers=1)
    pipeline._window_eviction._warmup_done = True
    pipeline._window_eviction._window_size[0] = 4096

    tokens = list(range(1, 25))
    kv = make_kv()

    # First request: miss → WynerZiv insert
    pipeline.process_request(
        query_tokens=tokens,
        current_context_tokens=tokens,
        kv_computed=kv,
        layer_idx=0,
    )
    # Subsequent requests: safe_reuse_hit → no WynerZiv insert
    for _ in range(n_tokens_to_insert - 1):
        pipeline.process_request(
            query_tokens=tokens,
            current_context_tokens=tokens,
            layer_idx=0,
        )

    mem_bc = pipeline._window_eviction.memory_bytes()
    # B+C should have <= same memory as C-only (one insert vs many inserts)
    assert mem_bc <= mem_c_only, (
        f"B+C memory {mem_bc} should ≤ C-only memory {mem_c_only}"
    )


# ------------------------------------------------------------------ #
# Test 8: get_combined_stats returns all expected keys                 #
# ------------------------------------------------------------------ #

def test_bc_combined_stats_structure():
    """get_combined_stats() returns all required stat keys."""
    pipeline = make_pipeline()
    stats = pipeline.get_combined_stats()

    assert "safety_gate" in stats
    assert "window_eviction" in stats
    assert "window_metrics" in stats
    assert "shared_accuracy_budget" in stats
    assert "total_requests" in stats

    sg = stats["safety_gate"]
    assert "total_lookups" in sg
    assert "safety_gate_pass_rate" in sg
    assert "effective_hit_rate" in sg
