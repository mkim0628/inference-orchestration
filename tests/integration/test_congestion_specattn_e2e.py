"""End-to-end integration tests for CongestionAdmissionSpecAttnDualReductionPipeline.

Cross Activity A+C: CONCUR admission scheduler + SpecAttn KV sparsification.
Validates the full pipeline from logit injection through scheduling, compression,
and accuracy preservation (evaluation_criteria.md §5 MANDATORY).
"""

import pytest
import torch
import torch.nn.functional as F

from src.cache.specattn_sparse_codec import SpecAttnCodecConfig, SpecAttnVerificationGuidedKVSparseCodec
from src.cache.congestion_specattn_pipeline import (
    CongestionAdmissionSpecAttnDualReductionPipeline,
    DualReductionConfig,
)
from src.scheduler.concur_congestion_admission_scheduler import (
    CONCURCongestionBasedAgentAdmissionScheduler,
    CongestionAdmissionConfig,
)
from src.metrics.perplexity import cosine_similarity_output
from src.engine.runner import InferenceRunner, InferenceRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_KV = 100
D_HEAD = 64
SEED = 42


@pytest.fixture
def pipeline_free() -> CongestionAdmissionSpecAttnDualReductionPipeline:
    """Pipeline with KV pool in FREE state (30% occupancy)."""
    cfg = DualReductionConfig(
        scheduler_config=CongestionAdmissionConfig(
            capacity_bytes=1_000_000,
            alpha_low=0.60,
            alpha_high=0.85,
        ),
        codec_config=SpecAttnCodecConfig(
            retention_ratio_by_layer=[0.80] * 12,
            global_retention_ratio=0.80,
        ),
        codec_adapt_on_congestion=True,
        retention_reduction_on_congestion=0.10,
        seed=SEED,
    )
    p = CongestionAdmissionSpecAttnDualReductionPipeline(cfg)
    p.update_kv_pool(300_000)  # 30% → FREE
    return p


@pytest.fixture
def pipeline_congested() -> CongestionAdmissionSpecAttnDualReductionPipeline:
    """Pipeline with KV pool in CONGESTED state (90% occupancy)."""
    cfg = DualReductionConfig(
        scheduler_config=CongestionAdmissionConfig(
            capacity_bytes=1_000_000,
            alpha_low=0.60,
            alpha_high=0.85,
        ),
        codec_config=SpecAttnCodecConfig(
            retention_ratio_by_layer=[0.80] * 12,
            global_retention_ratio=0.80,
        ),
        codec_adapt_on_congestion=True,
        retention_reduction_on_congestion=0.10,
        seed=SEED,
    )
    p = CongestionAdmissionSpecAttnDualReductionPipeline(cfg)
    p.update_kv_pool(900_000)  # 90% → CONGESTED
    return p


def _logits(n_kv: int = N_KV, seed: int = SEED) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(4, 8, n_kv)


def _kv(seed: int = SEED) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N_KV, D_HEAD)


def _q(seed: int = SEED + 100) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(8, D_HEAD)


# ---------------------------------------------------------------------------
# Basic put/get tests
# ---------------------------------------------------------------------------

def test_e2e_put_get_basic(
    pipeline_free: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """put → get roundtrip returns a tensor."""
    kv = _kv()
    pipeline_free.put("key_a", kv)
    result = pipeline_free.get("key_a")
    assert result is not None
    assert result.shape == kv.shape


def test_e2e_set_logits_then_put_applies_mask(
    pipeline_free: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """set_verification_logits → put → get_importance_mask() returns bool mask."""
    logits = _logits()
    pipeline_free.set_verification_logits(logits, layer_idx=0)
    kv = _kv()
    pipeline_free.put("masked_key", kv)
    mask = pipeline_free.get_importance_mask("masked_key")
    assert mask is not None
    assert mask.dtype == torch.bool
    assert mask.shape[0] == N_KV


def test_e2e_put_get_miss_returns_none(
    pipeline_free: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """get() on an unknown key returns None."""
    assert pipeline_free.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Congestion-triggered retention ratio adjustment
# ---------------------------------------------------------------------------

def test_e2e_congestion_triggers_retention_reduction(
    pipeline_congested: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """CONGESTED state → codec retention_ratios reduced by retention_reduction_on_congestion."""
    base = pipeline_congested._base_retention_ratios[0]
    current = pipeline_congested.codec.config.retention_ratio_by_layer[0]
    assert current < base, (
        f"Congestion should reduce retention ratio: base={base}, current={current}"
    )


def test_e2e_free_restores_retention(
    pipeline_congested: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """Moving from CONGESTED → FREE restores codec retention_ratios."""
    base = pipeline_congested._base_retention_ratios[0]
    # Move to FREE state
    pipeline_congested.update_kv_pool(200_000)  # 20% < 60% → FREE
    current = pipeline_congested.codec.config.retention_ratio_by_layer[0]
    assert abs(current - base) < 1e-9, (
        f"FREE state should restore retention_ratio to {base}, got {current}"
    )


# ---------------------------------------------------------------------------
# Scheduling behaviour
# ---------------------------------------------------------------------------

def test_e2e_schedule_during_congestion(
    pipeline_congested: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """CONGESTED state: schedule() returns empty list."""
    reqs = [object() for _ in range(10)]
    result = pipeline_congested.schedule(reqs)
    assert result == []


def test_e2e_schedule_during_free(
    pipeline_free: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """FREE state: schedule() returns all requests."""
    reqs = [object() for _ in range(5)]
    result = pipeline_free.schedule(reqs)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Accuracy preservation — MANDATORY (evaluation_criteria.md §5)
# ---------------------------------------------------------------------------

def test_e2e_accuracy_preserved_cosine_above_099(
    pipeline_free: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """retention_ratio=0.80: put → get cosine_sim >= 0.99 (MANDATORY §5)."""
    logits = _logits()
    pipeline_free.set_verification_logits(logits, layer_idx=0)

    torch.manual_seed(SEED)
    k_orig = torch.randn(N_KV, D_HEAD)
    pipeline_free.put("k_acc", k_orig)
    k_comp = pipeline_free.get("k_acc")

    logits2 = _logits(seed=SEED + 10)
    pipeline_free.set_verification_logits(logits2, layer_idx=0)
    torch.manual_seed(SEED + 1)
    v_orig = torch.randn(N_KV, D_HEAD)
    pipeline_free.put("v_acc", v_orig)
    v_comp = pipeline_free.get("v_acc")

    q = _q()
    cos_sim = cosine_similarity_output(
        q.float(), k_orig.float(), v_orig.float(),
        k_comp.float(), v_comp.float()
    )
    assert cos_sim >= 0.99, (
        f"E2E cosine_sim={cos_sim:.6f} < 0.99 (MANDATORY §5 violated)"
    )


# ---------------------------------------------------------------------------
# Memory reduction
# ---------------------------------------------------------------------------

def test_e2e_memory_reduction_above_30pct() -> None:
    """retention_ratio=0.70 pipeline: memory_reduction_ratio() >= 0.30 (MANDATORY)."""
    cfg = DualReductionConfig(
        scheduler_config=CongestionAdmissionConfig(capacity_bytes=1_000_000),
        codec_config=SpecAttnCodecConfig(
            retention_ratio_by_layer=[0.70] * 12,
            global_retention_ratio=0.70,
            low_importance_quant_int4=True,
        ),
    )
    pipeline = CongestionAdmissionSpecAttnDualReductionPipeline(cfg)
    pipeline.update_kv_pool(200_000)  # FREE state

    for i in range(50):
        torch.manual_seed(i)
        logits = torch.randn(4, 8, N_KV)
        pipeline.set_verification_logits(logits, layer_idx=0)
        kv = torch.randn(N_KV, D_HEAD)
        pipeline.put(f"key_{i}", kv)

    ratio = pipeline.codec.memory_reduction_ratio()
    assert ratio >= 0.30, (
        f"memory_reduction_ratio={ratio:.4f} < 0.30 (retention=0.70 → 30% evicted)"
    )
    assert pipeline.codec._total_tokens_original > 0


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def test_e2e_metrics_summary_all_keys(
    pipeline_free: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """metrics_summary() contains all required keys."""
    summary = pipeline_free.metrics_summary()
    required_keys = [
        "scheduler_overhead_ms_p50",
        "kv_pool_occupancy",
        "congestion_level",
        "codec_hit_rate",
        "codec_memory_reduction_ratio",
        "current_retention_ratios",
        "total_memory_bytes",
    ]
    for k in required_keys:
        assert k in summary, f"metrics_summary() missing key: {k}"


# ---------------------------------------------------------------------------
# Full CacheStore interface
# ---------------------------------------------------------------------------

def test_e2e_cachestore_interface_full(
    pipeline_free: CongestionAdmissionSpecAttnDualReductionPipeline,
) -> None:
    """All CacheStore abstract methods work: put/get/evict/hit_rate/memory_bytes/reset_stats."""
    kv = _kv()
    pipeline_free.put("x", kv)
    assert pipeline_free.get("x") is not None
    assert pipeline_free.get("missing") is None
    assert pipeline_free.hit_rate() > 0.0
    freed = pipeline_free.evict()
    assert freed >= 0
    mb = pipeline_free.memory_bytes()
    assert mb >= 0
    pipeline_free.reset_stats()
    assert pipeline_free.hit_rate() == 0.0


# ---------------------------------------------------------------------------
# Cross A+C vs. solo A comparison
# ---------------------------------------------------------------------------

def test_e2e_cross_ac_vs_solo_throughput() -> None:
    """Cross-1 admitted count >= A-1 solo admitted count (additive benefit)."""
    capacity = 1_000_000

    # Solo A-1 scheduler
    sched_cfg = CongestionAdmissionConfig(
        capacity_bytes=capacity,
        alpha_low=0.60,
        alpha_high=0.85,
    )
    solo_scheduler = CONCURCongestionBasedAgentAdmissionScheduler(sched_cfg)
    solo_scheduler.update_kv_pool(300_000)  # FREE

    reqs = list(range(20))
    solo_result = solo_scheduler.schedule(reqs)

    # Cross-1 pipeline (same pool state)
    cfg = DualReductionConfig(
        scheduler_config=sched_cfg,
        codec_config=SpecAttnCodecConfig(retention_ratio_by_layer=[0.80] * 12),
    )
    cross_pipeline = CongestionAdmissionSpecAttnDualReductionPipeline(cfg)
    cross_pipeline.update_kv_pool(300_000)

    cross_result = cross_pipeline.schedule(reqs)
    # Both in FREE state: both admit all 20
    assert len(cross_result) >= len(solo_result)


# ---------------------------------------------------------------------------
# InferenceRunner integration
# ---------------------------------------------------------------------------

def test_e2e_runner_integration() -> None:
    """InferenceRunner with DualReductionPipeline as cache runs run_batch() successfully."""
    cfg = DualReductionConfig(
        scheduler_config=CongestionAdmissionConfig(capacity_bytes=1_000_000_000),
        codec_config=SpecAttnCodecConfig(
            retention_ratio_by_layer=[0.80] * 12,
            global_retention_ratio=0.80,
        ),
        seed=SEED,
    )
    pipeline = CongestionAdmissionSpecAttnDualReductionPipeline(cfg)
    pipeline.update_kv_pool(100_000)  # FREE

    runner = InferenceRunner(
        cache=pipeline,
        scheduler=pipeline,
        num_layers=12,
        hidden_dim=D_HEAD,
        chunk_size=32,
        seed=SEED,
    )

    requests = [
        InferenceRequest(request_id=f"req_{i}", token_ids=list(range(64)), seed=i)
        for i in range(3)
    ]
    results = runner.run_batch(requests)
    assert len(results) == 3
    for r in results:
        assert r.ttft_ms >= 0.0
        assert r.output_tokens == 64
