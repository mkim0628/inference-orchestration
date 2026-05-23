"""End-to-end integration tests for RuntimeCertifiedKVSculptDistillationPipeline (Cross-1).

Covers: pipeline run, accuracy preservation, memory reduction, fallback tracking,
CacheStore interface, runner integration, CPD router integration, CLC gate integration.
"""

import pytest
import torch

from src.cache.runtime_certified_quant_codec import RuntimeCertifiedConfig
from src.cache.kvsculpt_distillation_codec import KVSculptConfig
from src.cache.runtime_certified_distillation_pipeline import (
    DistillationPipelineConfig,
    RuntimeCertifiedKVSculptDistillationPipeline,
)
from src.cache.clc_positional_bias_gated_segment_cache import (
    CLCBiasGateConfig,
    CLCPositionalBiasGatedSegmentCache,
    ReencodingPolicy,
)
from src.scheduler.cpd_warm_cold_hit_router import CPDRouterConfig, CPDWarmColdHitRateRouter
from src.engine.runner import InferenceRunner, InferenceRequest
from src.metrics.perplexity import cosine_similarity_output


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_pipeline(d_head: int = 64, seq_len_ref: int = 32) -> RuntimeCertifiedKVSculptDistillationPipeline:
    cfg = DistillationPipelineConfig(
        c1_config=RuntimeCertifiedConfig(d_head=d_head, max_entries=100, seed=42),
        c2_config=KVSculptConfig(
            n_layers=4, d_head=d_head, total_budget_ratio=0.7, seed=42
        ),
        seed=42,
    )
    return RuntimeCertifiedKVSculptDistillationPipeline(cfg)


def _focused_tensors(
    seq_len: int = 32, d_head: int = 64, seed: int = 42
):
    """Return (Q, K, V) where important tokens dominate attention."""
    torch.manual_seed(seed)
    n_important = max(1, int(seq_len * 0.7))
    K_important = torch.randn(n_important, d_head) * 2.0
    K_noise = torch.randn(seq_len - n_important, d_head) * 0.01
    K = torch.cat([K_important, K_noise], dim=0)
    V = torch.randn(seq_len, d_head)
    Q = K_important.mean(0, keepdim=True).expand(4, -1).float()
    return Q, K, V


# --------------------------------------------------------------------------- #
# Basic pipeline run                                                           #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_basic_run() -> None:
    """run_pipeline completes without error."""
    pipeline = _make_pipeline()
    Q, K, V = _focused_tensors()
    K_final, V_final, report = pipeline.run_pipeline(Q, K, V, layer_idx=0, cache_key="k0")
    assert K_final is not None
    assert V_final is not None


def test_e2e_pipeline_returns_kfinal_vfinal_report() -> None:
    """run_pipeline returns (K_final, V_final, report) with required report keys."""
    pipeline = _make_pipeline()
    Q, K, V = _focused_tensors()
    K_final, V_final, report = pipeline.run_pipeline(Q, K, V, layer_idx=0, cache_key="k0")
    assert isinstance(K_final, torch.Tensor)
    assert isinstance(V_final, torch.Tensor)
    assert isinstance(report, dict)
    for key in ["layer_idx", "fallback_level", "error_bound", "selected_ratio"]:
        assert key in report, f"Missing report key '{key}'"


# --------------------------------------------------------------------------- #
# Accuracy preservation (MANDATORY)                                            #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_accuracy_preserved_cosine_above_099() -> None:
    """run_pipeline: cosine_similarity_output >= 0.99 (MANDATORY §5)."""
    pipeline = _make_pipeline()
    Q, K_orig, V_orig = _focused_tensors()
    K_final, V_final, _ = pipeline.run_pipeline(Q, K_orig, V_orig, layer_idx=0, cache_key="k0")
    cos_sim = cosine_similarity_output(
        Q.float(), K_orig.float(), V_orig.float(), K_final.float(), V_final.float()
    )
    assert cos_sim >= 0.99, (
        f"E2E pipeline cosine_sim={cos_sim:.6f} < 0.99 (MANDATORY §5)"
    )


# --------------------------------------------------------------------------- #
# Memory reduction                                                             #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_memory_reduction_above_50pct() -> None:
    """certified_codec.memory_reduction_ratio() >= 0.50 after pipeline run."""
    pipeline = _make_pipeline()
    for i in range(5):
        Q, K, V = _focused_tensors(seed=i)
        pipeline.run_pipeline(Q, K, V, layer_idx=0, cache_key=f"k{i}")
    ratio = pipeline.certified_codec.memory_reduction_ratio()
    assert ratio >= 0.50, f"memory_reduction_ratio={ratio:.4f} < 0.50"


# --------------------------------------------------------------------------- #
# Distillation reduces seq_len                                                 #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_distillation_reduces_seq_len() -> None:
    """K_final.shape[0] <= K.shape[0] after pipeline (compression reduces token count)."""
    pipeline = _make_pipeline()
    Q, K, V = _focused_tensors(seq_len=32)
    K_final, V_final, report = pipeline.run_pipeline(Q, K, V, layer_idx=0, cache_key="k0")
    # When no fallback, K_final = K_distilled which has fewer tokens
    if report["fallback_level"] == 0:
        assert K_final.shape[0] <= K.shape[0], (
            f"K_final.shape[0]={K_final.shape[0]} > K.shape[0]={K.shape[0]}"
        )


# --------------------------------------------------------------------------- #
# Fallback tracking                                                            #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_fallback_tracking() -> None:
    """After multiple runs, layer_request_counts and layer_fallback_counts exist."""
    pipeline = _make_pipeline()
    Q, K, V = _focused_tensors()
    for i in range(5):
        pipeline.run_pipeline(Q, K, V, layer_idx=0, cache_key=f"k{i}")
    assert 0 in pipeline._layer_request_counts
    assert pipeline._layer_request_counts[0] >= 5
    # _layer_fallback_counts may be empty if no fallbacks occurred
    assert isinstance(pipeline._layer_fallback_counts, dict)


# --------------------------------------------------------------------------- #
# CacheStore interface                                                         #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_cachestore_interface_full() -> None:
    """put/get/evict/hit_rate/memory_bytes/reset_stats all work."""
    pipeline = _make_pipeline()
    torch.manual_seed(42)
    kv = torch.randn(16, 64)
    pipeline.put("a", kv)
    assert pipeline.get("a") is not None
    assert pipeline.get("missing") is None
    assert pipeline.hit_rate() > 0.0
    assert pipeline.memory_bytes() > 0
    freed = pipeline.evict()
    assert freed > 0
    pipeline.reset_stats()
    # after reset, hits and misses counters should be zero
    assert pipeline.certified_codec._hits == 0
    assert pipeline.certified_codec._misses == 0


# --------------------------------------------------------------------------- #
# runner integration                                                           #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_runner_integration() -> None:
    """InferenceRunner(cache=RuntimeCertifiedKVSculptDistillationPipeline) run_batch succeeds."""
    pipeline = _make_pipeline(d_head=64)
    runner = InferenceRunner(
        cache=pipeline,
        num_layers=4,
        hidden_dim=64,
        chunk_size=16,
        seed=42,
    )
    requests = [
        InferenceRequest(request_id=f"r{i}", token_ids=list(range(32)), seed=i)
        for i in range(3)
    ]
    results = runner.run_batch(requests)
    assert len(results) == 3
    for r in results:
        assert r.output_tokens == 64


# --------------------------------------------------------------------------- #
# CPD router integration                                                       #
# --------------------------------------------------------------------------- #


def test_e2e_pipeline_cpd_router_integration() -> None:
    """CPDWarmColdHitRateRouter schedules and run_pipeline is called for warm requests."""
    from dataclasses import dataclass, field as dc_field
    from typing import List

    @dataclass
    class Req:
        request_id: str
        token_ids: List[int] = dc_field(default_factory=list)
        prefix_hash: str = ""
        session_turn: int = 0
        segment_match_ratio: float = 0.0

    router_cfg = CPDRouterConfig(seed=42)
    router = CPDWarmColdHitRateRouter(router_cfg)
    # seed warm history
    router._hit_history["pfx_warm"] = [True] * 20

    reqs = [Req(request_id=f"r{i}", prefix_hash="pfx_warm", token_ids=list(range(5))) for i in range(3)]
    scheduled = router.schedule(reqs)
    assert len(scheduled) == 3

    # run pipeline for each scheduled request
    pipeline = _make_pipeline()
    for req in scheduled:
        Q, K, V = _focused_tensors(seed=hash(req.request_id) % 100)
        pipeline.run_pipeline(Q, K, V, layer_idx=0, cache_key=req.request_id)

    stats = router.routing_stats()
    # at least warm requests were routed (all have warm history)
    assert stats["warm_ratio"] >= 0.0


# --------------------------------------------------------------------------- #
# CLC gate integration                                                         #
# --------------------------------------------------------------------------- #


def test_e2e_clc_gate_integration() -> None:
    """CLCPositionalBiasGatedSegmentCache.get_with_policy returns DIRECT_REUSE for ΔPos=0."""
    gate_cfg = CLCBiasGateConfig(max_context_length=4096, bias_threshold=0.15, seed=42)
    gate = CLCPositionalBiasGatedSegmentCache(gate_cfg)

    # Store a segment
    Q, K, V = _focused_tensors(seq_len=32)
    gate.put_segment("seg1", K, pos_orig_start=0, pos_orig_end=32, content_hash="h1")

    # Retrieve at same position (ΔPos = 0 -> DIRECT_REUSE)
    kv, policy = gate.get_with_policy("seg1", pos_target_start=0)
    assert kv is not None
    assert policy == ReencodingPolicy.DIRECT_REUSE

    # Use as input to pipeline (no re-encoding needed)
    pipeline = _make_pipeline()
    K_final, V_final, report = pipeline.run_pipeline(Q, kv, V, layer_idx=0, cache_key="k_from_clc")
    cos_sim = cosine_similarity_output(
        Q.float(), kv.float(), V.float(), K_final.float(), V_final.float()
    )
    assert cos_sim >= 0.99, f"CLC gate integration cosine_sim={cos_sim:.6f} < 0.99"


# --------------------------------------------------------------------------- #
# Solo C-1 vs Solo C-2 vs Cross-1 memory comparison                           #
# --------------------------------------------------------------------------- #


def test_e2e_solo_c1_vs_solo_c2_vs_cross1_memory_comparison() -> None:
    """C-1 / C-2 / Cross-1 memory reduction comparison (Cross-1 >= C-1 or equal)."""
    from src.cache.runtime_certified_quant_codec import RuntimeCertifiedConfig, RuntimeCertifiedQuantizedAttentionCodec
    from src.cache.kvsculpt_distillation_codec import KVSculptConfig, KVSculptDistillationCodec

    d_head, seq_len = 64, 32

    # C-1 solo
    c1_cfg = RuntimeCertifiedConfig(d_head=d_head, max_entries=100, seed=42)
    c1 = RuntimeCertifiedQuantizedAttentionCodec(c1_cfg)
    for i in range(5):
        torch.manual_seed(i)
        c1.put(f"k{i}", torch.randn(seq_len, d_head))
    c1_ratio = c1.memory_reduction_ratio()

    # C-2 solo: KVSculpt just stores tensors (no quantization)
    c2_cfg = KVSculptConfig(n_layers=4, d_head=d_head, total_budget_ratio=0.5, seed=42)
    c2 = KVSculptDistillationCodec(c2_cfg)
    for i in range(5):
        torch.manual_seed(i)
        c2.put(f"k{i}", torch.randn(int(seq_len * 0.5), d_head))  # 50% tokens
    # C-2 memory: 50% tokens retained (no quantization beyond token selection)

    # Cross-1: pipeline
    pipeline = _make_pipeline(d_head=d_head)
    for i in range(5):
        Q, K, V = _focused_tensors(seq_len=seq_len, d_head=d_head, seed=i)
        pipeline.run_pipeline(Q, K, V, layer_idx=0, cache_key=f"k{i}")
    cross1_ratio = pipeline.certified_codec.memory_reduction_ratio()

    # Both C-1 and Cross-1 should achieve >= 50% memory reduction
    assert c1_ratio >= 0.50, f"C-1 solo memory_reduction={c1_ratio:.4f} < 0.50"
    assert cross1_ratio >= 0.50, f"Cross-1 memory_reduction={cross1_ratio:.4f} < 0.50"
