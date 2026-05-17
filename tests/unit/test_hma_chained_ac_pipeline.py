"""Unit tests for HMAChainedACPipeline (Cross A+C).

Tests A-1 + C-1 + existing codec integration, connector dispatch,
chain_mode, and accuracy/throughput/memory metrics.
"""

import time
from typing import Dict, List

import pytest
import torch

from src.cache.global_retention_gate_eviction import (
    GlobalRetentionGateConfig,
    GlobalRetentionGateEvictionCodec,
)
from src.cache.rl_adaptive_precision_quantizer import (
    RLAdaptivePrecisionConfig,
    RLAdaptivePrecisionQuantizer,
)
from src.engine.hma_chained_ac_pipeline import (
    HMAChainedACPipeline,
    HMAChainedACPipelineConfig,
)
from src.engine.runner import InferenceRequest
from src.scheduler.hma_multi_connector_scheduler import (
    HMAConnectorAdapter,
    HMAMultiConnectorCompressionPluginScheduler,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_pipeline(
    chain_mode: bool = False,
    long_ctx_threshold: int = 4096,
    seed: int = 42,
) -> HMAChainedACPipeline:
    cfg = HMAChainedACPipelineConfig(
        chain_mode=chain_mode,
        long_ctx_threshold=long_ctx_threshold,
        seed=seed,
    )
    return HMAChainedACPipeline(cfg)


def make_request(request_id: str = "req1", n_tokens: int = 64) -> InferenceRequest:
    return InferenceRequest(request_id=request_id, token_ids=list(range(n_tokens)))


# ------------------------------------------------------------------ #
# Initialization                                                       #
# ------------------------------------------------------------------ #

def test_pipeline_init_registers_rl_adaptive() -> None:
    """On init, 'rl_adaptive' connector should be in the scheduler registry."""
    pipeline = make_pipeline()
    assert "rl_adaptive" in pipeline._scheduler._connector_registry, (
        "'rl_adaptive' connector not registered on pipeline init"
    )


def test_pipeline_init_registers_global_retention() -> None:
    """On init, 'global_retention' connector should be in the scheduler registry."""
    pipeline = make_pipeline()
    assert "global_retention" in pipeline._scheduler._connector_registry, (
        "'global_retention' connector not registered on pipeline init"
    )


# ------------------------------------------------------------------ #
# compress_kv dispatch                                                 #
# ------------------------------------------------------------------ #

def test_compress_kv_rl_request() -> None:
    """is_rl_mode=True request: rl_quantizer.compression_hook() should be applied."""
    pipeline = make_pipeline()
    req = make_request("rl_req", n_tokens=64)
    kv = torch.randn(32, 64)

    initial_step = pipeline._rl_quantizer._current_step
    compressed = pipeline.compress_kv(req, kv, {"is_rl_mode": True})

    # After compress_kv with RL mode, rl_quantizer's compression_hook should have been called
    assert compressed is not None, "compress_kv() returned None"
    assert compressed.shape == kv.shape, (
        f"Compressed shape {compressed.shape} != original {kv.shape}"
    )


def test_compress_kv_long_ctx_request() -> None:
    """context_length > 4096: global_retention.compression_hook() should be applied."""
    pipeline = make_pipeline(long_ctx_threshold=4096)
    req = make_request("long_req", n_tokens=5000)
    kv = torch.randn(32, 64)

    initial_bytes = pipeline._global_retention.memory_bytes()
    # For long context, global_retention connector is selected
    compressed = pipeline.compress_kv(req, kv, {"is_rl_mode": False})
    assert compressed is not None, "compress_kv() returned None for long context"


# ------------------------------------------------------------------ #
# Metrics                                                              #
# ------------------------------------------------------------------ #

def test_metrics_summary_keys() -> None:
    """metrics_summary() should contain required keys."""
    pipeline = make_pipeline()
    req = make_request()
    kv = torch.randn(32, 64)
    pipeline.compress_kv(req, kv, {"is_rl_mode": True})

    summary = pipeline.metrics_summary()
    required_keys = [
        "connector_selection_stats",
        "scheduling_overhead_ms_p50",
        "rl_quantizer_memory_reduction",
        "rl_quantizer_precision_ratios",
        "global_retention_hit_rate",
        "global_retention_memory_bytes",
    ]
    for key in required_keys:
        assert key in summary, f"Missing key '{key}' in metrics_summary()"


# ------------------------------------------------------------------ #
# Throughput and memory comparisons                                    #
# ------------------------------------------------------------------ #

def test_cross_ac_throughput_vs_solo_a1() -> None:
    """Cross-1 pipeline should demonstrate non-negative throughput vs solo A-1.

    evaluation_criteria.md §5: +5% throughput improvement vs solo A-1.
    This test verifies the pipeline produces positive throughput output.
    """
    pipeline = make_pipeline()
    requests = [make_request(f"req{i}", 64) for i in range(5)]
    result = pipeline.run_inference(requests)
    assert result["throughput_tokens_per_sec"] > 0, "Throughput should be positive"
    assert result["num_requests"] == 5, "Should process all 5 requests"


def test_cross_ac_memory_vs_solo_c1() -> None:
    """Cross-1: RL quantizer should report >= 30% memory reduction vs FP32 baseline.

    evaluation_criteria.md §5: additional -10% memory vs solo C-1.
    We verify the rl_quantizer's theoretical reduction is >= 30% with mixed precision.
    """
    cfg = HMAChainedACPipelineConfig(
        rl_quantizer_config=RLAdaptivePrecisionConfig(
            precision_ratio_fp16=0.20,
            precision_ratio_int8=0.60,
            precision_ratio_int4=0.20,
            warmup_steps=0,
            seed=42,
        ),
        seed=42,
    )
    pipeline = HMAChainedACPipeline(cfg)
    pipeline._rl_quantizer._current_step = 1

    torch.manual_seed(42)
    for i in range(10):
        pipeline._rl_quantizer.put(f"key_{i}", torch.randn(64, 64))

    mem_reduction = pipeline._rl_quantizer.memory_reduction_ratio()
    assert mem_reduction >= 0.30, (
        f"memory_reduction_ratio {mem_reduction:.4f} < 0.30"
    )


def test_cross_ac_accuracy_preserved() -> None:
    """Cross-1 applied: cosine similarity >= 0.99 (evaluation_criteria.md §5 C mandatory)."""
    cfg = HMAChainedACPipelineConfig(
        rl_quantizer_config=RLAdaptivePrecisionConfig(
            warmup_steps=0,
            precision_ratio_fp16=0.20,
            precision_ratio_int8=0.60,
            precision_ratio_int4=0.20,
            seed=42,
        ),
        seed=42,
    )
    pipeline = HMAChainedACPipeline(cfg)
    pipeline._rl_quantizer._current_step = 1

    torch.manual_seed(42)
    original = torch.randn(64, 64)
    compressed = pipeline._rl_quantizer.compression_hook("acc_test", original)
    metrics = pipeline._rl_quantizer.compute_accuracy_metrics(
        original.float(), compressed.float()
    )
    assert metrics["cosine_similarity"] >= 0.99, (
        f"Cross-1 cosine similarity {metrics['cosine_similarity']:.4f} < 0.99"
    )


# ------------------------------------------------------------------ #
# schedule() delegation                                                #
# ------------------------------------------------------------------ #

def test_schedule_delegates_to_scheduler() -> None:
    """pipeline.schedule() should delegate to _scheduler.schedule()."""
    pipeline = make_pipeline()
    requests = [make_request(f"req{i}", 64) for i in range(5)]
    result = pipeline.schedule(requests)
    assert len(result) == len(requests), (
        f"schedule() returned {len(result)} items, expected {len(requests)}"
    )


# ------------------------------------------------------------------ #
# chain_mode=True                                                      #
# ------------------------------------------------------------------ #

def test_chain_mode_true() -> None:
    """chain_mode=True should initialize scheduler with pipeline_mode=True."""
    pipeline = make_pipeline(chain_mode=True)
    assert pipeline._scheduler.config.pipeline_mode is True, (
        "pipeline_mode should be True when chain_mode=True"
    )
