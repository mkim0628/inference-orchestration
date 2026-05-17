"""Cross A+C: HMAChainedACPipeline — HMA multi-connector plugin chaining A+C integration pipeline.

Integrates A-1 HMAMultiConnectorCompressionPluginScheduler connector registry with
C-1 RLAdaptivePrecisionQuantizer and existing codecs for request-profile-based
dynamic connector dispatch. Supports chain_mode for sequential connector application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.cache.base import CacheStore
from src.cache.rl_adaptive_precision_quantizer import (
    RLAdaptivePrecisionConfig,
    RLAdaptivePrecisionQuantizer,
)
from src.cache.global_retention_gate_eviction import (
    GlobalRetentionGateConfig,
    GlobalRetentionGateEvictionCodec,
)
from src.engine.runner import InferenceRequest, InferenceRunner
from src.scheduler.hma_multi_connector_scheduler import (
    HMAConnectorAdapter,
    HMAConnectorInterface,
    HMAMultiConnectorCompressionPluginScheduler,
    HMAMultiConnectorConfig,
)


@dataclass
class HMAChainedACPipelineConfig:
    """Configuration for HMAChainedACPipeline. All parameters YAML-externalizable."""
    chain_mode: bool = False
    default_connector: str = "global_retention"
    long_ctx_threshold: int = 4096
    memory_pressure_threshold: float = 0.8
    rl_quantizer_config: Optional[RLAdaptivePrecisionConfig] = None
    global_retention_config: Optional[GlobalRetentionGateConfig] = None
    seed: int = 42


class HMAChainedACPipeline:
    """HMA multi-connector plugin chaining A+C integration pipeline.

    Cross Activity A+C:
      - A-1 HMAMultiConnectorCompressionPluginScheduler connector registry
      - C-1 RLAdaptivePrecisionQuantizer (RL workloads)
      - existing GlobalRetentionGateEvictionCodec (long context)
      - existing RateQuantReverseWaterfillingCodec (short high-throughput, optional)

    Request profile based connector selection rules (YAML connector_dispatch_policy):
      is_rl_mode=True or num_completions>1  → "rl_adaptive"
      context_length > long_ctx_threshold   → "global_retention"
      memory_pressure > threshold           → "ratequant" (optional)
      default                               → default_connector

    chain_mode=True:
      selected connector → global_retention sequential application

    InferenceRunner integration:
      runner = InferenceRunner(cache=pipeline.cache, scheduler=pipeline)
      runner.run_batch(requests) → pipeline.schedule(requests)
    """

    def __init__(
        self,
        config: HMAChainedACPipelineConfig,
        rl_quantizer: Optional[RLAdaptivePrecisionQuantizer] = None,
        global_retention_codec: Optional[GlobalRetentionGateEvictionCodec] = None,
        extra_connectors: Optional[Dict[str, HMAConnectorInterface]] = None,
    ) -> None:
        self.config = config

        # C-1: RLAdaptivePrecisionQuantizer
        rl_cfg = config.rl_quantizer_config or RLAdaptivePrecisionConfig(seed=config.seed)
        self._rl_quantizer: RLAdaptivePrecisionQuantizer = (
            rl_quantizer or RLAdaptivePrecisionQuantizer(rl_cfg)
        )

        # Existing GlobalRetentionGateEvictionCodec
        gr_cfg = config.global_retention_config or GlobalRetentionGateConfig(seed=config.seed)
        self._global_retention: GlobalRetentionGateEvictionCodec = (
            global_retention_codec or GlobalRetentionGateEvictionCodec(gr_cfg)
        )

        # A-1: HMAMultiConnectorScheduler initialization
        sched_cfg = HMAMultiConnectorConfig(
            long_ctx_threshold=config.long_ctx_threshold,
            memory_pressure_threshold=config.memory_pressure_threshold,
            default_connector=config.default_connector,
            pipeline_mode=config.chain_mode,
            seed=config.seed,
        )
        self._scheduler = HMAMultiConnectorCompressionPluginScheduler(sched_cfg)

        # Register connectors
        self._scheduler.register_connector(
            "rl_adaptive",
            HMAConnectorAdapter("rl_adaptive", self._rl_quantizer),
        )
        self._scheduler.register_connector(
            "global_retention",
            HMAConnectorAdapter("global_retention", self._global_retention),
        )

        # Optional extra connectors (e.g., ratequant, lookahead)
        for name, conn in (extra_connectors or {}).items():
            self._scheduler.register_connector(name, conn)

        # Default cache: global_retention codec as CacheStore
        self.cache: CacheStore = self._global_retention

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """Scheduling entry point called by InferenceRunner.run_batch()."""
        return self._scheduler.schedule(requests)

    def compress_kv(
        self,
        request: InferenceRequest,
        kv: torch.Tensor,
        request_meta: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Compress KV with the appropriate connector for the request profile."""
        compressed, _ = self._scheduler.apply_connector(request, kv, request_meta)
        return compressed

    def run_inference(
        self,
        requests: List[InferenceRequest],
        num_layers: int = 12,
        hidden_dim: int = 64,
        chunk_size: int = 128,
    ) -> Dict:
        """Run inference with throughput, memory, and accuracy measurements.

        Args:
            requests: list of InferenceRequest
            num_layers: model layer count (for InferenceRunner)
            hidden_dim: hidden dimension (for InferenceRunner)
            chunk_size: token chunk size

        Returns:
            dict with throughput_tokens_per_sec, memory_bytes, metrics_summary
        """
        runner = InferenceRunner(
            cache=self.cache,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            chunk_size=chunk_size,
            seed=self.config.seed,
            scheduler=self,
        )
        import time
        t0 = time.monotonic()
        results = runner.run_batch(requests)
        elapsed = time.monotonic() - t0

        total_tokens = sum(r.output_tokens for r in results)
        throughput = total_tokens / elapsed if elapsed > 0 else 0.0

        return {
            "throughput_tokens_per_sec": throughput,
            "memory_bytes": self.cache.memory_bytes(),
            "num_requests": len(results),
            "total_output_tokens": total_tokens,
            "elapsed_sec": elapsed,
            "runner_metrics": runner.metrics_summary(),
            "pipeline_metrics": self.metrics_summary(),
        }

    def metrics_summary(self) -> Dict:
        """Integrated metrics for throughput, memory, and accuracy measurement."""
        return {
            "connector_selection_stats": self._scheduler.connector_selection_stats(),
            "scheduling_overhead_ms_p50": self._scheduler.scheduling_overhead_ms_p50(),
            "rl_quantizer_memory_reduction": self._rl_quantizer.memory_reduction_ratio(),
            "rl_quantizer_precision_ratios": self._rl_quantizer.current_precision_ratios(),
            "global_retention_hit_rate": self._global_retention.hit_rate(),
            "global_retention_memory_bytes": self._global_retention.memory_bytes(),
        }
