"""Cross Activity A+B+C: AMPDPrefillShareNonContiguousStack.

5-stage processing flow integrating:
  A-1 AMPDLazySegmentFetchScheduler — lazy fetch scheduler
  B-1 AMPDAdapShotLazyLoadPipeline  — lazy load + AdapShot RoPE reencoding pipeline
  C-1 DPAttentionAwareCompressionSelector — environment-aware compression selector

Evaluation criteria (evaluation_criteria.md §5):
  - Cross throughput vs solo: +5% or more
  - Cross memory vs solo: −10% or more
  - Accuracy preservation (C included): cosine >= 0.99 (MANDATORY)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.ampd_adapshot_lazy_pipeline import AMPDAdapShotLazyLoadPipeline, LazyPipelineConfig
from src.cache.dp_attention_aware_compression import (
    DPAttentionAwareCompressionSelector,
    DPAttentionCompressionConfig,
)
from src.engine.runner import InferenceRequest, InferenceRunner
from src.scheduler.ampd_lazy_segment_fetch import (
    AMPDLazySchedulerConfig,
    AMPDLazySegmentFetchScheduler,
    SegmentMeta,
    SegmentMetadataRegistry,
)


@dataclass
class AMPDStackConfig:
    scheduler_config: Optional[AMPDLazySchedulerConfig] = None
    pipeline_config: Optional[LazyPipelineConfig] = None
    compression_config: Optional[DPAttentionCompressionConfig] = None
    seed: int = 42


class AMPDPrefillShareNonContiguousStack:
    """AMPD + AdapShot + DP Attention-aware compression A+B+C unified stack.

    Cross Activity A+B+C:
      - A-1 AMPDLazySegmentFetchScheduler: lazy fetch scheduler
      - B-1 AMPDAdapShotLazyLoadPipeline: lazy load + reencoding pipeline
      - C-1 DPAttentionAwareCompressionSelector: environment-aware compression

    5-stage processing flow:
      Step 1: Segment metadata forwarded first (no KV transfer).
      Step 2: Fan-out distribution simulation (local put on single node).
      Step 3: Confirmed segments lazy-loaded + AdapShot reencoded (async overlap).
      Step 4: DP Attention-aware compression applied.
      Step 5: Non-contiguous attention computation fed (InferenceRunner-compatible).

    InferenceRunner integration:
      runner = InferenceRunner(cache=stack.cache, scheduler=stack)
      runner.run_batch(requests) → stack.schedule(requests) called internally.
    """

    def __init__(
        self,
        config: AMPDStackConfig,
        extra_codecs: Optional[Dict[str, CacheStore]] = None,
    ) -> None:
        self.config = config

        sched_cfg = config.scheduler_config or AMPDLazySchedulerConfig(seed=config.seed)
        pipeline_cfg = config.pipeline_config or LazyPipelineConfig(seed=config.seed)
        comp_cfg = config.compression_config or DPAttentionCompressionConfig(seed=config.seed)

        self.registry = SegmentMetadataRegistry()
        self.scheduler = AMPDLazySegmentFetchScheduler(sched_cfg, self.registry)
        self.pipeline = AMPDAdapShotLazyLoadPipeline(pipeline_cfg)
        self.compressor = DPAttentionAwareCompressionSelector(comp_cfg)

        for name, codec in (extra_codecs or {}).items():
            self.compressor.register_codec(name, codec)

        # InferenceRunner expects a CacheStore via cache attribute
        self.cache: CacheStore = self.pipeline

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """InferenceRunner.run_batch()-compatible schedule() entry point."""
        return self.scheduler.schedule(requests)

    def process_request_step1_metadata(
        self,
        request: InferenceRequest,
        candidate_segment_ids: List[str],
    ) -> None:
        """Step 1: forward segment metadata without KV transfer."""
        metas = [
            SegmentMeta(
                segment_id=seg_id,
                source_node_id="local",
                tier="HBM",
                approx_size_bytes=128 * 64 * 2,
                position_range=(0, 128),
            )
            for seg_id in candidate_segment_ids
        ]
        self.scheduler.pre_resolve_segments(request, candidate_segment_ids, metas)

    def process_step2_fanout(
        self,
        key: str,
        kv: torch.Tensor,
    ) -> None:
        """Step 2: fan-out distribution (single-node: local put to pipeline store)."""
        self.pipeline.put(key, kv)

    def process_step3_lazy_load(
        self,
        confirmed_ids: List[str],
        source_positions: Optional[List[int]] = None,
        target_positions: Optional[List[int]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """Step 3: lazy load + AdapShot reencoding (sync wrapper over async pipeline).

        Wraps async load_and_reencode_segment in a new event loop for
        compatibility with synchronous InferenceRunner callers.
        """
        src_pos = source_positions or [0] * len(confirmed_ids)
        tgt_pos = target_positions or [0] * len(confirmed_ids)

        async def _run() -> List[Optional[torch.Tensor]]:
            results = []
            for seg_id, sp, tp in zip(confirmed_ids, src_pos, tgt_pos):
                kv = await self.pipeline.load_and_reencode_segment(seg_id, sp, tp)
                results.append(kv)
            return results

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    def process_step4_compression(
        self,
        key: str,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """Step 4: DP Attention-aware compression."""
        return self.compressor.compression_hook(key, kv)

    def process_step5_put_compressed(
        self,
        key: str,
        kv: torch.Tensor,
    ) -> None:
        """Step 5: store compression output back into the pipeline cache."""
        compressed = self.compressor.compression_hook(key, kv)
        self.pipeline.put(key, compressed)

    def metrics_summary(self) -> Dict:
        """Unified metrics for cross-activity evaluation."""
        return {
            "scheduler_overhead_ms_p50": self.scheduler.scheduling_overhead_ms_p50(),
            "unnecessary_transfer_ratio": self.scheduler.unnecessary_transfer_ratio(),
            "pipeline_hit_rate": self.pipeline.hit_rate(),
            "pipeline_noncontiguous_hit_rate": self.pipeline.noncontiguous_hit_rate(),
            "pipeline_memory_bytes": self.pipeline.memory_bytes(),
            "compressor_hit_rate": self.compressor.hit_rate(),
            "compressor_memory_reduction": self.compressor.memory_reduction_ratio(),
            "compressor_effective_replicas": self.compressor.effective_kv_replicas(),
        }
