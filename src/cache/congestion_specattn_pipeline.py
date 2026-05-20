"""Cross A+C dual-reduction pipeline: CONCUR admission + SpecAttn sparsification.

Integrates CONCURCongestionBasedAgentAdmissionScheduler (Activity A) and
SpecAttnVerificationGuidedKVSparseCodec (Activity C) with a feedback loop:
KV pool congestion signal dynamically tightens the codec retention ratio.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.cache.base import CacheStore
from src.cache.specattn_sparse_codec import (
    SpecAttnCodecConfig,
    SpecAttnVerificationGuidedKVSparseCodec,
)
from src.scheduler.concur_congestion_admission_scheduler import (
    CONCURCongestionBasedAgentAdmissionScheduler,
    CongestionAdmissionConfig,
)


@dataclass
class DualReductionConfig:
    scheduler_config: Optional[CongestionAdmissionConfig] = None
    codec_config: Optional[SpecAttnCodecConfig] = None
    codec_adapt_on_congestion: bool = True
    retention_reduction_on_congestion: float = 0.10
    seed: int = 42


class CongestionAdmissionSpecAttnDualReductionPipeline(CacheStore):
    """CONCUR congestion admission × SpecAttn KV sparsification pipeline.

    Cross Activity A+C:
      Layer 1 (A): CONCURCongestionBasedAgentAdmissionScheduler — admission gate.
      Layer 2 (C): SpecAttnVerificationGuidedKVSparseCodec — KV sparsification.

    Feedback loop:
      KV pool occupancy > alpha_high → C retention_ratio -= retention_reduction_on_congestion
        (more aggressive sparsification relieves pool pressure).
      KV pool occupancy < alpha_low → restore retention_ratio to baseline values.

    CacheStore interface fully implemented; all put() calls go through the codec.
    """

    def __init__(self, config: DualReductionConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config

        sched_cfg = config.scheduler_config or CongestionAdmissionConfig(seed=config.seed)
        codec_cfg = config.codec_config or SpecAttnCodecConfig(seed=config.seed)

        self.scheduler = CONCURCongestionBasedAgentAdmissionScheduler(sched_cfg)
        self.codec = SpecAttnVerificationGuidedKVSparseCodec(codec_cfg)
        # Snapshot of original ratios used for FREE-state restoration
        self._base_retention_ratios: List[float] = list(codec_cfg.retention_ratio_by_layer)
        self._agent_kv_usage: Dict[str, int] = {}

    def set_verification_logits(
        self,
        attn_logits: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Forward verification-phase logits to the codec."""
        self.codec.set_verification_logits(attn_logits, layer_idx)

    def update_kv_pool(self, used_bytes: int) -> None:
        """Update KV pool state and trigger the A→C feedback loop.

        When congested, tighten retention ratio; when free, restore it.
        """
        self.scheduler.update_kv_pool(used_bytes)
        if self.config.codec_adapt_on_congestion:
            level = self.scheduler.monitor.congestion_level()
            if level == "CONGESTED":
                delta = self.config.retention_reduction_on_congestion
                for i in range(len(self.codec.config.retention_ratio_by_layer)):
                    self.codec.config.retention_ratio_by_layer[i] = max(
                        0.50,
                        self._base_retention_ratios[i] - delta,
                    )
            elif level == "FREE":
                self.codec.config.retention_ratio_by_layer = list(self._base_retention_ratios)

    def schedule(self, requests: List) -> List:
        """Delegate batch scheduling to the CONCUR scheduler."""
        return self.scheduler.schedule(requests)

    # ------------------------------------------------------------------ #
    # CacheStore abstract methods                                          #
    # ------------------------------------------------------------------ #

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return self.codec.compression_hook(key, value)

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        return self.codec.get_importance_mask(key)

    def put(self, key: str, value: torch.Tensor) -> None:
        self.codec.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.codec.get(key)

    def evict(self) -> int:
        return self.codec.evict()

    def hit_rate(self) -> float:
        return self.codec.hit_rate()

    def memory_bytes(self) -> int:
        return self.codec.memory_bytes()

    def reset_stats(self) -> None:
        self.codec.reset_stats()
        self.scheduler.reset_stats()
        self._agent_kv_usage.clear()

    def metrics_summary(self) -> Dict:
        """Unified metrics for cross A+C evaluation."""
        return {
            "scheduler_overhead_ms_p50": self.scheduler.scheduling_overhead_ms_p50(),
            "kv_pool_occupancy": self.scheduler.monitor.get_occupancy(),
            "congestion_level": self.scheduler.monitor.congestion_level(),
            "codec_hit_rate": self.codec.hit_rate(),
            "codec_memory_reduction_ratio": self.codec.memory_reduction_ratio(),
            "current_retention_ratios": list(self.codec.config.retention_ratio_by_layer),
            "total_memory_bytes": self.memory_bytes(),
        }
