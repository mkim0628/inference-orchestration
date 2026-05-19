from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.cache.base import CacheStore
from src.cache.thunder_agent_static_reservation_cache import (
    LLMProgramDAG,
    ProgramStep,
    ThunderAgentStaticSegmentReservationCache,
)
from src.cache.kvdrive_tier_compression_codec import (
    KVDriveTierDifferentiatedCompressionCodec,
    Tier,
    TierCompressionConfig,
)
from src.scheduler.kvdrive_attention_pipeline_scheduler import (
    KVDriveAttentionAwarePipelineScheduler,
    KVDriveSchedulerConfig,
)


@dataclass
class IntegratedStackConfig:
    scheduler_config: Optional[KVDriveSchedulerConfig] = None
    codec_config: Optional[TierCompressionConfig] = None
    max_entries: int = 512
    pin_threshold: float = 0.5
    seed: int = 42


class KVDriveThunderAgentIntegratedStack(CacheStore):
    """Unified A+B+C stack: 3-tier scheduling + static reservation + tier compression.

    Flow:
      Step 1 (parse_program): LLMProgramDAG parsing → reservation map.
      Step 2 (reserve_for_step): Pin high-reuse segments.
      Step 3 (put): Look up tier via KVTierRegistry → compress with TierCodec → store.
      Step 4 (get): Retrieve + decompress from segment_cache.

    CacheStore interface fully implemented.
    """

    def __init__(self, config: IntegratedStackConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config

        sched_cfg = config.scheduler_config or KVDriveSchedulerConfig(seed=config.seed)
        codec_cfg = config.codec_config or TierCompressionConfig(seed=config.seed)

        self.scheduler = KVDriveAttentionAwarePipelineScheduler(sched_cfg)
        self.codec = KVDriveTierDifferentiatedCompressionCodec(
            codec_cfg, default_tier="HBM"
        )
        self.segment_cache = ThunderAgentStaticSegmentReservationCache(
            max_entries=config.max_entries,
            pin_threshold=config.pin_threshold,
            seed=config.seed,
        )

    # ------------------------------------------------------------------ #
    # Program lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def parse_program(self, steps: List[ProgramStep]) -> None:
        """Parse agentic workflow into DAG and build reservation map."""
        self.segment_cache.parse_program(steps)

    def reserve_for_step(self, step_id: str) -> List[str]:
        """Pin high-reuse segments for the given step."""
        return self.segment_cache.reserve_segments(step_id)

    def release_step(self, step_id: str) -> None:
        """Unpin segments after step completes."""
        self.segment_cache.release_reservations(step_id)

    def schedule(self, requests: List[dict]) -> List[dict]:
        """Delegate to KVDriveAttentionAwarePipelineScheduler."""
        return self.scheduler.schedule(requests)

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Determine tier from KVTierRegistry, apply tier-specific compression."""
        tier: Tier = "HBM"
        try:
            token_id = int(key.split(":")[0], 16) % (2**31)
            info = self.scheduler.registry.get_tier(token_id)
            if info is not None:
                tier = info.tier  # type: ignore[assignment]
        except (ValueError, IndexError):
            pass
        return self.codec.compress_for_tier(value, tier)

    def put(self, key: str, value: torch.Tensor) -> None:
        """Compress via compression_hook, then store in segment_cache."""
        compressed = self.compression_hook(key, value)
        # Store directly into the segment_cache's internal store to honour its LRU/pinning.
        if key in self.segment_cache._store:
            self.segment_cache._store.move_to_end(key)
        else:
            if len(self.segment_cache._store) >= self.config.max_entries:
                self.segment_cache.evict()
        self.segment_cache._store[key] = compressed.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve from segment_cache (with non-contiguous hit tracking)."""
        return self.segment_cache.get(key)

    def evict(self) -> int:
        """Delegate LRU eviction to segment_cache (respects pinning)."""
        return self.segment_cache.evict()

    def hit_rate(self) -> float:
        return self.segment_cache.hit_rate()

    def memory_bytes(self) -> int:
        return self.segment_cache.memory_bytes()

    def reset_stats(self) -> None:
        self.segment_cache.reset_stats()
        self.scheduler.reset_stats()
        self.codec.reset_stats()

    # ------------------------------------------------------------------ #
    # Extended metrics                                                     #
    # ------------------------------------------------------------------ #

    def noncontiguous_hit_rate(self) -> float:
        return self.segment_cache.noncontiguous_hit_rate()

    def metrics_summary(self) -> Dict:
        """Return combined metrics across all three activities."""
        return {
            "scheduler_overhead_ms_p50": self.scheduler.scheduling_overhead_ms_p50(),
            "unnecessary_eviction_rate": self.scheduler.unnecessary_eviction_rate(),
            "segment_cache_hit_rate": self.segment_cache.hit_rate(),
            "noncontiguous_hit_rate": self.segment_cache.noncontiguous_hit_rate(),
            "reservation_hit_rate": self.segment_cache.reservation_hit_rate(),
            "codec_memory_reduction_ratio": self.codec.memory_reduction_ratio(),
            "total_memory_bytes": self.memory_bytes(),
        }
