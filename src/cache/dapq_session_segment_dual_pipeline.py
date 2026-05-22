"""DapQSessionSegmentDualReductionPipeline — Cross Activity B+C.

Combines SessionAwareTurnLevelSegmentCache (Activity B) with
DapQPositionAwareEvictionCodec (Activity C) for dual reduction using
a consistent position-aware principle for both segment selection and KV eviction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.cache.base import CacheStore
from src.cache.session_turn_level_segment_cache import (
    SessionAwareTurnLevelSegmentCache,
    SessionTurnLevelConfig,
    TurnSegmentEntry,
)
from src.cache.dapq_position_aware_eviction_codec import (
    DapQPositionAwareEvictionCodec,
    DapQEvictionConfig,
)


@dataclass
class DualReductionPipelineConfig:
    b_config: Optional[SessionTurnLevelConfig] = None
    c_config: Optional[DapQEvictionConfig] = None
    segment_keep_ratio: float = 0.50    # Step 2: keep top 50% segments
    kv_budget_ratio: float = 0.30       # Step 3: keep top 30% KV within selected segments
    decay_factor: float = 512.0         # position-aware score decay parameter
    seed: int = 42


class DapQSessionSegmentDualReductionPipeline(CacheStore):
    """DapQ position-aware eviction + session non-contiguous segment reuse dual reduction (B+C).

    Fully implements CacheStore interface.

    5-step integrated processing flow:
      Step 1 (Session segment retrieval, B):
        segment_cache.get_session_segments(session_id) → prior-turn segment list.

      Step 2 (Position-aware segment importance scoring, C principle):
        Compute position_reuse_score(segment, current_decode_pos) for each segment.
        Select top segment_keep_ratio segments.

      Step 3 (KV importance scoring within selected segments, C):
        Apply DapQ position-aware pseudo query importance to selected segment KVs.
        Keep top kv_budget_ratio KVs.

      Step 4 (Unselected segments + unselected KV handling):
        Unselected KVs: immediate eviction (overwrite with compressed values).

      Step 5 (Return final KV set):
        Return list of (TurnSegmentEntry, compressed_kv_tensor) for selected segments.

    Design consistency:
      Segment selection (B) and KV selection (C) both use the same position-aware
      principle → consistent importance criterion across granularity levels.

    Evaluation targets (evaluation_criteria.md §5):
      - Combined throughput: +5% vs single Activity (high priority)
      - Combined memory reduction: -10% vs single Activity (high priority)
      - Accuracy preservation (C included): cosine_sim >= 0.99 (MANDATORY)
    """

    def __init__(self, config: DualReductionPipelineConfig) -> None:
        if _TORCH_AVAILABLE:
            torch.manual_seed(config.seed)
        self.config = config
        b_cfg = config.b_config or SessionTurnLevelConfig(seed=config.seed)
        c_cfg = config.c_config or DapQEvictionConfig(
            budget_ratio=config.kv_budget_ratio,
            seed=config.seed,
        )
        self.segment_cache = SessionAwareTurnLevelSegmentCache(b_cfg)
        self.eviction_codec = DapQPositionAwareEvictionCodec(c_cfg)

    # ------------------------------------------------------------------ #
    # Dual reduction pipeline API                                          #
    # ------------------------------------------------------------------ #

    def process_session(
        self,
        session_id: str,
        current_decode_pos: float,
        pos_decode_int: Optional[int] = None,
    ) -> "List[Tuple[TurnSegmentEntry, torch.Tensor]]":
        """Execute B+C dual reduction pipeline.

        Args:
            session_id: target session identifier.
            current_decode_pos: current decode position (float for score computation).
            pos_decode_int: optional integer decode position for RoPE; defaults to
                            int(current_decode_pos).

        Returns:
            List of (TurnSegmentEntry, compressed_kv_tensor) for selected segments.
            Empty list on session cache miss.
        """
        # Step 1: retrieve session segments
        entries = self.segment_cache.get_session_segments(session_id)
        if not entries:
            return []

        # Step 2: position-aware segment selection
        scored = [
            (e, self.segment_cache.position_reuse_score(
                e, current_decode_pos, self.config.decay_factor))
            for e in entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        k_seg = max(1, int(len(scored) * self.config.segment_keep_ratio))
        selected_entries = [e for e, _ in scored[:k_seg]]

        # Step 3: apply DapQ eviction to KVs of selected segments
        pos = pos_decode_int if pos_decode_int is not None else int(current_decode_pos)
        result = []
        for entry in selected_entries:
            kv = self.segment_cache.get(entry.kv_pointer)
            if kv is None:
                continue
            compressed_kv = self.eviction_codec.compression_hook(entry.kv_pointer, kv)
            result.append((entry, compressed_kv))

        return result

    def dual_reduction_ratio(self) -> float:
        """Estimated B+C dual reduction ratio.

        = 1 - (segment_keep_ratio × kv_budget_ratio).
        Actual measurement uses memory_bytes() comparison.
        """
        return 1.0 - (self.config.segment_keep_ratio * self.config.kv_budget_ratio)

    def metrics_summary(self) -> Dict:
        """Return summary of key metrics from both sub-components."""
        return {
            "session_cache_hit_rate": self.segment_cache.hit_rate(),
            "session_noncontiguous_hit_rate": self.segment_cache.noncontiguous_hit_rate(),
            "eviction_memory_reduction_ratio": self.eviction_codec.memory_reduction_ratio(),
            "dual_reduction_estimate": self.dual_reduction_ratio(),
            "total_memory_bytes": self.memory_bytes(),
        }

    # ------------------------------------------------------------------ #
    # CacheStore interface (delegates to segment_cache)                   #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: "torch.Tensor") -> None:
        """Store KV tensor via segment_cache."""
        self.segment_cache.put(key, value)

    def get(self, key: str) -> "Optional[torch.Tensor]":
        """Retrieve KV tensor via segment_cache."""
        return self.segment_cache.get(key)

    def evict(self) -> int:
        """Evict via segment_cache session-aware LRU."""
        return self.segment_cache.evict()

    def hit_rate(self) -> float:
        """Cumulative cache hit rate from segment_cache."""
        return self.segment_cache.hit_rate()

    def memory_bytes(self) -> int:
        """Current memory footprint from segment_cache."""
        return self.segment_cache.memory_bytes()

    def compression_hook(self, key: str, value: "torch.Tensor") -> "torch.Tensor":
        """Apply DapQ position-aware eviction compression."""
        return self.eviction_codec.compression_hook(key, value)

    def get_importance_mask(self, key: str) -> "Optional[torch.Tensor]":
        """Return importance mask from eviction_codec."""
        return self.eviction_codec.get_importance_mask(key)

    def reset_stats(self) -> None:
        """Reset stats in both sub-components."""
        self.segment_cache.reset_stats()
        self.eviction_codec.reset_stats()
