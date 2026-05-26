"""Activity C-1 (optional): MLA c_KV two-axis compression codec.

Compresses MLA c_KV along two orthogonal axes:
  - Position axis: pointer-based deduplication for identical segment IDs
  - Depth axis: cosine-similarity-based layer sharing when cos_sim >= threshold

Accuracy-preserving design:
  - Position axis: c_KV is position-free (mathematical guarantee: zero loss)
  - Depth axis: only shares layers with cos_sim >= depth_sharing_threshold (>= 0.90)
  - Auto-fallback: if accuracy_delta > 1%, threshold rises to fallback_threshold
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.cache.base import CacheStore
from src.cache.irminsul_mla_segment_cache import IrminsulMLASegmentCache


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #


@dataclass
class MLATwoAxisConfig:
    """Configuration for MLA two-axis compression codec."""
    depth_sharing_threshold: float = 0.90   # layers with cos_sim >= threshold share KV
    depth_sharing_k: int = 2                # k=2 → 50% layer sharing target
    position_dedup_enabled: bool = True     # enable position-axis deduplication
    fallback_threshold: float = 0.95        # threshold raised here when accuracy > 1%


# ------------------------------------------------------------------ #
# MLATwoAxisCompressionCodec                                          #
# ------------------------------------------------------------------ #


class MLATwoAxisCompressionCodec(CacheStore):
    """MLA c_KV position-axis × depth-axis two-axis compression codec (Activity C-1).

    Position axis compression:
      - Agentic sessions: identical segment_id c_KV → pointer only (deduplication)
      - c_KV position-free property guarantees zero accuracy loss

    Depth axis compression:
      - cos_sim(c_kv[l], c_kv[l-1]) >= depth_sharing_threshold → layer l shares l-1
      - Below threshold: independent storage

    Combined reduction:
      total_reduction = 1 - (1 - pos_reduction) × (1 - depth_reduction)

    Accuracy-preserving fallback:
      auto_adjust_threshold(accuracy_delta): if |delta| > 1%, raise threshold
    """

    def __init__(
        self,
        base_cache: IrminsulMLASegmentCache,
        config: MLATwoAxisConfig,
    ) -> None:
        self.base_cache = base_cache
        self.config = config

        # Position-axis dedup: segment_id → c_KV reference tensor
        # Maps segment_id to first-stored c_KV; subsequent stores just increment ref count
        self._position_dedup: Dict[str, torch.Tensor] = {}
        self._position_ref_counts: Dict[str, int] = {}

        # Depth-axis sharing: (segment_id, layer) → shared_layer_idx or None
        # None means stored independently; int means it points to another layer
        self._depth_sharing_map: Dict[Tuple[str, int], int] = {}

        # Generic tensor store for base CacheStore put/get API
        self._generic_store: Dict[str, torch.Tensor] = {}

        # Stats
        self._hits = 0
        self._misses = 0
        self._position_deduplicated = 0   # entries saved by position dedup
        self._depth_shared = 0            # entries saved by depth sharing

    # ---------------------------------------------------------------- #
    # CacheStore abstract methods                                       #
    # ---------------------------------------------------------------- #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store tensor; applies position-axis dedup when enabled."""
        if self.config.position_dedup_enabled:
            if key in self._position_dedup:
                # Already stored — just increment ref count (pointer reuse)
                self._position_ref_counts[key] = self._position_ref_counts.get(key, 1) + 1
                self._position_deduplicated += 1
                return
            self._position_dedup[key] = value.detach().clone()
            self._position_ref_counts[key] = 1
        else:
            self._generic_store[key] = value.detach().clone()

        # Also delegate to base cache for full MLA segment operations
        self.base_cache.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve tensor by key."""
        # Try position dedup store first
        if key in self._position_dedup:
            self._hits += 1
            return self._position_dedup[key]
        # Try generic store
        if key in self._generic_store:
            self._hits += 1
            return self._generic_store[key]
        # Delegate to base cache
        result = self.base_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        """Evict one entry; delegates to base cache."""
        return self.base_cache.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Compressed memory footprint (deduplication reduces this)."""
        pos_bytes = sum(v.nbytes for v in self._position_dedup.values())
        generic_bytes = sum(v.nbytes for v in self._generic_store.values())
        return pos_bytes + generic_bytes

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._position_deduplicated = 0
        self._depth_shared = 0
        self.base_cache.reset_stats()

    # ---------------------------------------------------------------- #
    # Position-axis API                                                 #
    # ---------------------------------------------------------------- #

    def position_dedup_reduction_rate(self) -> float:
        """Fraction of entries saved by position-axis deduplication."""
        total_refs = sum(self._position_ref_counts.values())
        unique_entries = len(self._position_dedup)
        if total_refs == 0:
            return 0.0
        return 1.0 - unique_entries / total_refs

    # ---------------------------------------------------------------- #
    # Depth-axis API                                                    #
    # ---------------------------------------------------------------- #

    def compress_layer_kv(
        self,
        c_kv_by_layer: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], float]:
        """Compress layer-indexed c_KV dict using depth-axis sharing.

        Consecutive layers with cos_sim >= threshold share the same tensor.

        Returns:
            compressed: {layer_idx: c_KV tensor} — deduplicated via pointer sharing
            depth_reduction: fraction of layers that share with a previous layer
        """
        if not c_kv_by_layer:
            return {}, 0.0

        sorted_layers = sorted(c_kv_by_layer.keys())
        compressed: Dict[int, torch.Tensor] = {}
        shared_count = 0

        for i, layer_idx in enumerate(sorted_layers):
            c_kv = c_kv_by_layer[layer_idx]

            if i == 0:
                compressed[layer_idx] = c_kv
                continue

            prev_layer = sorted_layers[i - 1]
            prev_c_kv = compressed[prev_layer]

            # Compute cosine similarity between consecutive layers
            cos_sim = self._layer_cosine_sim(c_kv, prev_c_kv)

            if cos_sim >= self.config.depth_sharing_threshold:
                # Share: point to previous layer tensor (zero extra memory)
                compressed[layer_idx] = compressed[prev_layer]
                shared_count += 1
                self._depth_shared += 1
            else:
                compressed[layer_idx] = c_kv

        depth_reduction = shared_count / len(sorted_layers) if sorted_layers else 0.0
        return compressed, depth_reduction

    def depth_sharing_reduction_rate(
        self,
        c_kv_by_layer: Dict[int, torch.Tensor],
    ) -> float:
        """Compute depth-axis reduction rate for the given layer set."""
        if len(c_kv_by_layer) <= 1:
            return 0.0

        sorted_layers = sorted(c_kv_by_layer.keys())
        shared = 0

        for i in range(1, len(sorted_layers)):
            prev = sorted_layers[i - 1]
            curr = sorted_layers[i]
            cos_sim = self._layer_cosine_sim(
                c_kv_by_layer[curr], c_kv_by_layer[prev]
            )
            if cos_sim >= self.config.depth_sharing_threshold:
                shared += 1

        return shared / len(sorted_layers)

    def combined_reduction_rate(
        self,
        c_kv_by_layer: Dict[int, torch.Tensor],
        pos_reduction: Optional[float] = None,
    ) -> float:
        """Compute combined two-axis reduction rate.

        total_reduction = 1 - (1 - pos_reduction) × (1 - depth_reduction)
        """
        depth_reduction = self.depth_sharing_reduction_rate(c_kv_by_layer)
        if pos_reduction is None:
            pos_reduction = self.position_dedup_reduction_rate()

        return 1.0 - (1.0 - pos_reduction) * (1.0 - depth_reduction)

    # ---------------------------------------------------------------- #
    # Accuracy-preserving fallback                                      #
    # ---------------------------------------------------------------- #

    def auto_adjust_threshold(
        self,
        accuracy_delta: float,
        max_allowed_delta: float = 0.01,
    ) -> bool:
        """Raise depth_sharing_threshold to fallback_threshold if accuracy_delta > 1%.

        Returns True if threshold was adjusted.
        """
        if abs(accuracy_delta) > max_allowed_delta:
            self.config.depth_sharing_threshold = self.config.fallback_threshold
            return True
        return False

    # ---------------------------------------------------------------- #
    # Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _layer_cosine_sim(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between two layer c_KV tensors.

        Flattens each tensor to a vector before computing similarity.
        """
        a_flat = a.float().flatten()
        b_flat = b.float().flatten()

        # Pad or truncate to same length for comparison
        min_len = min(len(a_flat), len(b_flat))
        if min_len == 0:
            return 0.0

        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]

        dot = (a_flat * b_flat).sum()
        norm_a = a_flat.norm()
        norm_b = b_flat.norm()

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(dot / (norm_a * norm_b))
