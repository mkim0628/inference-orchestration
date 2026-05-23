"""CLC Positional Bias Gated Segment Cache (Activity B).

Implements the accuracy-limitation mechanism identified in 2603.20218:
position encoding mismatch on position-independent reuse causes accuracy degradation.
A lightweight gate measures ΔPos and applies a 3-stage selective re-encoding policy.
"""

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch

from src.cache.base import CacheStore


class ReencodingPolicy(Enum):
    DIRECT_REUSE = "direct_reuse"       # ΔPos ≤ bias_threshold -> no re-encoding
    PARTIAL_REENCODING = "partial"      # bias_threshold < ΔPos ≤ rope_threshold
    FULL_REENCODING = "full"            # ΔPos > rope_threshold -> full re-encoding


@dataclass
class CLCBiasGateConfig:
    max_context_length: int = 4096           # normalization reference context length
    bias_threshold: float = 0.15            # ΔPos ≤ this: direct reuse safe
    rope_distortion_threshold: float = 0.40  # ΔPos > this: full re-encoding required
    partial_reencoding_layer_ratio: float = 0.5  # partial: re-encode first 50% of layers
    max_entries: int = 1000
    seed: int = 42


@dataclass
class SegmentMeta:
    """Position metadata for a cached segment."""
    pos_orig_start: int    # original context start position
    pos_orig_end: int      # original context end position
    content_hash: str      # content hash (immutable)


class CLCPositionalBiasGatedSegmentCache(CacheStore):
    """CLC positional bias threshold gate for selective non-contiguous segment reuse (Activity B).

    Directly implements the CLC accuracy-limitation mechanism from 2603.20218:
      position-independent reuse causes position encoding mismatch -> accuracy degradation.
      -> measure ΔPos as a continuous value and decide 3-stage re-encoding policy.

    Positional bias measurement:
      ΔPos = |pos_target_start - pos_orig_start| / max_context_length  (0~1 normalized)

    3-stage selective re-encoding gate:
      ΔPos ≤ 0.15:  DIRECT_REUSE — skip AdapShot re-encoding (immediate reuse)
      0.15 < ΔPos ≤ 0.40: PARTIAL_REENCODING — re-encode first N//2 layers only
      ΔPos > 0.40: FULL_REENCODING — apply full AdapShot re-encoding

    Evaluation criteria (evaluation_criteria.md §3):
      - Overall Cache Hit Rate +5%p or more (high)
      - Non-contiguous segment hit rate ≥ 30% of total hits (high)
    """

    def __init__(self, config: CLCBiasGateConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._meta: Dict[str, SegmentMeta] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._direct_reuse_hits: int = 0
        self._partial_reencoding_hits: int = 0
        self._full_reencoding_hits: int = 0

    def compute_delta_pos(
        self,
        pos_orig_start: int,
        pos_target_start: int,
    ) -> float:
        """Normalized positional bias measurement.

        ΔPos = |pos_target_start - pos_orig_start| / max_context_length
        """
        return abs(pos_target_start - pos_orig_start) / max(1, self.config.max_context_length)

    def check_bias(
        self,
        segment_meta: SegmentMeta,
        pos_target_start: int,
    ) -> ReencodingPolicy:
        """Determine re-encoding policy from positional bias magnitude."""
        delta_pos = self.compute_delta_pos(segment_meta.pos_orig_start, pos_target_start)
        if delta_pos <= self.config.bias_threshold:
            return ReencodingPolicy.DIRECT_REUSE
        elif delta_pos <= self.config.rope_distortion_threshold:
            return ReencodingPolicy.PARTIAL_REENCODING
        else:
            return ReencodingPolicy.FULL_REENCODING

    def put_segment(
        self,
        key: str,
        value: torch.Tensor,
        pos_orig_start: int,
        pos_orig_end: int,
        content_hash: str,
    ) -> None:
        """Store segment KV together with position metadata."""
        meta = SegmentMeta(
            pos_orig_start=pos_orig_start,
            pos_orig_end=pos_orig_end,
            content_hash=content_hash,
        )
        self._meta[key] = meta
        self.put(key, value)

    def get_with_policy(
        self,
        key: str,
        pos_target_start: int,
    ) -> Tuple[Optional[torch.Tensor], ReencodingPolicy]:
        """Return cached segment with positional bias gate policy.

        Returns:
            (kv_tensor_or_None, reencoding_policy)
            kv_tensor is None on cache miss. Caller applies re-encoding policy.
        """
        kv = self.get(key)
        if kv is None:
            return None, ReencodingPolicy.FULL_REENCODING

        meta = self._meta.get(key)
        if meta is None:
            return kv, ReencodingPolicy.FULL_REENCODING

        policy = self.check_bias(meta, pos_target_start)

        if policy == ReencodingPolicy.DIRECT_REUSE:
            self._direct_reuse_hits += 1
        elif policy == ReencodingPolicy.PARTIAL_REENCODING:
            self._partial_reencoding_hits += 1
        else:
            self._full_reencoding_hits += 1

        return kv, policy

    def noncontiguous_direct_hit_rate(self) -> float:
        """Fraction of hits that were DIRECT_REUSE (no re-encoding needed)."""
        total_hits = (
            self._direct_reuse_hits
            + self._partial_reencoding_hits
            + self._full_reencoding_hits
        )
        return self._direct_reuse_hits / max(1, total_hits)

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        self._store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._store:
            return 0
        key, v = self._store.popitem(last=False)
        self._meta.pop(key, None)
        return v.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            "CLCPositionalBiasGatedSegmentCache does not support importance masking."
        )

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._direct_reuse_hits = 0
        self._partial_reencoding_hits = 0
        self._full_reencoding_hits = 0
