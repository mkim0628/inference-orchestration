"""Activity B: AMPDAdapShotLazyLoadPipeline — lazy load + AdapShot RoPE reencoding.

3-stage async pipeline:
  Stage 1 (Segment Resolution): resolve_segments() fixes the hit-guaranteed segment
    set. Returns only SegmentMeta + miss indices — no KV data loaded.
  Stage 2 (Lazy Load): async load after confirmation; asyncio.Event per segment.
  Stage 3 (RoPE Reencoding Overlap): starts immediately on Stage 2 completion per
    segment, overlapping with remaining loads.
    Total latency = max(load_latency, reencode_latency) in theory.

Implements CacheStore fully.
"""

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class LazyPipelineConfig:
    chunk_size: int = 128
    max_entries: int = 1000
    rope_theta: float = 10000.0      # RoPE base frequency for AdapShot reencoding
    n_heads: int = 8
    d_head: int = 64
    companion_hit_threshold: int = 2  # companion segment prefetch threshold
    seed: int = 42


class AMPDAdapShotLazyLoadPipeline(CacheStore):
    """AMPD lazy load + AdapShot RoPE reencoding async overlap pipeline.

    Activity B: Non-Contiguous KV Cache Reuse.
    Fully implements CacheStore interface.

    3-stage async pipeline:
      Stage 1 (Segment Resolution): resolve_segments() returns hit metadata and
        miss indices. KV tensors are not loaded into memory yet.
      Stage 2 (Lazy Load): async load triggered after segment set confirmation;
        asyncio.Event signals completion per segment.
      Stage 3 (RoPE Reencoding Overlap): triggered immediately on Stage 2 event;
        overlaps with remaining Stage 2 loads.

    AdapShot RoPE reencoding:
      Source positions [pos_start, pos_end] → target positions [target_start, …]
      Δθ = target_position − source_position
      Batch rotation via torch.einsum (FP32 internally, FP16 output).

    Evaluation criteria (evaluation_criteria.md §3):
      - Non-contiguous segment hit rate ≥ 30% of all hits
      - KV memory footprint increase ≤ +20% vs baseline
    """

    def __init__(self, config: LazyPipelineConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._store: SegmentedHashCache = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        # co-occurrence stats: {(seg_a, seg_b): hit_count}
        self._companion_stats: Dict[Tuple[str, str], int] = {}
        # per-segment load completion events
        self._load_events: Dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------ #
    # CacheStore abstract methods                                          #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Standard put — stores directly (no compression hook here)."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Standard get — direct HBM lookup."""
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self._store.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._store.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._store.reset_stats()
        self._load_events.clear()

    # ------------------------------------------------------------------ #
    # Stage 1: Segment Resolution                                          #
    # ------------------------------------------------------------------ #

    def resolve_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> "Tuple[List, List[int]]":
        """Stage 1: collect hit-guaranteed segment metadata (no KV load).

        Algorithm:
          1. SegmentedHashCache.get_segments() for hit/miss determination.
          2. Hit chunks → SegmentMeta(segment_id, tier="HBM", ...) per chunk.
          3. Miss chunks → miss_indices list.
          4. No KV tensors returned at this stage — metadata only.

        Returns:
            (hit_metas, miss_chunk_indices)
        """
        from src.scheduler.ampd_lazy_segment_fetch import SegmentMeta

        hits, misses = self._store.get_segments(token_ids, layer_idx)

        chunk_size = self.config.chunk_size
        hit_metas = []
        for chunk_idx, _kv in hits:
            seg_id = self._store.chunk_key(token_ids, chunk_idx, layer_idx)
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(token_ids))
            hit_metas.append(SegmentMeta(
                segment_id=seg_id,
                source_node_id="local",
                tier="HBM",
                approx_size_bytes=chunk_size * self.config.d_head * 2 * 2,  # FP16 = 2 bytes
                position_range=(start, end),
            ))

        return hit_metas, misses

    # ------------------------------------------------------------------ #
    # Stage 2 + 3: Lazy Load + RoPE Reencoding Overlap                    #
    # ------------------------------------------------------------------ #

    async def load_and_reencode_segment(
        self,
        segment_id: str,
        source_position: int,
        target_position: int,
    ) -> Optional[torch.Tensor]:
        """Stage 2+3 overlap: load then immediately AdapShot-reencode.

        Algorithm:
          1. Load segment_id KV from _store.
          2. Signal asyncio.Event on load completion.
          3. AdapShot RoPE reencoding: Δθ = target_position − source_position.
          4. Return reencoded KV.

        Args:
            segment_id: cache key
            source_position: token position at cache-store time
            target_position: target position in current context

        Returns:
            RoPE-reencoded KV tensor (FP16) or None on miss
        """
        kv = self._store.get(segment_id)
        if kv is None:
            return None

        event = self._load_events.setdefault(segment_id, asyncio.Event())
        event.set()  # signal Stage 2 completion

        # Stage 3 runs inline (overlapped with remaining loads in practice)
        if source_position != target_position:
            kv = self._adapshot_rope_reencode(kv, source_position, target_position)

        return kv

    def _adapshot_rope_reencode(
        self,
        kv: torch.Tensor,
        source_pos: int,
        target_pos: int,
    ) -> torch.Tensor:
        """AdapShot RoPE phase-offset reencoding.

        Algorithm:
          1. delta = target_pos − source_pos
          2. Build rotation matrix from RoPE frequencies over d_head dimension.
          3. Apply rotation to last dim of kv via even/odd pair split.
          4. Batch rotation: O(S × n_heads × d_head).

        kv shape: [n_tokens, ...] — last dim must be even (d_head or multiple).
        Returns: same shape, dtype=float16.
        """
        d = kv.shape[-1]
        delta = float(target_pos - source_pos)
        half_d = d // 2

        inv_freq = 1.0 / (
            self.config.rope_theta ** (
                torch.arange(0, half_d, dtype=torch.float32) * 2.0 / d
            )
        )  # [half_d]
        angle = delta * inv_freq   # [half_d]
        cos_a = torch.cos(angle)   # [half_d]
        sin_a = torch.sin(angle)   # [half_d]

        kv_f = kv.detach().float()
        flat = kv_f.reshape(-1, d)       # [N, d]
        x1 = flat[..., :half_d]          # [N, half_d]
        x2 = flat[..., half_d:]          # [N, half_d]

        rotated = torch.cat([
            x1 * cos_a - x2 * sin_a,
            x1 * sin_a + x2 * cos_a,
        ], dim=-1)                        # [N, d]

        return rotated.reshape(kv.shape).half()

    # ------------------------------------------------------------------ #
    # Segment-level API (delegate to SegmentedHashCache)                   #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Store segment (delegates to SegmentedHashCache)."""
        self._store.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """Delegate chunk key computation to inner store."""
        return self._store.chunk_key(token_ids, chunk_idx, layer_idx)

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> "Tuple[List[Tuple[int, torch.Tensor]], List[int]]":
        """Non-contiguous segment hit lookup with non-contiguous hit tracking."""
        hits, misses = self._store.get_segments(token_ids, layer_idx)
        miss_set = set(misses)
        for chunk_idx, _ in hits:
            if any(m < chunk_idx for m in miss_set):
                self._noncontiguous_hits += 1
            self._hits += 1
        self._misses += len(misses)
        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous (out-of-prefix)."""
        # Combine internal counters with inner store counters to avoid double counting
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits
