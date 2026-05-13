"""KVFoldAccumulativeRadixCache — foldl accumulator-based non-contiguous KV reuse (Activity B).

KV-Fold (2605.12471) foldl protocol integrated into the src/cache/ layer without training.
Uses SegmentedHashCache as the segment backend and manages foldl accumulation state
in a separate OrderedDict. Drift plateau detection allows early termination of the
foldl chain once the accumulated KV stabilises.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class KVFoldConfig:
    chunk_size: int = 128
    max_entries: int = 2000
    drift_threshold: float = 1e-3
    max_fold_depth: int = 511
    enable_streaming_fallback: bool = True
    window_size: int = 32
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 12
    seed: int = 42
    compressor: Optional[object] = None  # Optional[SRFTFusedINT4KVKernel]


@dataclass
class _FoldState:
    """Accumulated KV state — accumulator of a foldl chain."""
    accumulated_kv: torch.Tensor   # [n_accumulated_tokens, 2, n_heads, d_head]
    chunk_ids: List[int]
    fold_depth: int
    last_drift: float
    plateau_reached: bool


class KVFoldAccumulativeRadixCache(CacheStore):
    """foldl accumulator-based non-contiguous KV reuse cache (Activity B).

    KV-Fold (2605.12471) foldl protocol implemented training-free in src/cache/.
    Uses SegmentedHashCache as the segment backend; foldl accumulation states
    are kept in a separate OrderedDict.

    Usage:
      1. fold_chunk(chunk_tokens, layer_idx) — integrate chunk into foldl accumulator
      2. get_folded_prefix(key) — return accumulated KV as prefix
      3. put/get — standard CacheStore interface (RadixAttention prefix-match path)
    """

    def __init__(self, config: KVFoldConfig) -> None:
        self.config = config
        self._store = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._fold_states: OrderedDict[str, _FoldState] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._fold_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Standard prefix KV store (RadixAttention-compatible path)."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Standard prefix KV lookup. Returns None on miss."""
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        """LRU eviction covering both segment store and fold states."""
        freed = self._store.evict()
        if self._fold_states:
            oldest_key = next(iter(self._fold_states))
            state = self._fold_states.pop(oldest_key)
            freed += state.accumulated_kv.nbytes
        return freed

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        fold_bytes = sum(
            s.accumulated_kv.nbytes for s in self._fold_states.values()
        )
        return self._store.memory_bytes() + fold_bytes

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._fold_hits = 0
        self._store.reset_stats()

    # ------------------------------------------------------------------ #
    # foldl accumulation API                                               #
    # ------------------------------------------------------------------ #

    def fold_chunk(
        self,
        chunk_tokens: List[int],
        layer_idx: int = 0,
        existing_fold_key: Optional[str] = None,
    ) -> Tuple[str, torch.Tensor]:
        """Integrate a chunk into the foldl accumulation state.

        Returns:
          fold_key: str — key of the updated accumulation state
          accumulated_kv: Tensor [n_acc_tokens, 2, n_heads, d_head]
        """
        n_heads = self.config.n_heads
        d_head = self.config.d_head

        # 1. Load existing accumulation state or initialise
        if existing_fold_key and existing_fold_key in self._fold_states:
            state = self._fold_states[existing_fold_key]
        else:
            state = _FoldState(
                accumulated_kv=torch.empty(0, 2, n_heads, d_head),
                chunk_ids=[],
                fold_depth=0,
                last_drift=float("inf"),
                plateau_reached=False,
            )

        # 2. Early exit when plateau is reached — no further computation needed
        if state.plateau_reached and existing_fold_key is not None:
            return existing_fold_key, state.accumulated_kv

        # 3. Get/compute chunk KV
        chunk_key = self._store.chunk_key(chunk_tokens, 0, layer_idx)
        cached_kv = self._store.get(chunk_key)
        if cached_kv is None:
            new_chunk_kv = self._compute_chunk_kv(chunk_tokens, layer_idx, state.accumulated_kv)
            self._store.put(chunk_key, new_chunk_kv)
        else:
            new_chunk_kv = cached_kv

        # 4. foldl accumulation: cat accumulated_kv with new_chunk_kv
        if state.accumulated_kv.shape[0] == 0:
            updated_kv = new_chunk_kv
        else:
            updated_kv = torch.cat([state.accumulated_kv, new_chunk_kv], dim=0)

        # 5. Drift calculation: ||new_chunk_kv - prev_tail||_F / ||prev_tail||_F
        if state.accumulated_kv.shape[0] > 0:
            prev_tail = state.accumulated_kv[-new_chunk_kv.shape[0]:]
            drift = (
                (new_chunk_kv.float() - prev_tail.float()).norm().item()
                / prev_tail.float().norm().clamp(min=1e-8).item()
            )
        else:
            drift = float("inf")

        # 6. Plateau detection: drift below threshold after at least 2 steps
        plateau = drift < self.config.drift_threshold and state.fold_depth >= 2

        # 7. StreamingLLM fallback under memory pressure
        max_tokens = self.config.window_size * self.config.chunk_size
        if self.config.enable_streaming_fallback and updated_kv.shape[0] > max_tokens:
            sink_tokens = min(4, updated_kv.shape[0])
            window_tokens = max_tokens
            keep_recent = window_tokens - sink_tokens
            if keep_recent > 0:
                updated_kv = torch.cat(
                    [updated_kv[:sink_tokens], updated_kv[-keep_recent:]],
                    dim=0,
                )
            else:
                updated_kv = updated_kv[:sink_tokens]

        # 8. Apply SRFT+INT4 compression to old chunks when compressor is attached
        # (B+C integration: recent window stays FP16, older chunks are compressed)
        if self.config.compressor is not None and updated_kv.shape[0] > self.config.chunk_size:
            compressor = self.config.compressor
            # Only compress if the tensor is shaped correctly for the kernel
            if (updated_kv.dim() == 4 and
                    updated_kv.shape[2] == self.config.n_heads and
                    updated_kv.shape[3] == self.config.d_head):
                try:
                    updated_kv = compressor.compression_hook("fold", updated_kv)
                except Exception:
                    pass  # compression failure is non-fatal

        # 9. Generate new fold_key and store state
        chunk_hash = self._store.chunk_key(chunk_tokens, 0, layer_idx)
        new_fold_key = f"fold:{layer_idx}:{chunk_hash}:{state.fold_depth}"
        new_state = _FoldState(
            accumulated_kv=updated_kv,
            chunk_ids=state.chunk_ids + [hash(tuple(chunk_tokens))],
            fold_depth=state.fold_depth + 1,
            last_drift=drift,
            plateau_reached=plateau,
        )
        if len(self._fold_states) >= self.config.max_entries:
            self.evict()
        self._fold_states[new_fold_key] = new_state

        return new_fold_key, updated_kv

    def get_folded_prefix(self, fold_key: str) -> Optional[torch.Tensor]:
        """Return the accumulated KV state as a prefix tensor.

        Returns: [n_acc_tokens, 2, n_heads, d_head] float, or None on miss.
        """
        state = self._fold_states.get(fold_key)
        if state is None:
            return None
        self._fold_hits += 1
        self._hits += 1
        return state.accumulated_kv

    def register_prefolded_prefix(
        self,
        fold_key: str,
        accumulated_kv: torch.Tensor,
        chunk_ids: List[int],
    ) -> None:
        """Register a pre-accumulated KV from AgenticChunkPreCachingPipeline.

        Stores as a stable fold state (plateau_reached=True) and also registers
        the tensor under fold_key in the segment store for RadixAttention access.
        """
        state = _FoldState(
            accumulated_kv=accumulated_kv,
            chunk_ids=chunk_ids,
            fold_depth=len(chunk_ids),
            last_drift=0.0,
            plateau_reached=True,
        )
        self._fold_states[fold_key] = state
        # Also accessible via standard put/get path
        self._store.put(fold_key, accumulated_kv)

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that came from foldl accumulation or non-contiguous segments."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return (self._noncontiguous_hits + self._fold_hits) / total_hits

    def get_fold_depth_stats(self) -> Dict[str, float]:
        """foldl chain statistics: mean depth and plateau convergence ratio."""
        if not self._fold_states:
            return {"mean_depth": 0.0, "plateau_ratio": 0.0, "n_chains": 0}
        depths = [s.fold_depth for s in self._fold_states.values()]
        plateaus = [1 for s in self._fold_states.values() if s.plateau_reached]
        return {
            "mean_depth": sum(depths) / len(depths),
            "plateau_ratio": len(plateaus) / len(depths),
            "n_chains": len(depths),
        }

    # ------------------------------------------------------------------ #
    # Segment API (SegmentedHashCache compatible)                          #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Standard segment store (RadixAttention compatible)."""
        self._store.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Delegate to segment store (runner.py compatibility)."""
        return self._store.get_segments(token_ids, layer_idx)

    def get_segments_with_fold(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        fold_key: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int], Optional[torch.Tensor]]:
        """RadixAttention prefix match + foldl fallback combined lookup.

        Returns:
          hits: [(chunk_idx, kv)]
          misses: [chunk_idx]
          fold_prefix: Optional[Tensor] — foldl accumulated prefix (if available)
        """
        hits, misses = self._store.get_segments(token_ids, layer_idx)
        fold_prefix = self.get_folded_prefix(fold_key) if fold_key else None

        # Track non-contiguous hits (hits where an earlier chunk is a miss)
        if hits:
            miss_set = set(misses)
            for idx, _ in hits:
                if any(m < idx for m in miss_set):
                    self._noncontiguous_hits += 1

        return hits, misses, fold_prefix

    # ------------------------------------------------------------------ #
    # Internal utilities                                                   #
    # ------------------------------------------------------------------ #

    def _compute_chunk_kv(
        self,
        chunk_tokens: List[int],
        layer_idx: int,
        prefix_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesise chunk KV for simulation (no real model required).

        In a real engine this would come from src/engine/runner.py via a
        model forward pass. Uses a deterministic seed from token content.
        """
        n_tok = len(chunk_tokens)
        if n_tok == 0:
            n_tok = self.config.chunk_size
        seed = (sum(chunk_tokens) + layer_idx) % (2 ** 31)
        torch.manual_seed(seed)
        kv = torch.randn(n_tok, 2, self.config.n_heads, self.config.d_head)
        return kv

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """Delegate chunk key generation to segment store."""
        return self._store.chunk_key(token_ids, chunk_idx, layer_idx)
