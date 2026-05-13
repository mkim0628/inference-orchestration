"""AgenticChunkPreCachingPipeline — PBKV prediction + KVFold pre-accumulation (Activity A+B).

Integrates PBKVAgentSegmentPreservationScheduler (A) with KVFoldAccumulativeRadixCache (B):
  1. PBKV predictor decides which future chunks (chunk set S) to pre-accumulate
  2. KVFoldAccumulativeRadixCache foldl-processes those chunks in order
  3. The accumulated KV is registered as a "pre-folded prefix" in the RadixCache leaf
  4. At inference time only incremental foldl processing is needed (TTFT reduction)
  5. On prediction miss: fallback to RadixAttention (SegmentedHashCache) path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.kv_fold_accumulative import KVFoldAccumulativeRadixCache, KVFoldConfig
from src.engine.runner import InferenceRequest
from src.scheduler.pbkv_agent_segment_scheduler import (
    PBKVAgentSegmentPreservationScheduler,
    PBKVConfig,
)


@dataclass
class AgenticPreCachingConfig:
    kvfold: Optional[KVFoldConfig] = None
    pbkv: Optional[PBKVConfig] = None
    precache_top_k: int = 10
    precache_min_prob: float = 0.5

    def __post_init__(self) -> None:
        if self.kvfold is None:
            self.kvfold = KVFoldConfig()
        if self.pbkv is None:
            self.pbkv = PBKVConfig()


class AgenticChunkPreCachingPipeline(CacheStore):
    """PBKV prediction + KVFold pre-accumulation pipeline (Activity A+B).

    Full CacheStore implementation delegates to KVFoldAccumulativeRadixCache.

    Core flow:
      1. precache_predicted_chunks() — run prediction + pre-accumulate before batch
      2. get_with_precache() — use pre-folded prefix at inference time
      3. prediction miss → fallback_to_radix_attention()
    """

    def __init__(self, config: AgenticPreCachingConfig) -> None:
        self.config = config
        self.fold_cache = KVFoldAccumulativeRadixCache(config.kvfold)
        self.scheduler = PBKVAgentSegmentPreservationScheduler(
            self.fold_cache, config.pbkv
        )
        # fold_key → {'chunk_ids': [...], 'prob': float}
        self._prefolded_registry: Dict[str, Dict] = {}
        self._hits = 0
        self._misses = 0
        self._precache_hits = 0
        self._fallback_count = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface (delegate to KVFoldAccumulativeRadixCache)      #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        self.fold_cache.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        result = self.fold_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self.fold_cache.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self.fold_cache.memory_bytes()

    def reset_stats(self) -> None:
        self.fold_cache.reset_stats()
        self._hits = 0
        self._misses = 0
        self._precache_hits = 0
        self._fallback_count = 0

    # ------------------------------------------------------------------ #
    # A+B pipeline API                                                     #
    # ------------------------------------------------------------------ #

    def precache_predicted_chunks(
        self,
        agent_id: str,
        candidate_chunk_tokens_list: List[List[int]],
        layer_idx: int = 0,
    ) -> Optional[str]:
        """PBKV prediction + KVFold pre-accumulation.

        Returns fold_key of the pre-accumulated prefix, or None if no chunks
        passed the precache_min_prob threshold.
        """
        if not candidate_chunk_tokens_list:
            return None

        # 1. Score each candidate chunk via the PBKV predictor
        scored_chunks: List[Tuple[float, List[int]]] = []
        for chunk_tokens in candidate_chunk_tokens_list:
            req_proxy = InferenceRequest(
                request_id=f"{agent_id}_{hash(tuple(chunk_tokens))}",
                token_ids=chunk_tokens,
            )
            prob = self.scheduler._predict_segment_reuse(req_proxy)
            scored_chunks.append((prob, chunk_tokens))

        # 2. Select top-k chunks above the minimum probability threshold
        scored_chunks.sort(key=lambda x: -x[0])
        selected = [
            (p, t)
            for p, t in scored_chunks[: self.config.precache_top_k]
            if p >= self.config.precache_min_prob
        ]
        if not selected:
            return None

        # 3. Pre-accumulate selected chunks via KVFold (preserve original order)
        # Build a set of selected token-list hashes for O(1) membership test
        selected_hashes = {id(t): True for _, t in selected}
        selected_set = {tuple(t): True for _, t in selected}
        original_order = [t for p, t in scored_chunks if tuple(t) in selected_set]

        fold_key: Optional[str] = None
        for chunk_tokens in original_order:
            fold_key, _ = self.fold_cache.fold_chunk(chunk_tokens, layer_idx, fold_key)

        # 4. Register pre-folded prefix in RadixCache leaf node
        if fold_key is not None:
            accumulated = self.fold_cache.get_folded_prefix(fold_key)
            if accumulated is not None:
                chunk_ids = [hash(tuple(t)) for _, t in selected]
                self.fold_cache.register_prefolded_prefix(fold_key, accumulated, chunk_ids)
                self._prefolded_registry[fold_key] = {
                    "chunk_ids": chunk_ids,
                    "prob": sum(p for p, _ in selected) / len(selected),
                }

        return fold_key

    def get_with_precache(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        precache_fold_key: Optional[str] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int], Optional[torch.Tensor]]:
        """Lookup with pre-folded prefix priority.

        Returns:
          hits: [(chunk_idx, kv)]
          misses: [chunk_idx]
          fold_prefix: Optional[Tensor] — available pre-folded prefix
        """
        # 1. Check for pre-folded prefix
        fold_prefix: Optional[torch.Tensor] = None
        if precache_fold_key and precache_fold_key in self._prefolded_registry:
            fold_prefix = self.fold_cache.get_folded_prefix(precache_fold_key)
            if fold_prefix is not None:
                self._precache_hits += 1

        # 2. Segment lookup (RadixAttention + foldl fallback)
        hits, misses, _ = self.fold_cache.get_segments_with_fold(
            token_ids, layer_idx, precache_fold_key
        )
        return hits, misses, fold_prefix

    def fallback_to_radix_attention(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """On prediction miss, fall back to the SegmentedHashCache RadixAttention path."""
        self._fallback_count += 1
        return self.fold_cache._store.get_segments(token_ids, layer_idx)

    def noncontiguous_hit_rate(self) -> float:
        """Delegate to the underlying KVFoldAccumulativeRadixCache."""
        return self.fold_cache.noncontiguous_hit_rate()

    def precache_efficiency(self) -> float:
        """Fraction of total accesses that were served from a pre-folded prefix."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._precache_hits / total
