"""
QueryCentricTriAttentionCache — Activity B+C (Cross-1 integration).

Combines QueryCentricRecomputeCache (Activity B) with TriAttentionCodec
(Activity C) to create a dual-path storage strategy:
  - High-relevance segments: stored as raw KV (recompute quality preserved).
  - Low-relevance segments: compressed with TriAttentionCodec (10× ratio).

This avoids crowding out high-relevance segments with bulky low-relevance KV
while still retaining them in compressed form for potential future use.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.cache.base import CacheStore
from src.cache.query_centric_recompute import QueryCentricRecomputeCache
from src.cache.tri_attention_codec import TriAttentionCodec


class QueryCentricTriAttentionCache(CacheStore):
    """
    Cross-1: QueryCentricRecomputeCache + TriAttentionCodec unified cache.

    Design principle:
      - Segments with cosine similarity >= relevance_threshold to the query
        are stored as raw KV (preserving reconstruction quality).
      - Segments below the threshold are compressed by TriAttentionCodec
        and stored in a separate compressed store.
      - Recomputation always uses raw KV (never the lossy compressed version).

    put() stores without query context (relevance unknown); use put_with_query()
    to route to raw vs. compressed storage explicitly.
    """

    def __init__(
        self,
        capacity_bytes: int,
        codec: TriAttentionCodec,
        recompute_budget_ratio: float = 0.20,
        relevance_threshold: float = 0.60,
        compression_ratio: float = 0.10,
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.relevance_threshold = relevance_threshold
        self.compression_ratio = compression_ratio

        # Underlying QCRC handles raw KV + saliency metadata
        self._qcrc = QueryCentricRecomputeCache(
            capacity_bytes=capacity_bytes,
            recompute_budget_ratio=recompute_budget_ratio,
        )
        self._codec = codec

        # Separate compressed store for low-relevance segments
        self._compressed_store: Dict[str, Dict] = {}
        # Separate raw store for high-relevance segments (for fast lookup)
        self._raw_store: Dict[str, torch.Tensor] = {}

        # Stats (unified across both stores)
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store a KV segment without query context.

        Without query context the relevance cannot be determined, so the
        segment is forwarded to QCRC (raw storage) as a safe default.

        Args:
            key: Segment hash string.
            value: KV tensor [layers, heads, seq_len, head_dim].
        """
        self._qcrc.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a KV tensor.

        Lookup order: raw_store → compressed_store → QCRC store.

        Args:
            key: Segment hash string.

        Returns:
            KV tensor or None on miss.
        """
        if key in self._raw_store:
            self._hit_count += 1
            return self._raw_store[key]

        if key in self._compressed_store:
            self._hit_count += 1
            return self._codec.decompress(self._compressed_store[key])

        result = self._qcrc.get(key)
        if result is None:
            self._miss_count += 1
        else:
            self._hit_count += 1
        return result

    def evict(self) -> int:
        """Evict one entry from the compressed store, then QCRC if empty.

        Returns bytes freed.
        """
        if self._compressed_store:
            key = next(iter(self._compressed_store))
            entry = self._compressed_store.pop(key)
            return entry["kv"].nbytes
        if self._raw_store:
            key = next(iter(self._raw_store))
            tensor = self._raw_store.pop(key)
            return tensor.nbytes
        return self._qcrc.evict()

    def hit_rate(self) -> float:
        """Cumulative cache hit rate across all stores."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Total memory across raw store, compressed store, and QCRC."""
        raw_bytes = sum(t.nbytes for t in self._raw_store.values())
        compressed_bytes = sum(
            e["kv"].nbytes for e in self._compressed_store.values()
        )
        return raw_bytes + compressed_bytes + self._qcrc.memory_bytes()

    def reset_stats(self) -> None:
        """Reset hit/miss counters in this store and QCRC."""
        self._hit_count = 0
        self._miss_count = 0
        self._qcrc.reset_stats()

    # ------------------------------------------------------------------ #
    # Query-aware dual-path API                                            #
    # ------------------------------------------------------------------ #

    def put_with_query(
        self,
        key: str,
        value: torch.Tensor,
        keys_pre_rope: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> None:
        """Store with query-context-aware routing.

        High-relevance segments (cosine sim >= relevance_threshold) are stored
        as raw KV in both _raw_store and QCRC (for selective_recompute).
        Low-relevance segments are compressed and stored in _compressed_store.

        Args:
            key: Segment hash string.
            value: KV tensor [layers, heads, seq_len, head_dim].
            keys_pre_rope: Pre-RoPE K tensor (same shape as value).
            query_embedding: Query mean K vector [head_dim].
        """
        seg_emb = value.mean(dim=(0, 1, 2))  # [head_dim]
        relevance = F.cosine_similarity(
            query_embedding.unsqueeze(0).float(),
            seg_emb.unsqueeze(0).float(),
        ).item()

        if relevance >= self.relevance_threshold:
            # High-relevance: store raw for quality recomputation
            self._raw_store[key] = value.detach().clone()
            self._qcrc.put(key, value)
        else:
            # Low-relevance: compress to save capacity
            if self._codec.mu_k is not None:
                compressed = self._codec.compress(
                    value, keys_pre_rope, self.compression_ratio
                )
                self._compressed_store[key] = compressed
            else:
                # Codec not calibrated yet: fall back to raw storage in QCRC
                self._qcrc.put(key, value)

    def selective_recompute(
        self,
        query: torch.Tensor,
        cached_segments: List[str],
        budget: float = 0.20,
    ) -> List[str]:
        """Delegate recompute budget allocation to QCRC using only raw KV.

        Compressed segments are excluded because reconstruction quality is
        insufficient for recomputation (lossy approximation).

        Args:
            query: Query mean K vector [head_dim].
            cached_segments: Candidate segment hashes.
            budget: Max fraction of total tokens to recompute.

        Returns:
            Selected segment hashes for recomputation.
        """
        # Only segments with raw KV are eligible for recomputation
        raw_segments = [k for k in cached_segments if k in self._raw_store]
        # Include segments stored only in QCRC (put() without query)
        qcrc_segments = [
            k for k in cached_segments
            if k not in self._raw_store and k not in self._compressed_store
        ]
        eligible = raw_segments + qcrc_segments
        return self._qcrc.selective_recompute(query, eligible, budget)

    def compressed_keys(self) -> List[str]:
        """Return keys of all compressed segments."""
        return list(self._compressed_store.keys())

    def raw_keys(self) -> List[str]:
        """Return keys of all raw (high-relevance) segments."""
        return list(self._raw_store.keys())
