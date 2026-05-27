"""IndexMem Soft Hit Segment Cache — non-contiguous reuse with latent soft hits (Activity B-1).

Extends binary hit/miss model to a continuum:
  - Hard Hit: physical KV in cache -> return kv_tensor
  - Soft Hit: physical KV evicted but latent state in pool -> return latent_state
  - Miss: neither -> recompute needed

arXiv 2605.25475.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.indexmem_latent_memory_module import (
    IndexMemLatentMemoryModule,
    LatentMemoryConfig,
)


@dataclass
class HitResult:
    """Cache lookup result. Extends binary hit/miss to a continuum."""
    type: Literal["hard", "soft", "miss"]
    kv_tensor: Optional[torch.Tensor] = None      # physical KV on hard hit
    latent_state: Optional[torch.Tensor] = None   # latent state on soft hit
    segment_key: str = ""


@dataclass
class SoftHitSegmentConfig:
    chunk_size: int = 128
    max_physical_entries: int = 1000
    latent_pool_max_segments: int = 10000
    beta_soft: float = 0.1
    beta_weight: float = 0.5
    latent_dim: int = 64
    kv_dim: int = 128
    n_layers: int = 32
    seed: int = 42


class IndexMemSoftHitSegmentCache(CacheStore):
    """IndexMem Soft Hit Non-Contiguous Segment Cache (Activity B-1).

    Extends the binary non-contiguous reuse model to a continuum by maintaining
    latent states for physically evicted segments, enabling "soft hits".
    """

    def __init__(self, config: SoftHitSegmentConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._physical_store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._segment_latent_pool: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._latent_encoder = IndexMemLatentMemoryModule(
            LatentMemoryConfig(
                latent_dim=config.latent_dim,
                kv_dim=config.kv_dim,
                n_layers=config.n_layers,
                seed=config.seed,
            )
        )
        self._n_hard_hits: int = 0
        self._n_soft_hits: int = 0
        self._n_misses: int = 0
        self._n_noncontiguous_hard_hits: int = 0
        self._n_noncontiguous_soft_hits: int = 0

    # ---- CacheStore abstract method implementations ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store physical KV. Evict LRU + encode latent if over capacity."""
        if key in self._physical_store:
            self._physical_store.move_to_end(key)
        else:
            if len(self._physical_store) >= self.config.max_physical_entries:
                self.evict()
            self._physical_store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve physical KV. For soft hit path, use get_hit_result()."""
        if key in self._physical_store:
            self._physical_store.move_to_end(key)
            self._n_hard_hits += 1
            return self._physical_store[key]
        elif key in self._segment_latent_pool:
            self._n_soft_hits += 1
            return None  # soft hit: physical KV not available
        else:
            self._n_misses += 1
            return None

    def evict(self) -> int:
        """LRU eviction. Encodes latent state before removing physical KV."""
        if not self._physical_store:
            return 0
        evict_key, evicted_kv = self._physical_store.popitem(last=False)  # LRU = first

        # Encode evicted KV into latent state
        if evicted_kv.numel() > 0:
            encoded = self._encode_to_latent(evicted_kv, evict_key)
            self._add_to_latent_pool(evict_key, encoded)

        return evicted_kv.nbytes

    def hit_rate(self) -> float:
        """Binary hit rate (hard hits only). Backward compatible."""
        total = self._n_hard_hits + self._n_soft_hits + self._n_misses
        return self._n_hard_hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Physical KV memory footprint."""
        return sum(v.nbytes for v in self._physical_store.values())

    def reset_stats(self) -> None:
        self._n_hard_hits = 0
        self._n_soft_hits = 0
        self._n_misses = 0
        self._n_noncontiguous_hard_hits = 0
        self._n_noncontiguous_soft_hits = 0

    # ---- Soft hit extended API ----

    def get_hit_result(self, segment_key: str) -> HitResult:
        """Return HitResult including soft hit path.

        Algorithm:
          hard -> physical KV exists
          soft -> physical KV evicted but latent state in pool
          miss -> neither
        """
        if segment_key in self._physical_store:
            self._physical_store.move_to_end(segment_key)
            self._n_hard_hits += 1
            return HitResult(
                type="hard",
                kv_tensor=self._physical_store[segment_key],
                segment_key=segment_key,
            )
        elif segment_key in self._segment_latent_pool:
            self._segment_latent_pool.move_to_end(segment_key)
            self._n_soft_hits += 1
            return HitResult(
                type="soft",
                latent_state=self._segment_latent_pool[segment_key],
                segment_key=segment_key,
            )
        else:
            self._n_misses += 1
            return HitResult(type="miss", segment_key=segment_key)

    def soft_hit_residual(
        self,
        query_states: torch.Tensor,
        latent_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute residual from latent state for soft hit path.

        beta_soft automatically adjusted: increased when only soft hits available.
        """
        total = self._n_hard_hits + self._n_soft_hits
        # Adaptive beta: scale up beta_soft when mostly soft hits
        if total > 0 and self._n_hard_hits == 0:
            beta = min(self.config.beta_soft * 2.0, 0.2)
        elif total > 0:
            hard_ratio = self._n_hard_hits / total
            beta = self.config.beta_soft * (1.0 + (1.0 - hard_ratio))
            beta = min(beta, self.config.beta_soft * 2.0)
        else:
            beta = self.config.beta_soft

        # Simple cross-attention: query @ latent
        orig_shape = query_states.shape
        q_flat = query_states.float().reshape(-1, orig_shape[-1])  # [n_q, d_head]
        latent_f = latent_state.float()  # [latent_dim]

        # Project latent to query space via dot product
        d_head = q_flat.shape[-1]
        latent_dim = latent_f.shape[0]

        if latent_dim == d_head:
            # Direct dot product
            attn = torch.softmax(
                q_flat @ latent_f.unsqueeze(-1) / (d_head ** 0.5), dim=0
            )  # [n_q, 1]
            readout = attn * latent_f.unsqueeze(0)  # [n_q, d_head]
        else:
            # Truncate or pad latent to match d_head
            if latent_dim > d_head:
                lat_proj = latent_f[:d_head]
            else:
                lat_proj = torch.cat(
                    [latent_f, torch.zeros(d_head - latent_dim)], dim=0
                )
            attn = torch.softmax(
                q_flat @ lat_proj.unsqueeze(-1) / (d_head ** 0.5), dim=0
            )  # [n_q, 1]
            readout = attn * lat_proj.unsqueeze(0)  # [n_q, d_head]

        result = beta * readout.reshape(orig_shape)
        return result.to(query_states.dtype)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> str:
        """SegmentedHashCache-compatible API. Returns segment_key."""
        key = self._chunk_key(token_ids, chunk_idx, layer_idx)
        self.put(key, kv)
        return key

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, HitResult]], List[int]]:
        """Segment lookup with soft hit path included.

        Returns:
            hits: [(chunk_idx, HitResult), ...] — hard + soft hits
            miss_chunk_indices: [int, ...] — complete misses
        """
        n_chunks = max(1, (len(token_ids) + self.config.chunk_size - 1) // self.config.chunk_size)
        hits: List[Tuple[int, HitResult]] = []
        miss_indices: List[int] = []

        # Track which chunks precede hits for non-contiguous detection
        chunk_results: List[Optional[HitResult]] = []
        for i in range(n_chunks):
            key = self._chunk_key(token_ids, i, layer_idx)
            result = self.get_hit_result(key)
            chunk_results.append(result)

        miss_set: set = set()
        for i, result in enumerate(chunk_results):
            if result.type == "miss":
                miss_set.add(i)

        for i, result in enumerate(chunk_results):
            if result.type != "miss":
                hits.append((i, result))
                # Non-contiguous: any miss chunk index < current hit index
                if any(m < i for m in miss_set):
                    if result.type == "hard":
                        self._n_noncontiguous_hard_hits += 1
                    else:
                        self._n_noncontiguous_soft_hits += 1
            else:
                miss_indices.append(i)

        return hits, miss_indices

    def noncontiguous_hit_rate(self) -> float:
        """Non-contiguous hard hit rate."""
        total_hits = self._n_hard_hits
        if total_hits == 0:
            return 0.0
        return self._n_noncontiguous_hard_hits / total_hits

    def soft_hit_rate(self) -> float:
        """Soft hit rate: n_soft_hits / total."""
        total = self._n_hard_hits + self._n_soft_hits + self._n_misses
        return self._n_soft_hits / total if total > 0 else 0.0

    def weighted_hit_rate(self) -> float:
        """Weighted hit rate: (n_hard + beta_weight * n_soft) / total."""
        total = self._n_hard_hits + self._n_soft_hits + self._n_misses
        if total == 0:
            return 0.0
        return (
            self._n_hard_hits + self.config.beta_weight * self._n_soft_hits
        ) / total

    # ---- Internal helpers ----

    def _chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int,
    ) -> str:
        start = chunk_idx * self.config.chunk_size
        end = start + self.config.chunk_size
        chunk = token_ids[start:end]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

    def _encode_to_latent(self, kv: torch.Tensor, segment_key: str) -> torch.Tensor:
        """Encode KV tensor to latent state vector."""
        # Use latent encoder to get [latent_dim] representation
        latent = self._latent_encoder.encode_evicted(kv, segment_key, layer_idx=0)
        return latent.detach()

    def _add_to_latent_pool(self, key: str, latent: torch.Tensor) -> None:
        """Add latent state to pool with LRU eviction if over capacity."""
        if len(self._segment_latent_pool) >= self.config.latent_pool_max_segments:
            # Evict oldest (LRU)
            self._segment_latent_pool.popitem(last=False)
        self._segment_latent_pool[key] = latent
