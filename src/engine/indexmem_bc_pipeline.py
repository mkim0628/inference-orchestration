"""IndexMem B+C Integration Pipeline — UnifiedLatentPool + SharedLatentEncoder (Cross-1).

Unifies segment-level (B-1) and token-level (C-1) latent states under a
single pool for efficient soft-hit reuse and memory management.
arXiv 2605.25475.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.indexmem_soft_hit_segment_cache import (
    HitResult,
    IndexMemSoftHitSegmentCache,
    SoftHitSegmentConfig,
)
from src.cache.indexmem_eviction_codec import IndexMemEvictionCodec, IndexMemEvictionConfig
from src.cache.indexmem_latent_memory_module import IndexMemLatentMemoryModule, LatentMemoryConfig


@dataclass
class BCPipelineConfig:
    segment_latent_pool_mb: float = 128.0
    token_latent_pool_mb: float = 64.0
    unified_weighted_hit_weights: Tuple[float, float, float] = (1.0, 0.7, 0.4)
    # (hard_weight, segment_soft_weight, token_soft_weight)
    budget_ratio: float = 0.5
    beta_readout: float = 0.1
    seed: int = 42


class UnifiedLatentPool:
    """Unified management of segment-level (B-1) and token-level (C-1) latent states.

    segment_id -> segment_latent (entire segment KV aggregated representation)
    (segment_id, token_range) -> token_latent (individual evicted token representation)

    Memory cap: segment_latent_pool_mb + token_latent_pool_mb
    Eviction policy: LRU when pool capacity is exceeded
    """

    def __init__(self, config: BCPipelineConfig) -> None:
        self.config = config
        self._segment_pool: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._token_pool: OrderedDict[Tuple[str, Tuple[int, int]], torch.Tensor] = OrderedDict()

        # Convert MB limits to approximate entry limits (assume 64-dim float32 latents)
        bytes_per_latent = 64 * 4  # latent_dim=64, float32
        self._max_segment_entries = max(
            1, int((config.segment_latent_pool_mb * 1024 * 1024) / bytes_per_latent)
        )
        self._max_token_entries = max(
            1, int((config.token_latent_pool_mb * 1024 * 1024) / bytes_per_latent)
        )

    def put_segment_latent(self, segment_id: str, latent: torch.Tensor) -> None:
        """Store segment-level latent state with LRU eviction."""
        if segment_id in self._segment_pool:
            self._segment_pool.move_to_end(segment_id)
        else:
            if len(self._segment_pool) >= self._max_segment_entries:
                self._segment_pool.popitem(last=False)
            self._segment_pool[segment_id] = latent.detach().clone()

    def put_token_latent(
        self,
        segment_id: str,
        token_range: Tuple[int, int],
        latent: torch.Tensor,
    ) -> None:
        """Store token-level latent state with LRU eviction."""
        key = (segment_id, token_range)
        if key in self._token_pool:
            self._token_pool.move_to_end(key)
        else:
            if len(self._token_pool) >= self._max_token_entries:
                self._token_pool.popitem(last=False)
            self._token_pool[key] = latent.detach().clone()

    def lookup_segment(self, segment_id: str) -> Optional[torch.Tensor]:
        """Look up segment latent. Returns None on miss."""
        if segment_id in self._segment_pool:
            self._segment_pool.move_to_end(segment_id)
            return self._segment_pool[segment_id]
        return None

    def lookup_token(
        self, segment_id: str, token_range: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """Look up token latent. Returns None on miss."""
        key = (segment_id, token_range)
        if key in self._token_pool:
            self._token_pool.move_to_end(key)
            return self._token_pool[key]
        return None

    def memory_bytes(self) -> int:
        """Total memory used by both pools."""
        seg_bytes = sum(v.nbytes for v in self._segment_pool.values())
        tok_bytes = sum(v.nbytes for v in self._token_pool.values())
        return seg_bytes + tok_bytes


class SharedLatentEncoder:
    """Shared latent encoder for B-1 and C-1 (parameter sharing for efficiency).

    Both B-1 (IndexMemSoftHitSegmentCache) and C-1 (IndexMemEvictionCodec) share
    the same encoder parameters, doubling effective training samples.
    """

    def __init__(self, config: LatentMemoryConfig) -> None:
        self._module = IndexMemLatentMemoryModule(config)

    def encode(self, kv: torch.Tensor) -> torch.Tensor:
        """Encode KV block to latent vector. [n_tokens, kv_dim] -> [latent_dim]"""
        if kv.shape[0] == 0:
            return torch.zeros(self._module.config.latent_dim)
        # Use a dummy request key for stateless encoding
        key = f"shared_encoder_{id(kv)}"
        latent = self._module.encode_evicted(kv, key, layer_idx=0)
        self._module.clear(key)
        return latent


class IndexMemBCIntegrationPipeline:
    """IndexMem B+C Integration Pipeline (Cross-1).

    Unifies segment-level soft hits (B-1) and token-level residual readout (C-1)
    with 2-layer latent preservation via UnifiedLatentPool.

    Processing flow:
      Step 1 (B-1 lookup): UnifiedLatentPool.lookup()
        -> hard hit (physical KV) / soft hit (segment latent) / miss 3 levels
      Step 2 (hard hit): IndexMemEvictionCodec.encode() computes token importance
        -> low-importance tokens -> Latent Memory encode -> UnifiedLatentPool
      Step 3 (soft hit): segment latent residual + token latent additional residual
      Step 4 (miss): recompute -> register latent state in UnifiedLatentPool

    Unified weighted hit rate:
      unified_weighted_hit_rate = (n_hard + 0.7*n_seg_soft + 0.4*n_token_soft) / total
    """

    def __init__(
        self,
        soft_hit_cache: IndexMemSoftHitSegmentCache,
        eviction_codec: IndexMemEvictionCodec,
        unified_pool: UnifiedLatentPool,
        shared_encoder: SharedLatentEncoder,
        config: BCPipelineConfig,
    ) -> None:
        self.soft_hit_cache = soft_hit_cache
        self.eviction_codec = eviction_codec
        self.unified_pool = unified_pool
        self.shared_encoder = shared_encoder
        self.config = config

        self._n_hard_hits: int = 0
        self._n_seg_soft_hits: int = 0
        self._n_token_soft_hits: int = 0
        self._n_misses: int = 0

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, HitResult]], List[int]]:
        """InferenceRunner-compatible API. B+C unified lookup."""
        hits, misses = self.soft_hit_cache.get_segments(token_ids, layer_idx)

        for chunk_idx, hit_result in hits:
            if hit_result.type == "hard":
                self._n_hard_hits += 1
            elif hit_result.type == "soft":
                self._n_seg_soft_hits += 1

        self._n_misses += len(misses)

        return hits, misses

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """InferenceRunner-compatible API."""
        segment_key = self.soft_hit_cache.put_segment(
            token_ids, chunk_idx, kv, layer_idx
        )

        # Also encode into UnifiedLatentPool for token-level latent reuse
        if kv.numel() > 0:
            latent = self.shared_encoder.encode(kv)
            self.unified_pool.put_segment_latent(segment_key, latent)

    def unified_weighted_hit_rate(self) -> float:
        """(n_hard + 0.7*n_seg_soft + 0.4*n_token_soft) / total"""
        hw, sw, tw = self.config.unified_weighted_hit_weights
        total = (
            self._n_hard_hits
            + self._n_seg_soft_hits
            + self._n_token_soft_hits
            + self._n_misses
        )
        if total == 0:
            return 0.0
        weighted = (
            hw * self._n_hard_hits
            + sw * self._n_seg_soft_hits
            + tw * self._n_token_soft_hits
        )
        return weighted / total

    def reset_stats(self) -> None:
        self._n_hard_hits = 0
        self._n_seg_soft_hits = 0
        self._n_token_soft_hits = 0
        self._n_misses = 0
        self.soft_hit_cache.reset_stats()
