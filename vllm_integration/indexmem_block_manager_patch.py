"""indexmem_block_manager_patch.py — Activity B: IndexMem Soft Hit Segment Cache for vLLM 0.21.0.

2026-05-27: IndexMemSoftHitKVCacheManagerMixin — ports IndexMemSoftHitSegmentCache
            (Activity B) into vLLM's v1 KVCacheManager as a mixin.

            Provides a three-tier cache lookup model:
              Hard Hit : physical KV in block store → return kv_tensor directly
              Soft Hit : physical KV evicted but latent state in SegmentLatentPool →
                         return latent state for residual reconstruction
              Miss     : neither → recompute needed

            SegmentLatentPool:
              In-memory DRAM store (NOT HBM/GPU VRAM) for latent state vectors of
              evicted KV segments. Keyed by segment_id (SHA-256 of token content).
              LRU eviction when over latent_pool_max_segments capacity.

            Core API:
              store_soft_hit_segment(session_id, turn_id, token_ids, chunk_idx,
                                     kv_tensor, layer_idx):
                Store physical KV segment with latent encoding. Supports both
                hard-hit (physical KV) and soft-hit (latent state) paths.

              get_soft_hit_result(session_id, turn_id, token_ids, chunk_idx, layer_idx):
                Return HitResult with type "hard" | "soft" | "miss".

              allocate_soft_hit_block(segment_id):
                Reserve a logical block slot for a soft-hit segment.
                Returns block_idx (int) or None if no slot available.

              weighted_hit_rate():
                (n_hard + beta_weight * n_soft) / total — tracks combined cache
                effectiveness including soft-hit contribution.

            Integration with vLLM v1 architecture:
              IndexMemSoftHitKVCacheManagerMixin operates as a PARALLEL segment
              store alongside vLLM's native paged block pool. It does NOT replace
              or modify the native KV block allocation logic.

              The SegmentLatentPool stores latent states in DRAM — evicted segments
              that are no longer in GPU VRAM but can still contribute to attention
              quality via residual readout (IndexMemLatentMemoryModule pattern).

            Non-contiguous reuse path:
              The mixin tracks which segments were hit non-contiguously (a hit preceded
              by a miss in the chunk sequence). This is the Activity B metric:
              noncontiguous_fraction = noncontiguous_hits / total_hits >= 30%.

            weighted_hit_rate tracking:
              weighted_hit_rate = (n_hard + beta_weight * n_soft) / total
              beta_weight=0.5 by default (soft hits count as half a hard hit).

vLLM version: 0.21.0
Activity: B — IndexMem Soft Hit Non-Contiguous Segment Cache
"""

from __future__ import annotations

import sys
import pathlib
import hashlib
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, TYPE_CHECKING

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import vllm


def _vllm_version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:3])


assert _vllm_version_tuple(vllm.__version__) >= _vllm_version_tuple("0.4.0"), (
    f"vllm_integration requires vLLM >= 0.4.0, found {vllm.__version__}"
)


def _add_repo_root_to_path() -> None:
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _try_import_soft_hit_cache() -> tuple:
    """Import IndexMemSoftHitSegmentCache from src/."""
    _add_repo_root_to_path()
    try:
        from src.cache.indexmem_soft_hit_segment_cache import (
            IndexMemSoftHitSegmentCache,
            SoftHitSegmentConfig,
            HitResult,
        )
        return IndexMemSoftHitSegmentCache, SoftHitSegmentConfig, HitResult
    except ImportError:
        return None, None, None


# ---------------------------------------------------------------------------
# Import KVCacheManager (graceful fallback for CPU-only / no-GPU environments)
# ---------------------------------------------------------------------------

try:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    _KV_CACHE_MANAGER_AVAILABLE = True
except Exception:
    class KVCacheManager:  # type: ignore[no-redef]
        """Stub KVCacheManager for CPU-only environments."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    _KV_CACHE_MANAGER_AVAILABLE = False


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class SoftHitResult:
    """Cache lookup result for the three-tier soft-hit model."""
    type: Literal["hard", "soft", "miss"]
    kv_tensor: Optional[Any] = None        # physical KV on hard hit (torch.Tensor)
    latent_state: Optional[Any] = None     # latent vector on soft hit (torch.Tensor)
    segment_key: str = ""


@dataclass
class IndexMemSoftHitMixinConfig:
    """Configuration for IndexMemSoftHitKVCacheManagerMixin."""
    chunk_size: int = 128
    max_physical_entries: int = 1000
    latent_pool_max_segments: int = 10000
    beta_soft: float = 0.1              # readout strength for soft-hit residual
    beta_weight: float = 0.5            # soft-hit weight in weighted_hit_rate
    latent_dim: int = 64
    kv_dim: int = 128
    n_layers: int = 32
    seed: int = 42
    enabled: bool = True


# ===========================================================================
# SegmentLatentPool — DRAM store for evicted KV latent states
# ===========================================================================

class SegmentLatentPool:
    """In-memory (DRAM) latent state pool for evicted KV segments.

    Stores latent state vectors for physically evicted segments, enabling
    the "soft hit" path: even after a segment is evicted from GPU VRAM,
    its latent state remains in DRAM for residual reconstruction.

    Key design choices:
      - Stored in DRAM (CPU memory), NOT in GPU VRAM/HBM.
      - LRU eviction when over max_segments capacity.
      - Keyed by segment_id (string, typically SHA-256 based).
      - Latent vectors are CPU tensors (FP32) for memory efficiency.
    """

    def __init__(
        self,
        max_segments: int = 10000,
        latent_dim: int = 64,
    ) -> None:
        self.max_segments = max_segments
        self.latent_dim = latent_dim
        # OrderedDict for LRU eviction (most-recently-used at end)
        self._pool: OrderedDict[str, Any] = OrderedDict()
        self._n_evictions: int = 0

    def store(self, segment_id: str, latent: Any) -> None:
        """Store latent state vector. Evicts LRU if over capacity."""
        if segment_id in self._pool:
            self._pool.move_to_end(segment_id)
        else:
            if len(self._pool) >= self.max_segments:
                self._pool.popitem(last=False)  # evict LRU
                self._n_evictions += 1
            if _TORCH_AVAILABLE and isinstance(latent, torch.Tensor):
                self._pool[segment_id] = latent.detach().cpu().float()
            else:
                self._pool[segment_id] = latent

    def get(self, segment_id: str) -> Optional[Any]:
        """Retrieve latent state vector. Returns None on miss."""
        if segment_id not in self._pool:
            return None
        self._pool.move_to_end(segment_id)
        return self._pool[segment_id]

    def contains(self, segment_id: str) -> bool:
        """Check if segment_id has a latent state in the pool."""
        return segment_id in self._pool

    def evict(self, segment_id: str) -> bool:
        """Explicitly evict a segment from the pool. Returns True if found."""
        if segment_id in self._pool:
            del self._pool[segment_id]
            return True
        return False

    def size(self) -> int:
        """Current number of segments in the pool."""
        return len(self._pool)

    def memory_bytes(self) -> int:
        """Approximate memory usage of stored latent states (bytes)."""
        total = 0
        for v in self._pool.values():
            try:
                total += v.nbytes
            except AttributeError:
                total += self.latent_dim * 4  # assume FP32
        return total

    def stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        return {
            "pool_size": len(self._pool),
            "max_segments": self.max_segments,
            "total_evictions": self._n_evictions,
            "memory_bytes": self.memory_bytes(),
        }


# ===========================================================================
# Inline soft-hit cache stub (fallback when src/ is not importable)
# ===========================================================================

class _InlineSoftHitStore:
    """Minimal inline soft-hit cache (no src/ dependency).

    Maintains:
      _physical: OrderedDict[key → kv_tensor]  (hard-hit store)
      _latent_pool: SegmentLatentPool           (soft-hit store)
    """

    def __init__(self, config: IndexMemSoftHitMixinConfig) -> None:
        self.config = config
        self._physical: OrderedDict[str, Any] = OrderedDict()
        self._latent_pool = SegmentLatentPool(
            max_segments=config.latent_pool_max_segments,
            latent_dim=config.latent_dim,
        )
        self._n_hard = 0
        self._n_soft = 0
        self._n_miss = 0
        self._n_nc_hard = 0
        self._n_nc_soft = 0

    def put(self, key: str, kv: Any) -> None:
        """Store physical KV. LRU evict + encode latent if over capacity."""
        if key in self._physical:
            self._physical.move_to_end(key)
        else:
            if len(self._physical) >= self.config.max_physical_entries:
                self._evict_lru()
            if _TORCH_AVAILABLE and isinstance(kv, torch.Tensor):
                self._physical[key] = kv.detach().clone()
            else:
                self._physical[key] = kv

    def _evict_lru(self) -> None:
        """Evict LRU from physical store; encode latent state."""
        if not self._physical:
            return
        evict_key, evict_kv = self._physical.popitem(last=False)
        # Encode evicted KV to latent (mean pooling, FP32)
        if _TORCH_AVAILABLE and isinstance(evict_kv, torch.Tensor) and evict_kv.numel() > 0:
            latent = evict_kv.float().reshape(evict_kv.shape[0], -1).mean(dim=0)
            self._latent_pool.store(evict_key, latent)

    def get_hit_result(self, key: str) -> SoftHitResult:
        """Look up key; return HitResult with type hard/soft/miss."""
        if key in self._physical:
            self._physical.move_to_end(key)
            self._n_hard += 1
            return SoftHitResult(
                type="hard",
                kv_tensor=self._physical[key],
                segment_key=key,
            )
        elif self._latent_pool.contains(key):
            self._n_soft += 1
            return SoftHitResult(
                type="soft",
                latent_state=self._latent_pool.get(key),
                segment_key=key,
            )
        else:
            self._n_miss += 1
            return SoftHitResult(type="miss", segment_key=key)

    def soft_hit_residual(
        self,
        query_states: Any,
        latent_state: Any,
    ) -> Optional[Any]:
        """Compute beta-scaled residual from latent state."""
        if not _TORCH_AVAILABLE or latent_state is None:
            return None
        try:
            beta = self.config.beta_soft
            q_f = query_states.float().reshape(-1, query_states.shape[-1])
            d = q_f.shape[-1]
            lat = latent_state.float()
            ld = lat.shape[0]
            if ld > d:
                lat = lat[:d]
            elif ld < d:
                lat = torch.cat([lat, torch.zeros(d - ld, device=lat.device)])
            scale = d ** 0.5
            attn = torch.softmax(q_f @ lat.unsqueeze(-1) / scale, dim=0)
            readout = beta * (attn * lat.unsqueeze(0)).reshape(query_states.shape)
            return readout.to(query_states.dtype)
        except Exception:
            return None

    def weighted_hit_rate(self) -> float:
        total = self._n_hard + self._n_soft + self._n_miss
        if total == 0:
            return 0.0
        return (self._n_hard + self.config.beta_weight * self._n_soft) / total

    def soft_hit_rate(self) -> float:
        total = self._n_hard + self._n_soft + self._n_miss
        return self._n_soft / total if total > 0 else 0.0

    def hard_hit_rate(self) -> float:
        total = self._n_hard + self._n_soft + self._n_miss
        return self._n_hard / total if total > 0 else 0.0

    def noncontiguous_fraction(self) -> float:
        total_hits = self._n_hard + self._n_soft
        return (self._n_nc_hard + self._n_nc_soft) / total_hits if total_hits > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "n_hard": self._n_hard,
            "n_soft": self._n_soft,
            "n_miss": self._n_miss,
            "n_nc_hard": self._n_nc_hard,
            "n_nc_soft": self._n_nc_soft,
            "weighted_hit_rate": self.weighted_hit_rate(),
            "soft_hit_rate": self.soft_hit_rate(),
            "hard_hit_rate": self.hard_hit_rate(),
            "noncontiguous_fraction": self.noncontiguous_fraction(),
            "physical_store_size": len(self._physical),
            "latent_pool_size": self._latent_pool.size(),
            "latent_pool_memory_bytes": self._latent_pool.memory_bytes(),
        }


# ===========================================================================
# IndexMemSoftHitKVCacheManagerMixin — Activity B (2026-05-27)
# ===========================================================================

class IndexMemSoftHitKVCacheManagerMixin:
    """vLLM v1 KVCacheManager mixin: IndexMem soft-hit non-contiguous segment reuse.

    Activity B: Non-Contiguous KV Cache Reuse (IndexMem Soft Hit).

    This mixin adds a parallel soft-hit segment store alongside vLLM's native
    paged block pool. It does NOT replace or modify any native KVCacheManager
    methods — all existing methods are preserved.

    Three-tier cache lookup model:
      Hard Hit : physical KV in block store (full reuse, zero reconstruction error)
      Soft Hit : physical KV evicted, latent state in SegmentLatentPool → residual
      Miss     : neither → recompute required

    Core API:
        store_soft_hit_segment(session_id, turn_id, token_ids, chunk_idx,
                               kv_tensor, layer_idx):
            Store KV segment in physical store with latent encoding fallback.

        get_soft_hit_result(session_id, turn_id, token_ids, chunk_idx, layer_idx):
            Return SoftHitResult with type "hard" | "soft" | "miss".

        allocate_soft_hit_block(segment_id):
            Reserve a logical block slot for a soft-hit segment.
            Returns block_idx (int) or None.

        soft_hit_residual(query_states, latent_state):
            Compute beta-scaled latent residual for soft-hit reconstruction.

        weighted_hit_rate():
            (n_hard + beta_weight * n_soft) / total

    Non-contiguous tracking:
        The mixin tracks which segment hits are preceded by misses in the chunk
        sequence (non-contiguous). This drives the Activity B metric:
        noncontiguous_fraction = noncontiguous_hits / total_hits >= 30%.

    Usage:
        # Option A: mixin
        class MyKVCacheManager(IndexMemSoftHitKVCacheManagerMixin, KVCacheManager):
            pass

        # Option B: factory
        Mgr = make_indexmem_soft_hit_kv_cache_manager_class(KVCacheManager)
        mgr = Mgr(
            ...,  # standard KVCacheManager args
            indexmem_soft_hit_config=IndexMemSoftHitMixinConfig(
                budget_ratio=0.5, beta_soft=0.1
            ),
        )
    """

    def __init__(
        self,
        *args: Any,
        indexmem_soft_hit_config: Optional[IndexMemSoftHitMixinConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            indexmem_soft_hit_config: IndexMemSoftHitMixinConfig. If None, uses defaults.
            All other args/kwargs forwarded to the base KVCacheManager.__init__().
        """
        super().__init__(*args, **kwargs)

        if indexmem_soft_hit_config is None:
            indexmem_soft_hit_config = IndexMemSoftHitMixinConfig()
        self._im_cfg = indexmem_soft_hit_config

        # Try to import native IndexMemSoftHitSegmentCache from src/
        IndexMemSoftHitSegmentCache, SoftHitSegmentConfig, HitResult = (
            _try_import_soft_hit_cache()
        )

        self._im_use_native: bool = False
        self._im_cache: Any = None

        if IndexMemSoftHitSegmentCache is not None and _TORCH_AVAILABLE:
            try:
                src_cfg = SoftHitSegmentConfig(
                    chunk_size=indexmem_soft_hit_config.chunk_size,
                    max_physical_entries=indexmem_soft_hit_config.max_physical_entries,
                    latent_pool_max_segments=indexmem_soft_hit_config.latent_pool_max_segments,
                    beta_soft=indexmem_soft_hit_config.beta_soft,
                    beta_weight=indexmem_soft_hit_config.beta_weight,
                    latent_dim=indexmem_soft_hit_config.latent_dim,
                    kv_dim=indexmem_soft_hit_config.kv_dim,
                    n_layers=indexmem_soft_hit_config.n_layers,
                    seed=indexmem_soft_hit_config.seed,
                )
                self._im_cache = IndexMemSoftHitSegmentCache(src_cfg)
                self._im_use_native = True
            except Exception:
                self._im_cache = _InlineSoftHitStore(indexmem_soft_hit_config)
        else:
            self._im_cache = _InlineSoftHitStore(indexmem_soft_hit_config)

        # SegmentLatentPool (exposed separately for direct access)
        self._im_latent_pool = SegmentLatentPool(
            max_segments=indexmem_soft_hit_config.latent_pool_max_segments,
            latent_dim=indexmem_soft_hit_config.latent_dim,
        )

        # Soft-hit block allocation table: {segment_id: block_idx}
        self._im_soft_hit_blocks: Dict[str, int] = {}
        self._im_next_block_idx: int = 0

        # Activity B metrics
        self._im_store_count: int = 0
        self._im_n_hard: int = 0
        self._im_n_soft: int = 0
        self._im_n_miss: int = 0
        self._im_n_nc_hard: int = 0
        self._im_n_nc_soft: int = 0

    # -----------------------------------------------------------------------
    # Segment key computation
    # -----------------------------------------------------------------------

    def _im_chunk_key(
        self,
        session_id: str,
        turn_id: int,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int,
    ) -> str:
        """Compute a SHA-256 segment key for a chunk of tokens."""
        chunk_size = self._im_cfg.chunk_size
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = token_ids[start:end]
        raw = struct.pack(f"{len(chunk)}I", *chunk) if chunk else b""
        header = f"{session_id}|{turn_id}|l{layer_idx}".encode()
        return hashlib.sha256(header + raw).hexdigest()

    # -----------------------------------------------------------------------
    # Core segment store API
    # -----------------------------------------------------------------------

    def store_soft_hit_segment(
        self,
        session_id: str,
        turn_id: int,
        token_ids: List[int],
        chunk_idx: int,
        kv_tensor: Any,
        layer_idx: int = 0,
    ) -> str:
        """Store physical KV segment with latent encoding fallback.

        Args:
            session_id: Session identifier.
            turn_id: Conversation turn index.
            token_ids: Token ID sequence for content hashing.
            chunk_idx: Chunk index within the token sequence.
            kv_tensor: KV tensor [chunk_len, d_head] or [chunk_len, n_heads, d_head].
            layer_idx: Transformer layer index.

        Returns:
            Segment key string (SHA-256 based).
        """
        self._im_store_count += 1
        key = self._im_chunk_key(session_id, turn_id, token_ids, chunk_idx, layer_idx)

        if self._im_use_native:
            try:
                # Use native put_segment API
                stored_key = self._im_cache.put_segment(
                    token_ids=token_ids,
                    chunk_idx=chunk_idx,
                    kv=kv_tensor,
                    layer_idx=layer_idx,
                )
                return stored_key
            except Exception:
                pass

        # Fallback: inline store
        if isinstance(self._im_cache, _InlineSoftHitStore):
            self._im_cache.put(key, kv_tensor)
        return key

    def get_soft_hit_result(
        self,
        session_id: str,
        turn_id: int,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> SoftHitResult:
        """Look up a segment; return SoftHitResult with type hard/soft/miss.

        Args:
            session_id: Session identifier.
            turn_id: Conversation turn index.
            token_ids: Token ID sequence for content hashing.
            chunk_idx: Chunk index.
            layer_idx: Transformer layer index.

        Returns:
            SoftHitResult with type "hard", "soft", or "miss".
        """
        key = self._im_chunk_key(session_id, turn_id, token_ids, chunk_idx, layer_idx)

        if self._im_use_native:
            try:
                # Use native get_hit_result API
                native_result = self._im_cache.get_hit_result(key)
                result = SoftHitResult(
                    type=native_result.type,
                    kv_tensor=native_result.kv_tensor,
                    latent_state=native_result.latent_state,
                    segment_key=key,
                )
                self._im_update_hit_counts(result.type)
                return result
            except Exception:
                pass

        # Fallback: inline store
        if isinstance(self._im_cache, _InlineSoftHitStore):
            result = self._im_cache.get_hit_result(key)
            self._im_update_hit_counts(result.type)
            return result

        self._im_n_miss += 1
        return SoftHitResult(type="miss", segment_key=key)

    def _im_update_hit_counts(self, hit_type: str) -> None:
        """Update aggregate hit/miss counters."""
        if hit_type == "hard":
            self._im_n_hard += 1
        elif hit_type == "soft":
            self._im_n_soft += 1
        else:
            self._im_n_miss += 1

    def get_segments_with_soft_hits(
        self,
        session_id: str,
        turn_id: int,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, SoftHitResult]], List[int]]:
        """Look up all chunks for a token sequence; return hits + misses.

        Non-contiguous hit detection: a hit is "non-contiguous" if any
        earlier chunk in the sequence was a miss.

        Returns:
            hits: List of (chunk_idx, SoftHitResult) — hard and soft hits.
            miss_indices: List of chunk indices that were complete misses.
        """
        chunk_size = self._im_cfg.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)

        if self._im_use_native:
            try:
                # Use native get_segments API
                native_hits, miss_indices = self._im_cache.get_segments(
                    token_ids=token_ids, layer_idx=layer_idx
                )
                # Convert native HitResult to SoftHitResult
                hits = []
                for chunk_idx, native_result in native_hits:
                    soft_result = SoftHitResult(
                        type=native_result.type,
                        kv_tensor=native_result.kv_tensor,
                        latent_state=native_result.latent_state,
                        segment_key=native_result.segment_key,
                    )
                    hits.append((chunk_idx, soft_result))
                    self._im_update_hit_counts(soft_result.type)
                self._im_n_miss += len(miss_indices)
                # Non-contiguous tracking
                miss_set = set(miss_indices)
                for chunk_idx, result in hits:
                    if any(m < chunk_idx for m in miss_set):
                        if result.type == "hard":
                            self._im_n_nc_hard += 1
                        else:
                            self._im_n_nc_soft += 1
                return hits, miss_indices
            except Exception:
                pass

        # Fallback: inline chunk-by-chunk lookup
        chunk_results = []
        for i in range(n_chunks):
            result = self.get_soft_hit_result(
                session_id, turn_id, token_ids, i, layer_idx
            )
            chunk_results.append((i, result))

        miss_set = {i for i, r in chunk_results if r.type == "miss"}
        hits = []
        miss_indices = []
        for i, result in chunk_results:
            if result.type != "miss":
                hits.append((i, result))
                if any(m < i for m in miss_set):
                    if result.type == "hard":
                        self._im_n_nc_hard += 1
                    else:
                        self._im_n_nc_soft += 1
            else:
                miss_indices.append(i)

        return hits, miss_indices

    # -----------------------------------------------------------------------
    # Soft-hit block allocation
    # -----------------------------------------------------------------------

    def allocate_soft_hit_block(self, segment_id: str) -> Optional[int]:
        """Reserve a logical block slot for a soft-hit segment.

        Provides a block index that can be used to reference the soft-hit
        segment in block tables without allocating GPU VRAM (the actual KV
        is reconstructed via latent readout at attention time).

        Args:
            segment_id: Segment key string (from store_soft_hit_segment).

        Returns:
            block_idx (int): Logical block index. Starts at 0x8000_0000 to
                avoid collision with real block IDs in vLLM's block pool.
            None if segment_id not found in latent pool.
        """
        # Check if segment has a latent state (soft-hit available)
        has_soft_hit = False
        if self._im_use_native:
            try:
                # Check native latent pool
                result = self._im_cache.get_hit_result(segment_id)
                has_soft_hit = (result.type in ("hard", "soft"))
            except Exception:
                pass
        elif isinstance(self._im_cache, _InlineSoftHitStore):
            result = self._im_cache.get_hit_result(segment_id)
            has_soft_hit = (result.type in ("hard", "soft"))

        if not has_soft_hit:
            return None

        # Allocate a logical block index in soft-hit namespace
        if segment_id not in self._im_soft_hit_blocks:
            # Use high-bit range to avoid collision with real blocks
            block_idx = 0x80000000 | (self._im_next_block_idx & 0x7FFFFFFF)
            self._im_next_block_idx += 1
            self._im_soft_hit_blocks[segment_id] = block_idx

        return self._im_soft_hit_blocks[segment_id]

    # -----------------------------------------------------------------------
    # Soft-hit residual computation
    # -----------------------------------------------------------------------

    def soft_hit_residual(
        self,
        query_states: Any,
        latent_state: Any,
    ) -> Optional[Any]:
        """Compute beta-scaled residual from latent state for soft-hit reconstruction.

        Args:
            query_states: Query tensor [n_q, d_head].
            latent_state: Latent state vector from SoftHitResult.latent_state.

        Returns:
            Residual tensor (same shape as query_states) or None on failure.
        """
        if self._im_use_native and hasattr(self._im_cache, 'soft_hit_residual'):
            try:
                return self._im_cache.soft_hit_residual(query_states, latent_state)
            except Exception:
                pass

        if isinstance(self._im_cache, _InlineSoftHitStore):
            return self._im_cache.soft_hit_residual(query_states, latent_state)

        return None

    # -----------------------------------------------------------------------
    # Activity B metrics
    # -----------------------------------------------------------------------

    def weighted_hit_rate(self) -> float:
        """Weighted hit rate: (n_hard + beta_weight * n_soft) / total."""
        if self._im_use_native and hasattr(self._im_cache, 'weighted_hit_rate'):
            try:
                return self._im_cache.weighted_hit_rate()
            except Exception:
                pass
        total = self._im_n_hard + self._im_n_soft + self._im_n_miss
        if total == 0:
            return 0.0
        return (self._im_n_hard + self._im_cfg.beta_weight * self._im_n_soft) / total

    def soft_hit_rate(self) -> float:
        """Soft hit rate: n_soft / total."""
        if self._im_use_native and hasattr(self._im_cache, 'soft_hit_rate'):
            try:
                return self._im_cache.soft_hit_rate()
            except Exception:
                pass
        total = self._im_n_hard + self._im_n_soft + self._im_n_miss
        return self._im_n_soft / total if total > 0 else 0.0

    def noncontiguous_fraction(self) -> float:
        """Fraction of hits that are non-contiguous (preceded by a miss)."""
        if self._im_use_native and hasattr(self._im_cache, 'noncontiguous_hit_rate'):
            try:
                return self._im_cache.noncontiguous_hit_rate()
            except Exception:
                pass
        total_hits = self._im_n_hard + self._im_n_soft
        return (self._im_n_nc_hard + self._im_n_nc_soft) / total_hits if total_hits > 0 else 0.0

    def indexmem_soft_hit_metrics(self) -> Dict[str, Any]:
        """Return Activity B metrics for observability."""
        base: Dict[str, Any] = {
            "store_count": self._im_store_count,
            "n_hard": self._im_n_hard,
            "n_soft": self._im_n_soft,
            "n_miss": self._im_n_miss,
            "n_nc_hard": self._im_n_nc_hard,
            "n_nc_soft": self._im_n_nc_soft,
            "weighted_hit_rate": self.weighted_hit_rate(),
            "soft_hit_rate": self.soft_hit_rate(),
            "noncontiguous_fraction": self.noncontiguous_fraction(),
            "soft_hit_blocks_allocated": len(self._im_soft_hit_blocks),
            "latent_pool_size": self._im_latent_pool.size(),
            "use_native_cache": self._im_use_native,
        }
        if isinstance(self._im_cache, _InlineSoftHitStore):
            base.update(self._im_cache.stats())
        elif self._im_use_native and hasattr(self._im_cache, 'memory_bytes'):
            try:
                base["physical_memory_bytes"] = self._im_cache.memory_bytes()
            except Exception:
                pass
        return base

    def annotate_request_with_soft_hits(
        self,
        request: Any,
        session_id: str,
        turn_id: int,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> None:
        """Attach soft-hit segment metadata to a vLLM Request object.

        Queries the segment cache and attaches soft-hit results as runtime
        attributes on the Request object. The scheduler and model runner can
        read these attributes to route soft-hit segments to the latent
        readout path instead of full recomputation.

        Args:
            request: vLLM Request object (vllm.v1.request.Request).
            session_id: Session associated with this request.
            turn_id: Conversation turn index.
            token_ids: Token IDs for segment lookup.
            layer_idx: Transformer layer index.
        """
        hits, miss_indices = self.get_segments_with_soft_hits(
            session_id, turn_id, token_ids, layer_idx
        )
        soft_hits = [(i, r) for i, r in hits if r.type == "soft"]
        hard_hits = [(i, r) for i, r in hits if r.type == "hard"]

        try:
            object.__setattr__(request, "im_session_id", session_id)
            object.__setattr__(request, "im_hard_hits", hard_hits)
            object.__setattr__(request, "im_soft_hits", soft_hits)
            object.__setattr__(request, "im_miss_indices", miss_indices)
            object.__setattr__(request, "im_has_soft_hit", len(soft_hits) > 0)
            object.__setattr__(request, "im_weighted_hit_rate", self.weighted_hit_rate())
        except Exception:
            pass  # Graceful: some Request implementations may reject setattr


# ---------------------------------------------------------------------------
# make_indexmem_soft_hit_kv_cache_manager_class() factory
# ---------------------------------------------------------------------------

def make_indexmem_soft_hit_kv_cache_manager_class(
    base_class: type = KVCacheManager,
    config: Optional[IndexMemSoftHitMixinConfig] = None,
) -> type:
    """Factory: return a KVCacheManager subclass with IndexMem soft-hit support.

    Args:
        base_class: The vLLM KVCacheManager class to subclass.
        config: Optional IndexMemSoftHitMixinConfig to embed in the subclass.

    Returns:
        A new class combining IndexMemSoftHitKVCacheManagerMixin with base_class.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.indexmem_block_manager_patch import (
            IndexMemSoftHitMixinConfig,
            make_indexmem_soft_hit_kv_cache_manager_class,
        )

        Mgr = make_indexmem_soft_hit_kv_cache_manager_class(
            KVCacheManager,
            IndexMemSoftHitMixinConfig(chunk_size=128, beta_weight=0.5),
        )
        mgr = Mgr(
            kv_cache_config=...,
            max_model_len=4096,
            hash_block_size=16,
        )
        # mgr.store_soft_hit_segment(...)
        # mgr.get_soft_hit_result(...)
    """
    _cfg = config

    class _IndexMemSoftHitKVCacheManager(
        IndexMemSoftHitKVCacheManagerMixin,
        base_class,  # type: ignore[valid-type]
    ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if _cfg is not None and "indexmem_soft_hit_config" not in kwargs:
                kwargs["indexmem_soft_hit_config"] = _cfg
            super().__init__(*args, **kwargs)

    _IndexMemSoftHitKVCacheManager.__name__ = "IndexMemSoftHitKVCacheManager"
    _IndexMemSoftHitKVCacheManager.__qualname__ = "IndexMemSoftHitKVCacheManager"
    _IndexMemSoftHitKVCacheManager.__doc__ = (
        "KVCacheManager subclass with IndexMem soft-hit non-contiguous reuse "
        "(Activity B). Auto-generated by make_indexmem_soft_hit_kv_cache_manager_class()."
    )
    return _IndexMemSoftHitKVCacheManager


# ---------------------------------------------------------------------------
# apply_indexmem_block_manager_patch() — idempotent
# ---------------------------------------------------------------------------

_INDEXMEM_BM_PATCH_APPLIED: bool = False


def apply_indexmem_block_manager_patch(
    config: Optional[IndexMemSoftHitMixinConfig] = None,
) -> bool:
    """Attempt to monkey-patch vLLM's KVCacheManager with IndexMem soft-hit support.

    IDEMPOTENT — calling it multiple times is safe.

    In production, prefer make_indexmem_soft_hit_kv_cache_manager_class() for
    a clean subclass approach. This function is provided for environments where
    direct monkey-patching of the vLLM module is preferred.

    Returns:
        True if patch was applied or already applied; False on failure.
    """
    global _INDEXMEM_BM_PATCH_APPLIED
    if _INDEXMEM_BM_PATCH_APPLIED:
        return True

    try:
        import vllm.v1.core.kv_cache_manager as _km_mod
        _OrigKVM = _km_mod.KVCacheManager

        PatchedClass = make_indexmem_soft_hit_kv_cache_manager_class(
            _OrigKVM, config
        )
        _km_mod.KVCacheManager = PatchedClass
        _INDEXMEM_BM_PATCH_APPLIED = True
        return True
    except Exception:
        _INDEXMEM_BM_PATCH_APPLIED = True  # mark to avoid retries
        return False
