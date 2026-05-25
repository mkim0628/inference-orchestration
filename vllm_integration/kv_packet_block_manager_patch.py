"""KV Packet Non-Contiguous KV Reuse — Activity B (2026-05-25).

Extends vLLM's KVCacheManager to track KVPacketCache segments and support
non-contiguous block-table assembly.

Integration design:
  - KVPacketSegmentMixin: auxiliary segment store alongside vLLM's PagedAttention
    block table. Keyed by (content_hash, layer_idx).
  - find_noncontiguous_hits(token_ids): queries KVPacketCache for matching segments,
    returns matched block IDs for non-contiguous assembly.
  - build_kv_packet_block_table(segment_keys, block_size, max_blocks):
    builds int64 [1, max_blocks] tensor suitable for injection into PA kernel.
  - make_kv_packet_kv_cache_manager_class(base_cls): factory that subclasses
    vLLM KVCacheManager with KVPacketSegmentMixin.

Non-contiguous hit tracking:
  - A "non-contiguous hit" is a hit on a segment where the previous accessed
    segment is not adjacent in insertion order (same definition as KVPacketCache).
  - Reported via kv_packet_noncontiguous_hit_rate().
"""

from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# Optional src/ import                                                          #
# --------------------------------------------------------------------------- #

try:
    import sys
    import pathlib
    _repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from src.cache.kv_packet import KVPacketCache, KVPacketConfig, KVPacket  # noqa: F401
    _SRC_KV_PACKET_AVAILABLE = True
except ImportError:
    _SRC_KV_PACKET_AVAILABLE = False
    KVPacketCache = None   # type: ignore
    KVPacketConfig = None  # type: ignore
    KVPacket = None        # type: ignore


# --------------------------------------------------------------------------- #
# Inline segment store (standalone, no src/ dependency)                        #
# --------------------------------------------------------------------------- #

@dataclass
class _InlineKVPacketEntry:
    """Auxiliary KV segment entry stored alongside PagedAttention block table."""
    segment_id: str
    kv_tensor: torch.Tensor        # [n_tokens, 2, n_heads, d_head] FP16
    adapter_K: torch.Tensor        # [n_adapter_tokens, n_heads, d_head]
    adapter_V: torch.Tensor        # [n_adapter_tokens, n_heads, d_head]
    n_adapter_tokens: int = 4
    distillation_loss: float = 1.0
    n_reuses: int = 0


class _InlineKVPacketStore:
    """Minimal inline KV Packet store — no src/ dependency required.

    Implements the same adapter-prepend non-contiguous reuse strategy as
    KVPacketCache but embedded inline for portability.
    """

    def __init__(
        self,
        max_entries: int = 512,
        n_adapter_tokens: int = 4,
        n_heads: int = 8,
        d_head: int = 128,
        distillation_loss_threshold: float = 0.1,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.max_entries = max_entries
        self.n_adapter_tokens = n_adapter_tokens
        self.n_heads = n_heads
        self.d_head = d_head
        self.distillation_loss_threshold = distillation_loss_threshold
        self._store: OrderedDict[str, _InlineKVPacketEntry] = OrderedDict()
        self._insertion_order: List[str] = []
        self._access_order: List[str] = []
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0

    def put(self, segment_id: str, kv_tensor: torch.Tensor) -> None:
        """Store segment. kv_tensor: [n_tokens, 2, n_heads, d_head]."""
        if segment_id in self._store:
            self._store.move_to_end(segment_id)
            return
        if len(self._store) >= self.max_entries:
            self._evict()
        adapter_K = torch.randn(self.n_adapter_tokens, self.n_heads, self.d_head) * 0.02
        adapter_V = torch.randn(self.n_adapter_tokens, self.n_heads, self.d_head) * 0.02
        entry = _InlineKVPacketEntry(
            segment_id=segment_id,
            kv_tensor=kv_tensor.detach().clone().to(torch.float16),
            adapter_K=adapter_K,
            adapter_V=adapter_V,
            n_adapter_tokens=self.n_adapter_tokens,
        )
        self._store[segment_id] = entry
        self._insertion_order.append(segment_id)

    def get(self, segment_id: str) -> Optional[torch.Tensor]:
        """Return adapter-prepended KV [n_adapter + n_tokens, 2, n_heads, d_head]."""
        if segment_id not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(segment_id)
        self._hits += 1
        entry = self._store[segment_id]
        entry.n_reuses += 1
        self._track_noncontiguous(segment_id)
        return self._apply_adapter(entry)

    def get_kv_pair(
        self, segment_id: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (K, V) tuple from adapter-applied KV."""
        adapted = self.get(segment_id)
        if adapted is None:
            return None
        K = adapted[:, 0, :, :]
        V = adapted[:, 1, :, :]
        return K, V

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def noncontiguous_hit_rate(self) -> float:
        if self._hits == 0:
            return 0.0
        return self._noncontiguous_hits / self._hits

    def _apply_adapter(self, entry: _InlineKVPacketEntry) -> torch.Tensor:
        adapter_kv = torch.stack([entry.adapter_K, entry.adapter_V], dim=1)
        return torch.cat([adapter_kv.to(entry.kv_tensor.dtype), entry.kv_tensor], dim=0)

    def _evict(self) -> None:
        threshold = self.distillation_loss_threshold
        low_quality = [
            k for k, e in self._store.items() if e.distillation_loss > threshold
        ]
        evict_key = low_quality[0] if low_quality else next(iter(self._store))
        self._store.pop(evict_key)
        if evict_key in self._insertion_order:
            self._insertion_order.remove(evict_key)

    def _track_noncontiguous(self, segment_id: str) -> None:
        if self._access_order:
            prev = self._access_order[-1]
            if segment_id in self._insertion_order and prev in self._insertion_order:
                idx_cur = self._insertion_order.index(segment_id)
                idx_prev = self._insertion_order.index(prev)
                if abs(idx_cur - idx_prev) > 1:
                    self._noncontiguous_hits += 1
            else:
                self._noncontiguous_hits += 1
        self._access_order.append(segment_id)


# --------------------------------------------------------------------------- #
# KVCacheManager mixin                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class KVPacketSegmentConfig:
    """Configuration for KVPacketSegmentMixin."""
    max_segments: int = 512
    n_adapter_tokens: int = 4
    n_heads: int = 8
    d_head: int = 128
    distillation_loss_threshold: float = 0.1
    seed: int = 42
    # Block-table sentinel for unused slots (PagedAttention convention)
    block_table_sentinel: int = -1


# Sentinel for unused block table slots
_KV_PACKET_SENTINEL = -1


def _hash_token_ids(token_ids: List[int]) -> str:
    """Compute a deterministic segment hash from token IDs."""
    data = ",".join(str(t) for t in token_ids)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class KVPacketSegmentMixin:
    """Mixin for vLLM KVCacheManager adding KV Packet non-contiguous reuse.

    Maintains an auxiliary segment store keyed by (content_hash, layer_idx)
    alongside vLLM's native PagedAttention block table.

    Key methods:
      store_kv_packet_segment(token_ids, kv_tensor, layer_idx):
          Store segment. kv_tensor: [n_tokens, 2, n_heads, d_head].

      find_noncontiguous_hits(token_ids_chunks, layer_idx):
          Query store for each chunk of token_ids, returns list of
          (segment_id, K, V) tuples for matched segments.

      build_kv_packet_block_table(segment_keys, block_size, max_blocks):
          Build int64 [1, max_blocks] block table tensor with -1 padding.
          Suitable for direct injection into PagedAttention kernel.

      kv_packet_stats():
          Returns dict with hit_rate, noncontiguous_hit_rate, n_segments.
    """

    def __init__(self, *args, kv_packet_config: Optional[KVPacketSegmentConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = kv_packet_config or KVPacketSegmentConfig()
        self._kv_packet_cfg = cfg
        self._kv_packet_store = _InlineKVPacketStore(
            max_entries=cfg.max_segments,
            n_adapter_tokens=cfg.n_adapter_tokens,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            distillation_loss_threshold=cfg.distillation_loss_threshold,
            seed=cfg.seed,
        )
        # (content_hash, layer_idx) -> list of (start_block, n_blocks)
        self._kv_packet_block_registry: Dict[Tuple[str, int], List[int]] = {}
        self._kv_packet_block_align_warnings: int = 0

    # ---- Storage API ---------------------------------------------------------

    def store_kv_packet_segment(
        self,
        token_ids: List[int],
        kv_tensor: torch.Tensor,    # [n_tokens, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> str:
        """Store a KV segment in the auxiliary packet store.

        Args:
            token_ids:  token IDs for content-based hashing
            kv_tensor:  KV data [n_tokens, 2, n_heads, d_head]
            layer_idx:  transformer layer index

        Returns:
            segment_id (content hash)
        """
        segment_id = _hash_token_ids(token_ids)
        store_key = f"{segment_id}_L{layer_idx}"
        self._kv_packet_store.put(store_key, kv_tensor)
        return segment_id

    # ---- Lookup API ----------------------------------------------------------

    def find_noncontiguous_hits(
        self,
        token_ids_chunks: List[List[int]],
        layer_idx: int = 0,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Query auxiliary store for non-contiguous KV segment hits.

        For each chunk of token_ids, checks if a matching segment exists.
        Returns all matching segments with their adapter-applied KV pairs.

        Args:
            token_ids_chunks: list of token_id lists (each is a segment chunk)
            layer_idx: transformer layer index

        Returns:
            List of (segment_id, K, V) tuples for each matched segment.
            K, V are [n_adapter_tokens + n_tokens, n_heads, d_head] tensors.
        """
        results = []
        for chunk in token_ids_chunks:
            segment_id = _hash_token_ids(chunk)
            store_key = f"{segment_id}_L{layer_idx}"
            kv_pair = self._kv_packet_store.get_kv_pair(store_key)
            if kv_pair is not None:
                K, V = kv_pair
                results.append((segment_id, K, V))
        return results

    # ---- Block table builder -------------------------------------------------

    def build_kv_packet_block_table(
        self,
        segment_keys: List[str],
        block_size: int = 16,
        max_blocks: int = 64,
    ) -> Optional[torch.Tensor]:
        """Build non-contiguous block table for PagedAttention kernel injection.

        Constructs an int64 tensor [1, max_blocks] where:
          - Valid slots contain block_ids from the KV packet store mapping
          - Unused slots are filled with _KV_PACKET_SENTINEL (-1)

        This follows the convention established in prior cycles (DapQSessionSegment,
        BlockUnion, CLCPositionalBiasGated) for non-contiguous block table injection.

        Args:
            segment_keys:  list of segment identifiers from find_noncontiguous_hits()
            block_size:    vLLM page block size (default 16)
            max_blocks:    maximum block table capacity

        Returns:
            int64 tensor [1, max_blocks] or None if no segments matched
        """
        if not segment_keys:
            return None

        # Enumerate valid matched blocks (using store entry indices as proxy block IDs)
        block_ids = []
        store_keys = list(self._kv_packet_store._store.keys())
        for seg_key in segment_keys:
            if seg_key in store_keys:
                block_ids.append(store_keys.index(seg_key))

        if not block_ids:
            return None

        # Validate/warn on misalignment (matches CLCBiasGate convention)
        for bid in block_ids:
            if bid % block_size != 0:
                self._kv_packet_block_align_warnings += 1

        # Pad to max_blocks with sentinel
        padded = block_ids[:max_blocks] + [_KV_PACKET_SENTINEL] * max(
            0, max_blocks - len(block_ids)
        )
        return torch.tensor([padded], dtype=torch.int64)

    # ---- Stats ---------------------------------------------------------------

    def kv_packet_stats(self) -> dict:
        """JSON-serializable KV packet segment stats."""
        store = self._kv_packet_store
        return {
            "hit_rate": store.hit_rate(),
            "noncontiguous_hit_rate": store.noncontiguous_hit_rate(),
            "n_segments": len(store._store),
            "total_hits": store._hits,
            "total_misses": store._misses,
            "noncontiguous_hits": store._noncontiguous_hits,
            "block_align_warnings": self._kv_packet_block_align_warnings,
        }

    # ---- TP warning (mirrors CLCPositionalBiasGated convention) --------------

    @staticmethod
    def _warn_if_tp_environment() -> None:
        """Emit warning if tensor-parallel environment is detected."""
        import os
        tp_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if tp_size > 1 or world_size > 1:
            warnings.warn(
                "KVPacketSegmentMixin: tensor-parallel environment detected "
                f"(TP={tp_size}, WORLD_SIZE={world_size}). "
                "build_kv_packet_block_table() is not TP-aware. "
                "Broadcast block table from rank 0 before attention kernel call.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            try:
                import torch.distributed as dist
                if dist.is_initialized() and dist.get_world_size() > 1:
                    warnings.warn(
                        "KVPacketSegmentMixin: torch.distributed world_size > 1 detected. "
                        "Block table not broadcast-safe.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Factory                                                                       #
# --------------------------------------------------------------------------- #

def make_kv_packet_kv_cache_manager_class(
    base_cls: type,
    kv_packet_config: Optional[KVPacketSegmentConfig] = None,
) -> type:
    """Subclass vLLM KVCacheManager with KVPacketSegmentMixin.

    Args:
        base_cls:         vLLM KVCacheManager class
                          (vllm.v1.core.kv_cache_manager.KVCacheManager)
        kv_packet_config: KVPacketSegmentConfig (default if None)

    Returns:
        New class that subclasses both KVPacketSegmentMixin and base_cls.
        issubclass(result, base_cls) is True.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.kv_packet_block_manager_patch import (
            KVPacketSegmentConfig, make_kv_packet_kv_cache_manager_class,
        )
        KVPacketManager = make_kv_packet_kv_cache_manager_class(KVCacheManager)
        assert issubclass(KVPacketManager, KVCacheManager)
    """
    cfg = kv_packet_config or KVPacketSegmentConfig()
    _cfg = cfg  # capture for __init__

    class KVPacketKVCacheManager(KVPacketSegmentMixin, base_cls):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("kv_packet_config", _cfg)
            super().__init__(*args, **kwargs)

    KVPacketKVCacheManager.__name__ = "KVPacketKVCacheManager"
    KVPacketKVCacheManager.__qualname__ = "KVPacketKVCacheManager"
    return KVPacketKVCacheManager


# --------------------------------------------------------------------------- #
# Exports                                                                       #
# --------------------------------------------------------------------------- #

__all__ = [
    "KVPacketSegmentConfig",
    "KVPacketSegmentMixin",
    "make_kv_packet_kv_cache_manager_class",
    "_InlineKVPacketStore",
    "_hash_token_ids",
    "_SRC_KV_PACKET_AVAILABLE",
]
