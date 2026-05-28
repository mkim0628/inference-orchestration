"""Activity B-1: PegaFlow RDMA + Irminsul δ-rotation distributed non-contiguous KV cache.

Extends Irminsul MLA segment cache (05-26) with PegaFlow RDMA to reach remote nodes.
4-level lookup: local HBM → PegaFlow local → PegaFlow RDMA remote → miss.
"""

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.irminsul_mla_segment_cache import (
    IrminsulMLAConfig,
    IrminsulMLASegmentCache,
    apply_delta_rotation,
    cdc_chunk,
    cdc_segment_key,
)
from src.cache.pegaflow_kv_connector import MockPegaFlowConnector, PegaFlowConnectorConfig
from src.metrics.hit_rate import DistributedHitRateMetrics
from src.scheduler.pegaflow_rdma_router import (
    PegaFlowRDMACrossNodeRouter,
    PegaFlowRDMARouterConfig,
    PeerRegistry,
)

if TYPE_CHECKING:
    pass


@dataclass
class IrminsulKVEntry:
    """Irminsul MLA segment entry compatible with distributed cache (B-1).

    c_kv_tensor is position-free; k_r_tensor requires δ-rotation on reuse.
    """

    segment_id: bytes
    c_kv_tensor: torch.Tensor    # [n_tokens, n_heads, d_kv] — position-free
    k_r_tensor: torch.Tensor     # [n_tokens, n_heads, d_r]  — position-dependent
    source_position: int          # original sequence position (for δ calculation)


@dataclass
class DistributedSegmentCacheConfig:
    rdma_reuse_discount: float = 0.8
    bloom_sync_interval_ms: float = 500.0
    local_max_entries: int = 5000
    avg_chunk_size: int = 256
    seed: int = 42


def _segment_id_from_tokens(token_ids: List[int], chunk_idx: int) -> bytes:
    """Deterministic segment ID from token content and chunk index."""
    raw = struct.pack(f"{len(token_ids)}I", *token_ids)
    return hashlib.sha256(raw + chunk_idx.to_bytes(4, "little")).digest()


class PegaFlowIrminsulDistributedSegmentCache(CacheStore):
    """Distributed non-contiguous KV segment cache combining Irminsul + PegaFlow RDMA.

    4-level lookup order:
      1. Local HBM (IrminsulMLASegmentCache)
      2. PegaFlow local host/SSD (PegaFlowKVConnector.get)
      3. PegaFlow RDMA remote node (PegaFlowRDMACrossNodeRouter.route_segment_request)
      4. Miss → recompute

    On hit: δ-rotation applied to k_r; c_KV reused as-is.
    Eviction: local HBM LRU evict → async offload to PegaFlow local.
    """

    def __init__(
        self,
        local_irminsul_cache: IrminsulMLASegmentCache,
        pegaflow_local: CacheStore,
        pegaflow_rdma_router: PegaFlowRDMACrossNodeRouter,
        config: DistributedSegmentCacheConfig,
        local_node_id: str = "local",
    ) -> None:
        self.config = config
        self._local = local_irminsul_cache
        self._pegaflow_local = pegaflow_local
        self._rdma_router = pegaflow_rdma_router
        self._local_node_id = local_node_id
        self._metrics = DistributedHitRateMetrics()
        # Local distributed entry store (segment_id bytes → IrminsulKVEntry), LRU ordered
        self._entry_store: OrderedDict[bytes, IrminsulKVEntry] = OrderedDict()
        torch.manual_seed(config.seed)

    # ------------------------------------------------------------------ #
    # CacheStore abstract methods                                          #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store tensor under generic key in both local Irminsul and PegaFlow local."""
        self._local.put(key, value)
        self._pegaflow_local.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """4-level lookup via get_distributed (returns combined tensor on hit)."""
        seg_id = key.encode("utf-8")
        tensor, _ = self.get_distributed(seg_id, target_position=0)
        return tensor

    def evict(self) -> int:
        """Evict LRU entry from local distributed store; offload to PegaFlow."""
        if self._entry_store:
            seg_id, entry = self._entry_store.popitem(last=False)
            combined = torch.cat([entry.c_kv_tensor, entry.k_r_tensor], dim=-1)
            # Async offload to PegaFlow local (fire-and-forget via MockPegaFlowConnector)
            self._pegaflow_local.put(seg_id.hex(), combined)
            return combined.nelement() * combined.element_size()
        return self._local.evict()

    def hit_rate(self) -> float:
        """Overall hit rate (local + pegaflow_local + rdma_remote) / total."""
        return self._metrics.distributed_hit_rate()

    def memory_bytes(self) -> int:
        """Local HBM KV memory footprint in bytes."""
        dist_bytes = sum(
            e.c_kv_tensor.nelement() * e.c_kv_tensor.element_size()
            + e.k_r_tensor.nelement() * e.k_r_tensor.element_size()
            for e in self._entry_store.values()
        )
        return dist_bytes + self._local.memory_bytes()

    def reset_stats(self) -> None:
        self._metrics.reset()
        self._local.reset_stats()

    # ------------------------------------------------------------------ #
    # Distributed lookup core API                                          #
    # ------------------------------------------------------------------ #

    def get_distributed(
        self,
        segment_id: bytes,
        target_position: int,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """4-level distributed lookup + δ-rotation.

        Returns:
          (kv_tensor, hit_type): hit_type ∈ {"local_hard_hit", "pegaflow_local_hit",
                                              "rdma_remote_hit", "miss"}
        """
        # Level 1: local HBM distributed entry store
        if segment_id in self._entry_store:
            self._entry_store.move_to_end(segment_id)
            entry = self._entry_store[segment_id]
            kv = self.apply_delta_rotation(entry, target_position)
            self._metrics.record("local_hard_hit")
            return kv, "local_hard_hit"

        # Level 1b: Irminsul generic tensor store (via hex segment key)
        seg_hex = segment_id.hex()
        local_tensor = self._local.get(seg_hex)
        if local_tensor is not None:
            self._metrics.record("local_hard_hit")
            return local_tensor, "local_hard_hit"

        # Level 2: PegaFlow local host/SSD
        raw = self._pegaflow_local.get(seg_hex)
        if raw is not None:
            self._metrics.record("pegaflow_local_hit")
            return raw, "pegaflow_local_hit"

        # Level 3: PegaFlow RDMA remote
        rdma_result, hit_type_rdma = self._rdma_router.route_segment_request(
            segment_id, self._local_node_id
        )
        if rdma_result is not None and hit_type_rdma == "rdma_remote":
            self._metrics.record("rdma_remote_hit")
            return rdma_result, "rdma_remote_hit"

        # Level 4: Miss
        self._metrics.record("miss")
        return None, "miss"

    def apply_delta_rotation(
        self,
        entry: IrminsulKVEntry,
        target_position: int,
    ) -> torch.Tensor:
        """Apply Irminsul δ-rotation to k_r; c_KV is position-free.

        δ = target_position - source_position
        k_r_corrected = rope_rotate(k_r_tensor, δ)
        Returns concat(c_kv_tensor, k_r_corrected, dim=-1)
        """
        delta = target_position - entry.source_position
        c_kv = entry.c_kv_tensor

        # Flatten to 2D for apply_delta_rotation: [n_tokens, d_r]
        original_shape = entry.k_r_tensor.shape
        if entry.k_r_tensor.dim() > 2:
            k_r_2d = entry.k_r_tensor.reshape(-1, original_shape[-1])
        else:
            k_r_2d = entry.k_r_tensor

        if delta == 0:
            k_r_corrected = k_r_2d
        else:
            k_r_corrected = apply_delta_rotation(k_r_2d, delta)

        # Flatten c_kv to 2D for concat compatibility
        if c_kv.dim() > 2:
            c_kv_2d = c_kv.reshape(-1, c_kv.shape[-1])
        else:
            c_kv_2d = c_kv

        return torch.cat([c_kv_2d, k_r_corrected], dim=-1)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        source_position: int,
        layer_idx: int = 0,
    ) -> bytes:
        """CDC-chunk-aware segment storage. Returns segment_id bytes."""
        seg_id = _segment_id_from_tokens(token_ids, chunk_idx)

        entry = IrminsulKVEntry(
            segment_id=seg_id,
            c_kv_tensor=c_kv.detach().clone(),
            k_r_tensor=k_r.detach().clone(),
            source_position=source_position,
        )

        # Enforce capacity via LRU eviction
        while len(self._entry_store) >= self.config.local_max_entries:
            evicted_id, evicted_entry = self._entry_store.popitem(last=False)
            combined = torch.cat([evicted_entry.c_kv_tensor, evicted_entry.k_r_tensor], dim=-1)
            self._pegaflow_local.put(evicted_id.hex(), combined)

        self._entry_store[seg_id] = entry
        # Register with RDMA router's peer registry for remote discovery
        self._rdma_router.peer_registry.register_local_segment(seg_id)
        return seg_id

    def distributed_hit_rate_breakdown(self) -> dict:
        """Return per-level hit rate breakdown dict."""
        return self._metrics.summary()

    def get_metrics(self) -> DistributedHitRateMetrics:
        return self._metrics
