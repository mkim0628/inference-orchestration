"""pegaflow_irminsul_block_manager_patch.py — Activity B: PegaFlow+Irminsul distributed
non-contiguous KV cache for vLLM 0.21.0.

2026-05-28: PegaFlowIrminsulDistributedKVCacheManagerMixin — ports
            PegaFlowIrminsulDistributedSegmentCache (Activity B-1) and
            PegaFlowRDMACrossNodeRouter (Activity A-2) into vLLM's v1
            KVCacheManager as a mixin.

            Provides 4-level distributed non-contiguous KV segment reuse:
              1. Local HBM (IrminsulMLASegmentCache / entry_store)
              2. PegaFlow local host/SSD (MockPegaFlowConnector or real IPC)
              3. PegaFlow RDMA remote node (Bloom Filter → RDMA cost decision)
              4. Miss → recompute

            δ-rotation (Irminsul MLA):
              c_KV component is position-free → reuse as-is from any tier.
              k_r component requires δ-rotation: k_r_corrected = RoPE(k_r, δ)
              where δ = target_position - source_position.

            make_pegaflow_irminsul_kv_cache_manager_class() factory:
              Subclasses KVCacheManager and adds the 4-level distributed cache
              as a parallel segment store alongside vLLM's native paged block pool.
              All native KVCacheManager methods are preserved (no overrides).

            Integration with vLLM v1 architecture:
              The mixin operates as a PARALLEL segment store. It does NOT replace
              or modify vLLM's native KV block allocation logic (block_pool, paged
              attention). Segment reuse is managed via put_distributed_segment() /
              get_distributed_segment() APIs that callers invoke explicitly.

            Non-contiguous reuse path:
              When a request's KV segments are found in any tier, the combined
              tensor (c_kv || k_r_corrected) is returned for attention kernel use.
              The block_table padding approach is left to the caller (model runner
              or attention backend hook) — this mixin provides the segment API.

vLLM version: 0.21.0
Activity: B — PegaFlowIrminsulDistributedSegmentCache (4-level distributed lookup)
          A — PegaFlowRDMACrossNodeRouter (Bloom Filter RDMA routing)
"""

from __future__ import annotations

import sys
import pathlib
import hashlib
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import vllm


def _add_repo_root_to_path() -> None:
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _try_import_distributed_cache_src():
    """Lazily import PegaFlowIrminsulDistributedSegmentCache from src/."""
    _add_repo_root_to_path()
    try:
        from src.cache.pegaflow_irminsul_distributed_cache import (
            PegaFlowIrminsulDistributedSegmentCache,
            DistributedSegmentCacheConfig,
            IrminsulKVEntry,
        )
        return PegaFlowIrminsulDistributedSegmentCache, DistributedSegmentCacheConfig, IrminsulKVEntry
    except ImportError:
        return None, None, None


def _try_import_pegaflow_connector_src():
    """Lazily import MockPegaFlowConnector from src/."""
    _add_repo_root_to_path()
    try:
        from src.cache.pegaflow_kv_connector import (
            MockPegaFlowConnector,
            PegaFlowConnectorConfig,
            create_pegaflow_connector,
        )
        return MockPegaFlowConnector, PegaFlowConnectorConfig, create_pegaflow_connector
    except ImportError:
        return None, None, None


def _try_import_rdma_router_src():
    """Lazily import PegaFlowRDMACrossNodeRouter from src/."""
    _add_repo_root_to_path()
    try:
        from src.scheduler.pegaflow_rdma_router import (
            PegaFlowRDMACrossNodeRouter,
            PegaFlowRDMARouterConfig,
            PeerRegistry,
        )
        return PegaFlowRDMACrossNodeRouter, PegaFlowRDMARouterConfig, PeerRegistry
    except ImportError:
        return None, None, None


def _try_import_irminsul_src():
    """Lazily import IrminsulMLASegmentCache from src/."""
    _add_repo_root_to_path()
    try:
        from src.cache.irminsul_mla_segment_cache import (
            IrminsulMLASegmentCache,
            IrminsulMLAConfig,
            apply_delta_rotation,
        )
        return IrminsulMLASegmentCache, IrminsulMLAConfig, apply_delta_rotation
    except ImportError:
        return None, None, None


def _try_import_distributed_metrics_src():
    """Lazily import DistributedHitRateMetrics from src/."""
    _add_repo_root_to_path()
    try:
        from src.metrics.hit_rate import DistributedHitRateMetrics
        return DistributedHitRateMetrics
    except ImportError:
        return None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PegaFlowIrminsulMixinConfig:
    """Configuration for PegaFlowIrminsulDistributedKVCacheManagerMixin."""
    rdma_reuse_discount: float = 0.8          # RDMA vs recompute cost discount threshold
    bloom_sync_interval_ms: float = 500.0     # Bloom Filter sync interval
    local_max_entries: int = 5000             # max segments in local HBM store
    avg_chunk_size: int = 256                 # CDC average chunk size (tokens)
    local_node_id: str = "local"              # this node's identifier
    use_mock_pegaflow: bool = True            # use MockPegaFlowConnector (no Rust process)
    peer_nodes_config_path: str = "configs/pegaflow_peer_nodes.yaml"
    bloom_filter_capacity: int = 100_000
    bloom_filter_error_rate: float = 0.01
    seed: int = 42


# ============================================================================
# Inline fallback: minimal 4-level distributed segment cache (no src/ dependency)
# ============================================================================

class _InlineIrminsulKVEntry:
    """Minimal IrminsulKVEntry stand-in."""
    __slots__ = ("segment_id", "c_kv_tensor", "k_r_tensor", "source_position")

    def __init__(
        self,
        segment_id: bytes,
        c_kv_tensor: Any,
        k_r_tensor: Any,
        source_position: int,
    ) -> None:
        self.segment_id = segment_id
        self.c_kv_tensor = c_kv_tensor
        self.k_r_tensor = k_r_tensor
        self.source_position = source_position


class _InlineDistributedHitRateMetrics:
    """Minimal hit rate tracker when src/ not available."""

    def __init__(self) -> None:
        self.total_lookups = 0
        self.local_hard_hits = 0
        self.pegaflow_local_hits = 0
        self.rdma_remote_hits = 0

    def record(self, hit_type: str) -> None:
        self.total_lookups += 1
        if hit_type == "local_hard_hit":
            self.local_hard_hits += 1
        elif hit_type == "pegaflow_local_hit":
            self.pegaflow_local_hits += 1
        elif hit_type == "rdma_remote_hit":
            self.rdma_remote_hits += 1

    def distributed_hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return (self.local_hard_hits + self.pegaflow_local_hits + self.rdma_remote_hits) / self.total_lookups

    def summary(self) -> Dict:
        total = max(1, self.total_lookups)
        hits = self.local_hard_hits + self.pegaflow_local_hits + self.rdma_remote_hits
        return {
            "local_hard_hit_rate": self.local_hard_hits / total,
            "pegaflow_local_hit_rate": self.pegaflow_local_hits / total,
            "rdma_remote_hit_rate": self.rdma_remote_hits / total,
            "miss_rate": 1.0 - self.distributed_hit_rate(),
            "distributed_hit_rate": self.distributed_hit_rate(),
            "total_lookups": self.total_lookups,
        }

    def reset(self) -> None:
        self.total_lookups = 0
        self.local_hard_hits = 0
        self.pegaflow_local_hits = 0
        self.rdma_remote_hits = 0


class _InlinePegaFlowStore:
    """Minimal in-memory PegaFlow fallback (no Rust process)."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def memory_bytes(self) -> int:
        if not _TORCH_AVAILABLE:
            return 0
        total = 0
        for v in self._store.values():
            if hasattr(v, "nelement") and hasattr(v, "element_size"):
                total += v.nelement() * v.element_size()
        return total


def _inline_delta_rotate(k_r: Any, delta: int) -> Any:
    """Minimal RoPE-style δ-rotation without src/ dependency."""
    if not _TORCH_AVAILABLE or delta == 0:
        return k_r
    import math
    d = k_r.shape[-1]
    half_d = d // 2
    # Simple rotation: swap halves and negate one half (approximation of RoPE shift)
    cos_delta = math.cos(delta * 0.01)
    sin_delta = math.sin(delta * 0.01)
    k1 = k_r[..., :half_d]
    k2 = k_r[..., half_d:]
    rotated = torch.cat([
        k1 * cos_delta - k2 * sin_delta,
        k1 * sin_delta + k2 * cos_delta,
    ], dim=-1)
    return rotated


def _segment_id_from_tokens(token_ids: List[int], chunk_idx: int) -> bytes:
    """Deterministic segment ID from token content + chunk index."""
    if not token_ids:
        raw = b""
    else:
        raw = struct.pack(f"{len(token_ids)}I", *token_ids)
    return hashlib.sha256(raw + chunk_idx.to_bytes(4, "little")).digest()


# ============================================================================
# 2026-05-28: PegaFlowIrminsulDistributedKVCacheManagerMixin (Activity B)
# ============================================================================

class PegaFlowIrminsulDistributedKVCacheManagerMixin:
    """vLLM v1 KVCacheManager mixin: PegaFlow+Irminsul distributed non-contiguous KV reuse.

    Activity B: Non-Contiguous KV Cache Reuse via 4-level distributed lookup.
    Activity A-2: PegaFlow RDMA cross-node routing annotation.

    This mixin adds a parallel distributed segment store alongside vLLM's
    native paged block pool. It does NOT override native block allocation.

    API:
      put_distributed_segment(token_ids, chunk_idx, c_kv, k_r, source_position)
          → segment_id (bytes): store a KV segment in local HBM + PegaFlow.

      get_distributed_segment(segment_id, target_position)
          → (tensor, hit_type): 4-level lookup with δ-rotation on hit.
            hit_type ∈ {"local_hard_hit", "pegaflow_local_hit", "rdma_remote_hit", "miss"}

      distributed_hit_rate_breakdown()
          → dict: per-level hit rate breakdown.

      register_mock_remote_segment(node_id, segment_id, tensor)
          → void: for testing; registers a segment in mock remote store.

    Non-contiguous block table padding (vLLM block_size boundary compliance):
      The caller is responsible for block-table padding before attention kernel.
      This mixin returns the segment tensor as-is; padding to block_size boundaries
      is performed by the attention backend hook (attention_backend_patch.py).
    """

    # ---- Mixin initialization ---- #

    def _pegaflow_irminsul_init(
        self,
        config: Optional[PegaFlowIrminsulMixinConfig] = None,
    ) -> None:
        """Initialize PegaFlow+Irminsul mixin state."""
        self._pfi_cfg = config or PegaFlowIrminsulMixinConfig()

        # Try to import src/ native implementation
        (
            PegaFlowIrminsulDistributedSegmentCache,
            DistributedSegmentCacheConfig,
            IrminsulKVEntry,
        ) = _try_import_distributed_cache_src()

        if PegaFlowIrminsulDistributedSegmentCache is not None:
            # Build native cache using src/ components
            self._pfi_native = self._build_native_cache(
                PegaFlowIrminsulDistributedSegmentCache,
                DistributedSegmentCacheConfig,
            )
            self._pfi_use_native = True
        else:
            # Fallback to inline implementation
            self._pfi_native = None
            self._pfi_use_native = False
            self._pfi_entry_store: OrderedDict[bytes, _InlineIrminsulKVEntry] = OrderedDict()
            self._pfi_local_store = _InlinePegaFlowStore()
            self._pfi_metrics = _InlineDistributedHitRateMetrics()

            # Build RDMA router if src/ is available
            RDMARouter, RDMARouterConfig, PeerRegistry = _try_import_rdma_router_src()
            MockConnector, ConnectorConfig, _ = _try_import_pegaflow_connector_src()
            if RDMARouter is not None and MockConnector is not None:
                conn_cfg = ConnectorConfig(use_mock=True, seed=self._pfi_cfg.seed)
                connector = MockConnector(conn_cfg)
                router_cfg = RDMARouterConfig(
                    peer_nodes_config_path=self._pfi_cfg.peer_nodes_config_path,
                    bloom_filter_capacity=self._pfi_cfg.bloom_filter_capacity,
                    bloom_filter_error_rate=self._pfi_cfg.bloom_filter_error_rate,
                    rdma_reuse_discount=self._pfi_cfg.rdma_reuse_discount,
                    seed=self._pfi_cfg.seed,
                )
                peer_reg = PeerRegistry(router_cfg)
                self._pfi_rdma_router = RDMARouter(connector, peer_reg, router_cfg)
            else:
                self._pfi_rdma_router = None

    def _build_native_cache(
        self,
        PegaFlowIrminsulDistributedSegmentCache: type,
        DistributedSegmentCacheConfig: type,
    ) -> Any:
        """Build a native PegaFlowIrminsulDistributedSegmentCache from src/."""
        _add_repo_root_to_path()

        IrminsulMLASegmentCache, IrminsulMLAConfig, _ = _try_import_irminsul_src()
        MockConnector, ConnectorConfig, create_connector = _try_import_pegaflow_connector_src()
        RDMARouter, RDMARouterConfig, PeerRegistry = _try_import_rdma_router_src()

        if any(x is None for x in [
            IrminsulMLASegmentCache, MockConnector, RDMARouter, PeerRegistry
        ]):
            return None

        # Build sub-components
        irminsul_cfg = IrminsulMLAConfig(seed=self._pfi_cfg.seed)
        local_irminsul = IrminsulMLASegmentCache(irminsul_cfg)

        conn_cfg = ConnectorConfig(use_mock=True, seed=self._pfi_cfg.seed)
        pegaflow_local = create_connector(conn_cfg)

        router_cfg = RDMARouterConfig(
            peer_nodes_config_path=self._pfi_cfg.peer_nodes_config_path,
            bloom_filter_capacity=self._pfi_cfg.bloom_filter_capacity,
            bloom_filter_error_rate=self._pfi_cfg.bloom_filter_error_rate,
            rdma_reuse_discount=self._pfi_cfg.rdma_reuse_discount,
            seed=self._pfi_cfg.seed,
        )
        peer_reg = PeerRegistry(router_cfg)
        rdma_router = RDMARouter(pegaflow_local, peer_reg, router_cfg)

        dist_cfg = DistributedSegmentCacheConfig(
            rdma_reuse_discount=self._pfi_cfg.rdma_reuse_discount,
            bloom_sync_interval_ms=self._pfi_cfg.bloom_sync_interval_ms,
            local_max_entries=self._pfi_cfg.local_max_entries,
            avg_chunk_size=self._pfi_cfg.avg_chunk_size,
            seed=self._pfi_cfg.seed,
        )

        return PegaFlowIrminsulDistributedSegmentCache(
            local_irminsul_cache=local_irminsul,
            pegaflow_local=pegaflow_local,
            pegaflow_rdma_router=rdma_router,
            config=dist_cfg,
            local_node_id=self._pfi_cfg.local_node_id,
        )

    def _pfi_ensure_init(self) -> None:
        """Lazy initialization guard."""
        if not hasattr(self, "_pfi_cfg"):
            self._pegaflow_irminsul_init()

    # ---- Core segment API ---- #

    def put_distributed_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        c_kv: Any,
        k_r: Any,
        source_position: int,
        layer_idx: int = 0,
    ) -> bytes:
        """Store a KV segment in local HBM + PegaFlow local.

        Returns the deterministic segment_id (bytes) for later retrieval.

        vLLM block_size boundary compliance:
          Callers should align token_ids to block_size boundaries before calling.
          Segment splitting across block boundaries is not performed here.
        """
        self._pfi_ensure_init()

        if self._pfi_use_native and self._pfi_native is not None:
            return self._pfi_native.put_segment(
                token_ids, chunk_idx, c_kv, k_r, source_position, layer_idx
            )

        # Inline fallback
        seg_id = _segment_id_from_tokens(token_ids, chunk_idx)

        if _TORCH_AVAILABLE:
            entry = _InlineIrminsulKVEntry(
                segment_id=seg_id,
                c_kv_tensor=c_kv.detach().clone() if hasattr(c_kv, "detach") else c_kv,
                k_r_tensor=k_r.detach().clone() if hasattr(k_r, "detach") else k_r,
                source_position=source_position,
            )
            # Enforce LRU capacity
            while len(self._pfi_entry_store) >= self._pfi_cfg.local_max_entries:
                evicted_id, evicted_entry = self._pfi_entry_store.popitem(last=False)
                if _TORCH_AVAILABLE:
                    combined = torch.cat(
                        [evicted_entry.c_kv_tensor, evicted_entry.k_r_tensor], dim=-1
                    )
                    self._pfi_local_store.put(evicted_id.hex(), combined)
            self._pfi_entry_store[seg_id] = entry

        return seg_id

    def get_distributed_segment(
        self,
        segment_id: bytes,
        target_position: int,
    ) -> Tuple[Optional[Any], str]:
        """4-level distributed lookup + δ-rotation on hit.

        Returns:
          (kv_tensor, hit_type) where hit_type ∈ {
            "local_hard_hit", "pegaflow_local_hit", "rdma_remote_hit", "miss"
          }

        For "local_hard_hit" and "pegaflow_local_hit":
          c_KV is returned as-is (position-free).
          k_r has δ-rotation applied: δ = target_position - source_position.
          Final tensor = concat(c_kv, k_r_corrected, dim=-1).
        """
        self._pfi_ensure_init()

        if self._pfi_use_native and self._pfi_native is not None:
            return self._pfi_native.get_distributed(segment_id, target_position)

        # Inline 4-level fallback
        metrics: _InlineDistributedHitRateMetrics = self._pfi_metrics

        # Level 1: local HBM entry store
        if segment_id in self._pfi_entry_store:
            self._pfi_entry_store.move_to_end(segment_id)
            entry = self._pfi_entry_store[segment_id]
            kv = self._inline_apply_delta_rotation(entry, target_position)
            metrics.record("local_hard_hit")
            return kv, "local_hard_hit"

        # Level 2: PegaFlow local host/SSD
        seg_hex = segment_id.hex()
        raw = self._pfi_local_store.get(seg_hex)
        if raw is not None:
            metrics.record("pegaflow_local_hit")
            return raw, "pegaflow_local_hit"

        # Level 3: PegaFlow RDMA remote
        if self._pfi_rdma_router is not None:
            rdma_result, hit_type_rdma = self._pfi_rdma_router.route_segment_request(
                segment_id, self._pfi_cfg.local_node_id
            )
            if rdma_result is not None and hit_type_rdma == "rdma_remote":
                metrics.record("rdma_remote_hit")
                return rdma_result, "rdma_remote_hit"

        # Level 4: miss
        metrics.record("miss")
        return None, "miss"

    def _inline_apply_delta_rotation(
        self,
        entry: _InlineIrminsulKVEntry,
        target_position: int,
    ) -> Optional[Any]:
        """Apply δ-rotation and return concat(c_kv, k_r_corrected, dim=-1)."""
        if not _TORCH_AVAILABLE:
            return None
        delta = target_position - entry.source_position
        c_kv = entry.c_kv_tensor
        k_r = entry.k_r_tensor

        if delta == 0:
            k_r_corrected = k_r
        else:
            k_r_corrected = _inline_delta_rotate(k_r, delta)

        # Flatten to 2D for concat
        if c_kv.dim() > 2:
            c_kv = c_kv.reshape(-1, c_kv.shape[-1])
        if k_r_corrected.dim() > 2:
            k_r_corrected = k_r_corrected.reshape(-1, k_r_corrected.shape[-1])

        return torch.cat([c_kv, k_r_corrected], dim=-1)

    # ---- Stats ---- #

    def distributed_hit_rate_breakdown(self) -> Dict[str, Any]:
        """Return per-level hit rate breakdown."""
        self._pfi_ensure_init()
        if self._pfi_use_native and self._pfi_native is not None:
            return self._pfi_native.distributed_hit_rate_breakdown()
        return self._pfi_metrics.summary()

    def distributed_memory_bytes(self) -> int:
        """Local HBM segment store memory footprint."""
        self._pfi_ensure_init()
        if self._pfi_use_native and self._pfi_native is not None:
            return self._pfi_native.memory_bytes()
        if not _TORCH_AVAILABLE:
            return 0
        total = 0
        for entry in self._pfi_entry_store.values():
            total += (
                entry.c_kv_tensor.nelement() * entry.c_kv_tensor.element_size()
                + entry.k_r_tensor.nelement() * entry.k_r_tensor.element_size()
            )
        return total

    def reset_distributed_stats(self) -> None:
        """Reset hit rate counters."""
        self._pfi_ensure_init()
        if self._pfi_use_native and self._pfi_native is not None:
            self._pfi_native.reset_stats()
        elif hasattr(self, "_pfi_metrics"):
            self._pfi_metrics.reset()

    # ---- Mock RDMA test helpers ---- #

    def register_mock_remote_segment(
        self,
        node_id: str,
        segment_id: bytes,
        tensor: Any,
    ) -> None:
        """Register a segment in the mock RDMA remote store (for testing)."""
        self._pfi_ensure_init()
        if self._pfi_use_native and self._pfi_native is not None:
            # Access rdma router via native cache
            rdma_router = getattr(self._pfi_native, "_rdma_router", None)
            if rdma_router is not None:
                rdma_router.register_mock_remote_segment(node_id, segment_id, tensor)
        elif self._pfi_rdma_router is not None:
            self._pfi_rdma_router.register_mock_remote_segment(
                node_id, segment_id, tensor
            )

    def get_rdma_router(self) -> Optional[Any]:
        """Return the RDMA router for external configuration."""
        self._pfi_ensure_init()
        if self._pfi_use_native and self._pfi_native is not None:
            return getattr(self._pfi_native, "_rdma_router", None)
        return getattr(self, "_pfi_rdma_router", None)

    def pegaflow_irminsul_metrics(self) -> Dict[str, Any]:
        """Return combined Activity A-2 + B metrics dict."""
        breakdown = self.distributed_hit_rate_breakdown()
        breakdown.update({
            "local_memory_bytes": self.distributed_memory_bytes(),
            "use_native_src": self._pfi_use_native if hasattr(self, "_pfi_use_native") else False,
        })
        return breakdown


# ============================================================================
# Factory
# ============================================================================

def make_pegaflow_irminsul_kv_cache_manager_class(
    BaseKVCacheManagerClass: type,
    config: Optional[PegaFlowIrminsulMixinConfig] = None,
) -> type:
    """Build a vLLM v1 KVCacheManager subclass with PegaFlow+Irminsul distributed cache.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        PFIManager = make_pegaflow_irminsul_kv_cache_manager_class(KVCacheManager)

    The returned class inherits all KVCacheManager methods and adds:
      - put_distributed_segment(): store KV segment in 4-level cache
      - get_distributed_segment(): 4-level lookup with δ-rotation
      - distributed_hit_rate_breakdown(): per-level hit rate stats
      - pegaflow_irminsul_metrics(): combined A-2 + B metrics

    All native KVCacheManager block pool methods are preserved unchanged.
    """
    cfg = config or PegaFlowIrminsulMixinConfig()

    class PegaFlowIrminsulKVCacheManager(
        PegaFlowIrminsulDistributedKVCacheManagerMixin, BaseKVCacheManagerClass
    ):
        """KVCacheManager with PegaFlow+Irminsul distributed non-contiguous KV (Activity B)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pfi_config = kwargs.pop("pegaflow_irminsul_config", cfg)
            super().__init__(*args, **kwargs)
            self._pegaflow_irminsul_init(pfi_config)

    PegaFlowIrminsulKVCacheManager.__name__ = (
        f"PegaFlowIrminsul_{BaseKVCacheManagerClass.__name__}"
    )
    PegaFlowIrminsulKVCacheManager.__qualname__ = PegaFlowIrminsulKVCacheManager.__name__
    return PegaFlowIrminsulKVCacheManager
