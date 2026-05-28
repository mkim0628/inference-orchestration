"""rdma_attention_backend_patch.py — Activity B: RDMA remote segment attention backend
extension for vLLM 0.21.0.

2026-05-28: PegaFlowRDMASegmentAttentionHook — extends vLLM's attention backend to
            support non-contiguous KV segment reuse from remote RDMA nodes.

            Design:
              This hook implements the Activity B reuse path at the attention layer
              boundary. It intercepts KV after write (post-computation) and stores
              segments into the PegaFlow+Irminsul distributed cache. On read, it
              retrieves previously computed segments for reuse.

              vLLM accuracy contract (consistent with prior cycles):
                write_to_cache(): MUST return original key/value tensors unchanged
                                  so the primary attention kernel receives unmodified KV.
                read_from_cache(): Returns retrieved segment tensor (c_kv || k_r_corrected)
                                   for use in the non-contiguous reuse path (NOT the primary
                                   attention kernel path — callers handle substitution).

              Block boundary compliance:
                Segments are stored at token granularity. Before use in the primary
                attention kernel, the caller pads block_table entries to vLLM's
                block_size boundary. This hook does not pad tensors.

            Integration points:
              - Called after Q/K/V projection, before FlashAttention kernel.
              - write_to_cache() stores the computed K/V into the distributed segment cache.
              - read_from_cache() queries the 4-level distributed cache for segment reuse.
              - apply_pegaflow_rdma_segment_patch(): monkey-patches write_to_cache /
                read_from_cache onto a vLLM attention backend implementation class.

vLLM version: 0.21.0
Activity: B — Non-contiguous RDMA segment reuse at attention layer boundary
          A — PegaFlow RDMA routing (via PegaFlowIrminsulDistributedKVCacheManagerMixin)
"""

from __future__ import annotations

import sys
import pathlib
import hashlib
import time
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
    _add_repo_root_to_path()
    try:
        from src.cache.pegaflow_irminsul_distributed_cache import (
            PegaFlowIrminsulDistributedSegmentCache,
            DistributedSegmentCacheConfig,
        )
        return PegaFlowIrminsulDistributedSegmentCache, DistributedSegmentCacheConfig
    except ImportError:
        return None, None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PegaFlowRDMASegmentHookConfig:
    """Configuration for PegaFlowRDMASegmentAttentionHook."""
    n_layers: int = 32
    avg_chunk_tokens: int = 256           # average CDC chunk size in tokens
    kv_size_per_token_bytes: int = 512    # estimate for segment size calculation
    enabled: bool = True
    seed: int = 42


# ============================================================================
# 2026-05-28: PegaFlowRDMASegmentAttentionHook (Activity B)
# ============================================================================

class PegaFlowRDMASegmentAttentionHook:
    """Attention backend hook for Activity B: RDMA non-contiguous KV segment reuse.

    Accuracy contract:
      write_to_cache():
        - Stores K/V tensors into the distributed segment cache (side-effect only).
        - RETURNS the original key_tensor and value_tensor UNCHANGED.
        - The primary attention kernel (FlashAttention) always receives the full,
          unmodified KV — zero softmax distortion.

      read_from_cache():
        - Queries the 4-level distributed cache for a previously computed segment.
        - Returns (tensor, hit_type) for use in the non-contiguous reuse path.
        - The caller is responsible for injecting the reused tensor into the
          attention input (block_table padding).

    vLLM block_size boundary compliance:
      Segments are stored at token granularity within vLLM's block_size grid.
      Segment boundaries are aligned to avg_chunk_tokens for CDC compatibility.
      Segments that span block boundaries are not split; the caller must ensure
      token_ids passed to put_distributed_segment() fall within a single block.
    """

    def __init__(
        self,
        config: Optional[PegaFlowRDMASegmentHookConfig] = None,
        distributed_cache: Optional[Any] = None,
        enabled: bool = True,
    ) -> None:
        self.config = config or PegaFlowRDMASegmentHookConfig()
        self.enabled = enabled
        self._distributed_cache = distributed_cache
        self._write_count: int = 0
        self._read_hit_count: int = 0
        self._read_miss_count: int = 0

        if _TORCH_AVAILABLE:
            torch.manual_seed(self.config.seed)

    def set_distributed_cache(self, cache: Any) -> None:
        """Wire up a PegaFlowIrminsulDistributedSegmentCache or compatible object."""
        self._distributed_cache = cache

    def write_to_cache(
        self,
        kv_key: str,
        key_tensor: Any,
        value_tensor: Any,
        layer_idx: int = 0,
        token_ids: Optional[List[int]] = None,
        chunk_idx: int = 0,
        source_position: int = 0,
    ) -> Tuple[Any, Any]:
        """Store K/V into the distributed segment cache; return ORIGINAL tensors.

        Accuracy contract: returns (key_tensor, value_tensor) UNCHANGED.
        The primary attention kernel always receives the unmodified KV.

        Side-effect: stores the segment into the 4-level distributed cache
        for later non-contiguous reuse by other requests.

        Args:
          kv_key: unique string key for this segment (e.g., "session0:layer2:chunk3")
          key_tensor: K tensor from attention projection
          value_tensor: V tensor from attention projection
          layer_idx: transformer layer index
          token_ids: token IDs for CDC segment key (optional; derived from kv_key if None)
          chunk_idx: CDC chunk index within the sequence
          source_position: absolute position of this segment in the sequence

        Returns:
          (key_tensor, value_tensor): ORIGINAL tensors, UNCHANGED.
        """
        if not self.enabled or self._distributed_cache is None:
            return key_tensor, value_tensor

        try:
            # Derive token_ids from kv_key hash if not provided
            if token_ids is None:
                key_hash = int(hashlib.sha256(kv_key.encode()).hexdigest()[:8], 16)
                token_ids = [key_hash % 65536]  # minimal single-token ID for keying

            # Split key_tensor / value_tensor into c_kv and k_r components.
            # Per Irminsul MLA convention: last d_r dims are k_r, rest are c_kv.
            # When no explicit split info is available, treat full K as c_kv (position-free
            # path) and create a minimal k_r from the last half of K.
            if _TORCH_AVAILABLE and hasattr(key_tensor, "shape"):
                d = key_tensor.shape[-1]
                d_r = max(1, d // 4)   # use last d//4 dims as k_r
                c_kv = key_tensor[..., :-d_r]
                k_r = key_tensor[..., -d_r:]
            else:
                c_kv = key_tensor
                k_r = value_tensor

            # Store in distributed cache (side-effect only)
            if hasattr(self._distributed_cache, "put_distributed_segment"):
                # PegaFlowIrminsulDistributedKVCacheManagerMixin API
                self._distributed_cache.put_distributed_segment(
                    token_ids=token_ids,
                    chunk_idx=chunk_idx,
                    c_kv=c_kv,
                    k_r=k_r,
                    source_position=source_position,
                    layer_idx=layer_idx,
                )
            elif hasattr(self._distributed_cache, "put_segment"):
                # Direct PegaFlowIrminsulDistributedSegmentCache API
                self._distributed_cache.put_segment(
                    token_ids=token_ids,
                    chunk_idx=chunk_idx,
                    c_kv=c_kv,
                    k_r=k_r,
                    source_position=source_position,
                    layer_idx=layer_idx,
                )

            self._write_count += 1
        except Exception:
            pass  # Never block the primary attention path

        # Critical: return ORIGINAL tensors unchanged
        return key_tensor, value_tensor

    def read_from_cache(
        self,
        kv_key: str,
        layer_idx: int = 0,
        target_position: int = 0,
        segment_id: Optional[bytes] = None,
    ) -> Tuple[Optional[Any], str]:
        """Query the 4-level distributed cache for a non-contiguous segment.

        Returns:
          (tensor, hit_type): tensor is the reusable KV (c_kv || k_r_corrected)
                              or None on miss.
          hit_type ∈ {"local_hard_hit", "pegaflow_local_hit", "rdma_remote_hit", "miss"}

        This is the Activity B reuse path, NOT the primary attention kernel path.
        The caller injects the returned tensor into the attention input by padding
        the block_table (block_table[... , :segment_len] = retrieved_block_ids).
        """
        if not self.enabled or self._distributed_cache is None:
            return None, "miss"

        try:
            # Derive segment_id if not provided
            if segment_id is None:
                segment_id = hashlib.sha256(
                    f"{kv_key}:{layer_idx}".encode()
                ).digest()

            if hasattr(self._distributed_cache, "get_distributed_segment"):
                tensor, hit_type = self._distributed_cache.get_distributed_segment(
                    segment_id, target_position
                )
            elif hasattr(self._distributed_cache, "get_distributed"):
                tensor, hit_type = self._distributed_cache.get_distributed(
                    segment_id, target_position
                )
            else:
                return None, "miss"

            if tensor is not None:
                self._read_hit_count += 1
            else:
                self._read_miss_count += 1

            return tensor, hit_type

        except Exception:
            self._read_miss_count += 1
            return None, "miss"

    def hook_stats(self) -> Dict[str, Any]:
        """Return statistics for this hook instance."""
        total_reads = self._read_hit_count + self._read_miss_count
        return {
            "write_count": self._write_count,
            "read_hit_count": self._read_hit_count,
            "read_miss_count": self._read_miss_count,
            "read_hit_rate": (
                self._read_hit_count / total_reads if total_reads > 0 else 0.0
            ),
            "enabled": self.enabled,
            "has_distributed_cache": self._distributed_cache is not None,
        }


# ============================================================================
# Patch factory
# ============================================================================

def apply_pegaflow_rdma_segment_patch(
    AttentionImplClass: type,
    hook: PegaFlowRDMASegmentAttentionHook,
    layer_idx: int = 0,
) -> PegaFlowRDMASegmentAttentionHook:
    """Monkey-patch write_to_cache / read_from_cache onto an attention impl class.

    Adds Activity B non-contiguous RDMA segment reuse hooks.
    Does NOT modify the primary attention computation path (forward()).

    Usage:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        hook = PegaFlowRDMASegmentAttentionHook(config=..., distributed_cache=cache)
        apply_pegaflow_rdma_segment_patch(FlashAttentionImpl, hook, layer_idx=0)

    After patching:
        impl.write_to_cache(kv_key, k, v, layer_idx=0) → (k, v)  # original returned
        impl.read_from_cache(kv_key, layer_idx=0, target_position=0) → (tensor, hit_type)
    """
    _hook = hook
    _layer_idx = layer_idx

    def write_to_cache(
        self_impl: Any,
        kv_key: str,
        key_tensor: Any,
        value_tensor: Any,
        layer_idx: int = _layer_idx,
        token_ids: Optional[List[int]] = None,
        chunk_idx: int = 0,
        source_position: int = 0,
    ) -> Tuple[Any, Any]:
        return _hook.write_to_cache(
            kv_key, key_tensor, value_tensor, layer_idx,
            token_ids, chunk_idx, source_position,
        )

    def read_from_cache(
        self_impl: Any,
        kv_key: str,
        layer_idx: int = _layer_idx,
        target_position: int = 0,
        segment_id: Optional[bytes] = None,
    ) -> Tuple[Optional[Any], str]:
        return _hook.read_from_cache(kv_key, layer_idx, target_position, segment_id)

    AttentionImplClass.write_to_cache = write_to_cache
    AttentionImplClass.read_from_cache = read_from_cache
    return _hook


def extend_cache_config_pegaflow_rdma(
    cache_config: Any,
    rdma_reuse_discount: float = 0.8,
    bloom_filter_capacity: int = 100_000,
    bloom_filter_error_rate: float = 0.01,
    avg_chunk_tokens: int = 256,
) -> None:
    """Inject PegaFlow RDMA Activity B fields into vLLM CacheConfig.

    Compatible with vLLM's pydantic-based CacheConfig (uses object.__setattr__).

    Args:
      cache_config: vllm.config.CacheConfig instance
      rdma_reuse_discount: RDMA reuse decision threshold (default 0.8)
      bloom_filter_capacity: Bloom Filter capacity for peer segment index
      bloom_filter_error_rate: Bloom Filter false positive rate
      avg_chunk_tokens: CDC average chunk size in tokens
    """
    object.__setattr__(cache_config, "compression_method", "none")
    object.__setattr__(cache_config, "pegaflow_rdma_reuse_discount", rdma_reuse_discount)
    object.__setattr__(cache_config, "pegaflow_bloom_capacity", bloom_filter_capacity)
    object.__setattr__(cache_config, "pegaflow_bloom_error_rate", bloom_filter_error_rate)
    object.__setattr__(cache_config, "pegaflow_avg_chunk_tokens", avg_chunk_tokens)
    object.__setattr__(cache_config, "pegaflow_rdma_enabled", True)
