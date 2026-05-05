"""CompressedKVManager — CompressedDiffStore adapted for vLLM 0.20.1.

Activity B+C Cross-1: INT4-compressed masters + FP16 diff storage, integrated
with vLLM's block manager interface.

Integration point: vLLM's KVCacheManager (vllm.v1.core.kv_cache_manager).

Design:
  - CompressedKVManager wraps CompressedDiffStore logic with vLLM-aware APIs.
  - Master KV blocks are stored as INT4 (via NQKVCodecPatch) instead of FP16.
  - Diff blocks remain FP16 (small, compression overhead not worth it).
  - Provides block_id-based API that mirrors vLLM's block manager conventions.
  - CompressedMasterEntry uses (indices, mu, sigma, original_shape) — identical
    to CompressedDiffStore in src/cache/compressed_diff_store.py.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional

import torch

# ---------------------------------------------------------------------------
# vLLM imports — graceful fallback
# ---------------------------------------------------------------------------
try:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    _VLLM_KV_CACHE_MANAGER_AVAILABLE = True
except ImportError:
    _VLLM_KV_CACHE_MANAGER_AVAILABLE = False

try:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    _VLLM_KV_CACHE_CONFIG_AVAILABLE = True
except ImportError:
    _VLLM_KV_CACHE_CONFIG_AVAILABLE = False

# Local patch imports
from vllm_integration.nqkv_codec_patch import NQKVCodecPatch
from vllm_integration.diff_aware_kv_patch import DiffAwareKVPatch, DiffBlockEntry


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class CompressedMasterEntry:
    """INT4-compressed master KV storage entry (one vLLM block group)."""
    indices: torch.Tensor       # (num_quant_blocks, block_size) uint8
    mu: torch.Tensor            # (num_quant_blocks,) float16
    sigma: torch.Tensor         # (num_quant_blocks,) float16
    original_shape: torch.Size
    group_id: str


class CompressedKVManager:
    """CompressedDiffStore for vLLM: INT4 masters + FP16 diffs (Cross-1).

    Extends DiffAwareKVPatch by compressing master blocks with NQKVCodecPatch.

    Key API:
        store_block(block_id, kv_tensor)
            Compress and store kv_tensor as master for block_id.

        store_agent_diff(block_id, agent_id, kv_tensor)
            Compute and store diff relative to compressed master.

        retrieve_block(block_id, agent_id=None)
            Reconstruct KV: decompress master, apply agent diff if any.

        evict_block(block_id)
            Remove block: compressed master + all agent diffs.

    Tracking:
        block_id → CompressedMasterEntry  (self._compressed_masters)
        block_id → agent_id → DiffBlockEntry  (via _diff_patch internals)
    """

    def __init__(
        self,
        seq_block_size: int = 64,
        diff_threshold: float = 0.1,
        max_blocks: int = 1000,
        codec_block_size: int = 64,
    ) -> None:
        """
        Args:
            seq_block_size:    Sequence-level diff block size (tokens).
            diff_threshold:    RMS distance below which agent block == master.
            max_blocks:        Maximum number of tracked block groups before LRU evict.
            codec_block_size:  NQKVCodec quantisation block size (elements).
        """
        self.seq_block_size = seq_block_size
        self.diff_threshold = diff_threshold
        self.max_blocks = max_blocks

        self.codec = NQKVCodecPatch(block_size=codec_block_size)

        # LRU-ordered compressed masters: group_id → CompressedMasterEntry
        self._compressed_masters: OrderedDict[str, CompressedMasterEntry] = OrderedDict()
        # vLLM block_id → group_id
        self._block_to_group: Dict[int, str] = {}
        # Per-group per-agent diffs (reusing DiffAwareKVPatch's internal infra)
        self._diff_patch = DiffAwareKVPatch(
            seq_block_size=seq_block_size,
            diff_threshold=diff_threshold,
            max_groups=max_blocks,
        )

        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def store_block(
        self,
        block_id: int,
        kv_tensor: torch.Tensor,
        group_id: Optional[str] = None,
    ) -> None:
        """Compress and store kv_tensor as the master for block_id.

        If max_blocks is reached, the oldest LRU group is evicted first.

        Args:
            block_id:   vLLM physical block ID.
            kv_tensor:  Full FP16 KV tensor for this block.
            group_id:   Optional group key (default: str(block_id)).
        """
        gid = group_id if group_id is not None else str(block_id)
        self._block_to_group[block_id] = gid

        if len(self._compressed_masters) >= self.max_blocks and gid not in self._compressed_masters:
            oldest_gid = next(iter(self._compressed_masters))
            self._evict_by_group_id(oldest_gid)

        indices, mu, sigma = self.codec.encode(kv_tensor)
        self._compressed_masters[gid] = CompressedMasterEntry(
            indices=indices,
            mu=mu,
            sigma=sigma,
            original_shape=kv_tensor.shape,
            group_id=gid,
        )
        self._compressed_masters.move_to_end(gid)

    def store_agent_diff(
        self,
        block_id: int,
        agent_id: str,
        kv_tensor: torch.Tensor,
    ) -> None:
        """Compute and store diff relative to the compressed master for block_id.

        The master is decompressed transiently for diff computation.

        Args:
            block_id:  vLLM physical block ID.
            agent_id:  Request/agent identifier string.
            kv_tensor: Full FP16 KV tensor for this agent at this block.
        """
        gid = self._block_to_group.get(block_id, str(block_id))
        master_kv = self._decompress_master(gid)
        if master_kv is None:
            # No master registered — store as master
            self.store_block(block_id, kv_tensor, group_id=gid)
            return
        # Register transient master in DiffAwareKVPatch for diff computation
        self._diff_patch._register_master(master_kv, gid)
        self._diff_patch._put_agent_kv(agent_id, gid, kv_tensor)

    def retrieve_block(
        self,
        block_id: int,
        agent_id: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Reconstruct KV for (block_id, agent_id).

        If agent_id is None or no diff is found, returns the decompressed master.

        Returns None if block_id has no registered master.
        """
        gid = self._block_to_group.get(block_id, str(block_id))
        master_kv = self._decompress_master(gid)
        if master_kv is None:
            self._misses += 1
            return None

        if agent_id is None:
            self._hits += 1
            return master_kv

        # Try to find agent diff
        agent_diffs = self._diff_patch._diffs.get(gid)
        if agent_diffs is not None:
            diff_entry = agent_diffs.get(agent_id)
            if diff_entry is not None:
                # Apply diff blocks
                result = master_kv.float().clone()
                seq_len = result.shape[-2] if result.dim() >= 2 else result.numel()
                for blk_idx, diff_block in diff_entry.diff_blocks.items():
                    start = blk_idx * self.seq_block_size
                    end = min(start + self.seq_block_size, seq_len)
                    if result.dim() >= 2:
                        result[..., start:end, :] = result[..., start:end, :] + diff_block.float()
                    else:
                        result[start:end] = result[start:end] + diff_block.float()
                self._hits += 1
                return result.half()

        # Fall back to master
        self._hits += 1
        return master_kv

    def evict_block(self, block_id: int) -> int:
        """Evict compressed master and all agent diffs for block_id.

        Returns bytes freed.
        """
        gid = self._block_to_group.pop(block_id, str(block_id))
        return self._evict_by_group_id(gid)

    # ------------------------------------------------------------------
    # Statistics and memory accounting
    # ------------------------------------------------------------------

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Sum of compressed master bytes + all agent diff bytes."""
        total = 0
        for cm in self._compressed_masters.values():
            total += cm.indices.nbytes + cm.mu.nbytes + cm.sigma.nbytes
        for agent_diffs in self._diff_patch._diffs.values():
            for diff_entry in agent_diffs.values():
                total += sum(b.nbytes for b in diff_entry.diff_blocks.values())
        return total

    def compression_summary(self, sample_kv: torch.Tensor) -> dict:
        """Return compression ratio vs FP16 for a sample KV tensor."""
        return {
            "compression_ratio": self.codec.compression_ratio(sample_kv),
            "n_compressed_masters": len(self._compressed_masters),
            "memory_bytes": self.memory_bytes(),
        }

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._diff_patch.reset_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompress_master(self, group_id: str) -> Optional[torch.Tensor]:
        entry = self._compressed_masters.get(group_id)
        if entry is None:
            return None
        self._compressed_masters.move_to_end(group_id)
        return self.codec.decode(
            entry.indices, entry.mu, entry.sigma, entry.original_shape
        )

    def _evict_by_group_id(self, group_id: str) -> int:
        freed = 0
        if group_id in self._compressed_masters:
            cm = self._compressed_masters.pop(group_id)
            freed += cm.indices.nbytes + cm.mu.nbytes + cm.sigma.nbytes
        # Evict diffs via diff_patch
        freed += self._diff_patch._evict_group(group_id)
        # Clean stale block_to_group mappings
        stale = [bid for bid, gid in self._block_to_group.items() if gid == group_id]
        for bid in stale:
            del self._block_to_group[bid]
        return freed


__all__ = ["CompressedKVManager", "CompressedMasterEntry"]
