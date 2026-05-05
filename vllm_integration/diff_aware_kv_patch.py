"""DiffAwareKVPatch — DiffAwareSegmentStore adapted for vLLM 0.20.1.

Activity B-1: Non-contiguous KV cache reuse via master + block-sparse diff storage.

Integration point: vLLM's KVCacheManager (vllm.v1.core.kv_cache_manager).

Design:
  - DiffAwareKVPatch wraps the DiffAwareSegmentStore logic (master + diff)
    and adapts it to work alongside vLLM's physical block IDs.
  - vLLM allocates physical blocks (identified by integer block_id).
    DiffAwareKVPatch maps these block_ids to group_id/agent_id keys in the
    master+diff store.
  - register_master_block() / get_agent_block() provide the primary API.
  - No FAISS — search space is the number of master groups, not total tokens.
  - Works with vLLM's KVCacheBlocks structure via block_id integers.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

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


# ---------------------------------------------------------------------------
# Internal data structures (mirrors DiffAwareSegmentStore dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class MasterBlockEntry:
    """FP16 master KV for a block group."""
    kv: torch.Tensor     # FP16 full KV tensor for this block group
    group_id: str


@dataclass
class DiffBlockEntry:
    """Per-agent diff storage for one block group."""
    group_id: str
    agent_id: str
    # block_idx → diff tensor (FP16, only blocks where RMS dist > threshold)
    diff_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Block indices identical to master (pointer reference only)
    master_ref_blocks: Set[int] = field(default_factory=set)


class DiffAwareKVPatch:
    """DiffAwareSegmentStore adapted for vLLM block IDs (Activity B-1).

    Exposes:
      register_master_block(block_id, kv_tensor)
          Store a block as the master for its group. group_id is derived from
          block_id using the provided block_id_to_group callable (default:
          str(block_id)).

      get_agent_block(block_id, agent_id)
          Retrieve reconstructed KV for a (block_id, agent_id) pair.

      put_agent_block(block_id, agent_id, kv_tensor)
          Store per-agent diff relative to the registered master for block_id.

    No FAISS.  Search space = number of master groups (not total tokens).
    """

    def __init__(
        self,
        seq_block_size: int = 64,
        diff_threshold: float = 0.1,
        max_groups: int = 1000,
    ) -> None:
        """
        Args:
            seq_block_size:  Sequence-level block size for diff granularity
                             (distinct from vLLM's physical page block size).
            diff_threshold:  RMS distance threshold below which a block is
                             treated as identical to master.
            max_groups:      Maximum number of master groups before LRU eviction.
        """
        self.seq_block_size = seq_block_size
        self.diff_threshold = diff_threshold
        self.max_groups = max_groups

        # LRU-ordered masters: group_id → MasterBlockEntry
        self._masters: OrderedDict[str, MasterBlockEntry] = OrderedDict()
        # Per-group per-agent diffs: group_id → {agent_id → DiffBlockEntry}
        self._diffs: Dict[str, Dict[str, DiffBlockEntry]] = {}
        # vLLM block_id → group_id mapping
        self._block_to_group: Dict[int, str] = {}

        self._hits = 0
        self._misses = 0
        self._diff_hits = 0
        self._master_hits = 0

    # ------------------------------------------------------------------
    # Primary API — vLLM block_id based
    # ------------------------------------------------------------------

    def register_master_block(
        self,
        block_id: int,
        kv_tensor: torch.Tensor,
        group_id: Optional[str] = None,
    ) -> None:
        """Register a vLLM physical block's KV as the master for its group.

        Args:
            block_id:   vLLM physical block ID (integer from KVCacheManager).
            kv_tensor:  KV tensor for this block, any shape.
            group_id:   Optional explicit group key.  Defaults to str(block_id).
        """
        gid = group_id if group_id is not None else str(block_id)
        self._block_to_group[block_id] = gid
        self._register_master(kv_tensor, gid)

    def put_agent_block(
        self,
        block_id: int,
        agent_id: str,
        kv_tensor: torch.Tensor,
    ) -> None:
        """Store per-agent diff relative to the master for block_id.

        Args:
            block_id:  vLLM physical block ID.
            agent_id:  Agent/request identifier string.
            kv_tensor: Full KV tensor for this agent at this block.
        """
        gid = self._block_to_group.get(block_id, str(block_id))
        self._put_agent_kv(agent_id, gid, kv_tensor)

    def get_agent_block(
        self,
        block_id: int,
        agent_id: str,
    ) -> Optional[torch.Tensor]:
        """Reconstruct KV for a (block_id, agent_id) pair.

        Returns None if neither master nor diff is found.
        """
        gid = self._block_to_group.get(block_id, str(block_id))
        result = self._get_agent_kv(agent_id, gid)
        if result is not None:
            self._hits += 1
            self._diff_hits += 1
        else:
            # Fall back to master
            master = self._get_master_kv(gid)
            if master is not None:
                self._hits += 1
                self._master_hits += 1
                return master
            self._misses += 1
        return result

    def get_master_block(
        self,
        block_id: int,
    ) -> Optional[torch.Tensor]:
        """Return the master KV tensor for block_id, or None."""
        gid = self._block_to_group.get(block_id, str(block_id))
        result = self._get_master_kv(gid)
        if result is not None:
            self._hits += 1
            self._master_hits += 1
        else:
            self._misses += 1
        return result

    def evict_block(self, block_id: int) -> int:
        """Evict master and all agent diffs for a vLLM block. Returns bytes freed."""
        gid = self._block_to_group.pop(block_id, str(block_id))
        return self._evict_group(gid)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def diff_hit_stats(self) -> dict:
        """Non-contiguous hit breakdown."""
        total_hits = self._diff_hits + self._master_hits
        return {
            "diff_hit_rate": self._diff_hits / total_hits if total_hits > 0 else 0.0,
            "master_hit_rate": self._master_hits / total_hits if total_hits > 0 else 0.0,
            "overall_hit_rate": self.hit_rate(),
            "n_groups": len(self._masters),
            "n_tracked_blocks": len(self._block_to_group),
            "search_space_reduction": (
                max(len(self._block_to_group), 1) / max(len(self._masters), 1)
            ),
        }

    def memory_bytes(self) -> int:
        total = sum(e.kv.nbytes for e in self._masters.values())
        for agent_diffs in self._diffs.values():
            for diff_entry in agent_diffs.values():
                total += sum(b.nbytes for b in diff_entry.diff_blocks.values())
        return total

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._diff_hits = 0
        self._master_hits = 0

    # ------------------------------------------------------------------
    # Internal helpers (mirror DiffAwareSegmentStore internals)
    # ------------------------------------------------------------------

    def _register_master(self, master_kv: torch.Tensor, group_id: str) -> None:
        if len(self._masters) >= self.max_groups and group_id not in self._masters:
            oldest = next(iter(self._masters))
            self._evict_group(oldest)
        self._masters[group_id] = MasterBlockEntry(
            kv=master_kv.detach().clone(),
            group_id=group_id,
        )
        self._masters.move_to_end(group_id)
        if group_id not in self._diffs:
            self._diffs[group_id] = {}

    def _put_agent_kv(
        self,
        agent_id: str,
        group_id: str,
        agent_kv: torch.Tensor,
    ) -> None:
        master_kv = self._get_master_kv(group_id)
        if master_kv is None:
            self._register_master(agent_kv, group_id)
            return

        agent_kv_f = agent_kv.float()
        master_kv_f = master_kv.float()

        if agent_kv_f.dim() >= 2:
            seq_len = agent_kv_f.shape[-2]
        else:
            seq_len = agent_kv_f.numel()

        n_blocks = max(1, (seq_len + self.seq_block_size - 1) // self.seq_block_size)
        diff_entry = DiffBlockEntry(group_id=group_id, agent_id=agent_id)

        for blk in range(n_blocks):
            start = blk * self.seq_block_size
            end = min(start + self.seq_block_size, seq_len)
            if agent_kv_f.dim() >= 2:
                master_seg = master_kv_f[..., start:end, :]
                agent_seg = agent_kv_f[..., start:end, :]
            else:
                master_seg = master_kv_f[start:end]
                agent_seg = agent_kv_f[start:end]

            rms = ((master_seg - agent_seg).pow(2).mean()).sqrt().item()
            if rms > self.diff_threshold:
                diff_entry.diff_blocks[blk] = (agent_seg - master_seg).to(torch.float16)
            else:
                diff_entry.master_ref_blocks.add(blk)

        if group_id not in self._diffs:
            self._diffs[group_id] = {}
        self._diffs[group_id][agent_id] = diff_entry

    def _get_master_kv(self, group_id: str) -> Optional[torch.Tensor]:
        entry = self._masters.get(group_id)
        if entry is None:
            return None
        self._masters.move_to_end(group_id)
        return entry.kv

    def _get_agent_kv(
        self,
        agent_id: str,
        group_id: str,
    ) -> Optional[torch.Tensor]:
        master_kv = self._get_master_kv(group_id)
        if master_kv is None:
            return None
        agent_diffs = self._diffs.get(group_id)
        if agent_diffs is None:
            return None
        diff_entry = agent_diffs.get(agent_id)
        if diff_entry is None:
            return None

        result = master_kv.float().clone()
        if result.dim() >= 2:
            seq_len = result.shape[-2]
        else:
            seq_len = result.numel()

        for blk_idx, diff_block in diff_entry.diff_blocks.items():
            start = blk_idx * self.seq_block_size
            end = min(start + self.seq_block_size, seq_len)
            if result.dim() >= 2:
                result[..., start:end, :] = result[..., start:end, :] + diff_block.float()
            else:
                result[start:end] = result[start:end] + diff_block.float()

        return result.half()

    def _evict_group(self, group_id: str) -> int:
        freed = 0
        if group_id in self._masters:
            freed += self._masters.pop(group_id).kv.nbytes
        if group_id in self._diffs:
            for de in self._diffs.pop(group_id).values():
                freed += sum(b.nbytes for b in de.diff_blocks.values())
        # Clean up block_to_group mapping
        stale_blocks = [bid for bid, gid in self._block_to_group.items() if gid == group_id]
        for bid in stale_blocks:
            del self._block_to_group[bid]
        return freed


__all__ = ["DiffAwareKVPatch"]
