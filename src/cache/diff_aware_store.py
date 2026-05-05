"""DiffAwareSegmentStore — Master + block-sparse diff KV cache (Activity B-1).

Stores a shared master KV for an agent group, and only the diff blocks
for each individual agent. FAISS is explicitly NOT used — the search space
is limited to the number of masters (groups), structurally bypassing N>10K
nearest-neighbor search bottlenecks (resolves SUMMARY.md open item #2).
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import torch

from src.cache.base import CacheStore


@dataclass
class MasterEntry:
    """FP16 master KV entry for a group."""
    kv: torch.Tensor
    group_id: str


@dataclass
class DiffEntry:
    """Per-agent diff storage entry."""
    group_id: str
    agent_id: str
    # Only blocks where L2(master_block, agent_block) > diff_threshold are stored
    diff_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Block indices that are identical to master (stored as pointer reference only)
    master_ref_blocks: Set[int] = field(default_factory=set)


class DiffAwareSegmentStore(CacheStore):
    """Agent-group master + block-sparse diff KV cache (Activity B-1).

    NOTE: FAISS not used. Search space = number of groups (masters).
    This structurally bypasses N>10K search bottleneck.
    """

    def __init__(
        self,
        block_size: int = 64,
        diff_threshold: float = 0.1,
        max_groups: int = 100,
    ) -> None:
        self.block_size = block_size
        self.diff_threshold = diff_threshold
        self.max_groups = max_groups
        # LRU ordering: OrderedDict[group_id, MasterEntry]
        self._masters: OrderedDict[str, MasterEntry] = OrderedDict()
        # Per-group per-agent diffs: {group_id: {agent_id: DiffEntry}}
        self._diffs: Dict[str, Dict[str, DiffEntry]] = {}
        self._hits = 0
        self._misses = 0
        self._diff_hits = 0
        self._master_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Route by key format:
        - "master:{group_id}"             → register_master(value, group_id)
        - "agent:{group_id}:{agent_id}"   → put_agent_kv(agent_id, group_id, value)
        - other                           → register_master(value, group_id=key)
        """
        if key.startswith("master:"):
            group_id = key[len("master:"):]
            self.register_master(value, group_id)
        elif key.startswith("agent:"):
            parts = key.split(":", 2)
            if len(parts) == 3:
                _, group_id, agent_id = parts
                self.put_agent_kv(agent_id, group_id, value)
            else:
                self.register_master(value, key)
        else:
            self.register_master(value, key)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Route by key format:
        - "master:{group_id}"           → return master KV
        - "agent:{group_id}:{agent_id}" → return reconstructed agent KV
        """
        if key.startswith("master:"):
            group_id = key[len("master:"):]
            result = self._get_master_kv(group_id)
            if result is not None:
                self._hits += 1
                self._master_hits += 1
            else:
                self._misses += 1
            return result
        elif key.startswith("agent:"):
            parts = key.split(":", 2)
            if len(parts) == 3:
                _, group_id, agent_id = parts
                result = self.get_agent_kv(agent_id, group_id)
                if result is not None:
                    self._hits += 1
                    self._diff_hits += 1
                else:
                    self._misses += 1
                return result
        self._misses += 1
        return None

    def evict(self) -> int:
        """Evict the oldest LRU group (master + all agent diffs).

        Returns number of bytes freed.
        """
        if not self._masters:
            return 0
        oldest_group_id = next(iter(self._masters))
        return self._evict_group(oldest_group_id)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Sum of master KV bytes + all agent diff block bytes."""
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

    # ------------------------------------------------------------------ #
    # Extended API                                                         #
    # ------------------------------------------------------------------ #

    def register_master(self, master_kv: torch.Tensor, group_id: str) -> None:
        """Register the shared KV for a group as master.

        Evicts oldest LRU group if max_groups is exceeded.
        """
        if len(self._masters) >= self.max_groups and group_id not in self._masters:
            self.evict()
        self._masters[group_id] = MasterEntry(
            kv=master_kv.detach().clone(),
            group_id=group_id,
        )
        self._masters.move_to_end(group_id)
        if group_id not in self._diffs:
            self._diffs[group_id] = {}

    def put_agent_kv(
        self,
        agent_id: str,
        group_id: str,
        agent_kv: torch.Tensor,
    ) -> None:
        """Store only the diff blocks between agent KV and master KV.

        Blocks where L2(master, agent) <= diff_threshold: store master ref pointer only.
        Blocks where L2(master, agent) > diff_threshold: store diff tensor (FP16).
        """
        master_kv = self._get_master_kv(group_id)
        if master_kv is None:
            # No master for this group — register as master
            self.register_master(agent_kv, group_id)
            return

        agent_kv_f = agent_kv.float()
        master_kv_f = master_kv.float()

        # Flatten on all dimensions except the last two (seq_len, d_head)
        # Operate on flattened token dim (last-but-one axis)
        # Shape: (..., seq_len, d_head) — block along seq_len dimension
        if agent_kv_f.dim() >= 2:
            seq_len = agent_kv_f.shape[-2]
        else:
            seq_len = agent_kv_f.numel()

        n_blocks = max(1, (seq_len + self.block_size - 1) // self.block_size)

        diff_entry = DiffEntry(group_id=group_id, agent_id=agent_id)

        for block_idx in range(n_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, seq_len)

            if agent_kv_f.dim() >= 2:
                master_block = master_kv_f[..., start:end, :]
                agent_block = agent_kv_f[..., start:end, :]
            else:
                master_block = master_kv_f[start:end]
                agent_block = agent_kv_f[start:end]

            l2_dist = (master_block - agent_block).norm().item()

            if l2_dist > self.diff_threshold:
                # Store the diff block in FP16 (original precision, not compressed)
                diff = (agent_block - master_block).to(torch.float16)
                diff_entry.diff_blocks[block_idx] = diff
            else:
                diff_entry.master_ref_blocks.add(block_idx)

        if group_id not in self._diffs:
            self._diffs[group_id] = {}
        self._diffs[group_id][agent_id] = diff_entry

    def get_agent_kv(
        self,
        agent_id: str,
        group_id: str,
    ) -> Optional[torch.Tensor]:
        """Reconstruct agent KV from master + stored diff blocks.

        Returns None if group or agent not found.
        """
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

        for block_idx, diff_block in diff_entry.diff_blocks.items():
            start = block_idx * self.block_size
            end = min(start + self.block_size, seq_len)
            if result.dim() >= 2:
                result[..., start:end, :] = (
                    result[..., start:end, :] + diff_block.float()
                )
            else:
                result[start:end] = result[start:end] + diff_block.float()

        return result.half()

    def diff_hit_stats(self) -> dict:
        """Non-contiguous hit breakdown statistics.

        Returns dict with keys:
            diff_hit_rate, master_hit_rate, overall_hit_rate,
            n_groups, search_space_reduction
        """
        total_hits = self._diff_hits + self._master_hits
        diff_hit_rate = self._diff_hits / total_hits if total_hits > 0 else 0.0
        master_hit_rate = self._master_hits / total_hits if total_hits > 0 else 0.0
        overall = self.hit_rate()
        n_groups = len(self._masters)
        # Baseline assumes one segment per token — groups are much fewer
        total_agents = sum(len(d) for d in self._diffs.values())
        baseline_segments = max(n_groups + total_agents, 1)
        search_space_reduction = baseline_segments / max(n_groups, 1)

        return {
            "diff_hit_rate": diff_hit_rate,
            "master_hit_rate": master_hit_rate,
            "overall_hit_rate": overall,
            "n_groups": n_groups,
            "search_space_reduction": search_space_reduction,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_master_kv(self, group_id: str) -> Optional[torch.Tensor]:
        """Return master KV tensor. Overridden by CompressedDiffStore for INT4 decode."""
        entry = self._masters.get(group_id)
        if entry is None:
            return None
        # Accessing updates LRU order
        self._masters.move_to_end(group_id)
        return entry.kv

    def _register_lru_only(self, group_id: str) -> None:
        """Register a dummy MasterEntry for LRU tracking only (used by CompressedDiffStore)."""
        if len(self._masters) >= self.max_groups and group_id not in self._masters:
            oldest = next(iter(self._masters))
            self._evict_group(oldest)
        # Store a zero-byte sentinel; CompressedDiffStore holds the real data separately
        sentinel = MasterEntry(kv=torch.empty(0), group_id=group_id)
        self._masters[group_id] = sentinel
        self._masters.move_to_end(group_id)
        if group_id not in self._diffs:
            self._diffs[group_id] = {}

    def _evict_group(self, group_id: str) -> int:
        """Evict a specific group: master + all agent diffs. Returns bytes freed."""
        freed = 0
        if group_id in self._masters:
            freed += self._masters.pop(group_id).kv.nbytes
        if group_id in self._diffs:
            for diff_entry in self._diffs.pop(group_id).values():
                freed += sum(b.nbytes for b in diff_entry.diff_blocks.values())
        return freed
