"""CompressedDiffStore — DiffAwareSegmentStore + NQKVCodec integration (Cross-1).

Master KV is stored as INT4 (compressed via NQKVCodec).
Diff blocks remain FP16 (uncompressed) — small size makes compression overhead negligible.
Reconstruction: master INT4 dequantize + diff block merge.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from src.cache.diff_aware_store import DiffAwareSegmentStore
from src.cache.nqkv_codec import NQKVCodec


@dataclass
class CompressedMasterEntry:
    """INT4-compressed master KV storage entry."""
    indices: torch.Tensor      # (num_blocks, block_size) uint8
    mu: torch.Tensor           # (num_blocks,) float16
    sigma: torch.Tensor        # (num_blocks,) float16
    original_shape: torch.Size
    group_id: str


class CompressedDiffStore(DiffAwareSegmentStore):
    """DiffAwareSegmentStore with INT4-compressed masters (Cross-1).

    Master KV → NQKVCodec INT4 encoded
    Diff blocks → FP16 original (sparse)
    Retrieval → master INT4 decode + diff merge
    """

    def __init__(
        self,
        block_size: int = 64,
        diff_threshold: float = 0.1,
        max_groups: int = 100,
        codec_block_size: int = 64,
    ) -> None:
        super().__init__(
            block_size=block_size,
            diff_threshold=diff_threshold,
            max_groups=max_groups,
        )
        self.codec = NQKVCodec(block_size=codec_block_size)
        # Separate storage for compressed masters
        self._compressed_masters: dict = {}  # group_id → CompressedMasterEntry

    def register_master(self, master_kv: torch.Tensor, group_id: str) -> None:
        """Compress master KV with NQKVCodec (INT4) and store.

        LRU management is delegated to the parent's _masters OrderedDict
        via _register_lru_only.
        """
        # Encode master KV to INT4
        indices, mu, sigma = self.codec.encode(master_kv)
        self._compressed_masters[group_id] = CompressedMasterEntry(
            indices=indices,
            mu=mu,
            sigma=sigma,
            original_shape=master_kv.shape,
            group_id=group_id,
        )
        # Use parent LRU sentinel (no FP16 master stored in _masters)
        self._register_lru_only(group_id)

    def _get_master_kv(self, group_id: str) -> Optional[torch.Tensor]:
        """Dequantize INT4 master and return as FP16 (overrides parent)."""
        entry = self._compressed_masters.get(group_id)
        if entry is None:
            return None
        # Update LRU ordering in parent _masters
        if group_id in self._masters:
            self._masters.move_to_end(group_id)
        return self.codec.decode(
            entry.indices, entry.mu, entry.sigma, entry.original_shape
        )

    def evict(self) -> int:
        """Evict the oldest LRU group, including the compressed master."""
        if not self._masters:
            return 0
        oldest_group_id = next(iter(self._masters))
        return self._evict_group(oldest_group_id)

    def _evict_group(self, group_id: str) -> int:
        """Evict group: compressed master + all agent diffs. Returns bytes freed."""
        freed = 0
        # Remove LRU sentinel from parent _masters
        if group_id in self._masters:
            self._masters.pop(group_id)
        # Remove compressed master and count bytes
        if group_id in self._compressed_masters:
            cm = self._compressed_masters.pop(group_id)
            freed += cm.indices.nbytes + cm.mu.nbytes + cm.sigma.nbytes
        # Remove all agent diffs for this group
        if group_id in self._diffs:
            for diff_entry in self._diffs.pop(group_id).values():
                freed += sum(b.nbytes for b in diff_entry.diff_blocks.values())
        return freed

    def memory_bytes(self) -> int:
        """Sum of compressed master bytes + all agent diff block bytes."""
        total = 0
        for cm in self._compressed_masters.values():
            total += cm.indices.nbytes + cm.mu.nbytes + cm.sigma.nbytes
        for agent_diffs in self._diffs.values():
            for diff_entry in agent_diffs.values():
                total += sum(b.nbytes for b in diff_entry.diff_blocks.values())
        return total
