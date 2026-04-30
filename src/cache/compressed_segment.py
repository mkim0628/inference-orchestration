from typing import TYPE_CHECKING, List, Optional, Tuple
import torch

from src.cache.segmented import SegmentedHashCache
from src.cache.compression import CompressionCodec

if TYPE_CHECKING:
    from src.cache.segment_adapter import SegmentAdapter


class CompressedSegmentCache(SegmentedHashCache):
    """Compressed non-contiguous segment cache (Activity B+C Cross-1).

    Stores KV segments in compressed form (FP16/INT8) to maximise the
    number of reusable segments within a fixed memory budget.

    Optionally accepts a SegmentAdapter that is applied to decompressed KV
    tensors on non-contiguous hits to reduce distribution shift.
    """

    def __init__(
        self,
        codec: CompressionCodec,
        chunk_size: int = 128,
        max_entries: int = 1000,
        adapter: Optional["SegmentAdapter"] = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, max_entries=max_entries)
        self.codec = codec
        self.adapter = adapter
        # Tracks layer_idx per stored key so decode can use the right precision
        self._key_layer: dict = {}

    def put_segment(  # type: ignore[override]
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Compress then store the KV segment (layer-scoped key)."""
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)
        tensor_id = hash(key) % (2**31)
        compressed = self.codec.encode(kv, layer_idx, tensor_id)
        self._key_layer[key] = layer_idx
        self.put(key, compressed)

    def get_segments(  # type: ignore[override]
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Retrieve and decompress cached segments for a given layer."""
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        miss_set_building: List[int] = []
        hit_pairs: List[Tuple[int, torch.Tensor]] = []

        for i in range(n_chunks):
            key = self.chunk_key(token_ids, i, layer_idx)
            compressed = self.get(key)
            if compressed is not None:
                tensor_id = hash(key) % (2**31)
                kv = self.codec.decode(compressed, layer_idx, tensor_id)
                hit_pairs.append((i, kv))
            else:
                miss_set_building.append(i)
                misses.append(i)

        miss_set = set(miss_set_building)
        for idx, kv in hit_pairs:
            is_noncontiguous = any(m < idx for m in miss_set)
            if is_noncontiguous:
                self._noncontiguous_hits += 1
                if self.adapter is not None:
                    with torch.no_grad():
                        kv = self.adapter(kv)
            hits.append((idx, kv))

        return hits, misses

    def memory_bytes(self) -> int:
        """Actual compressed storage size."""
        return sum(v.nbytes for v in self._store.values())

    def compression_ratio(self) -> float:
        """Average compression ratio across all stored segments."""
        return self.codec.average_compression_ratio()
