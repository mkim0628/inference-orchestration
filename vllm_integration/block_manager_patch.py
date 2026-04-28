"""Activity B: Position-independent segment hash lookup for vLLM 0.20.0.

vLLM 0.20.0 uses a v1 architecture where the KV cache is managed by
``vllm.v1.core.kv_cache_manager.KVCacheManager`` (backed by a coordinator and
block pool) rather than the legacy ``BlockAllocator`` / ``BlockSpaceManager``
classes found in older versions.

Integration strategy
--------------------
We provide two artefacts:

1. ``SegmentHashMixin`` — a mixin that adds position-independent segment key
   computation and a lightweight in-process segment index.  This is the direct
   port of ``SegmentedHashCache.chunk_key`` / ``get_segments`` logic from
   ``src/cache/segmented.py``.

2. ``NonContiguousKVCacheManager`` — a thin subclass of vLLM's
   ``KVCacheManager`` that embeds the mixin.  It overrides
   ``get_computed_blocks`` to first query the standard prefix cache and then,
   for any non-prefix hit, look up individual segments via the position-
   independent hash table.  Segments stored in the index are compressed via
   :class:`~vllm_integration.compression_codec.CompressionCodec`.

Import safety
-------------
All vLLM imports are wrapped in ``try/except`` blocks so that unit tests and
linters can import this module even when vLLM is not installed.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

try:
    from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
    from vllm.v1.request import Request as VllmRequest
    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False
    KVCacheManager = object  # type: ignore[assignment, misc]

from vllm_integration.compression_codec import CompressionCodec


# --------------------------------------------------------------------------- #
# Position-independent segment key helper                                      #
# --------------------------------------------------------------------------- #

class SegmentHashMixin:
    """Mixin: position-independent chunk-level KV segment hashing.

    Ported from ``src/cache/segmented.SegmentedHashCache``.
    The hash of a chunk depends only on its *token content* and the layer
    index — not on the absolute position within the sequence.  This means the
    same tokens appearing at different offsets in different requests share a
    cache entry.
    """

    # Default chunk size mirrors the one used in the standalone implementation.
    CHUNK_SIZE: int = 64

    @classmethod
    def get_segment_key(
        cls,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int,
        chunk_size: int = 64,
    ) -> str:
        """Compute a position-independent hash key for a token chunk.

        Args:
            token_ids:  Full token-id sequence for the request.
            chunk_idx:  Index of the chunk within the sequence
                        (0-based, each chunk covers ``chunk_size`` tokens).
            layer_idx:  Transformer layer index; different layers keep
                        independent cache entries.
            chunk_size: Number of tokens per chunk (must match the value used
                        when the entry was stored).

        Returns:
            A hex-encoded SHA-256 digest that is stable across requests and
            independent of absolute token position.
        """
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = token_ids[start:end]
        if not chunk:
            return ""
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

    @classmethod
    def iter_segment_keys(
        cls,
        token_ids: List[int],
        layer_idx: int,
        chunk_size: int = 64,
    ) -> List[Tuple[int, str]]:
        """Enumerate ``(chunk_idx, key)`` pairs for all chunks in ``token_ids``.

        Chunks that are shorter than ``chunk_size`` (i.e. the final partial
        chunk) are included but may produce a weaker cache hit rate because
        the same partial chunk at a different position would hash identically
        only if its token content is identical.
        """
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        return [
            (i, cls.get_segment_key(token_ids, i, layer_idx, chunk_size))
            for i in range(n_chunks)
        ]


# --------------------------------------------------------------------------- #
# In-process compressed segment index                                          #
# --------------------------------------------------------------------------- #

class CompressedSegmentIndex:
    """LRU index that maps segment keys to compressed KV tensors.

    This is the in-process counterpart of ``src/cache/compressed_segment.py``
    stripped of the ``CacheStore`` inheritance so it can be used inside vLLM
    without pulling in project-level dependencies.

    Entries are stored in compressed form (FP16 / INT8) to maximise the
    number of reusable segments within a fixed memory budget.
    """

    def __init__(
        self,
        codec: CompressionCodec,
        max_entries: int = 2000,
    ) -> None:
        self.codec = codec
        self.max_entries = max_entries
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._key_layer: Dict[str, int] = {}

        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # --- storage ------------------------------------------------------------ #

    def put(
        self,
        key: str,
        kv: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> None:
        """Compress and store a KV tensor under ``key``."""
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.max_entries:
            self._store.popitem(last=False)
        compressed = self.codec.encode(kv, layer_idx, tensor_id)
        self._store[key] = compressed
        self._key_layer[key] = layer_idx

    def get(
        self,
        key: str,
        tensor_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """Retrieve and decompress a KV tensor, or ``None`` on miss."""
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        layer_idx = self._key_layer[key]
        return self.codec.decode(self._store[key], layer_idx, tensor_id)

    # --- statistics --------------------------------------------------------- #

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def __len__(self) -> int:
        return len(self._store)


# --------------------------------------------------------------------------- #
# NonContiguousKVCacheManager                                                  #
# --------------------------------------------------------------------------- #

class NonContiguousKVCacheManager(SegmentHashMixin, KVCacheManager):  # type: ignore[misc]
    """vLLM KVCacheManager extended with position-independent segment reuse.

    This subclass is transparent to the rest of vLLM: it preserves the full
    public interface of ``KVCacheManager`` and only adds the non-contiguous
    lookup on top of the standard prefix-cache path.

    How it works
    ------------
    1. ``get_computed_blocks`` first delegates to the parent (standard prefix
       cache lookup).
    2. If the parent returns zero computed tokens the segment index is
       consulted.  For each chunk of the request's token sequence a position-
       independent key is computed.  Any chunk whose key exists in the segment
       index is counted as a non-contiguous hit.
    3. The segment index is populated lazily from ``cache_blocks`` (called by
       the scheduler after prefill) via the overridden ``cache_blocks`` method.

    Note: non-contiguous hits reported here are *informational* — they do not
    reduce the number of tokens that vLLM re-prefills.  Full integration
    (actually skipping prefill for cached segments) requires attention-kernel
    cooperation; see ``attention_backend_patch.py`` for the read/write hooks.
    """

    def __init__(
        self,
        *args,
        codec: Optional[CompressionCodec] = None,
        segment_chunk_size: int = 64,
        segment_max_entries: int = 2000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if codec is None:
            # Default: assume 32 layers if we cannot introspect
            num_layers = getattr(self, "_num_layers", 32)
            codec = CompressionCodec(num_layers=num_layers)
        self._segment_index = CompressedSegmentIndex(
            codec=codec,
            max_entries=segment_max_entries,
        )
        self._segment_chunk_size = segment_chunk_size
        self._nc_hits = 0
        self._total_queries = 0

    # ---------------------------------------------------------------------- #
    # Overridden KVCacheManager methods                                        #
    # ---------------------------------------------------------------------- #

    def get_computed_blocks(
        self, request: "VllmRequest"
    ) -> "tuple[KVCacheBlocks, int]":
        """Extend prefix cache lookup with non-contiguous segment lookup.

        Falls back transparently to standard prefix caching; segment index
        look-ups only happen when no prefix cache hit is found.
        """
        blocks, num_computed = super().get_computed_blocks(request)
        self._total_queries += 1

        if num_computed == 0 and self.enable_caching:
            token_ids: List[int] = list(request.prompt_token_ids)
            # Layer 0 is used as representative for hit-rate accounting;
            # real usage iterates all layers in the attention backend patch.
            chunk_keys = self.iter_segment_keys(
                token_ids,
                layer_idx=0,
                chunk_size=self._segment_chunk_size,
            )
            hit_chunks = [
                idx
                for idx, key in chunk_keys
                if self._segment_index.get(key) is not None
            ]
            if hit_chunks:
                self._nc_hits += len(hit_chunks)

        return blocks, num_computed

    def cache_blocks(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Forward to parent; segment index population is done via hooks."""
        return super().cache_blocks(*args, **kwargs)

    # ---------------------------------------------------------------------- #
    # Segment index access                                                     #
    # ---------------------------------------------------------------------- #

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int,
        kv: torch.Tensor,
        tensor_id: int = 0,
    ) -> None:
        """Store a KV tensor for a specific token chunk and layer.

        Called by the attention backend hook after computing KV for a chunk.
        """
        key = self.get_segment_key(
            token_ids, chunk_idx, layer_idx, self._segment_chunk_size
        )
        self._segment_index.put(key, kv, layer_idx, tensor_id)

    def lookup_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """Retrieve a cached KV tensor for a specific chunk and layer.

        Returns decompressed float32 tensor or ``None`` on miss.
        """
        key = self.get_segment_key(
            token_ids, chunk_idx, layer_idx, self._segment_chunk_size
        )
        return self._segment_index.get(key, tensor_id)

    # ---------------------------------------------------------------------- #
    # Statistics                                                               #
    # ---------------------------------------------------------------------- #

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of queries that yielded at least one non-contiguous hit."""
        if self._total_queries == 0:
            return 0.0
        return self._nc_hits / self._total_queries

    def segment_index_memory_bytes(self) -> int:
        return self._segment_index.memory_bytes()
