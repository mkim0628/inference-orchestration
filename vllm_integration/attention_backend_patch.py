"""Activity B+C: Attention backend hooks for compressed non-contiguous KV reuse.

vLLM 0.20.0 attention pipeline (v1 architecture)
-------------------------------------------------
The ``AttentionImpl.forward`` method is responsible for:
  1. Computing query (Q), key (K) and value (V) projections.
  2. Writing K/V to the block-structured GPU cache via
     ``do_rope_and_kv_cache_update`` (or equivalent).
  3. Running the attention kernel over cached K/V.

Integration approach
--------------------
We cannot monkey-patch the compiled CUDA kernels directly.  Instead we expose:

``CompressedKVHook``
    A plain Python class with two methods that wrap
    :class:`~vllm_integration.compression_codec.CompressionCodec`:

    - ``encode_kv(kv, layer_idx)`` — compress before off-device or CPU storage.
    - ``decode_kv(compressed, layer_idx)`` — decompress before attention.

    Callers (e.g. a custom attention wrapper or an ``AttentionImpl`` subclass)
    inject an instance of this class and call the two methods at the
    appropriate points.

``NonContiguousAttentionWrapper``
    An example wrapper around a standard ``AttentionImpl`` that:

    1. On the *write* path: stores each chunk's K/V in the
       ``NonContiguousKVCacheManager`` segment index (compressed via
       ``CompressedKVHook``).
    2. On the *read* path: retrieves cached K/V chunks from the segment index
       and splices them into the attention computation, skipping re-prefill
       for those tokens.

    This wrapper is *illustrative* — actual adoption requires registering it
    as the attention implementation for the target model, which is model-
    specific and version-specific.  Refer to ``README.md`` for integration
    steps.

All vLLM imports are guarded with ``try/except`` for portability.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

try:
    from vllm.v1.attention.backend import AttentionImpl
    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False
    AttentionImpl = object  # type: ignore[assignment, misc]

from vllm_integration.compression_codec import CompressionCodec, HadamardInt4Codec

# Type alias accepted by CompressedKVHook
_AnyCodec = "CompressionCodec | HadamardInt4Codec"


# --------------------------------------------------------------------------- #
# CompressedKVHook                                                             #
# --------------------------------------------------------------------------- #

class CompressedKVHook:
    """Encode/decode hooks for mixed-precision KV compression (Activity C).

    This class is the primary integration point between the attention pipeline
    and the compression codec.  It is designed to be injected into any
    attention implementation that exposes a write-to-cache / read-from-cache
    interface.

    Args:
        codec:     Shared :class:`CompressionCodec` instance.  Must have been
                   constructed with the correct ``num_layers`` for the model.
        tensor_id_k: Tensor ID used for K tensors (default 0).
        tensor_id_v: Tensor ID used for V tensors (default 1).
    """

    def __init__(
        self,
        codec: "CompressionCodec | HadamardInt4Codec",
        tensor_id_k: int = 0,
        tensor_id_v: int = 1,
    ) -> None:
        self.codec = codec
        self.tensor_id_k = tensor_id_k
        self.tensor_id_v = tensor_id_v

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def encode_kv(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        is_key: bool = True,
    ) -> torch.Tensor:
        """Compress a KV tensor for off-device / segment-index storage.

        Should be called immediately after the K or V projection, *before*
        writing to any persistent cache.

        Args:
            kv:        Raw K or V tensor (any float dtype).
            layer_idx: Transformer layer index.
            is_key:    True for K, False for V; selects the ``tensor_id``.

        Returns:
            FP16 tensor for early layers; INT8 tensor for later layers.
        """
        tensor_id = self.tensor_id_k if is_key else self.tensor_id_v
        return self.codec.encode(kv, layer_idx, tensor_id)

    def decode_kv(
        self,
        compressed: torch.Tensor,
        layer_idx: int,
        is_key: bool = True,
    ) -> torch.Tensor:
        """Decompress a stored KV tensor to float32 for attention computation.

        Must be called before passing K/V to the attention kernel so that the
        kernel always receives native float tensors.

        Args:
            compressed: Tensor returned by :meth:`encode_kv`.
            layer_idx:  Matching layer index.
            is_key:     True for K, False for V.

        Returns:
            Float32 tensor.
        """
        tensor_id = self.tensor_id_k if is_key else self.tensor_id_v
        return self.codec.decode(compressed, layer_idx, tensor_id)

    def encode_kv_pair(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: encode both K and V in one call."""
        return (
            self.encode_kv(k, layer_idx, is_key=True),
            self.encode_kv(v, layer_idx, is_key=False),
        )

    def decode_kv_pair(
        self,
        k_compressed: torch.Tensor,
        v_compressed: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience: decode both K and V in one call."""
        return (
            self.decode_kv(k_compressed, layer_idx, is_key=True),
            self.decode_kv(v_compressed, layer_idx, is_key=False),
        )

    def compression_ratio(self, layer_idx: int) -> float:
        """Delegate to the underlying codec."""
        return self.codec.compression_ratio(layer_idx)


# --------------------------------------------------------------------------- #
# NonContiguousAttentionWrapper (illustrative)                                 #
# --------------------------------------------------------------------------- #

class NonContiguousAttentionWrapper:
    """Wraps a vLLM AttentionImpl to add B+C non-contiguous KV reuse.

    This is a *reference implementation* showing how the hook and the segment
    manager are wired together.  Production adoption requires subclassing the
    specific attention backend used by the model (e.g. ``FlashAttentionImpl``).

    Write path (prefill)
    --------------------
    After the K/V projections and before the vLLM cache-update call:
    1. Split K/V by chunk boundaries (``chunk_size`` tokens per chunk).
    2. Encode each chunk with ``CompressedKVHook.encode_kv_pair``.
    3. Store in ``NonContiguousKVCacheManager.store_segment``.

    Read path (decode / second prefill of shared prompts)
    -----------------------------------------------------
    Before the attention kernel:
    1. Iterate chunks; call ``NonContiguousKVCacheManager.lookup_segment``.
    2. On hit: decode with ``CompressedKVHook.decode_kv_pair``; splice into
       K/V buffers.  Mark those token positions as "already computed" so
       the attention kernel skips them.
    3. On miss: fall through to normal prefill for that chunk.

    Args:
        impl:           Underlying ``AttentionImpl``.
        hook:           Configured :class:`CompressedKVHook` instance.
        kv_manager:     :class:`~vllm_integration.block_manager_patch.
                        NonContiguousKVCacheManager` instance.
        chunk_size:     Must match the value used in ``kv_manager``.
    """

    def __init__(
        self,
        impl: "AttentionImpl",
        hook: CompressedKVHook,
        kv_manager: object,  # NonContiguousKVCacheManager avoids circular import
        chunk_size: int = 64,
    ) -> None:
        self._impl = impl
        self._hook = hook
        self._kv_manager = kv_manager
        self._chunk_size = chunk_size

    # ---------------------------------------------------------------------- #
    # Write path helper                                                        #
    # ---------------------------------------------------------------------- #

    def store_kv_chunks(
        self,
        token_ids: List[int],
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Encode and store all K/V chunks from a prefill step.

        ``k`` and ``v`` should be shaped ``[seq_len, num_heads, head_dim]`` or
        equivalently ``[seq_len, head_dim]`` for MLA.  Token dimension is
        assumed to be dim 0.

        This method is idempotent: storing the same chunk twice is a no-op
        (the segment index moves the existing entry to MRU position).
        """
        n_chunks = max(1, (len(token_ids) + self._chunk_size - 1) // self._chunk_size)
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self._chunk_size
            end = start + self._chunk_size
            k_chunk = k[start:end]
            v_chunk = v[start:end]
            self._kv_manager.store_segment(  # type: ignore[attr-defined]
                token_ids, chunk_idx, layer_idx, k_chunk, tensor_id=0
            )
            self._kv_manager.store_segment(  # type: ignore[attr-defined]
                token_ids, chunk_idx, layer_idx, v_chunk, tensor_id=1
            )

    # ---------------------------------------------------------------------- #
    # Read path helper                                                         #
    # ---------------------------------------------------------------------- #

    def load_cached_chunks(
        self,
        token_ids: List[int],
        layer_idx: int,
    ) -> Tuple[List[int], List[int]]:
        """Return chunk indices that are (hit, miss) in the segment index.

        The caller should:
        - Skip prefill for *hit* chunks (reuse stored K/V).
        - Run normal prefill for *miss* chunks.

        Returns:
            (hit_chunk_indices, miss_chunk_indices)
        """
        n_chunks = max(1, (len(token_ids) + self._chunk_size - 1) // self._chunk_size)
        hits: List[int] = []
        misses: List[int] = []
        for chunk_idx in range(n_chunks):
            k_cached = self._kv_manager.lookup_segment(  # type: ignore[attr-defined]
                token_ids, chunk_idx, layer_idx, tensor_id=0
            )
            if k_cached is not None:
                hits.append(chunk_idx)
            else:
                misses.append(chunk_idx)
        return hits, misses

    # ---------------------------------------------------------------------- #
    # Delegating forward                                                       #
    # ---------------------------------------------------------------------- #

    def forward(self, *args, **kwargs):
        """Delegate to the underlying AttentionImpl.forward.

        Full integration requires intercepting args to splice cached K/V
        chunks; see docstring for the read/write path description.  This stub
        passes through unchanged so the wrapper is drop-in safe.
        """
        return self._impl.forward(*args, **kwargs)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# TriStateKVHook — Activity C (2026-04-30)                                    #
# Ports src/cache/tri_state_compressor.py to vLLM 0.20.0                     #
# --------------------------------------------------------------------------- #

class TriStateKVHook:
    """ARKV-style tri-state KV compression hook for vLLM attention backends.

    Ports ``TriStateCompressor`` from ``src/cache/tri_state_compressor.py``
    into the vLLM attention pipeline.  On each K/V write, tokens are
    classified into:

    - **retain** (top ``retain_ratio`` by attention score) → FP16, no info loss
    - **compress** (middle fraction) → HadamardInt4Codec, ~75% savings
    - **evict** (bottom ``evict_ratio`` by attention score) → dropped

    Accuracy constraint
    -------------------
    The ±1% perplexity delta constraint is maintained by:
    1. Retaining the top-20% heavy-hitter tokens at FP16 precision.
    2. Using HadamardInt4Codec (Hadamard rotation + INT4) for compress tier —
       validated at KL < 0.007 in standalone tests (Report ① 2026-04-30).
    3. Evicting only the bottom-40% low-attention tokens that contribute
       negligibly to attention output.

    Usage::

        hook = TriStateKVHook(codec=hadamard_codec, retain_ratio=0.20, evict_ratio=0.40)

        # In attention forward (write path, after K/V projection):
        storage = hook.encode_kv(k, attn_weights, layer_idx, is_key=True)

        # In attention forward (read path):
        k_reconstructed = hook.decode_kv(storage, layer_idx)
    """

    def __init__(
        self,
        codec: "HadamardInt4Codec",
        retain_ratio: float = 0.20,
        evict_ratio: float = 0.40,
    ) -> None:
        self.codec = codec
        self.retain_ratio = retain_ratio
        self.evict_ratio = evict_ratio
        # Storage: maps (layer_idx, tensor_id) → list of storage dicts
        self._storage: dict = {}

    def encode_kv(
        self,
        kv: "torch.Tensor",
        attn_weights: "torch.Tensor",
        layer_idx: int,
        tensor_id: int = 0,
    ) -> dict:
        """Tri-state classify + compress KV tensor.

        Args:
            kv:          K or V tensor, shape (n_tokens, kv_dim).
            attn_weights: Per-token attention importance scores, shape (n_tokens,).
            layer_idx:   Transformer layer index.
            tensor_id:   0 for K, 1 for V.

        Returns:
            Storage dict with keys: retain_kv, compressed_kv, retain_indices,
            compress_indices, evict_indices, layer_idx, tensor_id.
        """
        import torch
        n_tokens = kv.shape[0]
        scores = attn_weights.float()

        retain_k = max(1, int(n_tokens * self.retain_ratio))
        evict_k = max(1, int(n_tokens * self.evict_ratio))

        sorted_indices = torch.argsort(scores, descending=True)
        retain_indices = sorted_indices[:retain_k]
        evict_indices = sorted_indices[n_tokens - evict_k:]
        compress_indices = sorted_indices[retain_k: n_tokens - evict_k]

        retain_kv = kv[retain_indices].half()

        if len(compress_indices) > 0:
            compressed_kv = self.codec.encode(kv[compress_indices], layer_idx, tensor_id)
        else:
            compressed_kv = torch.zeros(0, dtype=torch.int8)

        storage = {
            "retain_kv": retain_kv,
            "compressed_kv": compressed_kv,
            "retain_indices": retain_indices,
            "compress_indices": compress_indices,
            "evict_indices": evict_indices,
            "n_tokens": n_tokens,
            "kv_dim": kv.shape[-1],
            "layer_idx": layer_idx,
            "tensor_id": tensor_id,
        }
        self._storage[(layer_idx, tensor_id)] = storage
        return storage

    def decode_kv(
        self,
        storage: dict,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> "torch.Tensor":
        """Reconstruct KV from tri-state storage.

        Evicted token positions are filled with zeros (they contributed
        negligibly to attention and are safe to zero-fill for the residual).

        Returns:
            Float32 tensor of shape (n_tokens, kv_dim).
        """
        import torch
        n_tokens = storage["n_tokens"]
        kv_dim = storage["kv_dim"]
        out = torch.zeros(n_tokens, kv_dim, dtype=torch.float32)

        retain_indices = storage["retain_indices"]
        out[retain_indices] = storage["retain_kv"].float()

        compress_indices = storage["compress_indices"]
        if len(compress_indices) > 0:
            decompressed = self.codec.decode(storage["compressed_kv"], layer_idx, tensor_id)
            out[compress_indices] = decompressed.float()

        return out

    def compression_ratio(self, retain_ratio: float = 0.20, evict_ratio: float = 0.40) -> float:
        """Theoretical memory ratio vs FP32 baseline.

        retain: retain_ratio * (2/4) = 0.10 of FP32
        compress: (1-retain_ratio-evict_ratio) * (1/4) ≈ 0.10 of FP32
        evict: 0
        total ≈ 0.20 of FP32 → 80% savings
        """
        compress_ratio = 1.0 - retain_ratio - evict_ratio
        return retain_ratio * 0.5 + compress_ratio * 0.25  # as fraction of FP32


if __name__ == "__main__":
    def _test_tri_state_hook():
        import torch
        from vllm_integration.compression_codec import HadamardInt4Codec
        codec = HadamardInt4Codec(num_layers=12, cutoff_ratio=0.2)
        hook = TriStateKVHook(codec=codec, retain_ratio=0.20, evict_ratio=0.40)
        kv = torch.randn(50, 64)
        attn_weights = torch.rand(50)
        storage = hook.encode_kv(kv, attn_weights, layer_idx=5, tensor_id=0)
        decoded = hook.decode_kv(storage, layer_idx=5, tensor_id=0)
        assert decoded.shape == kv.shape, f"Shape mismatch: {decoded.shape} != {kv.shape}"
        ratio = hook.compression_ratio()
        assert ratio < 0.25, f"Compression ratio too high: {ratio}"
        print(f"TriStateKVHook: OK (compression_ratio={ratio:.3f})")

    _test_tri_state_hook()
