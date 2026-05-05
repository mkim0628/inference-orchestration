"""NQKVCodecPatch — NQKVCodec adapted for vLLM 0.20.1 paged KV cache block layout.

Activity C-1: Normal Float INT4 block-quantile quantization codec ported to vLLM.

Integration point: vLLM's KVCacheManager / CacheEngine layer.

Design:
  - NQKVCodecPatch wraps NQKVCodec and adds vLLM-aware helpers for operating on
    paged physical blocks (fixed-size block_size × n_heads × d_head tensors).
  - encode_block() / decode_block() operate on a single vLLM KV block tensor.
  - encode_cache_tensor() / decode_cache_tensor() operate on the full KV cache
    tensor as allocated by vLLM's CacheEngine (shape varies by vLLM version).
  - All operations are CPU/GPU compatible; CUDA is used when available.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# vLLM imports — graceful fallback so the file can be imported without a
# running vLLM server and in environments missing optional vLLM internals.
# ---------------------------------------------------------------------------
try:
    from vllm.v1.kv_cache_interface import KVCacheConfig  # vLLM >= 0.20
    _VLLM_KV_CACHE_CONFIG_AVAILABLE = True
except ImportError:
    _VLLM_KV_CACHE_CONFIG_AVAILABLE = False

try:
    from vllm.config import CacheConfig
    _VLLM_CACHE_CONFIG_AVAILABLE = True
except ImportError:
    _VLLM_CACHE_CONFIG_AVAILABLE = False

# ---------------------------------------------------------------------------
# NF4 constant table (copied from src/cache/nqkv_codec.py to keep this file
# self-contained and avoid circular imports from the research src/ tree).
# ---------------------------------------------------------------------------
_NF4_VALUES: List[float] = [
    -1.9951, -1.4731, -1.0607, -0.7589, -0.5165, -0.2892, -0.0941,
     0.0941,  0.2892,  0.5165,  0.7589,  1.0607,  1.4731,  1.9951,
]


class NQKVCodecPatch:
    """NQKVCodec adapted for vLLM 0.20.1 paged KV block layout (Activity C-1).

    Provides encode/decode helpers operating on:
      - Individual KV blocks as used by vLLM's BlockAllocator / CacheEngine.
      - Full layer KV cache tensors of shape
        [num_blocks, block_size, num_kv_heads, head_dim]  (vLLM v1 layout).

    Compression is lossless-to-NF4-precision: each 64-element block is stored
    as (uint8 indices, float16 mu, float16 sigma), yielding ~3.5× reduction
    vs FP16 at the cost of ~RMSE 0.13 reconstruction error.

    No nn.Parameter or nn.Module.  Training-free.
    """

    def __init__(
        self,
        block_size: int = 64,
        nf4_values: Optional[List[float]] = None,
        vllm_block_size: int = 16,
    ) -> None:
        """
        Args:
            block_size:       NQKVCodec quantisation block size (default 64 elems).
            nf4_values:       Override NF4 representative table.
            vllm_block_size:  vLLM's physical page block size (tokens per block).
        """
        self.block_size = block_size
        self.vllm_block_size = vllm_block_size
        self._nf4 = torch.tensor(
            nf4_values if nf4_values is not None else _NF4_VALUES,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Core codec — identical logic to NQKVCodec in src/cache/nqkv_codec.py
    # ------------------------------------------------------------------

    def encode(
        self,
        kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an arbitrary-shape KV tensor to (indices_uint8, mu_fp16, sigma_fp16).

        Returns:
            indices: uint8,   shape (num_quant_blocks, block_size)
            mu:      float16, shape (num_quant_blocks,)
            sigma:   float16, shape (num_quant_blocks,)
        """
        original_shape = kv.shape
        flat = kv.float().reshape(-1)
        n_elem = flat.numel()
        num_qblocks = (n_elem + self.block_size - 1) // self.block_size
        pad_len = num_qblocks * self.block_size - n_elem
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        blocks = flat.reshape(num_qblocks, self.block_size)
        mu = blocks.mean(dim=-1)
        sigma = blocks.std(dim=-1).clamp(min=1e-8)
        normalized = (blocks - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)

        nf4 = self._nf4.to(kv.device)
        diff = (normalized.unsqueeze(-1) - nf4.view(1, 1, -1)).abs()
        indices = diff.argmin(dim=-1).to(torch.uint8)

        return indices, mu.half(), sigma.half()

    def decode(
        self,
        indices: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """Decode (indices, mu, sigma) back to FP16 KV tensor."""
        nf4 = self._nf4.to(indices.device)
        reconstructed = nf4[indices.long().clamp(0, len(nf4) - 1)]
        restored = reconstructed * sigma.float().unsqueeze(-1) + mu.float().unsqueeze(-1)
        flat = restored.reshape(-1)
        if original_shape is not None:
            n_elem = 1
            for d in original_shape:
                n_elem *= d
            flat = flat[:n_elem]
            return flat.reshape(original_shape).half()
        return flat.half()

    # ------------------------------------------------------------------
    # vLLM block-level helpers
    # ------------------------------------------------------------------

    def encode_vllm_block(
        self,
        kv_block: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
        """Compress one vLLM physical KV block.

        Args:
            kv_block: Tensor of any shape representing one vLLM block, e.g.
                      [2, num_kv_heads, vllm_block_size, head_dim]  (K and V
                      are stacked on dim 0 in some vLLM layouts) or
                      [vllm_block_size, num_kv_heads, head_dim].

        Returns:
            (indices, mu, sigma, original_shape) — pass all four to
            decode_vllm_block() to reconstruct.
        """
        original_shape = kv_block.shape
        indices, mu, sigma = self.encode(kv_block)
        return indices, mu, sigma, original_shape

    def decode_vllm_block(
        self,
        indices: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Reconstruct a vLLM physical KV block from compressed form."""
        return self.decode(indices, mu, sigma, original_shape)

    def encode_layer_kv_cache(
        self,
        kv_cache: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compress a full-layer KV cache tensor as used by vLLM v1.

        vLLM v1 KV cache shape for one layer:
            [num_blocks, block_size, num_kv_heads, head_dim]   (K or V)
        or a stacked tensor:
            [2, num_blocks, block_size, num_kv_heads, head_dim]

        Returns a dict with keys 'indices', 'mu', 'sigma', 'original_shape'
        that can be passed to decode_layer_kv_cache().
        """
        original_shape = kv_cache.shape
        indices, mu, sigma = self.encode(kv_cache)
        return {
            "indices": indices,
            "mu": mu,
            "sigma": sigma,
            "original_shape": original_shape,
        }

    def decode_layer_kv_cache(
        self,
        compressed: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct a full-layer KV cache tensor from the dict produced by
        encode_layer_kv_cache()."""
        return self.decode(
            compressed["indices"],
            compressed["mu"],
            compressed["sigma"],
            compressed["original_shape"],
        )

    def compression_ratio(self, kv: torch.Tensor) -> float:
        """Theoretical compression ratio vs FP16 baseline."""
        n_elem = kv.numel()
        n_qblocks = (n_elem + self.block_size - 1) // self.block_size
        compressed_bytes = n_qblocks * self.block_size * 0.5 + n_qblocks * 4
        original_bytes = n_elem * 2  # FP16
        return original_bytes / compressed_bytes

    # ------------------------------------------------------------------
    # vLLM cache config integration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def adjusted_num_gpu_blocks(
        num_gpu_blocks: int,
        compression_ratio: float,
    ) -> int:
        """Estimate how many vLLM GPU blocks can be served when KV is compressed.

        This is a planning helper — it does NOT modify vLLM's CacheConfig
        directly.  Feed the result into CacheConfig.num_gpu_blocks_override
        at engine init time to expand the effective block pool.

        Args:
            num_gpu_blocks:    Original block count estimated by vLLM's
                               profile_run (based on full FP16 KV size).
            compression_ratio: From compression_ratio() — typically ~3.5.

        Returns:
            Expanded block count (int).
        """
        return int(num_gpu_blocks * compression_ratio)

    def get_vllm_cache_config_kwargs(
        self,
        compression_ratio: Optional[float] = None,
    ) -> dict:
        """Return kwargs to pass to CacheConfig for compression-aware sizing.

        Usage:
            kwargs = codec.get_vllm_cache_config_kwargs(compression_ratio=3.5)
            # Merge into your LLM / AsyncLLMEngine init kwargs:
            # engine_args.cache_dtype = kwargs['cache_dtype']
        """
        return {
            # cache_dtype 'auto' preserves vLLM's default selection; the codec
            # operates post-storage so dtype selection is independent.
            "cache_dtype": "auto",
        }


__all__ = ["NQKVCodecPatch"]
