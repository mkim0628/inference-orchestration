"""NQKVCodec — Normal Float INT4 block-quantile quantization codec (Activity C-1).

No matrix transformation overhead. Only (mu, sigma) FP16 scalars per block stored
for dequantization. Drop-in replacement for CompressionCodec interface.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# Normal Float INT4 quantile representative values (14 values → 16 levels).
# Derived from equally-spaced quantiles of the standard normal distribution.
# 14 representative values for the 14 usable quantile mid-points
# (two extreme quantiles: indices 0 and 13 as tail representatives).
NF4_VALUES: List[float] = [
    -1.9951,
    -1.4731,
    -1.0607,
    -0.7589,
    -0.5165,
    -0.2892,
    -0.0941,
     0.0941,
     0.2892,
     0.5165,
     0.7589,
     1.0607,
     1.4731,
     1.9951,
]

NF4_QUANTILES: List[float] = [
    -1.7408,
    -1.2315,
    -0.9004,
    -0.6340,
    -0.4001,
    -0.1876,
     0.0000,
     0.1876,
     0.4001,
     0.6340,
     0.9004,
     1.2315,
     1.7408,
]


class NQKVCodec:
    """Normal Float INT4 block-quantile quantization codec (Activity C-1).

    No matrix transformation. Per-block (mu, sigma) FP16 scalars for inverse quantization.
    Implements a CompressionCodec-compatible interface.
    Training-free: no nn.Parameter or nn.Module.
    """

    def __init__(
        self,
        block_size: int = 64,
        nf4_values: Optional[List[float]] = None,
    ) -> None:
        self.block_size = block_size
        self.nf4_values = torch.tensor(
            nf4_values if nf4_values is not None else NF4_VALUES,
            dtype=torch.float32,
        )

    def encode(
        self,
        kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode FP16/FP32 KV tensor to (indices_uint8, mu_fp16, sigma_fp16).

        Args:
            kv: Tensor of any shape. Total elements padded to block_size multiple.

        Returns:
            indices: uint8 tensor, shape=(num_blocks, block_size), values in [0, 13]
            mu:      float16 tensor, shape=(num_blocks,) — per-block mean
            sigma:   float16 tensor, shape=(num_blocks,) — per-block std dev
        """
        original_shape = kv.shape
        flat = kv.float().reshape(-1)
        n_elem = flat.numel()
        num_blocks = (n_elem + self.block_size - 1) // self.block_size
        pad_len = num_blocks * self.block_size - n_elem

        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len))

        blocks = flat.reshape(num_blocks, self.block_size)

        mu = blocks.mean(dim=-1)
        sigma = blocks.std(dim=-1).clamp(min=1e-8)

        normalized = (blocks - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)

        # Find closest NF4 representative for each normalized value
        # normalized: (num_blocks, block_size), nf4: (14,)
        nf4 = self.nf4_values.to(kv.device)
        diff = (normalized.unsqueeze(-1) - nf4.view(1, 1, -1)).abs()
        indices = diff.argmin(dim=-1).to(torch.uint8)  # (num_blocks, block_size)

        return indices, mu.half(), sigma.half()

    def decode(
        self,
        indices: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """Decode (indices_uint8, mu_fp16, sigma_fp16) back to FP16 KV tensor.

        Args:
            indices:        uint8 tensor, shape=(num_blocks, block_size)
            mu:             float16 tensor, shape=(num_blocks,)
            sigma:          float16 tensor, shape=(num_blocks,)
            original_shape: if provided, output is trimmed and reshaped to this

        Returns:
            Restored FP16 tensor.
        """
        nf4 = self.nf4_values.to(indices.device)
        reconstructed = nf4[indices.long().clamp(0, len(nf4) - 1)]  # (num_blocks, block_size)

        mu_f = mu.float().unsqueeze(-1)
        sigma_f = sigma.float().unsqueeze(-1)
        restored = reconstructed * sigma_f + mu_f  # (num_blocks, block_size)

        flat = restored.reshape(-1)

        if original_shape is not None:
            n_elem = 1
            for d in original_shape:
                n_elem *= d
            flat = flat[:n_elem]
            return flat.reshape(original_shape).half()

        return flat.half()

    def compression_ratio(self, kv: torch.Tensor) -> float:
        """Theoretical compression ratio vs FP16 baseline.

        INT4 = 0.5 bytes/elem + 2 FP16 scalars (4 bytes) per block overhead.
        """
        n_elem = kv.numel()
        n_blocks = (n_elem + self.block_size - 1) // self.block_size
        # Each block: block_size * 0.5 bytes (4-bit packing) + 4 bytes (mu+sigma FP16)
        compressed_bytes = n_blocks * self.block_size * 0.5 + n_blocks * 4
        original_bytes = n_elem * 2  # FP16
        return original_bytes / compressed_bytes
