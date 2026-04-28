"""Activity C: Mixed-precision KV cache quantization codec.

Port of src/cache/compression.py for use inside vLLM 0.20.0.

Design:
- Early layers (layer_idx < cutoff) are kept in FP16 — they carry the most
  critical information and FP16 still halves the FP32 footprint.
- Later layers use symmetric per-tensor INT8 quantization (~75 % savings vs
  FP32).  Scale factors are stored so decode is lossless given the stored
  scale.

This module is intentionally dependency-free (no vLLM imports) so it can be
imported before the vLLM engine is initialised.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class CompressionCodec:
    """Mixed-precision KV cache codec (Activity C).

    Args:
        num_layers:    Total number of transformer layers in the model.
        cutoff_ratio:  Fraction of layers kept in FP16 (default 1/3).
                       Remaining layers are quantized to INT8.
    """

    def __init__(self, num_layers: int, cutoff_ratio: float = 1 / 3) -> None:
        self.num_layers = num_layers
        self.cutoff = max(1, int(num_layers * cutoff_ratio))
        # Stores INT8 scale factors indexed by (layer_idx, tensor_id).
        # tensor_id disambiguates K vs V tensors within the same layer.
        self._scales: Dict[Tuple[int, int], float] = {}

    # ---------------------------------------------------------------------- #
    # Public encode / decode API                                               #
    # ---------------------------------------------------------------------- #

    def encode(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Compress a KV tensor for storage.

        Args:
            kv:        Input tensor in any floating-point dtype.
            layer_idx: Index of the transformer layer this KV belongs to.
            tensor_id: 0 for K, 1 for V (or any integer to distinguish
                       multiple tensors stored for the same layer).

        Returns:
            FP16 tensor for early layers; INT8 tensor for later layers.
        """
        if layer_idx < self.cutoff:
            return kv.to(torch.float16)

        # Symmetric per-tensor INT8 quantization.
        abs_max = kv.abs().max().item()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        self._scales[(layer_idx, tensor_id)] = scale
        quantized = (
            kv.float().div(scale).round().clamp(-128, 127).to(torch.int8)
        )
        return quantized

    def decode(
        self,
        compressed: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Decompress a stored KV tensor back to float32.

        Args:
            compressed: Tensor returned by :meth:`encode`.
            layer_idx:  Matching layer index used during encode.
            tensor_id:  Matching tensor_id used during encode.

        Returns:
            Float32 tensor ready for attention computation.
        """
        if layer_idx < self.cutoff:
            return compressed.to(torch.float32)

        scale = self._scales.get((layer_idx, tensor_id), 1.0)
        return compressed.to(torch.float32).mul(scale)

    # ---------------------------------------------------------------------- #
    # Utility                                                                  #
    # ---------------------------------------------------------------------- #

    def compression_ratio(self, layer_idx: int) -> float:
        """Bytes saved relative to FP32 for a single layer.

        Returns:
            0.5 for FP16 layers (2 bytes vs 4 bytes).
            0.75 for INT8 layers (1 byte vs 4 bytes).
        """
        if layer_idx < self.cutoff:
            return 0.5
        return 0.75

    def average_compression_ratio(self) -> float:
        """Weighted-average compression ratio across all layers."""
        early = self.cutoff * 0.5
        late = (self.num_layers - self.cutoff) * 0.75
        return (early + late) / self.num_layers

    def __repr__(self) -> str:
        return (
            f"CompressionCodec(num_layers={self.num_layers}, "
            f"cutoff={self.cutoff}, "
            f"avg_ratio={self.average_compression_ratio():.2f})"
        )
