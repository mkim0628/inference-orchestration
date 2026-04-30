from typing import Dict, Tuple
import torch


class HadamardInt4Codec:
    """Hadamard rotation + INT4-range quantization codec (SAW-INT4 style).

    Early layers (< cutoff) are stored as FP16 to preserve critical information.
    Later layers are Hadamard-rotated then quantized to the INT4 range [-8, 7]
    and stored as int8 (one value per byte).  Hadamard rotation distributes
    outliers uniformly, enabling near-lossless INT4-range quantization.

    Memory savings vs FP32 baseline:
      FP16 early layers : −50%
      INT8-stored INT4  : −75%  (true bit-packing would give −87.5%)
    """

    def __init__(self, num_layers: int, cutoff_ratio: float = 0.2) -> None:
        self.num_layers = num_layers
        self.cutoff = max(1, int(num_layers * cutoff_ratio))
        # Per-row scales stored as tensors: key = (layer_idx, tensor_id) → Tensor shape (n, 1)
        self._scales: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hadamard_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Hadamard helpers                                                     #
    # ------------------------------------------------------------------ #

    def _next_power_of_two(self, n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    def _hadamard_matrix(self, dim: int) -> torch.Tensor:
        if dim in self._hadamard_cache:
            return self._hadamard_cache[dim]
        if dim == 1:
            h = torch.ones(1, 1, dtype=torch.float32)
        else:
            half = self._hadamard_matrix(dim // 2)
            top = torch.cat([half, half], dim=1)
            bot = torch.cat([half, -half], dim=1)
            h = torch.cat([top, bot], dim=0) / (2 ** 0.5)
        self._hadamard_cache[dim] = h
        return h

    def _apply_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        orig_dim = x.shape[-1]
        pad_dim = self._next_power_of_two(orig_dim)
        if pad_dim != orig_dim:
            pad = torch.zeros(*x.shape[:-1], pad_dim - orig_dim, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        H = self._hadamard_matrix(pad_dim).to(x.device)
        rotated = x @ H.T
        return rotated[..., :orig_dim]

    def _inverse_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        # Normalized Hadamard is orthogonal: H^{-1} = H^T = H
        orig_dim = x.shape[-1]
        pad_dim = self._next_power_of_two(orig_dim)
        if pad_dim != orig_dim:
            pad = torch.zeros(*x.shape[:-1], pad_dim - orig_dim, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        H = self._hadamard_matrix(pad_dim).to(x.device)
        restored = x @ H
        return restored[..., :orig_dim]

    # ------------------------------------------------------------------ #
    # CompressionCodec-compatible interface                               #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        kv_f = kv.float()
        if layer_idx < self.cutoff:
            return kv_f.half()
        rotated = self._apply_hadamard(kv_f)
        # Per-row quantization: each token row gets its own scale (SAW-INT4 approach).
        # This distributes quantization budget optimally across tokens.
        if rotated.dim() >= 2:
            scale = rotated.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / 7.0
        else:
            scale_val = rotated.abs().max().item()
            scale = torch.tensor([[scale_val / 7.0 if scale_val > 0 else 1.0]])
        self._scales[(layer_idx, tensor_id)] = scale.detach().clone()
        quantized = (rotated / scale).round().clamp(-8, 7).to(torch.int8)
        return quantized

    def decode(
        self,
        compressed: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        if layer_idx < self.cutoff:
            return compressed.float()
        scale = self._scales.get((layer_idx, tensor_id))
        if scale is None:
            scale = torch.ones(1, 1)
        dequantized = compressed.float() * scale.to(compressed.device)
        return self._inverse_hadamard(dequantized)

    def compression_ratio(self, layer_idx: int) -> float:
        return 0.5 if layer_idx < self.cutoff else 0.75

    def average_compression_ratio(self) -> float:
        early = self.cutoff * 0.5
        late = (self.num_layers - self.cutoff) * 0.75
        return (early + late) / self.num_layers


class CompressionCodec:
    """Mixed-precision KV cache quantization codec (Activity C).

    Early layers (< cutoff) use FP16 to preserve critical information.
    Remaining layers use symmetric per-tensor INT8 quantization (~50% savings).
    """

    def __init__(self, num_layers: int, cutoff_ratio: float = 1 / 3) -> None:
        self.num_layers = num_layers
        self.cutoff = max(1, int(num_layers * cutoff_ratio))
        # Stores INT8 scale factors: key = (layer_idx, tensor_id)
        self._scales: Dict[Tuple[int, int], float] = {}

    def encode(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Compress KV tensor; FP16 for early layers, INT8 for later layers."""
        if layer_idx < self.cutoff:
            return kv.to(torch.float16)

        abs_max = kv.abs().max().item()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        self._scales[(layer_idx, tensor_id)] = scale
        quantized = (kv.float() / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized

    def decode(
        self,
        compressed: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Decompress back to float32."""
        if layer_idx < self.cutoff:
            return compressed.to(torch.float32)

        scale = self._scales.get((layer_idx, tensor_id), 1.0)
        return compressed.to(torch.float32) * scale

    def compression_ratio(self, layer_idx: int) -> float:
        """Theoretical bytes saved vs FP32 baseline."""
        if layer_idx < self.cutoff:
            return 0.5  # FP16 = 2 bytes vs FP32 = 4 bytes → 50% savings
        return 0.75  # INT8 = 1 byte vs FP32 = 4 bytes → 75% savings

    def average_compression_ratio(self) -> float:
        early = self.cutoff * 0.5
        late = (self.num_layers - self.cutoff) * 0.75
        return (early + late) / self.num_layers
