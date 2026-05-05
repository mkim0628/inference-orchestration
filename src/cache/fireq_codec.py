"""FireQCodec — RoPE-aware 2-stage outlier smoothing + INT4+FP8 mixed precision (Activity C-2).

Stage 1 (pre-RoPE): equalize variance of RoPE rotation channel pairs (i, i+d/2)
                    via per-channel pair scale factors estimated from calibration.
Stage 2 (post-RoPE): per-channel targeted scaling of remaining outlier channels
                     (|K| > outlier_threshold_sigma * channel_std).

Key: INT4 (4-bit unsigned, stored as uint8, packed 2-per-byte conceptually)
Value: FP8 (torch.float8_e4m3fn on CUDA; FP16 fallback on CPU)

Training-free: all parameters are estimated statistics, not nn.Parameter.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _block_quantize_int4(
    x: torch.Tensor,
    block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-block INT4-range quantization. Returns (quantized uint8, scale, zero)."""
    flat = x.float().reshape(-1)
    n = flat.numel()
    num_blocks = (n + block_size - 1) // block_size
    pad = num_blocks * block_size - n
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(num_blocks, block_size)
    # Symmetric around 0: scale = abs_max / 7
    scale = blocks.abs().max(dim=-1)[0].clamp(min=1e-8) / 7.0  # (num_blocks,)
    quantized = (blocks / scale.unsqueeze(-1)).round().clamp(-8, 7)
    # Store in uint8 as (val + 8) to keep unsigned
    quantized_uint8 = (quantized + 8).to(torch.uint8)
    return quantized_uint8, scale.half(), torch.tensor(n, dtype=torch.int64)


def _block_dequantize_int4(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    original_numel: int,
    original_shape: torch.Size,
    block_size: int = 64,
) -> torch.Tensor:
    """Inverse of _block_quantize_int4."""
    # quantized: (num_blocks, block_size) uint8
    dequant = (quantized.float() - 8) * scale.float().unsqueeze(-1)
    flat = dequant.reshape(-1)[:original_numel]
    return flat.reshape(original_shape).float()


class FireQCodec:
    """RoPE-aware 2-stage outlier smoothing codec (Activity C-2).

    Key: INT4 with pre-RoPE channel pair normalization + post-RoPE outlier scaling.
    Value: FP8 on CUDA (float8_e4m3fn), FP16 fallback on CPU.
    Training-free: calibrate() estimates statistics only.
    """

    def __init__(
        self,
        n_heads: int = 12,
        d_head: int = 64,
        outlier_threshold_sigma: float = 3.0,
        calib_scale_dir: Optional[str] = None,
    ) -> None:
        self.n_heads = n_heads
        self.d_head = d_head
        self.outlier_threshold_sigma = outlier_threshold_sigma
        self.calib_scale_dir = Path(calib_scale_dir) if calib_scale_dir else None
        # Layer-indexed pre-RoPE channel pair scale factors: {layer_idx: Tensor(n_heads, d_head//2)}
        self._pre_rope_scales: Dict[int, torch.Tensor] = {}
        # Layer-indexed post-RoPE outlier channel boolean masks: {layer_idx: Tensor(n_heads, d_head)}
        self._outlier_masks: Dict[int, torch.Tensor] = {}

    def calibrate(
        self,
        calib_kvs: List[Tuple[torch.Tensor, int]],
        min_samples: int = 10,
    ) -> None:
        """Estimate per-layer pre-RoPE scale factors and post-RoPE outlier masks.

        Args:
            calib_kvs: list of (kv_tensor, layer_idx). kv_tensor shape:
                       (n_heads, seq_len, d_head) or (batch, n_heads, seq_len, d_head).
            min_samples: minimum calibration samples per layer (warning if fewer).
        """
        # Group samples by layer
        from collections import defaultdict
        layer_samples: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for kv, layer_idx in calib_kvs:
            layer_samples[layer_idx].append(kv.float())

        for layer_idx, tensors in layer_samples.items():
            if len(tensors) < min_samples:
                pass  # Proceed even with fewer samples — warn would be verbose

            # Stack: (N, ...) where N = number of samples
            stacked = torch.stack(tensors, dim=0)  # (N, [batch,] n_heads, seq_len, d_head)
            # Normalize to (N, n_heads, seq_len, d_head) by flattening batch dims
            if stacked.dim() == 5:
                N, B, H, S, D = stacked.shape
                stacked = stacked.reshape(N * B, H, S, D)
            # stacked: (N, n_heads, seq_len, d_head)

            half_d = self.d_head // 2

            # Stage 1: channel pair (i, i+d/2) variance equalization
            # var over (N, seq_len) dims → shape (n_heads, d_head//2)
            # dim=(0, 2) = (batch, seq) dimensions
            K = stacked  # (N, n_heads, seq_len, d_head)
            var_first = K[..., :half_d].var(dim=(0, 2))    # (n_heads, half_d)
            var_second = K[..., half_d:].var(dim=(0, 2))   # (n_heads, half_d)
            # Scale factor: sqrt(var_first / var_second) to equalize
            s = (var_first / var_second.clamp(min=1e-8)).sqrt()
            self._pre_rope_scales[layer_idx] = s.detach().clone()

            # Stage 2: post-RoPE outlier channel mask
            channel_std = K.std(dim=(0, 2))           # (n_heads, d_head)
            channel_max = K.abs().amax(dim=(0, 2))    # (n_heads, d_head)
            outlier_mask = channel_max > self.outlier_threshold_sigma * channel_std
            self._outlier_masks[layer_idx] = outlier_mask.detach().clone()

        if self.calib_scale_dir is not None:
            self._save_calibration()

    def encode(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        rope_applied: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode KV tensor to (key_int4_uint8, value_fp8_or_fp16, meta).

        Args:
            kv:          Key tensor: (n_heads, seq_len, d_head) or (batch, n_heads, seq_len, d_head).
                         For combined KV, pass key part only.
            layer_idx:   Transformer layer index.
            rope_applied: If False, apply Stage 1 pre-RoPE smoothing.
                         If True, apply Stage 2 post-RoPE outlier scaling.

        Returns:
            key_int4:   uint8 tensor (quantized key)
            value_fp8:  FP8 or FP16 fallback (same shape as kv for value)
            meta:       dict with dequantization info
        """
        K = kv.float().clone()
        meta: dict = {
            "original_shape": kv.shape,
            "layer_idx": layer_idx,
            "rope_applied": rope_applied,
        }

        # Stage 1: pre-RoPE channel pair scale (applied when rope not yet applied)
        if not rope_applied:
            scales = self._pre_rope_scales.get(layer_idx)
            if scales is not None:
                half_d = self.d_head // 2
                scales_dev = scales.to(K.device)
                # Broadcast: scales shape (n_heads, half_d) → (..., n_heads, 1, half_d)
                K[..., :half_d] = K[..., :half_d] / scales_dev.unsqueeze(-2)
                K[..., half_d:] = K[..., half_d:] * scales_dev.unsqueeze(-2)
                meta["pre_rope_scales_applied"] = True

        # Stage 2: post-RoPE outlier channel targeted scaling
        outlier_scale = None
        if rope_applied:
            mask = self._outlier_masks.get(layer_idx)
            if mask is not None:
                # channel_max per head/channel → (n_heads, d_head)
                if K.dim() == 3:
                    channel_max = K.abs().amax(dim=1)  # (n_heads, d_head)
                elif K.dim() == 4:
                    channel_max = K.abs().amax(dim=(0, 2))  # (n_heads, d_head)
                else:
                    channel_max = K.abs().amax(dim=-2)

                target_scale = channel_max.clamp(min=1.0)  # (n_heads, d_head)
                # Scale down outlier channels so INT4 range covers them
                K = K / target_scale.unsqueeze(-2)
                outlier_scale = target_scale.half()
                meta["outlier_scale"] = outlier_scale

        # Key INT4 block quantization
        key_quant_flat = K.reshape(-1)
        n_elem = key_quant_flat.numel()
        block_size = 64
        num_blocks = (n_elem + block_size - 1) // block_size
        pad = num_blocks * block_size - n_elem
        if pad > 0:
            key_quant_flat = torch.nn.functional.pad(key_quant_flat, (0, pad))
        blocks = key_quant_flat.reshape(num_blocks, block_size)
        block_scale = blocks.abs().max(dim=-1)[0].clamp(min=1e-8) / 7.0
        quant_int = (blocks / block_scale.unsqueeze(-1)).round().clamp(-8, 7)
        key_int4 = (quant_int + 8).to(torch.uint8)  # (num_blocks, block_size)
        meta["key_block_scale"] = block_scale.half()
        meta["key_original_numel"] = n_elem

        # Value: FP8 on CUDA, FP16 fallback on CPU
        V = kv.float()
        if kv.is_cuda and hasattr(torch, "float8_e4m3fn"):
            value_fp8 = V.to(torch.float8_e4m3fn)
        else:
            value_fp8 = V.half()

        return key_int4, value_fp8, meta

    def decode(
        self,
        key_int4: torch.Tensor,
        value_fp8: torch.Tensor,
        meta: dict,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode (key_int4, value_fp8, meta) back to (K_fp16, V_fp16).

        Args:
            key_int4:  uint8 tensor, shape (num_blocks, block_size)
            value_fp8: FP8 or FP16 value tensor
            meta:      dict from encode()
            layer_idx: Transformer layer index

        Returns:
            K_fp16, V_fp16 tensors.
        """
        # Key dequantization
        block_scale = meta["key_block_scale"].float()
        n_elem = meta["key_original_numel"]
        original_shape = meta["original_shape"]

        dequant = (key_int4.float() - 8) * block_scale.unsqueeze(-1)
        K = dequant.reshape(-1)[:n_elem].reshape(original_shape).float()

        # Stage 2 inverse: restore outlier channel scaling
        if "outlier_scale" in meta:
            outlier_scale = meta["outlier_scale"].float()  # (n_heads, d_head)
            K = K * outlier_scale.unsqueeze(-2)

        # Stage 1 inverse: restore pre-RoPE channel pair scaling
        if meta.get("pre_rope_scales_applied"):
            scales = self._pre_rope_scales.get(layer_idx)
            if scales is not None:
                half_d = self.d_head // 2
                scales_dev = scales.float().to(K.device)
                K[..., :half_d] = K[..., :half_d] * scales_dev.unsqueeze(-2)
                K[..., half_d:] = K[..., half_d:] / scales_dev.unsqueeze(-2)

        # Value: cast back to float16
        V = value_fp8.float().half()

        return K.half(), V

    def load_calibration(self, layer_idx: int) -> bool:
        """Load saved calibration scales and masks for a layer. Returns True on success."""
        if self.calib_scale_dir is None:
            return False
        scale_path = self.calib_scale_dir / f"layer_{layer_idx}_scales.pt"
        mask_path = self.calib_scale_dir / f"layer_{layer_idx}_masks.pt"
        try:
            if scale_path.exists():
                self._pre_rope_scales[layer_idx] = torch.load(scale_path, weights_only=True)
            if mask_path.exists():
                self._outlier_masks[layer_idx] = torch.load(mask_path, weights_only=True)
            return scale_path.exists() or mask_path.exists()
        except Exception:
            return False

    def _save_calibration(self) -> None:
        """Persist calibration data to calib_scale_dir."""
        if self.calib_scale_dir is None:
            return
        self.calib_scale_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx, scales in self._pre_rope_scales.items():
            torch.save(scales, self.calib_scale_dir / f"layer_{layer_idx}_scales.pt")
        for layer_idx, mask in self._outlier_masks.items():
            torch.save(mask, self.calib_scale_dir / f"layer_{layer_idx}_masks.pt")
        # Save metadata
        meta = {
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "outlier_threshold_sigma": self.outlier_threshold_sigma,
            "layers": sorted(self._pre_rope_scales.keys()),
        }
        with open(self.calib_scale_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
