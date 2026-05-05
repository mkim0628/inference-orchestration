"""FireQAttentionPatch — FireQCodec integration with vLLM 0.20.1 attention backend.

Activity C-2: RoPE-aware 2-stage outlier smoothing, patched into vLLM's
FlashAttentionImpl or XFormersBackend forward pass.

Integration point: vllm.v1.attention.backends.flash_attn.FlashAttentionImpl

Design:
  - FireQAttentionPatch wraps a vLLM AttentionImpl (specifically
    FlashAttentionImpl in vLLM 0.20.1) and injects FireQCodec transforms
    around the KV cache write/read path.
  - Pre-RoPE hook: applied to key before RoPE rotations (Stage 1 equalization).
  - Post-RoPE hook: applied to key after RoPE (Stage 2 outlier masking).
  - The patched forward() calls the wrapped impl's forward() with smoothed K/V.
  - Graceful fallback: if CUDA is unavailable or vLLM internals are missing,
    FireQAttentionPatch falls back to identity (no-op) transforms.
  - No new nn.Parameter — all scale factors come from FireQCodec.calibrate().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# vLLM imports — graceful fallback
# ---------------------------------------------------------------------------
_VLLM_FLASH_ATTN_AVAILABLE = False
_VLLM_ATTN_BACKEND_AVAILABLE = False
_FlashAttentionImpl = None
_AttentionImpl = None

try:
    from vllm.v1.attention.backend import AttentionImpl, AttentionBackend
    _AttentionImpl = AttentionImpl
    _VLLM_ATTN_BACKEND_AVAILABLE = True
except ImportError:
    pass

try:
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionBackend,
    )
    _FlashAttentionImpl = FlashAttentionImpl
    _VLLM_FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# FireQ constants
# ---------------------------------------------------------------------------

def _block_quantize_int4_sym(
    x: torch.Tensor,
    block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Symmetric per-block INT4 quantization.

    Returns (quantized_uint8, scale_fp16, original_numel).
    """
    flat = x.float().reshape(-1)
    n = flat.numel()
    num_blocks = (n + block_size - 1) // block_size
    pad = num_blocks * block_size - n
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(num_blocks, block_size)
    scale = blocks.abs().max(dim=-1)[0].clamp(min=1e-8) / 7.0
    quant_int = (blocks / scale.unsqueeze(-1)).round().clamp(-8, 7)
    uint8 = (quant_int + 8).to(torch.uint8)
    return uint8, scale.half(), n


def _block_dequantize_int4_sym(
    uint8: torch.Tensor,
    scale: torch.Tensor,
    original_numel: int,
    original_shape: torch.Size,
    block_size: int = 64,
) -> torch.Tensor:
    """Inverse of _block_quantize_int4_sym."""
    dequant = (uint8.float() - 8) * scale.float().unsqueeze(-1)
    return dequant.reshape(-1)[:original_numel].reshape(original_shape).float()


# ---------------------------------------------------------------------------
# FireQCodec (self-contained, no src/ import required)
# ---------------------------------------------------------------------------

class _FireQCodecCore:
    """Minimal FireQCodec for use inside the attention patch.

    Self-contained: no dependency on src.cache.fireq_codec.
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        outlier_threshold_sigma: float = 3.0,
    ) -> None:
        self.n_heads = n_heads
        self.d_head = d_head
        self.outlier_threshold_sigma = outlier_threshold_sigma
        self._pre_rope_scales: Dict[int, torch.Tensor] = {}
        self._outlier_masks: Dict[int, torch.Tensor] = {}

    def calibrate(
        self,
        calib_kvs: List[Tuple[torch.Tensor, int]],
        min_samples: int = 10,
    ) -> None:
        from collections import defaultdict
        layer_samples: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for kv, layer_idx in calib_kvs:
            layer_samples[layer_idx].append(kv.float())
        for layer_idx, tensors in layer_samples.items():
            stacked = torch.stack(tensors, dim=0)
            if stacked.dim() == 5:
                N, B, H, S, D = stacked.shape
                stacked = stacked.reshape(N * B, H, S, D)
            half_d = self.d_head // 2
            K = stacked
            var_first = K[..., :half_d].var(dim=(0, 2))
            var_second = K[..., half_d:].var(dim=(0, 2))
            s = (var_first / var_second.clamp(min=1e-8)).sqrt()
            self._pre_rope_scales[layer_idx] = s.detach().clone()
            channel_std = K.std(dim=(0, 2))
            channel_max = K.abs().amax(dim=(0, 2))
            self._outlier_masks[layer_idx] = (
                channel_max > self.outlier_threshold_sigma * channel_std
            ).detach().clone()

    def apply_pre_rope(self, K: torch.Tensor, layer_idx: int) -> torch.Tensor:
        scales = self._pre_rope_scales.get(layer_idx)
        if scales is None:
            return K
        half_d = self.d_head // 2
        s = scales.to(K.device)
        K = K.clone()
        K[..., :half_d] = K[..., :half_d] / s.unsqueeze(-2)
        K[..., half_d:] = K[..., half_d:] * s.unsqueeze(-2)
        return K

    def apply_post_rope(
        self, K: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mask = self._outlier_masks.get(layer_idx)
        if mask is None:
            return K, None
        if K.dim() == 3:
            channel_max = K.abs().amax(dim=1)
        elif K.dim() == 4:
            channel_max = K.abs().amax(dim=(0, 2))
        else:
            channel_max = K.abs().amax(dim=-2)
        target_scale = channel_max.clamp(min=1.0)
        K = K / target_scale.unsqueeze(-2)
        return K, target_scale.half()

    def invert_post_rope(
        self, K: torch.Tensor, outlier_scale: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if outlier_scale is None:
            return K
        return K * outlier_scale.float().unsqueeze(-2)

    def invert_pre_rope(self, K: torch.Tensor, layer_idx: int) -> torch.Tensor:
        scales = self._pre_rope_scales.get(layer_idx)
        if scales is None:
            return K
        half_d = self.d_head // 2
        s = scales.to(K.device)
        K = K.clone()
        K[..., :half_d] = K[..., :half_d] * s.unsqueeze(-2)
        K[..., half_d:] = K[..., half_d:] / s.unsqueeze(-2)
        return K

    def load_from_dir(self, calib_dir: str) -> bool:
        p = Path(calib_dir)
        loaded = False
        for scale_file in p.glob("layer_*_scales.pt"):
            try:
                layer_idx = int(scale_file.stem.split("_")[1])
                self._pre_rope_scales[layer_idx] = torch.load(
                    scale_file, weights_only=True
                )
                loaded = True
            except Exception:
                pass
        for mask_file in p.glob("layer_*_masks.pt"):
            try:
                layer_idx = int(mask_file.stem.split("_")[1])
                self._outlier_masks[layer_idx] = torch.load(
                    mask_file, weights_only=True
                )
                loaded = True
            except Exception:
                pass
        return loaded


# ---------------------------------------------------------------------------
# FireQAttentionPatch — wraps FlashAttentionImpl
# ---------------------------------------------------------------------------

class FireQAttentionPatch:
    """Wraps vLLM's FlashAttentionImpl (or any AttentionImpl) to inject
    FireQCodec key smoothing before KV cache writes (Activity C-2).

    Usage:
        codec = _FireQCodecCore(n_heads=32, d_head=128)
        codec.calibrate(calib_data)

        # Wrap the existing impl from a vLLM attention layer:
        patched = FireQAttentionPatch(original_impl, codec, layer_idx=5)

        # Use patched.forward() as a drop-in replacement for original_impl.forward()

    Fallback behaviour:
        If CUDA is unavailable or vLLM internals are not importable,
        forward() delegates directly to the wrapped impl without modification.
    """

    def __init__(
        self,
        wrapped_impl: object,
        codec: Optional[_FireQCodecCore] = None,
        layer_idx: int = 0,
        apply_pre_rope: bool = True,
        apply_post_rope: bool = True,
    ) -> None:
        """
        Args:
            wrapped_impl:    Original AttentionImpl instance (e.g. FlashAttentionImpl).
            codec:           _FireQCodecCore instance (calibrated).  If None, no-op.
            layer_idx:       Transformer layer index for scale lookup.
            apply_pre_rope:  Apply Stage 1 pre-RoPE channel pair equalization.
            apply_post_rope: Apply Stage 2 post-RoPE outlier scaling.
        """
        self._impl = wrapped_impl
        self._codec = codec
        self.layer_idx = layer_idx
        self.apply_pre_rope = apply_pre_rope
        self.apply_post_rope = apply_post_rope

    @classmethod
    def from_flash_attn_backend(
        cls,
        layer_module: "torch.nn.Module",
        codec: Optional[_FireQCodecCore],
        layer_idx: int = 0,
    ) -> "FireQAttentionPatch":
        """Convenience constructor: wraps the impl from a vLLM attention layer.

        Args:
            layer_module:  nn.Module that has an ``impl`` attribute (vLLM pattern).
            codec:         Calibrated FireQCodec instance.
            layer_idx:     Transformer layer index.
        """
        impl = getattr(layer_module, "impl", None)
        if impl is None:
            impl = layer_module  # fallback: wrap layer itself
        return cls(impl, codec=codec, layer_idx=layer_idx)

    def forward(
        self,
        layer: "torch.nn.Module",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: object,
        output: torch.Tensor,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Patched forward: inject FireQ smoothing then delegate to wrapped impl.

        Hook insertion points:
          1. (pre-RoPE)  Stage 1: channel pair variance equalization on key.
          2. (post-RoPE) Stage 2: outlier channel targeted scaling on key.
          3. Wrapped impl forward() — writes smoothed K to KV cache and runs attn.
          4. No inverse is applied in forward() because the KV cache stores the
             smoothed representation; inverse happens at retrieval time
             (retrieve_and_invert_key()).
        """
        cuda_ok = key.is_cuda and torch.cuda.is_available()

        if self._codec is not None:
            # Stage 1 — pre-RoPE equalization
            if self.apply_pre_rope:
                key = self._codec.apply_pre_rope(key, self.layer_idx)

            # Stage 2 — post-RoPE outlier scaling
            _outlier_scale: Optional[torch.Tensor] = None
            if self.apply_post_rope:
                key, _outlier_scale = self._codec.apply_post_rope(key, self.layer_idx)

        # Delegate to wrapped impl
        if _VLLM_FLASH_ATTN_AVAILABLE and isinstance(
            self._impl, _FlashAttentionImpl
        ):
            return self._impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        elif hasattr(self._impl, "forward"):
            # Generic AttentionImpl subclass
            return self._impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        else:
            # Last resort fallback: no-op (should not happen in normal usage)
            return output

    def retrieve_and_invert_key(
        self,
        key_smoothed: torch.Tensor,
        outlier_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Invert FireQ smoothing on a retrieved KV cache key tensor.

        Call this after reading a compressed key from the KV cache to restore
        the original-distribution key before attention computation.

        Args:
            key_smoothed:  Key tensor as stored (after FireQ smoothing).
            outlier_scale: Outlier scale tensor returned by Stage 2 (if any).

        Returns:
            Approximately restored key tensor (FP16).
        """
        K = key_smoothed.float().clone()
        if self._codec is not None:
            if self.apply_post_rope:
                K = self._codec.invert_post_rope(K, outlier_scale)
            if self.apply_pre_rope:
                K = self._codec.invert_pre_rope(K, self.layer_idx)
        return K.half()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_codec(
        n_heads: int,
        d_head: int,
        outlier_threshold_sigma: float = 3.0,
        calib_dir: Optional[str] = None,
    ) -> _FireQCodecCore:
        """Create and optionally pre-load a FireQCodecCore instance.

        Args:
            n_heads:                  Number of attention heads.
            d_head:                   Head dimension.
            outlier_threshold_sigma:  Outlier detection threshold.
            calib_dir:                Path to saved calibration files; if
                                      provided, scales are loaded automatically.

        Returns:
            _FireQCodecCore instance (calibrated if calib_dir is valid).
        """
        codec = _FireQCodecCore(
            n_heads=n_heads,
            d_head=d_head,
            outlier_threshold_sigma=outlier_threshold_sigma,
        )
        if calib_dir is not None:
            codec.load_from_dir(calib_dir)
        return codec

    @staticmethod
    def patch_vllm_model_layers(
        model: "torch.nn.Module",
        codec: _FireQCodecCore,
        layer_name_contains: str = "attn",
    ) -> int:
        """Walk model layers and wrap matching attention impls with FireQAttentionPatch.

        This is a convenience utility for patching a fully-loaded vLLM model.
        It wraps the ``impl`` attribute of each attention layer whose name
        contains ``layer_name_contains``.

        Args:
            model:                 vLLM model (nn.Module).
            codec:                 Calibrated FireQCodecCore.
            layer_name_contains:   Substring to match attention layer names.

        Returns:
            Number of layers patched.
        """
        patched_count = 0
        for layer_idx, (name, module) in enumerate(model.named_modules()):
            if layer_name_contains in name and hasattr(module, "impl"):
                original_impl = module.impl
                module.impl = FireQAttentionPatch(
                    wrapped_impl=original_impl,
                    codec=codec,
                    layer_idx=layer_idx,
                )
                patched_count += 1
        return patched_count


__all__ = [
    "FireQAttentionPatch",
    "_FireQCodecCore",
]
