"""Activity C (2026-05-26): CacheConfig Extension for MLATwoAxisCompressionCodec.

Extends vLLM 0.21.0's CacheConfig with compression_method and MLA two-axis
codec parameters, without modifying vLLM source.

Approach: Python dataclass that wraps/extends CacheConfig fields, added to
VllmConfig via a side-channel attribute (_irminsul_compression_config).

vLLM version: 0.21.0
Activity: C-1 (MLA Two-Axis Compression Codec)
Source: src/cache/mla_two_axis_compression_codec.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# CacheConfig compression_method values (mirrors Activity C spec)
CompressionMethod = Literal["none", "int8", "fp8", "eviction", "low_rank", "mla_two_axis"]


# ---------------------------------------------------------------------------
# IrminsulCompressionConfig — CacheConfig extension for C-1
# ---------------------------------------------------------------------------


@dataclass
class IrminsulCompressionConfig:
    """Extension to vLLM CacheConfig for Irminsul MLA two-axis compression (C-1).

    This config is attached as vllm_config._irminsul_compression_config.
    It does NOT modify the vLLM CacheConfig dataclass directly — that would
    require vLLM source changes.

    Fields:
        compression_method: which C-1 codec to activate
        depth_sharing_threshold: cos_sim threshold for depth-axis layer sharing
        depth_sharing_k: target layer sharing ratio (k=2 → ~50% sharing)
        position_dedup_enabled: enable position-axis deduplication
        fallback_threshold: raised when accuracy_delta > max_allowed_accuracy_delta
        max_allowed_accuracy_delta: accuracy delta threshold for fallback
    """

    compression_method: CompressionMethod = "none"
    """Compression method. "none" = disabled. "mla_two_axis" = Activity C-1."""

    depth_sharing_threshold: float = 0.90
    """Cosine similarity threshold for depth-axis layer sharing.

    Only layers where cos_sim(c_kv[l], c_kv[l-1]) >= this threshold share KV.
    Must be >= 0.90 for accuracy preservation (Activity C §4).
    """

    depth_sharing_k: int = 2
    """Target sharing ratio: k=2 → up to 50% of layers share a previous layer."""

    position_dedup_enabled: bool = True
    """Enable position-axis deduplication.

    Identical segment_id (SHA256 content hash) → pointer reuse, no copy.
    Mathematical guarantee: c_KV position-free → zero accuracy loss.
    """

    fallback_threshold: float = 0.95
    """Threshold raised to this value when accuracy_delta > max_allowed_accuracy_delta."""

    max_allowed_accuracy_delta: float = 0.01
    """Maximum allowed accuracy delta (perplexity / task accuracy change).

    If |delta| > this value, depth_sharing_threshold is raised to fallback_threshold.
    Set to 0.01 (1%) per Activity C accuracy preservation requirement.
    """

    # Quantization parameters (for future int8/fp8 codec integration)
    quantization_bits: int = 8
    """Quantization bit width when compression_method is "int8" or "fp8"."""

    def is_enabled(self) -> bool:
        """Return True if compression is active (not "none")."""
        return self.compression_method != "none"

    def is_mla_two_axis(self) -> bool:
        """Return True if MLA two-axis compression (C-1) is active."""
        return self.compression_method == "mla_two_axis"

    def to_hook_config(self) -> Any:
        """Convert to MLATwoAxisHookConfig for use with MLATwoAxisCompressionHook."""
        try:
            from vllm_integration.irminsul_attention_backend_patch import (
                MLATwoAxisHookConfig,
            )
            return MLATwoAxisHookConfig(
                depth_sharing_threshold=self.depth_sharing_threshold,
                depth_sharing_k=self.depth_sharing_k,
                position_dedup_enabled=self.position_dedup_enabled,
                fallback_threshold=self.fallback_threshold,
                max_allowed_accuracy_delta=self.max_allowed_accuracy_delta,
                compression_method=self.compression_method,
            )
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# install_compression_config — attach to VllmConfig
# ---------------------------------------------------------------------------


def install_compression_config(
    vllm_config: Any,
    compression_config: Optional[IrminsulCompressionConfig] = None,
) -> IrminsulCompressionConfig:
    """Attach an IrminsulCompressionConfig to a vLLM VllmConfig instance.

    Does not modify vLLM's CacheConfig; attaches as a side-channel attribute.

    After this call:
        vllm_config._irminsul_compression_config  →  IrminsulCompressionConfig

    Returns the compression config for further configuration.
    """
    if compression_config is None:
        compression_config = IrminsulCompressionConfig()
    vllm_config._irminsul_compression_config = compression_config  # type: ignore[attr-defined]
    return compression_config


def get_compression_config(vllm_config: Any) -> Optional[IrminsulCompressionConfig]:
    """Retrieve the attached compression config from a VllmConfig, or None."""
    return getattr(vllm_config, "_irminsul_compression_config", None)


# ---------------------------------------------------------------------------
# Block size adjustment helper (Activity C §: OOM prevention)
# ---------------------------------------------------------------------------


def compute_compressed_block_count(
    num_gpu_blocks: int,
    compression_method: CompressionMethod,
    compression_ratio: float = 0.5,
) -> int:
    """Compute adjusted GPU block count accounting for compression memory savings.

    When compression is active, the effective memory per block is reduced by
    compression_ratio.  This allows allocating more blocks within the same
    GPU memory budget, increasing effective KV cache capacity.

    Args:
        num_gpu_blocks: base number of GPU blocks from vLLM profiling
        compression_method: which compression codec is active
        compression_ratio: expected memory reduction (0.5 = 50% reduction → 2× blocks)

    Returns:
        Adjusted block count. If compression is disabled, returns num_gpu_blocks.
    """
    if compression_method == "none":
        return num_gpu_blocks

    if compression_method == "mla_two_axis":
        # MLA two-axis: position dedup + depth sharing
        # Conservative estimate: ~30% reduction → 1.3× blocks
        # (actual measured in Activity C evaluation)
        effective_ratio = max(0.0, min(compression_ratio, 0.7))
        adjusted = int(num_gpu_blocks * (1.0 + effective_ratio))
        return adjusted

    if compression_method in ("int8", "fp8"):
        # INT8/FP8 quantization: ~50% reduction → 2× blocks
        effective_ratio = max(0.0, min(compression_ratio, 0.5))
        adjusted = int(num_gpu_blocks * (1.0 + effective_ratio))
        return adjusted

    return num_gpu_blocks
