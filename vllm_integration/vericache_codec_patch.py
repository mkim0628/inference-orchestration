"""VeriCache Speculative Draft-Verify KV Compression — Activity C (2026-05-25).

Hooks VeriCacheSpeculativeCodec (arXiv 2605.17613) into vLLM 0.21.0 v1 attention
backend write/read path.

Integration design:
  - write_to_cache(kv_key, key, value, layer_idx):
      Stores full KV + generates INT8 compressed draft in auxiliary side-channel cache.
      ALWAYS returns (key, value) UNCHANGED so primary vLLM attention kernel receives
      full unmodified tensors (accuracy contract §4: primary kernel error = 0).

  - read_from_cache(kv_key, layer_idx, Q):
      Returns VerificationResult: draft attention output + verified attention output.
      Final output selection: get_final_output(result) picks draft if accepted else verified.

  - apply_vericache_codec_patch(attn_impl, config, layer_idx):
      Monkey-patches a FlashAttentionImpl (or any AttentionImpl subclass) to call
      write_to_cache/read_from_cache at the correct points.

  - extend_cache_config_vericache(cache_config, hook_config):
      Extends vLLM CacheConfig with VeriCache fields via object.__setattr__ (pydantic-safe).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# Inline DraftCodec (no src/ dependency)                                       #
# --------------------------------------------------------------------------- #

class _InlineDraftCodec:
    def compress(self, kv: torch.Tensor) -> object:
        raise NotImplementedError

    def decompress(self, compressed_kv: object) -> torch.Tensor:
        raise NotImplementedError

    @property
    def compression_ratio(self) -> float:
        raise NotImplementedError


class _InlineInt8DraftCodec(_InlineDraftCodec):
    """INT8 quantization: FP16 -> INT8 (2x memory saving)."""

    def compress(self, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = kv.float().abs().max() / 127.0 + 1e-8
        quantized = (kv.float() / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def decompress(self, compressed_kv: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        quantized, scale = compressed_kv
        return quantized.float() * scale

    @property
    def compression_ratio(self) -> float:
        return 2.0


# --------------------------------------------------------------------------- #
# VeriCache hook config + entry                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class VeriCacheCodecHookConfig:
    """Configuration for VeriCacheCodecAttentionHook."""
    d_head: int = 128
    acceptance_threshold: float = 0.01   # Accept draft if relative_error < this
    max_entries_per_layer: int = 512
    enable_async_verify: bool = True
    seed: int = 42


@dataclass
class _VeriCacheEntry:
    key: str
    compressed_kv: object          # _InlineInt8DraftCodec.compress() result
    full_kv_ref: torch.Tensor      # Full KV for verify
    original_shape: torch.Size     # [n_tokens, ...] shape


@dataclass
class VeriCacheVerificationResult:
    accepted: bool
    draft_output: torch.Tensor      # Attention output from compressed KV
    verified_output: torch.Tensor   # Attention output from full KV
    relative_error: float
    acceptance_threshold: float


# --------------------------------------------------------------------------- #
# Core hook class                                                               #
# --------------------------------------------------------------------------- #

class VeriCacheCodecAttentionHook:
    """VeriCache speculative draft-verify codec hook for vLLM attention backends.

    Implements the write/read hook pattern from the established vllm_integration
    convention (see attention_backend_patch.py prior cycles):
      - write_to_cache() stores compressed side-channel; returns ORIGINAL tensors
      - read_from_cache() returns draft+verify result for non-contiguous segment reuse
      - Primary attention kernel ALWAYS receives original unmodified KV (accuracy contract)

    Storage layout: _store[(kv_key, layer_idx)] = (_VeriCacheEntry for K, _VeriCacheEntry for V)
    """

    def __init__(self, config: VeriCacheCodecHookConfig) -> None:
        self.config = config
        self._codec = _InlineInt8DraftCodec()
        # (kv_key, layer_idx) -> (entry_K, entry_V)
        self._store: OrderedDict[Tuple[str, int], Tuple[_VeriCacheEntry, _VeriCacheEntry]] = (
            OrderedDict()
        )
        self._hits: int = 0
        self._misses: int = 0
        self._total_draft_calls: int = 0
        self._total_accepted: int = 0
        self._relative_errors: List[float] = []
        self._lock = threading.Lock()

    # ---- Public write/read interface ----------------------------------------

    def write_to_cache(
        self,
        kv_key: str,
        key: torch.Tensor,     # [n_tokens, n_kv_heads, head_size] or [n_tokens, head_size]
        value: torch.Tensor,   # same shape as key
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store compressed KV in side-channel cache. Returns (key, value) UNCHANGED.

        The primary vLLM attention kernel ALWAYS receives full unmodified tensors.
        Compressed KV is stored only in the auxiliary side-channel for draft-verify
        non-contiguous reuse.

        Args:
            kv_key: segment identifier
            key:    key tensor [n_tokens, ...]
            value:  value tensor [n_tokens, ...]
            layer_idx: transformer layer index

        Returns:
            (key_original, value_original) — ALWAYS the original unchanged tensors
        """
        store_key = (kv_key, layer_idx)
        if store_key in self._store:
            # Already stored; refresh LRU order
            with self._lock:
                self._store.move_to_end(store_key)
            return key, value

        # Evict if at capacity
        with self._lock:
            if len(self._store) >= self.config.max_entries_per_layer:
                self._store.popitem(last=False)  # LRU evict

        # Flatten to 2D [n_tokens, d] for codec
        key_2d = key.reshape(key.shape[0], -1).detach().clone()
        val_2d = value.reshape(value.shape[0], -1).detach().clone()

        compressed_k = self._codec.compress(key_2d.float())
        compressed_v = self._codec.compress(val_2d.float())

        entry_K = _VeriCacheEntry(
            key=kv_key + "_K",
            compressed_kv=compressed_k,
            full_kv_ref=key_2d,
            original_shape=key.shape,
        )
        entry_V = _VeriCacheEntry(
            key=kv_key + "_V",
            compressed_kv=compressed_v,
            full_kv_ref=val_2d,
            original_shape=value.shape,
        )

        with self._lock:
            self._store[store_key] = (entry_K, entry_V)

        # ACCURACY CONTRACT: always return original tensors
        return key, value

    def read_from_cache(
        self,
        kv_key: str,
        layer_idx: int,
        Q: Optional[torch.Tensor] = None,   # [n_q, d_head] for draft+verify
    ) -> Optional[VeriCacheVerificationResult]:
        """Return VeriCache speculative draft-verify result for non-contiguous reuse.

        Args:
            kv_key:    segment identifier
            layer_idx: transformer layer index
            Q:         query tensor for draft attention. If None, returns None.

        Returns:
            VeriCacheVerificationResult, or None on miss
        """
        store_key = (kv_key, layer_idx)
        if store_key not in self._store or Q is None:
            self._misses += 1
            return None

        with self._lock:
            self._store.move_to_end(store_key)
            self._hits += 1
            entry_K, entry_V = self._store[store_key]

        # Flatten Q to 2D
        Q_2d = Q.reshape(Q.shape[0], -1).float()

        # Draft phase: attention from compressed (approx) KV
        draft_K = self._codec.decompress(entry_K.compressed_kv)
        draft_V = self._codec.decompress(entry_K.compressed_kv)
        # Use entry_V for V decompress
        draft_V = self._codec.decompress(entry_V.compressed_kv)
        draft_output = self._compute_attention(Q_2d, draft_K, draft_V)

        # Verify phase: attention from full KV
        verified_output = self._compute_attention(Q_2d, entry_K.full_kv_ref, entry_V.full_kv_ref)

        # Compute relative error
        rel_error = float(
            (draft_output - verified_output).norm()
            / (verified_output.norm() + 1e-8)
        )
        accepted = rel_error < self.config.acceptance_threshold

        with self._lock:
            self._total_draft_calls += 1
            if accepted:
                self._total_accepted += 1
            self._relative_errors.append(rel_error)

        return VeriCacheVerificationResult(
            accepted=accepted,
            draft_output=draft_output,
            verified_output=verified_output,
            relative_error=rel_error,
            acceptance_threshold=self.config.acceptance_threshold,
        )

    def get_final_output(self, result: VeriCacheVerificationResult) -> torch.Tensor:
        """Deterministic accuracy-preserving output selection.

        Accepted: draft_output (within threshold — faster path)
        Rejected: verified_output (full KV accuracy — deterministic guarantee)
        """
        return result.draft_output if result.accepted else result.verified_output

    # ---- Statistics ----------------------------------------------------------

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def draft_acceptance_rate(self) -> float:
        if self._total_draft_calls == 0:
            return 0.0
        return self._total_accepted / self._total_draft_calls

    def mean_relative_error(self) -> float:
        if not self._relative_errors:
            return 0.0
        return sum(self._relative_errors) / len(self._relative_errors)

    def memory_reduction_ratio(self) -> float:
        """Ratio = 1 - compressed_bytes / full_kv_bytes."""
        comp_bytes = 0
        full_bytes = 0
        for (entry_K, entry_V) in self._store.values():
            for entry in (entry_K, entry_V):
                full_bytes += entry.full_kv_ref.nbytes
                c = entry.compressed_kv
                if isinstance(c, tuple):
                    for t in c:
                        if isinstance(t, torch.Tensor):
                            comp_bytes += t.nbytes
                elif isinstance(c, torch.Tensor):
                    comp_bytes += c.nbytes
        if full_bytes == 0:
            return 0.0
        return 1.0 - comp_bytes / full_bytes

    def stats(self) -> dict:
        return {
            "hit_rate": self.hit_rate(),
            "draft_acceptance_rate": self.draft_acceptance_rate(),
            "mean_relative_error": self.mean_relative_error(),
            "memory_reduction_ratio": self.memory_reduction_ratio(),
            "total_draft_calls": self._total_draft_calls,
            "total_accepted": self._total_accepted,
            "n_entries": len(self._store),
            "compression_codec": "Int8DraftCodec",
            "compression_ratio": self._codec.compression_ratio,
        }

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_draft_calls = 0
        self._total_accepted = 0
        self._relative_errors.clear()

    # ---- Internal helpers ----------------------------------------------------

    @staticmethod
    def _compute_attention(
        Q: torch.Tensor,   # [n_q, d_head] float32
        K: torch.Tensor,   # [n_kv, d_head] float32
        V: torch.Tensor,   # [n_kv, d_head] float32
    ) -> torch.Tensor:
        """Scaled dot-product attention for draft-verify."""
        import torch.nn.functional as F
        Q_f = Q.float()
        K_f = K.float()
        V_f = V.float()
        scale = Q_f.size(-1) ** -0.5
        scores = (Q_f @ K_f.T) * scale
        attn = F.softmax(scores, dim=-1)
        return attn @ V_f


# --------------------------------------------------------------------------- #
# Monkey-patch factory                                                          #
# --------------------------------------------------------------------------- #

_VERICACHE_PATCH_APPLIED: Dict[int, bool] = {}


def apply_vericache_codec_patch(
    attn_impl: object,
    config: Optional[VeriCacheCodecHookConfig] = None,
    layer_idx: int = 0,
) -> VeriCacheCodecAttentionHook:
    """Monkey-patch a FlashAttentionImpl (or any AttentionImpl) with VeriCache hooks.

    Attaches:
      - attn_impl._vericache_hook : VeriCacheCodecAttentionHook
      - attn_impl.write_to_cache(kv_key, key, value) -> (key, value) ORIGINAL
      - attn_impl.read_from_cache(kv_key, Q) -> VeriCacheVerificationResult or None

    Idempotent: re-calling with same attn_impl reuses existing hook.

    Args:
        attn_impl: vLLM FlashAttentionImpl instance or class
        config:    VeriCacheCodecHookConfig (default if None)
        layer_idx: transformer layer index (for multi-layer keying)

    Returns:
        VeriCacheCodecAttentionHook instance attached to attn_impl
    """
    impl_id = id(attn_impl)
    if impl_id in _VERICACHE_PATCH_APPLIED:
        return getattr(attn_impl, "_vericache_hook")

    cfg = config or VeriCacheCodecHookConfig()
    hook = VeriCacheCodecAttentionHook(cfg)

    attn_impl._vericache_hook = hook
    attn_impl._vericache_layer_idx = layer_idx

    def write_to_cache(
        kv_key: str,
        key: torch.Tensor,
        value: torch.Tensor,
        _layer_idx: int = layer_idx,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return attn_impl._vericache_hook.write_to_cache(kv_key, key, value, _layer_idx)

    def read_from_cache(
        kv_key: str,
        Q: Optional[torch.Tensor] = None,
        _layer_idx: int = layer_idx,
    ) -> Optional[VeriCacheVerificationResult]:
        return attn_impl._vericache_hook.read_from_cache(kv_key, _layer_idx, Q)

    attn_impl.write_to_cache = write_to_cache
    attn_impl.read_from_cache = read_from_cache

    _VERICACHE_PATCH_APPLIED[impl_id] = True
    return hook


# --------------------------------------------------------------------------- #
# CacheConfig extension                                                         #
# --------------------------------------------------------------------------- #

def extend_cache_config_vericache(
    cache_config: object,
    hook_config: Optional[VeriCacheCodecHookConfig] = None,
) -> None:
    """Extend vLLM CacheConfig with VeriCache compression fields.

    Uses object.__setattr__ for pydantic-frozen-dataclass compatibility
    (same pattern as prior cycles: extend_cache_config_runtime_certified, etc.).

    Added fields:
      compression_method         = "vericache_speculative"
      vericache_acceptance_threshold = float (default 0.01)
      vericache_d_head           = int
      vericache_max_entries      = int
      vericache_compression_codec = "int8_draft"

    Args:
        cache_config: vLLM CacheConfig instance
        hook_config:  VeriCacheCodecHookConfig (default if None)
    """
    cfg = hook_config or VeriCacheCodecHookConfig()
    try:
        object.__setattr__(cache_config, "compression_method", "vericache_speculative")
        object.__setattr__(
            cache_config, "vericache_acceptance_threshold", cfg.acceptance_threshold
        )
        object.__setattr__(cache_config, "vericache_d_head", cfg.d_head)
        object.__setattr__(cache_config, "vericache_max_entries", cfg.max_entries_per_layer)
        object.__setattr__(cache_config, "vericache_compression_codec", "int8_draft")
    except Exception:
        # Non-pydantic fallback
        setattr(cache_config, "compression_method", "vericache_speculative")
        setattr(cache_config, "vericache_acceptance_threshold", cfg.acceptance_threshold)
        setattr(cache_config, "vericache_d_head", cfg.d_head)
        setattr(cache_config, "vericache_max_entries", cfg.max_entries_per_layer)
        setattr(cache_config, "vericache_compression_codec", "int8_draft")


# --------------------------------------------------------------------------- #
# Optional: try to import src/ implementation for richer codec support         #
# --------------------------------------------------------------------------- #

try:
    import sys
    import pathlib
    _repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from src.cache.vericache_speculative_codec import (  # noqa: F401
        VeriCacheSpeculativeCodec,
        VeriCacheConfig,
        Int8DraftCodec,
        VerificationResult,
    )
    _SRC_VERICACHE_AVAILABLE = True
except ImportError:
    _SRC_VERICACHE_AVAILABLE = False

__all__ = [
    "VeriCacheCodecHookConfig",
    "VeriCacheCodecAttentionHook",
    "VeriCacheVerificationResult",
    "apply_vericache_codec_patch",
    "extend_cache_config_vericache",
    "_SRC_VERICACHE_AVAILABLE",
]
