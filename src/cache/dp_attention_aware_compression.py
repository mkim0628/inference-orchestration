"""Activity C: DPAttentionAwareCompressionSelector — DP Attention-aware compression policy.

Detects DP Attention environment state (n_gpus, dp_attn_enabled) and selects the
optimal compression codec per-environment:
  - Single GPU / DP Attention disabled: high-compression codec preferred.
  - Multi-GPU + DP Attention enabled: marginal-utility-based selective compression
    (skip if marginal_utility < threshold, because DP Attention already reduces
    effective KV replicas from N to 1).

Reuses existing codec implementations as wrappers; no new codec abstractions.
Implements CacheStore fully.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from src.cache.base import CacheStore


@dataclass
class DPAttentionCompressionConfig:
    # DP Attention environment
    dp_attn_enabled: bool = False        # overridden by DP_ATTN_ENABLED env var
    n_gpus: int = 1                      # overridden by auto_detect_gpus if True
    auto_detect_gpus: bool = True        # call torch.cuda.device_count() at init

    # Codec selection policy
    single_gpu_codec: str = "global_retention"   # codec name for single-GPU path
    dp_attn_codec: str = "global_retention"      # codec name for DP Attention path
    dp_attn_compression_skip_threshold: float = 0.5  # skip compression when marginal_utility < this

    max_entries: int = 1000
    seed: int = 42


class DPAttentionAwareCompressionSelector(CacheStore):
    """DP Attention-aware environment-adaptive compression policy selector.

    Activity C: KV Cache Compression.
    CacheStore interface fully implemented.

    Environment detection:
      - n_gpus: torch.cuda.device_count() when auto_detect_gpus=True.
      - dp_attn_enabled: config flag or DP_ATTN_ENABLED="1" environment variable.
      - effective_kv_replicas = n_gpus (DP Attention disabled) or 1 (enabled).

    Compression policy:
      - effective_kv_replicas > 1 (single GPU or DP Attention disabled):
          High-compression codec preferred; each compressed KV directly reduces total memory.
      - effective_kv_replicas == 1 (DP Attention enabled):
          Marginal utility = 1 - 1/compression_ratio.
          Skip compression when marginal_utility < dp_attn_compression_skip_threshold.

    Dual savings quantification:
      effective_reduction = 1 - 1 / (effective_kv_replicas * compression_ratio)

    Accuracy preservation:
      Lower compression intensity under DP Attention → smaller accuracy delta.
      Each registered codec's accuracy guarantee is inherited.

    Evaluation criteria (evaluation_criteria.md §4):
      - Accuracy preservation: perplexity change ±1% (MANDATORY)
      - KV Memory Reduction ≥ −30%
      - Effective Context Length ≥ 2×
    """

    def __init__(
        self,
        config: DPAttentionCompressionConfig,
        codec_registry: Optional[Dict[str, CacheStore]] = None,
        env_change_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        torch.manual_seed(config.seed)

        # GPU count detection
        self._n_gpus = config.n_gpus
        if config.auto_detect_gpus:
            try:
                detected = torch.cuda.device_count()
                self._n_gpus = max(1, detected)
            except Exception:
                self._n_gpus = 1

        # DP Attention state (env var takes precedence over config)
        env_flag = os.environ.get("DP_ATTN_ENABLED", "")
        self._dp_attn_enabled = config.dp_attn_enabled or env_flag in ("1", "true", "True")

        self._effective_kv_replicas = 1 if self._dp_attn_enabled else self._n_gpus

        self._codec_registry: Dict[str, CacheStore] = codec_registry or {}
        self._env_change_callback = env_change_callback

        # Fallback store when no codec matches or compression is skipped
        self._fallback_store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0
        # Per-codec compression ratio tracking (used for marginal utility computation)
        self._codec_compression_ratios: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Environment-aware codec selection                                    #
    # ------------------------------------------------------------------ #

    def effective_kv_replicas(self) -> int:
        """Return current effective_kv_replicas value."""
        return self._effective_kv_replicas

    def select_codec(self) -> Optional[CacheStore]:
        """Select compression codec based on current environment.

        Algorithm:
          1. effective_kv_replicas > 1 (single GPU / DP Attention disabled):
               → config.single_gpu_codec (high-compression first).
          2. effective_kv_replicas == 1 (DP Attention enabled):
               → marginal_utility = 1 - 1/compression_ratio.
               → if marginal_utility < dp_attn_compression_skip_threshold: return None.
               → else: config.dp_attn_codec.
        """
        if self._effective_kv_replicas > 1:
            codec_name = self.config.single_gpu_codec
        else:
            ratio = self._codec_compression_ratios.get(self.config.dp_attn_codec, 2.0)
            marginal_utility = 1.0 - 1.0 / max(ratio, 1.0)
            if marginal_utility < self.config.dp_attn_compression_skip_threshold:
                return None
            codec_name = self.config.dp_attn_codec

        return self._codec_registry.get(codec_name)

    def register_codec(
        self,
        name: str,
        codec: CacheStore,
        compression_ratio: float = 2.0,
    ) -> None:
        """Register a codec in the registry with its expected compression ratio."""
        self._codec_registry[name] = codec
        self._codec_compression_ratios[name] = compression_ratio

    def register_env_change_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when DP Attention state changes."""
        self._env_change_callback = callback

    def update_dp_attn_state(
        self,
        dp_attn_enabled: bool,
        n_gpus: Optional[int] = None,
    ) -> None:
        """Runtime DP Attention state change — auto-switches compression policy.

        Algorithm:
          1. Update _dp_attn_enabled and optionally _n_gpus.
          2. Recompute _effective_kv_replicas.
          3. Invoke env_change_callback if registered.
        """
        self._dp_attn_enabled = dp_attn_enabled
        if n_gpus is not None:
            self._n_gpus = n_gpus
        self._effective_kv_replicas = 1 if self._dp_attn_enabled else self._n_gpus
        if self._env_change_callback is not None:
            self._env_change_callback()

    def effective_memory_reduction_ratio(self, compression_ratio: float) -> float:
        """Dual savings formula: 1 - 1 / (effective_kv_replicas × compression_ratio).

        DP Attention (N-GPU) + compression (C×) → 1 - 1/(N*C).
        """
        return 1.0 - 1.0 / max(self._effective_kv_replicas * compression_ratio, 1.0)

    # ------------------------------------------------------------------ #
    # CacheStore abstract methods                                          #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Environment-aware compression then store."""
        self._total_bytes_original += value.nbytes
        compressed = self.compression_hook(key, value)
        self._total_bytes_stored += compressed.nbytes

        codec = self.select_codec()
        if codec is not None:
            codec.put(key, compressed)
        else:
            if len(self._fallback_store) >= self.config.max_entries:
                self.evict()
            self._fallback_store[key] = compressed.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        codec = self.select_codec()
        result: Optional[torch.Tensor] = None
        if codec is not None:
            result = codec.get(key)
        if result is None:
            result = self._fallback_store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        codec = self.select_codec()
        if codec is not None:
            return codec.evict()
        if self._fallback_store:
            _, v = self._fallback_store.popitem(last=False)
            return v.nbytes
        return 0

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        codec = self.select_codec()
        if codec is not None:
            return codec.memory_bytes()
        return sum(v.nbytes for v in self._fallback_store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0
        for codec in self._codec_registry.values():
            codec.reset_stats()
        self._fallback_store.clear()

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Delegate to selected codec's compression_hook, or return identity."""
        codec = self.select_codec()
        if codec is not None and hasattr(codec, "compression_hook"):
            return codec.compression_hook(key, value)
        return value

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def memory_reduction_ratio(self) -> float:
        """Actual memory reduction ratio measured in bytes."""
        if self._total_bytes_original == 0:
            return 0.0
        return 1.0 - self._total_bytes_stored / self._total_bytes_original

    def dp_attn_compression_matrix_entry(
        self,
        codec_name: str,
        compression_ratio: float,
    ) -> Dict:
        """Return a single entry for the DP Attention × compression experiment matrix.

        Records into results/<exp>/dp_attn_compression_matrix.json externally.
        """
        return {
            "dp_attn_enabled": self._dp_attn_enabled,
            "n_gpus": self._n_gpus,
            "effective_kv_replicas": self._effective_kv_replicas,
            "codec_name": codec_name,
            "compression_ratio": compression_ratio,
            "effective_memory_reduction": self.effective_memory_reduction_ratio(compression_ratio),
            "actual_memory_reduction": self.memory_reduction_ratio(),
        }
