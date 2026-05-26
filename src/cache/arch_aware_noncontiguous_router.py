"""Activity B-1: Architecture-aware non-contiguous KV cache router.

Detects MLA vs GQA/MHA from ModelConfig and routes cache lookups accordingly.
MLA → IrminsulMLASegmentCache (δ-rotation)
GQA/MHA → fallback CacheStore (RoPEReencodingNonContiguousCache or SegmentedHashCache)
"""

import fnmatch
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from src.cache.base import CacheStore
from src.cache.irminsul_mla_segment_cache import IrminsulMLASegmentCache


# ------------------------------------------------------------------ #
# Model configuration                                                 #
# ------------------------------------------------------------------ #


@dataclass
class ModelConfig:
    """Model architecture parameters for attention type detection."""
    model_name: str
    kv_lora_rank: Optional[int] = None      # MLA-only: c_KV compression rank
    qk_rope_head_dim: Optional[int] = None  # MLA-only: k_r dimension (=64)
    num_kv_heads: int = 8
    d_head: int = 64
    rope_base: float = 10000.0


# ------------------------------------------------------------------ #
# Architecture registry (loads from configs/arch_registry.yaml)      #
# ------------------------------------------------------------------ #

_ARCH_REGISTRY: Optional[List[Dict]] = None


def _load_arch_registry() -> List[Dict]:
    global _ARCH_REGISTRY
    if _ARCH_REGISTRY is not None:
        return _ARCH_REGISTRY

    registry_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "arch_registry.yaml"
    )
    registry_path = os.path.normpath(registry_path)

    if os.path.exists(registry_path):
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        _ARCH_REGISTRY = data.get("architectures", [])
    else:
        _ARCH_REGISTRY = []

    return _ARCH_REGISTRY


def detect_attention_arch(model_config: ModelConfig) -> str:
    """Detect attention architecture from model configuration.

    Returns: "MLA" | "GQA" | "MHA"

    Priority order:
      1. arch_registry.yaml model_name glob match
      2. kv_lora_rank + qk_rope_head_dim=64 → MLA
      3. num_kv_heads <= 4 → GQA
      4. default → MHA
    """
    registry = _load_arch_registry()
    name_lower = model_config.model_name.lower()

    for entry in registry:
        pattern = entry.get("model_pattern", "")
        if fnmatch.fnmatch(name_lower, pattern.lower()):
            return entry.get("arch", "MHA")

    # Heuristic detection from config fields
    if (
        model_config.kv_lora_rank is not None
        and model_config.qk_rope_head_dim is not None
        and model_config.qk_rope_head_dim == 64
    ):
        return "MLA"

    if model_config.num_kv_heads <= 4:
        return "GQA"

    return "MHA"


# ------------------------------------------------------------------ #
# ArchitectureAwareNonContiguousRouter                                #
# ------------------------------------------------------------------ #


class ArchitectureAwareNonContiguousRouter(CacheStore):
    """Routes non-contiguous KV lookups to the appropriate backend.

    MLA models → IrminsulMLASegmentCache (δ-rotation, training-free)
    GQA/MHA models → fallback CacheStore (RoPE re-encoding or segmented hash)

    Fully implements CacheStore interface by delegating to the active backend.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        mla_cache: IrminsulMLASegmentCache,
        gqa_mha_cache: CacheStore,
    ) -> None:
        self.model_config = model_config
        self.mla_cache = mla_cache
        self.gqa_mha_cache = gqa_mha_cache
        self._arch = detect_attention_arch(model_config)

    # ---------------------------------------------------------------- #
    # CacheStore abstract methods                                       #
    # ---------------------------------------------------------------- #

    def put(self, key: str, value: torch.Tensor) -> None:
        self._active_cache.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._active_cache.get(key)

    def evict(self) -> int:
        return self._active_cache.evict()

    def hit_rate(self) -> float:
        return self._active_cache.hit_rate()

    def memory_bytes(self) -> int:
        return self._active_cache.memory_bytes()

    def reset_stats(self) -> None:
        self._active_cache.reset_stats()

    # ---------------------------------------------------------------- #
    # Properties and routing                                            #
    # ---------------------------------------------------------------- #

    @property
    def _active_cache(self) -> CacheStore:
        if self._arch == "MLA":
            return self.mla_cache
        return self.gqa_mha_cache

    @property
    def arch(self) -> str:
        """Currently detected architecture type."""
        return self._arch

    def get_segments_routed(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Route segment lookup to architecture-appropriate cache backend.

        MLA: IrminsulMLASegmentCache.get_segments_mla() — returns (chunk_idx, c_kv, k_r)
             collapsed to (chunk_idx, combined_tensor) for compatibility.
        GQA/MHA: falls back to gqa_mha_cache.get_segments() if available,
                 otherwise returns empty hits with all chunks as misses.

        Returns:
            hits: [(chunk_idx, kv_tensor), ...]
            miss_chunk_indices: [chunk_idx, ...]
        """
        if self._arch == "MLA":
            return self.mla_cache.get_segments(token_ids, layer_idx)

        if hasattr(self.gqa_mha_cache, "get_segments"):
            return self.gqa_mha_cache.get_segments(token_ids, layer_idx)

        # Minimal fallback for CacheStore implementations without get_segments
        return [], list(range(max(1, len(token_ids) // 128)))

    # InferenceRunner-compatible segment API (delegates to active cache)
    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """InferenceRunner-compatible get_segments API."""
        return self.get_segments_routed(token_ids, 0, layer_idx)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """InferenceRunner-compatible put_segment API."""
        if self._arch == "MLA":
            self.mla_cache.put_segment(token_ids, chunk_idx, kv, layer_idx)
        elif hasattr(self.gqa_mha_cache, "put_segment"):
            self.gqa_mha_cache.put_segment(token_ids, chunk_idx, kv, layer_idx)
