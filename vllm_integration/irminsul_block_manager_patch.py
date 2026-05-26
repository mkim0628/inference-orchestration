"""Activity B (2026-05-26): Non-Contiguous MLA Segment Reuse — vLLM v1 Block Manager Patch.

Extends vLLM 0.21.0's KVCacheManager with:
  1. IrminsulMLASegmentMixin — CDC-based non-contiguous block lookup (B-1)
  2. CDCHashBlockRegistry attachment — unified cross-tier addressing (B-2)
  3. build_noncontiguous_block_table() — padded block table for non-contiguous vLLM blocks
  4. Architecture detection from VllmConfig (MLA / GQA / MHA)

vLLM integration point: vllm.v1.core.kv_cache_manager.KVCacheManager

Design:
  - vLLM's own prefix caching (block_pool SHA256 trie) handles contiguous prefixes
  - This patch adds a secondary CDC segment store for arbitrary non-contiguous hits
  - Non-contiguous hits return (chunk_idx, c_kv, k_r_corrected) tuples
  - block_table padded with -1 sentinel (vLLM PagedAttention convention)
  - No vLLM source modification: monkey-patch / mixin approach

vLLM version: 0.21.0
Activity: B (Non-Contiguous KV Cache Reuse — MLA δ-rotation + CDC hashing)
Sources:
  src/cache/irminsul_mla_segment_cache.py (B-1)
  src/cache/arch_aware_noncontiguous_router.py (B-1 router)
  src/cache/cdc_content_hash_interface.py (B-2)
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm_integration.mla_segment_cache_integration import (
    IrminsulMLASegmentMixin,
    install_irminsul_mla_hooks,
)
from vllm_integration.cdc_hash_integration import (
    CDCHashBlockRegistry,
    install_cdc_hash_registry,
)


# ---------------------------------------------------------------------------
# Architecture detection helper
# ---------------------------------------------------------------------------


def detect_vllm_model_arch(vllm_config: Any) -> str:
    """Detect attention architecture from vLLM VllmConfig.

    Returns "MLA" | "GQA" | "MHA".

    Priority:
      1. model_config.architectures list (DeepSeek* / Kimi* → MLA)
      2. kv_lora_rank + qk_rope_head_dim=64 → MLA
      3. num_kv_heads heuristic (≤4 → GQA)
      4. default → MHA
    """
    try:
        model_cfg = vllm_config.model_config
        # Check architecture string list
        archs = getattr(model_cfg, "architectures", None) or []
        for arch in archs:
            arch_lower = str(arch).lower()
            if any(kw in arch_lower for kw in ("deepseek", "mla", "kimi", "moonlight")):
                return "MLA"

        # Check for MLA-specific numeric fields
        kv_lora_rank = getattr(model_cfg, "kv_lora_rank", None)
        qk_rope_head_dim = getattr(model_cfg, "qk_rope_head_dim", None)
        if kv_lora_rank is not None and qk_rope_head_dim == 64:
            return "MLA"

        # GQA heuristic
        num_kv_heads = getattr(model_cfg, "num_key_value_heads", None)
        if num_kv_heads is not None and num_kv_heads <= 4:
            return "GQA"

    except AttributeError:
        pass

    return "MHA"


# ---------------------------------------------------------------------------
# Non-contiguous block table builder
# ---------------------------------------------------------------------------


def build_noncontiguous_block_table(
    hits: List[Tuple[int, torch.Tensor, torch.Tensor]],
    max_blocks: int,
    block_size: int = 16,
) -> torch.Tensor:
    """Build a vLLM-compatible block table tensor for non-contiguous MLA segments.

    vLLM's PagedAttention expects block_table: int64 [1, max_blocks] with
    valid block IDs at covered positions and -1 as the empty sentinel.

    For non-contiguous MLA reuse the actual KV data is already in the
    (c_kv, k_r_corrected) tensors from find_noncontiguous_mla_hits(); this
    table encodes *which chunks* were hit so the model runner can merge
    segment data with full-compute outputs.

    Args:
        hits: [(chunk_idx, c_kv, k_r_corrected), ...]
        max_blocks: block table width (pad to this length with -1)
        block_size: tokens per vLLM block (default 16)

    Returns:
        block_table: int64 tensor [1, max_blocks]
    """
    table = torch.full((1, max_blocks), -1, dtype=torch.int64)
    for chunk_idx, _c_kv, _k_r in hits:
        if chunk_idx < max_blocks:
            table[0, chunk_idx] = chunk_idx
    return table


def build_mla_kv_batch(
    hits: List[Tuple[int, torch.Tensor, torch.Tensor]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Concatenate (c_kv, k_r_corrected) tensors from all hit chunks.

    Returns (c_kv_batch, k_r_batch) shaped [total_tokens, dim],
    or None if hits is empty.

    The concatenated tensors are suitable for direct injection into the
    FlashMLA / FlashInfer MLA attention kernels.
    """
    if not hits:
        return None

    c_kvs = [c_kv for _, c_kv, _ in hits]
    k_rs = [k_r for _, _, k_r in hits]
    c_kv_batch = torch.cat(c_kvs, dim=0)
    k_r_batch = torch.cat(k_rs, dim=0)
    return c_kv_batch, k_r_batch


# ---------------------------------------------------------------------------
# Main patch class
# ---------------------------------------------------------------------------


class IrminsulNonContiguousBlockManagerPatch:
    """Coordinates the B-1 + B-2 patches on a vLLM KVCacheManager (2026-05-26).

    Usage:
        patch = IrminsulNonContiguousBlockManagerPatch(vllm_config)
        registry = patch.install(kv_cache_manager)

    After install, on each prefill step:
        hits, miss_chunks = manager.find_noncontiguous_mla_hits(token_ids)
        kv_batch        = build_mla_kv_batch(hits)          # for kernel
        block_table     = build_noncontiguous_block_table(hits, max_blocks)
        # manager.find_mla_segment() / manager.store_mla_segment() for per-layer ops
        # registry.lookup() / registry.store() for cross-tier ops

    Preserves all existing vLLM KVCacheManager behaviour.
    """

    def __init__(
        self,
        vllm_config: Any = None,
        avg_chunk_size: int = 256,
        min_chunk_size: int = 64,
        max_chunk_size: int = 1024,
        rope_base: float = 10000.0,
        k_r_dim: int = 64,
        max_entries: int = 2000,
        model_name: str = "default",
        s3_client: Any = None,
        s3_bucket: str = "kvcache",
    ) -> None:
        self.vllm_config = vllm_config
        self.avg_chunk_size = avg_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.rope_base = rope_base
        self.k_r_dim = k_r_dim
        self.max_entries = max_entries
        self.model_name = model_name
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket

        self._arch = "MHA"
        if vllm_config is not None:
            self._arch = detect_vllm_model_arch(vllm_config)

    def install(self, manager: Any) -> CDCHashBlockRegistry:
        """Install B-1 MLA segment mixin and B-2 CDC hash registry.

        Returns CDCHashBlockRegistry for downstream use.
        """
        # B-1: Irminsul MLA segment store (δ-rotation + CDC)
        install_irminsul_mla_hooks(
            manager,
            avg_chunk_size=self.avg_chunk_size,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            rope_base=self.rope_base,
            k_r_dim=self.k_r_dim,
            max_entries=self.max_entries,
        )

        # B-2: CDC hash registry (unified cross-tier address space)
        registry = install_cdc_hash_registry(
            manager,
            model_name=self.model_name,
            s3_client=self.s3_client,
            s3_bucket=self.s3_bucket,
        )

        # Expose detected arch for downstream components
        manager._irminsul_model_arch = self._arch  # type: ignore[attr-defined]

        return registry

    @property
    def arch(self) -> str:
        """Detected attention architecture for the configured vLLM model."""
        return self._arch
