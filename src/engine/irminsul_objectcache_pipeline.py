"""Cross-1 (A+B): Irminsul + ObjectCache end-to-end pipeline.

Integrates:
  - B-1: IrminsulMLASegmentCache (CDC chunking + δ-rotation)
  - B-2: CDCContentHashSegmentIDInterface (unified tier lookup)
  - A-1: ObjectCacheS3TierRouter (break-even S3 tier routing)

InferenceRunner-compatible API: exposes get_segments() and put_segment().
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from src.cache.arch_aware_noncontiguous_router import ArchitectureAwareNonContiguousRouter
from src.cache.cdc_content_hash_interface import CDCContentHashSegmentIDInterface
from src.cache.irminsul_mla_segment_cache import (
    IrminsulMLAConfig,
    IrminsulMLASegmentCache,
    apply_delta_rotation,
    cdc_chunk,
    cdc_segment_key,
)
from src.scheduler.objectcache_s3_tier_router import ObjectCacheS3TierRouter


# ------------------------------------------------------------------ #
# Pipeline configuration                                              #
# ------------------------------------------------------------------ #


@dataclass
class IrminsulObjectCachePipelineConfig:
    """Configuration for the A+B integrated pipeline."""
    layer_idx: int = 0
    enable_s3_tier: bool = False    # Whether to attempt S3 lookups
    mla_config: Optional[IrminsulMLAConfig] = None


# ------------------------------------------------------------------ #
# IrminsulObjectCachePipeline                                         #
# ------------------------------------------------------------------ #


class IrminsulObjectCachePipeline:
    """Irminsul (B-1) + ObjectCache (A-1) A+B integrated pipeline.

    Processing flow:
      Step 1 (B-1): CDC chunking + segment_id = SHA256(chunk_tokens)
      Step 2 (B-2): CDCContentHashSegmentIDInterface.lookup(segment_id)
                    → HBM/DRAM/SSD hit → return immediately
      Step 3 (A-1): Full miss → ObjectCacheS3TierRouter checks break-even
                    → if active, S3 mock lookup (real: layerwise RDMA)
      Step 4 (B-1): MLA model: apply_delta_rotation(k_r, delta) → position corrected
                    GQA/MHA: AdapShot RoPE re-encoding fallback
      Step 5: Return corrected (c_kv, k_r_corrected) → attention input

    InferenceRunner compatible: exposes get_segments() and put_segment().
    """

    def __init__(
        self,
        arch_router: ArchitectureAwareNonContiguousRouter,
        s3_router: ObjectCacheS3TierRouter,
        segment_interface: CDCContentHashSegmentIDInterface,
        config: IrminsulObjectCachePipelineConfig,
    ) -> None:
        self.arch_router = arch_router
        self.s3_router = s3_router
        self.segment_interface = segment_interface
        self.config = config

        # Use mla_config from router's mla_cache if not explicitly provided
        if config.mla_config is None and isinstance(
            arch_router.mla_cache, IrminsulMLASegmentCache
        ):
            self._mla_cfg = arch_router.mla_cache.config
        else:
            self._mla_cfg = config.mla_config or IrminsulMLAConfig()

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """InferenceRunner-compatible API.

        Returns same format as SegmentedHashCache.get_segments():
            hits: [(chunk_idx, kv_tensor), ...]
            misses: [chunk_idx, ...]
        """
        arch = self.arch_router.arch

        if arch == "MLA":
            return self._get_segments_mla(token_ids, layer_idx)
        else:
            return self._get_segments_fallback(token_ids, layer_idx)

    def _get_segments_mla(
        self,
        token_ids: List[int],
        layer_idx: int,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """MLA path: CDC → tier lookup → δ-rotation.

        Lookup order:
          1. CDCContentHashSegmentIDInterface (HBM/DRAM/SSD generic tier)
          2. IrminsulMLASegmentCache directly (MLA-specific store with δ-rotation)
          3. S3 tier (only if enabled and break-even condition met)
        """
        mla_cache = self.arch_router.mla_cache
        chunks = cdc_chunk(
            token_ids,
            avg_chunk_size=self._mla_cfg.avg_chunk_size,
            min_chunk_size=self._mla_cfg.min_chunk_size,
            max_chunk_size=self._mla_cfg.max_chunk_size,
        )

        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []
        position = 0

        for chunk_idx, chunk in enumerate(chunks):
            segment_id = cdc_segment_key(chunk)

            # Step 2a: Try generic tier lookup (HBM/DRAM/SSD via interface)
            kv_tensor, hit_tier = self.segment_interface.lookup(segment_id, layer_idx)

            if kv_tensor is not None:
                hits.append((chunk_idx, kv_tensor))
                position += len(chunk)
                continue

            # Step 2b: Try MLA-specific segment store (primary source of truth)
            result = mla_cache.get_mla_segment_with_delta_rotation(
                segment_id, position, layer_idx
            )
            if result is not None:
                c_kv, k_r_corrected = result
                combined = torch.cat([c_kv, k_r_corrected], dim=-1)
                hits.append((chunk_idx, combined))
                # Promote to interface HBM for future generic tier hits
                self.segment_interface.store(segment_id, combined, tier="HBM", layer_idx=layer_idx)
                position += len(chunk)
                continue

            # Step 3: S3 tier (only if enabled and router break-even active)
            if self.config.enable_s3_tier and self.s3_router.s3_tier_active:
                # S3 lookup is handled by the interface if s3_client is configured
                # (already attempted in step 2a above — no additional action needed)
                pass

            misses.append(chunk_idx)
            position += len(chunk)

        return hits, misses

    def _get_segments_fallback(
        self,
        token_ids: List[int],
        layer_idx: int,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """GQA/MHA fallback path: delegate to gqa_mha_cache."""
        gqa_cache = self.arch_router.gqa_mha_cache
        if hasattr(gqa_cache, "get_segments"):
            return gqa_cache.get_segments(token_ids, layer_idx)
        # Minimal fallback
        return [], list(range(max(1, len(token_ids) // 128)))

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """InferenceRunner-compatible segment storage.

        Routes to architecture-appropriate backend and also stores in
        the unified interface's HBM tier for cross-tier visibility.
        """
        arch = self.arch_router.arch

        if arch == "MLA":
            mla_cache = self.arch_router.mla_cache
            mla_cache.put_segment(token_ids, chunk_idx, kv, layer_idx)

            # Also register in the unified tier interface
            chunks = cdc_chunk(
                token_ids,
                avg_chunk_size=self._mla_cfg.avg_chunk_size,
                min_chunk_size=self._mla_cfg.min_chunk_size,
                max_chunk_size=self._mla_cfg.max_chunk_size,
            )
            if chunk_idx < len(chunks):
                segment_id = cdc_segment_key(chunks[chunk_idx])
                self.segment_interface.store(segment_id, kv, tier="HBM", layer_idx=layer_idx)
        else:
            gqa_cache = self.arch_router.gqa_mha_cache
            if hasattr(gqa_cache, "put_segment"):
                gqa_cache.put_segment(token_ids, chunk_idx, kv, layer_idx)
            else:
                # Generic fallback: store by chunk index key
                self.arch_router.put(f"chunk_{chunk_idx}_{layer_idx}", kv)

    def hit_rate(self) -> float:
        """Overall hit rate from the active architecture backend."""
        return self.arch_router.hit_rate()

    def memory_bytes(self) -> int:
        """Memory footprint from active backend."""
        return self.arch_router.memory_bytes()

    def reset_stats(self) -> None:
        """Reset hit/miss stats on the active backend."""
        self.arch_router.reset_stats()
