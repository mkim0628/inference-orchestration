"""Activity B-2: CDC Content Hash Integration — vLLM v1 Unified Block ID.

Maps CDC SHA256 content hashes (from IrminsulMLASegmentCache) to vLLM's
block ID space, enabling unified cross-tier segment addressing.

Integration approach:
  - Wraps vLLM's block_pool as the HBM tier
  - Provides a CDCHashBlockRegistry that maps segment_id → vLLM block_id
  - Transparent tier waterfall: HBM → DRAM → SSD → S3 → miss

vLLM version: 0.21.0
Activity: B-2 (CDC Content Hash Unified Segment ID Interface)
Source: src/cache/cdc_content_hash_interface.py
"""

from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# CDC key helper (standalone, no src/ import)
# ---------------------------------------------------------------------------


def _cdc_segment_key(chunk_tokens: List[int]) -> str:
    """SHA256 of chunk token bytes — position-independent key."""
    raw = struct.pack(f"{len(chunk_tokens)}I", *chunk_tokens)
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Segment metadata
# ---------------------------------------------------------------------------


@dataclass
class VLLMSegmentMetadata:
    """Metadata for a cached segment, mapped to vLLM block IDs."""
    segment_id: str           # SHA256(CDC_chunk_token_bytes)
    source_position: int
    n_tokens: int
    model_arch: str           # "MLA" | "GQA" | "MHA"
    storage_tier: str         # "HBM" | "DRAM" | "SSD" | "S3"
    vllm_block_id: int = -1   # vLLM block_id (-1 = not mapped to vLLM block)
    layer_idx: int = 0


# ---------------------------------------------------------------------------
# CDCHashBlockRegistry — unified B-1 + vLLM block address space
# ---------------------------------------------------------------------------


class CDCHashBlockRegistry:
    """Maps CDC content hashes to vLLM block IDs and tier tensors.

    This bridges the research implementation's SHA256 segment key space with
    vLLM 0.21.0's block_pool integer block ID space.

    Design:
      - HBM tier: dict[segment_id → torch.Tensor] (mirrors vLLM GPU pages)
      - DRAM/SSD/S3: optional external CacheStore objects (same interface)
      - segment_id_to_block_id: bidirectional mapping for vLLM block table

    Lookup order: HBM → DRAM → SSD → S3 → miss
    S3 unavailability is silently treated as miss (no exception).
    """

    def __init__(
        self,
        model_name: str = "default",
        s3_client: Optional[Any] = None,      # boto3.client('s3') or None
        s3_bucket: str = "kvcache",
        max_hbm_segments: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.max_hbm_segments = max_hbm_segments

        # HBM tensor store: segment_id_layer_key → tensor
        self._hbm_store: Dict[str, torch.Tensor] = {}

        # Optional lower-tier stores (must expose .get(key) / .put(key, val))
        self._dram_store: Optional[Any] = None
        self._ssd_store: Optional[Any] = None

        # segment_id → vLLM block_id mapping (populated by allocator)
        self._segment_to_block: Dict[str, int] = {}
        self._block_to_segment: Dict[int, str] = {}

        # Per-tier hit counters
        self._tier_hits: Dict[str, int] = {
            "HBM": 0, "DRAM": 0, "SSD": 0, "S3": 0
        }
        self._misses = 0

    # -- Tier attachment ---------------------------------------------------

    def attach_dram_store(self, store: Any) -> None:
        """Attach a DRAM-tier CacheStore (must have .get/.put API)."""
        self._dram_store = store

    def attach_ssd_store(self, store: Any) -> None:
        """Attach an SSD-tier CacheStore (must have .get/.put API)."""
        self._ssd_store = store

    # -- vLLM block ID mapping --------------------------------------------

    def register_segment_block(self, segment_id: str, block_id: int) -> None:
        """Register a mapping from content-hash segment_id to vLLM block_id."""
        self._segment_to_block[segment_id] = block_id
        self._block_to_segment[block_id] = segment_id

    def get_block_id_for_segment(self, segment_id: str) -> Optional[int]:
        """Return vLLM block_id for a segment_id, or None if not mapped."""
        return self._segment_to_block.get(segment_id)

    def get_segment_for_block_id(self, block_id: int) -> Optional[str]:
        """Return segment_id for a vLLM block_id, or None if not mapped."""
        return self._block_to_segment.get(block_id)

    # -- Unified lookup API -----------------------------------------------

    def lookup(
        self,
        segment_id: str,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Tier-waterfall lookup: HBM → DRAM → SSD → S3 → miss.

        Returns (tensor, tier_name) on hit, (None, "miss") on miss.
        Maps to: CDCContentHashSegmentIDInterface.lookup()
        """
        key = self._tier_key(segment_id, layer_idx)

        # HBM
        kv = self._hbm_store.get(key)
        if kv is not None:
            self._tier_hits["HBM"] += 1
            return kv, "HBM"

        # DRAM
        if self._dram_store is not None:
            kv = self._dram_store.get(key)
            if kv is not None:
                self._tier_hits["DRAM"] += 1
                self._hbm_promote(key, kv)
                return kv, "DRAM"

        # SSD
        if self._ssd_store is not None:
            kv = self._ssd_store.get(key)
            if kv is not None:
                self._tier_hits["SSD"] += 1
                self._hbm_promote(key, kv)
                return kv, "SSD"

        # S3
        if self.s3_client is not None:
            kv = self._s3_get(segment_id, layer_idx)
            if kv is not None:
                self._tier_hits["S3"] += 1
                self._hbm_promote(key, kv)
                return kv, "S3"

        self._misses += 1
        return None, "miss"

    def store(
        self,
        segment_id: str,
        kv: torch.Tensor,
        tier: str = "HBM",
        layer_idx: int = 0,
    ) -> None:
        """Store KV tensor in the specified tier.

        Maps to: CDCContentHashSegmentIDInterface.store()
        """
        key = self._tier_key(segment_id, layer_idx)
        if tier == "HBM":
            self._hbm_promote(key, kv)
        elif tier == "DRAM" and self._dram_store is not None:
            self._dram_store.put(key, kv)
        elif tier == "SSD" and self._ssd_store is not None:
            self._ssd_store.put(key, kv)
        elif tier == "S3" and self.s3_client is not None:
            self._s3_put(segment_id, layer_idx, kv)

    # -- S3 object key convention -----------------------------------------

    def s3_object_key(self, segment_id: str, layer_idx: int) -> str:
        """S3 object key: {model_name}/{segment_id}_{layer_idx}.kvcache"""
        return f"{self.model_name}/{segment_id}_{layer_idx}.kvcache"

    @staticmethod
    def make_segment_id(chunk_tokens: List[int]) -> str:
        """Make a position-independent segment ID from chunk token content."""
        return _cdc_segment_key(chunk_tokens)

    # -- Stats -------------------------------------------------------------

    def tier_hit_stats(self) -> Dict[str, int]:
        return dict(self._tier_hits)

    def total_misses(self) -> int:
        return self._misses

    # -- Private helpers ---------------------------------------------------

    def _tier_key(self, segment_id: str, layer_idx: int) -> str:
        return f"{segment_id}_{layer_idx}"

    def _hbm_promote(self, key: str, kv: torch.Tensor) -> None:
        """Store in HBM with LRU eviction when at capacity."""
        if len(self._hbm_store) >= self.max_hbm_segments:
            # Remove an arbitrary oldest entry (dict insertion order since 3.7)
            oldest = next(iter(self._hbm_store))
            del self._hbm_store[oldest]
        self._hbm_store[key] = kv.detach().clone()

    def _s3_get(self, segment_id: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Fetch tensor from S3; silently returns None on any error."""
        try:
            obj_key = self.s3_object_key(segment_id, layer_idx)
            response = self.s3_client.get_object(  # type: ignore[union-attr]
                Bucket=self.s3_bucket, Key=obj_key
            )
            buf = io.BytesIO(response["Body"].read())
            return torch.load(buf, weights_only=True)
        except Exception:
            return None

    def _s3_put(self, segment_id: str, layer_idx: int, kv: torch.Tensor) -> None:
        """Upload tensor to S3; silently ignores errors."""
        try:
            obj_key = self.s3_object_key(segment_id, layer_idx)
            buf = io.BytesIO()
            torch.save(kv, buf)
            buf.seek(0)
            self.s3_client.put_object(  # type: ignore[union-attr]
                Bucket=self.s3_bucket, Key=obj_key, Body=buf.read()
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# install_cdc_hash_registry — attach to vLLM KVCacheManager
# ---------------------------------------------------------------------------


def install_cdc_hash_registry(
    manager: object,
    model_name: str = "default",
    s3_client: Optional[Any] = None,
    s3_bucket: str = "kvcache",
    max_hbm_segments: int = 4096,
) -> CDCHashBlockRegistry:
    """Create and attach a CDCHashBlockRegistry to a vLLM KVCacheManager.

    After this call:
        manager._cdc_registry → CDCHashBlockRegistry instance

    The registry provides transparent cross-tier lookup using CDC SHA256 keys,
    and can map segment_ids to vLLM integer block_ids.

    Returns the CDCHashBlockRegistry for direct use.
    """
    registry = CDCHashBlockRegistry(
        model_name=model_name,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
        max_hbm_segments=max_hbm_segments,
    )
    manager._cdc_registry = registry  # type: ignore[attr-defined]
    return registry
