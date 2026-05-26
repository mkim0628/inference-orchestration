"""Activity B-2: CDC content-hash unified segment ID interface.

Provides a unified address space (SHA256 content hash) for Irminsul (B-1)
and ObjectCache S3 tier (A-1), enabling transparent cross-tier segment lookup.

Lookup order: HBM → DRAM → SSD → S3 → miss (recompute)
"""

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.irminsul_mla_segment_cache import cdc_segment_key


# ------------------------------------------------------------------ #
# Segment metadata                                                    #
# ------------------------------------------------------------------ #


@dataclass
class SegmentMetadata:
    """Metadata for a cached segment across tiers."""
    segment_id: str                  # SHA256(CDC_chunk_token_bytes)
    source_position: int
    n_tokens: int
    model_arch: str                  # "MLA" | "GQA" | "MHA"
    storage_tier: str                # "HBM" | "DRAM" | "SSD" | "S3"
    layer_idx: int = 0


# ------------------------------------------------------------------ #
# CDCContentHashSegmentIDInterface                                    #
# ------------------------------------------------------------------ #


class CDCContentHashSegmentIDInterface:
    """Irminsul (B-1) + ObjectCache (A-1) unified segment address system.

    Unified key: SegmentID = SHA256(CDC_chunk_token_bytes)
    Lookup order: HBM → DRAM → SSD → S3 → recompute

    S3 access is abstracted via boto3/minio client.
    When S3 is unavailable, falls back to "miss" without raising exceptions.
    """

    def __init__(
        self,
        hbm_cache: CacheStore,
        dram_cache: Optional[CacheStore] = None,
        ssd_cache: Optional[CacheStore] = None,
        s3_client: Optional[object] = None,   # boto3.client('s3') or None
        s3_bucket: str = "kvcache",
        model_name: str = "default",
    ) -> None:
        self.hbm_cache = hbm_cache
        self.dram_cache = dram_cache
        self.ssd_cache = ssd_cache
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.model_name = model_name

        # Track per-tier hit counts for diagnostics
        self._tier_hits: Dict[str, int] = {"HBM": 0, "DRAM": 0, "SSD": 0, "S3": 0}
        self._misses = 0

    def lookup(
        self,
        segment_id: str,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Look up segment through all tiers in order.

        Returns:
            (kv_tensor, tier_name): tensor and tier that had the hit
            (None, "miss"): all tiers missed
        """
        # HBM (always present)
        kv = self.hbm_cache.get(self._tier_key(segment_id, layer_idx))
        if kv is not None:
            self._tier_hits["HBM"] += 1
            return kv, "HBM"

        # DRAM
        if self.dram_cache is not None:
            kv = self.dram_cache.get(self._tier_key(segment_id, layer_idx))
            if kv is not None:
                self._tier_hits["DRAM"] += 1
                # Promote to HBM
                self.hbm_cache.put(self._tier_key(segment_id, layer_idx), kv)
                return kv, "DRAM"

        # SSD
        if self.ssd_cache is not None:
            kv = self.ssd_cache.get(self._tier_key(segment_id, layer_idx))
            if kv is not None:
                self._tier_hits["SSD"] += 1
                # Promote to HBM
                self.hbm_cache.put(self._tier_key(segment_id, layer_idx), kv)
                return kv, "SSD"

        # S3 (remote object storage)
        if self.s3_client is not None:
            kv = self._s3_get(segment_id, layer_idx)
            if kv is not None:
                self._tier_hits["S3"] += 1
                # Promote to HBM
                self.hbm_cache.put(self._tier_key(segment_id, layer_idx), kv)
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
        """Store KV tensor at the specified tier."""
        key = self._tier_key(segment_id, layer_idx)
        if tier == "HBM":
            self.hbm_cache.put(key, kv)
        elif tier == "DRAM" and self.dram_cache is not None:
            self.dram_cache.put(key, kv)
        elif tier == "SSD" and self.ssd_cache is not None:
            self.ssd_cache.put(key, kv)
        elif tier == "S3" and self.s3_client is not None:
            self._s3_put(segment_id, layer_idx, kv)

    def s3_object_key(self, segment_id: str, layer_idx: int) -> str:
        """Generate S3 object key: model_name/segment_id_layer.kvcache"""
        return f"{self.model_name}/{segment_id}_{layer_idx}.kvcache"

    @staticmethod
    def make_segment_id(chunk_tokens: List[int]) -> str:
        """SHA256 hash of CDC chunk content (position-independent)."""
        return cdc_segment_key(chunk_tokens)

    def tier_hit_stats(self) -> Dict[str, int]:
        """Return per-tier hit counts."""
        return dict(self._tier_hits)

    def total_misses(self) -> int:
        """Return total miss count."""
        return self._misses

    # ---------------------------------------------------------------- #
    # Private helpers                                                   #
    # ---------------------------------------------------------------- #

    def _tier_key(self, segment_id: str, layer_idx: int) -> str:
        """Internal key scoped to layer."""
        return f"{segment_id}_{layer_idx}"

    def _s3_get(self, segment_id: str, layer_idx: int) -> Optional[torch.Tensor]:
        """Fetch tensor from S3; returns None on any error or unavailability."""
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
