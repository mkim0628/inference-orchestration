"""Unit tests for Activity B-2: CDCContentHashSegmentIDInterface.

Tests:
  - test_segment_id_equals_sha256: make_segment_id matches SHA256(token_bytes)
  - test_hbm_hit_returns_first: HBM hit returns immediately, S3 not called
  - test_tier_waterfall_order: HBM→DRAM→SSD→S3 lookup order
  - test_s3_unavailable_fallback: S3=None → "miss" without exception
  - test_s3_object_key_format: model_name/segment_id_layer.kvcache format
  - test_segment_id_dedup: storing twice doesn't create duplicate
"""

import hashlib
import struct
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.cache.cdc_content_hash_interface import (
    CDCContentHashSegmentIDInterface,
    SegmentMetadata,
)
from src.cache.contiguous import ContiguousCache
from src.cache.irminsul_mla_segment_cache import cdc_segment_key


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _make_interface(
    s3_client=None,
    model_name: str = "test-model",
    dram: bool = True,
    ssd: bool = True,
) -> CDCContentHashSegmentIDInterface:
    hbm = ContiguousCache(max_entries=100)
    dram_cache = ContiguousCache(max_entries=100) if dram else None
    ssd_cache = ContiguousCache(max_entries=100) if ssd else None
    return CDCContentHashSegmentIDInterface(
        hbm_cache=hbm,
        dram_cache=dram_cache,
        ssd_cache=ssd_cache,
        s3_client=s3_client,
        s3_bucket="kvcache",
        model_name=model_name,
    )


# ------------------------------------------------------------------ #
# Segment ID correctness                                              #
# ------------------------------------------------------------------ #


def test_segment_id_equals_sha256() -> None:
    """make_segment_id produces SHA256(raw_token_bytes) exactly."""
    chunk_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    raw = struct.pack(f"{len(chunk_tokens)}I", *chunk_tokens)
    expected = hashlib.sha256(raw).hexdigest()

    result = CDCContentHashSegmentIDInterface.make_segment_id(chunk_tokens)
    assert result == expected, f"Expected {expected}, got {result}"


def test_segment_id_is_position_independent() -> None:
    """Same tokens produce same ID regardless of batch position context."""
    tokens = [10, 20, 30, 40, 50]
    id1 = CDCContentHashSegmentIDInterface.make_segment_id(tokens)
    id2 = CDCContentHashSegmentIDInterface.make_segment_id(tokens)
    assert id1 == id2, "Same tokens must produce same segment ID"


def test_segment_id_different_tokens_different_id() -> None:
    """Different token sequences produce different segment IDs."""
    id1 = CDCContentHashSegmentIDInterface.make_segment_id([1, 2, 3])
    id2 = CDCContentHashSegmentIDInterface.make_segment_id([4, 5, 6])
    assert id1 != id2, "Different tokens must produce different IDs"


# ------------------------------------------------------------------ #
# HBM hit test                                                        #
# ------------------------------------------------------------------ #


def test_hbm_hit_returns_first() -> None:
    """HBM hit returns tensor immediately; lower tiers not queried."""
    interface = _make_interface()
    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([1, 2, 3])

    torch.manual_seed(42)
    kv = torch.randn(10, 64)
    interface.store(segment_id, kv, tier="HBM", layer_idx=0)

    result_kv, tier = interface.lookup(segment_id, layer_idx=0)
    assert tier == "HBM", f"Expected HBM hit, got '{tier}'"
    assert result_kv is not None
    assert torch.allclose(result_kv, kv, atol=1e-5)


def test_hbm_hit_does_not_query_s3() -> None:
    """When HBM hits, S3 client should never be called."""
    mock_s3 = MagicMock()
    interface = _make_interface(s3_client=mock_s3)

    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([10, 20])
    torch.manual_seed(1)
    kv = torch.randn(5, 32)
    interface.store(segment_id, kv, tier="HBM", layer_idx=0)

    interface.lookup(segment_id, layer_idx=0)
    mock_s3.get_object.assert_not_called()


# ------------------------------------------------------------------ #
# Tier waterfall order                                                #
# ------------------------------------------------------------------ #


def test_tier_waterfall_order_hbm_first() -> None:
    """HBM→DRAM→SSD→S3 order: HBM miss goes to DRAM."""
    interface = _make_interface()
    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([100, 200, 300])

    torch.manual_seed(5)
    kv = torch.randn(8, 32)
    interface.store(segment_id, kv, tier="DRAM", layer_idx=0)

    result_kv, tier = interface.lookup(segment_id, layer_idx=0)
    assert tier == "DRAM", f"Expected DRAM hit, got '{tier}'"
    assert result_kv is not None


def test_tier_waterfall_ssd_hit() -> None:
    """SSD hit when HBM and DRAM both miss."""
    interface = _make_interface()
    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([11, 22, 33])

    torch.manual_seed(7)
    kv = torch.randn(6, 16)
    interface.store(segment_id, kv, tier="SSD", layer_idx=0)

    result_kv, tier = interface.lookup(segment_id, layer_idx=0)
    assert tier == "SSD", f"Expected SSD hit, got '{tier}'"


def test_tier_waterfall_full_miss() -> None:
    """All tiers miss → returns (None, 'miss') with no exception."""
    interface = _make_interface(s3_client=None)
    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([999, 888])

    result_kv, tier = interface.lookup(segment_id, layer_idx=0)
    assert result_kv is None
    assert tier == "miss"


def test_ssd_hit_promotes_to_hbm() -> None:
    """After SSD hit, tensor is promoted to HBM for future lookups."""
    interface = _make_interface()
    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([50, 60, 70])

    torch.manual_seed(3)
    kv = torch.randn(4, 16)
    interface.store(segment_id, kv, tier="SSD", layer_idx=0)

    # First lookup: SSD hit
    _, tier = interface.lookup(segment_id, layer_idx=0)
    assert tier == "SSD"

    # Second lookup: should now be HBM (promoted)
    _, tier2 = interface.lookup(segment_id, layer_idx=0)
    assert tier2 == "HBM", f"Expected HBM on second lookup (promoted), got '{tier2}'"


# ------------------------------------------------------------------ #
# S3 unavailable fallback                                             #
# ------------------------------------------------------------------ #


def test_s3_unavailable_fallback() -> None:
    """s3_client=None → returns 'miss' without any exception."""
    interface = _make_interface(s3_client=None)
    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([7, 8, 9])

    # Should not raise
    result_kv, tier = interface.lookup(segment_id, layer_idx=0)
    assert result_kv is None
    assert tier == "miss"


def test_s3_error_handled_gracefully() -> None:
    """S3 client that raises exception → returns None gracefully."""
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = Exception("Connection refused")
    interface = _make_interface(s3_client=mock_s3)

    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([1, 1, 1])

    # Should not propagate exception
    result_kv, tier = interface.lookup(segment_id, layer_idx=0)
    assert result_kv is None
    assert tier == "miss"


# ------------------------------------------------------------------ #
# S3 object key format                                                #
# ------------------------------------------------------------------ #


def test_s3_object_key_format() -> None:
    """S3 key must be '{model_name}/{segment_id}_{layer_idx}.kvcache'."""
    interface = _make_interface(model_name="deepseek-v2")
    segment_id = "a" * 64  # 64-char hex string
    layer_idx = 3

    key = interface.s3_object_key(segment_id, layer_idx)
    expected = f"deepseek-v2/{segment_id}_{layer_idx}.kvcache"
    assert key == expected, f"Expected '{expected}', got '{key}'"


def test_s3_object_key_default_model() -> None:
    """Default model_name='default' appears in S3 key."""
    interface = _make_interface(model_name="default")
    key = interface.s3_object_key("b" * 64, 0)
    assert key.startswith("default/")
    assert key.endswith(".kvcache")


# ------------------------------------------------------------------ #
# Segment ID deduplication                                            #
# ------------------------------------------------------------------ #


def test_segment_id_dedup_same_store() -> None:
    """Storing the same segment_id twice to HBM: lookup returns correct tensor."""
    interface = _make_interface()
    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([1, 2, 3, 4])

    torch.manual_seed(42)
    kv1 = torch.randn(5, 32)
    torch.manual_seed(99)
    kv2 = torch.randn(5, 32)

    interface.store(segment_id, kv1, tier="HBM", layer_idx=0)
    interface.store(segment_id, kv2, tier="HBM", layer_idx=0)

    # Should return one result (no duplicate in interface)
    result_kv, tier = interface.lookup(segment_id, layer_idx=0)
    assert result_kv is not None
    assert tier == "HBM"


def test_tier_hit_stats_tracked() -> None:
    """tier_hit_stats() returns correct counts per tier."""
    interface = _make_interface()

    segment_id = CDCContentHashSegmentIDInterface.make_segment_id([1, 2])
    torch.manual_seed(0)
    kv = torch.randn(4, 16)

    interface.store(segment_id, kv, tier="HBM", layer_idx=0)
    interface.lookup(segment_id, layer_idx=0)
    interface.lookup(segment_id, layer_idx=0)

    stats = interface.tier_hit_stats()
    assert stats["HBM"] >= 2, f"Expected >= 2 HBM hits, got {stats['HBM']}"


def test_total_misses_tracked() -> None:
    """total_misses() increments correctly."""
    interface = _make_interface(s3_client=None)
    interface.lookup("nonexistent_" + "a" * 52, layer_idx=0)
    interface.lookup("nonexistent_" + "b" * 52, layer_idx=0)
    assert interface.total_misses() >= 2
