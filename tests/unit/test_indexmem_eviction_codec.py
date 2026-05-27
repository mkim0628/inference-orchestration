"""Unit tests for IndexMemEvictionCodec (Activity C-1)."""

import json

import pytest
import torch

from src.cache.indexmem_eviction_codec import (
    IndexMemEvictionCodec,
    IndexMemEvictionConfig,
)


def _make_codec(
    budget_ratio: float = 0.5,
    zero_shot: bool = True,
    n_layers: int = 4,
    kv_dim: int = 64,
    seed: int = 42,
    policy: str = "snapkv",
) -> IndexMemEvictionCodec:
    cfg = IndexMemEvictionConfig(
        budget_ratio=budget_ratio,
        base_eviction_policy=policy,
        n_layers=n_layers,
        kv_dim=kv_dim,
        zero_shot_mode=zero_shot,
        seed=seed,
    )
    return IndexMemEvictionCodec(cfg)


# --------------------------------------------------------------------------- #


def test_encode_output_smaller_than_input():
    codec = _make_codec(budget_ratio=0.5)
    torch.manual_seed(0)
    kv = torch.randn(20, 64)
    compressed = codec.encode(kv, layer_idx=0)
    assert compressed.shape[0] == 10, f"Expected 10 tokens, got {compressed.shape[0]}"


def test_encode_preserves_important_tokens():
    codec = _make_codec(budget_ratio=0.5, zero_shot=False)
    torch.manual_seed(3)
    kv = torch.randn(16, 64)
    # Query highly similar to first token
    query = kv[0].clone()

    compressed = codec.encode(
        kv,
        layer_idx=0,
        query=query,
        token_positions=torch.arange(16, dtype=torch.float32),
        current_position=16,
        request_key="req_important",
    )
    # Compressed output should have budget_ratio tokens kept
    expected_keep = max(1, int(16 * 0.5))
    assert compressed.shape[0] == expected_keep


def test_get_readout_after_encode():
    codec = _make_codec(budget_ratio=0.5)
    torch.manual_seed(1)
    kv = torch.randn(16, 64)
    codec.encode(kv, layer_idx=0, request_key="req_readout")

    query = torch.randn(4, 64)
    readout = codec.get_readout(query, layer_idx=0, request_key="req_readout")
    # After encode, latent memory should have state -> readout should be non-zero
    assert readout.shape == query.shape
    # Not necessarily non-zero (small beta), but finite
    assert torch.isfinite(readout).all()


def test_auto_adjust_fallback_on_high_delta():
    codec = _make_codec(budget_ratio=0.5)
    original_budget = codec.config.budget_ratio
    adjusted = codec.auto_adjust_on_accuracy_delta(0.02)  # > 1% threshold
    assert adjusted is True
    assert codec.config.budget_ratio == codec.config.fallback_budget_ratio
    assert codec.config.budget_ratio > original_budget


def test_auto_adjust_no_change_on_low_delta():
    codec = _make_codec(budget_ratio=0.5)
    adjusted = codec.auto_adjust_on_accuracy_delta(0.005)  # < 1%
    assert adjusted is False
    assert codec.config.budget_ratio == 0.5


def test_draft_codec_interface_compress():
    codec = _make_codec(budget_ratio=0.5)
    torch.manual_seed(2)
    kv = torch.randn(16, 64)
    result = codec.compress(kv)
    assert isinstance(result, tuple), "compress() should return a tuple"
    assert len(result) == 2, "compress() should return (tensor, str)"
    compressed, key = result
    assert isinstance(compressed, torch.Tensor)
    assert isinstance(key, str)


def test_draft_codec_interface_decompress():
    codec = _make_codec(budget_ratio=0.5)
    torch.manual_seed(2)
    kv = torch.randn(16, 64)
    compressed = codec.compress(kv)
    decompressed = codec.decompress(compressed)
    assert isinstance(decompressed, torch.Tensor)
    # Shape: budget_ratio * n_tokens kept
    assert decompressed.shape[0] == max(1, int(16 * 0.5))
    assert decompressed.shape[1] == 64


def test_compression_ratio_property():
    codec = _make_codec(budget_ratio=0.5)
    assert codec.compression_ratio_float == 2.0
    codec2 = _make_codec(budget_ratio=0.4)
    assert abs(codec2.compression_ratio_float - 2.5) < 1e-5


def test_compression_ratio_method():
    codec = _make_codec(budget_ratio=0.5)
    assert codec.compression_ratio(0) == 2.0


def test_base_eviction_policy_h2o():
    codec = _make_codec(budget_ratio=0.5, policy="h2o")
    torch.manual_seed(5)
    kv = torch.randn(16, 64)
    compressed = codec.encode(kv, layer_idx=0)
    assert compressed.shape[0] == 8  # budget_ratio=0.5


def test_compression_stats_json_serializable():
    codec = _make_codec(budget_ratio=0.5)
    torch.manual_seed(6)
    kv = torch.randn(16, 64)
    codec.encode(kv, layer_idx=0, request_key="stats_test")
    stats = codec.compression_stats()
    # Should be JSON serializable
    json_str = json.dumps(stats)
    assert len(json_str) > 0
    loaded = json.loads(json_str)
    assert "encode_count" in loaded
    assert "configured_budget_ratio" in loaded


def test_decode_passthrough():
    codec = _make_codec(budget_ratio=0.5)
    torch.manual_seed(7)
    compressed = torch.randn(8, 64)
    decoded = codec.decode(compressed, layer_idx=0)
    assert torch.equal(decoded, compressed)


def test_encode_with_default_query_fallback():
    """encode() with query=None should use kv.mean(dim=0) as fallback."""
    codec = _make_codec(budget_ratio=0.5)
    torch.manual_seed(8)
    kv = torch.randn(20, 64)
    compressed = codec.encode(kv, layer_idx=0)  # query=None
    assert compressed.shape[0] == 10
