"""Unit tests for VeriCacheSpeculativeCodec (Activity C, 2026-05-25).

Tests cover Int8DraftCodec, TokenEvictionDraftCodec, VeriCacheSpeculativeCodec,
and the deterministic accuracy guarantee.
"""

import pytest
import torch

from src.cache.vericache_speculative_codec import (
    Int8DraftCodec,
    TokenEvictionDraftCodec,
    VeriCacheConfig,
    VeriCacheSpeculativeCodec,
    VerificationResult,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_kv(n_tokens: int = 32, d_head: int = 64, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_tokens, d_head)


def _make_q(n_q: int = 8, d_head: int = 64, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_q, d_head)


def _make_codec(
    d_head: int = 64,
    acceptance_threshold: float = 0.01,
    seed: int = 42,
) -> VeriCacheSpeculativeCodec:
    cfg = VeriCacheConfig(
        d_head=d_head,
        acceptance_threshold=acceptance_threshold,
        max_entries=512,
        seed=seed,
    )
    return VeriCacheSpeculativeCodec(cfg)


# --------------------------------------------------------------------------- #
# Int8DraftCodec tests
# --------------------------------------------------------------------------- #


def test_int8_codec_compression():
    """INT8 codec produces 2x memory compression vs FP16."""
    codec = Int8DraftCodec()
    kv = torch.randn(64, 128).half()  # FP16
    quantized, scale = codec.compress(kv)

    assert quantized.dtype == torch.int8
    assert isinstance(scale, torch.Tensor)
    assert scale.item() > 0

    # INT8 uses 1 byte/element vs FP16's 2 bytes/element: ~2x compression
    assert codec.compression_ratio == 2.0

    # Compressed memory ~ half of original (ignoring small scale overhead)
    orig_bytes = kv.numel() * 2  # FP16
    comp_bytes = quantized.numel() * 1  # INT8
    assert comp_bytes == orig_bytes // 2


def test_int8_codec_decompress_shape():
    """INT8 decompressed tensor has same shape as input."""
    codec = Int8DraftCodec()
    kv = torch.randn(32, 64)
    quantized, scale = codec.compress(kv)
    restored = codec.decompress((quantized, scale))

    assert restored.shape == kv.shape
    assert restored.dtype == torch.float32


def test_int8_codec_roundtrip_accuracy():
    """INT8 quantization error is small (within ~1%)."""
    codec = Int8DraftCodec()
    kv = torch.randn(128, 64)
    quantized, scale = codec.compress(kv)
    restored = codec.decompress((quantized, scale))

    rel_error = (kv - restored).norm() / (kv.norm() + 1e-8)
    assert rel_error.item() < 0.02  # INT8 quantization introduces < 2% error typically


# --------------------------------------------------------------------------- #
# TokenEvictionDraftCodec tests
# --------------------------------------------------------------------------- #


def test_token_eviction_codec():
    """TokenEviction codec keeps correct number of tokens per keep_ratio."""
    for keep_ratio in [0.3, 0.5, 0.7]:
        codec = TokenEvictionDraftCodec(keep_ratio=keep_ratio)
        n_tokens, d_head = 100, 64
        kv = torch.randn(n_tokens, d_head)
        kept_kv, kept_idx = codec.compress(kv)

        expected_n_keep = max(1, int(n_tokens * keep_ratio))
        assert kept_kv.shape == (expected_n_keep, d_head), \
            f"keep_ratio={keep_ratio}: expected {expected_n_keep} tokens, got {kept_kv.shape[0]}"
        assert kept_idx.shape == (expected_n_keep,)
        assert codec.compression_ratio == pytest.approx(1.0 / keep_ratio, rel=1e-5)


def test_token_eviction_keeps_high_norm_tokens():
    """TokenEviction keeps high-norm tokens (high importance)."""
    codec = TokenEvictionDraftCodec(keep_ratio=0.5)
    kv = torch.zeros(20, 32)
    # Make first 10 tokens have high norm
    kv[:10] = torch.randn(10, 32) * 10.0
    kv[10:] = torch.randn(10, 32) * 0.01

    kept_kv, kept_idx = codec.compress(kv)
    # All kept indices should be in the high-norm range (first 10)
    assert (kept_idx < 10).all(), f"Expected high-norm tokens kept, got indices {kept_idx}"


def test_token_eviction_decompress_returns_kept():
    """TokenEviction decompress returns kept tokens without padding."""
    codec = TokenEvictionDraftCodec(keep_ratio=0.5)
    kv = torch.randn(40, 32)
    compressed = codec.compress(kv)
    restored = codec.decompress(compressed)

    n_keep = max(1, int(40 * 0.5))
    assert restored.shape == (n_keep, 32)


# --------------------------------------------------------------------------- #
# VeriCacheSpeculativeCodec put/get tests
# --------------------------------------------------------------------------- #


def test_put_stores_entry():
    """put() stores KV entry in internal store."""
    codec = _make_codec()
    kv = _make_kv()
    codec.put("seg1", kv)
    assert "seg1" in codec._store
    assert len(codec._store) == 1


def test_get_returns_approx_tensor():
    """get() returns approximate KV tensor with correct shape and dtype."""
    codec = _make_codec(d_head=64)
    kv = _make_kv(n_tokens=32, d_head=64)
    codec.put("seg1", kv)
    result = codec.get("seg1")

    assert result is not None
    assert result.shape == kv.shape
    # dtype: Int8DraftCodec returns float32
    assert result.dtype in (torch.float32, torch.float16)


def test_get_miss_returns_none():
    """get() returns None on cache miss."""
    codec = _make_codec()
    result = codec.get("nonexistent_key")
    assert result is None


def test_put_kv_pair_stores_separate_keys():
    """put_kv_pair() stores K and V under key+'_K' and key+'_V'."""
    codec = _make_codec(d_head=64)
    K = _make_kv(n_tokens=16, d_head=64, seed=1)
    V = _make_kv(n_tokens=16, d_head=64, seed=2)
    codec.put_kv_pair("seg1", K, V)

    assert "seg1_K" in codec._store
    assert "seg1_V" in codec._store


# --------------------------------------------------------------------------- #
# draft_and_verify tests
# --------------------------------------------------------------------------- #


def test_draft_and_verify_shape():
    """draft_and_verify() returns VerificationResult with correct output shapes."""
    d_head = 64
    n_tokens = 32
    n_q = 8
    codec = _make_codec(d_head=d_head)
    K = _make_kv(n_tokens, d_head, seed=1)
    V = _make_kv(n_tokens, d_head, seed=2)
    Q = _make_q(n_q, d_head)

    codec.put_kv_pair("seg1", K, V)
    result = codec.draft_and_verify("seg1_K", "seg1_V", Q)

    assert result is not None
    assert isinstance(result, VerificationResult)
    assert result.draft_output.shape == (n_q, d_head)
    assert result.verified_output.shape == (n_q, d_head)
    assert isinstance(result.accepted, bool)
    assert isinstance(result.relative_error, float)
    assert result.relative_error >= 0.0


def test_draft_and_verify_miss_returns_none():
    """draft_and_verify() returns None when keys are not in store."""
    codec = _make_codec()
    Q = _make_q()
    result = codec.draft_and_verify("missing_K", "missing_V", Q)
    assert result is None


def test_draft_and_verify_accepted_when_small_error():
    """draft_and_verify() accepts when relative_error is below threshold."""
    d_head = 64
    codec = _make_codec(d_head=d_head, acceptance_threshold=1.0)  # Accept everything
    K = _make_kv(32, d_head, seed=1)
    V = _make_kv(32, d_head, seed=2)
    Q = _make_q(8, d_head)

    codec.put_kv_pair("seg1", K, V)
    result = codec.draft_and_verify("seg1_K", "seg1_V", Q)

    assert result is not None
    assert result.accepted is True  # threshold=1.0 means everything accepted


# --------------------------------------------------------------------------- #
# Deterministic guarantee tests
# --------------------------------------------------------------------------- #


def test_deterministic_guarantee():
    """Core VeriCache guarantee: final output is always within acceptance_threshold of full KV.

    Tests both branches:
    1. All-reject branch (threshold=0.0): final_output == verified_output
    2. All-accept branch (threshold=1.0): final_output == draft_output, relative_error < 1.0
    """
    d_head = 64
    n_tokens = 32
    n_q = 8
    torch.manual_seed(42)
    K = torch.randn(n_tokens, d_head)
    V = torch.randn(n_tokens, d_head)
    Q = torch.randn(n_q, d_head)

    # Branch 1: All reject (threshold=0.0 -> every draft rejected)
    cfg_reject = VeriCacheConfig(d_head=d_head, acceptance_threshold=0.0, seed=42)
    codec_reject = VeriCacheSpeculativeCodec(cfg_reject)
    codec_reject.put_kv_pair("seg", K, V)
    result_reject = codec_reject.draft_and_verify("seg_K", "seg_V", Q)

    assert result_reject is not None
    assert result_reject.accepted is False  # threshold=0.0 -> rejected

    final_reject = codec_reject.get_final_output(result_reject)
    # Rejected: final_output MUST equal verified_output (full KV accuracy)
    max_diff = (final_reject - result_reject.verified_output).abs().max().item()
    assert max_diff < 1e-5, f"Rejected draft: final_output != verified_output (max_diff={max_diff})"

    # Branch 2: All accept (threshold=1.0 -> every draft accepted)
    cfg_accept = VeriCacheConfig(d_head=d_head, acceptance_threshold=1.0, seed=42)
    codec_accept = VeriCacheSpeculativeCodec(cfg_accept)
    codec_accept.put_kv_pair("seg", K, V)
    result_accept = codec_accept.draft_and_verify("seg_K", "seg_V", Q)

    assert result_accept is not None
    assert result_accept.accepted is True  # threshold=1.0 -> accepted

    final_accept = codec_accept.get_final_output(result_accept)
    # Accepted: final_output == draft_output, relative_error < threshold (1.0)
    max_diff = (final_accept - result_accept.draft_output).abs().max().item()
    assert max_diff < 1e-5, f"Accepted draft: final_output != draft_output (max_diff={max_diff})"
    assert result_accept.relative_error < 1.0


def test_deterministic_guarantee_final_error_below_threshold():
    """Final output relative error vs full KV is always <= acceptance_threshold.

    This is the mathematical invariant: regardless of draft codec compression quality,
    the final output error is bounded by acceptance_threshold.
    """
    d_head = 64
    threshold = 0.01
    torch.manual_seed(42)
    K = torch.randn(32, d_head)
    V = torch.randn(32, d_head)
    Q = torch.randn(8, d_head)

    cfg = VeriCacheConfig(d_head=d_head, acceptance_threshold=threshold, seed=42)
    codec = VeriCacheSpeculativeCodec(cfg)
    codec.put_kv_pair("seg", K, V)
    result = codec.draft_and_verify("seg_K", "seg_V", Q)

    assert result is not None
    final = codec.get_final_output(result)

    # Compute full KV output directly
    full_out = VeriCacheSpeculativeCodec._compute_attention(Q, K, V)
    final_error = float(
        (final.float() - full_out.float()).norm() / (full_out.float().norm() + 1e-8)
    )

    # The invariant: final_error <= threshold (accepted) or final_error ~= 0 (rejected = full KV)
    assert final_error <= threshold + 1e-6, \
        f"Final error {final_error:.6f} exceeds threshold {threshold}"


# --------------------------------------------------------------------------- #
# Eviction, hit_rate, memory tests
# --------------------------------------------------------------------------- #


def test_evict_lru():
    """evict() removes LRU entry and returns bytes freed."""
    codec = _make_codec()
    kv1 = _make_kv(16, 64, seed=1)
    kv2 = _make_kv(16, 64, seed=2)
    codec.put("k1", kv1)
    codec.put("k2", kv2)

    freed = codec.evict()
    assert freed > 0
    assert len(codec._store) == 1


def test_hit_rate():
    """hit_rate() reflects actual hit/miss pattern."""
    codec = _make_codec()
    kv = _make_kv()
    codec.put("k1", kv)

    codec.get("k1")    # hit
    codec.get("k1")    # hit
    codec.get("miss")  # miss

    assert codec.hit_rate() == pytest.approx(2 / 3, rel=1e-4)


def test_memory_bytes_compressed_only():
    """memory_bytes() counts compressed KV only (not full_kv_ref)."""
    codec = _make_codec(d_head=64)
    kv = torch.randn(64, 64).half()  # FP16
    codec.put("k1", kv)

    compressed_mem = codec.memory_bytes()
    full_mem = codec.memory_bytes_full_kv()

    # Compressed should be less than full KV
    assert compressed_mem > 0
    assert compressed_mem < full_mem


def test_memory_reduction_ratio():
    """memory_reduction_ratio() returns value in [0, 1]."""
    codec = _make_codec(d_head=64)
    kv = torch.randn(64, 64).half()
    codec.put("k1", kv)

    ratio = codec.memory_reduction_ratio()
    assert 0.0 <= ratio <= 1.0


def test_reset_stats():
    """reset_stats() resets all counters to zero."""
    codec = _make_codec()
    kv = _make_kv()
    codec.put("k1", kv)
    codec.get("k1")
    codec.get("miss")

    codec.reset_stats()
    assert codec.hit_rate() == 0.0
    assert codec._hits == 0
    assert codec._misses == 0
    assert codec._total_draft_calls == 0
    assert codec._total_accepted == 0
    assert len(codec._relative_errors) == 0


# --------------------------------------------------------------------------- #
# set_draft_codec tests
# --------------------------------------------------------------------------- #


def test_set_draft_codec_token_eviction():
    """set_draft_codec() swaps codec and new entries use new codec."""
    d_head = 64
    codec = _make_codec(d_head=d_head)
    codec.set_draft_codec(TokenEvictionDraftCodec(keep_ratio=0.5))

    kv = _make_kv(32, d_head)
    codec.put("k1", kv)

    # get() should work with the new codec
    result = codec.get("k1")
    assert result is not None
    # TokenEviction returns kept tokens only (16 out of 32)
    assert result.shape[0] == max(1, int(32 * 0.5))


def test_set_draft_codec_and_draft_verify():
    """draft_and_verify() works after codec swap."""
    d_head = 64
    codec = _make_codec(d_head=d_head, acceptance_threshold=1.0)
    codec.set_draft_codec(TokenEvictionDraftCodec(keep_ratio=0.5))

    K = _make_kv(32, d_head, seed=1)
    V = _make_kv(32, d_head, seed=2)
    Q = _make_q(8, d_head)

    codec.put_kv_pair("seg", K, V)
    result = codec.draft_and_verify("seg_K", "seg_V", Q)

    assert result is not None
    assert isinstance(result.accepted, bool)


# --------------------------------------------------------------------------- #
# get_importance_mask tests
# --------------------------------------------------------------------------- #


def test_get_importance_mask_token_eviction():
    """get_importance_mask() returns bool mask with TokenEviction codec."""
    d_head = 64
    n_tokens = 32
    codec = _make_codec(d_head=d_head)
    codec.set_draft_codec(TokenEvictionDraftCodec(keep_ratio=0.5))

    kv = _make_kv(n_tokens, d_head)
    codec.put("k1", kv)
    mask = codec.get_importance_mask("k1")

    assert mask is not None
    assert mask.dtype == torch.bool
    assert mask.shape == (n_tokens,)
    assert mask.sum().item() == max(1, int(n_tokens * 0.5))


def test_get_importance_mask_int8_codec():
    """get_importance_mask() returns None with Int8DraftCodec (scale is float, not long)."""
    codec = _make_codec()
    kv = _make_kv()
    codec.put("k1", kv)
    mask = codec.get_importance_mask("k1")
    # Int8 codec: second element is scale (float), not long tensor -> None
    assert mask is None


def test_get_importance_mask_miss():
    """get_importance_mask() returns None for missing key."""
    codec = _make_codec()
    mask = codec.get_importance_mask("nonexistent")
    assert mask is None


# --------------------------------------------------------------------------- #
# CacheStore interface compliance
# --------------------------------------------------------------------------- #


def test_cachestore_interface_compliance():
    """VeriCacheSpeculativeCodec implements all required CacheStore abstract methods."""
    from src.cache.base import CacheStore
    codec = _make_codec()
    assert isinstance(codec, CacheStore)

    # Verify all abstract methods are implemented
    kv = _make_kv()
    codec.put("k1", kv)
    val = codec.get("k1")
    assert val is not None

    freed = codec.evict()
    assert isinstance(freed, int)

    rate = codec.hit_rate()
    assert isinstance(rate, float)

    mem = codec.memory_bytes()
    assert isinstance(mem, int)

    codec.reset_stats()  # should not raise


def test_speculative_stats_keys():
    """speculative_stats() returns dict with all required keys."""
    codec = _make_codec(d_head=64)
    K = _make_kv(16, 64, seed=1)
    V = _make_kv(16, 64, seed=2)
    Q = _make_q(4, 64)

    codec.put_kv_pair("seg", K, V)
    codec.draft_and_verify("seg_K", "seg_V", Q)

    stats = codec.speculative_stats()
    for key in [
        "draft_acceptance_rate",
        "mean_relative_error",
        "total_draft_calls",
        "total_accepted",
        "hit_rate",
        "memory_reduction_ratio",
        "compression_codec",
        "compression_ratio",
        "n_entries",
    ]:
        assert key in stats, f"Missing key in speculative_stats: {key}"
