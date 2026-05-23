"""Unit tests for RuntimeCertifiedQuantizedAttentionCodec (Activity C).

Covers: quantization round-trips, CacheStore interface, error bound
conservativeness, fallback ladder, memory reduction, and reproducibility.
"""

import pytest
import torch

from src.cache.runtime_certified_quant_codec import (
    RuntimeCertifiedConfig,
    RuntimeCertifiedQuantizedAttentionCodec,
)
from src.metrics.perplexity import attention_output_relative_error


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def config() -> RuntimeCertifiedConfig:
    return RuntimeCertifiedConfig(d_head=64, max_entries=100, seed=42)


@pytest.fixture
def codec(config: RuntimeCertifiedConfig) -> RuntimeCertifiedQuantizedAttentionCodec:
    return RuntimeCertifiedQuantizedAttentionCodec(config)


def _make_kv(seq_len: int = 16, d_head: int = 64, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(seq_len, d_head)


def _make_query(n_q: int = 4, d_head: int = 64, seed: int = 99) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_q, d_head)


# --------------------------------------------------------------------------- #
# Quantization round-trip accuracy                                             #
# --------------------------------------------------------------------------- #


def test_quantize_int8_round_trip_error_small() -> None:
    """quantize_int8 -> dequantize_int8: relative_error < 0.01."""
    torch.manual_seed(42)
    x = torch.randn(32, 64)
    x_int8, scale, zero = RuntimeCertifiedQuantizedAttentionCodec._quantize_int8(x)
    x_restored = RuntimeCertifiedQuantizedAttentionCodec._dequantize_int8(x_int8, scale, zero)
    rel_err = (x.float() - x_restored).norm() / x.float().norm().clamp(min=1e-8)
    assert rel_err.item() < 0.01, f"INT8 round-trip relative error {rel_err:.6f} >= 0.01"


def test_quantize_int4_round_trip_error_small() -> None:
    """quantize_int4 -> dequantize_int4: raw KV-tensor L2 relative_error < 0.20.

    INT4 has only 16 discrete levels; raw KV-tensor L2 error is inherently higher
    (~10-20%) than INT8. The accuracy-preserving guarantee is on attention outputs,
    not raw tensor error. The threshold 0.20 matches Spec.md INT4 expectations.
    """
    torch.manual_seed(42)
    x = torch.randn(32, 64)
    packed, scale, zero = RuntimeCertifiedQuantizedAttentionCodec._quantize_int4(x)
    x_restored = RuntimeCertifiedQuantizedAttentionCodec._dequantize_int4(packed, scale, zero, 64)
    rel_err = (x.float() - x_restored).norm() / x.float().norm().clamp(min=1e-8)
    assert rel_err.item() < 0.20, f"INT4 round-trip relative error {rel_err:.6f} >= 0.20"


# --------------------------------------------------------------------------- #
# put / get storage                                                            #
# --------------------------------------------------------------------------- #


def test_put_stores_int8_and_int4(codec: RuntimeCertifiedQuantizedAttentionCodec) -> None:
    """put(key, tensor) stores entry with key_int8 of dtype int8."""
    kv = _make_kv()
    codec.put("k1", kv)
    assert "k1" in codec._store
    assert codec._store["k1"].key_int8.dtype == torch.int8


def test_put_cpu_backup_exists(codec: RuntimeCertifiedQuantizedAttentionCodec) -> None:
    """After put, key_fp16_backup resides on CPU."""
    kv = _make_kv()
    codec.put("k1", kv)
    assert codec._store["k1"].key_fp16_backup.device == torch.device("cpu")


def test_get_returns_restored_tensor_level0(
    codec: RuntimeCertifiedQuantizedAttentionCodec,
) -> None:
    """put -> get at fallback_level=0 returns tensor with same shape."""
    kv = _make_kv(seq_len=16, d_head=64)
    codec.put("k1", kv)
    out = codec.get("k1")
    assert out is not None
    assert out.shape == kv.shape


# --------------------------------------------------------------------------- #
# Error bound conservativeness (MANDATORY)                                     #
# --------------------------------------------------------------------------- #


def test_compute_error_bound_conservative() -> None:
    """100 synthetic sequences: error_bound >= actual_relative_error (MANDATORY)."""
    config = RuntimeCertifiedConfig(d_head=64, max_entries=1000, seed=42)
    codec = RuntimeCertifiedQuantizedAttentionCodec(config)
    violations = 0
    for i in range(100):
        torch.manual_seed(i)
        seq_len = torch.randint(8, 32, (1,)).item()
        K_orig = torch.randn(seq_len, 64)
        V_orig = torch.randn(seq_len, 64)
        Q = torch.randn(4, 64)

        k_int8, k_scale, k_zero = codec._quantize_int8(K_orig.float())
        K_restored = codec._dequantize_int8(k_int8, k_scale, k_zero)
        v_packed, v_scale, v_zero = codec._quantize_int4(V_orig.float())
        V_restored = codec._dequantize_int4(v_packed, v_scale, v_zero, 64)

        error_bound, _, _ = codec.compute_error_bound(
            Q.float(), K_orig.float(), K_restored, V_restored, V_orig.float()
        )
        actual_error = attention_output_relative_error(
            Q.float(), K_orig.float(), V_orig.float(), K_restored, V_restored
        )
        if error_bound < actual_error - 1e-6:
            violations += 1

    assert violations == 0, (
        f"error_bound < actual_error in {violations}/100 sequences (MANDATORY violated)"
    )


# --------------------------------------------------------------------------- #
# Fallback ladder decision                                                     #
# --------------------------------------------------------------------------- #


def test_decide_fallback_level0_below_threshold(
    codec: RuntimeCertifiedQuantizedAttentionCodec,
) -> None:
    """error_bound=0.003 <= error_threshold=0.005 -> fallback_level=0."""
    level = codec.decide_fallback_level(error_bound=0.003, delta_attn_bound=0.001)
    assert level == 0


def test_decide_fallback_level1_above_threshold(
    codec: RuntimeCertifiedQuantizedAttentionCodec,
) -> None:
    """error_bound=0.008 > threshold=0.005, delta_attn=0.002 <= threshold/2=0.0025 -> level=1."""
    level = codec.decide_fallback_level(error_bound=0.008, delta_attn_bound=0.002)
    assert level == 1


def test_decide_fallback_level2_high_attn_distortion(
    codec: RuntimeCertifiedQuantizedAttentionCodec,
) -> None:
    """delta_attn=0.004 > threshold/2=0.0025 -> level=2."""
    level = codec.decide_fallback_level(error_bound=0.01, delta_attn_bound=0.004)
    assert level == 2


# --------------------------------------------------------------------------- #
# certify_and_update                                                           #
# --------------------------------------------------------------------------- #


def test_certify_and_update_changes_fallback_level(
    codec: RuntimeCertifiedQuantizedAttentionCodec,
) -> None:
    """certify_and_update(Q) sets fallback_level on the stored entry."""
    kv = _make_kv(seq_len=16, d_head=64)
    codec.put("k1", kv)
    Q = _make_query(n_q=4, d_head=64)
    new_level, error_bound = codec.certify_and_update("k1", Q)
    assert new_level in (0, 1, 2)
    assert error_bound >= 0.0
    assert codec._store["k1"].fallback_level == new_level


# --------------------------------------------------------------------------- #
# Memory reduction (MANDATORY)                                                 #
# --------------------------------------------------------------------------- #


def test_memory_reduction_ratio_above_50pct() -> None:
    """INT8K+INT4V: memory_reduction_ratio() >= 0.50 vs FP16 (MANDATORY)."""
    config = RuntimeCertifiedConfig(d_head=64, max_entries=1000, seed=42)
    codec = RuntimeCertifiedQuantizedAttentionCodec(config)
    for i in range(10):
        kv = _make_kv(seq_len=64, d_head=64, seed=i)
        codec.put(f"k{i}", kv)
    ratio = codec.memory_reduction_ratio()
    assert ratio >= 0.50, f"memory_reduction_ratio={ratio:.4f} < 0.50 (MANDATORY)"


# --------------------------------------------------------------------------- #
# compression_hook accuracy (MANDATORY)                                        #
# --------------------------------------------------------------------------- #


def test_compression_hook_relative_error_below_1pct() -> None:
    """compression_hook: attention_output_relative_error < 0.01 (MANDATORY)."""
    config = RuntimeCertifiedConfig(d_head=64, max_entries=100, seed=42)
    codec = RuntimeCertifiedQuantizedAttentionCodec(config)
    torch.manual_seed(42)
    K_orig = torch.randn(32, 64)
    V_orig = torch.randn(32, 64)
    Q = torch.randn(4, 64)

    K_comp = codec.compression_hook("key", K_orig)
    err = attention_output_relative_error(
        Q.float(), K_orig.float(), V_orig.float(), K_comp.float(), V_orig.float()
    )
    assert err < 0.01, f"compression_hook relative_error={err:.6f} >= 0.01 (MANDATORY)"


# --------------------------------------------------------------------------- #
# Fallback rate tracking                                                       #
# --------------------------------------------------------------------------- #


def test_fallback_rate_level1_tracked() -> None:
    """Level 1 fallback increments _fallback_count_level1."""
    config = RuntimeCertifiedConfig(d_head=64, max_entries=100, seed=42)
    codec = RuntimeCertifiedQuantizedAttentionCodec(config)
    kv = _make_kv(seq_len=16, d_head=64)
    codec.put("k1", kv)
    # force level 1
    codec._store["k1"].fallback_level = 1
    codec.get("k1")
    assert codec._fallback_count_level1 >= 1


# --------------------------------------------------------------------------- #
# certified_accuracy_report                                                    #
# --------------------------------------------------------------------------- #


def test_certified_accuracy_report_keys(
    codec: RuntimeCertifiedQuantizedAttentionCodec,
) -> None:
    """certified_accuracy_report() contains all required keys."""
    report = codec.certified_accuracy_report()
    required_keys = [
        "fallback_rate_level1",
        "fallback_rate_level2",
        "error_bound_mean",
        "error_bound_p99",
        "memory_reduction_ratio",
        "error_threshold",
    ]
    for k in required_keys:
        assert k in report, f"Missing key '{k}' in certified_accuracy_report()"


# --------------------------------------------------------------------------- #
# Full CacheStore interface                                                    #
# --------------------------------------------------------------------------- #


def test_cachestore_interface_full() -> None:
    """put/get/evict/hit_rate/memory_bytes/reset_stats all work."""
    config = RuntimeCertifiedConfig(d_head=64, max_entries=10, seed=42)
    codec = RuntimeCertifiedQuantizedAttentionCodec(config)
    kv = _make_kv()
    codec.put("a", kv)
    assert codec.get("a") is not None
    assert codec.get("missing") is None
    assert codec.hit_rate() > 0.0
    assert codec.memory_bytes() > 0
    freed = codec.evict()
    assert freed > 0
    codec.reset_stats()
    assert codec._hits == 0
    assert codec._misses == 0


def test_evict_lru_first() -> None:
    """max_entries=2, 3rd put evicts first entry."""
    config = RuntimeCertifiedConfig(d_head=64, max_entries=2, seed=42)
    codec = RuntimeCertifiedQuantizedAttentionCodec(config)
    codec.put("a", _make_kv(seed=1))
    codec.put("b", _make_kv(seed=2))
    codec.put("c", _make_kv(seed=3))  # triggers eviction of "a"
    assert "a" not in codec._store
    assert "b" in codec._store
    assert "c" in codec._store


def test_hit_rate_tracking() -> None:
    """2 puts, 1 hit + 1 miss -> hit_rate() == 0.5."""
    config = RuntimeCertifiedConfig(d_head=64, max_entries=100, seed=42)
    codec = RuntimeCertifiedQuantizedAttentionCodec(config)
    codec.put("a", _make_kv(seed=1))
    codec.put("b", _make_kv(seed=2))
    codec.get("a")       # hit
    codec.get("missing") # miss
    assert codec.hit_rate() == 0.5


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #


def test_seed_reproducibility() -> None:
    """Same seed + same input -> same quantization result."""
    def _run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        x = torch.randn(16, 64)
        x_int8, scale, zero = RuntimeCertifiedQuantizedAttentionCodec._quantize_int8(x)
        return RuntimeCertifiedQuantizedAttentionCodec._dequantize_int8(x_int8, scale, zero)

    r1 = _run(42)
    r2 = _run(42)
    assert torch.allclose(r1, r2), "Quantization is not reproducible with same seed"
