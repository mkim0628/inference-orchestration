"""Activity C MANDATORY accuracy-preservation tests for SRFTFusedINT4KVKernel.

All proxy thresholds match evaluation_criteria.md §4:
  - attention_output_relative_error < 0.01  (±1% perplexity)
  - attention_kl_divergence < 0.015
  - cosine_similarity_output >= 0.99
"""

import json
import os

import pytest
import torch

from src.cache.srft_int4_kv_kernel import SRFTFusedINT4KVKernel, SRFTInt4Config
from src.metrics.perplexity import (
    attention_kl_divergence,
    attention_output_relative_error,
    cosine_similarity_output,
)


@pytest.fixture
def kernel_default() -> SRFTFusedINT4KVKernel:
    """group_size=128, INT4, SRFT ON default kernel."""
    config = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, n_bits=4, seed=42)
    return SRFTFusedINT4KVKernel(config)


def test_encode_decode_shape(kernel_default: SRFTFusedINT4KVKernel) -> None:
    """encode → decode preserves shape [n_tokens, 2, n_heads, d_head]."""
    torch.manual_seed(42)
    kv = torch.randn(64, 2, 8, 64)
    encoded = kernel_default.encode(kv)
    recovered = kernel_default.decode(encoded)
    assert recovered.shape == kv.shape


def test_memory_reduction_ratio(kernel_default: SRFTFusedINT4KVKernel) -> None:
    """INT4 nibble packing achieves >= 60% memory reduction. evaluation_criteria.md §4."""
    ratio = kernel_default.memory_reduction_ratio(n_tokens=64, d_head=64, n_heads=8)
    assert ratio >= 0.60, f"Memory reduction {ratio:.4f} below 60% target"


def test_accuracy_relative_error_within_1pct(kernel_default: SRFTFusedINT4KVKernel) -> None:
    """PRIMARY ±1% accuracy: relative output error < 0.01.
    group_size=128, SRFT ON. evaluation_criteria.md §4 MANDATORY."""
    torch.manual_seed(99)  # independent seed from calibration
    kv = torch.randn(32, 2, 8, 64)
    k_orig = kv[:, 0, 0, :]
    v_orig = kv[:, 1, 0, :]
    encoded = kernel_default.encode(kv)
    kv_rec = kernel_default.decode(encoded)
    k_comp = kv_rec[:, 0, 0, :]
    v_comp = kv_rec[:, 1, 0, :]
    q = torch.randn(8, 64)
    error = attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp)
    assert error < 0.01, f"Relative error {error:.4f} exceeds ±1% limit"


def test_kl_divergence_within_threshold(kernel_default: SRFTFusedINT4KVKernel) -> None:
    """KL divergence < 0.015. evaluation_criteria.md §4."""
    torch.manual_seed(77)
    kv = torch.randn(32, 2, 8, 64)
    encoded = kernel_default.encode(kv)
    kv_rec = kernel_default.decode(encoded)
    q = torch.randn(8, 64)
    kl = attention_kl_divergence(q, kv[:, 0, 0, :], kv_rec[:, 0, 0, :])
    assert kl < 0.015, f"KL divergence {kl:.4f} exceeds 0.015 threshold"


def test_cosine_similarity_above_threshold(kernel_default: SRFTFusedINT4KVKernel) -> None:
    """Cosine similarity >= 0.99. evaluation_criteria.md §4."""
    torch.manual_seed(55)
    kv = torch.randn(32, 2, 8, 64)
    encoded = kernel_default.encode(kv)
    kv_rec = kernel_default.decode(encoded)
    q = torch.randn(8, 64)
    sim = cosine_similarity_output(
        q,
        kv[:, 0, 0, :],
        kv[:, 1, 0, :],
        kv_rec[:, 0, 0, :],
        kv_rec[:, 1, 0, :],
    )
    assert sim >= 0.99, f"Cosine similarity {sim:.4f} below 0.99 threshold"


def test_srft_vs_plain_int4_accuracy() -> None:
    """SRFT ON error <= plain INT4 error with outlier channels.
    Demonstrates SRFT Gaussianisation effect. evaluation_criteria.md §4."""
    torch.manual_seed(42)
    kv = torch.randn(32, 2, 8, 64)
    kv[:, :, :, ::8] *= 10.0  # inject outliers in every 8th channel

    config_srft = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, use_srft=True, seed=42)
    config_plain = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, use_srft=False, seed=42)
    kernel_srft = SRFTFusedINT4KVKernel(config_srft)
    kernel_plain = SRFTFusedINT4KVKernel(config_plain)

    q = torch.randn(8, 64)
    k_orig, v_orig = kv[:, 0, 0, :], kv[:, 1, 0, :]

    enc_srft = kernel_srft.encode(kv)
    rec_srft = kernel_srft.decode(enc_srft)
    err_srft = attention_output_relative_error(
        q, k_orig, v_orig, rec_srft[:, 0, 0, :], rec_srft[:, 1, 0, :]
    )

    enc_plain = kernel_plain.encode(kv)
    rec_plain = kernel_plain.decode(enc_plain)
    err_plain = attention_output_relative_error(
        q, k_orig, v_orig, rec_plain[:, 0, 0, :], rec_plain[:, 1, 0, :]
    )

    assert err_srft <= err_plain, (
        f"SRFT error {err_srft:.4f} should be <= plain INT4 error {err_plain:.4f}"
    )


def test_independent_seed_accuracy(kernel_default: SRFTFusedINT4KVKernel) -> None:
    """Accuracy preserved with completely independent seed. evaluation_criteria.md §4 MANDATORY."""
    torch.manual_seed(999)
    test_kv = torch.randn(32, 2, 8, 64)
    q = torch.randn(8, 64)
    encoded = kernel_default.encode(test_kv)
    kv_rec = kernel_default.decode(encoded)
    error = attention_output_relative_error(
        q,
        test_kv[:, 0, 0, :],
        test_kv[:, 1, 0, :],
        kv_rec[:, 0, 0, :],
        kv_rec[:, 1, 0, :],
    )
    assert error < 0.01, f"Independent test error {error:.4f} exceeds ±1%"


def test_group_size_sweep() -> None:
    """group_size 64/128/256 sweep — compression vs accuracy curve.
    Saves results to results/2026-05-13/perplexity_sweep.json."""
    torch.manual_seed(42)
    kv = torch.randn(64, 2, 8, 64)
    q = torch.randn(8, 64)
    results: dict = {}
    for gs in [64, 128, 256]:
        config = SRFTInt4Config(n_heads=8, d_head=64, group_size=gs, seed=42)
        kernel = SRFTFusedINT4KVKernel(config)
        encoded = kernel.encode(kv)
        kv_rec = kernel.decode(encoded)
        err = attention_output_relative_error(
            q,
            kv[:, 0, 0, :],
            kv[:, 1, 0, :],
            kv_rec[:, 0, 0, :],
            kv_rec[:, 1, 0, :],
        )
        mem_red = kernel.memory_reduction_ratio(n_tokens=64, d_head=64, n_heads=8)
        results[str(gs)] = {
            "group_size": gs,
            "memory_reduction": mem_red,
            "relative_error": err,
            "pass_1pct": err < 0.01,
        }
        assert kv_rec.shape == kv.shape
    os.makedirs("results/2026-05-13", exist_ok=True)
    with open("results/2026-05-13/perplexity_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
