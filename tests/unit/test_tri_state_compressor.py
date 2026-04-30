"""Unit tests for TriStateCompressor (Activity C).

Tests verify:
  - Token classification ratios (retain/compress/evict)
  - Encode/decode roundtrip attention-based KL divergence < 0.05 (consistent with
    existing HadamardInt4Codec accuracy tests; INT4 inherent error makes raw-KV
    KL higher, but attention-output KL stays within the 0.05 bound validated in
    test_compression_accuracy.py::test_attention_kl_divergence)
  - Cosine similarity >= 0.90 for non-evicted tokens
  - Compression ratio <= 0.25 (>= 75% savings vs FP32)
  - Accuracy preservation proxy: retain tier (FP16) keeps attention-KL near 0,
    combined retain+compress tier keeps attention-KL < 0.05
"""

import torch
import torch.nn.functional as F
import pytest

from src.cache.compression import HadamardInt4Codec
from src.cache.tri_state_compressor import TriStateCompressor


@pytest.fixture
def codec() -> HadamardInt4Codec:
    return HadamardInt4Codec(num_layers=12, cutoff_ratio=0.2)


@pytest.fixture
def compressor(codec: HadamardInt4Codec) -> TriStateCompressor:
    return TriStateCompressor(codec=codec, retain_ratio=0.20, evict_ratio=0.40)


def _attention_kl(
    kv_orig: torch.Tensor,
    kv_decoded: torch.Tensor,
    n_queries: int = 8,
    seed: int = 0,
) -> float:
    """Attention-score KL divergence (batchmean) as perplexity proxy.

    Mirrors the methodology in test_compression_accuracy.py::TestHadamardInt4Accuracy.
    """
    torch.manual_seed(seed)
    q = torch.randn(n_queries, kv_orig.shape[-1])
    scale = kv_orig.shape[-1] ** -0.5

    attn_orig = F.softmax(q @ kv_orig.float().T * scale, dim=-1)
    attn_dec = F.softmax(q @ kv_decoded.float().T * scale, dim=-1)

    kl = F.kl_div(
        attn_dec.log().clamp(min=-100),
        attn_orig,
        reduction="batchmean",
    )
    return kl.item()


def test_classify_ratios(compressor: TriStateCompressor) -> None:
    """100 tokens → retain≈20, compress≈40, evict≈40 (±1 due to rounding)."""
    torch.manual_seed(0)
    n_tokens = 100
    kv = torch.randn(n_tokens, 64)
    attn_weights = torch.rand(n_tokens)

    result = compressor.classify(kv, attn_weights, layer_idx=6)

    n_retain = result["retain_indices"].numel()
    n_compress = result["compress_indices"].numel()
    n_evict = result["evict_indices"].numel()

    assert n_retain + n_compress + n_evict == n_tokens, (
        f"Partition sizes {n_retain}+{n_compress}+{n_evict} != {n_tokens}"
    )
    assert abs(n_retain - 20) <= 1, f"retain count {n_retain} not ≈ 20"
    assert abs(n_evict - 40) <= 1, f"evict count {n_evict} not ≈ 40"
    assert abs(n_compress - 40) <= 1, f"compress count {n_compress} not ≈ 40"


def test_encode_decode_roundtrip_kl(compressor: TriStateCompressor) -> None:
    """Attention-based KL divergence of non-evicted tokens should be < 0.05.

    INT4 quantisation on Gaussian data has inherent ~10-20% raw-KV L2 error,
    but attention-output KL stays below 0.05 (same threshold as
    test_compression_accuracy.py::TestHadamardInt4Accuracy::test_attention_kl_divergence).
    """
    torch.manual_seed(1)
    n_tokens = 50
    kv = torch.randn(n_tokens, 64)
    attn_weights = torch.rand(n_tokens)

    storage = compressor.encode(kv, attn_weights, layer_idx=6, tensor_id=1)
    decoded = compressor.decode(storage, layer_idx=6, tensor_id=1)

    non_evict = torch.cat([storage["retain_indices"], storage["compress_indices"]])
    if non_evict.numel() < 2:
        pytest.skip("Too few non-evicted tokens for KL test")

    kl = _attention_kl(kv[non_evict], decoded[non_evict], seed=10)

    assert kl < 0.05, (
        f"Attention-KL divergence {kl:.6f} >= 0.05 "
        "(INT4 compress tier exceeds attention-output accuracy threshold)"
    )


def test_encode_decode_cosine(compressor: TriStateCompressor) -> None:
    """Cosine similarity between decoded and original >= 0.90 for non-evicted tokens."""
    torch.manual_seed(2)
    n_tokens = 80
    kv = torch.randn(n_tokens, 64)
    attn_weights = torch.rand(n_tokens)

    storage = compressor.encode(kv, attn_weights, layer_idx=6, tensor_id=2)
    decoded = compressor.decode(storage, layer_idx=6, tensor_id=2)

    retain_indices = storage["retain_indices"]
    compress_indices = storage["compress_indices"]
    non_evict_indices = torch.cat([retain_indices, compress_indices])

    if non_evict_indices.numel() == 0:
        pytest.skip("No non-evicted tokens")

    orig = kv[non_evict_indices].float()
    dec = decoded[non_evict_indices].float()

    cos_sim = F.cosine_similarity(orig, dec, dim=-1)
    mean_cos = cos_sim.mean().item()

    assert mean_cos >= 0.90, f"Mean cosine similarity {mean_cos:.4f} < 0.90"


def test_compression_ratio_above_75pct(compressor: TriStateCompressor) -> None:
    """compression_ratio() <= 0.25 means >= 75% savings vs FP32."""
    ratio = compressor.compression_ratio(retain_ratio=0.20, evict_ratio=0.40)
    assert ratio <= 0.25, (
        f"Compression ratio {ratio:.4f} > 0.25 (less than 75% savings)"
    )


def test_accuracy_preservation_proxy(compressor: TriStateCompressor) -> None:
    """±1% perplexity proxy via attention-KL < 0.05 for retain+compress tiers.

    The retain tier (FP16) has near-zero KL; the compress tier (INT4) contributes
    the bulk but stays within the 0.05 bound validated by existing codec tests.
    FP16-only encode path (early layer) achieves attention-KL ≈ 0.
    """
    torch.manual_seed(3)
    n_tokens = 100
    kv = torch.randn(n_tokens, 64)
    attn_weights = torch.rand(n_tokens)

    # Early layer: codec uses FP16 for compress tier — high accuracy
    storage_fp16 = compressor.encode(kv, attn_weights, layer_idx=0, tensor_id=3)
    decoded_fp16 = compressor.decode(storage_fp16, layer_idx=0, tensor_id=3)

    non_evict_fp16 = torch.cat([
        storage_fp16["retain_indices"], storage_fp16["compress_indices"]
    ])
    kl_fp16 = _attention_kl(kv[non_evict_fp16], decoded_fp16[non_evict_fp16], seed=20)
    assert kl_fp16 < 0.01, (
        f"FP16 layer attention-KL {kl_fp16:.6f} >= 0.01 "
        "(retain + FP16-compress should be near-lossless)"
    )

    # Late layer: codec uses INT4 for compress tier — bounded accuracy
    storage_int4 = compressor.encode(kv, attn_weights, layer_idx=6, tensor_id=4)
    decoded_int4 = compressor.decode(storage_int4, layer_idx=6, tensor_id=4)

    non_evict_int4 = torch.cat([
        storage_int4["retain_indices"], storage_int4["compress_indices"]
    ])
    kl_int4 = _attention_kl(kv[non_evict_int4], decoded_int4[non_evict_int4], seed=21)
    assert kl_int4 < 0.05, (
        f"INT4 layer attention-KL {kl_int4:.6f} >= 0.05 "
        "(exceeds accuracy threshold for ±1% perplexity proxy)"
    )
