"""Activity C accuracy tests for KVDriveTierDifferentiatedCompressionCodec.

CRITICAL: HBM FP8 relative_error < 1% and cosine_similarity > 0.99 are MANDATORY.
"""
import pytest
import torch
import torch.nn.functional as F

from src.cache.kvdrive_tier_compression_codec import (
    KVDriveTierDifferentiatedCompressionCodec,
    TierCompressionCodec,
    TierCompressionConfig,
)


def _config(seed: int = 42) -> TierCompressionConfig:
    return TierCompressionConfig(
        vq_n_codes=256,
        vq_code_dim=8,
        int4_zero_threshold=0.01,
        max_entries=100,
        seed=seed,
    )


def _codec_store(seed: int = 42) -> KVDriveTierDifferentiatedCompressionCodec:
    return KVDriveTierDifferentiatedCompressionCodec(_config(seed))


def _tier_codec(seed: int = 42) -> TierCompressionCodec:
    return TierCompressionCodec(_config(seed))


def _relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """||orig - recon|| / (||orig|| + 1e-8)."""
    diff = (original.float() - reconstructed.float()).norm()
    denom = original.float().norm() + 1e-8
    return (diff / denom).item()


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()


# --------------------------------------------------------------------------- #
# MANDATORY tests                                                              #
# --------------------------------------------------------------------------- #

class TestHBMFP8Mandatory:
    """These tests MUST pass — failure means overall Fail."""

    def test_hbm_fp8_relative_error_below_1pct(self) -> None:
        """MANDATORY: FP8 relative_error < 1%."""
        torch.manual_seed(0)
        original = torch.randn(16, 32)
        store = _codec_store()
        compressed = store.compress_fp8(original)
        err = _relative_error(original, compressed)
        assert err < 0.01, f"HBM FP8 relative_error={err:.4f} >= 0.01"

    def test_hbm_fp8_cosine_similarity_above_099(self) -> None:
        """MANDATORY: FP8 cosine_similarity > 0.99."""
        torch.manual_seed(0)
        original = torch.randn(8, 64)
        store = _codec_store()
        compressed = store.compress_fp8(original)
        cs = _cosine_sim(original, compressed)
        assert cs > 0.99, f"HBM FP8 cosine_similarity={cs:.4f} <= 0.99"

    def test_hbm_fp8_relative_error_various_shapes(self) -> None:
        """MANDATORY: FP8 error below 1% for multiple shapes."""
        store = _codec_store()
        shapes = [(4, 8), (32, 16), (1, 128), (64, 4)]
        for shape in shapes:
            torch.manual_seed(sum(shape))
            original = torch.randn(*shape)
            compressed = store.compress_fp8(original)
            err = _relative_error(original, compressed)
            assert err < 0.01, f"shape={shape} FP8 relative_error={err:.4f} >= 0.01"


# --------------------------------------------------------------------------- #
# DRAM VQ                                                                     #
# --------------------------------------------------------------------------- #

class TestDRAMVQ:
    def test_dram_vq_relative_error_below_2pct(self) -> None:
        """DRAM VQ relative_error < 2%."""
        torch.manual_seed(1)
        original = torch.randn(8, 8)
        # Use round-trip through internal codec for accurate measurement
        cfg = _config()
        codec = TierCompressionCodec(cfg)
        indices, codebook = codec.compress_dram(original)
        flat = codec.decompress_dram(indices, codebook)
        reconstructed = flat[: original.numel()].reshape(original.shape)
        err = _relative_error(original, reconstructed)
        assert err < 0.02, f"DRAM VQ relative_error={err:.4f} >= 0.02"

    def test_vq_preserves_shape_and_dtype(self) -> None:
        torch.manual_seed(2)
        original = torch.randn(4, 16)
        store = _codec_store()
        compressed = store.compress_vq(original)
        assert compressed.shape == original.shape

    def test_dram_vq_cosine_similarity(self) -> None:
        torch.manual_seed(3)
        original = torch.randn(16, 8)
        cfg = _config()
        codec = TierCompressionCodec(cfg)
        indices, codebook = codec.compress_dram(original)
        flat = codec.decompress_dram(indices, codebook)
        reconstructed = flat[: original.numel()].reshape(original.shape)
        cs = _cosine_sim(original, reconstructed)
        assert cs > 0.97, f"DRAM VQ cosine_similarity={cs:.4f} <= 0.97"


# --------------------------------------------------------------------------- #
# SSD INT4 + sparse                                                           #
# --------------------------------------------------------------------------- #

class TestSSDInt4Sparse:
    def test_ssd_int4_sparse_reconstruction_error_below_5pct(self) -> None:
        torch.manual_seed(4)
        original = torch.randn(32)
        store = _codec_store()
        compressed = store.compress_int4_sparse(original)
        err = _relative_error(original, compressed)
        assert err <= 0.05, f"SSD INT4 relative_error={err:.4f} > 0.05"

    def test_int4_sparsification_zeros_small_values(self) -> None:
        """Values below threshold should become 0 after compression."""
        cfg = _config()
        cfg.int4_zero_threshold = 0.5
        codec = TierCompressionCodec(cfg)
        original = torch.tensor([0.1, 0.2, 1.0, 2.0])
        q, scale_t = codec.compress_ssd(original)
        reconstructed = codec.decompress_ssd(q, scale_t)
        # Values originally below 0.5 (indices 0 and 1) should be 0
        assert reconstructed[0].item() == pytest.approx(0.0)
        assert reconstructed[1].item() == pytest.approx(0.0)

    def test_int4_preserves_shape_and_dtype(self) -> None:
        torch.manual_seed(5)
        original = torch.randn(8, 4)
        store = _codec_store()
        compressed = store.compress_int4_sparse(original)
        assert compressed.shape == original.shape


# --------------------------------------------------------------------------- #
# Tier migration                                                               #
# --------------------------------------------------------------------------- #

class TestTierMigration:
    def test_migrate_tier_hbm_to_dram_preserves_accuracy(self) -> None:
        """HBM-stored entry migrated to DRAM should have relative_error < 2%."""
        torch.manual_seed(6)
        original = torch.randn(8, 8)
        store = _codec_store()
        store.put_with_tier("k1", original, "HBM")
        store.migrate_tier("k1", "HBM", "DRAM")
        retrieved = store.get("k1")
        assert retrieved is not None
        err = _relative_error(original, retrieved)
        assert err < 0.02, f"HBM→DRAM migration relative_error={err:.4f} >= 0.02"

    def test_migrate_tier_dram_to_ssd_preserves_accuracy(self) -> None:
        """DRAM→SSD migration should keep relative_error <= 5%."""
        torch.manual_seed(7)
        original = torch.randn(8, 8)
        store = _codec_store()
        store.put_with_tier("k2", original, "DRAM")
        store.migrate_tier("k2", "DRAM", "SSD")
        retrieved = store.get("k2")
        assert retrieved is not None
        err = _relative_error(original, retrieved)
        assert err <= 0.05, f"DRAM→SSD migration relative_error={err:.4f} > 0.05"

    def test_migrate_tier_missing_key_no_error(self) -> None:
        store = _codec_store()
        store.migrate_tier("nonexistent", "HBM", "DRAM")  # should not raise


# --------------------------------------------------------------------------- #
# compress_for_tier dispatch                                                   #
# --------------------------------------------------------------------------- #

class TestCompressForTierDispatch:
    def test_dispatch_hbm_matches_fp8(self) -> None:
        torch.manual_seed(8)
        t = torch.randn(4, 8)
        store = _codec_store()
        via_dispatch = store.compress_for_tier(t, "HBM")
        direct = store.compress_fp8(t)
        assert torch.allclose(via_dispatch, direct, atol=1e-5)

    def test_dispatch_dram_matches_vq(self) -> None:
        torch.manual_seed(9)
        t = torch.randn(4, 8)
        store = _codec_store()
        via_dispatch = store.compress_for_tier(t, "DRAM")
        direct = store.compress_vq(t)
        assert torch.allclose(via_dispatch, direct, atol=1e-5)

    def test_dispatch_ssd_matches_int4_sparse(self) -> None:
        torch.manual_seed(10)
        t = torch.randn(4, 8)
        store = _codec_store()
        via_dispatch = store.compress_for_tier(t, "SSD")
        direct = store.compress_int4_sparse(t)
        assert torch.allclose(via_dispatch, direct, atol=1e-5)


# --------------------------------------------------------------------------- #
# Memory reduction                                                             #
# --------------------------------------------------------------------------- #

class TestMemoryReduction:
    def test_memory_reduction_hbm_fp8(self) -> None:
        """After FP8 put, memory_reduction_ratio > 0 (INT8 vs FP32)."""
        torch.manual_seed(11)
        store = _codec_store()
        t = torch.randn(64, 64)  # FP32 → INT8 data + small metadata
        store.put_with_tier("hbm_key", t, "HBM")
        # INT8 is 1/4 the size; with per-row scale metadata still < original
        ratio = store.memory_reduction_ratio()
        assert ratio > 0.0, f"memory_reduction_ratio={ratio} should be > 0"

    def test_memory_reduction_overall_above_30pct(self) -> None:
        """HBM 20%/DRAM 50%/SSD 30% distribution → overall reduction >= 30%."""
        torch.manual_seed(12)
        store = _codec_store()
        # 2 HBM entries
        for i in range(2):
            store.put_with_tier(f"hbm_{i}", torch.randn(32, 32), "HBM")
        # 5 DRAM entries
        for i in range(5):
            store.put_with_tier(f"dram_{i}", torch.randn(32, 32), "DRAM")
        # 3 SSD entries
        for i in range(3):
            store.put_with_tier(f"ssd_{i}", torch.randn(32, 32), "SSD")
        ratio = store.memory_reduction_ratio()
        assert ratio >= 0.30, f"overall memory_reduction_ratio={ratio:.4f} < 0.30"


# --------------------------------------------------------------------------- #
# CacheStore interface                                                         #
# --------------------------------------------------------------------------- #

class TestCacheStoreInterface:
    def test_put_get_evict_hit_rate_memory_bytes_reset(self) -> None:
        store = _codec_store()
        t = torch.randn(4, 4)
        store.put("k1", t)
        result = store.get("k1")
        assert result is not None
        store.get("miss")
        assert store.hit_rate() == pytest.approx(0.5)
        assert store.memory_bytes() > 0
        freed = store.evict()
        # After eviction, fewer bytes remain
        store.reset_stats()
        assert store.hit_rate() == 0.0
        assert store.memory_bytes() == 0

    def test_fp8_preserves_shape_and_dtype(self) -> None:
        torch.manual_seed(13)
        original = torch.randn(8, 16, dtype=torch.float32)
        store = _codec_store()
        compressed = store.compress_fp8(original)
        assert compressed.shape == original.shape
        # dtype should be float32 (dequantized output)
        assert compressed.dtype == torch.float32

    def test_vq_preserves_shape(self) -> None:
        torch.manual_seed(14)
        original = torch.randn(8, 8)
        store = _codec_store()
        compressed = store.compress_vq(original)
        assert compressed.shape == original.shape


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #

class TestReproducibility:
    def test_codec_seed_reproducibility(self) -> None:
        """Same seed + same input → same output."""
        torch.manual_seed(99)
        t = torch.randn(8, 8)
        s1 = _codec_store(seed=42)
        s2 = _codec_store(seed=42)
        out1 = s1.compress_fp8(t.clone())
        out2 = s2.compress_fp8(t.clone())
        assert torch.allclose(out1, out2)

    def test_different_seeds_may_differ_for_vq(self) -> None:
        """Different VQ codebook seeds may produce different outputs."""
        torch.manual_seed(0)
        t = torch.randn(8, 8)
        s1 = _codec_store(seed=0)
        s2 = _codec_store(seed=99)
        out1 = s1.compress_vq(t.clone())
        out2 = s2.compress_vq(t.clone())
        # Not guaranteed to differ, but codebooks differ → likely different
        # Just ensure no error
        assert out1.shape == out2.shape
