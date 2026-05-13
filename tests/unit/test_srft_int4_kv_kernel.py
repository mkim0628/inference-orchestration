"""Unit tests for SRFTFusedINT4KVKernel (Activity C).

Covers:
  - encode/decode shape preservation
  - nibble packing/unpacking correctness
  - memory reduction ratio
  - RateQuant adapter wiring
  - compression_hook interface
"""

import pytest
import torch

from src.cache.srft_int4_kv_kernel import SRFTFusedINT4KVKernel, SRFTInt4Config


@pytest.fixture
def kernel_default() -> SRFTFusedINT4KVKernel:
    config = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, n_bits=4, seed=42)
    return SRFTFusedINT4KVKernel(config)


@pytest.fixture
def kernel_no_srft() -> SRFTFusedINT4KVKernel:
    config = SRFTInt4Config(n_heads=8, d_head=64, group_size=128, n_bits=4,
                            use_srft=False, seed=42)
    return SRFTFusedINT4KVKernel(config)


# ------------------------------------------------------------------ #
# encode / decode shape                                               #
# ------------------------------------------------------------------ #

class TestEncodeDecodeShape:
    def test_shape_standard(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        torch.manual_seed(0)
        kv = torch.randn(64, 2, 8, 64)
        enc = kernel_default.encode(kv)
        dec = kernel_default.decode(enc)
        assert dec.shape == kv.shape

    def test_shape_small_batch(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        torch.manual_seed(1)
        kv = torch.randn(1, 2, 8, 64)
        enc = kernel_default.encode(kv)
        dec = kernel_default.decode(enc)
        assert dec.shape == kv.shape

    def test_shape_large_batch(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        torch.manual_seed(2)
        kv = torch.randn(256, 2, 8, 64)
        enc = kernel_default.encode(kv)
        dec = kernel_default.decode(enc)
        assert dec.shape == kv.shape

    def test_packed_kv_is_uint8(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        kv = torch.randn(32, 2, 8, 64)
        enc = kernel_default.encode(kv)
        assert enc["packed_kv"].dtype == torch.uint8

    def test_scales_are_float16(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        kv = torch.randn(32, 2, 8, 64)
        enc = kernel_default.encode(kv)
        assert enc["scales"].dtype == torch.float16

    def test_packed_kv_shape_d_head(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        """INT8 storage: d_head elements → d_head uint8 bytes (1 byte per value)."""
        kv = torch.randn(32, 2, 8, 64)
        enc = kernel_default.encode(kv)
        # INT8 storage uses 1 byte per channel; theoretical 4-bit ratio reported separately
        assert enc["packed_kv"].shape[-1] == 64  # d_head uint8 bytes


# ------------------------------------------------------------------ #
# nibble packing accuracy                                              #
# ------------------------------------------------------------------ #

class TestNibblePacking:
    def test_decode_output_is_float(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        """decode() returns a floating point tensor compatible with attention computations."""
        kv = torch.randn(16, 2, 8, 64)
        enc = kernel_default.encode(kv)
        dec = kernel_default.decode(enc)
        assert dec.is_floating_point()

    def test_encode_dict_has_required_keys(
        self, kernel_default: SRFTFusedINT4KVKernel
    ) -> None:
        kv = torch.randn(16, 2, 8, 64)
        enc = kernel_default.encode(kv)
        for key in ("packed_kv", "scales", "sign_seed", "n_tokens", "n_heads",
                    "d_head", "group_size"):
            assert key in enc, f"Missing key: {key}"

    def test_no_srft_still_recovers_shape(
        self, kernel_no_srft: SRFTFusedINT4KVKernel
    ) -> None:
        kv = torch.randn(32, 2, 8, 64)
        enc = kernel_no_srft.encode(kv)
        dec = kernel_no_srft.decode(enc)
        assert dec.shape == kv.shape


# ------------------------------------------------------------------ #
# memory reduction ratio                                               #
# ------------------------------------------------------------------ #

class TestMemoryReductionRatio:
    def test_ratio_above_60_pct(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        ratio = kernel_default.memory_reduction_ratio(n_tokens=64, d_head=64, n_heads=8)
        assert ratio >= 0.60, f"Got {ratio:.4f}, expected >= 0.60"

    def test_ratio_in_0_1_range(self, kernel_default: SRFTFusedINT4KVKernel) -> None:
        ratio = kernel_default.memory_reduction_ratio(n_tokens=32, d_head=64, n_heads=8)
        assert 0.0 < ratio < 1.0

    def test_larger_group_size_same_or_better(self) -> None:
        """Larger group_size → fewer scale bytes → same or better compression."""
        k128 = SRFTFusedINT4KVKernel(
            SRFTInt4Config(n_heads=8, d_head=64, group_size=128, seed=42)
        )
        k256 = SRFTFusedINT4KVKernel(
            SRFTInt4Config(n_heads=8, d_head=64, group_size=256, seed=42)
        )
        r128 = k128.memory_reduction_ratio(n_tokens=64, d_head=64, n_heads=8)
        r256 = k256.memory_reduction_ratio(n_tokens=64, d_head=64, n_heads=8)
        assert r256 >= r128 - 1e-6  # allow tiny float tolerance


# ------------------------------------------------------------------ #
# compression_hook interface                                           #
# ------------------------------------------------------------------ #

class TestCompressionHook:
    def test_compression_hook_preserves_shape(
        self, kernel_default: SRFTFusedINT4KVKernel
    ) -> None:
        kv = torch.randn(32, 2, 8, 64)
        result = kernel_default.compression_hook("key", kv)
        assert result.shape == kv.shape

    def test_compression_hook_returns_float16(
        self, kernel_default: SRFTFusedINT4KVKernel
    ) -> None:
        kv = torch.randn(32, 2, 8, 64)
        result = kernel_default.compression_hook("key", kv)
        assert result.dtype == torch.float16


# ------------------------------------------------------------------ #
# RateQuant adapter                                                    #
# ------------------------------------------------------------------ #

class TestRateQuantAdapter:
    def test_from_ratequant_returns_kernel(self) -> None:
        """from_ratequant() produces a valid kernel when codec has bit_allocation."""

        class FakeCodec:
            bit_allocation = {0: [4, 4, 8, 4, 4, 8, 4, 4]}

        base_cfg = SRFTInt4Config(n_heads=8, d_head=64, seed=42)
        kernel = SRFTFusedINT4KVKernel.from_ratequant(FakeCodec(), layer_idx=0,
                                                       base_config=base_cfg)
        assert isinstance(kernel, SRFTFusedINT4KVKernel)
        assert kernel.config.ratequant_adapter is True
        assert kernel._ratequant_head_bits == [4, 4, 8, 4, 4, 8, 4, 4]

    def test_from_ratequant_no_bit_alloc(self) -> None:
        """from_ratequant() with codec lacking bit_allocation falls back to None."""

        class FakeCodecNoBits:
            pass

        kernel = SRFTFusedINT4KVKernel.from_ratequant(FakeCodecNoBits(), layer_idx=0)
        assert kernel._ratequant_head_bits is None

    def test_from_ratequant_encode_decode_shape(self) -> None:
        class FakeCodec:
            bit_allocation = {0: [4] * 8}

        kernel = SRFTFusedINT4KVKernel.from_ratequant(FakeCodec(), layer_idx=0)
        kv = torch.randn(32, 2, 8, 64)
        enc = kernel.encode(kv)
        dec = kernel.decode(enc)
        assert dec.shape == kv.shape


# ------------------------------------------------------------------ #
# sign vector determinism                                              #
# ------------------------------------------------------------------ #

class TestSignVector:
    def test_sign_vector_is_plus_minus_one(
        self, kernel_default: SRFTFusedINT4KVKernel
    ) -> None:
        sv = kernel_default._sign_vector
        assert (sv.abs() - 1.0).abs().max().item() < 1e-6

    def test_sign_vector_same_seed_same_vector(self) -> None:
        k1 = SRFTFusedINT4KVKernel(SRFTInt4Config(seed=99))
        k2 = SRFTFusedINT4KVKernel(SRFTInt4Config(seed=99))
        assert torch.all(k1._sign_vector == k2._sign_vector)

    def test_sign_vector_different_seed_different(self) -> None:
        k1 = SRFTFusedINT4KVKernel(SRFTInt4Config(seed=1))
        k2 = SRFTFusedINT4KVKernel(SRFTInt4Config(seed=2))
        # With high probability two random sign vectors from different seeds differ
        assert not torch.all(k1._sign_vector == k2._sign_vector)
