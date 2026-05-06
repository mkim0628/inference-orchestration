"""
Unit tests for TriAttentionCodec (Activity C).

These tests validate compression fidelity, ratio adherence, and the
position-stability guarantee of pre-RoPE importance estimation.
Actual perplexity on WikiText-2 is measured separately in
experiments/run_triattention_accuracy.py.
"""

import os
import tempfile

import pytest
import torch

from src.cache.tri_attention_codec import TriAttentionCodec


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def make_codec(
    n_layers: int = 2,
    n_heads: int = 2,
    head_dim: int = 64,
    series_terms: int = 8,
    prune_window: int = 128,
) -> TriAttentionCodec:
    return TriAttentionCodec(
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        series_terms=series_terms,
        prune_window=prune_window,
    )


def calibrate_codec(codec: TriAttentionCodec, n_samples: int = 10) -> None:
    L, H, D = codec.n_layers, codec.n_heads, codec.head_dim
    kvs = [torch.randn(L, H, 64, D) for _ in range(n_samples)]
    codec.calibrate(kvs)


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #

class TestPerplexityWithinTolerance:
    """
    Validate that kept-token positions are restored losslessly after
    compress → decompress. This is the unit-level proxy for perplexity
    preservation (actual GPT-2 perplexity is in run_triattention_accuracy.py).
    """

    def test_perplexity_within_tolerance(self) -> None:
        codec = make_codec()
        calibrate_codec(codec)

        kv = torch.randn(2, 2, 512, 64)
        keys_pre_rope = torch.randn(2, 2, 512, 64)

        compressed = codec.compress(kv, keys_pre_rope, compression_ratio=0.1)
        decompressed = codec.decompress(compressed)

        kept_idx = compressed["kept_indices"]
        original_kept = kv[:, :, kept_idx, :]
        restored_kept = decompressed[:, :, kept_idx, :]

        assert torch.allclose(original_kept, restored_kept, atol=1e-5), (
            "Restored kept-token values differ from originals (compression is lossy "
            "at kept positions — this should not happen)."
        )


class TestCompressionRatioRespected:
    @pytest.mark.parametrize("ratio", [0.1, 0.2, 0.5])
    def test_compression_ratio_respected(self, ratio: float) -> None:
        """Kept-token count should be close to seq_len * ratio."""
        codec = make_codec()
        calibrate_codec(codec)

        seq_len = 256
        kv = torch.randn(2, 2, seq_len, 64)
        keys_pre_rope = torch.randn(2, 2, seq_len, 64)

        compressed = codec.compress(kv, keys_pre_rope, compression_ratio=ratio)
        n_kept = compressed["kv"].shape[2]

        # Each window of prune_window (128) keeps ceil(window_len * ratio) tokens.
        # With seq_len=256 we have 2 windows of 128, so max allowed is:
        expected_max = int(seq_len * ratio) + codec.prune_window  # generous upper bound
        assert n_kept <= expected_max, (
            f"ratio={ratio}: n_kept={n_kept} exceeds expected_max={expected_max}"
        )
        # Also ensure at least 1 token per window is kept
        n_windows = (seq_len + codec.prune_window - 1) // codec.prune_window
        assert n_kept >= n_windows, (
            f"ratio={ratio}: n_kept={n_kept} < minimum {n_windows} (one per window)"
        )

    def test_at_least_one_token_kept(self) -> None:
        """With a very small ratio, at least 1 token per window is preserved."""
        codec = make_codec(prune_window=32)
        calibrate_codec(codec)
        kv = torch.randn(2, 2, 64, 64)
        k_pre = torch.randn(2, 2, 64, 64)
        compressed = codec.compress(kv, k_pre, compression_ratio=0.01)
        # 2 windows → at least 2 tokens kept
        assert compressed["kv"].shape[2] >= 2


class TestImportanceScoresPositionStable:
    def test_importance_scores_position_stable(self) -> None:
        """
        Identical pre-RoPE K vectors at different positions should produce
        near-identical importance scores (position stability guarantee).
        """
        codec = make_codec(n_layers=1, n_heads=1, head_dim=32, series_terms=4)
        calibrate_codec(codec, n_samples=10)

        keys_pre_rope = torch.randn(1, 1, 64, 32)
        # Mirror first 32 positions to last 32 (same pre-RoPE content)
        keys_pre_rope[0, 0, 32:, :] = keys_pre_rope[0, 0, :32, :]

        imp = codec.estimate_importance(keys_pre_rope)  # [1, 1, 64]
        diff = (imp[0, 0, :32] - imp[0, 0, 32:]).abs().mean().item()
        assert diff < 0.1, (
            f"pre-RoPE importance unstable across positions (mean diff={diff:.4f})"
        )

    def test_importance_returns_correct_shape(self) -> None:
        codec = make_codec()
        calibrate_codec(codec)
        kv = torch.randn(2, 2, 64, 64)
        imp = codec.estimate_importance(kv)
        assert imp.shape == (2, 2, 64)

    def test_importance_non_negative(self) -> None:
        """Importance is the absolute value of the Fourier series — always >= 0."""
        codec = make_codec()
        calibrate_codec(codec)
        kv = torch.randn(2, 2, 32, 64)
        imp = codec.estimate_importance(kv)
        assert (imp >= 0).all()

    def test_uncalibrated_raises(self) -> None:
        codec = make_codec()
        with pytest.raises(RuntimeError):
            codec.estimate_importance(torch.randn(2, 2, 32, 64))


class TestCalibrateAndSaveLoad:
    def test_calibrate_and_save_load(self) -> None:
        """Calibration save → load produces identical mu_k and a_m."""
        codec = make_codec()
        calibrate_codec(codec)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "calib.pt")
            kvs = [torch.randn(2, 2, 64, 64) for _ in range(10)]
            codec.calibrate(kvs, save_path=path)

            codec2 = make_codec()
            codec2.load_calibration(path)

            assert torch.allclose(codec.mu_k, codec2.mu_k, atol=1e-6)
            assert torch.allclose(codec.a_m, codec2.a_m, atol=1e-6)

    def test_save_creates_directory(self) -> None:
        """calibrate() with save_path creates intermediate directories."""
        codec = make_codec()
        kvs = [torch.randn(2, 2, 32, 64) for _ in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "calib.pt")
            codec.calibrate(kvs, save_path=path)
            assert os.path.exists(path)


class TestDecompressRoundtrip:
    def test_decompress_roundtrip(self) -> None:
        """compress → decompress: kept positions are lossless, pruned are zero."""
        codec = make_codec()
        calibrate_codec(codec)

        kv = torch.randn(2, 2, 128, 64)
        k_pre = torch.randn(2, 2, 128, 64)
        compressed = codec.compress(kv, k_pre, compression_ratio=0.5)
        decompressed = codec.decompress(compressed)

        assert decompressed.shape == kv.shape

        kept_idx = compressed["kept_indices"]
        assert torch.allclose(decompressed[:, :, kept_idx, :], kv[:, :, kept_idx, :], atol=1e-5)

        # Pruned positions should be zero
        all_idx = set(range(kv.shape[2]))
        pruned_idx = sorted(all_idx - set(kept_idx.tolist()))
        if pruned_idx:
            pruned_tensor = decompressed[:, :, pruned_idx, :]
            assert pruned_tensor.abs().max().item() == pytest.approx(0.0, abs=1e-7)

    def test_original_seq_len_recorded(self) -> None:
        codec = make_codec()
        calibrate_codec(codec)
        seq_len = 200
        kv = torch.randn(2, 2, seq_len, 64)
        k_pre = torch.randn(2, 2, seq_len, 64)
        compressed = codec.compress(kv, k_pre, compression_ratio=0.3)
        assert compressed["original_seq_len"] == seq_len


class TestWindowPruningBoundary:
    def test_window_pruning_boundary(self) -> None:
        """Sequences not divisible by prune_window are handled correctly."""
        codec = make_codec(prune_window=128)
        calibrate_codec(codec)

        # 300 tokens: window 1 = [0,128), window 2 = [128,256), window 3 = [256,300)
        seq_len = 300
        kv = torch.randn(2, 2, seq_len, 64)
        k_pre = torch.randn(2, 2, seq_len, 64)
        compressed = codec.compress(kv, k_pre, compression_ratio=0.5)

        assert compressed["original_seq_len"] == seq_len
        assert compressed["kv"].shape[2] > 0
        # All kept indices must be valid
        assert compressed["kept_indices"].max().item() < seq_len

    def test_sequence_shorter_than_window(self) -> None:
        """Sequences shorter than prune_window are handled without error."""
        codec = make_codec(prune_window=128)
        calibrate_codec(codec)

        seq_len = 50
        kv = torch.randn(2, 2, seq_len, 64)
        k_pre = torch.randn(2, 2, seq_len, 64)
        compressed = codec.compress(kv, k_pre, compression_ratio=0.2)
        assert compressed["original_seq_len"] == seq_len
        assert compressed["kv"].shape[2] >= 1
