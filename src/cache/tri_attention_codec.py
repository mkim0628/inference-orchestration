"""
TriAttentionCodec — Activity C.

Pre-RoPE trigonometric importance estimation for position-stable KV pruning.

Core insight (TriAttention, arXiv 2604.04921): in post-RoPE space, query
vectors are rotated differently at each position, making importance estimates
unstable. In pre-RoPE space, Q/K vectors cluster around a non-zero mean
(mu_k), and the distance from that centroid can be expressed as a
trigonometric Fourier series — yielding position-stable importance scores.

NOT a CacheStore subclass — this is a pure codec invoked by CacheStore
implementations (e.g. QueryCentricTriAttentionCache).
"""

import os
from typing import Any, Dict, List, Optional

import torch

from src.cache.base import CacheStore


class TriAttentionCodec:
    """
    Pre-RoPE trigonometric series importance estimation + windowed pruning.

    Calibration (one-time):
        1. Estimate mu_k: per-(layer, head) mean of K vectors.
        2. Fit Fourier coefficients a_m by least-squares regression:
           importance_approx(k_i) = sum_m a_m * (sin(m*d_i) + cos(m*d_i))
           where d_i = ||k_i - mu_k||.

    Compression:
        For each 128-token window, keep the top compression_ratio fraction
        by aggregated per-token importance (mean over layers and heads).
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        compression_ratio: float = 0.10,
        series_terms: int = 8,
        prune_window: int = 128,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.series_terms = series_terms
        self.prune_window = prune_window

        self.mu_k: Optional[torch.Tensor] = None   # [n_layers, n_heads, head_dim]
        self.a_m: Optional[torch.Tensor] = None    # [n_layers, n_heads, series_terms]

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        calibration_kvs: List[torch.Tensor],
        save_path: Optional[str] = None,
    ) -> None:
        """Estimate mu_k and a_m from calibration KV tensors.

        Args:
            calibration_kvs: List of tensors [layers, heads, seq_len, head_dim].
                             At least 10 requests recommended.
            save_path: Optional path to save calibration as a .pt file.
        """
        # Concatenate all calibration keys along the seq dimension
        all_keys = torch.cat(calibration_kvs, dim=2)  # [L, H, T, D]
        # Cast to float32 for numerical stability during calibration
        all_keys = all_keys.float()

        self.mu_k = all_keys.mean(dim=2)  # [L, H, D]
        distances = (all_keys - self.mu_k.unsqueeze(2)).norm(dim=-1)  # [L, H, T]
        self.a_m = self._fit_fourier_coefficients(distances, all_keys)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            torch.save({"mu_k": self.mu_k, "a_m": self.a_m}, save_path)

    def load_calibration(self, load_path: str) -> None:
        """Load previously saved calibration from disk.

        Args:
            load_path: Path to a .pt file produced by calibrate().
        """
        ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
        self.mu_k = ckpt["mu_k"]
        self.a_m = ckpt["a_m"]

    # ------------------------------------------------------------------ #
    # Core algorithm                                                       #
    # ------------------------------------------------------------------ #

    def estimate_importance(self, keys_pre_rope: torch.Tensor) -> torch.Tensor:
        """Compute per-token importance via trigonometric Fourier series.

        importance_i = |sum_{m=1}^{M} a_m * (sin(m*d_i) + cos(m*d_i))|
        where d_i = ||k_i - mu_k||.

        Args:
            keys_pre_rope: Pre-RoPE K tensor [layers, heads, seq_len, head_dim].

        Returns:
            Importance scores [layers, heads, seq_len].
        """
        if self.mu_k is None or self.a_m is None:
            raise RuntimeError("calibrate() or load_calibration() must be called first")

        device = keys_pre_rope.device
        mu_k = self.mu_k.to(device=device, dtype=keys_pre_rope.dtype)
        a_m = self.a_m.to(device=device, dtype=keys_pre_rope.dtype)

        # Distance from centroid: [L, H, S]
        diff = keys_pre_rope - mu_k.unsqueeze(2)
        d = diff.norm(dim=-1)

        # Fourier series accumulation
        importance = torch.zeros_like(d)
        for m in range(1, self.series_terms + 1):
            m_d = m * d
            importance = importance + a_m[:, :, m - 1].unsqueeze(2) * (
                torch.sin(m_d) + torch.cos(m_d)
            )

        return importance.abs()

    def compress(
        self,
        kv_tensor: torch.Tensor,
        keys_pre_rope: torch.Tensor,
        compression_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Prune low-importance tokens within 128-token windows.

        Args:
            kv_tensor: Full KV tensor [layers, heads, seq_len, head_dim].
            keys_pre_rope: Pre-RoPE K tensor (same shape). MUST be pre-RoPE.
            compression_ratio: Fraction of tokens to keep (overrides instance
                               default when provided).

        Returns:
            Dict with keys: "kv", "kept_indices", "original_seq_len",
            "compression_ratio".
        """
        ratio = compression_ratio if compression_ratio is not None else self.compression_ratio
        seq_len = kv_tensor.shape[2]

        importance = self.estimate_importance(keys_pre_rope)  # [L, H, S]
        token_importance = importance.mean(dim=(0, 1))         # [S]

        kept_parts: List[torch.Tensor] = []
        for window_start in range(0, seq_len, self.prune_window):
            window_end = min(window_start + self.prune_window, seq_len)
            window_imp = token_importance[window_start:window_end]
            n_keep = max(1, int(len(window_imp) * ratio))
            top_local = window_imp.topk(n_keep).indices + window_start
            kept_parts.append(top_local)

        kept_indices = torch.cat(kept_parts).sort().values
        compressed_kv = kv_tensor[:, :, kept_indices, :]

        return {
            "kv": compressed_kv,
            "kept_indices": kept_indices,
            "original_seq_len": seq_len,
            "compression_ratio": ratio,
        }

    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct full-length KV with zeros at pruned positions.

        Note: Information lost during compression is not recoverable; this
        produces an approximate reconstruction suitable for cache lookup but
        not for exact KV reuse.

        Args:
            compressed: Dict returned by compress().

        Returns:
            Reconstructed tensor [layers, heads, original_seq_len, head_dim].
        """
        kv_c = compressed["kv"]
        kept_indices = compressed["kept_indices"]
        original_len = compressed["original_seq_len"]
        layers, heads, _, dim = kv_c.shape
        reconstructed = torch.zeros(
            layers, heads, original_len, dim,
            dtype=kv_c.dtype, device=kv_c.device,
        )
        reconstructed[:, :, kept_indices, :] = kv_c
        return reconstructed

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fit_fourier_coefficients(
        self,
        distances: torch.Tensor,
        all_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Least-squares fit of Fourier coefficients.

        Minimises ||X * a - y|| where:
          y = per-token K-vector norm (attention-norm proxy)
          X[:, m] = sin(m*d) + cos(m*d)

        Args:
            distances: ||k_i - mu_k|| tensor [layers, heads, total_seq].
            all_keys: Raw K vectors [layers, heads, total_seq, head_dim].

        Returns:
            Coefficients [layers, heads, series_terms].
        """
        target_norms = all_keys.norm(dim=-1)  # [L, H, T]
        layers, heads, total_seq = distances.shape
        a_m = torch.zeros(layers, heads, self.series_terms, dtype=distances.dtype)

        for li in range(layers):
            for hi in range(heads):
                dist = distances[li, hi]  # [T]
                y = target_norms[li, hi]  # [T]

                # Design matrix: [T, series_terms]
                X = torch.stack(
                    [torch.sin(m * dist) + torch.cos(m * dist)
                     for m in range(1, self.series_terms + 1)],
                    dim=1,
                )

                # Least-squares solution; fall back to zeros on failure
                try:
                    result = torch.linalg.lstsq(X, y.unsqueeze(1))
                    coeff = result.solution.squeeze(1)
                    # lstsq may return a solution shorter than series_terms
                    # if the system is underdetermined; pad if necessary
                    if coeff.shape[0] < self.series_terms:
                        pad = torch.zeros(
                            self.series_terms - coeff.shape[0],
                            dtype=coeff.dtype,
                        )
                        coeff = torch.cat([coeff, pad])
                    a_m[li, hi] = coeff[: self.series_terms]
                except Exception:
                    # Graceful fallback: uniform unit coefficients
                    a_m[li, hi] = torch.ones(self.series_terms, dtype=distances.dtype)

        return a_m
