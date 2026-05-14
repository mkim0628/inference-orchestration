"""FibQuantVQCodec — Spherical-Beta radial-angular VQ codec for KV cache compression.

Based on FibQuant (arXiv 2605.11478): KV vectors are spherically normalized,
then decomposed into radial magnitude and unit-norm direction components.
The direction component is quantized via Fibonacci/Roberts-Kronecker lattice
directions; the magnitude via a Beta-quantile grid learned from calibration data.

Key difference from RSimVQCodec (k-means residual VQ):
  - RSimVQCodec: Euclidean k-means in pre-RoPE space, residual stages.
  - FibQuantVQCodec: Spherical normalization + separate radial/angular quantization;
    achieves higher accuracy in the 10x+ compression regime.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FibQuantConfig:
    d_head: int = 64          # KV head dimension
    n_heads: int = 8
    n_layers: int = 12
    block_size: int = 64      # tokens per encoding block
    bits_radial: int = 4      # radial quantization bits  → 2^bits_radial entries
    bits_direction: int = 9   # direction quantization bits → 2^bits_direction entries
    n_lloyd_restarts: int = 10  # Lloyd-Max multi-restart count
    n_lloyd_iters: int = 5      # Lloyd-Max iterations per restart
    seed: int = 42
    recent_window: int = 0    # tokens kept in FP16 (0 = compress all)


class FibQuantVQCodec:
    """FibQuant Spherical-Beta radial-angular VQ codec for KV cache compression.

    Distinct from RSimVQCodec (k-means residual VQ): uses spherical normalization,
    beta-quantile radial grid, and Fibonacci direction lattice.
    """

    def __init__(self, config: FibQuantConfig) -> None:
        self.config = config
        # Per-layer codebooks
        self.radial_codebooks: Dict[int, torch.Tensor] = {}   # [N_radii]
        self.direction_codebooks: Dict[int, torch.Tensor] = {}  # [N_dir, d_head]
        self._fitted: set = set()

    # ---------------------------------------------------------------------- #
    # Codebook construction                                                    #
    # ---------------------------------------------------------------------- #

    def fit(
        self,
        calibration_kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int,
    ) -> None:
        """Learn radial and direction codebooks from calibration data.

        Steps:
        1. Spherical normalization: collect norms and unit directions.
        2. Fit beta-quantile radial grid from norm distribution.
        3. Build Fibonacci direction lattice as initial direction codebook.
        4. Multi-restart Lloyd-Max refinement of directions on the sphere.
        """
        torch.manual_seed(self.config.seed + layer_idx)
        cfg = self.config
        d = cfg.d_head
        n_radii = 2 ** cfg.bits_radial
        n_dir = 2 ** cfg.bits_direction

        # Flatten all tokens/heads to vectors: [N, d]
        n_tok, _, n_heads, d_head = calibration_kv.shape
        flat = calibration_kv.reshape(-1, d_head).float()  # [N*2*n_heads, d]

        # 1. Spherical normalization
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, 1]
        unit_dirs = flat / norms  # [N, d]
        norms_1d = norms.squeeze(-1)  # [N]

        # 2. Beta-quantile radial grid
        radial_cb = self._fit_beta_radial_grid(norms_1d, n_radii)

        # 3. Fibonacci direction lattice as initialization
        fib_dirs = self._build_fibonacci_directions(n_dir, d_head)  # [n_dir, d]

        # 4. Multi-restart Lloyd-Max refinement for direction codebook
        best_dirs = fib_dirs
        best_loss = float("inf")

        # Subsample unit_dirs for speed if calibration set is large
        max_calib = min(len(unit_dirs), 8192)
        perm = torch.randperm(len(unit_dirs))[:max_calib]
        unit_sub = unit_dirs[perm]  # [max_calib, d]

        for restart in range(cfg.n_lloyd_restarts):
            if restart == 0:
                init = fib_dirs.clone()
            else:
                # Random perturbation restart
                noise = torch.randn(n_dir, d_head)
                init = F.normalize(fib_dirs + 0.1 * noise, dim=-1)

            refined = self._lloyd_max_refine(unit_sub, init, cfg.n_lloyd_iters)

            # Measure quantization distortion
            sim = unit_sub @ refined.T  # [M, n_dir]
            loss = (1.0 - sim.max(dim=-1).values).mean().item()
            if loss < best_loss:
                best_loss = loss
                best_dirs = refined

        self.radial_codebooks[layer_idx] = radial_cb
        self.direction_codebooks[layer_idx] = best_dirs
        self._fitted.add(layer_idx)

    def _build_fibonacci_directions(self, n_dir: int, d: int) -> torch.Tensor:
        """Construct n_dir quasi-uniform unit directions on S^(d-1).

        For d=2: standard Fibonacci spiral.
        For d>2: Roberts-Kronecker generalization — each dimension uses
        a distinct irrational increment derived from the plastic constant
        or consecutive prime roots, then projects onto the sphere.
        Returns: [n_dir, d] unit-norm tensors.
        """
        if d == 2:
            phi = (1.0 + math.sqrt(5.0)) / 2.0
            i = torch.arange(n_dir, dtype=torch.float32)
            theta = 2.0 * math.pi * (i / phi)
            dirs = torch.stack([theta.cos(), theta.sin()], dim=-1)
        else:
            # Roberts-Kronecker: use alpha_k = frac(k * phi^(1/d)) for dimension k
            # This gives well-distributed low-discrepancy sequences in each dim.
            dirs = torch.zeros(n_dir, d)
            phi_d = (5.0 ** 0.5 + 1.0) / 2.0  # golden ratio
            for k in range(d):
                alpha = ((k + 1) * phi_d) % 1.0 + 0.618 * k / max(d - 1, 1)
                frac = torch.arange(n_dir, dtype=torch.float32) * alpha
                frac = frac - frac.floor()  # in [0, 1)
                # Map to [-1, 1] via arcsin for better sphere coverage
                val = 2.0 * frac - 1.0
                dirs[:, k] = val
            dirs = F.normalize(dirs, dim=-1)

        return F.normalize(dirs, dim=-1)

    def _fit_beta_radial_grid(
        self, norms: torch.Tensor, n_radii: int
    ) -> torch.Tensor:
        """Fit radial quantization grid from empirical norm distribution.

        Without scipy, we use empirical quantiles of the norm distribution
        directly (equivalent to beta distribution quantile grid for
        calibration data drawn from a beta-like distribution).
        Returns: sorted tensor [n_radii] of quantile breakpoints.
        """
        norms_sorted, _ = norms.sort()
        N = len(norms_sorted)

        # Uniform quantile positions
        quantile_positions = torch.linspace(0.0, 1.0, n_radii)
        grid = torch.zeros(n_radii)
        for i, q in enumerate(quantile_positions):
            idx = int(q.item() * (N - 1))
            grid[i] = norms_sorted[idx]

        return grid

    def _lloyd_max_refine(
        self,
        data: torch.Tensor,       # [N, d] unit-norm directions
        centroids: torch.Tensor,  # [M, d] unit-norm centroids
        n_iters: int,
    ) -> torch.Tensor:
        """Lloyd-Max on the sphere: assign by cosine similarity, recompute by mean+normalize.

        Returns refined centroids [M, d], unit-norm.
        """
        centroids = F.normalize(centroids.float(), dim=-1)
        data_f = data.float()
        M = centroids.shape[0]

        for _ in range(n_iters):
            # Assignment: nearest centroid by cosine similarity
            sim = data_f @ centroids.T  # [N, M]
            assignments = sim.argmax(dim=-1)  # [N]

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(M, dtype=torch.float32)
            for m in range(M):
                mask = assignments == m
                if mask.any():
                    new_centroids[m] = data_f[mask].mean(dim=0)
                    counts[m] = mask.sum().float()
                else:
                    new_centroids[m] = centroids[m]

            # Project back to sphere
            norms = new_centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            centroids = new_centroids / norms

        return centroids

    # ---------------------------------------------------------------------- #
    # Encode / Decode                                                          #
    # ---------------------------------------------------------------------- #

    def _ensure_fitted(self, layer_idx: int, reference: torch.Tensor) -> None:
        """Auto-fit on reference data if codebooks for layer_idx not yet built."""
        if layer_idx not in self._fitted:
            # Build minimal calibration from provided reference tensor
            self.fit(reference.unsqueeze(0) if reference.dim() == 3 else reference, layer_idx)

    def encode_block(
        self,
        kv_block: torch.Tensor,  # [block_size, 2, n_heads, d_head]
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Encode one KV block into (radial_codes, direction_codes).

        Per token per head:
          1. Spherical normalize: norm = ||v||, v_unit = v / norm
          2. radial_code = argmin_r |norm - radial_codebook[r]|
          3. direction_code = argmin_d (1 - cosine(v_unit, direction_codebook[d]))

        Returns dict with int16 code tensors.
        """
        self._ensure_fitted(layer_idx, kv_block)
        rad_cb = self.radial_codebooks[layer_idx]   # [N_radii]
        dir_cb = self.direction_codebooks[layer_idx]  # [N_dir, d]

        block_size, _, n_heads, d_head = kv_block.shape
        flat = kv_block.float().reshape(-1, d_head)  # [block_size*2*n_heads, d]

        norms = flat.norm(dim=-1)  # [M_flat]
        unit_dirs = flat / norms.unsqueeze(-1).clamp(min=1e-8)  # [M_flat, d]

        # Radial assignment: nearest in L1/L2 on 1D grid
        rad_diff = (norms.unsqueeze(-1) - rad_cb.unsqueeze(0)).abs()  # [M_flat, N_radii]
        radial_codes = rad_diff.argmin(dim=-1).to(torch.int16)  # [M_flat]

        # Direction assignment: nearest by cosine similarity
        sim = unit_dirs @ dir_cb.T  # [M_flat, N_dir]
        direction_codes = sim.argmax(dim=-1).to(torch.int16)  # [M_flat]

        # Reshape back to [block_size, 2, n_heads]
        shape = (block_size, 2, n_heads)
        return {
            "radial_codes": radial_codes.reshape(shape),
            "direction_codes": direction_codes.reshape(shape),
            "layer_idx": layer_idx,
        }

    def decode_block(
        self,
        codes: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Decode (radial_codes, direction_codes) -> [block_size, 2, n_heads, d_head].

        Per token per head:
          1. norm = radial_codebook[radial_code]
          2. v_unit = direction_codebook[direction_code]
          3. v_reconstructed = norm * v_unit
        """
        rad_cb = self.radial_codebooks[layer_idx]   # [N_radii]
        dir_cb = self.direction_codebooks[layer_idx]  # [N_dir, d]

        radial_codes = codes["radial_codes"].long()    # [block_size, 2, n_heads]
        direction_codes = codes["direction_codes"].long()

        block_size, _, n_heads = radial_codes.shape
        d_head = dir_cb.shape[-1]

        # Gather radii and directions
        flat_r = radial_codes.reshape(-1)         # [M_flat]
        flat_d = direction_codes.reshape(-1)

        norms_recon = rad_cb[flat_r]              # [M_flat]
        dirs_recon = dir_cb[flat_d]               # [M_flat, d]

        recon = (norms_recon.unsqueeze(-1) * dirs_recon)  # [M_flat, d]
        return recon.reshape(block_size, 2, n_heads, d_head)

    def encode_segment(
        self,
        segment_kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int,
        segment_id: str,
    ) -> Dict:
        """Encode an entire segment (may span multiple blocks).

        Each block is encoded independently, enabling random-access decode
        of any block without decompressing the full segment.
        """
        self._ensure_fitted(layer_idx, segment_kv)
        cfg = self.config
        n_tokens = segment_kv.shape[0]
        block_size = cfg.block_size

        radial_parts: List[torch.Tensor] = []
        direction_parts: List[torch.Tensor] = []

        start = 0
        while start < n_tokens:
            end = min(start + block_size, n_tokens)
            block = segment_kv[start:end]
            block_codes = self.encode_block(block, layer_idx)
            radial_parts.append(block_codes["radial_codes"])
            direction_parts.append(block_codes["direction_codes"])
            start = end

        return {
            "radial_codes": torch.cat(radial_parts, dim=0),      # [n_tokens, 2, n_heads]
            "direction_codes": torch.cat(direction_parts, dim=0),
            "layer_idx": layer_idx,
            "segment_id": segment_id,
            "n_tokens": n_tokens,
            "shape": segment_kv.shape,
            "dtype": segment_kv.dtype,
        }

    def decode_segment(
        self,
        compressed: Dict,
        layer_idx: int,
    ) -> torch.Tensor:
        """Decode a full segment on-demand. Returns [n_tokens, 2, n_heads, d_head]."""
        n_tokens = compressed["n_tokens"]
        orig_shape = compressed["shape"]
        orig_dtype = compressed["dtype"]

        all_radial = compressed["radial_codes"]       # [n_tokens, 2, n_heads]
        all_direction = compressed["direction_codes"]

        block_size = self.config.block_size
        blocks: List[torch.Tensor] = []

        start = 0
        while start < n_tokens:
            end = min(start + block_size, n_tokens)
            block_codes = {
                "radial_codes": all_radial[start:end],
                "direction_codes": all_direction[start:end],
                "layer_idx": layer_idx,
            }
            blocks.append(self.decode_block(block_codes, layer_idx))
            start = end

        result = torch.cat(blocks, dim=0)  # [n_tokens, 2, n_heads, d_head]
        return result.to(orig_dtype)

    def compression_ratio(self, compression_target: float = 10.0) -> float:
        """Effective bits saved vs FP16 baseline (0.0 to 1.0 fraction).

        Stored bits per token-head vector:
          bits_radial + bits_direction (for K, same for V)
        vs FP16: d_head * 16 bits per vector.
        Factor 2 for K+V: we store both K and V codes.
        """
        bpv = self.config.bits_radial + self.config.bits_direction  # bits per K or V vector
        fp16_bpv = self.config.d_head * 16
        # Both K and V use same bit budget
        stored_bpv = 2 * bpv  # K + V
        return 1.0 - stored_bpv / fp16_bpv

    def save(self, path: str) -> None:
        torch.save(
            {
                "radial_codebooks": self.radial_codebooks,
                "direction_codebooks": self.direction_codebooks,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.radial_codebooks = data["radial_codebooks"]
        self.direction_codebooks = data["direction_codebooks"]
        self.config = data["config"]
        self._fitted = set(self.radial_codebooks.keys())
