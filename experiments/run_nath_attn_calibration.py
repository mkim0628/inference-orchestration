"""NAtHDDROffloadingScheduler calibration: measure cumulative attention score distribution.

Runs calibration over synthetic sequences to determine per-dataset tier_boundaries
[p1, p2, p3] that enforce max_eviction_ratio <= 3%.

Output: configs/nath_tier_boundaries.yaml

Usage:
  python experiments/run_nath_attn_calibration.py \\
    --n_sequences 100 --n_layers 12 --n_heads 8 \\
    --max_eviction_ratio 0.03 \\
    --output configs/nath_tier_boundaries.yaml --seed 42
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
import yaml

from src.scheduler.nath_ddr_offloading import (
    NAtHDDROffloadingConfig,
    NAtHDDROffloadingScheduler,
)


def simulate_attention_scores(
    n_sequences: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Generate synthetic per-token cumulative attention scores for calibration.

    Simulates multi-step decoding attention accumulation via EMA.
    Returns list of {token_key: score} dicts, one per sequence.
    """
    torch.manual_seed(seed)
    scheduler_cfg = NAtHDDROffloadingConfig(seed=seed)
    scheduler = NAtHDDROffloadingScheduler(scheduler_cfg)

    all_scores: List[Dict[str, float]] = []

    for seq_idx in range(n_sequences):
        seq_len = int(torch.randint(32, max_seq_len + 1, (1,)).item())
        token_keys = [f"seq{seq_idx}:tok{i}" for i in range(seq_len)]

        # Simulate n_decode_steps steps of attention score accumulation
        n_decode_steps = int(torch.randint(10, 50, (1,)).item())

        # Attention scores: Pareto-distributed (few tokens get most attention)
        for _ in range(n_decode_steps):
            # Each token gets a random attention score (heavy-tailed)
            raw_scores = torch.zeros(seq_len).exponential_(lambd=1.0)
            raw_scores = raw_scores / raw_scores.sum()
            for i, k in enumerate(token_keys):
                scheduler.update_attention_score(k, float(raw_scores[i]))

        seq_scores = {k: scheduler._attn_score_ema.get(k, 0.0) for k in token_keys}
        all_scores.append(seq_scores)

    return all_scores


def run_calibration(
    n_sequences: int = 100,
    n_layers: int = 12,
    n_heads: int = 8,
    max_seq_len: int = 512,
    max_eviction_ratio: float = 0.03,
    output_path: str = "configs/nath_tier_boundaries.yaml",
    seed: int = 42,
) -> Dict[str, float]:
    """Calibrate NAtH tier boundaries from attention score distribution.

    Algorithm:
      1. Collect all per-token cumulative attention scores from calibration runs.
      2. Fit p1, p2, p3 such that:
           - Tier 4 fraction  ≈ max_eviction_ratio (bottom 3%)
           - Tier 1 fraction  ≈ 30% (top-performing tokens on HBM)
           - Tier 2 fraction  ≈ 40% (DDR FP16 prefetch tier)
           - Tier 3 fraction  ≈ 27% (DDR INT8 compressed tier)

    Returns:
        {"p1": float, "p2": float, "p3": float}
    """
    torch.manual_seed(seed)
    seq_score_dicts = simulate_attention_scores(
        n_sequences=n_sequences,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        seed=seed,
    )

    # Aggregate all scores into a flat tensor
    all_scores: List[float] = []
    for seq_scores in seq_score_dicts:
        all_scores.extend(seq_scores.values())

    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    n_total = len(scores_tensor)

    # p3 = fraction NOT permanently evicted = 1 - max_eviction_ratio
    p3 = 1.0 - max_eviction_ratio  # e.g. 0.97 → bottom 3% are Tier 4

    # p1: top 30% on HBM — empirically chosen for balance; adjust from data
    # Use the 70th percentile as the HBM boundary (top 30% stay in HBM)
    p1 = 0.30

    # p2: between p1 and p3 — DDR FP16 tier; defaults to top 70% boundary
    p2 = 0.70

    # Verify: fraction below threshold for Tier 4 should match max_eviction_ratio
    thr3 = float(scores_tensor.quantile(1.0 - p3))
    actual_tier4_fraction = float((scores_tensor < thr3).float().mean())

    # Adjust p3 if the actual Tier-4 fraction deviates significantly
    if abs(actual_tier4_fraction - max_eviction_ratio) > 0.005:
        # Binary search for p3 that gives the desired eviction ratio
        lo, hi = 0.90, 0.999
        for _ in range(30):
            mid = (lo + hi) / 2.0
            thr = float(scores_tensor.quantile(1.0 - mid))
            frac = float((scores_tensor < thr).float().mean())
            if frac > max_eviction_ratio:
                hi = mid
            else:
                lo = mid
        p3 = (lo + hi) / 2.0

    result = {"p1": round(p1, 4), "p2": round(p2, 4), "p3": round(p3, 4)}

    output_data = {
        "nath_tier_boundaries": result,
        "calibration": {
            "n_sequences": n_sequences,
            "n_tokens_total": n_total,
            "max_eviction_ratio": max_eviction_ratio,
            "seed": seed,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False)

    print(f"[run_nath_attn_calibration] Calibrated tier boundaries: {result}")
    print(f"  Saved to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate NAtH DDR offloading tier boundaries")
    parser.add_argument("--n_sequences", type=int, default=100)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_eviction_ratio", type=float, default=0.03)
    parser.add_argument("--output", type=str, default="configs/nath_tier_boundaries.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_calibration(
        n_sequences=args.n_sequences,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        max_eviction_ratio=args.max_eviction_ratio,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
