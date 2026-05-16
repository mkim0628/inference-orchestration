"""Train GlobalRetentionGateEvictionCodec retention gate (W_r) + shared projection (W_final).

LLM weights are frozen; only RetentionGate parameters are learned.
Training cost: 500-1000 samples, 3-5 epochs, < 0.5 GPU-hour.

Objective: MSE between attention output before and after eviction.
The gate learns to keep tokens whose absence causes maximum attention-output distortion.

Usage:
  python experiments/train_retention_gate.py \\
    --n_samples 500 --n_epochs 5 --lr 1e-3 \\
    --budget_ratio 0.3 --n_layers 12 --n_heads 8 --d_model 512 \\
    --output configs/retention_gate_weights.pt --seed 42
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import torch

from src.cache.global_retention_gate_eviction import (
    GlobalRetentionGateConfig,
    GlobalRetentionGateEvictionCodec,
)


def generate_calibration_data(
    n_samples: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    min_seq_len: int = 32,
    max_seq_len: int = 256,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Generate synthetic KV tensors for calibration.

    Each sample has shape [n_tokens, n_layers, n_heads, d_head].
    d_head = d_model // n_heads.
    """
    torch.manual_seed(seed)
    d_head = d_model // n_heads
    data = []
    for i in range(n_samples):
        n_tokens = torch.randint(min_seq_len, max_seq_len + 1, (1,)).item()
        # Simulate natural KV distribution: Gaussian with a few high-magnitude tokens
        kv = torch.randn(n_tokens, n_layers, n_heads, d_head)
        # Inject importance signal: ~10% of tokens get 3× magnitude (simulate key tokens)
        n_important = max(1, int(n_tokens * 0.1))
        important_idx = torch.randperm(n_tokens)[:n_important]
        kv[important_idx] *= 3.0
        data.append(kv)
    return data


def train(
    n_samples: int = 500,
    n_epochs: int = 5,
    lr: float = 1e-3,
    n_layers: int = 12,
    n_heads: int = 8,
    d_model: int = 512,
    budget_ratio: float = 0.3,
    output_path: str = "configs/retention_gate_weights.pt",
    seed: int = 42,
) -> Dict[str, float]:
    """Fine-tune retention gate on synthetic calibration data.

    Returns:
        {"final_loss": float, "n_samples": int, "training_time_sec": float}
    """
    torch.manual_seed(seed)
    t0 = time.time()

    config = GlobalRetentionGateConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        budget_ratio=budget_ratio,
        seed=seed,
    )
    codec = GlobalRetentionGateEvictionCodec(config)

    calibration_data = generate_calibration_data(
        n_samples=n_samples,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        seed=seed,
    )

    result = codec.train_retention_gate(
        calibration_data=calibration_data,
        n_epochs=n_epochs,
        lr=lr,
    )

    training_time = time.time() - t0
    result["training_time_sec"] = training_time

    # Persist trained weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    codec.save(output_path)
    print(f"[train_retention_gate] Saved to {output_path}")
    print(f"  final_loss={result['final_loss']:.6f}  "
          f"n_samples={result['n_samples']}  "
          f"time={training_time:.1f}s")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GlobalRetentionGate W_r + W_final")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--budget_ratio", type=float, default=0.3)
    parser.add_argument("--output", type=str, default="configs/retention_gate_weights.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        lr=args.lr,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        budget_ratio=args.budget_ratio,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
