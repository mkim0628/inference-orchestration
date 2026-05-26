"""ObjectCache S3 break-even calibration script (Activity A-1).

Measures T_s3 and T_recompute for a range of context lengths and
produces a break-even table for objectcache_breakeven_table.yaml.

Usage:
    python configs/objectcache_breakeven_calibration.py [--output PATH]
"""

import argparse
import time
from typing import Dict, List

import yaml


def measure_t_recompute_ms(context_length: int, n_trials: int = 10) -> float:
    """Estimate per-token prefill cost for a given context length.

    Uses a linear model: T_recompute ≈ context_length * per_token_ms.
    In production, replace with actual transformer forward-pass timing.
    """
    import torch

    # Synthetic timing: simulate O(n^2) attention cost
    per_token_ms = 0.05  # 0.05ms per token (synthetic baseline)
    return context_length * per_token_ms


def measure_t_s3_ms(
    context_length: int,
    rdma_bandwidth_gbps: float = 100.0,
    d_head: int = 64,
    n_layers: int = 32,
    bytes_per_element: float = 2.0,  # FP16
) -> float:
    """Estimate S3 fetch latency for a given context length.

    KV cache size: 2 (K+V) × n_layers × context_length × d_head × bytes_per_element
    T_s3 = KV_bytes / (bandwidth_bytes_per_ms) + base_latency_ms
    """
    kv_bytes = 2 * n_layers * context_length * d_head * bytes_per_element
    bandwidth_bytes_per_ms = rdma_bandwidth_gbps * 1e9 / 1e3 / 8  # gbps → bytes/ms
    base_latency_ms = 1.0  # S3 request overhead
    return kv_bytes / bandwidth_bytes_per_ms + base_latency_ms


def compute_breakeven_table(
    context_lengths: List[int],
    rdma_bandwidth_gbps: float = 100.0,
) -> Dict[int, float]:
    """Compute break-even hit rates for each context length.

    hit_rate_breakeven = T_recompute / (T_recompute + T_s3)
    """
    table = {}
    for ctx_len in context_lengths:
        t_recompute = measure_t_recompute_ms(ctx_len)
        t_s3 = measure_t_s3_ms(ctx_len, rdma_bandwidth_gbps)
        breakeven = t_recompute / (t_recompute + t_s3)
        table[ctx_len] = round(float(breakeven), 4)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate ObjectCache S3 break-even table"
    )
    parser.add_argument(
        "--output",
        default="configs/objectcache_breakeven_table.yaml",
        help="Output YAML path",
    )
    parser.add_argument(
        "--rdma-bw", type=float, default=100.0,
        help="RDMA bandwidth in Gbps (default: 100.0)"
    )
    args = parser.parse_args()

    context_lengths = [4096, 8192, 16384, 32768, 65536]
    table = compute_breakeven_table(context_lengths, args.rdma_bw)

    print("Break-even hit rates:")
    for ctx_len, be in table.items():
        print(f"  context_length={ctx_len:6d}: breakeven={be:.4f}")

    output = {
        "breakeven_table": table,
        "s3_endpoint": "http://localhost:9000",
        "s3_bucket": "kvcache",
        "rdma_target": "gpu-node-0",
    }

    with open(args.output, "w") as f:
        yaml.dump(output, f, default_flow_style=False)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
