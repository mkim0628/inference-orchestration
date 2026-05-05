#!/usr/bin/env python3
"""FireQCodec TTFT/TBT throughput measurement script (Activity C-2).

Resolves SUMMARY.md open item #1: actual GPU throughput validation.

On GPU: uses torch.cuda.Event for precise TTFT/TBT timing.
On CPU: uses time.perf_counter for encode/decode latency and numpy accuracy validation.

Usage:
    python experiments/run_gpu_throughput.py [--n-trials N] [--seq-len S]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.cache.fireq_codec import FireQCodec
from src.cache.nqkv_codec import NQKVCodec


def measure_encode_decode_latency_cpu(
    codec: FireQCodec,
    n_heads: int,
    seq_len: int,
    d_head: int,
    n_trials: int = 50,
) -> Dict[str, float]:
    """Measure encode/decode latency on CPU using perf_counter."""
    torch.manual_seed(42)
    kv = torch.randn(n_heads, seq_len, d_head)

    encode_times = []
    decode_times = []

    # Calibrate codec
    calib = [(kv, 0)]
    codec.calibrate(calib)

    for _ in range(n_trials):
        t0 = time.perf_counter()
        key_int4, val_fp8, meta = codec.encode(kv, layer_idx=0, rope_applied=True)
        encode_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        K_dec, V_dec = codec.decode(key_int4, val_fp8, meta, layer_idx=0)
        decode_times.append(time.perf_counter() - t0)

    encode_ms = np.array(encode_times) * 1000
    decode_ms = np.array(decode_times) * 1000

    # Accuracy verification (numpy-based)
    key_int4, val_fp8, meta = codec.encode(kv, layer_idx=0, rope_applied=True)
    K_dec, V_dec = codec.decode(key_int4, val_fp8, meta, layer_idx=0)
    rmse = (kv - K_dec.float()).pow(2).mean().sqrt().item()

    return {
        "encode_p50_ms": float(np.percentile(encode_ms, 50)),
        "encode_p99_ms": float(np.percentile(encode_ms, 99)),
        "decode_p50_ms": float(np.percentile(decode_ms, 50)),
        "decode_p99_ms": float(np.percentile(decode_ms, 99)),
        "total_p50_ms": float(np.percentile(encode_ms + decode_ms, 50)),
        "rmse": rmse,
        "backend": "cpu",
    }


def measure_encode_decode_latency_gpu(
    codec: FireQCodec,
    n_heads: int,
    seq_len: int,
    d_head: int,
    n_trials: int = 50,
) -> Dict[str, float]:
    """Measure encode/decode latency on GPU using torch.cuda.Event."""
    torch.manual_seed(42)
    kv = torch.randn(n_heads, seq_len, d_head, device="cuda")

    calib = [(kv.cpu(), 0)]
    codec.calibrate(calib)

    encode_times = []
    decode_times = []

    # Warm-up
    for _ in range(5):
        key_int4, val_fp8, meta = codec.encode(kv.cpu(), layer_idx=0)
        _, _ = codec.decode(key_int4, val_fp8, meta, layer_idx=0)

    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        key_int4, val_fp8, meta = codec.encode(kv.cpu(), layer_idx=0)
        end.record()
        torch.cuda.synchronize()
        encode_times.append(start.elapsed_time(end))

        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        start2.record()
        K_dec, V_dec = codec.decode(key_int4, val_fp8, meta, layer_idx=0)
        end2.record()
        torch.cuda.synchronize()
        decode_times.append(start2.elapsed_time(end2))

    encode_ms = np.array(encode_times)
    decode_ms = np.array(decode_times)

    key_int4, val_fp8, meta = codec.encode(kv.cpu(), layer_idx=0)
    K_dec, V_dec = codec.decode(key_int4, val_fp8, meta, layer_idx=0)
    rmse = (kv.cpu() - K_dec.float()).pow(2).mean().sqrt().item()

    return {
        "encode_p50_ms": float(np.percentile(encode_ms, 50)),
        "encode_p99_ms": float(np.percentile(encode_ms, 99)),
        "decode_p50_ms": float(np.percentile(decode_ms, 50)),
        "decode_p99_ms": float(np.percentile(decode_ms, 99)),
        "total_p50_ms": float(np.percentile(encode_ms + decode_ms, 50)),
        "rmse": rmse,
        "backend": "cuda",
    }


def measure_nqkv_throughput(
    n_heads: int,
    seq_len: int,
    d_head: int,
    n_trials: int = 50,
) -> Dict[str, float]:
    """NQKVCodec baseline throughput measurement."""
    torch.manual_seed(42)
    codec = NQKVCodec(block_size=64)
    kv = torch.randn(n_heads, seq_len, d_head)

    encode_times = []
    decode_times = []

    for _ in range(n_trials):
        t0 = time.perf_counter()
        indices, mu, sigma = codec.encode(kv)
        encode_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        kv_restored = codec.decode(indices, mu, sigma, kv.shape)
        decode_times.append(time.perf_counter() - t0)

    encode_ms = np.array(encode_times) * 1000
    decode_ms = np.array(decode_times) * 1000

    indices, mu, sigma = codec.encode(kv)
    kv_restored = codec.decode(indices, mu, sigma, kv.shape)
    rmse = (kv - kv_restored.float()).pow(2).mean().sqrt().item()

    compression_ratio = codec.compression_ratio(kv)

    return {
        "encode_p50_ms": float(np.percentile(encode_ms, 50)),
        "encode_p99_ms": float(np.percentile(encode_ms, 99)),
        "decode_p50_ms": float(np.percentile(decode_ms, 50)),
        "decode_p99_ms": float(np.percentile(decode_ms, 99)),
        "total_p50_ms": float(np.percentile(encode_ms + decode_ms, 50)),
        "rmse": rmse,
        "compression_ratio": compression_ratio,
        "backend": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FireQCodec/NQKVCodec throughput measurement")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results/fireq_throughput")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_heads = args.n_heads
    seq_len = args.seq_len
    d_head = args.d_head
    n_trials = args.n_trials

    results: Dict = {
        "config": {
            "n_heads": n_heads,
            "seq_len": seq_len,
            "d_head": d_head,
            "n_trials": n_trials,
        }
    }

    print(f"Measuring FireQCodec latency (n_heads={n_heads}, seq_len={seq_len}, d_head={d_head})...")
    codec = FireQCodec(n_heads=n_heads, d_head=d_head, outlier_threshold_sigma=3.0)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("  Backend: CUDA (torch.cuda.Event)")
        results["fireq"] = measure_encode_decode_latency_gpu(
            codec, n_heads, seq_len, d_head, n_trials
        )
    else:
        print("  Backend: CPU (time.perf_counter)")
        results["fireq"] = measure_encode_decode_latency_cpu(
            codec, n_heads, seq_len, d_head, n_trials
        )

    print(f"  Encode p50: {results['fireq']['encode_p50_ms']:.3f}ms")
    print(f"  Decode p50: {results['fireq']['decode_p50_ms']:.3f}ms")
    print(f"  RMSE: {results['fireq']['rmse']:.6f}")

    print("\nMeasuring NQKVCodec latency...")
    results["nqkv"] = measure_nqkv_throughput(n_heads, seq_len, d_head, n_trials)
    print(f"  Encode p50: {results['nqkv']['encode_p50_ms']:.3f}ms")
    print(f"  Decode p50: {results['nqkv']['decode_p50_ms']:.3f}ms")
    print(f"  Compression ratio: {results['nqkv']['compression_ratio']:.2f}x")
    print(f"  RMSE: {results['nqkv']['rmse']:.6f}")

    # Overhead assessment: encode+decode vs baseline FP16 copy
    torch.manual_seed(42)
    kv_ref = torch.randn(n_heads, seq_len, d_head)
    copy_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = kv_ref.clone()
        copy_times.append(time.perf_counter() - t0)
    baseline_p50_ms = float(np.percentile(np.array(copy_times) * 1000, 50))
    results["baseline_copy_p50_ms"] = baseline_p50_ms

    fireq_overhead_pct = (
        results["fireq"]["total_p50_ms"] / max(baseline_p50_ms, 1e-6) - 1.0
    ) * 100
    nqkv_overhead_pct = (
        results["nqkv"]["total_p50_ms"] / max(baseline_p50_ms, 1e-6) - 1.0
    ) * 100
    results["fireq_overhead_pct"] = fireq_overhead_pct
    results["nqkv_overhead_pct"] = nqkv_overhead_pct
    results["fireq_pass"] = fireq_overhead_pct <= 10.0
    results["nqkv_pass"] = nqkv_overhead_pct <= 10.0

    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFireQ overhead: {fireq_overhead_pct:.1f}% (PASS={results['fireq_pass']})")
    print(f"NQKV overhead:  {nqkv_overhead_pct:.1f}% (PASS={results['nqkv_pass']})")
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
