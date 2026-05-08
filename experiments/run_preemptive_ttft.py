"""Preemptive vs non-preemptive TTFT p50/p99 comparison (Activity A).

Simulates burst-load and normal-load scenarios to measure:
- latency.ttft_p50_ms  (preemptive scheduler)
- latency.ttft_p99_ms  (preemptive scheduler)
- hit_rate.overall_hit_rate
- hit_rate.scheduler_hit_rate_delta  (preemptive - baseline)
- throughput.tokens_per_sec

Baseline reference: results/bc_2026-04-28/metrics.json
  ttft_p50 = 6.40 ms, ttft_p99 = 6.66 ms

Runs without real GPU — all KV tensors are synthetic CPU/GPU-agnostic.

Usage:
    python experiments/run_preemptive_ttft.py
"""

import json
import math
import os
import random
import statistics
import time
from typing import Dict, List, Tuple

import torch

from src.cache.static_dynamic_segment import StaticDynamicSegmentCache
from src.engine.runner import InferenceRequest, InferenceRunner
from src.scheduler.preemptive_kv_offload import PreemptiveKVOffloadScheduler

RESULTS_DIR = "results/2026-05-08"
SEED = 42

# Baseline TTFT values from bc_2026-04-28/metrics.json
BASELINE_TTFT_P50_MS = 6.404165686390339
BASELINE_TTFT_P99_MS = 6.66221616544081
BASELINE_HIT_RATE = 0.5


def _make_requests(
    n: int,
    base_tokens: int,
    shared_prefix_len: int,
    burst_factor: float,
    seed: int,
) -> List[InferenceRequest]:
    """Generate a mix of requests with shared prefix (cache-friendly) and burst tokens."""
    rng = random.Random(seed)
    shared = list(range(shared_prefix_len))
    requests: List[InferenceRequest] = []
    for i in range(n):
        # Burst adds extra tokens to simulate bursty demand
        extra = int(base_tokens * burst_factor * rng.random())
        unique_suffix = [rng.randint(1000, 5000) for _ in range(extra)]
        tokens = shared + unique_suffix
        requests.append(
            InferenceRequest(
                request_id=f"req_{i:04d}",
                token_ids=tokens,
                output_length=32,
                seed=seed + i,
            )
        )
    return requests


def _run_scenario(
    requests: List[InferenceRequest],
    cache_capacity_bytes: int,
    use_preemptive: bool,
    threshold_preempt: float,
    seed: int,
) -> Dict:
    """Run inference on a batch of requests, return metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    cache = StaticDynamicSegmentCache(
        capacity_bytes=cache_capacity_bytes,
        max_invalidation_range=2,
    )

    # Mark the first 4 chunks as static (shared prefix → non-contiguous reuse)
    # Pre-warm with a few static-segment keys
    for i in range(4):
        key = f"static_seg_{i}"
        cache.put(key, torch.randn(128, 64))
        cache.mark_static(key)

    scheduler = None
    if use_preemptive:
        scheduler = PreemptiveKVOffloadScheduler(
            cache=cache,
            cache_capacity_bytes=cache_capacity_bytes,
            threshold_preempt=threshold_preempt,
            consumption_rate_window=32,
            fairness_max_wait=10,
            preempt_compress=False,
        )

    runner = InferenceRunner(
        cache=cache,
        num_layers=4,
        hidden_dim=64,
        chunk_size=128,
        seed=seed,
        scheduler=scheduler,
    )

    # Simulate gradual fill of the cache to create realistic occupancy
    # Use a subset of requests as warm-up to build hit rate
    warmup = requests[: len(requests) // 4]
    for req in warmup:
        runner.run(req)

    # Reset metrics before measurement
    runner.hit_metrics.reset()
    runner.latency_metrics = type(runner.latency_metrics)()

    t_start = time.monotonic()
    results = runner.run_batch(requests)
    t_elapsed = time.monotonic() - t_start

    total_output_tokens = sum(r.output_tokens for r in results)
    tokens_per_sec = total_output_tokens / max(t_elapsed, 1e-6)

    hit_summary = runner.hit_metrics.summary()
    lat_summary = runner.latency_metrics.summary()

    return {
        "ttft_p50_ms": lat_summary["ttft_p50_ms"],
        "ttft_p99_ms": lat_summary["ttft_p99_ms"],
        "overall_hit_rate": hit_summary["overall_hit_rate"],
        "noncontiguous_fraction": hit_summary["noncontiguous_fraction"],
        "tokens_per_sec": tokens_per_sec,
        "num_requests": len(results),
        "elapsed_sec": t_elapsed,
    }


def run_preemptive_ttft_evaluation() -> Dict:
    """Compare preemptive vs non-preemptive scheduling under normal and burst loads."""
    torch.manual_seed(SEED)
    random.seed(SEED)

    # 4 GiB cache (but synthetic tensors are small — simulate via capacity parameter)
    # Use a tight capacity to trigger preemption behavior
    CACHE_CAPACITY = 64 * 1024 * 1024  # 64 MiB — tight enough for preemption
    N_REQUESTS = 60
    SHARED_PREFIX = 256  # tokens shared across requests

    print("=== Preemptive TTFT Evaluation ===")
    print(f"Requests: {N_REQUESTS}, shared_prefix: {SHARED_PREFIX} tokens")
    print(f"Baseline TTFT p50={BASELINE_TTFT_P50_MS:.3f}ms, p99={BASELINE_TTFT_P99_MS:.3f}ms")

    # Normal load: burst_factor=0.5 (moderate token variation)
    normal_requests = _make_requests(
        n=N_REQUESTS,
        base_tokens=SHARED_PREFIX,
        shared_prefix_len=SHARED_PREFIX,
        burst_factor=0.5,
        seed=SEED,
    )

    # Burst load: burst_factor=3.0 (heavy token surge, triggers preemption)
    burst_requests = _make_requests(
        n=N_REQUESTS,
        base_tokens=SHARED_PREFIX,
        shared_prefix_len=SHARED_PREFIX,
        burst_factor=3.0,
        seed=SEED + 1,
    )

    print("\n--- Normal load (burst_factor=0.5) ---")
    baseline_normal = _run_scenario(
        normal_requests, CACHE_CAPACITY, use_preemptive=False,
        threshold_preempt=0.85, seed=SEED
    )
    preemptive_normal = _run_scenario(
        normal_requests, CACHE_CAPACITY, use_preemptive=True,
        threshold_preempt=0.85, seed=SEED
    )
    print(f"  Baseline:   p50={baseline_normal['ttft_p50_ms']:.3f}ms, p99={baseline_normal['ttft_p99_ms']:.3f}ms, hit={baseline_normal['overall_hit_rate']:.3f}")
    print(f"  Preemptive: p50={preemptive_normal['ttft_p50_ms']:.3f}ms, p99={preemptive_normal['ttft_p99_ms']:.3f}ms, hit={preemptive_normal['overall_hit_rate']:.3f}")

    print("\n--- Burst load (burst_factor=3.0) ---")
    baseline_burst = _run_scenario(
        burst_requests, CACHE_CAPACITY, use_preemptive=False,
        threshold_preempt=0.85, seed=SEED + 1
    )
    preemptive_burst = _run_scenario(
        burst_requests, CACHE_CAPACITY, use_preemptive=True,
        threshold_preempt=0.85, seed=SEED + 1
    )
    print(f"  Baseline:   p50={baseline_burst['ttft_p50_ms']:.3f}ms, p99={baseline_burst['ttft_p99_ms']:.3f}ms, hit={baseline_burst['overall_hit_rate']:.3f}")
    print(f"  Preemptive: p50={preemptive_burst['ttft_p50_ms']:.3f}ms, p99={preemptive_burst['ttft_p99_ms']:.3f}ms, hit={preemptive_burst['overall_hit_rate']:.3f}")

    # Primary reported metrics use the normal-load preemptive result
    # (TTFT p50 overhead must stay ≤ +5% of baseline)
    ttft_p50 = preemptive_normal["ttft_p50_ms"]
    ttft_p99 = preemptive_normal["ttft_p99_ms"]
    hit_rate = preemptive_normal["overall_hit_rate"]
    tokens_per_sec = preemptive_normal["tokens_per_sec"]
    scheduler_hit_rate_delta = (
        preemptive_normal["overall_hit_rate"] - baseline_normal["overall_hit_rate"]
    )

    # TTFT p50 overhead check (must be ≤ +5% of baseline)
    ttft_p50_overhead_pct = (ttft_p50 - BASELINE_TTFT_P50_MS) / BASELINE_TTFT_P50_MS * 100
    ttft_p99_overhead_pct = (ttft_p99 - BASELINE_TTFT_P99_MS) / BASELINE_TTFT_P99_MS * 100

    # Burst p99 reduction (target ≥ -60%)
    burst_p99_delta_pct = (
        (preemptive_burst["ttft_p99_ms"] - BASELINE_TTFT_P99_MS) / BASELINE_TTFT_P99_MS * 100
    )

    print(f"\n  TTFT p50 overhead vs baseline: {ttft_p50_overhead_pct:+.1f}% (target ≤+5%)")
    print(f"  TTFT p99 overhead vs baseline: {ttft_p99_overhead_pct:+.1f}%")
    print(f"  Burst p99 delta vs baseline:   {burst_p99_delta_pct:+.1f}% (target ≤-60%)")
    print(f"  Scheduler hit rate delta:      {scheduler_hit_rate_delta:+.3f}")
    print(f"  Throughput (tokens/sec):       {tokens_per_sec:.1f}")

    return {
        "normal_load": {
            "baseline": baseline_normal,
            "preemptive": preemptive_normal,
        },
        "burst_load": {
            "baseline": baseline_burst,
            "preemptive": preemptive_burst,
        },
        "summary": {
            "ttft_p50_ms": ttft_p50,
            "ttft_p99_ms": ttft_p99,
            "overall_hit_rate": hit_rate,
            "scheduler_hit_rate_delta": scheduler_hit_rate_delta,
            "tokens_per_sec": tokens_per_sec,
            "ttft_p50_overhead_pct_vs_baseline": ttft_p50_overhead_pct,
            "ttft_p99_overhead_pct_vs_baseline": ttft_p99_overhead_pct,
            "burst_p99_delta_pct_vs_baseline": burst_p99_delta_pct,
            "noncontiguous_fraction": preemptive_normal["noncontiguous_fraction"],
        },
        "baseline_reference": {
            "source": "bc_2026-04-28/metrics.json",
            "ttft_p50_ms": BASELINE_TTFT_P50_MS,
            "ttft_p99_ms": BASELINE_TTFT_P99_MS,
            "overall_hit_rate": BASELINE_HIT_RATE,
        },
        "pass_criteria": {
            "ttft_p50_overhead_pass": ttft_p50_overhead_pct <= 5.0,
            "hit_rate_delta_pass": scheduler_hit_rate_delta >= -0.05,
        },
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results dir: {RESULTS_DIR}")

    ttft_metrics = run_preemptive_ttft_evaluation()

    # Load existing metrics.json and merge latency/throughput/hit_rate sections
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    summary = ttft_metrics["summary"]

    # Inject top-level latency, throughput, hit_rate sections
    existing["latency"] = {
        "ttft_p50_ms": summary["ttft_p50_ms"],
        "ttft_p99_ms": summary["ttft_p99_ms"],
        "ttft_p50_overhead_pct_vs_baseline": summary["ttft_p50_overhead_pct_vs_baseline"],
        "ttft_p99_overhead_pct_vs_baseline": summary["ttft_p99_overhead_pct_vs_baseline"],
        "burst_p99_delta_pct_vs_baseline": summary["burst_p99_delta_pct_vs_baseline"],
        "baseline_ttft_p50_ms": BASELINE_TTFT_P50_MS,
        "baseline_ttft_p99_ms": BASELINE_TTFT_P99_MS,
        "ttft_p50_pass": summary["ttft_p50_ms"] <= BASELINE_TTFT_P50_MS * 1.05,
    }
    existing["throughput"] = {
        "tokens_per_sec": summary["tokens_per_sec"],
        "baseline_tokens_per_sec": None,  # no baseline throughput recorded
    }
    existing["hit_rate"] = {
        "overall_hit_rate": summary["overall_hit_rate"],
        "scheduler_hit_rate_delta": summary["scheduler_hit_rate_delta"],
        "noncontiguous_fraction": summary["noncontiguous_fraction"],
        "noncontiguous_hit_rate_pass": summary["noncontiguous_fraction"] >= 0.30,
    }
    existing["preemptive_ttft_detail"] = ttft_metrics

    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nMetrics merged into {metrics_path}")
    overall_pass = (
        ttft_metrics["pass_criteria"]["ttft_p50_overhead_pass"]
    )
    print(f"=== Preemptive TTFT: {'PASS' if overall_pass else 'FAIL'} ===")


if __name__ == "__main__":
    main()
