"""Generate results/2026-05-25/metrics.json from SpeculativePacketPipeline benchmark.

Runs ~100 random queries through the pipeline, collects stats, and saves metrics.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from src.cache.kv_packet import KVPacketConfig
from src.cache.vericache_speculative_codec import (
    Int8DraftCodec,
    VeriCacheConfig,
)
from src.engine.speculative_packet_pipeline import (
    SpeculativePacketPipeline,
    SpeculativePacketPipelineConfig,
)

N_HEADS = 4
D_HEAD = 32
D_FLAT = N_HEADS * D_HEAD
N_TOKENS = 32
N_SEGMENTS = 10
N_RUNS = 100
SEED = 42


def make_pipeline() -> SpeculativePacketPipeline:
    cfg = SpeculativePacketPipelineConfig(
        kv_packet_config=KVPacketConfig(
            n_heads=N_HEADS,
            d_head=D_HEAD,
            n_adapter_tokens=4,
            adapter_steps=10,
            max_packets=64,
            seed=SEED,
        ),
        vericache_config=VeriCacheConfig(
            d_head=D_FLAT,
            acceptance_threshold=0.01,
            max_entries=256,
            seed=SEED,
        ),
        use_token_eviction_codec=False,
        seed=SEED,
    )
    return SpeculativePacketPipeline(cfg)


def main() -> None:
    torch.manual_seed(SEED)
    pipeline = make_pipeline()

    # Store N_SEGMENTS segments and train adapters
    for i in range(N_SEGMENTS):
        torch.manual_seed(i)
        kv = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
        pipeline.store_segment(f"seg{i}", kv)
        # Use a related context for adapter distillation
        ctx = torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)
        pipeline.train_segment_adapter(f"seg{i}", ctx)

    # Run ~100 queries across segments in non-contiguous order
    cosine_sims: list[float] = []
    final_relative_errors: list[float] = []

    for run_i in range(N_RUNS):
        # Access segments in shuffled order to generate non-contiguous hits
        seg_idx = (run_i * 3 + 7) % N_SEGMENTS  # non-sequential pattern
        seg_id = f"seg{seg_idx}"
        torch.manual_seed(run_i + 1000)
        Q = torch.randn(4, D_FLAT)

        result = pipeline.run(seg_id, Q)

        # Compute cosine similarity between final output and a reference
        # (full KV attention via VeriCache verified path)
        final_out = result.final_output
        if final_out is not None and final_out.norm() > 0:
            # Use VeriCache verified output as ground truth reference
            vr = pipeline.vericache.draft_and_verify(seg_id + "_K", seg_id + "_V", Q)
            if vr is not None:
                ref = vr.verified_output.reshape(final_out.shape)
                cos_sim = float(
                    F.cosine_similarity(
                        final_out.flatten().unsqueeze(0).float(),
                        ref.flatten().unsqueeze(0).float(),
                    ).item()
                )
                cosine_sims.append(cos_sim)
                # Relative error of final output vs ground truth
                rel_err = float(
                    (final_out.float() - ref.float()).norm()
                    / (ref.float().norm() + 1e-8)
                )
                final_relative_errors.append(rel_err)

    summary = pipeline.pipeline_summary()
    spec_stats = pipeline.vericache.speculative_stats()

    pipeline_cosine_similarity = (
        sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
    )
    final_relative_error_max = (
        max(final_relative_errors) if final_relative_errors else 0.0
    )

    metrics = {
        "draft_acceptance_rate": spec_stats["draft_acceptance_rate"],
        "mean_relative_error": spec_stats["mean_relative_error"],
        "memory_reduction_ratio": spec_stats["memory_reduction_ratio"],
        "compression_codec_name": spec_stats["compression_codec"],
        "noncontiguous_hit_rate": summary.get("noncontiguous_hit_rate", 0.0),
        "kv_packet_hit_rate": summary.get("kv_packet_hit_rate", 0.0),
        "b_hit_rate": summary.get("b_hit_rate", 0.0),
        "c_draft_acceptance_rate": summary.get("c_draft_acceptance_rate", 0.0),
        "pipeline_cosine_similarity": pipeline_cosine_similarity,
        "final_relative_error_max": final_relative_error_max,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "2026-05-25")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {out_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
