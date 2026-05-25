"""Speculative Packet Pipeline — Cross-1 B+C Integration (2026-05-25).

Orchestrates KVPacketCache (Activity B) + VeriCacheSpeculativeCodec (Activity C).
Non-contiguous packet segments feed directly into VeriCache draft-verify as compressed KV source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.kv_packet import KVPacketCache, KVPacketConfig
from src.cache.vericache_speculative_codec import (
    VeriCacheSpeculativeCodec,
    VeriCacheConfig,
    Int8DraftCodec,
    TokenEvictionDraftCodec,
    VerificationResult,
)


@dataclass
class SpeculativePacketPipelineConfig:
    kv_packet_config: Optional[KVPacketConfig] = None
    vericache_config: Optional[VeriCacheConfig] = None
    use_token_eviction_codec: bool = False    # True: TokenEviction, False: INT8
    token_eviction_keep_ratio: float = 0.5
    seed: int = 42


@dataclass
class PipelineResult:
    """B+C pipeline execution result."""
    segment_id: str
    path: str                          # "b_hit_c_draft", "b_miss_fallback", "c_reject_verified"
    final_output: torch.Tensor         # Final attention output [n_q, d_head]
    b_hit: bool                        # B: KV Packet hit
    c_accepted: bool                   # C: VeriCache draft accepted
    relative_error: Optional[float]    # VeriCache verify relative_error
    memory_compressed_bytes: int       # C compressed KV memory
    noncontiguous_adapter_applied: bool


class SpeculativePacketPipeline:
    """KV Packet (B) + VeriCache Speculative Codec (C) integrated B+C pipeline.

    Processing flow:

    Step 1 (B-1, non-contiguous hit detection):
      KVPacketCache.get_for_vericache(segment_id)
      -> hit: returns (K_packet, V_packet) with adapter applied (FLOPs = 0)

    Step 2 (C-1, VeriCache store):
      On hit: VeriCacheSpeculativeCodec.put_kv_pair(segment_id, K_packet, V_packet)
      -> generates compressed KV + stores full KV reference

    Step 3 (C-1, speculative drafting):
      VeriCacheSpeculativeCodec.draft_and_verify(key_K, key_V, Q)
      -> returns VerificationResult

    Step 4 (C-1, deterministic output selection):
      VeriCacheSpeculativeCodec.get_final_output(result)
      -> accepted: draft_output (compressed KV, within threshold)
      -> rejected: verified_output (full KV, deterministic guarantee)

    B miss: falls back to direct full KV computation (standard attention path)
    """

    def __init__(self, config: SpeculativePacketPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        kv_cfg = config.kv_packet_config or KVPacketConfig(seed=config.seed)
        vc_cfg = config.vericache_config or VeriCacheConfig(seed=config.seed)
        self.kv_packet_cache = KVPacketCache(kv_cfg)
        self.vericache = VeriCacheSpeculativeCodec(vc_cfg)

        # Configure compression codec
        if config.use_token_eviction_codec:
            self.vericache.set_draft_codec(
                TokenEvictionDraftCodec(config.token_eviction_keep_ratio)
            )
        else:
            self.vericache.set_draft_codec(Int8DraftCodec())

        self._pipeline_stats: List[dict] = []

    def store_segment(
        self,
        segment_id: str,
        kv_block: torch.Tensor,    # [n_tokens, 2, n_heads, d_head]
    ) -> None:
        """Store segment in KV Packet cache (includes adapter initialization)."""
        self.kv_packet_cache.put(segment_id, kv_block)

    def train_segment_adapter(
        self,
        segment_id: str,
        context_kv: torch.Tensor,  # [m_ref, 2, n_heads, d_head]
    ) -> float:
        """Train adapter via self-supervised distillation. Recommended once after store."""
        return self.kv_packet_cache.train_adapter(segment_id, context_kv)

    def run(
        self,
        segment_id: str,
        Q: torch.Tensor,            # [n_q, d_head]
        fallback_K: Optional[torch.Tensor] = None,  # Fallback K on B miss
        fallback_V: Optional[torch.Tensor] = None,  # Fallback V on B miss
    ) -> PipelineResult:
        """Run B+C integrated pipeline.

        Args:
            segment_id: segment identifier
            Q: query tensor [n_q, d_head]
            fallback_K/V: full KV to use when B misses (None returns zero tensor)

        Returns:
            PipelineResult
        """
        key_K = segment_id + "_K"
        key_V = segment_id + "_V"

        # Step 1: Check KV Packet hit
        kv_pair = self.kv_packet_cache.get_for_vericache(segment_id)

        if kv_pair is None:
            # B miss: fall back to direct full KV computation
            if fallback_K is not None and fallback_V is not None:
                final_out = VeriCacheSpeculativeCodec._compute_attention(
                    Q, fallback_K, fallback_V
                )
            else:
                final_out = torch.zeros(Q.shape[0], Q.shape[-1], dtype=Q.dtype)
            result = PipelineResult(
                segment_id=segment_id,
                path="b_miss_fallback",
                final_output=final_out,
                b_hit=False,
                c_accepted=False,
                relative_error=None,
                memory_compressed_bytes=0,
                noncontiguous_adapter_applied=False,
            )
            self._pipeline_stats.append({"path": "b_miss_fallback"})
            return result

        # Step 2: Store packet KV in VeriCache (generates compressed copy)
        K_packet, V_packet = kv_pair
        # Flatten n_heads dimension for VeriCache (expects [n_tokens, d_head] 2D tensors)
        K_2d = K_packet.reshape(K_packet.shape[0], -1)   # [n_tokens, n_heads*d_head]
        V_2d = V_packet.reshape(V_packet.shape[0], -1)
        self.vericache.put_kv_pair(segment_id, K_2d, V_2d)

        # Step 3: Speculative drafting + verification
        Q_2d = Q.reshape(Q.shape[0], -1) if Q.dim() > 2 else Q
        verify_result = self.vericache.draft_and_verify(key_K, key_V, Q_2d)

        if verify_result is None:
            # Draft failed: fallback
            final_out = torch.zeros(Q.shape[0], Q.shape[-1], dtype=Q.dtype)
            result = PipelineResult(
                segment_id=segment_id,
                path="b_hit_c_miss_fallback",
                final_output=final_out,
                b_hit=True,
                c_accepted=False,
                relative_error=None,
                memory_compressed_bytes=self.vericache.memory_bytes(),
                noncontiguous_adapter_applied=True,
            )
            self._pipeline_stats.append({"path": "b_hit_c_miss_fallback"})
            return result

        # Step 4: Deterministic output selection
        final_out = self.vericache.get_final_output(verify_result)
        path = (
            "b_hit_c_draft"
            if verify_result.accepted
            else "c_reject_verified"
        )

        result = PipelineResult(
            segment_id=segment_id,
            path=path,
            final_output=final_out.reshape(Q.shape[0], -1),
            b_hit=True,
            c_accepted=verify_result.accepted,
            relative_error=verify_result.relative_error,
            memory_compressed_bytes=self.vericache.memory_bytes(),
            noncontiguous_adapter_applied=True,
        )
        self._pipeline_stats.append({
            "path": path,
            "c_accepted": verify_result.accepted,
            "relative_error": verify_result.relative_error,
        })
        return result

    def pipeline_summary(self) -> dict:
        """JSON-serializable pipeline statistics."""
        if not self._pipeline_stats:
            return {}
        b_hits = [s for s in self._pipeline_stats if s["path"] != "b_miss_fallback"]
        return {
            "total_runs": len(self._pipeline_stats),
            "b_hit_rate": len(b_hits) / max(1, len(self._pipeline_stats)),
            "c_draft_acceptance_rate": self.vericache.draft_acceptance_rate(),
            "mean_relative_error": self.vericache.mean_relative_error(),
            "kv_packet_hit_rate": self.kv_packet_cache.hit_rate(),
            "noncontiguous_hit_rate": self.kv_packet_cache.noncontiguous_hit_rate(),
            "vericache_memory_reduction": self.vericache.memory_reduction_ratio(),
            **self.vericache.speculative_stats(),
        }
