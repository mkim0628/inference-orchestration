"""
DualFilterSegmentSelector — Activity B+C.

Two-stage filtering pipeline that independently combines query-relevance
(Activity B) with pre-RoPE token importance (Activity C) to select and
filter KV cache segments.

Stage 1: Query-relevance filter — keep top stage1_filter_ratio of candidates
         by cosine similarity between the query embedding and each segment's
         mean K embedding.
Stage 2: TriAttentionCodec importance filter — within each selected segment,
         retain only the top stage2_token_budget fraction of tokens by
         pre-RoPE trigonometric importance scores.

This selector is intentionally independent of QueryCentricTriAttentionCache
so it can be used as a standalone post-processing step before attention.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.cache.query_centric_recompute import QueryCentricRecomputeCache
from src.cache.tri_attention_codec import TriAttentionCodec


class DualFilterSegmentSelector:
    """
    Two-stage segment filter: query relevance then token importance.

    Attributes:
        qcrc: QueryCentricRecomputeCache instance for relevance metadata.
        codec: TriAttentionCodec instance for token-level importance scoring.
        stage1_filter_ratio: Fraction of candidates to pass stage 1 (0..1].
        stage2_token_budget: Fraction of tokens to retain per segment (0..1].
    """

    def __init__(
        self,
        qcrc: QueryCentricRecomputeCache,
        codec: TriAttentionCodec,
        stage1_filter_ratio: float = 0.40,
        stage2_token_budget: float = 0.20,
    ) -> None:
        if not (0.0 < stage1_filter_ratio <= 1.0):
            raise ValueError("stage1_filter_ratio must be in (0, 1]")
        if not (0.0 < stage2_token_budget <= 1.0):
            raise ValueError("stage2_token_budget must be in (0, 1]")

        self.qcrc = qcrc
        self.codec = codec
        self.stage1_filter_ratio = stage1_filter_ratio
        self.stage2_token_budget = stage2_token_budget

    def select(
        self,
        query_embedding: torch.Tensor,
        candidate_segments: List[str],
        segment_store: Dict[str, Dict],
        keys_pre_rope: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply two-stage filtering and return filtered KV tensors.

        Stage 1: Rank candidates by cosine similarity(query, segment_embedding)
                 and keep the top stage1_filter_ratio fraction.
        Stage 2: Compress each selected segment's KV to retain only the top
                 stage2_token_budget fraction of tokens by importance; if
                 codec is uncalibrated or pre-RoPE keys are unavailable the
                 segment is returned unfiltered.

        Args:
            query_embedding: Query mean K vector [head_dim].
            candidate_segments: Segment hash list to filter.
            segment_store: Mapping {key: {"kv": Tensor, "embedding": Tensor}}.
                           "embedding" must be a [head_dim] tensor; if absent
                           the segment's KV mean is used instead.
            keys_pre_rope: Optional mapping {key: pre-RoPE K tensor} for
                           stage-2 token importance filtering. If None or a
                           key is missing, stage-2 is skipped for that segment.

        Returns:
            Dict {seg_key: filtered_kv_tensor} for segments that passed both
            stages. Tensors retain their original [L, H, S', D] shape where
            S' ≤ original S.
        """
        if keys_pre_rope is None:
            keys_pre_rope = {}

        # ---------------------------------------------------------------- #
        # Stage 1: query-relevance filter                                   #
        # ---------------------------------------------------------------- #
        scores: Dict[str, float] = {}
        for key in candidate_segments:
            if key not in segment_store:
                continue
            entry = segment_store[key]
            emb = entry.get("embedding")
            if emb is None:
                # Fall back to KV mean as embedding
                emb = entry["kv"].mean(dim=(0, 1, 2))
            scores[key] = F.cosine_similarity(
                query_embedding.unsqueeze(0).float(),
                emb.unsqueeze(0).float(),
            ).item()

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        n_stage1 = max(1, int(len(sorted_keys) * self.stage1_filter_ratio))
        stage1_passed = sorted_keys[:n_stage1]

        # ---------------------------------------------------------------- #
        # Stage 2: TriAttentionCodec token importance filter               #
        # ---------------------------------------------------------------- #
        result: Dict[str, torch.Tensor] = {}
        codec_ready = self.codec.mu_k is not None

        for key in stage1_passed:
            kv = segment_store[key]["kv"]
            k_pre = keys_pre_rope.get(key)

            if codec_ready and k_pre is not None:
                compressed = self.codec.compress(kv, k_pre, self.stage2_token_budget)
                result[key] = self.codec.decompress(compressed)
            else:
                # Codec uncalibrated or pre-RoPE keys unavailable: pass through
                result[key] = kv

        return result

    def stage1_scores(
        self,
        query_embedding: torch.Tensor,
        candidate_segments: List[str],
        segment_store: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Return cosine-similarity scores for all candidates (stage-1 only).

        Useful for diagnostics and threshold tuning without applying stage 2.

        Args:
            query_embedding: Query mean K vector [head_dim].
            candidate_segments: Segment hash list.
            segment_store: Mapping {key: {"kv": Tensor, "embedding": Tensor}}.

        Returns:
            Dict {key: cosine_similarity_score}.
        """
        scores: Dict[str, float] = {}
        for key in candidate_segments:
            if key not in segment_store:
                continue
            entry = segment_store[key]
            emb = entry.get("embedding")
            if emb is None:
                emb = entry["kv"].mean(dim=(0, 1, 2))
            scores[key] = F.cosine_similarity(
                query_embedding.unsqueeze(0).float(),
                emb.unsqueeze(0).float(),
            ).item()
        return scores
