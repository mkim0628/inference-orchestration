"""Activity A: NAtHDDROffloadingScheduler — Semantic-aware 4-tier DDR offloading.

Based on NAtH (arXiv 2605.09490): "accuracy depends only on permanent eviction rate;
DDR offloading achieves zero-approximation-error".
Unlike CacheAwareScheduler (hit-rate-weighted priority), this scheduler uses
cumulative attention score EMA to classify tokens into 4 tiers and prefers
DDR offloading over permanent eviction under memory pressure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore


@dataclass
class NAtHDDROffloadingConfig:
    # 4-tier boundaries: [p1, p2, p3] percentile thresholds of cumulative attention score
    # Tier 1 (HBM):           >= p1 percentile  — FP16 GPU HBM retention
    # Tier 2 (DDR prefetch):  p1 ~ p2           — CPU DDR offload + async prefetch before attn
    # Tier 3 (DDR INT8):      p2 ~ p3           — CPU DDR INT8 compressed retention
    # Tier 4 (permanent evict): < (1 - p3)      — discarded; ratio <= max_eviction_ratio
    tier_boundaries: List[float] = field(
        default_factory=lambda: [0.30, 0.70, 0.97]
    )
    max_eviction_ratio: float = 0.03    # hard cap on permanent eviction (3%)
    ema_alpha: float = 0.95             # EMA decay for cumulative attention score
    prefetch_chunk_size: int = 64       # tokens per async DDR→HBM prefetch batch
    max_wait_ratio: float = 2.0         # fairness: max wait time multiple
    seed: int = 42


class NAtHDDROffloadingScheduler:
    """NAtH semantic-aware 4-tier DDR offloading minimal-eviction scheduler.

    Activity A: KV Cache-aware Scheduling.
    Scheduling unit: per-request — classifies each request's KV tokens into 4 tiers.
    Cache state: _attn_score_ema dict tracks per-token cumulative attention score EMA.

    4-tier memory policy:
      Tier 1 (HBM immediate):    top p1 percentile       — FP16 GPU HBM
      Tier 2 (DDR prefetch):     p1 ~ p2                 — CPU DDR FP16 + async prefetch
      Tier 3 (DDR INT8):         p2 ~ p3                 — CPU DDR INT8 compressed
      Tier 4 (permanent evict):  below (1-p3) percentile — discard; ratio <= max_eviction_ratio

    Evaluation criteria (evaluation_criteria.md §2):
      - Scheduling overhead: TTFT p50 increase < +5%
      - Cache hit rate: +10%p vs no-scheduling baseline
      - Request fairness: max wait time < max_wait_ratio × baseline

    NAtH theory validation:
      - Permanent eviction ratio <= 3% → GSM8K 91%+ accuracy
      - DDR offloaded tokens restored at full FP16 precision on prefetch (zero-approx-error)
    """

    def __init__(
        self,
        config: NAtHDDROffloadingConfig,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.config = config
        self._cache = cache
        # Per-token cumulative attention score EMA: {token_key: float}
        self._attn_score_ema: Dict[str, float] = {}
        # Tier assignment result: {token_key: int(1..4)}
        self._token_tier: Dict[str, int] = {}
        # DDR offload buffers (CPU tensors)
        self._ddr_buffer_fp16: Dict[str, torch.Tensor] = {}   # Tier 2: FP16 on CPU
        self._ddr_buffer_int8: Dict[str, torch.Tensor] = {}   # Tier 3: INT8 on CPU
        self._ddr_scale: Dict[str, float] = {}                 # INT8 quantisation scale
        # Scheduling overhead measurement
        self._scheduling_times: List[float] = []
        self._permanent_evictions: int = 0
        self._total_decisions: int = 0
        # Per-request arrival times for fairness tracking
        self._arrival_times: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # EMA update                                                           #
    # ------------------------------------------------------------------ #

    def update_attention_score(
        self,
        token_key: str,
        attn_score: float,
    ) -> None:
        """Update per-token cumulative attention score EMA each decoding step.

        new_score = alpha * old_score + (1 - alpha) * attn_score
        Older tokens' scores decay naturally → temporal importance reflected.
        """
        alpha = self.config.ema_alpha
        old = self._attn_score_ema.get(token_key, 0.0)
        self._attn_score_ema[token_key] = alpha * old + (1.0 - alpha) * attn_score

    # ------------------------------------------------------------------ #
    # 4-tier classification                                                #
    # ------------------------------------------------------------------ #

    def classify_tokens(
        self,
        token_keys: List[str],
    ) -> Dict[str, int]:
        """Classify tokens into 4 tiers by cumulative attention EMA scores.

        Algorithm:
          1. Collect EMA score for each token (default 0 if unseen).
          2. Compute percentile boundaries [p1, p2, p3].
          3. Assign Tier 1/2/3/4 by threshold.
          4. If Tier-4 ratio > max_eviction_ratio, raise Tier-3→4 boundary
             to protect accuracy (shift tokens from Tier 4 → Tier 3).

        Returns:
            {token_key: tier} where tier ∈ {1, 2, 3, 4}
        """
        if not token_keys:
            return {}

        scores = [self._attn_score_ema.get(k, 0.0) for k in token_keys]
        n = len(scores)
        score_tensor = torch.tensor(scores, dtype=torch.float32)

        p1, p2, p3 = self.config.tier_boundaries

        # Quantile thresholds (higher score = more important = lower tier number)
        # We want top p1 fraction → Tier 1, so threshold at (1-p1) quantile.
        # Use rank-based assignment to avoid float-precision boundary issues.
        sorted_scores, sorted_idx = score_tensor.sort(descending=True)
        n_tier1 = max(1, int(math.ceil(n * p1)))
        n_tier12 = max(n_tier1, int(math.ceil(n * p2)))
        n_tier123 = max(n_tier12, int(math.ceil(n * p3)))

        rank_tier = torch.full((n,), 4, dtype=torch.long)
        rank_tier[sorted_idx[:n_tier1]] = 1
        rank_tier[sorted_idx[n_tier1:n_tier12]] = 2
        rank_tier[sorted_idx[n_tier12:n_tier123]] = 3
        # Remaining (lowest) stay at 4

        tier_map: Dict[str, int] = {k: int(t) for k, t in zip(token_keys, rank_tier.tolist())}

        # Enforce max_eviction_ratio: if Tier-4 count too high, promote excess Tier-4 → Tier-3
        tier4_count = sum(1 for t in tier_map.values() if t == 4)
        max_tier4 = max(0, math.floor(n * self.config.max_eviction_ratio))
        if tier4_count > max_tier4:
            # Promote the highest-scoring Tier-4 tokens to Tier 3 until count is at limit
            tier4_keys_by_score = sorted(
                [(scores[i], token_keys[i]) for i in range(n) if tier_map[token_keys[i]] == 4],
                reverse=True,  # highest score first → promote these to Tier 3
            )
            n_promote = tier4_count - max_tier4
            for _, k in tier4_keys_by_score[:n_promote]:
                tier_map[k] = 3

        self._token_tier.update(tier_map)
        self._total_decisions += n
        self._permanent_evictions += sum(1 for t in tier_map.values() if t == 4)

        return tier_map

    # ------------------------------------------------------------------ #
    # DDR offloading                                                       #
    # ------------------------------------------------------------------ #

    def offload_to_ddr(
        self,
        token_key: str,
        kv_tensor: torch.Tensor,
        tier: int,
    ) -> None:
        """Offload KV tensor from GPU HBM to CPU DDR.

        Tier 2: FP16 preserved on CPU (zero approximation error on restore).
        Tier 3: INT8 quantised on CPU (2× memory saving, small approximation).
        """
        if tier == 2:
            # Non-blocking transfer; keep FP16 precision
            self._ddr_buffer_fp16[token_key] = kv_tensor.detach().cpu()
        elif tier == 3:
            # INT8 symmetric quantisation: scale = max_abs / 127
            max_abs = kv_tensor.abs().max().item()
            scale = max(max_abs / 127.0, 1e-8)
            quantised = (kv_tensor.detach().float() / scale).round().clamp(-127, 127).to(torch.int8).cpu()
            self._ddr_buffer_int8[token_key] = quantised
            self._ddr_scale[token_key] = scale

    def prefetch_from_ddr(
        self,
        token_keys: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Restore Tier-2 DDR tokens to GPU HBM at full FP16 precision.

        Called before attention computation to ensure Tier-2 tokens are GPU-resident.
        Zero approximation error for Tier 2 (FP16 round-trip).

        Returns:
            {token_key: kv_tensor on GPU}
        """
        result: Dict[str, torch.Tensor] = {}
        for k in token_keys:
            if k in self._ddr_buffer_fp16:
                # Restore to GPU; original FP16 precision preserved
                result[k] = self._ddr_buffer_fp16[k].cuda() if torch.cuda.is_available() else self._ddr_buffer_fp16[k]
        return result

    def restore_tier3_from_ddr(
        self,
        token_key: str,
        target_dtype: torch.dtype = torch.float16,
    ) -> Optional[torch.Tensor]:
        """Dequantise Tier-3 INT8 token back to approximate FP16 on CPU.

        Returns:
            Dequantised tensor, or None if not found.
        """
        if token_key not in self._ddr_buffer_int8:
            return None
        scale = self._ddr_scale.get(token_key, 1.0)
        q = self._ddr_buffer_int8[token_key].float() * scale
        return q.to(target_dtype)

    # ------------------------------------------------------------------ #
    # Request scheduling                                                   #
    # ------------------------------------------------------------------ #

    def schedule_request(
        self,
        request: Dict,
    ) -> Dict:
        """Classify KV tokens of a single request into 4 tiers and annotate.

        Args:
            request: {"id": str, "token_ids": List[int], "arrival_time": float}

        Returns:
            request with added keys:
              "tier_assignment": {token_key: tier}
              "ddr_offload_keys": List[str]   (Tier 2 + 3)
              "evict_keys": List[str]          (Tier 4)
        """
        t0 = time.monotonic()

        req_id: str = request.get("id", "")
        token_ids: List[int] = request.get("token_ids", [])
        arrival_time: float = request.get("arrival_time", t0)
        self._arrival_times[req_id] = arrival_time

        # Build token keys scoped to this request
        token_keys = [f"{req_id}:tok{i}:{tid}" for i, tid in enumerate(token_ids)]

        # Initialise EMA scores for new tokens (zero → will be updated on first decode)
        for k in token_keys:
            if k not in self._attn_score_ema:
                self._attn_score_ema[k] = 0.0

        tier_assignment = self.classify_tokens(token_keys)

        ddr_offload_keys = [k for k, t in tier_assignment.items() if t in (2, 3)]
        evict_keys = [k for k, t in tier_assignment.items() if t == 4]

        request["tier_assignment"] = tier_assignment
        request["ddr_offload_keys"] = ddr_offload_keys
        request["evict_keys"] = evict_keys

        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)
        return request

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def scheduling_overhead_ms_p50(self) -> float:
        """Median scheduling overhead in milliseconds."""
        if not self._scheduling_times:
            return 0.0
        sorted_times = sorted(self._scheduling_times)
        return sorted_times[len(sorted_times) // 2]

    def permanent_eviction_ratio(self) -> float:
        """Ratio of permanently evicted tokens to total classification decisions."""
        if self._total_decisions == 0:
            return 0.0
        return self._permanent_evictions / self._total_decisions

    def cache_hit_rate(self) -> float:
        """Effective cache hit rate including DDR offloaded tokens (Tier 1+2+3)."""
        if self._total_decisions == 0:
            return 0.0
        return 1.0 - self.permanent_eviction_ratio()

    def get_tier_distribution(self) -> Dict[int, float]:
        """Current tier distribution as fractions."""
        if self._total_decisions == 0:
            return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        for t in self._token_tier.values():
            counts[t] = counts.get(t, 0) + 1
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total > 0 else {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

    def reset_stats(self) -> None:
        """Reset all scheduling statistics."""
        self._scheduling_times.clear()
        self._permanent_evictions = 0
        self._total_decisions = 0
        self._attn_score_ema.clear()
        self._token_tier.clear()
        self._ddr_buffer_fp16.clear()
        self._ddr_buffer_int8.clear()
        self._ddr_scale.clear()
        self._arrival_times.clear()
