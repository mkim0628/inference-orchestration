"""Cross Activity A+C: NAtHRetentionTierDecider — dual-signal 4-tier decider.

Combines:
  A-2 NAtHDDROffloadingScheduler (cumulative attention score EMA)
  C-2 GlobalRetentionGateEvictionCodec (global retention gate score)
into a single final_score for 4-tier memory classification.

final_score_i = α × norm(attn_score_i) + (1−α) × retention_score_i

Supports dynamic tier-boundary adjustment:
  - Tier-4 ratio > max_eviction_ratio → raise Tier-3→4 boundary
  - HBM pressure > 0.8 → lower Tier-1→2 boundary
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.scheduler.nath_ddr_offloading import (
    NAtHDDROffloadingConfig,
    NAtHDDROffloadingScheduler,
)
from src.cache.global_retention_gate_eviction import GlobalRetentionGateEvictionCodec


@dataclass
class NAtHRetentionTierDeciderConfig:
    alpha: float = 0.5              # weight of cumulative attn score (1-alpha → retention)
    max_eviction_ratio: float = 0.03
    # None → inherit from NAtHDDROffloadingConfig defaults
    tier_boundaries: Optional[List[float]] = None
    seed: int = 42


class NAtHRetentionTierDecider:
    """Dual-signal 4-tier decider: NAtH cumulative attention + global retention gate.

    Cross Activity A+C:
      A-2 NAtHDDROffloadingScheduler + C-2 GlobalRetentionGateEvictionCodec.

    Dual-signal combination:
      final_score_i = α × L2_norm(attn_score_i) + (1−α) × retention_score_i

    Dynamic tier-boundary adjustment:
      - permanent eviction ratio > max_eviction_ratio → raise Tier-3→4 boundary
      - HBM pressure > 0.8                            → lower Tier-1→2 boundary

    Measurement (target 9):
      Solo A-2 / Solo C-2 / Combined Cross-1 compared on throughput/memory/accuracy.
    """

    def __init__(
        self,
        config: NAtHRetentionTierDeciderConfig,
        nath_scheduler: NAtHDDROffloadingScheduler,
        retention_codec: GlobalRetentionGateEvictionCodec,
    ) -> None:
        self.config = config
        self._nath = nath_scheduler
        self._retention = retention_codec
        self._tier_stats: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

        # Mutable tier boundaries — start from scheduler config or defaults
        boundaries = config.tier_boundaries
        if boundaries is None:
            boundaries = list(nath_scheduler.config.tier_boundaries)
        self._tier_boundaries: List[float] = list(boundaries)

    # ------------------------------------------------------------------ #
    # Core decision                                                        #
    # ------------------------------------------------------------------ #

    def decide_tier(
        self,
        token_keys: List[str],
        kv_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        """Classify tokens into 4 tiers using dual-signal combined score.

        Algorithm:
          1. Retrieve cumulative attention EMA from NAtHDDROffloadingScheduler.
          2. Retrieve global retention scores from GlobalRetentionGateEvictionCodec.
          3. L2-normalise both signals, then combine with alpha weight.
          4. Assign tiers by quantile of combined scores.
          5. Enforce max_eviction_ratio; adjust boundaries if violated.

        Returns:
            {token_key: tier} where tier ∈ {1, 2, 3, 4}
        """
        n = len(token_keys)
        if n == 0:
            return {}

        # ── Signal 1: cumulative attention EMA ─────────────────────────
        attn_scores = torch.tensor(
            [self._nath._attn_score_ema.get(k, 0.0) for k in token_keys],
            dtype=torch.float32,
        )

        # ── Signal 2: global retention gate score ──────────────────────
        if kv_tensor is not None:
            retention_scores = self._retention.get_global_retention_score(kv=kv_tensor)
            # Trim/pad to match n tokens
            if retention_scores.shape[0] != n:
                if retention_scores.shape[0] > n:
                    retention_scores = retention_scores[:n]
                else:
                    pad = torch.ones(n - retention_scores.shape[0])
                    retention_scores = torch.cat([retention_scores, pad])
        else:
            retention_scores = self._retention.get_global_retention_score(
                token_ids=list(range(n))
            )
            if retention_scores.shape[0] != n:
                retention_scores = torch.ones(n)

        # ── L2 normalise each signal ────────────────────────────────────
        attn_norm = self._l2_normalize(attn_scores)
        ret_norm = self._l2_normalize(retention_scores.float())

        # ── Combine ─────────────────────────────────────────────────────
        alpha = self.config.alpha
        combined = alpha * attn_norm + (1.0 - alpha) * ret_norm  # [n]

        # ── Tier assignment by rank (avoids float-precision boundary issues) ─
        p1, p2, p3 = self._tier_boundaries
        sorted_combined, sorted_idx = combined.sort(descending=True)
        n_tier1 = max(1, int(math.ceil(n * p1)))
        n_tier12 = max(n_tier1, int(math.ceil(n * p2)))
        n_tier123 = max(n_tier12, int(math.ceil(n * p3)))

        rank_tier = torch.full((n,), 4, dtype=torch.long)
        rank_tier[sorted_idx[:n_tier1]] = 1
        rank_tier[sorted_idx[n_tier1:n_tier12]] = 2
        rank_tier[sorted_idx[n_tier12:n_tier123]] = 3

        tier_map: Dict[str, int] = {k: int(t) for k, t in zip(token_keys, rank_tier.tolist())}

        # ── Enforce max_eviction_ratio ──────────────────────────────────
        tier4_count = sum(1 for t in tier_map.values() if t == 4)
        max_tier4 = max(0, math.floor(n * self.config.max_eviction_ratio))
        if tier4_count > max_tier4:
            tier4_keys_by_score = sorted(
                [(float(combined[token_keys.index(k)]), k)
                 for k in token_keys if tier_map[k] == 4],
                reverse=True,
            )
            n_promote = tier4_count - max_tier4
            for _, k in tier4_keys_by_score[:n_promote]:
                tier_map[k] = 3

        # ── Update statistics ───────────────────────────────────────────
        for t in tier_map.values():
            self._tier_stats[t] = self._tier_stats.get(t, 0) + 1

        return tier_map

    # ------------------------------------------------------------------ #
    # Dynamic boundary adjustment                                          #
    # ------------------------------------------------------------------ #

    def adjust_tier_boundaries(
        self,
        current_eviction_ratio: float,
        hbm_pressure: float = 0.0,
    ) -> None:
        """Dynamically adjust tier boundaries to maintain accuracy and HBM constraints.

        - current_eviction_ratio > max_eviction_ratio:
            Raise Tier-3→4 boundary (p3) to reduce Tier-4 fraction.
        - hbm_pressure > 0.8:
            Lower Tier-1→2 boundary (p1) to push more tokens to DDR offload.
        """
        p1, p2, p3 = self._tier_boundaries

        if current_eviction_ratio > self.config.max_eviction_ratio:
            # Raise p3 by 10% of remaining headroom (capped at 0.99)
            headroom = 1.0 - p3
            p3 = min(0.99, p3 + 0.1 * headroom)

        if hbm_pressure > 0.8:
            # Lower p1 by 20% of its current value (more tokens offloaded to DDR)
            p1 = max(0.05, p1 * 0.8)

        # Ensure ordering p1 < p2 < p3
        p2 = max(p1 + 0.01, min(p2, p3 - 0.01))
        self._tier_boundaries = [p1, p2, p3]

        # Propagate to the underlying NAtH scheduler
        self._nath.config.tier_boundaries = self._tier_boundaries

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def tier_distribution(self) -> Dict[int, float]:
        """Fraction of tokens assigned to each tier so far."""
        total = sum(self._tier_stats.values())
        if total == 0:
            return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        return {k: v / total for k, v in self._tier_stats.items()}

    def reset_stats(self) -> None:
        """Reset tier statistics."""
        self._tier_stats = {1: 0, 2: 0, 3: 0, 4: 0}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        """L2-normalise a 1-D tensor to [0, 1] range via min-max scaling.

        Falls back to uniform if all values are identical.
        """
        x_min = x.min()
        x_max = x.max()
        span = x_max - x_min
        if span < 1e-8:
            return torch.full_like(x, 0.5)
        return (x - x_min) / span
