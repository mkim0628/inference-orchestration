"""Activity A-1: ObjectCache S3 4th-tier break-even dynamic router.

Routes KV cache requests to S3 when the break-even hit rate condition is met,
using EMA hit-rate tracking and hysteresis to avoid oscillation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from src.cache.cdc_content_hash_interface import CDCContentHashSegmentIDInterface
from src.engine.runner import InferenceRequest
from src.scheduler.base import BaseScheduler


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #


@dataclass
class S3TierConfig:
    """ObjectCache S3 tier router configuration."""
    context_lengths: List[int] = field(
        default_factory=lambda: [4096, 8192, 16384, 32768, 65536]
    )
    breakeven_table: Dict[int, float] = field(
        default_factory=lambda: {
            4096: 0.15,
            8192: 0.18,
            16384: 0.22,
            32768: 0.28,
            65536: 0.35,
        }
    )
    hysteresis_band: float = 0.05
    ema_gamma: float = 0.9
    max_s3_requests_per_batch: int = 4
    s3_enabled_by_default: bool = False
    rdma_bandwidth_gbps: float = 100.0


def _load_breakeven_table(config_path: Optional[str] = None) -> Dict[int, float]:
    """Load break-even table from YAML config if available."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs",
            "objectcache_breakeven_table.yaml"
        )
    config_path = os.path.normpath(config_path)

    if not os.path.exists(config_path):
        return {4096: 0.15, 8192: 0.18, 16384: 0.22, 32768: 0.28, 65536: 0.35}

    with open(config_path) as f:
        data = yaml.safe_load(f)
    table = data.get("breakeven_table", {})
    return {int(k): float(v) for k, v in table.items()}


# ------------------------------------------------------------------ #
# ObjectCacheS3TierRouter                                             #
# ------------------------------------------------------------------ #


class ObjectCacheS3TierRouter(BaseScheduler):
    """S3 object storage 4th KV cache tier break-even dynamic router.

    Break-even hit rate formula:
        hit_rate_breakeven = T_recompute / (T_recompute + T_s3)

    EMA hit rate update:
        hit_rate_ema = γ × current_hit_rate + (1 - γ) × hit_rate_ema

    S3 tier activation condition:
        hit_rate_ema >= breakeven(context_length) + hysteresis_band

    S3 tier deactivation condition:
        hit_rate_ema < breakeven(context_length) - hysteresis_band
    """

    def __init__(
        self,
        config: S3TierConfig,
        segment_interface: CDCContentHashSegmentIDInterface,
    ) -> None:
        self.config = config
        self.segment_interface = segment_interface
        self._hit_rate_ema: float = 0.0
        self._s3_active: bool = config.s3_enabled_by_default

        # Load break-even table (config overrides YAML if provided)
        if config.breakeven_table:
            self._breakeven_table = dict(config.breakeven_table)
        else:
            self._breakeven_table = _load_breakeven_table()

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Route requests, setting s3_tier metadata flag when S3 is activated.

        Steps:
          1. For each request, compute context_length-based break-even
          2. Compare EMA hit rate with break-even + hysteresis
          3. Apply max_s3_requests_per_batch cap
          4. Mark s3_tier=True in request metadata for eligible requests
          5. Reorder: non-S3 requests first (cache warming effect)
        """
        s3_requests: List[InferenceRequest] = []
        non_s3_requests: List[InferenceRequest] = []

        s3_count = 0

        for req in requests:
            context_length = len(req.token_ids)
            breakeven = self.get_breakeven_for_context(context_length)

            # Determine if S3 should be active for this request
            should_use_s3 = (
                self._s3_active
                and self._hit_rate_ema >= breakeven + self.config.hysteresis_band
                and s3_count < self.config.max_s3_requests_per_batch
            )

            if should_use_s3:
                # Attach S3 routing metadata
                if not hasattr(req, "metadata") or req.metadata is None:
                    req.metadata = {}  # type: ignore[attr-defined]
                req.metadata["s3_tier"] = True  # type: ignore[attr-defined]
                s3_requests.append(req)
                s3_count += 1
            else:
                if hasattr(req, "metadata") and req.metadata is not None:
                    req.metadata.pop("s3_tier", None)  # type: ignore[attr-defined]
                non_s3_requests.append(req)

        # Non-S3 requests first (cache warming), then S3 requests
        return non_s3_requests + s3_requests

    def update_hit_rate_ema(self, current_hit_rate: float) -> None:
        """Update EMA hit rate and toggle S3 tier activation state.

        hit_rate_ema = γ × current + (1-γ) × ema
        Activation:   ema >= breakeven + hysteresis → activate
        Deactivation: ema <  breakeven - hysteresis → deactivate
        """
        gamma = self.config.ema_gamma
        self._hit_rate_ema = (
            gamma * current_hit_rate + (1.0 - gamma) * self._hit_rate_ema
        )

        # Use median context length for EMA-based toggling
        median_ctx = self.config.context_lengths[len(self.config.context_lengths) // 2]
        breakeven = self.get_breakeven_for_context(median_ctx)

        if self._hit_rate_ema >= breakeven + self.config.hysteresis_band:
            self._s3_active = True
        elif self._hit_rate_ema < breakeven - self.config.hysteresis_band:
            self._s3_active = False
        # In the hysteresis band: no change (prevents oscillation)

    def compute_breakeven_hit_rate(
        self,
        t_recompute_ms: float,
        t_s3_ms: float,
    ) -> float:
        """Compute break-even hit rate.

        hit_rate_breakeven = t_recompute / (t_recompute + t_s3)

        At this hit rate, savings from S3 cache retrieval exactly equals
        the cost of the additional S3 latency overhead.
        """
        if t_recompute_ms + t_s3_ms <= 0.0:
            return 0.0
        return t_recompute_ms / (t_recompute_ms + t_s3_ms)

    def get_breakeven_for_context(self, context_length: int) -> float:
        """Look up break-even hit rate for a given context length.

        Uses nearest lower bound from the break-even table.
        Falls back to max table value for very long contexts.
        """
        sorted_lengths = sorted(self._breakeven_table.keys())
        if not sorted_lengths:
            return 0.20  # sensible default

        # Find the largest table entry <= context_length
        best_key = sorted_lengths[0]
        for length in sorted_lengths:
            if length <= context_length:
                best_key = length
            else:
                break

        return self._breakeven_table[best_key]

    @property
    def s3_tier_active(self) -> bool:
        """Whether S3 tier is currently activated."""
        return self._s3_active

    @property
    def hit_rate_ema(self) -> float:
        """Current EMA hit rate."""
        return self._hit_rate_ema
