"""Activity A (2026-05-26): ObjectCache S3 4-Tier Break-Even Scheduler — vLLM v1 Patch.

Extends vLLM 0.21.0's v1 Scheduler with S3 tier routing based on break-even
hit rate calculation and EMA hysteresis.

Integration approach: subclass vllm.v1.core.sched.scheduler.Scheduler,
overriding schedule() to apply pre-scheduling S3 tier routing decisions.

Key behaviors:
  - Break-even hit rate: hit_rate_breakeven = T_recompute / (T_recompute + T_s3)
  - EMA hit rate tracking: ema = γ * current + (1-γ) * ema
  - S3 activation: ema >= breakeven(context_len) + hysteresis_band
  - S3 deactivation: ema < breakeven(context_len) - hysteresis_band
  - Batch cap: max_s3_requests_per_batch S3-routed requests per schedule()
  - Scheduling order: non-S3 first (cache warming effect)

Tests gating:
  - Activity A test: test_s3_activation_above_threshold, test_breakeven_formula
  - TTFT overhead: S3 routing adds scheduling metadata only; actual S3 I/O is
    async and bounded by max_s3_requests_per_batch

vLLM version: 0.21.0
Activity: A-1 (ObjectCacheS3TierBreakEvenRoutingPolicy)
Source: src/scheduler/objectcache_s3_tier_router.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VLLMObjectCacheS3Config:
    """Configuration for the vLLM ObjectCache S3 tier scheduler patch."""
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


# ---------------------------------------------------------------------------
# S3 tier routing logic (standalone, composable)
# ---------------------------------------------------------------------------


class S3TierRoutingEngine:
    """Core break-even routing logic, extracted from ObjectCacheS3TierRouter.

    This is the standalone routing engine that can be composed into any
    vLLM scheduler variant.

    Maps to: ObjectCacheS3TierRouter (src/scheduler/objectcache_s3_tier_router.py)
    """

    def __init__(self, config: VLLMObjectCacheS3Config) -> None:
        self.config = config
        self._hit_rate_ema: float = 0.0
        self._s3_active: bool = config.s3_enabled_by_default
        self._breakeven_table: Dict[int, float] = dict(config.breakeven_table)

    def compute_breakeven_hit_rate(
        self,
        t_recompute_ms: float,
        t_s3_ms: float,
    ) -> float:
        """Break-even hit rate: T_recompute / (T_recompute + T_s3).

        At this hit rate, the expected savings from S3 cache retrieval
        exactly equal the added latency cost of the S3 round trip.
        Below this rate, recomputation is cheaper.
        """
        total = t_recompute_ms + t_s3_ms
        if total <= 0.0:
            return 0.0
        return t_recompute_ms / total

    def update_hit_rate_ema(self, current_hit_rate: float) -> None:
        """Update EMA and toggle S3 tier active state with hysteresis.

        EMA update: ema = γ * current + (1-γ) * ema
        Toggle logic uses median context length from config table.
        """
        gamma = self.config.ema_gamma
        self._hit_rate_ema = gamma * current_hit_rate + (1.0 - gamma) * self._hit_rate_ema

        # Use median context length for global toggle
        median_ctx = self.config.context_lengths[len(self.config.context_lengths) // 2]
        breakeven = self.get_breakeven_for_context(median_ctx)

        if self._hit_rate_ema >= breakeven + self.config.hysteresis_band:
            self._s3_active = True
        elif self._hit_rate_ema < breakeven - self.config.hysteresis_band:
            self._s3_active = False
        # Within hysteresis band: no state change (prevents oscillation)

    def get_breakeven_for_context(self, context_length: int) -> float:
        """Look up break-even hit rate for given context length.

        Uses nearest lower-bound entry from the break-even table.
        Falls back to the smallest table entry for very short contexts.
        """
        sorted_lengths = sorted(self._breakeven_table.keys())
        if not sorted_lengths:
            return 0.20  # sensible default

        best_key = sorted_lengths[0]
        for length in sorted_lengths:
            if length <= context_length:
                best_key = length
            else:
                break
        return self._breakeven_table[best_key]

    def should_use_s3_for_request(self, context_length: int) -> bool:
        """Per-request S3 routing decision based on EMA vs break-even + hysteresis."""
        if not self._s3_active:
            return False
        breakeven = self.get_breakeven_for_context(context_length)
        return self._hit_rate_ema >= breakeven + self.config.hysteresis_band

    @property
    def s3_tier_active(self) -> bool:
        return self._s3_active

    @property
    def hit_rate_ema(self) -> float:
        return self._hit_rate_ema


# ---------------------------------------------------------------------------
# vLLM Scheduler subclass
# ---------------------------------------------------------------------------


class ObjectCacheS3SchedulerMixin:
    """Mixin for vLLM Scheduler that adds S3 tier routing.

    Intended to be used with multiple inheritance:

        from vllm.v1.core.sched.scheduler import Scheduler
        class ObjectCacheS3Scheduler(ObjectCacheS3SchedulerMixin, Scheduler):
            ...

    Or installed as a monkey-patch via install_objectcache_s3_scheduler_hooks().

    The mixin wraps schedule() to apply S3 routing metadata to vLLM Requests
    before scheduling decisions, and re-orders them so non-S3 requests execute
    first (cache warming effect).
    """

    _s3_routing_engine: S3TierRoutingEngine

    def _s3_pre_schedule_hook(self) -> None:
        """Pre-schedule hook: annotate waiting requests with S3 routing intent.

        Called at the start of schedule() to:
          1. Count S3-eligible requests in self.waiting
          2. Apply max_s3_requests_per_batch cap
          3. Set request metadata s3_tier=True/False
          4. Move S3-routed requests to end of waiting queue (non-S3 first)

        The actual S3 prefetch is assumed to be handled by an external
        KVConnector (e.g. vllm.distributed.kv_transfer) not directly by
        the scheduler.
        """
        engine = self._s3_routing_engine
        if not hasattr(self, "waiting"):
            return

        # Annotate each waiting request
        s3_count = 0
        s3_requests = []
        non_s3_requests = []

        for req in list(getattr(self, "waiting", [])):
            ctx_len = getattr(req, "num_prompt_tokens", 0)
            should_s3 = (
                engine.should_use_s3_for_request(ctx_len)
                and s3_count < engine.config.max_s3_requests_per_batch
            )
            if should_s3:
                if not hasattr(req, "metadata") or req.metadata is None:
                    try:
                        req.metadata = {}
                    except AttributeError:
                        pass
                if hasattr(req, "metadata") and isinstance(req.metadata, dict):
                    req.metadata["s3_tier"] = True
                s3_requests.append(req)
                s3_count += 1
            else:
                if hasattr(req, "metadata") and isinstance(req.metadata, dict):
                    req.metadata.pop("s3_tier", None)
                non_s3_requests.append(req)

        # Reorder: non-S3 first
        if s3_requests:
            try:
                waiting = getattr(self, "waiting")
                if hasattr(waiting, "_queue"):
                    # RequestQueue backed by deque
                    from collections import deque
                    waiting._queue = deque(non_s3_requests + s3_requests)
                elif isinstance(waiting, list):
                    self.waiting = non_s3_requests + s3_requests  # type: ignore
            except Exception:
                pass

    def s3_update_hit_rate(self, hit_rate: float) -> None:
        """Update EMA hit rate after a batch completes.

        Should be called by the engine core after processing each batch.
        """
        if hasattr(self, "_s3_routing_engine"):
            self._s3_routing_engine.update_hit_rate_ema(hit_rate)

    @property
    def s3_tier_active(self) -> bool:
        if hasattr(self, "_s3_routing_engine"):
            return self._s3_routing_engine.s3_tier_active
        return False


def make_objectcache_s3_scheduler_class(
    base_scheduler_class: Any,
) -> Any:
    """Factory: create an ObjectCache S3-aware Scheduler subclass.

    Usage:
        from vllm.v1.core.sched.scheduler import Scheduler
        S3Scheduler = make_objectcache_s3_scheduler_class(Scheduler)
        scheduler = S3Scheduler(vllm_config, kv_cache_config, ...,
                                s3_config=VLLMObjectCacheS3Config())

    The returned class inherits all native Scheduler behaviour and adds:
        schedule() — calls _s3_pre_schedule_hook() before base schedule()
        s3_update_hit_rate(hit_rate) — updates EMA
        s3_tier_active — property

    Does NOT break any existing SchedulerInterface methods.
    """

    class ObjectCacheS3Scheduler(ObjectCacheS3SchedulerMixin, base_scheduler_class):
        """vLLM Scheduler with ObjectCache S3 4th-tier routing (Activity A-1)."""

        def __init__(
            self,
            *args: Any,
            s3_config: Optional[VLLMObjectCacheS3Config] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            if s3_config is None:
                s3_config = VLLMObjectCacheS3Config()
            self._s3_routing_engine = S3TierRoutingEngine(s3_config)

        def schedule(self) -> Any:
            """Override: apply S3 pre-routing, then call vLLM base schedule()."""
            self._s3_pre_schedule_hook()
            return super().schedule()

    ObjectCacheS3Scheduler.__name__ = "ObjectCacheS3Scheduler"
    ObjectCacheS3Scheduler.__qualname__ = "ObjectCacheS3Scheduler"
    return ObjectCacheS3Scheduler


def install_objectcache_s3_hooks(
    scheduler_instance: Any,
    s3_config: Optional[VLLMObjectCacheS3Config] = None,
) -> S3TierRoutingEngine:
    """Attach S3 routing engine to an existing vLLM Scheduler instance.

    Non-destructive monkey-patch. Returns the S3TierRoutingEngine for stats.

    After this call, scheduler_instance gains:
        _s3_routing_engine: S3TierRoutingEngine
        s3_update_hit_rate(hit_rate) method
        s3_tier_active property
        The existing schedule() is wrapped with _s3_pre_schedule_hook()
    """
    import types

    if s3_config is None:
        s3_config = VLLMObjectCacheS3Config()

    engine = S3TierRoutingEngine(s3_config)
    scheduler_instance._s3_routing_engine = engine  # type: ignore[attr-defined]

    # Wrap schedule() with pre-hook
    original_schedule = scheduler_instance.schedule

    def _patched_schedule(self_or_none=None):
        # Bound method or plain callable
        inst = scheduler_instance
        # Apply S3 routing pre-hook
        mixin = ObjectCacheS3SchedulerMixin
        mixin._s3_pre_schedule_hook(inst)
        return original_schedule()

    scheduler_instance.schedule = _patched_schedule  # type: ignore[method-assign]

    # Attach helper methods
    scheduler_instance.s3_update_hit_rate = (  # type: ignore[attr-defined]
        types.MethodType(
            lambda self, hr: self._s3_routing_engine.update_hit_rate_ema(hr),
            scheduler_instance,
        )
    )

    return engine
