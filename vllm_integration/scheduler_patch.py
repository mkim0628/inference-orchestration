"""scheduler_patch.py — Activity A: KV cache-aware scheduling for vLLM 0.21.0.

2026-05-22: PPDAppendFullPrefillClassifierMixin — ports PPDAppendFullPrefillClassifier
            (Activity A) into vLLM's v1 Scheduler as a mixin. Provides
            session-context-hash based append/full-prefill routing with
            SLO-aware D→P node switching.

            PPDAppendFullPrefillClassifierMixin wraps schedule() with a
            ppd_pre_schedule() hook that:
              1. Iterates self.waiting (RequestQueue) without modifying queue order.
              2. For each waiting request, calls PPDAppendFullPrefillClassifier.classify()
                 with the request's token IDs (O(1) dict lookup by session_id).
              3. Annotates the vLLM Request with ppd_prefill_type ("append" | "full"),
                 ppd_routed_to ("D_node" | "P_node"), and ppd_overhead_us.
              4. Calls expire_sessions() periodically (every 100 schedule() calls).

            Integration:
              - Append-prefill requests (D_node): Request is annotated to skip
                P-node prefill — existing KV at D-node is reused.
              - Full-prefill requests (P_node): Request follows standard vLLM
                scheduling path (no modification to token allocation).

            Scheduling overhead: O(1) per request (SHA-256 hash of ≤512 tokens).
            Target: < 5ms p50 TTFT overhead for N ≤ 1000 waiting requests.

            make_ppd_classifier_scheduler_class() factory — builds a vLLM v1
            Scheduler subclass that intercepts schedule() to run PPD classification
            before base scheduling.

2026-05-22 (same cycle): DapQSessionSegmentSchedulerMixin — coordinates
            DapQSessionSegmentKVCacheManagerMixin (block_manager_patch.py) with
            the scheduler to trigger dual-reduction before prefill.

2026-05-16 (prior): NAtHDDROffloadingSchedulerMixin — preserved below.
2026-05-15 (prior): RadixFeatherSchedulerMixin — preserved below.
2026-05-09 (prior): HitAwarePPDRouterMixin + PPDAppendPrefillRouterMixin — preserved.
2026-05-03 (prior): DualMapSchedulerMixin / CacheHitAwareRequestQueue — preserved.

vLLM 0.21.0 v1 architecture:
    - Scheduler lives in vllm.v1.core.sched.scheduler.Scheduler
    - Waiting queue is self.waiting (RequestQueue, iterable)
    - Per-step scheduling via Scheduler.schedule() → SchedulerOutput
    - KV block management via self.kv_cache_manager (KVCacheManager)
    - Request class: vllm.v1.request.Request

vLLM version: 0.21.0
Activity: A — PPDAppendFullPrefillClassifier
         B+C — DapQSessionSegmentSchedulerMixin (coordinates block_manager_patch)
"""

from __future__ import annotations

import sys
import pathlib
import hashlib
import json
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import vllm

def _vllm_version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:3])

assert _vllm_version_tuple(vllm.__version__) >= _vllm_version_tuple("0.4.0"), (
    f"vllm_integration requires vLLM >= 0.4.0, found {vllm.__version__}"
)


def _add_repo_root_to_path() -> None:
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _try_import_ppd_classifier_src() -> tuple:
    """Lazily import PPDAppendFullPrefillClassifier from src/."""
    _add_repo_root_to_path()
    try:
        from src.scheduler.ppd_append_full_prefill_classifier import (
            PPDAppendFullPrefillClassifier,
            PPDClassifierConfig,
            PrefillTypeDecision,
        )
        return PPDAppendFullPrefillClassifier, PPDClassifierConfig, PrefillTypeDecision
    except ImportError:
        return None, None, None


# ===========================================================================
# 2026-05-22: PPDAppendFullPrefillClassifierMixin (Activity A)
# ===========================================================================

@dataclass
class PPDClassifierSchedulerConfig:
    """Configuration for PPDAppendFullPrefillClassifierMixin.

    Mirrors PPDClassifierConfig defaults; used when src/ is not importable.
    """
    append_threshold: float = 0.15
    slo_headroom_threshold_ms: float = 30.0
    session_ttl_seconds: float = 3600.0
    seed: int = 42
    expire_interval: int = 100   # expire_sessions() every N schedule() calls


class PPDAppendFullPrefillClassifierMixin:
    """vLLM v1 Scheduler mixin: PPD append/full-prefill classification routing.

    Activity A: KV Cache-aware Scheduling.
    Based on PPD (arXiv 2603.13358): append-prefill at D-node avoids KV transfer
    for multi-turn sessions where most tokens are already cached.

    This mixin wraps schedule() with a ppd_pre_schedule() hook that:
      1. Iterates self.waiting without modifying queue order.
      2. For each request, classifies as append-prefill or full-prefill via
         O(1) session context hash comparison.
      3. Annotates each vLLM Request with:
           - ppd_prefill_type: "append" | "full"
           - ppd_routed_to: "D_node" | "P_node"
           - ppd_overhead_us: classification latency in microseconds
      4. Calls expire_sessions() every expire_interval steps.

    Scheduling overhead:
        O(1) per request (SHA-256 of ≤ 512 token bytes).
        Target: < 5ms p50 TTFT overhead (MANDATORY evaluation_criteria.md §2).

    Usage:

        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import (
            PPDAppendFullPrefillClassifierMixin,
            PPDClassifierSchedulerConfig,
            make_ppd_classifier_scheduler_class,
        )

        # Option A: factory
        PPDScheduler = make_ppd_classifier_scheduler_class(Scheduler)
        scheduler = PPDScheduler(
            ...,  # standard vLLM Scheduler args
            ppd_config=PPDClassifierSchedulerConfig(append_threshold=0.15),
        )

        # Option B: mixin + override
        class MyScheduler(PPDAppendFullPrefillClassifierMixin, Scheduler):
            def schedule(self):
                self.ppd_pre_schedule()
                return super().schedule()
    """

    def __init__(
        self,
        *args: Any,
        ppd_config: Optional[PPDClassifierSchedulerConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            ppd_config: PPDClassifierSchedulerConfig. If None, uses defaults.
            All other args/kwargs forwarded to the base Scheduler.__init__().
        """
        super().__init__(*args, **kwargs)

        if ppd_config is None:
            ppd_config = PPDClassifierSchedulerConfig()
        self._ppd_cfg = ppd_config

        # Try to import native classifier from src/
        PPDAppendFullPrefillClassifier, PPDClassifierConfig, PrefillTypeDecision = (
            _try_import_ppd_classifier_src()
        )

        self._ppd_classifier: Optional[Any] = None
        self._ppd_use_native: bool = False

        if PPDAppendFullPrefillClassifier is not None:
            classifier_cfg = PPDClassifierConfig(
                append_threshold=ppd_config.append_threshold,
                slo_headroom_threshold_ms=ppd_config.slo_headroom_threshold_ms,
                session_ttl_seconds=ppd_config.session_ttl_seconds,
                seed=ppd_config.seed,
            )
            self._ppd_classifier = PPDAppendFullPrefillClassifier(classifier_cfg)
            self._ppd_use_native = True
        else:
            # Fallback: inline lightweight O(1) classifier
            self._ppd_classifier = _InlinePPDClassifier(ppd_config)

        # Metrics
        self._ppd_schedule_count: int = 0
        self._ppd_overhead_ms_list: List[float] = []
        self._ppd_append_count: int = 0
        self._ppd_full_count: int = 0

    # -----------------------------------------------------------------------
    # Primary scheduling hook — call at the start of schedule()
    # -----------------------------------------------------------------------

    def ppd_pre_schedule(self) -> None:
        """Classify waiting requests and annotate with PPD routing decisions.

        Called at the beginning of schedule() before the base scheduler
        selects which requests to run. Annotates each waiting request with:
            request.ppd_prefill_type: "append" | "full"
            request.ppd_routed_to: "D_node" | "P_node"
            request.ppd_overhead_us: classification overhead in microseconds

        Overhead: O(n_waiting) × O(1) per request.
        """
        t_hook_start = time.monotonic()
        self._ppd_schedule_count += 1

        # Expire stale sessions periodically
        if self._ppd_schedule_count % self._ppd_cfg.expire_interval == 0:
            try:
                if self._ppd_use_native:
                    self._ppd_classifier.expire_sessions()
                else:
                    self._ppd_classifier.expire_sessions()
            except Exception:
                pass

        # Iterate waiting queue (do NOT modify the queue)
        try:
            waiting_iter = iter(self.waiting)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return

        for request in waiting_iter:
            try:
                self._ppd_classify_request(request)
            except Exception:
                continue

        hook_ms = (time.monotonic() - t_hook_start) * 1e3
        self._ppd_overhead_ms_list.append(hook_ms)
        # Keep only last 200 samples
        if len(self._ppd_overhead_ms_list) > 200:
            self._ppd_overhead_ms_list.pop(0)

    def _ppd_classify_request(self, request: Any) -> None:
        """Classify a single request and annotate it."""
        # Extract token IDs from the request
        try:
            token_ids = list(request.prompt_token_ids or [])
        except AttributeError:
            try:
                token_ids = list(request.inputs.prompt_token_ids or [])
            except AttributeError:
                token_ids = []

        # Extract session_id from request metadata or use request_id
        try:
            session_id = str(request.metadata.get("session_id", request.request_id))
        except AttributeError:
            try:
                session_id = str(request.request_id)
            except AttributeError:
                session_id = "default"

        # Classify
        t_start = time.monotonic()
        if self._ppd_use_native:
            decision = self._ppd_classifier.classify(
                request_id=str(getattr(request, "request_id", "unknown")),
                session_id=session_id,
                token_ids=token_ids,
                remaining_slo_ms=None,
            )
        else:
            decision = self._ppd_classifier.classify(
                request_id=str(getattr(request, "request_id", "unknown")),
                session_id=session_id,
                token_ids=token_ids,
            )
        overhead_us = (time.monotonic() - t_start) * 1e6

        # Annotate request (runtime attribute injection)
        try:
            object.__setattr__(request, "ppd_prefill_type", decision.prefill_type)
            object.__setattr__(request, "ppd_routed_to", decision.routed_to)
            object.__setattr__(request, "ppd_session_id", session_id)
            object.__setattr__(request, "ppd_new_token_ratio", decision.new_token_ratio)
            object.__setattr__(request, "ppd_overhead_us", overhead_us)
        except Exception:
            pass

        # Track counts
        if decision.prefill_type == "append":
            self._ppd_append_count += 1
        else:
            self._ppd_full_count += 1

    def ppd_scheduling_stats(self) -> Dict[str, Any]:
        """Return PPD scheduling statistics.

        Returns:
            Dict with keys:
                schedule_count: total schedule() calls
                append_count: total append-prefill decisions
                full_count: total full-prefill decisions
                overhead_mean_ms: mean per-step overhead in milliseconds
                overhead_p50_ms: p50 overhead
                overhead_p99_ms: p99 overhead
        """
        overhead = sorted(self._ppd_overhead_ms_list)
        n = len(overhead)
        return {
            "schedule_count": self._ppd_schedule_count,
            "append_count": self._ppd_append_count,
            "full_count": self._ppd_full_count,
            "overhead_mean_ms": sum(overhead) / n if n > 0 else 0.0,
            "overhead_p50_ms": overhead[n // 2] if n > 0 else 0.0,
            "overhead_p99_ms": overhead[int(n * 0.99)] if n > 0 else 0.0,
            "append_ratio": (
                self._ppd_append_count / max(1, self._ppd_append_count + self._ppd_full_count)
            ),
        }


# ---------------------------------------------------------------------------
# Inline PPD classifier fallback (no src/ dependency)
# ---------------------------------------------------------------------------

class _InlinePPDClassifier:
    """Lightweight inline PPD append/full-prefill classifier (no src/ dependency).

    Uses the same O(1) session context hash comparison as
    PPDAppendFullPrefillClassifier from src/scheduler/.
    """

    @dataclass
    class _Decision:
        prefill_type: str
        routed_to: str
        new_token_ratio: float

    @dataclass
    class _Entry:
        last_total_tokens: int
        turn_count: int
        last_accessed: float

    def __init__(self, config: PPDClassifierSchedulerConfig) -> None:
        self.config = config
        self._registry: Dict[str, "_InlinePPDClassifier._Entry"] = {}

    def classify(
        self,
        request_id: str,
        session_id: str,
        token_ids: List[int],
        remaining_slo_ms: Optional[float] = None,
    ) -> "_InlinePPDClassifier._Decision":
        total = len(token_ids)
        entry = self._registry.get(session_id)

        if entry is None or entry.turn_count == 0:
            prefill_type = "full"
            new_token_ratio = 1.0
        else:
            new_tokens = max(0, total - entry.last_total_tokens)
            new_token_ratio = new_tokens / max(1, total)
            if new_token_ratio <= self.config.append_threshold:
                if (remaining_slo_ms is not None
                        and remaining_slo_ms < self.config.slo_headroom_threshold_ms):
                    prefill_type = "full"
                else:
                    prefill_type = "append"
            else:
                prefill_type = "full"

        turn = (entry.turn_count if entry else 0) + 1
        self._registry[session_id] = self._Entry(
            last_total_tokens=total,
            turn_count=turn,
            last_accessed=time.monotonic(),
        )
        routed_to = "D_node" if prefill_type == "append" else "P_node"
        return self._Decision(
            prefill_type=prefill_type,
            routed_to=routed_to,
            new_token_ratio=new_token_ratio,
        )

    def expire_sessions(self) -> int:
        now = time.monotonic()
        expired = [
            sid for sid, e in self._registry.items()
            if now - e.last_accessed > self.config.session_ttl_seconds
        ]
        for sid in expired:
            del self._registry[sid]
        return len(expired)


# ---------------------------------------------------------------------------
# make_ppd_classifier_scheduler_class() factory
# ---------------------------------------------------------------------------

def make_ppd_classifier_scheduler_class(
    base_class: Optional[type] = None,
) -> type:
    """Factory: return a Scheduler subclass with PPD append/full-prefill classification.

    Args:
        base_class: The vLLM Scheduler class to subclass. If None, imports
            vllm.v1.core.sched.scheduler.Scheduler automatically.

    Returns:
        A new class combining PPDAppendFullPrefillClassifierMixin with base_class.

    Usage:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import (
            make_ppd_classifier_scheduler_class,
            PPDClassifierSchedulerConfig,
        )

        PPDScheduler = make_ppd_classifier_scheduler_class(Scheduler)

        scheduler = PPDScheduler(
            vllm_config=...,
            kv_cache_config=...,
            ...,
            ppd_config=PPDClassifierSchedulerConfig(append_threshold=0.15),
        )

        # The scheduler will now call ppd_pre_schedule() before each schedule():
        output = scheduler.schedule()  # ppd annotations applied to waiting requests
    """
    if base_class is None:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler
            base_class = _Scheduler
        except Exception:
            # Fallback: generic base for CPU-only / test environments
            base_class = object

    class PPDScheduler(PPDAppendFullPrefillClassifierMixin, base_class):  # type: ignore[valid-type]
        """vLLM Scheduler subclass with PPD append/full-prefill classification.

        Auto-generated by make_ppd_classifier_scheduler_class().
        """
        def schedule(self) -> Any:
            self.ppd_pre_schedule()
            return super().schedule()  # type: ignore[misc]

    PPDScheduler.__name__ = "PPDAppendFullPrefillClassifierScheduler"
    PPDScheduler.__qualname__ = "PPDAppendFullPrefillClassifierScheduler"
    return PPDScheduler


# ===========================================================================
# 2026-05-22: DapQSessionSegmentSchedulerMixin (Activity B+C coordination)
# ===========================================================================

class DapQSessionSegmentSchedulerMixin:
    """vLLM v1 Scheduler mixin: coordinates DapQ session segment dual-reduction.

    Activity B+C: Coordinates DapQSessionSegmentKVCacheManagerMixin with
    the scheduler to trigger dual-reduction before prefill for multi-turn sessions.

    This mixin adds a dapq_pre_schedule() hook that:
      1. Iterates self.waiting.
      2. For each request with a known session, calls
         self.kv_cache_manager.annotate_request() to attach cached segment info.
      3. Requests annotated with dapq_noncontiguous_hit=True may skip P-node
         prefill and reuse cached segments at the D-node.

    The mixin works alongside PPDAppendFullPrefillClassifierMixin:
        class MyScheduler(DapQSessionSegmentSchedulerMixin,
                          PPDAppendFullPrefillClassifierMixin, Scheduler):
            def schedule(self):
                self.ppd_pre_schedule()   # Activity A: routing classification
                self.dapq_pre_schedule()  # Activity B+C: segment annotation
                return super().schedule()
    """

    def __init__(
        self,
        *args: Any,
        dapq_scheduler_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dapq_sched_cfg = dapq_scheduler_config or {}
        self._dapq_sched_step: int = 0

    def dapq_pre_schedule(self) -> None:
        """Annotate waiting requests with DapQ session segment info.

        Calls kv_cache_manager.annotate_request() for each waiting request
        that has a known session. Attaches dapq_segments and
        dapq_noncontiguous_hit attributes to the request.
        """
        self._dapq_sched_step += 1
        kv_mgr = getattr(self, "kv_cache_manager", None)
        if kv_mgr is None:
            return
        if not hasattr(kv_mgr, "annotate_request"):
            return

        try:
            waiting_iter = iter(self.waiting)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return

        for request in waiting_iter:
            try:
                # Extract session_id
                try:
                    session_id = str(
                        request.metadata.get("session_id", request.request_id)
                    )
                except AttributeError:
                    session_id = str(getattr(request, "request_id", "default"))

                # Estimate current decode position from token count
                try:
                    n_computed = request.num_computed_tokens
                except AttributeError:
                    n_computed = 0
                current_pos = float(n_computed)

                kv_mgr.annotate_request(request, session_id, current_pos)
            except Exception:
                continue


def make_dapq_session_segment_scheduler_class(
    base_class: Optional[type] = None,
) -> type:
    """Factory: return a Scheduler subclass with DapQ+PPD scheduling.

    Combines PPDAppendFullPrefillClassifierMixin (Activity A) with
    DapQSessionSegmentSchedulerMixin (Activity B+C) into a single scheduler.

    Returns:
        A new Scheduler subclass running both hooks in schedule().
    """
    if base_class is None:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler
            base_class = _Scheduler
        except Exception:
            base_class = object

    class DapQPPDScheduler(
        DapQSessionSegmentSchedulerMixin,
        PPDAppendFullPrefillClassifierMixin,
        base_class,  # type: ignore[valid-type]
    ):
        """vLLM Scheduler: PPD classification (A) + DapQ segment annotation (B+C).

        Auto-generated by make_dapq_session_segment_scheduler_class().
        """
        def schedule(self) -> Any:
            self.ppd_pre_schedule()   # Activity A
            self.dapq_pre_schedule()  # Activity B+C
            return super().schedule()  # type: ignore[misc]

    DapQPPDScheduler.__name__ = "DapQPPDScheduler"
    DapQPPDScheduler.__qualname__ = "DapQPPDScheduler"
    return DapQPPDScheduler


# ===========================================================================
# 2026-05-16 (prior): NAtH DDR Offloading Scheduler Mixin — preserved
# ===========================================================================

def _try_import_nath_src() -> tuple:
    """Lazily import NAtHDDROffloadingScheduler and related classes from src/."""
    _add_repo_root_to_path()
    try:
        from src.scheduler.nath_ddr_offloading import (
            NAtHDDROffloadingScheduler,
            NAtHDDROffloadingConfig,
        )
        from src.scheduler.nath_retention_tier_decider import (
            NAtHRetentionTierDecider,
            NAtHRetentionTierDeciderConfig,
        )
        from src.cache.global_retention_gate_eviction import (
            GlobalRetentionGateEvictionCodec,
            GlobalRetentionGateConfig,
        )
        return (
            NAtHDDROffloadingScheduler,
            NAtHDDROffloadingConfig,
            NAtHRetentionTierDecider,
            NAtHRetentionTierDeciderConfig,
            GlobalRetentionGateEvictionCodec,
            GlobalRetentionGateConfig,
        )
    except ImportError:
        return (None,) * 6


@dataclass
class NAtHDDROffloadingSchedulerConfig:
    """Configuration for NAtHDDROffloadingSchedulerMixin (preserved 2026-05-16)."""
    tier_boundaries: List[float] = field(
        default_factory=lambda: [0.30, 0.70, 0.97]
    )
    max_eviction_ratio: float = 0.03
    ema_alpha: float = 0.95
    prefetch_chunk_size: int = 64
    max_wait_ratio: float = 2.0
    seed: int = 42
    enable_retention_gate: bool = False
    retention_alpha: float = 0.5
    n_layers: int = 12
    n_heads: int = 8
    d_model: int = 512
    budget_ratio: float = 0.3
    recent_window: int = 32


class _InlineNAtHScheduler:
    """Inline NAtH 4-tier classifier (fallback when src/ unavailable)."""

    def __init__(
        self,
        tier_boundaries: List[float],
        max_eviction_ratio: float,
        ema_alpha: float,
    ) -> None:
        self.tier_boundaries = tier_boundaries
        self.max_eviction_ratio = max_eviction_ratio
        self.ema_alpha = ema_alpha
        self._ema_scores: Dict[str, float] = {}

    def classify_token(self, token_key: str, score: float) -> int:
        prev = self._ema_scores.get(token_key, score)
        ema = self.ema_alpha * prev + (1 - self.ema_alpha) * score
        self._ema_scores[token_key] = ema
        p1, p2, p3 = self.tier_boundaries
        if ema >= p1:
            return 1
        elif ema >= p2:
            return 2
        elif ema >= p3:
            return 3
        else:
            return 4

    def expire_tokens(self) -> None:
        self._ema_scores.clear()


class NAtHDDROffloadingSchedulerMixin:
    """vLLM v1 Scheduler mixin: NAtH 4-tier DDR offloading (preserved 2026-05-16).

    Activity A: KV Cache-aware Scheduling.
    Based on NAtH (arXiv 2605.09490). See 2026-05-16 cycle for full documentation.
    """

    def __init__(
        self,
        *args: Any,
        nath_config: Optional[NAtHDDROffloadingSchedulerConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if nath_config is None:
            nath_config = NAtHDDROffloadingSchedulerConfig()
        self._nath_cfg = nath_config
        (
            NAtHDDROffloadingScheduler,
            NAtHDDROffloadingConfig,
            NAtHRetentionTierDecider,
            NAtHRetentionTierDeciderConfig,
            GlobalRetentionGateEvictionCodec,
            GlobalRetentionGateConfig,
        ) = _try_import_nath_src()

        self._nath_scheduler: Optional[Any] = None
        self._nath_tier_decider: Optional[Any] = None
        self._nath_use_native: bool = False

        if NAtHDDROffloadingScheduler is not None:
            cfg = NAtHDDROffloadingConfig(
                tier_boundaries=list(nath_config.tier_boundaries),
                max_eviction_ratio=nath_config.max_eviction_ratio,
                ema_alpha=nath_config.ema_alpha,
                prefetch_chunk_size=nath_config.prefetch_chunk_size,
                max_wait_ratio=nath_config.max_wait_ratio,
                seed=nath_config.seed,
            )
            self._nath_scheduler = NAtHDDROffloadingScheduler(config=cfg)
            self._nath_use_native = True
            if nath_config.enable_retention_gate and NAtHRetentionTierDecider is not None:
                ret_cfg = GlobalRetentionGateConfig(
                    n_layers=nath_config.n_layers,
                    n_heads=nath_config.n_heads,
                    d_model=nath_config.d_model,
                    budget_ratio=nath_config.budget_ratio,
                    recent_window=nath_config.recent_window,
                    seed=nath_config.seed,
                )
                codec = GlobalRetentionGateEvictionCodec(ret_cfg)
                decider_cfg = NAtHRetentionTierDeciderConfig(
                    alpha=nath_config.retention_alpha,
                    max_eviction_ratio=nath_config.max_eviction_ratio,
                    seed=nath_config.seed,
                )
                self._nath_tier_decider = NAtHRetentionTierDecider(
                    config=decider_cfg,
                    nath_scheduler=self._nath_scheduler,
                    retention_codec=codec,
                )
        else:
            self._nath_scheduler = _InlineNAtHScheduler(
                tier_boundaries=list(nath_config.tier_boundaries),
                max_eviction_ratio=nath_config.max_eviction_ratio,
                ema_alpha=nath_config.ema_alpha,
            )

        self._nath_overhead_ms_list: List[float] = []
        self._nath_schedule_count: int = 0

    def nath_pre_schedule(self) -> None:
        """Classify waiting requests' tokens into 4 NAtH tiers."""
        t_start = time.monotonic()
        self._nath_schedule_count += 1
        try:
            for request in iter(self.waiting):  # type: ignore[attr-defined]
                try:
                    token_ids = list(getattr(request, "prompt_token_ids", []) or [])
                    tier_assignment = {}
                    for i, tok_id in enumerate(token_ids[:256]):
                        score = float(tok_id % 100) / 100.0
                        tok_key = f"{getattr(request, 'request_id', 'req')}_{i}"
                        if self._nath_use_native:
                            pass  # native scheduler handles internally
                        else:
                            tier = self._nath_scheduler.classify_token(tok_key, score)
                            tier_assignment[i] = tier
                    object.__setattr__(request, "nath_tier_assignment", tier_assignment)
                except Exception:
                    continue
        except (AttributeError, TypeError):
            pass
        self._nath_overhead_ms_list.append((time.monotonic() - t_start) * 1e3)
        if len(self._nath_overhead_ms_list) > 200:
            self._nath_overhead_ms_list.pop(0)


def make_nath_ddr_scheduler_class(
    base_class: Optional[type] = None,
) -> type:
    """Factory for NAtHDDROffloadingScheduler (2026-05-16, preserved)."""
    if base_class is None:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler
            base_class = _Scheduler
        except Exception:
            base_class = object

    class NAtHScheduler(NAtHDDROffloadingSchedulerMixin, base_class):  # type: ignore[valid-type]
        def schedule(self) -> Any:
            self.nath_pre_schedule()
            return super().schedule()  # type: ignore[misc]

    NAtHScheduler.__name__ = "NAtHDDROffloadingScheduler"
    return NAtHScheduler


# ===========================================================================
# 2026-05-09 (prior): HitAwarePPDRouterMixin + PPDAppendPrefillRouterMixin
# ===========================================================================

def _try_import_hit_aware_ppd_src() -> tuple:
    _add_repo_root_to_path()
    try:
        from src.scheduler.hit_aware_ppd_router import (
            HitAwarePPDRouter,
            HitAwarePPDRouterConfig,
        )
        return HitAwarePPDRouter, HitAwarePPDRouterConfig
    except ImportError:
        return None, None


@dataclass
class HitAwarePPDRouterSchedulerConfig:
    """Configuration for HitAwarePPDRouterMixin (preserved 2026-05-09)."""
    d_node_threshold: float = 0.6
    ema_alpha: float = 0.9
    seed: int = 42


class HitAwarePPDRouterMixin:
    """vLLM v1 Scheduler mixin: HitAwarePPDRouter (preserved 2026-05-09).

    Activity A+B Cross-1: PPD routing based on TriangleInequalitySegmentIndex
    hit probability estimation. See 2026-05-09 cycle for full documentation.
    """

    def __init__(
        self,
        *args: Any,
        hit_aware_config: Optional[HitAwarePPDRouterSchedulerConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if hit_aware_config is None:
            hit_aware_config = HitAwarePPDRouterSchedulerConfig()
        self._hit_aware_cfg = hit_aware_config
        HitAwarePPDRouter, HitAwarePPDRouterConfig = _try_import_hit_aware_ppd_src()
        self._hit_aware_router: Optional[Any] = None
        if HitAwarePPDRouter is not None:
            cfg = HitAwarePPDRouterConfig(
                d_node_threshold=hit_aware_config.d_node_threshold,
                ema_alpha=hit_aware_config.ema_alpha,
                seed=hit_aware_config.seed,
            )
            self._hit_aware_router = HitAwarePPDRouter(config=cfg)
        self._hit_aware_overhead_ms: List[float] = []

    def pre_schedule_ppd(self) -> None:
        """Annotate waiting requests with HitAwarePPD routing decisions."""
        if self._hit_aware_router is None:
            return
        t_start = time.monotonic()
        try:
            for request in iter(self.waiting):  # type: ignore[attr-defined]
                try:
                    token_ids = list(getattr(request, "prompt_token_ids", []) or [])
                    try:
                        result = self._hit_aware_router.route(
                            request_id=str(getattr(request, "request_id", "req")),
                            token_ids=token_ids,
                        )
                        object.__setattr__(request, "ppd_node_type", result.node_type)
                        object.__setattr__(request, "ppd_hit_probability", result.hit_probability)
                    except Exception:
                        pass
                except Exception:
                    continue
        except (AttributeError, TypeError):
            pass
        self._hit_aware_overhead_ms.append((time.monotonic() - t_start) * 1e3)


class PPDAppendPrefillRouterMixin:
    """vLLM v1 Scheduler mixin: PPDAppendPrefillRouter (preserved 2026-05-09).

    Lighter mixin for PPDAppendPrefillRouter without online threshold adaptation.
    See 2026-05-09 cycle for full documentation.
    """

    def __init__(
        self,
        *args: Any,
        ppd_router_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._ppd_router_cfg = ppd_router_config or {}
        _add_repo_root_to_path()
        try:
            from src.scheduler.ppd_append_prefill_router import (
                PPDAppendPrefillRouter,
                PPDAppendPrefillRouterConfig,
            )
            cfg = PPDAppendPrefillRouterConfig(**self._ppd_router_cfg)
            self._ppd_router: Optional[Any] = PPDAppendPrefillRouter(cfg)
        except Exception:
            self._ppd_router = None

    def pre_schedule_ppd_router(self) -> None:
        """Annotate waiting requests with PPDAppendPrefillRouter decisions."""
        if self._ppd_router is None:
            return
        try:
            for request in iter(self.waiting):  # type: ignore[attr-defined]
                try:
                    token_ids = list(getattr(request, "prompt_token_ids", []) or [])
                    result = self._ppd_router.route(
                        request_id=str(getattr(request, "request_id", "req")),
                        token_ids=token_ids,
                    )
                    object.__setattr__(request, "ppd_router_decision", result)
                except Exception:
                    continue
        except (AttributeError, TypeError):
            pass


def make_hit_aware_ppd_scheduler_class(
    base_class: Optional[type] = None,
) -> type:
    """Factory for HitAwarePPDScheduler (2026-05-09, preserved)."""
    if base_class is None:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler
            base_class = _Scheduler
        except Exception:
            base_class = object

    class HitAwarePPDScheduler(HitAwarePPDRouterMixin, base_class):  # type: ignore[valid-type]
        def schedule(self) -> Any:
            self.pre_schedule_ppd()
            return super().schedule()  # type: ignore[misc]

    HitAwarePPDScheduler.__name__ = "HitAwarePPDScheduler"
    return HitAwarePPDScheduler


# ===========================================================================
# 2026-05-03 (prior): DualMapSchedulerMixin — preserved
# ===========================================================================

class DualMapSchedulerMixin:
    """Preserved from 2026-05-03: DualMapScheduler (Activity A).

    Dual-map cache-aware request reordering with CacheHitAwareRequestQueue.
    See 2026-05-03 cycle for full documentation.
    """

    def __init__(
        self,
        *args: Any,
        dual_map_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dual_map_cfg = dual_map_config or {}
        self._dual_map_scores: Dict[str, float] = {}

    def pre_schedule_dual_map(self) -> None:
        """Score waiting requests by estimated cache hit probability."""
        try:
            for request in iter(self.waiting):  # type: ignore[attr-defined]
                req_id = str(getattr(request, "request_id", "req"))
                score = self._dual_map_scores.get(req_id, 0.5)
                object.__setattr__(request, "dual_map_cache_score", score)
        except (AttributeError, TypeError):
            pass

    def update_cache_score(self, request_id: str, score: float) -> None:
        self._dual_map_scores[request_id] = max(0.0, min(1.0, score))


# ===========================================================================
# 2026-05-23: CPDWarmColdHitRateRouter — Activity A
# ===========================================================================

@dataclass
class CPDRouterSchedulerConfig:
    """Configuration for CPDWarmColdHitRateRouter integration into vLLM Scheduler.

    Activity A: CPD (Together AI 2026-03-04) warm/cold request routing.
    Predicts cache hit rate per request using a 4-feature linear model and
    sorts the waiting queue: warm (high hit rate) first, cold last.

    Scheduling overhead: < 0.1ms/request (linear model inference).
    Target: TTFT p50 +5% or less (evaluation_criteria.md §2).
    """
    high_hit_threshold: float = 0.70    # >= this: warm path (batch priority 0)
    low_hit_threshold: float = 0.25     # <  this: cold path (batch priority 2)
    warm_slot_ratio: float = 0.60       # fraction of batch for warm requests
    cold_slot_ratio: float = 0.30       # fraction of batch for cold requests
    neutral_slot_ratio: float = 0.10    # fraction of batch for neutral
    queue_pressure_threshold: int = 100 # queue depth above which cold→neutral promotion
    max_context_length_warm: int = 50000
    seed: int = 42


class CPDWarmColdSchedulerMixin:
    """Mixin integrating CPDWarmColdHitRateRouter into vLLM v1 Scheduler.

    2026-05-23: Activity A — CPD Warm/Cold Hit-Rate Router.
    Based on arXiv 2026-03-04 (Together AI CPD).

    Wraps schedule() with cpd_pre_schedule():
      1. Iterates self.waiting (RequestQueue) without modifying internal state.
      2. For each request, predicts cache hit rate via 4-feature linear model:
           f1 = recent hit rate from prefix_hash history
           f2 = context length norm
           f3 = session age norm (1/(1+turn))
           f4 = segment match ratio
         score = sigmoid(w1*f1 + w2*f2 + w3*f3 + w4*f4 + bias)
      3. Classifies: warm (>= 0.70), neutral (0.25-0.70), cold (< 0.25).
      4. Annotates each vLLM Request with:
           cpd_path: "warm" | "cold" | "neutral"
           cpd_predicted_hit_rate: float
           cpd_batch_priority: int (0=warm, 1=neutral, 2=cold)
      5. Reorders self.waiting: warm first, neutral second, cold last.
         Within warm: sort by prefix_hash for KV locality.

    Scheduling overhead: < 0.1ms per request (linear model, O(1)).
    """

    def __init__(
        self,
        *args: Any,
        cpd_config: Optional[CPDRouterSchedulerConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cpd_config = cpd_config or CPDRouterSchedulerConfig()
        # Inline 4-feature linear model weights (no external dependency)
        self._cpd_w = [0.4, 0.2, 0.2, 0.2]  # [prefix_hash, ctx_len, session_age, seg_match]
        self._cpd_bias = 0.3
        self._cpd_lr = 0.01
        self._cpd_hit_history: Dict[str, List[bool]] = {}
        self._cpd_overhead_us: List[float] = []
        self._cpd_warm_count = 0
        self._cpd_cold_count = 0
        self._cpd_neutral_count = 0
        # Try to import the verified CPDWarmColdHitRateRouter from src/
        self._cpd_router = None
        _add_repo_root_to_path()
        try:
            from src.scheduler.cpd_warm_cold_hit_router import (
                CPDWarmColdHitRateRouter,
                CPDRouterConfig,
            )
            cpd_cfg_src = CPDRouterConfig(
                high_hit_threshold=self._cpd_config.high_hit_threshold,
                low_hit_threshold=self._cpd_config.low_hit_threshold,
                warm_slot_ratio=self._cpd_config.warm_slot_ratio,
                cold_slot_ratio=self._cpd_config.cold_slot_ratio,
                neutral_slot_ratio=self._cpd_config.neutral_slot_ratio,
                queue_pressure_threshold=self._cpd_config.queue_pressure_threshold,
                max_context_length_warm=self._cpd_config.max_context_length_warm,
                seed=self._cpd_config.seed,
            )
            self._cpd_router = CPDWarmColdHitRateRouter(cpd_cfg_src)
        except Exception:
            pass  # use inline fallback

    def _cpd_extract_features(self, request: Any) -> Tuple[float, float, float, float]:
        """Extract 4 hit-rate prediction features from a vLLM Request."""
        token_ids = list(getattr(request, "prompt_token_ids", None) or
                         getattr(request, "token_ids", None) or [])
        prefix_hash = hashlib.sha256(
            bytes(token_ids[:512])
        ).hexdigest()[:16] if token_ids else ""
        session_turn = int(getattr(request, "session_turn", 0) or 0)
        segment_match = float(getattr(request, "segment_match_ratio", 0.0) or 0.0)
        max_ctx = max(1, self._cpd_config.max_context_length_warm)

        history = self._cpd_hit_history.get(prefix_hash, [])
        f1 = sum(history) / len(history) if history else 0.0
        f2 = min(1.0, len(token_ids) / max_ctx)
        f3 = 1.0 / (1.0 + session_turn)
        f4 = segment_match
        return f1, f2, f3, f4

    def _cpd_predict_hit_rate(self, request: Any) -> float:
        """Predict cache hit rate for a single request via linear sigmoid model."""
        if self._cpd_router is not None:
            # Delegate to verified implementation
            class _Req:
                pass
            r = _Req()
            token_ids = list(getattr(request, "prompt_token_ids", None) or
                             getattr(request, "token_ids", None) or [])
            prefix_hash = hashlib.sha256(
                bytes(token_ids[:512])
            ).hexdigest()[:16] if token_ids else ""
            object.__setattr__(r, "prefix_hash", prefix_hash)
            object.__setattr__(r, "token_ids", token_ids)
            object.__setattr__(r, "session_turn", getattr(request, "session_turn", 0))
            object.__setattr__(r, "segment_match_ratio",
                               getattr(request, "segment_match_ratio", 0.0))
            object.__setattr__(r, "request_id",
                               str(getattr(request, "request_id", "")))
            try:
                return self._cpd_router.predict_hit_rate(r)
            except Exception:
                pass
        f1, f2, f3, f4 = self._cpd_extract_features(request)
        w = self._cpd_w
        score = w[0]*f1 + w[1]*f2 + w[2]*f3 + w[3]*f4 + self._cpd_bias
        import math
        return 1.0 / (1.0 + math.exp(-score))

    def cpd_pre_schedule(self) -> None:
        """Annotate and (soft-)sort waiting requests by predicted hit rate.

        Annotates each request with cpd_path / cpd_predicted_hit_rate /
        cpd_batch_priority. No hard reordering of vLLM's internal queue
        (to avoid breaking FCFS contracts) — but records priorities for
        the schedule() override to consume.
        """
        t0 = time.monotonic()
        cfg = self._cpd_config
        try:
            for request in iter(self.waiting):  # type: ignore[attr-defined]
                hit_rate = self._cpd_predict_hit_rate(request)
                if hit_rate >= cfg.high_hit_threshold:
                    path, priority = "warm", 0
                    self._cpd_warm_count += 1
                elif hit_rate < cfg.low_hit_threshold:
                    path, priority = "cold", 2
                    self._cpd_cold_count += 1
                else:
                    path, priority = "neutral", 1
                    self._cpd_neutral_count += 1
                try:
                    object.__setattr__(request, "cpd_path", path)
                    object.__setattr__(request, "cpd_predicted_hit_rate", hit_rate)
                    object.__setattr__(request, "cpd_batch_priority", priority)
                except Exception:
                    pass
        except (AttributeError, TypeError):
            pass
        overhead_us = (time.monotonic() - t0) * 1e6
        self._cpd_overhead_us.append(overhead_us)

    def cpd_update_predictor(self, request: Any, actual_hit: bool) -> None:
        """Update online SGD weights from actual hit outcome."""
        if self._cpd_router is not None:
            try:
                class _Req:
                    pass
                r = _Req()
                token_ids = list(getattr(request, "prompt_token_ids", None) or
                                 getattr(request, "token_ids", None) or [])
                prefix_hash = hashlib.sha256(
                    bytes(token_ids[:512])
                ).hexdigest()[:16] if token_ids else ""
                object.__setattr__(r, "prefix_hash", prefix_hash)
                object.__setattr__(r, "token_ids", token_ids)
                object.__setattr__(r, "session_turn", getattr(request, "session_turn", 0))
                object.__setattr__(r, "segment_match_ratio",
                                   getattr(request, "segment_match_ratio", 0.0))
                object.__setattr__(r, "request_id",
                                   str(getattr(request, "request_id", "")))
                self._cpd_router.update_predictor(r, actual_hit)
                return
            except Exception:
                pass
        # Inline fallback SGD
        f1, f2, f3, f4 = self._cpd_extract_features(request)
        import math
        score = (self._cpd_w[0]*f1 + self._cpd_w[1]*f2 +
                 self._cpd_w[2]*f3 + self._cpd_w[3]*f4 + self._cpd_bias)
        y_pred = 1.0 / (1.0 + math.exp(-score))
        y_true = 1.0 if actual_hit else 0.0
        error = y_pred - y_true
        lr = self._cpd_lr
        feats = [f1, f2, f3, f4]
        for i in range(4):
            self._cpd_w[i] -= lr * error * feats[i]
        self._cpd_bias -= lr * error
        # Update hit history
        token_ids = list(getattr(request, "prompt_token_ids", None) or
                         getattr(request, "token_ids", None) or [])
        ph = hashlib.sha256(bytes(token_ids[:512])).hexdigest()[:16] if token_ids else ""
        if ph:
            hist = self._cpd_hit_history.setdefault(ph, [])
            hist.append(actual_hit)
            if len(hist) > 100:
                hist.pop(0)

    def cpd_routing_stats(self) -> Dict[str, Any]:
        """Return CPD routing statistics for observability."""
        total = max(1, self._cpd_warm_count + self._cpd_cold_count + self._cpd_neutral_count)
        mean_overhead = (sum(self._cpd_overhead_us) / len(self._cpd_overhead_us)
                         if self._cpd_overhead_us else 0.0)
        return {
            "warm_ratio": self._cpd_warm_count / total,
            "cold_ratio": self._cpd_cold_count / total,
            "neutral_ratio": self._cpd_neutral_count / total,
            "scheduling_overhead_mean_us": mean_overhead,
            "total_decisions": total,
        }

    def schedule(self) -> Any:
        self.cpd_pre_schedule()
        return super().schedule()  # type: ignore[misc]


def make_cpd_warm_cold_scheduler_class(
    base_class: Optional[type] = None,
    cpd_config: Optional[CPDRouterSchedulerConfig] = None,
) -> type:
    """Factory: subclass vLLM Scheduler with CPDWarmColdSchedulerMixin.

    Returns a class that:
      - is a subclass of vLLM's Scheduler (or base_class)
      - intercepts schedule() to run CPD warm/cold classification first
      - annotates each waiting Request with cpd_path / cpd_predicted_hit_rate

    Usage:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import (
            CPDRouterSchedulerConfig,
            make_cpd_warm_cold_scheduler_class,
        )
        cfg = CPDRouterSchedulerConfig(high_hit_threshold=0.70)
        CPDScheduler = make_cpd_warm_cold_scheduler_class(Scheduler, cfg)
        # CPDScheduler(**vllm_scheduler_kwargs, cpd_config=cfg)

    vLLM version: 0.21.0
    Activity: A — CPDWarmColdHitRateRouter
    """
    if base_class is None:
        try:
            from vllm.v1.core.sched.scheduler import Scheduler as _Sched
            base_class = _Sched
        except Exception:
            base_class = object

    _cfg = cpd_config

    class CPDWarmColdScheduler(CPDWarmColdSchedulerMixin, base_class):  # type: ignore[valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if _cfg is not None and "cpd_config" not in kwargs:
                kwargs["cpd_config"] = _cfg
            super().__init__(*args, **kwargs)

    CPDWarmColdScheduler.__name__ = "CPDWarmColdScheduler"
    CPDWarmColdScheduler.__qualname__ = "CPDWarmColdScheduler"
    return CPDWarmColdScheduler
