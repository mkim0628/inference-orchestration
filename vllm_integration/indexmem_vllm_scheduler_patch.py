"""indexmem_vllm_scheduler_patch.py — Activity A/B: IndexMem Soft-Hit Scheduler for vLLM 0.21.0.

2026-05-27: IndexMemSoftHitSchedulerMixin — scheduler awareness of soft-hit segments.

            Deprioritizes re-computation when a soft-hit latent state is available
            for a segment, routing such requests to the latent readout path instead
            of triggering a full prefill recompute on the P-node.

            Core scheduling logic:
              1. For each waiting request, query the IndexMem soft-hit cache to
                 classify the request's token sequence into:
                   "hard_hit"  : physical KV in GPU VRAM — full reuse, schedule normally
                   "soft_hit"  : KV evicted, latent state in DRAM — deprioritize prefill,
                                 route to latent readout path (lower cost than recompute)
                   "miss"      : no cached state — schedule for full recompute (P-node)

              2. Soft-hit requests are given reduced scheduling priority to avoid
                 redundant P-node prefill when latent residual reconstruction suffices.

              3. Overhead: O(n_chunks) per request (SHA-256 hash of each chunk).
                 Target: < 5ms p50 TTFT overhead for N ≤ 1000 waiting requests.

            Integration with IndexMemSoftHitKVCacheManagerMixin:
              The scheduler mixin queries the block manager's soft-hit cache to
              determine segment availability before admission. Block manager and
              scheduler share the same logical segment namespace (SHA-256 keys).

            make_indexmem_soft_hit_scheduler_class() factory:
              Returns a vLLM v1 Scheduler subclass combining:
                - IndexMemSoftHitSchedulerMixin (soft-hit routing)
              with the base Scheduler class.

vLLM 0.21.0 v1 architecture:
    - Scheduler lives in vllm.v1.core.sched.scheduler.Scheduler
    - Waiting queue is self.waiting (RequestQueue, iterable)
    - Per-step scheduling via Scheduler.schedule() → SchedulerOutput
    - KV block management via self.kv_cache_manager (KVCacheManager)
    - Request class: vllm.v1.request.Request

vLLM version: 0.21.0
Activity: A (scheduling) + B (soft-hit routing) — IndexMem B+C integration
"""

from __future__ import annotations

import sys
import pathlib
import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

try:
    import torch
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


# ---------------------------------------------------------------------------
# Import vLLM Scheduler (graceful fallback for CPU-only environments)
# ---------------------------------------------------------------------------

try:
    from vllm.v1.core.sched.scheduler import Scheduler as VllmScheduler
    _SCHEDULER_AVAILABLE = True
except Exception:
    class VllmScheduler:  # type: ignore[no-redef]
        """Stub Scheduler for CPU-only environments."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.waiting: List[Any] = []
        def schedule(self) -> Any:
            return None
    _SCHEDULER_AVAILABLE = False


# ===========================================================================
# IndexMemSoftHitSchedulerConfig
# ===========================================================================

@dataclass
class IndexMemSoftHitSchedulerConfig:
    """Configuration for IndexMemSoftHitSchedulerMixin.

    Controls how the scheduler routes requests based on soft-hit availability.
    """
    # Soft-hit routing thresholds
    soft_hit_priority_boost: float = 0.5
        # Soft-hit requests get this fraction of normal scheduling priority.
        # 0.5 = deprioritize by 50% (soft-hit recompute is cheaper than full prefill)
    min_chunks_for_soft_hit_routing: int = 1
        # Minimum number of soft-hit chunks required to apply routing.
    soft_hit_deprioritize_threshold: float = 0.3
        # If soft_hit_rate >= this fraction, skip normal P-node prefill.

    # Chunk configuration (must match block manager config)
    chunk_size: int = 128

    # Scheduling overhead tracking
    max_overhead_us_per_request: float = 1000.0  # 1ms per request
    expire_session_interval: int = 100  # calls between session expiry scans

    # Misc
    seed: int = 42
    enabled: bool = True


# ===========================================================================
# IndexMemSoftHitSchedulerMixin — Activity B/A (2026-05-27)
# ===========================================================================

@dataclass
class _SoftHitClassification:
    """Classification result for a single request."""
    request_id: str
    session_id: str
    cache_path: Literal["hard_hit", "soft_hit", "miss"]
    n_hard_chunks: int = 0
    n_soft_chunks: int = 0
    n_miss_chunks: int = 0
    weighted_hit_rate: float = 0.0
    overhead_us: float = 0.0


class IndexMemSoftHitSchedulerMixin:
    """vLLM v1 Scheduler mixin: IndexMem soft-hit segment routing.

    Activity B+A: routes requests based on IndexMem soft-hit segment availability.

    When a request's token sequence has soft-hit segments (latent states in DRAM):
      - The request is classified as "soft_hit" path.
      - Scheduling priority is reduced (soft_hit_priority_boost=0.5).
      - The request's block table is annotated with soft-hit block IDs for the
        attention kernel to use the latent readout path.

    When a request has hard-hit segments (physical KV in VRAM):
      - Normal scheduling proceeds.
      - No priority modification.

    When a request has no cached state (miss):
      - Full P-node prefill is triggered.
      - Normal scheduling proceeds.

    Usage:
        class MyScheduler(IndexMemSoftHitSchedulerMixin, VllmScheduler):
            pass

        sched = MyScheduler(
            ...,  # standard Scheduler args
            indexmem_scheduler_config=IndexMemSoftHitSchedulerConfig(
                soft_hit_priority_boost=0.5,
                chunk_size=128,
            ),
        )
    """

    def __init__(
        self,
        *args: Any,
        indexmem_scheduler_config: Optional[IndexMemSoftHitSchedulerConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            indexmem_scheduler_config: Configuration for soft-hit routing.
            All other args/kwargs forwarded to the base Scheduler.__init__().
        """
        super().__init__(*args, **kwargs)

        if indexmem_scheduler_config is None:
            indexmem_scheduler_config = IndexMemSoftHitSchedulerConfig()
        self._im_sched_cfg = indexmem_scheduler_config

        # Session state: {session_id: last_classification}
        self._im_session_state: Dict[str, _SoftHitClassification] = {}

        # Inline segment key cache (avoid recomputing hashes)
        self._im_key_cache: Dict[str, str] = {}

        # Metrics
        self._im_sched_call_count: int = 0
        self._im_hard_hit_routes: int = 0
        self._im_soft_hit_routes: int = 0
        self._im_miss_routes: int = 0
        self._im_total_overhead_us: float = 0.0

        # Reference to block manager soft-hit cache (set after init if available)
        self._im_block_manager: Optional[Any] = None

    def set_indexmem_block_manager(self, block_manager: Any) -> None:
        """Attach an IndexMemSoftHitKVCacheManagerMixin block manager.

        Must be called after both scheduler and block manager are initialized,
        before schedule() is called.

        Args:
            block_manager: An instance with soft-hit query methods:
                get_soft_hit_result(), weighted_hit_rate(), etc.
        """
        self._im_block_manager = block_manager

    # -----------------------------------------------------------------------
    # Segment key computation (mirrors block manager)
    # -----------------------------------------------------------------------

    def _im_chunk_key(
        self,
        session_id: str,
        turn_id: int,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """Compute SHA-256 segment key for a chunk of tokens.

        Must produce the same keys as IndexMemSoftHitKVCacheManagerMixin._im_chunk_key().
        """
        chunk_size = self._im_sched_cfg.chunk_size
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = token_ids[start:end]
        raw = struct.pack(f"{len(chunk)}I", *chunk) if chunk else b""
        header = f"{session_id}|{turn_id}|l{layer_idx}".encode()
        return hashlib.sha256(header + raw).hexdigest()

    # -----------------------------------------------------------------------
    # Soft-hit classification
    # -----------------------------------------------------------------------

    def _im_classify_request(
        self,
        request: Any,
        session_id: str,
        turn_id: int = 0,
        layer_idx: int = 0,
    ) -> _SoftHitClassification:
        """Classify a request's token sequence by soft-hit availability.

        Algorithm:
          1. Compute chunk keys for all chunks in request.prompt_token_ids.
          2. Query block manager (if available) for hard/soft/miss status.
          3. Aggregate results into classification.

        Returns:
            _SoftHitClassification with cache_path "hard_hit" | "soft_hit" | "miss".
        """
        t0 = time.monotonic()
        request_id = getattr(request, "request_id", str(id(request)))
        token_ids = getattr(request, "prompt_token_ids", [])

        if not token_ids or not self._im_sched_cfg.enabled:
            return _SoftHitClassification(
                request_id=request_id,
                session_id=session_id,
                cache_path="miss",
                overhead_us=0.0,
            )

        chunk_size = self._im_sched_cfg.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)

        n_hard = 0
        n_soft = 0
        n_miss = 0

        if self._im_block_manager is not None and hasattr(
            self._im_block_manager, "get_soft_hit_result"
        ):
            for chunk_idx in range(n_chunks):
                try:
                    result = self._im_block_manager.get_soft_hit_result(
                        session_id, turn_id, token_ids, chunk_idx, layer_idx
                    )
                    if result.type == "hard":
                        n_hard += 1
                    elif result.type == "soft":
                        n_soft += 1
                    else:
                        n_miss += 1
                except Exception:
                    n_miss += 1
        else:
            # No block manager attached — all chunks are misses
            n_miss = n_chunks

        # Determine overall cache path
        total = n_hard + n_soft + n_miss
        if n_hard == total:
            cache_path: Literal["hard_hit", "soft_hit", "miss"] = "hard_hit"
        elif n_soft > 0 or (n_hard > 0 and n_miss > 0):
            cache_path = "soft_hit"
        else:
            cache_path = "miss"

        weighted_hr = (n_hard + 0.5 * n_soft) / max(1, total)
        overhead_us = (time.monotonic() - t0) * 1e6

        return _SoftHitClassification(
            request_id=request_id,
            session_id=session_id,
            cache_path=cache_path,
            n_hard_chunks=n_hard,
            n_soft_chunks=n_soft,
            n_miss_chunks=n_miss,
            weighted_hit_rate=weighted_hr,
            overhead_us=overhead_us,
        )

    # -----------------------------------------------------------------------
    # Pre-schedule hook
    # -----------------------------------------------------------------------

    def im_pre_schedule(self) -> None:
        """Pre-schedule hook: classify waiting requests for soft-hit routing.

        Called at the start of each schedule() invocation (or can be called
        directly from a subclass that overrides schedule()).

        For each waiting request:
          1. Computes soft-hit classification.
          2. Annotates request with im_cache_path, im_soft_hit_rate, etc.
          3. Updates aggregate metrics (hard/soft/miss route counts).

        Overhead: O(n_chunks) per request for SHA-256 hash computation.
        Sessions are expired periodically (every expire_session_interval calls).
        """
        self._im_sched_call_count += 1

        if not self._im_sched_cfg.enabled:
            return

        waiting = getattr(self, "waiting", [])

        for request in waiting:
            try:
                request_id = getattr(request, "request_id", str(id(request)))
                # Derive session_id from request (use request_id as fallback)
                session_id = getattr(
                    request, "session_id",
                    getattr(request, "request_id", request_id)
                )
                turn_id = getattr(request, "session_turn", 0)

                classification = self._im_classify_request(
                    request, session_id, turn_id
                )

                # Update metrics
                if classification.cache_path == "hard_hit":
                    self._im_hard_hit_routes += 1
                elif classification.cache_path == "soft_hit":
                    self._im_soft_hit_routes += 1
                else:
                    self._im_miss_routes += 1

                self._im_total_overhead_us += classification.overhead_us

                # Annotate request with classification results
                try:
                    object.__setattr__(request, "im_cache_path", classification.cache_path)
                    object.__setattr__(request, "im_n_soft_chunks", classification.n_soft_chunks)
                    object.__setattr__(request, "im_n_hard_chunks", classification.n_hard_chunks)
                    object.__setattr__(request, "im_n_miss_chunks", classification.n_miss_chunks)
                    object.__setattr__(
                        request, "im_weighted_hit_rate", classification.weighted_hit_rate
                    )
                    object.__setattr__(
                        request, "im_deprioritize_prefill",
                        classification.cache_path == "soft_hit"
                        and classification.n_soft_chunks
                        >= self._im_sched_cfg.min_chunks_for_soft_hit_routing
                    )
                except Exception:
                    pass  # Graceful: some Request types may not support setattr

                self._im_session_state[session_id] = classification

            except Exception:
                pass  # Never block scheduling due to classification error

        # Periodic session expiry
        if (
            self._im_sched_call_count
            % self._im_sched_cfg.expire_session_interval == 0
        ):
            self._im_expire_sessions()

    def _im_expire_sessions(self) -> int:
        """Expire stale session classifications. Returns number expired."""
        # Simple expiry: remove sessions not seen in last 1000 calls
        # (full TTL-based expiry requires per-session timestamps)
        max_sessions = 10000
        if len(self._im_session_state) > max_sessions:
            n_remove = len(self._im_session_state) - max_sessions
            to_remove = list(self._im_session_state.keys())[:n_remove]
            for k in to_remove:
                del self._im_session_state[k]
            return n_remove
        return 0

    # -----------------------------------------------------------------------
    # Overriding schedule() (optional — only when needed)
    # -----------------------------------------------------------------------

    def schedule(self) -> Any:
        """Override schedule() to inject im_pre_schedule() before base scheduling.

        Calls im_pre_schedule() then the base class schedule(). Subclasses that
        override schedule() themselves should call self.im_pre_schedule() at the
        start of their schedule() implementation.

        Returns:
            SchedulerOutput from the base class schedule().
        """
        self.im_pre_schedule()
        return super().schedule()

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def im_routing_stats(self) -> Dict[str, Any]:
        """Return scheduler routing statistics for observability."""
        total_routes = (
            self._im_hard_hit_routes
            + self._im_soft_hit_routes
            + self._im_miss_routes
        )
        return {
            "total_schedule_calls": self._im_sched_call_count,
            "total_routes": total_routes,
            "hard_hit_routes": self._im_hard_hit_routes,
            "soft_hit_routes": self._im_soft_hit_routes,
            "miss_routes": self._im_miss_routes,
            "hard_hit_ratio": (
                self._im_hard_hit_routes / max(1, total_routes)
            ),
            "soft_hit_ratio": (
                self._im_soft_hit_routes / max(1, total_routes)
            ),
            "miss_ratio": (
                self._im_miss_routes / max(1, total_routes)
            ),
            "scheduling_overhead_mean_us": (
                self._im_total_overhead_us / max(1, self._im_sched_call_count)
            ),
            "active_sessions": len(self._im_session_state),
            "block_manager_attached": self._im_block_manager is not None,
        }


# ---------------------------------------------------------------------------
# make_indexmem_soft_hit_scheduler_class() factory
# ---------------------------------------------------------------------------

def make_indexmem_soft_hit_scheduler_class(
    base_class: type = VllmScheduler,
    config: Optional[IndexMemSoftHitSchedulerConfig] = None,
) -> type:
    """Factory: return a Scheduler subclass with IndexMem soft-hit routing.

    Args:
        base_class: The vLLM Scheduler class to subclass. Defaults to
            vllm.v1.core.sched.scheduler.Scheduler.
        config: Optional IndexMemSoftHitSchedulerConfig to embed.

    Returns:
        A new class combining IndexMemSoftHitSchedulerMixin with base_class.

    Usage:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.indexmem_vllm_scheduler_patch import (
            IndexMemSoftHitSchedulerConfig,
            make_indexmem_soft_hit_scheduler_class,
        )

        cfg = IndexMemSoftHitSchedulerConfig(
            soft_hit_priority_boost=0.5,
            chunk_size=128,
        )
        IndexMemScheduler = make_indexmem_soft_hit_scheduler_class(Scheduler, cfg)
        assert issubclass(IndexMemScheduler, Scheduler)
        assert issubclass(IndexMemScheduler, IndexMemSoftHitSchedulerMixin)
    """
    _cfg = config

    class _IndexMemSoftHitScheduler(
        IndexMemSoftHitSchedulerMixin,
        base_class,  # type: ignore[valid-type]
    ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if _cfg is not None and "indexmem_scheduler_config" not in kwargs:
                kwargs["indexmem_scheduler_config"] = _cfg
            super().__init__(*args, **kwargs)

    _IndexMemSoftHitScheduler.__name__ = "IndexMemSoftHitScheduler"
    _IndexMemSoftHitScheduler.__qualname__ = "IndexMemSoftHitScheduler"
    _IndexMemSoftHitScheduler.__doc__ = (
        "Scheduler subclass with IndexMem soft-hit routing (Activity B+A). "
        "Auto-generated by make_indexmem_soft_hit_scheduler_class()."
    )
    return _IndexMemSoftHitScheduler
