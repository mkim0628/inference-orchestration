"""hexagent_scheduler_patch.py — Activity A: HexAGenT DAG Workflow Scheduler for vLLM 0.21.0.

2026-05-28: HexAGeTWorkflowSchedulerMixin — ports the HexAGenT online-public DAG
            workflow scheduler (src/scheduler/hexagent_workflow_scheduler.py) into
            vLLM's v1 Scheduler as a mixin.

            Implements 3-way optimization:
              1. Standalone completion horizon per task (t_prefill + t_decode + t_kv_transfer)
              2. SLO-risk-weighted priority for ready task scheduling
              3. KV capacity constraint + heterogeneous GPU transfer latency in objective

            HexAGeTWorkflowSchedulerMixin wraps schedule() with a hexagent_pre_schedule()
            hook that:
              1. Iterates self.waiting (RequestQueue) without modifying queue order.
              2. For each waiting request, computes a HexAGenT priority score using
                 the request's token count as a proxy for kv_demand_estimate and
                 the request's arrival_time + configured SLO to derive slo_deadline.
              3. Annotates the vLLM Request with:
                   - hexagent_priority: float (higher = scheduled earlier)
                   - hexagent_slo_risk: float (0.0 = slack, 1.0+ = SLO violation imminent)
                   - hexagent_horizon_ms: float (estimated standalone completion horizon ms)
                   - hexagent_gpu_type: str ("A100" | "H100" | "H200" | "any")
              4. Re-orders self.waiting based on hexagent_priority (highest first).

            Multi-node KV routing annotation (Activity A-2 PegaFlow RDMA):
              If a request has a known segment set (annotated via set_request_segments()),
              the mixin checks PegaFlow RDMA router availability and annotates the request
              with hexagent_rdma_local (bool) — True means local cache hit is preferred.

            make_hexagent_workflow_scheduler_class() factory — builds a vLLM v1
            Scheduler subclass that intercepts schedule() to run HexAGenT DAG
            prioritization before base scheduling.

vLLM version: 0.21.0
Activity: A — HexAGenT DAG Workflow Scheduler Mixin (Activity A-1 + A-2 RDMA routing annotation)
"""

from __future__ import annotations

import sys
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import vllm


def _add_repo_root_to_path() -> None:
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _try_import_hexagent_src():
    """Lazily import HexAGeTWorkflowScheduler from src/."""
    _add_repo_root_to_path()
    try:
        from src.scheduler.hexagent_workflow_scheduler import (
            HexAGeTWorkflowScheduler,
            HexAGeTSchedulerConfig,
            TaskNode,
            WorkflowDAG,
        )
        return HexAGeTWorkflowScheduler, HexAGeTSchedulerConfig, TaskNode, WorkflowDAG
    except ImportError:
        return None, None, None, None


def _try_import_rdma_router_src():
    """Lazily import PegaFlowRDMACrossNodeRouter from src/."""
    _add_repo_root_to_path()
    try:
        from src.scheduler.pegaflow_rdma_router import (
            PegaFlowRDMACrossNodeRouter,
            PegaFlowRDMARouterConfig,
            PeerRegistry,
        )
        return PegaFlowRDMACrossNodeRouter, PegaFlowRDMARouterConfig, PeerRegistry
    except ImportError:
        return None, None, None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HexAGeTSchedulerMixinConfig:
    """Configuration for HexAGeTWorkflowSchedulerMixin.

    Maps to HexAGeTSchedulerConfig from src/; used as fallback when src/ is
    not importable.
    """
    schedule_cycle_ms: float = 50.0
    risk_weight: float = 2.0
    alpha_gpu_affinity: float = 0.5
    slo_budget_ms: float = 5000.0         # default SLO per request (ms from arrival)
    kv_size_per_token_bytes: int = 512    # token → KV bytes estimate
    ema_decay: float = 0.9
    rdma_bandwidth_table_path: str = "configs/gpu_rdma_bandwidth_table.yaml"
    peer_nodes_config_path: str = "configs/pegaflow_peer_nodes.yaml"
    seed: int = 42
    reorder_waiting_queue: bool = True    # True: re-sort self.waiting by hexagent_priority
    expire_interval: int = 200            # expire stale state every N schedule() calls


# Base priorities by inferred task type (based on token count heuristic)
_HEXAGENT_BASE_PRIORITY: Dict[str, float] = {
    "planning": 4.0,
    "synthesis": 3.0,
    "tool_call": 2.0,
    "refine": 1.0,
    "unknown": 1.5,
}

# Approximate GPU throughput in bytes/ms
_GPU_THROUGHPUT_BYTES_PER_MS: Dict[str, float] = {
    "A100": 2_000_000.0,
    "H100": 4_000_000.0,
    "H200": 6_000_000.0,
    "any": 2_000_000.0,
}


# ============================================================================
# Inline fallback: HexAGenT horizon + SLO-risk priority (no src/ dependency)
# ============================================================================

class _InlineHexAGeTScheduler:
    """Pure-Python HexAGenT horizon + SLO-risk scorer for use when src/ unavailable."""

    def __init__(self, config: HexAGeTSchedulerMixinConfig) -> None:
        self.config = config
        self._request_state: Dict[str, Dict] = {}    # request_id → {arrival, task_type}
        self._rdma_bandwidth: Dict[str, float] = self._default_bandwidth()

    @staticmethod
    def _default_bandwidth() -> Dict[str, float]:
        return {
            "A100_to_A100": 200.0,
            "A100_to_H100": 200.0,
            "A100_to_H200": 200.0,
            "H100_to_H100": 400.0,
            "H100_to_H200": 400.0,
            "H200_to_H200": 800.0,
            "default": 100.0,
        }

    def _rdma_bandwidth_gbps(self, src_gpu: str, tgt_gpu: str) -> float:
        key = f"{src_gpu}_to_{tgt_gpu}"
        alt = f"{tgt_gpu}_to_{src_gpu}"
        return (
            self._rdma_bandwidth.get(key)
            or self._rdma_bandwidth.get(alt)
            or self._rdma_bandwidth.get("default", 100.0)
        )

    def standalone_completion_horizon(
        self,
        n_tokens: int,
        kv_demand_bytes: int,
        gpu_type: str = "any",
        current_time_ms: float = 0.0,
    ) -> float:
        """Estimate standalone completion horizon in ms from now.

        horizon_ms = current_time_ms + t_prefill_ms + t_decode_ms + t_kv_transfer_ms
        """
        throughput = _GPU_THROUGHPUT_BYTES_PER_MS.get(gpu_type, 2_000_000.0)
        # Rough estimates
        t_prefill_ms = kv_demand_bytes / throughput
        t_decode_ms = n_tokens * 0.05   # 0.05ms per output token estimate
        t_kv_transfer_ms = kv_demand_bytes / (_GPU_THROUGHPUT_BYTES_PER_MS["A100"] * 10)
        return current_time_ms + t_prefill_ms + t_decode_ms + t_kv_transfer_ms

    def slo_risk_score(
        self,
        horizon_ms: float,
        slo_deadline_ms: float,
    ) -> float:
        """SLO risk score: 0.0 = slack, 1.0+ = violation imminent."""
        if slo_deadline_ms <= 0.0:
            return 0.0
        return max(0.0, horizon_ms - slo_deadline_ms) / slo_deadline_ms

    def priority(
        self,
        request_id: str,
        n_tokens: int,
        kv_demand_bytes: int,
        arrival_time_s: float,
        task_type: str = "unknown",
        gpu_type: str = "any",
    ) -> Tuple[float, float, float]:
        """Compute (priority, slo_risk, horizon_ms) for a request.

        Returns:
          (priority_score, slo_risk, horizon_ms)
          priority = base_priority + risk_weight × slo_risk
        """
        now_ms = time.monotonic() * 1000.0
        arrival_ms = arrival_time_s * 1000.0
        slo_deadline_ms = arrival_ms + self.config.slo_budget_ms

        horizon_ms = self.standalone_completion_horizon(
            n_tokens, kv_demand_bytes, gpu_type, now_ms
        )
        risk = self.slo_risk_score(horizon_ms, slo_deadline_ms)
        base = _HEXAGENT_BASE_PRIORITY.get(task_type, 1.5)
        prio = base + self.config.risk_weight * risk
        return prio, risk, horizon_ms - now_ms

    def expire_stale_state(self) -> int:
        """Remove stale per-request state; returns count removed."""
        now_s = time.monotonic()
        stale = [
            rid for rid, s in self._request_state.items()
            if now_s - s.get("last_seen", now_s) > 3600.0
        ]
        for rid in stale:
            del self._request_state[rid]
        return len(stale)

    def record_request(self, request_id: str, task_type: str = "unknown") -> None:
        now_s = time.monotonic()
        if request_id not in self._request_state:
            self._request_state[request_id] = {
                "arrival": now_s,
                "task_type": task_type,
                "last_seen": now_s,
            }
        else:
            self._request_state[request_id]["last_seen"] = now_s

    def get_arrival_time(self, request_id: str) -> float:
        return self._request_state.get(request_id, {}).get(
            "arrival", time.monotonic()
        )

    def get_task_type(self, request_id: str) -> str:
        return self._request_state.get(request_id, {}).get("task_type", "unknown")


# ============================================================================
# 2026-05-28: HexAGeTWorkflowSchedulerMixin (Activity A)
# ============================================================================

class HexAGeTWorkflowSchedulerMixin:
    """vLLM v1 Scheduler mixin: HexAGenT online-public DAG workflow scheduling.

    Activity A: KV Cache-aware Scheduling via HexAGenT DAG.
    Based on HexAGenT (arXiv 2605.16637): workflow-horizon-aware SLO-risk scheduling.

    This mixin wraps schedule() with a hexagent_pre_schedule() hook that:
      1. Iterates self.waiting (RequestQueue) without modifying queue order (unless
         reorder_waiting_queue=True, which re-sorts by hexagent_priority).
      2. For each request, computes a HexAGenT priority score using:
           - n_tokens × kv_size_per_token_bytes as kv_demand estimate
           - arrival_time + slo_budget_ms as slo_deadline
           - standalone_completion_horizon() to estimate task completion horizon
           - SLO-risk score = max(0, horizon - deadline) / deadline
      3. Annotates each vLLM Request with:
           - hexagent_priority: float (higher = scheduled earlier)
           - hexagent_slo_risk: float
           - hexagent_horizon_ms: float (remaining horizon ms from now)
           - hexagent_kv_demand_bytes: int (estimated KV footprint)

    Integration with vLLM v1 architecture:
      - Wraps schedule() call; base class schedule() handles actual block allocation.
      - Does NOT modify token allocation or block management.
      - Scheduling overhead target: < 1ms per 1000 waiting requests.

    Multi-node RDMA routing (Activity A-2):
      If PegaFlowRDMACrossNodeRouter is available, annotates requests with
      hexagent_rdma_local=True when the local PegaFlow store has the segment.
      This annotation is used by PegaFlowIrminsulDistributedKVCacheManagerMixin
      (block_manager_patch.py) to prefer local cache over RDMA transfer.
    """

    # ---- Mixin initialization ---- #

    def _hexagent_init(
        self,
        config: Optional[HexAGeTSchedulerMixinConfig] = None,
        rdma_router: Optional[Any] = None,
    ) -> None:
        """Initialize HexAGenT mixin state. Call from __init__ or lazily."""
        self._hexagent_cfg = config or HexAGeTSchedulerMixinConfig()
        self._hexagent_call_count: int = 0

        # Try to import src/ native implementation
        HexAGeTScheduler, HexAGeTSchedulerConfig, _, _ = _try_import_hexagent_src()
        if HexAGeTScheduler is not None and HexAGeTSchedulerConfig is not None:
            native_cfg = HexAGeTSchedulerConfig(
                schedule_cycle_ms=self._hexagent_cfg.schedule_cycle_ms,
                risk_weight=self._hexagent_cfg.risk_weight,
                alpha_gpu_affinity=self._hexagent_cfg.alpha_gpu_affinity,
                ema_decay=self._hexagent_cfg.ema_decay,
                rdma_bandwidth_table_path=self._hexagent_cfg.rdma_bandwidth_table_path,
                seed=self._hexagent_cfg.seed,
            )
            self._hexagent_native = HexAGeTScheduler(native_cfg)
            self._hexagent_use_native = True
        else:
            self._hexagent_native = _InlineHexAGeTScheduler(self._hexagent_cfg)
            self._hexagent_use_native = False

        # RDMA router for Activity A-2 multi-node routing annotation
        self._hexagent_rdma_router = rdma_router

        # Routing statistics
        self._hexagent_priority_total: float = 0.0
        self._hexagent_high_risk_count: int = 0   # requests with slo_risk > 0.5
        self._hexagent_reorder_count: int = 0
        self._hexagent_overhead_sum_us: float = 0.0
        self._hexagent_overhead_n: int = 0

    def _hexagent_ensure_init(self) -> None:
        """Lazy initialization guard."""
        if not hasattr(self, "_hexagent_cfg"):
            self._hexagent_init()

    # ---- Core pre-schedule hook ---- #

    def hexagent_pre_schedule(self) -> None:
        """Annotate and optionally re-order self.waiting by HexAGenT priority.

        Must be called from schedule() before the base scheduling logic.
        """
        self._hexagent_ensure_init()
        t0 = time.monotonic()

        waiting = getattr(self, "waiting", None)
        if waiting is None or len(waiting) == 0:
            return

        self._hexagent_call_count += 1

        # Expire stale state periodically
        if (
            self._hexagent_call_count % self._hexagent_cfg.expire_interval == 0
            and not self._hexagent_use_native
        ):
            self._hexagent_native.expire_stale_state()

        scored: List[Tuple[float, int, Any]] = []   # (priority, index, request)
        for idx, req in enumerate(waiting):
            priority, slo_risk, horizon_ms = self._hexagent_score_request(req)
            # Annotate request in-place
            req.hexagent_priority = priority
            req.hexagent_slo_risk = slo_risk
            req.hexagent_horizon_ms = horizon_ms
            req.hexagent_kv_demand_bytes = self._hexagent_kv_demand(req)
            scored.append((priority, idx, req))

            # Multi-node routing annotation (Activity A-2)
            if self._hexagent_rdma_router is not None:
                self._hexagent_annotate_rdma_routing(req)

            # Stats
            self._hexagent_priority_total += priority
            if slo_risk > 0.5:
                self._hexagent_high_risk_count += 1

        # Optionally re-sort waiting queue by priority (highest first)
        if self._hexagent_cfg.reorder_waiting_queue and len(scored) > 1:
            scored.sort(key=lambda x: -x[0])
            try:
                # vLLM v1 RequestQueue supports direct list manipulation
                sorted_reqs = [r for _, _, r in scored]
                if hasattr(waiting, "_reqs"):
                    waiting._reqs = sorted_reqs
                elif hasattr(waiting, "requests"):
                    waiting.requests = sorted_reqs
                elif isinstance(waiting, list):
                    waiting.clear()
                    waiting.extend(sorted_reqs)
                self._hexagent_reorder_count += 1
            except Exception:
                pass  # Queue is immutable or unsupported — skip reorder

        elapsed_us = (time.monotonic() - t0) * 1e6
        self._hexagent_overhead_sum_us += elapsed_us
        self._hexagent_overhead_n += 1

    def _hexagent_score_request(self, req: Any) -> Tuple[float, float, float]:
        """Compute (priority, slo_risk, horizon_ms) for a single vLLM Request."""
        n_tokens = len(getattr(req, "prompt_token_ids", []) or [])
        arrival_time_s = getattr(req, "arrival_time", time.monotonic())
        gpu_type = getattr(req, "hexagent_gpu_type", "any")

        kv_demand = self._hexagent_kv_demand(req)

        if self._hexagent_use_native:
            # Native path: use HexAGeTWorkflowScheduler for accurate horizon estimation
            # Create a minimal TaskNode proxy
            from src.scheduler.hexagent_workflow_scheduler import TaskNode
            task = TaskNode(
                task_id=getattr(req, "request_id", "unknown"),
                task_type="unknown",
                kv_demand_estimate=kv_demand,
                gpu_type_preference=gpu_type,
                dependency_ids=[],
                slo_deadline=arrival_time_s * 1000.0 + self._hexagent_cfg.slo_budget_ms,
            )
            current_time = time.monotonic() * 1000.0
            native: Any = self._hexagent_native
            prio = native.priority(task, current_time)
            risk = native.slo_risk_score(task, current_time)
            horizon_ms = native.standalone_completion_horizon(task, current_time) - current_time
            return prio, risk, horizon_ms
        else:
            # Inline fallback path
            inline: _InlineHexAGeTScheduler = self._hexagent_native
            req_id = getattr(req, "request_id", "unknown")
            inline.record_request(req_id)
            task_type = inline.get_task_type(req_id)
            arrival = inline.get_arrival_time(req_id)
            prio, risk, horizon_ms = inline.priority(
                req_id, n_tokens, kv_demand, arrival, task_type, gpu_type
            )
            return prio, risk, horizon_ms

    def _hexagent_kv_demand(self, req: Any) -> int:
        """Estimate KV demand in bytes for a request."""
        n_tokens = len(getattr(req, "prompt_token_ids", []) or [])
        return n_tokens * self._hexagent_cfg.kv_size_per_token_bytes

    def _hexagent_annotate_rdma_routing(self, req: Any) -> None:
        """Annotate request with RDMA routing hint (Activity A-2).

        Sets req.hexagent_rdma_local = True if a local PegaFlow cache hit
        is expected, False if RDMA remote routing would be needed.
        """
        if self._hexagent_rdma_router is None:
            req.hexagent_rdma_local = True   # default: assume local
            return
        # Check if any pre-registered segment ID for this request is local
        seg_ids = getattr(req, "hexagent_segment_ids", [])
        has_local_hit = False
        try:
            local_node_id = getattr(self, "_hexagent_local_node_id", "local")
            for seg_id in seg_ids:
                if isinstance(seg_id, (str, bytes)):
                    seg_bytes = seg_id.encode() if isinstance(seg_id, str) else seg_id
                    result = self._hexagent_rdma_router.local_connector.get(seg_bytes.hex())
                    if result is not None:
                        has_local_hit = True
                        break
        except Exception:
            has_local_hit = True
        req.hexagent_rdma_local = has_local_hit

    # ---- schedule() override ---- #

    def schedule(self) -> Any:
        """Intercept schedule() to run HexAGenT pre-scheduling before base logic."""
        self.hexagent_pre_schedule()
        return super().schedule()  # type: ignore[misc]

    # ---- Public utility methods ---- #

    def set_hexagent_rdma_router(self, router: Any) -> None:
        """Wire up a PegaFlowRDMACrossNodeRouter for Activity A-2 routing annotation."""
        self._hexagent_ensure_init()
        self._hexagent_rdma_router = router

    def hexagent_routing_stats(self) -> Dict[str, Any]:
        """Return HexAGenT scheduling statistics."""
        self._hexagent_ensure_init()
        n = max(1, self._hexagent_overhead_n)
        return {
            "schedule_call_count": self._hexagent_call_count,
            "reorder_count": self._hexagent_reorder_count,
            "high_slo_risk_requests": self._hexagent_high_risk_count,
            "scheduling_overhead_mean_us": self._hexagent_overhead_sum_us / n,
            "use_native_src": self._hexagent_use_native,
        }


# ============================================================================
# Factory
# ============================================================================

def make_hexagent_workflow_scheduler_class(
    BaseSchedulerClass: type,
    config: Optional[HexAGeTSchedulerMixinConfig] = None,
    rdma_router: Optional[Any] = None,
) -> type:
    """Build a vLLM v1 Scheduler subclass with HexAGenT DAG scheduling.

    Usage:
        from vllm.v1.core.sched.scheduler import Scheduler
        HexAGeTScheduler = make_hexagent_workflow_scheduler_class(Scheduler)

    The returned class inherits all vLLM Scheduler methods and wraps
    schedule() with hexagent_pre_schedule() (HexAGenT priority + SLO-risk).
    """
    cfg = config or HexAGeTSchedulerMixinConfig()

    class HexAGeTVllmScheduler(HexAGeTWorkflowSchedulerMixin, BaseSchedulerClass):
        """vLLM Scheduler with HexAGenT DAG workflow priority (Activity A)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            hexagent_config = kwargs.pop("hexagent_config", cfg)
            rdma_router_arg = kwargs.pop("hexagent_rdma_router", rdma_router)
            super().__init__(*args, **kwargs)
            self._hexagent_init(hexagent_config, rdma_router_arg)

    HexAGeTVllmScheduler.__name__ = f"HexAGeT_{BaseSchedulerClass.__name__}"
    HexAGeTVllmScheduler.__qualname__ = HexAGeTVllmScheduler.__name__
    return HexAGeTVllmScheduler
