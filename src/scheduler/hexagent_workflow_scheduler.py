"""Activity A-1: HexAGenT Workflow Horizon-Aware KV Capacity Scheduler.

Online-public DAG workflow scheduler for agentic LLM workloads.
Reference: HexAGenT arXiv 2605.16637
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch
import yaml

from src.scheduler.base import BaseScheduler


TaskStatus = Literal["pending", "ready", "running", "done"]

# Base priority by task type (higher = processed first)
_TASK_TYPE_BASE_PRIORITY: Dict[str, float] = {
    "planning": 4.0,
    "synthesis": 3.0,
    "tool_call": 2.0,
    "refine": 1.0,
}

# Approximate GPU throughput in bytes/ms for KV transfer size estimation
_GPU_THROUGHPUT_BYTES_PER_MS: Dict[str, float] = {
    "A100": 2_000_000.0,   # ~2 GB/s effective throughput
    "H100": 4_000_000.0,
    "H200": 6_000_000.0,
    "any": 2_000_000.0,
}


@dataclass
class TaskNode:
    task_id: str
    task_type: Literal["planning", "tool_call", "synthesis", "refine"]
    kv_demand_estimate: int           # estimated KV bytes
    gpu_type_preference: str          # "A100" | "H100" | "H200" | "any"
    dependency_ids: List[str]         # preceding task_id list
    slo_deadline: float               # absolute monotonic timestamp
    status: TaskStatus = "pending"
    # Runtime estimates (EMA-updated)
    t_prefill_ms: float = 0.0
    t_decode_ms: float = 0.0
    t_kv_transfer_ms: float = 0.0
    assigned_gpu: Optional[str] = None


@dataclass
class WorkflowDAG:
    workflow_id: str
    nodes: Dict[str, TaskNode]        # task_id → TaskNode
    edges: List[Tuple[str, str]]      # (upstream_id, downstream_id)
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class HexAGeTSchedulerConfig:
    schedule_cycle_ms: float = 50.0
    risk_weight: float = 2.0
    alpha_gpu_affinity: float = 0.5
    heartbeat_interval_ms: float = 100.0
    kv_size_per_token_bytes: int = 512
    n_layers: int = 32
    ema_decay: float = 0.9
    rdma_bandwidth_table_path: str = "configs/gpu_rdma_bandwidth_table.yaml"
    seed: int = 42


class HexAGeTWorkflowScheduler(BaseScheduler):
    """HexAGenT online-public DAG workflow scheduler.

    3-way optimization:
      1. Standalone completion horizon estimation per task.
      2. SLO-risk-weighted priority for ready task scheduling.
      3. KV capacity constraint + heterogeneous GPU transfer latency in objective.
    """

    def __init__(self, config: HexAGeTSchedulerConfig) -> None:
        self.config = config
        self._dags: Dict[str, WorkflowDAG] = {}
        # Eviction registry: (task_id, kv_bytes) for currently-deferred low-priority tasks
        self._deferred_kv: Dict[str, int] = {}
        self._rdma_bandwidth: Dict[str, float] = {}
        torch.manual_seed(config.seed)
        self._load_rdma_bandwidth_table()

    def _load_rdma_bandwidth_table(self) -> None:
        try:
            with open(self.config.rdma_bandwidth_table_path) as f:
                data = yaml.safe_load(f)
            self._rdma_bandwidth = data.get("rdma_bandwidth_gbps", {})
        except (FileNotFoundError, Exception):
            # Fallback defaults when YAML not accessible
            self._rdma_bandwidth = {
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
        alt_key = f"{tgt_gpu}_to_{src_gpu}"
        return (
            self._rdma_bandwidth.get(key)
            or self._rdma_bandwidth.get(alt_key)
            or self._rdma_bandwidth.get("default", 100.0)
        )

    def register_workflow(self, dag: WorkflowDAG) -> None:
        """Register a new workflow DAG and compute initial ready statuses."""
        self._dags[dag.workflow_id] = dag
        self._update_ready_status(dag)

    def _update_ready_status(self, dag: WorkflowDAG) -> None:
        """Recompute pending → ready transitions based on resolved dependencies."""
        # Build set of completed task_ids
        done_ids: Set[str] = {
            tid for tid, node in dag.nodes.items() if node.status == "done"
        }
        for tid, node in dag.nodes.items():
            if node.status == "pending":
                if all(dep in done_ids for dep in node.dependency_ids):
                    node.status = "ready"

    def reveal_dag_edges(
        self,
        workflow_id: str,
        new_edges: List[Tuple[str, str]],
        new_nodes: Optional[Dict[str, TaskNode]] = None,
    ) -> None:
        """Append newly revealed edges/nodes to an existing DAG (online-public DAG).

        Called when a task completes and its successors become known.
        """
        dag = self._dags[workflow_id]
        dag.edges.extend(new_edges)
        if new_nodes:
            dag.nodes.update(new_nodes)
        self._update_ready_status(dag)

    def mark_task_done(self, workflow_id: str, task_id: str) -> None:
        """Mark a task as done and trigger ready-status propagation."""
        dag = self._dags[workflow_id]
        dag.nodes[task_id].status = "done"
        self._update_ready_status(dag)

    def standalone_completion_horizon(
        self,
        task: TaskNode,
        current_time: float,
    ) -> float:
        """Estimate standalone (independent) completion time horizon.

        horizon = current_time + t_prefill_ms + t_decode_ms + t_kv_transfer_ms
        """
        return (
            current_time
            + task.t_prefill_ms
            + task.t_decode_ms
            + task.t_kv_transfer_ms
        )

    def update_task_ema(
        self,
        task: TaskNode,
        observed_prefill_ms: float,
        observed_decode_ms: float,
        observed_kv_transfer_ms: float,
    ) -> None:
        """EMA-update runtime estimates after task observation."""
        alpha = 1.0 - self.config.ema_decay
        task.t_prefill_ms = (
            self.config.ema_decay * task.t_prefill_ms + alpha * observed_prefill_ms
        )
        task.t_decode_ms = (
            self.config.ema_decay * task.t_decode_ms + alpha * observed_decode_ms
        )
        task.t_kv_transfer_ms = (
            self.config.ema_decay * task.t_kv_transfer_ms
            + alpha * observed_kv_transfer_ms
        )

    def slo_risk_score(self, task: TaskNode, current_time: float) -> float:
        """SLO risk score: 0.0 = slack, 1.0+ = SLO violation imminent.

        risk = max(0, horizon - deadline) / deadline
        """
        horizon = self.standalone_completion_horizon(task, current_time)
        if task.slo_deadline <= 0.0:
            return 0.0
        risk = max(0.0, horizon - task.slo_deadline) / task.slo_deadline
        return risk

    def priority(self, task: TaskNode, current_time: float) -> float:
        """Scheduling priority (higher = processed first).

        priority = base_priority + risk_weight × slo_risk_score
        """
        base = _TASK_TYPE_BASE_PRIORITY.get(task.task_type, 1.0)
        risk = self.slo_risk_score(task, current_time)
        return base + self.config.risk_weight * risk

    def gpu_affinity_score(self, task: TaskNode, gpu_id: str) -> float:
        """Heterogeneous GPU affinity score (lower = preferred).

        score = -T_kv_transfer_to(gpu_id) - alpha × T_exec(task, gpu_id)
        """
        src_gpu = task.assigned_gpu or task.gpu_type_preference
        bw_gbps = self._rdma_bandwidth_gbps(src_gpu, gpu_id)
        # Convert GB/s → bytes/ms
        bw_bytes_per_ms = bw_gbps * 1e9 / 1000.0
        if bw_bytes_per_ms > 0:
            t_transfer = task.kv_demand_estimate / bw_bytes_per_ms
        else:
            t_transfer = float("inf")

        throughput = _GPU_THROUGHPUT_BYTES_PER_MS.get(gpu_id, 2_000_000.0)
        t_exec = task.kv_demand_estimate / throughput

        return -t_transfer - self.config.alpha_gpu_affinity * t_exec

    def _try_evict_to_fit(self, task: TaskNode, remaining_kv: int) -> int:
        """Attempt to evict deferred low-priority KV to make room for task.

        Returns bytes freed (may be 0 if nothing to evict).
        """
        freed = 0
        # Evict deferred tasks with smallest KV until enough space
        sorted_deferred = sorted(self._deferred_kv.items(), key=lambda x: x[1])
        for defer_id, defer_kv in sorted_deferred:
            if freed + defer_kv >= task.kv_demand_estimate - remaining_kv:
                freed += defer_kv
                del self._deferred_kv[defer_id]
                break
            freed += defer_kv
            del self._deferred_kv[defer_id]
        return freed

    def build_batch(
        self,
        ready_tasks: List[TaskNode],
        kv_available_bytes: int,
    ) -> List[TaskNode]:
        """Build a batch respecting KV capacity, ordered by SLO-risk priority.

        Tasks that don't fit are added to deferred-eviction registry.
        """
        current_time = time.monotonic()
        sorted_tasks = sorted(
            ready_tasks,
            key=lambda t: self.priority(t, current_time),
            reverse=True,
        )
        batch: List[TaskNode] = []
        remaining_kv = kv_available_bytes
        for task in sorted_tasks:
            if task.kv_demand_estimate <= remaining_kv:
                batch.append(task)
                remaining_kv -= task.kv_demand_estimate
            else:
                evicted = self._try_evict_to_fit(task, remaining_kv)
                net_remaining = remaining_kv + evicted
                if task.kv_demand_estimate <= net_remaining:
                    batch.append(task)
                    remaining_kv = net_remaining - task.kv_demand_estimate
                else:
                    # Defer this task — register its KV demand for future eviction
                    self._deferred_kv[task.task_id] = task.kv_demand_estimate
        return batch

    def get_ready_tasks(self, workflow_id: str) -> List[TaskNode]:
        """Return all ready tasks in the workflow DAG."""
        dag = self._dags[workflow_id]
        return [n for n in dag.nodes.values() if n.status == "ready"]

    def get_dag(self, workflow_id: str) -> WorkflowDAG:
        """Return DAG for the given workflow."""
        return self._dags[workflow_id]

    def schedule(self, requests: List) -> List:
        """BaseScheduler interface implementation.

        Converts requests (TaskNode or plain objects with task_id) to a
        priority-sorted list. Falls back to identity sort for non-TaskNode inputs.
        """
        if not requests:
            return requests
        if isinstance(requests[0], TaskNode):
            current_time = time.monotonic()
            return sorted(
                requests,
                key=lambda t: self.priority(t, current_time),
                reverse=True,
            )
        return requests
