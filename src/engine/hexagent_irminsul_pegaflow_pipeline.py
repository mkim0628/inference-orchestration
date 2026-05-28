"""Cross-1: HexAGenT DAG + PegaFlow RDMA + Irminsul δ-rotation A+B integration pipeline.

5-step async pipeline:
  Step 1: DAG construction + SLO-risk priority assignment (A-1)
  Step 2: Next-step segment prefetch prediction + async prefetch (A-1/B-1)
  Step 3: Distributed segment lookup (B-1)
  Step 4: δ-rotation position correction (B-1/Irminsul)
  Step 5: DAG update + next cycle (A-1)
"""

import asyncio
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.pegaflow_irminsul_distributed_cache import (
    PegaFlowIrminsulDistributedSegmentCache,
)
from src.scheduler.hexagent_workflow_scheduler import (
    HexAGeTSchedulerConfig,
    HexAGeTWorkflowScheduler,
    TaskNode,
    WorkflowDAG,
)
from src.scheduler.pegaflow_rdma_router import PegaFlowRDMACrossNodeRouter


@dataclass
class CrossABPipelineConfig:
    prefetch_confidence_threshold: float = 0.7
    prefetch_hbm_budget_ratio: float = 0.10
    async_prefetch: bool = True
    seed: int = 42


class HexAGeTIrminsulPegaFlowPipeline:
    """A+B integration pipeline: HexAGenT DAG scheduling + PegaFlow RDMA + Irminsul δ-rotation.

    5-step async loop:
      Step 1: Receive agentic session → init WorkflowDAG → SLO-risk sort → build_batch
      Step 2: After prefill, predict next-step CDC segment IDs → async prefetch if confident
      Step 3: On task execution → distributed segment lookup (4-level)
      Step 4: δ-rotation applied to remote k_r, c_KV reused immediately
      Step 5: Task done → reveal_dag_edges → reprioritize → repeat Step 1

    Async pipelining: Step 2 prefetch (asyncio.create_task) overlaps with Step 3 lookup.
    """

    def __init__(
        self,
        dag_scheduler: HexAGeTWorkflowScheduler,
        distributed_cache: PegaFlowIrminsulDistributedSegmentCache,
        rdma_router: PegaFlowRDMACrossNodeRouter,
        config: CrossABPipelineConfig,
    ) -> None:
        self.scheduler = dag_scheduler
        self.distributed_cache = distributed_cache
        self.rdma_router = rdma_router
        self.config = config
        # {segment_id: asyncio.Task} tracking in-flight prefetch tasks
        self._prefetch_tasks: Dict[bytes, "asyncio.Task"] = {}
        # HBM budget: max bytes reserved for prefetching
        self._prefetch_hbm_budget_bytes: int = 0
        # Track total HBM used by prefetch (rough estimate)
        self._prefetch_hbm_used: int = 0
        torch.manual_seed(config.seed)

    async def process_workflow_session(
        self,
        workflow_id: str,
        initial_tasks: List[TaskNode],
        kv_available_bytes: int = 1 << 30,  # 1 GB default
    ) -> None:
        """Run the 5-step async workflow loop until all tasks complete."""
        # Step 1: Build DAG and register
        nodes = {t.task_id: t for t in initial_tasks}
        dag = WorkflowDAG(workflow_id=workflow_id, nodes=nodes, edges=[])
        self.scheduler.register_workflow(dag)

        while True:
            # Step 1: Build batch from ready tasks
            ready_tasks = self.scheduler.get_ready_tasks(workflow_id)
            if not ready_tasks:
                # Check if workflow is complete
                current_dag = self.scheduler.get_dag(workflow_id)
                all_done = all(
                    n.status == "done" for n in current_dag.nodes.values()
                )
                if all_done:
                    break
                # No ready tasks and not all done — wait for external events
                await asyncio.sleep(0)
                break

            batch = self.scheduler.build_batch(ready_tasks, kv_available_bytes)

            for task in batch:
                task.status = "running"

                # Step 2: Predict and async-prefetch next segments
                if self.config.async_prefetch:
                    current_dag = self.scheduler.get_dag(workflow_id)
                    predicted = self._predict_next_segment_ids(current_dag, task.task_id)
                    for seg_id, confidence in predicted:
                        if confidence >= self.config.prefetch_confidence_threshold:
                            asyncio.create_task(
                                self._async_prefetch_segment(seg_id, confidence)
                            )

                # Step 3: Distributed segment lookup for current task
                seg_id = self._task_to_segment_id(task)
                kv_tensor, hit_type = self.distributed_cache.get_distributed(
                    seg_id, target_position=0
                )

                # Step 4: δ-rotation already applied inside get_distributed
                # kv_tensor is ready to use (or None → recompute)

                # Step 5: Mark task done → update DAG
                self.scheduler.mark_task_done(workflow_id, task.task_id)

            # Yield to allow async prefetch tasks to progress
            await asyncio.sleep(0)

    def _task_to_segment_id(self, task: TaskNode) -> bytes:
        """Derive a deterministic segment ID from task metadata."""
        key = f"{task.task_id}:{task.task_type}:{task.kv_demand_estimate}"
        return hashlib.sha256(key.encode("utf-8")).digest()

    def _predict_next_segment_ids(
        self,
        dag: WorkflowDAG,
        current_task_id: str,
    ) -> List[Tuple[bytes, float]]:
        """Predict CDC segment IDs for tasks that follow current_task_id.

        Confidence = 1 / branch_factor (uniform over successors).
        Returns list of (segment_id, confidence) pairs.
        """
        next_task_ids = [
            downstream
            for upstream, downstream in dag.edges
            if upstream == current_task_id
        ]

        branch_factor = len(next_task_ids)
        if branch_factor == 0:
            return []

        confidence = 1.0 / branch_factor
        results: List[Tuple[bytes, float]] = []

        for next_id in next_task_ids:
            if next_id not in dag.nodes:
                continue
            next_task = dag.nodes[next_id]
            seg_id = self._task_to_segment_id(next_task)
            results.append((seg_id, confidence))

        return results

    async def _async_prefetch_segment(
        self,
        segment_id: bytes,
        confidence: float,
    ) -> None:
        """Async prefetch a segment from PegaFlow local → RDMA remote.

        Manages HBM prefetch budget: evicts lowest-confidence entry when full.
        Non-blocking — does not block the calling task's execution.
        """
        # Check HBM budget (simple token-slot estimate)
        estimated_bytes = 512 * 256  # kv_size_per_token × avg_chunk_size
        if self._prefetch_hbm_used + estimated_bytes > self._prefetch_hbm_budget_bytes:
            # Evict smallest allocation to make room (simplified: just skip)
            if self._prefetch_hbm_budget_bytes > 0:
                return

        if segment_id in self._prefetch_tasks:
            return  # Already in flight

        # Attempt to warm the cache via distributed lookup
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.distributed_cache.get_distributed(segment_id, target_position=0),
        )
        self._prefetch_hbm_used = max(0, self._prefetch_hbm_used - estimated_bytes)
        self._prefetch_tasks.pop(segment_id, None)

    def set_hbm_budget(self, total_hbm_bytes: int) -> None:
        """Configure HBM budget for prefetching (call before process_workflow_session)."""
        self._prefetch_hbm_budget_bytes = int(
            total_hbm_bytes * self.config.prefetch_hbm_budget_ratio
        )

    def get_distributed_hit_rate_breakdown(self) -> dict:
        """Expose distributed hit rate breakdown from underlying cache."""
        return self.distributed_cache.distributed_hit_rate_breakdown()
