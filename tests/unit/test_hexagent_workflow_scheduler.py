"""Unit tests for HexAGeTWorkflowScheduler (Activity A-1)."""

import time
from typing import Dict, List

import pytest
import torch

from src.scheduler.hexagent_workflow_scheduler import (
    HexAGeTSchedulerConfig,
    HexAGeTWorkflowScheduler,
    TaskNode,
    WorkflowDAG,
)


def _make_config(**kwargs) -> HexAGeTSchedulerConfig:
    defaults = dict(
        schedule_cycle_ms=50.0,
        risk_weight=2.0,
        alpha_gpu_affinity=0.5,
        kv_size_per_token_bytes=512,
        n_layers=32,
        ema_decay=0.9,
        seed=42,
        rdma_bandwidth_table_path="configs/gpu_rdma_bandwidth_table.yaml",
    )
    defaults.update(kwargs)
    return HexAGeTSchedulerConfig(**defaults)


def _make_task(
    task_id: str,
    task_type: str = "tool_call",
    kv_demand: int = 1024,
    gpu_pref: str = "A100",
    dep_ids: List[str] = None,
    slo_deadline: float = None,
    status: str = "ready",
) -> TaskNode:
    if dep_ids is None:
        dep_ids = []
    if slo_deadline is None:
        slo_deadline = time.monotonic() + 10.0
    task = TaskNode(
        task_id=task_id,
        task_type=task_type,
        kv_demand_estimate=kv_demand,
        gpu_type_preference=gpu_pref,
        dependency_ids=dep_ids,
        slo_deadline=slo_deadline,
        status=status,
    )
    return task


def test_workflow_dag_init_from_tasks():
    tasks = [_make_task("t1"), _make_task("t2", dep_ids=["t1"])]
    nodes = {t.task_id: t for t in tasks}
    dag = WorkflowDAG(workflow_id="wf1", nodes=nodes, edges=[("t1", "t2")])
    assert "t1" in dag.nodes
    assert "t2" in dag.nodes
    assert ("t1", "t2") in dag.edges
    assert dag.workflow_id == "wf1"


def test_standalone_completion_horizon_basic():
    config = _make_config()
    scheduler = HexAGeTWorkflowScheduler(config)
    task = _make_task("t1")
    task.t_prefill_ms = 5.0
    task.t_decode_ms = 3.0
    task.t_kv_transfer_ms = 1.0
    current_time = time.monotonic()
    horizon = scheduler.standalone_completion_horizon(task, current_time)
    assert horizon > current_time
    assert abs(horizon - (current_time + 9.0)) < 1e-6


def test_slo_risk_score_zero_when_slack():
    config = _make_config()
    scheduler = HexAGeTWorkflowScheduler(config)
    task = _make_task("t1")
    task.t_prefill_ms = 1.0
    task.t_decode_ms = 1.0
    task.t_kv_transfer_ms = 1.0
    # Deadline far in the future
    task.slo_deadline = time.monotonic() + 1000.0
    risk = scheduler.slo_risk_score(task, time.monotonic())
    assert risk == 0.0


def test_slo_risk_score_positive_when_overdue():
    config = _make_config()
    scheduler = HexAGeTWorkflowScheduler(config)
    task = _make_task("t1")
    task.t_prefill_ms = 500.0
    task.t_decode_ms = 500.0
    task.t_kv_transfer_ms = 500.0
    # Deadline is in the past / very tight
    task.slo_deadline = time.monotonic() + 0.001
    risk = scheduler.slo_risk_score(task, time.monotonic())
    assert risk > 0.0


def test_build_batch_kv_capacity_constraint():
    config = _make_config()
    scheduler = HexAGeTWorkflowScheduler(config)
    t1 = _make_task("t1", kv_demand=500)
    t2 = _make_task("t2", kv_demand=500)
    t3 = _make_task("t3", kv_demand=200)
    # Only 700 bytes available → t1 + t3 fit, t2 should be deferred
    batch = scheduler.build_batch([t1, t2, t3], kv_available_bytes=700)
    total_demand = sum(t.kv_demand_estimate for t in batch)
    assert total_demand <= 700


def test_build_batch_priority_ordering():
    config = _make_config(risk_weight=100.0)
    scheduler = HexAGeTWorkflowScheduler(config)
    t_low = _make_task("t_low", task_type="refine", kv_demand=100)
    t_high = _make_task("t_high", task_type="planning", kv_demand=100)
    # Give t_low a very large prefill latency so horizon far exceeds its deadline
    t_low.t_prefill_ms = 999_999.0
    t_low.slo_deadline = time.monotonic() + 1.0  # deadline 1s out but horizon is huge
    # t_high has plenty of slack
    t_high.slo_deadline = time.monotonic() + 10_000.0
    batch = scheduler.build_batch([t_high, t_low], kv_available_bytes=10_000)
    assert len(batch) == 2
    # t_low should be first due to high SLO risk (large horizon / deadline overflow)
    assert batch[0].task_id == "t_low"


def test_reveal_dag_edges_updates_ready_status():
    config = _make_config()
    scheduler = HexAGeTWorkflowScheduler(config)
    t1 = _make_task("t1", status="pending", dep_ids=[])
    t2 = _make_task("t2", status="pending", dep_ids=["t1"])
    dag = WorkflowDAG(workflow_id="wf1", nodes={"t1": t1, "t2": t2}, edges=[])
    scheduler.register_workflow(dag)

    # After registration t1 should be ready (no deps), t2 still pending
    assert dag.nodes["t1"].status == "ready"
    assert dag.nodes["t2"].status == "pending"

    # Mark t1 done then reveal edges — t2 should become ready
    scheduler.mark_task_done("wf1", "t1")
    scheduler.reveal_dag_edges("wf1", [("t1", "t2")])
    assert dag.nodes["t2"].status == "ready"


def test_gpu_affinity_score_prefers_low_transfer_cost():
    config = _make_config()
    scheduler = HexAGeTWorkflowScheduler(config)
    task = _make_task("t1", gpu_pref="A100", kv_demand=1_000_000)
    task.assigned_gpu = "A100"
    # H200 has higher bandwidth (800 GB/s in table) than default (100 GB/s)
    score_h200 = scheduler.gpu_affinity_score(task, "H200")
    score_unknown = scheduler.gpu_affinity_score(task, "unknown_gpu")
    # Lower absolute value → shorter transfer time → better affinity
    # With higher bandwidth, transfer time is lower, so score_h200 should be less negative
    assert score_h200 > score_unknown


def test_ema_update_execution_time():
    config = _make_config(ema_decay=0.9)
    scheduler = HexAGeTWorkflowScheduler(config)
    task = _make_task("t1")
    task.t_prefill_ms = 0.0

    # After one update with large value, EMA should move toward it
    scheduler.update_task_ema(task, observed_prefill_ms=100.0, observed_decode_ms=0.0, observed_kv_transfer_ms=0.0)
    assert task.t_prefill_ms == pytest.approx(10.0, rel=1e-5)  # 0.9*0 + 0.1*100

    # After repeated updates the value converges
    for _ in range(20):
        scheduler.update_task_ema(task, 100.0, 0.0, 0.0)
    assert task.t_prefill_ms > 50.0  # Should be converging toward 100


def test_schedule_interface_returns_list():
    config = _make_config()
    scheduler = HexAGeTWorkflowScheduler(config)
    tasks = [_make_task("t1"), _make_task("t2")]
    result = scheduler.schedule(tasks)
    assert isinstance(result, list)
    assert len(result) == 2


def test_schedule_deterministic_with_seed():
    config = _make_config(seed=42)
    sched1 = HexAGeTWorkflowScheduler(config)
    sched2 = HexAGeTWorkflowScheduler(config)

    tasks1 = [_make_task(f"t{i}", kv_demand=100 * i) for i in range(1, 5)]
    tasks2 = [_make_task(f"t{i}", kv_demand=100 * i) for i in range(1, 5)]

    result1 = sched1.schedule(tasks1)
    result2 = sched2.schedule(tasks2)

    assert [t.task_id for t in result1] == [t.task_id for t in result2]
