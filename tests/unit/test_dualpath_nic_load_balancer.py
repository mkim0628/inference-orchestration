"""Unit tests for DualPathNICLoadBalancer (Activity A).

Tests:
  - update_nic_status() state update and retrieval
  - prefill_nic_util < 0.80 -> path="single"
  - prefill_nic_util >= 0.80 + idle decode present -> path="dual"
  - idle decode absent -> path="single"
  - min_load_first sorting
  - round-robin even distribution
  - max_dual_path_per_node=4 exclusion
  - complete_dual_path() active_dual_path decrement
  - decision latency < 0.1ms (p99)
  - schedule() routing_decision attachment
  - scheduling_stats() dict
"""

import time
import dataclasses
import torch
import pytest

from src.scheduler.dualpath_nic_load_balancer import (
    DualPathNICLoadBalancer,
    DualPathNICConfig,
    NodeNICStatus,
    RoutingDecision,
)


# ---- helpers -----------------------------------------------------------------

def _make_lb(
    nic_saturation_threshold: float = 0.80,
    idle_nic_threshold: float = 0.30,
    max_dual_path_per_node: int = 4,
    stale_threshold_ms: float = 5000.0,  # long stale window for tests
    seed: int = 42,
) -> DualPathNICLoadBalancer:
    cfg = DualPathNICConfig(
        nic_saturation_threshold=nic_saturation_threshold,
        idle_nic_threshold=idle_nic_threshold,
        max_dual_path_per_node=max_dual_path_per_node,
        stale_threshold_ms=stale_threshold_ms,
        seed=seed,
    )
    return DualPathNICLoadBalancer(cfg)


@dataclasses.dataclass
class FakeRequest:
    request_id: str = "req-0"
    routing_decision: object = None


def _setup_prefill_and_decode(
    lb: DualPathNICLoadBalancer,
    prefill_util: float = 0.85,
    n_decode: int = 2,
    decode_util: float = 0.10,
) -> None:
    lb.update_nic_status("prefill-0", "prefill", prefill_util)
    for i in range(n_decode):
        lb.update_nic_status(f"decode-{i}", "decode", decode_util, active_dual_path=0)


# ---- update_nic_status -------------------------------------------------------

def test_update_nic_status_stores_node() -> None:
    lb = _make_lb()
    lb.update_nic_status("node-1", "prefill", 0.50)
    assert "node-1" in lb._node_status
    status = lb._node_status["node-1"]
    assert status.node_type == "prefill"
    assert abs(status.nic_utilization - 0.50) < 1e-9


def test_update_nic_status_overwrite() -> None:
    lb = _make_lb()
    lb.update_nic_status("n1", "decode", 0.10)
    lb.update_nic_status("n1", "decode", 0.25)
    assert abs(lb._node_status["n1"].nic_utilization - 0.25) < 1e-9


def test_update_nic_status_active_dual_path_stored() -> None:
    lb = _make_lb()
    lb.update_nic_status("d1", "decode", 0.10, active_dual_path=2)
    assert lb._node_status["d1"].active_dual_path == 2


# ---- single path when prefill NIC below threshold ----------------------------

def test_single_path_when_prefill_nic_below_threshold() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80)
    lb.update_nic_status("pf0", "prefill", 0.60)
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "single", f"expected single, got {decision.path}"
    assert decision.relay_decode_node_id is None


def test_single_path_when_no_prefill_nodes_registered() -> None:
    lb = _make_lb()
    # No nodes registered: prefill_util defaults to 0.0 < threshold
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "single"


def test_single_path_when_exactly_at_threshold_minus_epsilon() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80)
    lb.update_nic_status("pf0", "prefill", 0.7999)
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "single"


# ---- dual path when prefill NIC saturated and idle decode present ------------

def test_dual_path_when_prefill_saturated_and_decode_idle() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80, idle_nic_threshold=0.30)
    _setup_prefill_and_decode(lb, prefill_util=0.85, n_decode=2, decode_util=0.10)
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "dual", f"expected dual, got {decision.path}"
    assert decision.relay_decode_node_id is not None
    assert "decode" in decision.relay_decode_node_id


def test_dual_path_relay_node_is_decode_type() -> None:
    lb = _make_lb()
    lb.update_nic_status("pf0", "prefill", 0.90)
    lb.update_nic_status("dc0", "decode", 0.05)
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "dual"
    assert lb._node_status[decision.relay_decode_node_id].node_type == "decode"


# ---- fallback to single when no idle decode ----------------------------------

def test_single_fallback_when_no_idle_decode() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80, idle_nic_threshold=0.30)
    lb.update_nic_status("pf0", "prefill", 0.90)
    # Decode nodes are busy (above idle threshold)
    lb.update_nic_status("dc0", "decode", 0.50)
    lb.update_nic_status("dc1", "decode", 0.80)
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "single"


def test_single_fallback_no_decode_nodes_at_all() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80)
    lb.update_nic_status("pf0", "prefill", 0.95)
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "single"


# ---- min_load_first sorting --------------------------------------------------

def test_min_load_first_selects_lowest_util_decode() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80, idle_nic_threshold=0.50)
    lb.update_nic_status("pf0", "prefill", 0.90)
    lb.update_nic_status("dc0", "decode", 0.40)  # higher util
    lb.update_nic_status("dc1", "decode", 0.05)  # lower util — should be chosen first
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "dual"
    # First round-robin slot (rr_index=0) from min-load-first sorted list
    assert decision.relay_decode_node_id == "dc1"


# ---- round-robin distribution ------------------------------------------------

def test_round_robin_distributes_across_idle_decode_nodes() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80, idle_nic_threshold=0.50)
    lb.update_nic_status("pf0", "prefill", 0.90)
    lb.update_nic_status("dc0", "decode", 0.10)
    lb.update_nic_status("dc1", "decode", 0.15)

    assigned = {}
    for i in range(10):
        req = FakeRequest(request_id=f"r{i}")
        # Reset active_dual_path so nodes stay eligible
        lb._node_status["dc0"].active_dual_path = 0
        lb._node_status["dc1"].active_dual_path = 0
        d = lb.decide_routing(req)
        if d.path == "dual":
            nid = d.relay_decode_node_id
            assigned[nid] = assigned.get(nid, 0) + 1

    assert len(assigned) == 2, "round-robin should use both decode nodes"
    # Both should receive roughly equal assignments
    counts = list(assigned.values())
    assert abs(counts[0] - counts[1]) <= 2


# ---- max_dual_path_per_node exclusion ----------------------------------------

def test_max_dual_path_exceeded_node_excluded() -> None:
    lb = _make_lb(
        nic_saturation_threshold=0.80,
        idle_nic_threshold=0.50,
        max_dual_path_per_node=4,
    )
    lb.update_nic_status("pf0", "prefill", 0.90)
    # dc0 already at max capacity
    lb.update_nic_status("dc0", "decode", 0.10, active_dual_path=4)
    # dc1 has capacity
    lb.update_nic_status("dc1", "decode", 0.15, active_dual_path=0)

    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "dual"
    assert decision.relay_decode_node_id == "dc1", (
        f"expected dc1, got {decision.relay_decode_node_id}"
    )


def test_all_decode_nodes_at_max_fallback_single() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80, max_dual_path_per_node=2)
    lb.update_nic_status("pf0", "prefill", 0.90)
    lb.update_nic_status("dc0", "decode", 0.10, active_dual_path=2)
    lb.update_nic_status("dc1", "decode", 0.10, active_dual_path=2)
    req = FakeRequest()
    decision = lb.decide_routing(req)
    assert decision.path == "single"


# ---- complete_dual_path ------------------------------------------------------

def test_complete_dual_path_decrements_active() -> None:
    lb = _make_lb()
    lb.update_nic_status("dc0", "decode", 0.10, active_dual_path=3)
    lb.complete_dual_path("dc0")
    assert lb._node_status["dc0"].active_dual_path == 2


def test_complete_dual_path_clamps_to_zero() -> None:
    lb = _make_lb()
    lb.update_nic_status("dc0", "decode", 0.10, active_dual_path=0)
    lb.complete_dual_path("dc0")
    assert lb._node_status["dc0"].active_dual_path == 0


def test_complete_dual_path_unknown_node_no_error() -> None:
    lb = _make_lb()
    lb.complete_dual_path("nonexistent-node")  # should not raise


# ---- decision latency --------------------------------------------------------

def test_decision_latency_below_01ms_p99() -> None:
    """100 routing decisions: p99 decision_latency_ms < 0.1ms."""
    lb = _make_lb(stale_threshold_ms=60000.0)
    lb.update_nic_status("pf0", "prefill", 0.50)

    for i in range(100):
        lb.decide_routing(FakeRequest(request_id=f"r{i}"))

    p99 = lb.p99_decision_latency_ms()
    assert p99 < 0.1, f"p99 decision_latency={p99:.4f}ms >= 0.1ms"


# ---- schedule() interface ----------------------------------------------------

def test_schedule_attaches_routing_decision() -> None:
    lb = _make_lb()
    lb.update_nic_status("pf0", "prefill", 0.50)
    requests = [FakeRequest(request_id=f"r{i}") for i in range(5)]
    result = lb.schedule(requests)
    assert len(result) == 5
    for req in result:
        assert req.routing_decision is not None
        assert isinstance(req.routing_decision, RoutingDecision)


def test_schedule_returns_all_requests() -> None:
    lb = _make_lb()
    lb.update_nic_status("pf0", "prefill", 0.90)
    lb.update_nic_status("dc0", "decode", 0.10)
    requests = [FakeRequest(request_id=f"r{i}") for i in range(10)]
    result = lb.schedule(requests)
    assert len(result) == 10


# ---- scheduling_stats --------------------------------------------------------

def test_scheduling_stats_returns_dict() -> None:
    lb = _make_lb()
    lb.update_nic_status("pf0", "prefill", 0.60)
    for i in range(5):
        lb.decide_routing(FakeRequest(request_id=f"r{i}"))
    stats = lb.scheduling_stats()
    assert isinstance(stats, dict)
    assert "single_path_count" in stats
    assert "dual_path_count" in stats
    assert "dual_path_ratio" in stats
    assert "decision_latency_p99_ms" in stats
    assert "decision_latency_mean_ms" in stats


def test_scheduling_stats_counts_match_decisions() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80)
    lb.update_nic_status("pf0", "prefill", 0.50)
    for i in range(7):
        lb.decide_routing(FakeRequest(request_id=f"r{i}"))
    stats = lb.scheduling_stats()
    assert stats["single_path_count"] == 7
    assert stats["dual_path_count"] == 0
    assert stats["dual_path_ratio"] == 0.0


def test_reset_stats() -> None:
    lb = _make_lb()
    lb.update_nic_status("pf0", "prefill", 0.60)
    lb.decide_routing(FakeRequest())
    lb.reset_stats()
    stats = lb.scheduling_stats()
    assert stats["single_path_count"] == 0
    assert stats["dual_path_count"] == 0
    assert stats["decision_latency_p99_ms"] == 0.0


# ---- dual_path_ratio ---------------------------------------------------------

def test_dual_path_ratio_zero_when_all_single() -> None:
    lb = _make_lb()
    lb.update_nic_status("pf0", "prefill", 0.50)
    for i in range(5):
        lb.decide_routing(FakeRequest(request_id=f"r{i}"))
    assert lb.dual_path_ratio() == 0.0


def test_dual_path_ratio_positive_when_dual_used() -> None:
    lb = _make_lb(nic_saturation_threshold=0.80)
    lb.update_nic_status("pf0", "prefill", 0.90)
    lb.update_nic_status("dc0", "decode", 0.10)
    for i in range(4):
        lb._node_status["dc0"].active_dual_path = 0
        lb.decide_routing(FakeRequest(request_id=f"r{i}"))
    assert lb.dual_path_ratio() > 0.0
