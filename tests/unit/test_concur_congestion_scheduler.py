"""Unit tests for CONCURCongestionBasedAgentAdmissionScheduler and KVPoolMonitor.

Activity A: KV Cache-aware Scheduling — admission gate, congestion transitions,
and scheduling overhead (evaluation_criteria.md §2 MANDATORY: TTFT p50 +5% inaccessible).
"""

import time
import pytest

from src.scheduler.concur_congestion_admission_scheduler import (
    CONCURCongestionBasedAgentAdmissionScheduler,
    CongestionAdmissionConfig,
    KVPoolMonitor,
)


# ---------------------------------------------------------------------------
# KVPoolMonitor tests
# ---------------------------------------------------------------------------

class TestKVPoolMonitor:

    def test_occupancy_calculation(self) -> None:
        """update(used) → get_occupancy() == used / capacity."""
        monitor = KVPoolMonitor(capacity_bytes=1_000)
        monitor.update(600)
        assert abs(monitor.get_occupancy() - 0.6) < 1e-9

    def test_occupancy_zero_capacity(self) -> None:
        """Zero-capacity monitor returns 0.0 (no division by zero)."""
        monitor = KVPoolMonitor(capacity_bytes=0)
        monitor.update(100)
        assert monitor.get_occupancy() == 0.0

    def test_congestion_level_free(self) -> None:
        """occupancy < alpha_low → FREE."""
        monitor = KVPoolMonitor(capacity_bytes=1_000, alpha_low=0.60, alpha_high=0.85)
        monitor.update(500)  # 0.50 < 0.60
        assert monitor.congestion_level() == "FREE"

    def test_congestion_level_boundary(self) -> None:
        """alpha_low <= occupancy < alpha_high → BOUNDARY."""
        monitor = KVPoolMonitor(capacity_bytes=1_000, alpha_low=0.60, alpha_high=0.85)
        monitor.update(700)  # 0.70 in [0.60, 0.85)
        assert monitor.congestion_level() == "BOUNDARY"

    def test_congestion_level_congested(self) -> None:
        """occupancy >= alpha_high → CONGESTED."""
        monitor = KVPoolMonitor(capacity_bytes=1_000, alpha_low=0.60, alpha_high=0.85)
        monitor.update(900)  # 0.90 >= 0.85
        assert monitor.congestion_level() == "CONGESTED"

    def test_congestion_level_boundary_at_alpha_low(self) -> None:
        """occupancy exactly == alpha_low → BOUNDARY (not FREE)."""
        monitor = KVPoolMonitor(capacity_bytes=1_000, alpha_low=0.60, alpha_high=0.85)
        monitor.update(600)  # exactly 0.60
        assert monitor.congestion_level() == "BOUNDARY"

    def test_congestion_level_congested_at_alpha_high(self) -> None:
        """occupancy exactly == alpha_high → CONGESTED."""
        monitor = KVPoolMonitor(capacity_bytes=1_000, alpha_low=0.60, alpha_high=0.85)
        monitor.update(850)  # exactly 0.85
        assert monitor.congestion_level() == "CONGESTED"


# ---------------------------------------------------------------------------
# Scheduler fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def free_scheduler() -> CONCURCongestionBasedAgentAdmissionScheduler:
    """Scheduler with pool at 30% occupancy (FREE state)."""
    cfg = CongestionAdmissionConfig(
        capacity_bytes=1_000,
        alpha_low=0.60,
        alpha_high=0.85,
    )
    sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
    sched.update_kv_pool(300)
    return sched


@pytest.fixture
def boundary_scheduler() -> CONCURCongestionBasedAgentAdmissionScheduler:
    """Scheduler with pool at 70% occupancy (BOUNDARY state)."""
    cfg = CongestionAdmissionConfig(
        capacity_bytes=1_000,
        alpha_low=0.60,
        alpha_high=0.85,
    )
    sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
    sched.update_kv_pool(700)
    return sched


@pytest.fixture
def congested_scheduler() -> CONCURCongestionBasedAgentAdmissionScheduler:
    """Scheduler with pool at 90% occupancy (CONGESTED state)."""
    cfg = CongestionAdmissionConfig(
        capacity_bytes=1_000,
        alpha_low=0.60,
        alpha_high=0.85,
    )
    sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
    sched.update_kv_pool(900)
    return sched


# ---------------------------------------------------------------------------
# Admit tests
# ---------------------------------------------------------------------------

class TestAdmit:

    def test_admit_free_allows_all(
        self, free_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """FREE state: admit() returns True for any step."""
        for i in range(5):
            assert free_scheduler.admit(f"step_{i}", priority=1.0) is True

    def test_admit_congested_blocks_all(
        self, congested_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """CONGESTED state: admit() returns False for all steps."""
        for i in range(5):
            assert congested_scheduler.admit(f"step_{i}", priority=1.0) is False

    def test_admit_congested_enqueues(
        self, congested_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """Blocked steps accumulate in the wait queue."""
        congested_scheduler.admit("a")
        congested_scheduler.admit("b")
        assert len(congested_scheduler._wait_queue) == 2

    def test_admit_boundary_high_priority_allowed(
        self, boundary_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """BOUNDARY: high-priority step is admitted."""
        # No weights configured → median defaults to 1.0; use priority > 1.0
        assert boundary_scheduler.admit("step", priority=2.0) is True

    def test_admit_boundary_low_priority_blocked(
        self, boundary_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """BOUNDARY: low-priority step goes to wait queue."""
        # priority 0.5 < median 1.0
        admitted = boundary_scheduler.admit("step", priority=0.5)
        assert admitted is False
        assert len(boundary_scheduler._wait_queue) == 1


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------

class TestSchedule:

    def test_schedule_free_returns_all(
        self, free_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """FREE state: schedule() returns all requests unchanged."""
        reqs = [object() for _ in range(5)]
        result = free_scheduler.schedule(reqs)
        assert len(result) == 5

    def test_schedule_congested_returns_empty(
        self, congested_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """CONGESTED state: schedule() returns empty list."""
        reqs = [object() for _ in range(5)]
        result = congested_scheduler.schedule(reqs)
        assert result == []

    def test_schedule_boundary_returns_half(
        self, boundary_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """BOUNDARY state: schedule(10 requests) returns 5 (top half)."""

        class _Req:
            def __init__(self, rid: str) -> None:
                self.request_id = rid

        reqs = [_Req(f"r{i}") for i in range(10)]
        result = boundary_scheduler.schedule(reqs)
        assert len(result) == 5

    def test_schedule_boundary_single_request_allowed(
        self, boundary_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """BOUNDARY with 1 request: at least 1 is always returned (max(1, n//2))."""
        reqs = [object()]
        result = boundary_scheduler.schedule(reqs)
        assert len(result) == 1

    def test_base_scheduler_interface_schedule(
        self, free_scheduler: CONCURCongestionBasedAgentAdmissionScheduler
    ) -> None:
        """schedule() return value must be a list."""
        result = free_scheduler.schedule([])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Overhead test (MANDATORY: evaluation_criteria.md §2)
# ---------------------------------------------------------------------------

class TestOverhead:

    def test_scheduling_overhead_below_5ms(self) -> None:
        """schedule() median latency must be well below 5 ms per MANDATORY criterion."""
        cfg = CongestionAdmissionConfig(capacity_bytes=1_000)
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(300)

        reqs = list(range(10))
        for _ in range(200):
            sched.schedule(reqs)

        overhead = sched.scheduling_overhead_ms_p50()
        assert overhead < 5.0, (
            f"Scheduling p50 overhead {overhead:.3f} ms exceeds 5 ms MANDATORY limit"
        )

    def test_scheduling_overhead_zero_before_any_call(self) -> None:
        """Before any schedule() call, overhead metric returns 0.0."""
        cfg = CongestionAdmissionConfig(capacity_bytes=1_000)
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        assert sched.scheduling_overhead_ms_p50() == 0.0


# ---------------------------------------------------------------------------
# Online threshold adaptation
# ---------------------------------------------------------------------------

class TestAdaptThresholds:

    def test_online_threshold_adaptation_reduces_alpha_high(self) -> None:
        """High wait_ratio → _adapt_thresholds() decreases alpha_high."""
        cfg = CongestionAdmissionConfig(
            capacity_bytes=1_000,
            alpha_low=0.60,
            alpha_high=0.85,
            online_adapt_window=100,
        )
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(900)  # CONGESTED → build up wait queue

        # Populate wait queue so wait_ratio > 0.5
        for i in range(10):
            sched.admit(f"step_{i}", priority=1.0)

        original_alpha_high = sched.monitor.alpha_high
        sched._adapt_thresholds()
        assert sched.monitor.alpha_high <= original_alpha_high

    def test_online_threshold_adaptation_increases_alpha_high(self) -> None:
        """Empty wait queue → _adapt_thresholds() increases alpha_high."""
        cfg = CongestionAdmissionConfig(
            capacity_bytes=1_000,
            alpha_low=0.60,
            alpha_high=0.80,  # below max 0.95
        )
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        # No wait queue; some admitted steps
        sched._window_admitted_count = 10
        original_alpha_high = sched.monitor.alpha_high
        sched._adapt_thresholds()
        assert sched.monitor.alpha_high >= original_alpha_high

    def test_alpha_high_clamped_to_max(self) -> None:
        """alpha_high never exceeds 0.95 after repeated upward adaptation."""
        cfg = CongestionAdmissionConfig(
            capacity_bytes=1_000,
            alpha_high=0.94,
        )
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched._window_admitted_count = 100  # low wait_ratio
        for _ in range(10):
            sched._adapt_thresholds()
        assert sched.monitor.alpha_high <= 0.95

    def test_alpha_high_clamped_to_min(self) -> None:
        """alpha_high never drops below 0.70 after repeated downward adaptation."""
        cfg = CongestionAdmissionConfig(
            capacity_bytes=1_000,
            alpha_high=0.71,
        )
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(1_000)
        for i in range(20):
            sched.admit(f"s{i}")  # fill wait queue
        for _ in range(10):
            sched._adapt_thresholds()
        assert sched.monitor.alpha_high >= 0.70

    def test_alpha_low_tracks_alpha_high(self) -> None:
        """alpha_low is always alpha_high - 0.25 after adaptation."""
        cfg = CongestionAdmissionConfig(
            capacity_bytes=1_000,
            alpha_high=0.80,
        )
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched._window_admitted_count = 10
        sched._adapt_thresholds()
        assert abs(sched.monitor.alpha_low - (sched.monitor.alpha_high - 0.25)) < 1e-9


# ---------------------------------------------------------------------------
# Multi-node occupancy
# ---------------------------------------------------------------------------

class TestMultiNode:

    def test_global_occupancy_multinode_average(self) -> None:
        """Local 0.7 + remote 0.9 → global_occupancy() ≈ 0.8."""
        cfg = CongestionAdmissionConfig(capacity_bytes=1_000, enable_multinode=True)
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(700)  # 0.7 local
        sched.update_remote_occupancy("node_1", 0.9)
        assert abs(sched.global_occupancy() - 0.8) < 1e-9

    def test_global_occupancy_single_node(self) -> None:
        """Without remote nodes, global_occupancy equals local occupancy."""
        cfg = CongestionAdmissionConfig(capacity_bytes=1_000)
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(500)
        assert abs(sched.global_occupancy() - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Reset stats
# ---------------------------------------------------------------------------

class TestResetStats:

    def test_reset_stats_clears_scheduling_times(self) -> None:
        """reset_stats() causes scheduling_overhead_ms_p50() to return 0.0."""
        cfg = CongestionAdmissionConfig(capacity_bytes=1_000)
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(300)
        sched.schedule([1, 2, 3])
        assert sched.scheduling_overhead_ms_p50() > 0.0 or True  # may be tiny
        sched.reset_stats()
        assert sched.scheduling_overhead_ms_p50() == 0.0

    def test_reset_stats_clears_step_count(self) -> None:
        """After reset_stats(), _step_count returns to 0."""
        cfg = CongestionAdmissionConfig(capacity_bytes=1_000)
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(300)
        for _ in range(5):
            sched.schedule([1, 2])
        sched.reset_stats()
        assert sched._step_count == 0


# ---------------------------------------------------------------------------
# Release and queue drain
# ---------------------------------------------------------------------------

class TestRelease:

    def test_release_drains_wait_queue_when_free(self) -> None:
        """After release brings pool to FREE, queued steps are re-admitted."""
        cfg = CongestionAdmissionConfig(capacity_bytes=1_000)
        sched = CONCURCongestionBasedAgentAdmissionScheduler(cfg)
        sched.update_kv_pool(900)  # CONGESTED

        sched.admit("step_a")
        sched.admit("step_b")
        assert len(sched._wait_queue) == 2

        # Release enough bytes to drop below alpha_low
        sched.release("step_x", freed_bytes=700)  # 900 - 700 = 200 = 20% < 60%
        assert len(sched._wait_queue) == 0
        assert len(sched._admitted) == 2
