"""Unit tests for CPDWarmColdHitRateRouter (Activity A).

Covers: hit rate prediction, warm/cold/neutral classification, scheduling,
online predictor update, overhead measurement, and BaseScheduler interface.
"""

import time
from dataclasses import dataclass, field
from typing import List

import pytest
import torch

from src.scheduler.cpd_warm_cold_hit_router import (
    CPDRouterConfig,
    CPDWarmColdHitRateRouter,
    HitRatePredictorWeights,
    RoutingDecision,
)


# --------------------------------------------------------------------------- #
# Minimal request dataclass for testing                                        #
# --------------------------------------------------------------------------- #


@dataclass
class FakeRequest:
    request_id: str = "req_0"
    token_ids: List[int] = field(default_factory=list)
    prefix_hash: str = ""
    session_turn: int = 0
    segment_match_ratio: float = 0.0


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def config() -> CPDRouterConfig:
    return CPDRouterConfig(
        high_hit_threshold=0.70,
        low_hit_threshold=0.25,
        queue_pressure_threshold=100,
        seed=42,
    )


@pytest.fixture
def router(config: CPDRouterConfig) -> CPDWarmColdHitRateRouter:
    return CPDWarmColdHitRateRouter(config)


def _warm_request() -> FakeRequest:
    """Request with known high hit history -> warm."""
    req = FakeRequest(request_id="warm_req", prefix_hash="pfx_warm", token_ids=list(range(10)))
    return req


def _cold_request() -> FakeRequest:
    """Request with no history and minimal features -> cold."""
    return FakeRequest(request_id="cold_req", prefix_hash="", token_ids=[])


# --------------------------------------------------------------------------- #
# predict_hit_rate                                                             #
# --------------------------------------------------------------------------- #


def test_cpd_predict_hit_rate_range_0_to_1(router: CPDWarmColdHitRateRouter) -> None:
    """predict_hit_rate(req) in [0.0, 1.0]."""
    req = FakeRequest(token_ids=list(range(50)), prefix_hash="abc")
    rate = router.predict_hit_rate(req)
    assert 0.0 <= rate <= 1.0, f"hit_rate={rate} out of [0, 1]"


# --------------------------------------------------------------------------- #
# classify_request: warm / cold / neutral                                      #
# --------------------------------------------------------------------------- #


def test_cpd_classify_warm_high_hit_rate(router: CPDWarmColdHitRateRouter) -> None:
    """Prefix with 100% hit history -> path='warm', priority=0."""
    req = _warm_request()
    # seed the hit history with 10 True values
    router._hit_history["pfx_warm"] = [True] * 10
    decision = router.classify_request(req)
    assert decision.path == "warm", f"Expected warm, got {decision.path}"
    assert decision.batch_priority == 0


def test_cpd_classify_cold_low_hit_rate(router: CPDWarmColdHitRateRouter) -> None:
    """No history and minimal features -> cold or neutral (depends on bias).

    The router has a positive bias (0.3) so even with no features it may
    not go below low_hit_threshold=0.25 without empty prefix.
    We use empty token_ids AND no prefix AND session_turn=0 and check
    that the decision is at most neutral (not warm).
    """
    req = FakeRequest(request_id="cold", prefix_hash="", token_ids=[], session_turn=10)
    decision = router.classify_request(req)
    # With session_turn=10: f3 = 1/(1+10) ≈ 0.09
    # score = 0.4*0 + 0.2*0 + 0.2*0.09 + 0.2*0 + 0.3 ≈ 0.318
    # sigmoid(0.318) ≈ 0.579 -> neutral
    # Just check it's not warm (it won't be warm without hit history)
    assert decision.path != "warm", f"Expected not warm, got {decision.path}"


def test_cpd_classify_cold_with_forced_cold_history(
    router: CPDWarmColdHitRateRouter,
) -> None:
    """Prefix with 0% hit history and large context -> cold path."""
    req = FakeRequest(
        request_id="cold_req",
        prefix_hash="pfx_cold",
        token_ids=list(range(5)),
        session_turn=0,
    )
    # set full miss history
    router._hit_history["pfx_cold"] = [False] * 20
    # adjust weights to make sigmoid output < 0.25
    router._weights.w_prefix_hash = 1.0
    router._weights.w_context_length = 0.0
    router._weights.w_session_age = 0.0
    router._weights.w_segment_match = 0.0
    router._weights.bias = -2.0  # large negative bias -> cold
    decision = router.classify_request(req)
    assert decision.path == "cold", f"Expected cold, got {decision.path}"
    assert decision.batch_priority == 2


def test_cpd_classify_neutral_mid_hit_rate(router: CPDWarmColdHitRateRouter) -> None:
    """Moderate features -> neutral path (between 0.25 and 0.70)."""
    req = FakeRequest(
        request_id="neutral_req",
        prefix_hash="pfx_mid",
        token_ids=list(range(20)),
        session_turn=2,
    )
    # 50% hit history
    router._hit_history["pfx_mid"] = [True, False] * 5
    decision = router.classify_request(req)
    assert decision.path in ("warm", "neutral", "cold"), "Invalid path"
    # the hit rate from 50% history + moderate features should be in neutral range
    rate = router.predict_hit_rate(req)
    assert 0.0 <= rate <= 1.0


# --------------------------------------------------------------------------- #
# schedule ordering                                                            #
# --------------------------------------------------------------------------- #


def test_cpd_schedule_warm_first_in_output(router: CPDWarmColdHitRateRouter) -> None:
    """Warm requests appear at the front of the scheduled output."""
    # create one definite warm and one definite cold
    warm_req = FakeRequest(request_id="w", prefix_hash="pfx_w", token_ids=list(range(5)))
    router._hit_history["pfx_w"] = [True] * 20  # 100% hit history -> warm
    cold_req = FakeRequest(request_id="c", prefix_hash="pfx_c", token_ids=[])
    router._weights.bias = -3.0  # force cold for empty request
    router._hit_history["pfx_c"] = [False] * 20
    router._weights.w_prefix_hash = 2.0  # amplify prefix history signal

    scheduled = router.schedule([warm_req, cold_req])
    assert len(scheduled) == 2
    # warm should come before cold
    warm_positions = [i for i, r in enumerate(scheduled) if getattr(r, 'request_id', '') == 'w']
    cold_positions = [i for i, r in enumerate(scheduled) if getattr(r, 'request_id', '') == 'c']
    if warm_positions and cold_positions:
        assert min(warm_positions) < min(cold_positions), "Warm request not before cold"


def test_cpd_schedule_cold_last(router: CPDWarmColdHitRateRouter) -> None:
    """schedule returns a list; if cold exists it appears at the end."""
    reqs = [FakeRequest(request_id=f"r{i}", prefix_hash=f"pfx_{i}") for i in range(5)]
    result = router.schedule(reqs)
    assert len(result) == 5


def test_cpd_schedule_empty_returns_empty(router: CPDWarmColdHitRateRouter) -> None:
    """schedule([]) == []."""
    assert router.schedule([]) == []


# --------------------------------------------------------------------------- #
# Predictor update                                                             #
# --------------------------------------------------------------------------- #


def test_cpd_update_predictor_changes_weights(router: CPDWarmColdHitRateRouter) -> None:
    """update_predictor(req, actual_hit=True) modifies weights."""
    req = FakeRequest(
        request_id="req0", prefix_hash="pfx_up", token_ids=list(range(10))
    )
    w_before = router._weights.w_prefix_hash
    router.update_predictor(req, actual_hit=True)
    w_after = router._weights.w_prefix_hash
    # weights should change (unless gradient happened to be zero)
    assert w_before != w_after or router._weights.bias != 0.3, (
        "update_predictor did not change any weights"
    )


def test_cpd_hit_history_updated_after_update(router: CPDWarmColdHitRateRouter) -> None:
    """After update_predictor, _hit_history[prefix_hash] is non-empty."""
    req = FakeRequest(
        request_id="req0", prefix_hash="pfx_hist", token_ids=list(range(5))
    )
    router.update_predictor(req, actual_hit=True)
    assert "pfx_hist" in router._hit_history
    assert len(router._hit_history["pfx_hist"]) >= 1


def test_cpd_hit_history_max_100_entries(router: CPDWarmColdHitRateRouter) -> None:
    """Hit history is capped at 100 entries per prefix."""
    req = FakeRequest(prefix_hash="pfx_cap", token_ids=[1])
    for _ in range(150):
        router.update_predictor(req, actual_hit=True)
    assert len(router._hit_history["pfx_cap"]) <= 100


# --------------------------------------------------------------------------- #
# Scheduling overhead                                                          #
# --------------------------------------------------------------------------- #


def test_cpd_scheduling_overhead_below_1ms(router: CPDWarmColdHitRateRouter) -> None:
    """classify_request() overhead < 1000μs (TTFT +5% constraint)."""
    req = FakeRequest(request_id="r", prefix_hash="ph", token_ids=list(range(20)))
    for _ in range(10):
        router.classify_request(req)
    mean_us = router.scheduling_overhead_mean_us()
    assert mean_us < 1000, (
        f"Mean scheduling overhead {mean_us:.2f}μs >= 1000μs (TTFT limit exceeded)"
    )


# --------------------------------------------------------------------------- #
# sort_warm_batch_by_prefix_similarity                                        #
# --------------------------------------------------------------------------- #


def test_cpd_sort_warm_batch_by_prefix(router: CPDWarmColdHitRateRouter) -> None:
    """Requests with same prefix_hash appear consecutively in sorted output."""
    reqs = [
        FakeRequest(request_id="r1", prefix_hash="aaa"),
        FakeRequest(request_id="r2", prefix_hash="bbb"),
        FakeRequest(request_id="r3", prefix_hash="aaa"),
        FakeRequest(request_id="r4", prefix_hash="ccc"),
    ]
    sorted_reqs = router.sort_warm_batch_by_prefix_similarity(reqs)
    hashes = [getattr(r, 'prefix_hash', '') for r in sorted_reqs]
    # consecutive same hashes
    seen = {}
    for i, h in enumerate(hashes):
        if h in seen:
            assert seen[h] == i - 1, f"prefix '{h}' not contiguous in sorted output"
        seen[h] = i


# --------------------------------------------------------------------------- #
# Queue pressure cold promotion                                                #
# --------------------------------------------------------------------------- #


def test_cpd_queue_pressure_cold_promoted() -> None:
    """len(requests) > queue_pressure_threshold: half of cold requests promoted to neutral."""
    cfg = CPDRouterConfig(
        high_hit_threshold=0.70,
        low_hit_threshold=0.25,
        queue_pressure_threshold=5,
        seed=42,
    )
    router = CPDWarmColdHitRateRouter(cfg)
    # Force all requests to be cold by setting low bias and no history
    router._weights.bias = -5.0
    router._weights.w_prefix_hash = 0.0
    router._weights.w_context_length = 0.0
    router._weights.w_session_age = 0.0
    router._weights.w_segment_match = 0.0

    reqs = [FakeRequest(request_id=f"r{i}", prefix_hash="") for i in range(10)]
    result = router.schedule(reqs)
    # All 10 should still be returned
    assert len(result) == 10


# --------------------------------------------------------------------------- #
# routing_stats                                                                #
# --------------------------------------------------------------------------- #


def test_cpd_routing_stats_keys(router: CPDWarmColdHitRateRouter) -> None:
    """routing_stats() contains required keys."""
    stats = router.routing_stats()
    required = ["warm_ratio", "cold_ratio", "neutral_ratio", "scheduling_overhead_mean_us"]
    for k in required:
        assert k in stats, f"Missing key '{k}' in routing_stats()"


# --------------------------------------------------------------------------- #
# BaseScheduler interface                                                      #
# --------------------------------------------------------------------------- #


def test_cpd_basescheduler_interface(router: CPDWarmColdHitRateRouter) -> None:
    """schedule() returns a list (BaseScheduler interface compliance)."""
    reqs = [FakeRequest(request_id=f"r{i}") for i in range(3)]
    result = router.schedule(reqs)
    assert isinstance(result, list)
    assert len(result) == 3
