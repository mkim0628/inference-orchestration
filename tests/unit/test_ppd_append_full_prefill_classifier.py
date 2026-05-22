"""Unit tests for PPDAppendFullPrefillClassifier (Activity A).

Covers:
  - First turn always classified as full-prefill
  - Second turn with few new tokens → append
  - Second turn with many new tokens → full
  - SLO pressure forces full even under append conditions
  - Classification overhead < 1000 μs (O(1) guarantee)
  - Registry updated after classify()
  - Session TTL expiry
  - reset_session() removes entry
  - Multiple sessions are independent
  - new_token_ratio calculation edge case
"""

import time
import pytest

from src.scheduler.ppd_append_full_prefill_classifier import (
    PPDAppendFullPrefillClassifier,
    PPDClassifierConfig,
    PrefillTypeDecision,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


def _make_classifier(
    append_threshold: float = 0.15,
    slo_headroom_threshold_ms: float = 30.0,
    session_ttl_seconds: float = 3600.0,
) -> PPDAppendFullPrefillClassifier:
    cfg = PPDClassifierConfig(
        append_threshold=append_threshold,
        slo_headroom_threshold_ms=slo_headroom_threshold_ms,
        session_ttl_seconds=session_ttl_seconds,
        seed=42,
    )
    return PPDAppendFullPrefillClassifier(cfg)


def _tokens(n: int, start: int = 0) -> list:
    return list(range(start, start + n))


# ------------------------------------------------------------------ #
# First turn always full                                               #
# ------------------------------------------------------------------ #


def test_ppd_classifier_turn1_always_full_prefill() -> None:
    """First request in any session is always classified as full-prefill."""
    clf = _make_classifier()
    decision = clf.classify("req1", "session1", _tokens(100))
    assert decision.prefill_type == "full", (
        f"First turn should be full-prefill, got {decision.prefill_type}"
    )
    assert decision.routed_to == "P_node", (
        f"Full-prefill should route to P_node, got {decision.routed_to}"
    )


# ------------------------------------------------------------------ #
# Append conditions                                                    #
# ------------------------------------------------------------------ #


def test_ppd_classifier_turn2_small_new_tokens_append() -> None:
    """Turn 2 with new_token_ratio < append_threshold → append-prefill → D_node."""
    clf = _make_classifier(append_threshold=0.15)
    # Turn 1: establish baseline of 100 tokens
    clf.classify("req1", "sess1", _tokens(100))
    # Turn 2: 115 tokens total → 15 new → ratio = 15/115 ≈ 0.130 < 0.15
    decision = clf.classify("req2", "sess1", _tokens(115))
    assert decision.prefill_type == "append", (
        f"Expected append-prefill, got {decision.prefill_type} (ratio={decision.new_token_ratio:.4f})"
    )
    assert decision.routed_to == "D_node", (
        f"Append-prefill should route to D_node, got {decision.routed_to}"
    )
    assert decision.new_token_ratio < 0.15, (
        f"Expected new_token_ratio < 0.15, got {decision.new_token_ratio:.4f}"
    )


def test_ppd_classifier_turn2_large_new_tokens_full() -> None:
    """Turn 2 with new_token_ratio > append_threshold → full-prefill."""
    clf = _make_classifier(append_threshold=0.15)
    # Turn 1: 50 tokens
    clf.classify("req1", "sess1", _tokens(50))
    # Turn 2: 200 tokens → 150 new → ratio = 150/200 = 0.75 > 0.15
    decision = clf.classify("req2", "sess1", _tokens(200))
    assert decision.prefill_type == "full", (
        f"Expected full-prefill for large token increase, got {decision.prefill_type}"
    )


# ------------------------------------------------------------------ #
# SLO pressure                                                         #
# ------------------------------------------------------------------ #


def test_ppd_classifier_slo_pressure_forces_full() -> None:
    """Append conditions + remaining_slo_ms < threshold → full-prefill (P_node offload)."""
    clf = _make_classifier(append_threshold=0.15, slo_headroom_threshold_ms=30.0)
    # Turn 1
    clf.classify("req1", "sess1", _tokens(100))
    # Turn 2: would be append, but SLO pressure forces full
    decision = clf.classify(
        "req2", "sess1", _tokens(115),
        remaining_slo_ms=10.0  # < 30ms threshold
    )
    assert decision.prefill_type == "full", (
        f"SLO pressure should force full-prefill, got {decision.prefill_type}"
    )
    assert decision.routed_to == "P_node"


# ------------------------------------------------------------------ #
# Performance: overhead < 1ms                                         #
# ------------------------------------------------------------------ #


def test_ppd_classifier_overhead_below_1ms() -> None:
    """classify() overhead should be < 1000 μs (O(1) hash comparison)."""
    clf = _make_classifier()
    clf.classify("warmup", "sess0", _tokens(100))  # warmup
    overheads = []
    for i in range(20):
        tokens = _tokens(100 + i * 5)
        decision = clf.classify(f"req{i}", "sess_perf", tokens)
        overheads.append(decision.classifier_overhead_us)
    mean_overhead = sum(overheads) / len(overheads)
    assert mean_overhead < 1000.0, (
        f"Mean classification overhead {mean_overhead:.1f} μs >= 1000 μs"
    )


# ------------------------------------------------------------------ #
# Registry updates                                                     #
# ------------------------------------------------------------------ #


def test_ppd_classifier_registry_updated_after_classify() -> None:
    """_registry[session_id] exists and turn_count increments after each classify()."""
    clf = _make_classifier()
    assert "sess1" not in clf._registry

    clf.classify("req1", "sess1", _tokens(100))
    assert "sess1" in clf._registry, "_registry should contain session after classify()"
    assert clf._registry["sess1"].turn_count == 1

    clf.classify("req2", "sess1", _tokens(115))
    assert clf._registry["sess1"].turn_count == 2


# ------------------------------------------------------------------ #
# Session TTL expiry                                                   #
# ------------------------------------------------------------------ #


def test_ppd_classifier_session_ttl_expire() -> None:
    """Sessions with TTL < elapsed time are removed by expire_sessions()."""
    clf = _make_classifier(session_ttl_seconds=0.001)  # 1 ms TTL
    clf.classify("req1", "sess_expiring", _tokens(100))
    assert clf.registry_size() == 1

    time.sleep(0.002)  # wait for TTL to expire
    expired = clf.expire_sessions()
    assert expired >= 1, f"Expected at least 1 expired session, got {expired}"
    assert clf.registry_size() == 0


# ------------------------------------------------------------------ #
# reset_session                                                        #
# ------------------------------------------------------------------ #


def test_ppd_classifier_reset_session_removes_entry() -> None:
    """reset_session() removes the session from _registry."""
    clf = _make_classifier()
    clf.classify("req1", "sess1", _tokens(100))
    assert "sess1" in clf._registry

    clf.reset_session("sess1")
    assert "sess1" not in clf._registry, "Session should be removed after reset_session()"


def test_ppd_classifier_reset_nonexistent_session_no_error() -> None:
    """reset_session() on a non-existent session does not raise."""
    clf = _make_classifier()
    clf.reset_session("never_existed")  # should not raise


# ------------------------------------------------------------------ #
# Multiple sessions independent                                        #
# ------------------------------------------------------------------ #


def test_ppd_classifier_multiple_sessions_independent() -> None:
    """Session A and Session B have independent turn counts."""
    clf = _make_classifier()
    clf.classify("r1", "sessA", _tokens(100))
    clf.classify("r2", "sessB", _tokens(200))
    clf.classify("r3", "sessA", _tokens(120))

    assert clf._registry["sessA"].turn_count == 2
    assert clf._registry["sessB"].turn_count == 1


# ------------------------------------------------------------------ #
# new_token_ratio calculation                                          #
# ------------------------------------------------------------------ #


def test_ppd_classifier_new_token_ratio_calculation() -> None:
    """prev_total=100, current_total=115 → new_token_ratio = 15/115 ≈ 0.130 → append."""
    clf = _make_classifier(append_threshold=0.15)
    clf.classify("r1", "s1", _tokens(100))
    decision = clf.classify("r2", "s1", _tokens(115))

    expected_ratio = 15 / 115
    assert abs(decision.new_token_ratio - expected_ratio) < 1e-6, (
        f"Expected ratio {expected_ratio:.6f}, got {decision.new_token_ratio:.6f}"
    )
    assert decision.prefill_type == "append", (
        f"Ratio {decision.new_token_ratio:.4f} < 0.15 should be append"
    )


def test_ppd_classifier_exact_threshold_is_append() -> None:
    """new_token_ratio exactly at append_threshold (<=) → append-prefill."""
    clf = _make_classifier(append_threshold=0.15)
    # prev=100, current needs ratio = exactly 0.15: new = 0.15 * total → total=100/(1-0.15)≈117.6
    # Use total=200 and new=30: ratio=30/200=0.15 == threshold → append
    clf.classify("r1", "s1", _tokens(170))
    decision = clf.classify("r2", "s1", _tokens(200))
    # new_tokens = 30, ratio = 30/200 = 0.15 == threshold → append
    assert decision.prefill_type == "append", (
        f"Ratio exactly at threshold should be append, got {decision.prefill_type}"
    )
