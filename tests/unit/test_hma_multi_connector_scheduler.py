"""Unit tests for HMAMultiConnectorCompressionPluginScheduler (Activity A).

Tests connector registration, selection O(1) dispatch, pipeline mode chaining,
scheduling overhead measurement, and request fairness.
"""

import time
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.engine.runner import InferenceRequest
from src.scheduler.hma_multi_connector_scheduler import (
    HMAConnectorAdapter,
    HMAConnectorInterface,
    HMAMultiConnectorCompressionPluginScheduler,
    HMAMultiConnectorConfig,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_scheduler(
    default_connector: str = "global_retention",
    pipeline_mode: bool = False,
    long_ctx_threshold: int = 4096,
    memory_pressure_threshold: float = 0.8,
) -> HMAMultiConnectorCompressionPluginScheduler:
    cfg = HMAMultiConnectorConfig(
        default_connector=default_connector,
        pipeline_mode=pipeline_mode,
        long_ctx_threshold=long_ctx_threshold,
        memory_pressure_threshold=memory_pressure_threshold,
        seed=42,
    )
    return HMAMultiConnectorCompressionPluginScheduler(cfg)


def make_request(request_id: str = "req1", n_tokens: int = 64) -> InferenceRequest:
    return InferenceRequest(request_id=request_id, token_ids=list(range(n_tokens)))


class MockConnector(HMAConnectorInterface):
    """Minimal mock connector that tracks compress() calls."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.compress_calls = 0

    @property
    def connector_name(self) -> str:
        return self._name

    def compress(self, kv: torch.Tensor, request_profile: Dict) -> torch.Tensor:
        self.compress_calls += 1
        return kv

    def decompress(self, compressed_kv: torch.Tensor, request_profile: Dict) -> torch.Tensor:
        return compressed_kv


# ------------------------------------------------------------------ #
# Registration                                                         #
# ------------------------------------------------------------------ #

def test_register_connector() -> None:
    """register_connector() should add connector to registry."""
    sched = make_scheduler()
    conn = MockConnector("rl_adaptive")
    sched.register_connector("rl_adaptive", conn)
    assert "rl_adaptive" in sched._connector_registry
    assert sched._connector_registry["rl_adaptive"] is conn
    assert sched._connector_selection_counts["rl_adaptive"] == 0


# ------------------------------------------------------------------ #
# Connector selection                                                  #
# ------------------------------------------------------------------ #

def test_select_connector_rl_mode() -> None:
    """is_rl_mode=True → 'rl_adaptive' should be selected."""
    sched = make_scheduler()
    sched.register_connector("rl_adaptive", MockConnector("rl_adaptive"))
    sched.register_connector("global_retention", MockConnector("global_retention"))
    req = make_request(n_tokens=64)
    selected = sched.select_connector(req, {"is_rl_mode": True})
    assert selected == "rl_adaptive", f"Expected 'rl_adaptive', got '{selected}'"


def test_select_connector_num_completions() -> None:
    """num_completions>1 → 'rl_adaptive' should be selected."""
    sched = make_scheduler()
    sched.register_connector("rl_adaptive", MockConnector("rl_adaptive"))
    sched.register_connector("global_retention", MockConnector("global_retention"))
    req = make_request(n_tokens=64)
    selected = sched.select_connector(req, {"num_completions": 4})
    assert selected == "rl_adaptive", f"Expected 'rl_adaptive', got '{selected}'"


def test_select_connector_long_context() -> None:
    """context_length > 4096, is_rl_mode=False → 'global_retention' should be selected."""
    sched = make_scheduler(long_ctx_threshold=4096)
    sched.register_connector("rl_adaptive", MockConnector("rl_adaptive"))
    sched.register_connector("global_retention", MockConnector("global_retention"))
    req = make_request(n_tokens=5000)
    selected = sched.select_connector(req, {"is_rl_mode": False})
    assert selected == "global_retention", f"Expected 'global_retention', got '{selected}'"


def test_select_connector_high_pressure() -> None:
    """context_length ≤ 4096, memory_pressure=0.9 → 'ratequant' selected (if registered)."""
    sched = make_scheduler(memory_pressure_threshold=0.8)
    sched.register_connector("global_retention", MockConnector("global_retention"))
    sched.register_connector("ratequant", MockConnector("ratequant"))
    req = make_request(n_tokens=128)
    selected = sched.select_connector(req, {"memory_pressure": 0.9})
    assert selected == "ratequant", f"Expected 'ratequant', got '{selected}'"


def test_select_connector_default_fallback() -> None:
    """If selected connector not in registry, fall back to first available."""
    sched = make_scheduler(default_connector="missing_connector")
    sched.register_connector("fallback_conn", MockConnector("fallback_conn"))
    req = make_request(n_tokens=64)
    selected = sched.select_connector(req, {})
    assert selected == "fallback_conn", f"Expected 'fallback_conn', got '{selected}'"


# ------------------------------------------------------------------ #
# apply_connector                                                      #
# ------------------------------------------------------------------ #

def test_apply_connector_calls_compress() -> None:
    """apply_connector() should call the selected connector's compress()."""
    sched = make_scheduler(default_connector="test_conn")
    mock_conn = MockConnector("test_conn")
    sched.register_connector("test_conn", mock_conn)
    req = make_request()
    kv = torch.randn(32, 64)
    _, name = sched.apply_connector(req, kv)
    assert mock_conn.compress_calls == 1, (
        f"compress() should have been called once, got {mock_conn.compress_calls}"
    )
    assert name == "test_conn"


def test_pipeline_mode_chains_global_retention() -> None:
    """pipeline_mode=True and primary != global_retention → global_retention.compress() also called."""
    sched = make_scheduler(pipeline_mode=True)
    rl_conn = MockConnector("rl_adaptive")
    gr_conn = MockConnector("global_retention")
    sched.register_connector("rl_adaptive", rl_conn)
    sched.register_connector("global_retention", gr_conn)

    req = make_request(n_tokens=64)
    kv = torch.randn(32, 64)
    sched.apply_connector(req, kv, {"is_rl_mode": True})
    assert rl_conn.compress_calls == 1, "Primary connector compress() should be called"
    assert gr_conn.compress_calls == 1, (
        "global_retention compress() should also be called in pipeline_mode"
    )


# ------------------------------------------------------------------ #
# Scheduling overhead                                                  #
# ------------------------------------------------------------------ #

def test_scheduling_overhead_below_01ms() -> None:
    """select_connector() overhead should be < 0.1ms (O(1) dict lookup)."""
    sched = make_scheduler()
    sched.register_connector("global_retention", MockConnector("global_retention"))
    req = make_request()

    # Warm up
    sched.select_connector(req, {})
    sched.reset_stats()

    # Measure 100 selections
    N = 100
    for _ in range(N):
        sched.select_connector(req, {})

    p50 = sched.scheduling_overhead_ms_p50()
    assert p50 < 0.1, f"select_connector() p50 overhead {p50:.4f}ms >= 0.1ms"


# ------------------------------------------------------------------ #
# Statistics                                                           #
# ------------------------------------------------------------------ #

def test_connector_selection_stats() -> None:
    """connector_selection_stats() should return accurate per-connector counts."""
    sched = make_scheduler(default_connector="global_retention")
    rl_conn = MockConnector("rl_adaptive")
    gr_conn = MockConnector("global_retention")
    sched.register_connector("rl_adaptive", rl_conn)
    sched.register_connector("global_retention", gr_conn)

    req_rl = make_request("req_rl", n_tokens=64)
    req_long = make_request("req_long", n_tokens=5000)
    req_default = make_request("req_default", n_tokens=128)

    sched.apply_connector(req_rl, torch.randn(16, 64), {"is_rl_mode": True})
    sched.apply_connector(req_long, torch.randn(16, 64), {})
    sched.apply_connector(req_default, torch.randn(16, 64), {})

    stats = sched.connector_selection_stats()
    assert stats.get("rl_adaptive", 0) == 1, f"rl_adaptive count: {stats}"
    assert stats.get("global_retention", 0) == 2, f"global_retention count: {stats}"


def test_schedule_returns_all_requests() -> None:
    """schedule(requests) should return a list of the same length as input."""
    sched = make_scheduler()
    requests = [make_request(f"req{i}", 64) for i in range(10)]
    result = sched.schedule(requests)
    assert len(result) == len(requests), (
        f"Expected {len(requests)} requests, got {len(result)}"
    )


def test_reset_stats_clears_all() -> None:
    """reset_stats() should clear all counters to zero/empty."""
    sched = make_scheduler(default_connector="global_retention")
    sched.register_connector("global_retention", MockConnector("global_retention"))
    req = make_request()
    sched.apply_connector(req, torch.randn(16, 64))
    sched.schedule([req])

    sched.reset_stats()

    assert len(sched._scheduling_times) == 0, "scheduling_times should be empty"
    assert len(sched._request_connector_map) == 0, "request_connector_map should be empty"
    assert len(sched._arrival_times) == 0, "arrival_times should be empty"
    # connector_selection_counts should be cleared
    for name, count in sched._connector_selection_counts.items():
        assert count == 0, f"connector {name} count should be 0 after reset"
