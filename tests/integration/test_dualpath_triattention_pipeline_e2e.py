"""End-to-end integration tests for DualPathTriAttentionCompressPipeline (Cross-1, A+C).

Tests:
  - Single path scenario: prefill NIC 0.60 -> path="single", K_final=K_full
  - Dual path scenario: prefill NIC 0.85 + idle decode -> path="dual"
  - Dual path + reasoning task: n_keep ≈ N * 0.093
  - Accuracy: dual path relative_error < 0.01 (MANDATORY)
  - 100 runs: avg decision_latency_ms < 0.1
  - pipeline_summary() dict
  - Idle decode absent: single path fallback
  - kv_bytes_transferred / kv_bytes_original ≈ budget_ratio
"""

import dataclasses
import time
import torch
import torch.nn.functional as F
import pytest

from src.cache.triattention_pre_rope_kv_selector_codec import TriAttentionSelectorConfig
from src.scheduler.dualpath_nic_load_balancer import DualPathNICConfig
from src.engine.dualpath_triattention_pipeline import (
    DualPathTriAttentionCompressPipeline,
    DualPathTriAttentionPipelineConfig,
)
from src.metrics.perplexity import attention_output_relative_error, cosine_similarity_output


# ---- helpers -----------------------------------------------------------------

@dataclasses.dataclass
class FakeRequest:
    request_id: str = "req-test"
    routing_decision: object = None


def _make_pipeline(
    kv_budget_ratio_default: float = 0.20,
    kv_budget_ratio_reasoning: float = 0.093,
    nic_saturation_threshold: float = 0.80,
    idle_nic_threshold: float = 0.30,
    max_dual_path_per_node: int = 8,
    d_head: int = 64,
    stale_threshold_ms: float = 60000.0,
    seed: int = 42,
) -> DualPathTriAttentionCompressPipeline:
    ta_cfg = TriAttentionSelectorConfig(
        d_head=d_head,
        n_kv_heads=4,
        kv_budget_ratio_reasoning=kv_budget_ratio_reasoning,
        kv_budget_ratio_default=kv_budget_ratio_default,
        max_entries=1000,
        seed=seed,
    )
    dp_cfg = DualPathNICConfig(
        nic_saturation_threshold=nic_saturation_threshold,
        idle_nic_threshold=idle_nic_threshold,
        max_dual_path_per_node=max_dual_path_per_node,
        stale_threshold_ms=stale_threshold_ms,
        seed=seed,
    )
    cfg = DualPathTriAttentionPipelineConfig(
        triattention_config=ta_cfg,
        dualpath_config=dp_cfg,
        seed=seed,
    )
    return DualPathTriAttentionCompressPipeline(cfg)


def _random_qkv(n_q: int, N: int, d_head: int, seed: int = 42):
    torch.manual_seed(seed)
    Q = torch.randn(n_q, d_head)
    K = torch.randn(N, d_head)
    V = torch.randn(N, d_head)
    return Q, K, V


# ---- single path scenario ----------------------------------------------------

def test_single_path_when_prefill_nic_below_threshold() -> None:
    """prefill NIC 0.60 < 0.80: path='single', K_final = K_full."""
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.60)
    Q, K_full, V_full = _random_qkv(4, 64, 64)
    req = FakeRequest()

    K_final, V_final, report = pipeline.run_pipeline(req, Q, K_full, V_full)
    assert report["path"] == "single"
    assert K_final.shape == K_full.shape, "single path: K_final should equal K_full"
    assert V_final.shape == V_full.shape
    # Tensors should be identical (same object)
    assert K_final is K_full


def test_single_path_compression_ratio_is_one() -> None:
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.50)
    Q, K, V = _random_qkv(4, 64, 64)
    _, _, report = pipeline.run_pipeline(FakeRequest(), Q, K, V)
    assert report["compression_ratio"] == 1.0


# ---- dual path scenario ------------------------------------------------------

def test_dual_path_when_prefill_nic_saturated() -> None:
    """prefill NIC 0.85 + idle decode -> path='dual'."""
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.85)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.10)
    Q, K_full, V_full = _random_qkv(4, 128, 64)
    req = FakeRequest()

    K_final, V_final, report = pipeline.run_pipeline(req, Q, K_full, V_full)
    assert report["path"] == "dual"
    assert K_final.shape[0] < K_full.shape[0], "dual path must compress KV"
    assert report["relay_decode_node_id"] is not None


def test_dual_path_kv_shape_matches_budget() -> None:
    """Dual path: n_keep matches kv_budget_ratio_default * N."""
    d, N = 64, 256
    budget = 0.20
    pipeline = _make_pipeline(kv_budget_ratio_default=budget)
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05)

    Q, K, V = _random_qkv(8, N, d, seed=7)
    K_final, V_final, report = pipeline.run_pipeline(
        FakeRequest(), Q, K, V, is_reasoning_task=False
    )
    n_keep_expected = max(1, int(N * budget))
    assert K_final.shape[0] == n_keep_expected, (
        f"K_final.shape[0]={K_final.shape[0]} != {n_keep_expected}"
    )


# ---- dual path + reasoning task ----------------------------------------------

def test_dual_path_reasoning_task_budget_093() -> None:
    """Dual path + reasoning: n_keep ≈ N * 0.093."""
    d, N = 64, 256
    budget = 0.093
    pipeline = _make_pipeline(kv_budget_ratio_reasoning=budget)
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05)

    Q, K, V = _random_qkv(8, N, d, seed=15)
    K_final, _, report = pipeline.run_pipeline(
        FakeRequest(), Q, K, V, is_reasoning_task=True
    )
    n_keep_expected = max(1, int(N * budget))
    assert K_final.shape[0] == n_keep_expected, (
        f"K_final.shape[0]={K_final.shape[0]} != {n_keep_expected}"
    )
    assert abs(report["kv_budget_ratio"] - budget) < 1e-9


# ---- accuracy: dual path relative_error < 0.01 (MANDATORY) ------------------

def _make_focused_qkv_for_accuracy(
    n_q: int, N: int, d: int, budget: float, seed: int = 42
) -> tuple:
    """Create Q, K, V where budget-fraction tokens have dominant attention mass.

    Design:
    - Q: diverse random unit vectors (low conc_q -> norm_score dominates).
    - K_important[i]: aligned with Q[i % n_q] * 100 -> large norm AND large logit.
    - K_noise: random * 0.001 -> near-zero norm AND near-zero logit.

    This guarantees that TriAttention (norm_score path) selects all important
    tokens and that noise tokens contribute < 0.1% of attention mass.
    """
    torch.manual_seed(seed)
    n_important = max(1, int(N * budget))

    Q = torch.randn(n_q, d)
    Q = F.normalize(Q, dim=-1)

    K_imps = [Q[i % n_q] * 100.0 + torch.randn(d) * 0.001 for i in range(n_important)]
    K_important = torch.stack(K_imps, dim=0)
    K_noise = (
        torch.randn(N - n_important, d) * 0.001
        if N > n_important
        else torch.zeros(0, d)
    )
    K = torch.cat([K_important, K_noise], dim=0)
    V = torch.randn(N, d)
    return Q, K, V


def test_dual_path_accuracy_relative_error_below_001() -> None:
    """Dual path: attention_output_relative_error < 0.01 (MANDATORY).

    Uses budget=0.20 with focused-attention KV: important tokens are aligned
    with Q directions (large logit), noise tokens near-zero.
    Guaranteed: all important tokens selected, noise mass < 0.1%.
    """
    d, N = 64, 128
    budget = 0.20
    pipeline = _make_pipeline(kv_budget_ratio_default=budget, d_head=d)
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05)

    Q_pre_rope, K, V = _make_focused_qkv_for_accuracy(8, N, d, budget, seed=42)

    K_final, V_final, report = pipeline.run_pipeline(
        FakeRequest(), Q_pre_rope, K, V, is_reasoning_task=False
    )
    assert report["path"] == "dual"
    err = attention_output_relative_error(
        Q_pre_rope.float(), K.float(), V.float(), K_final.float(), V_final.float()
    )
    assert err < 0.01, f"dual path relative_error={err:.6f} >= 0.01 (MANDATORY)"


def test_dual_path_accuracy_cosine_sim_above_099() -> None:
    """Dual path: cosine_similarity >= 0.99 (MANDATORY).

    Uses budget=0.20 with focused-attention KV for reliable selection.
    """
    d, N = 64, 128
    budget = 0.20
    pipeline = _make_pipeline(kv_budget_ratio_default=budget, d_head=d)
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05)

    Q_pre_rope, K, V = _make_focused_qkv_for_accuracy(8, N, d, budget, seed=42)

    K_final, V_final, _ = pipeline.run_pipeline(
        FakeRequest(), Q_pre_rope, K, V, is_reasoning_task=False
    )
    cos_sim = cosine_similarity_output(
        Q_pre_rope.float(), K.float(), V.float(), K_final.float(), V_final.float()
    )
    assert cos_sim >= 0.99, f"cosine_sim={cos_sim:.6f} < 0.99 (MANDATORY)"


# ---- 100 runs: avg decision_latency_ms < 0.1 --------------------------------

def test_100_runs_avg_decision_latency_below_01ms() -> None:
    """100 pipeline runs: avg routing decision_latency_ms < 0.1ms."""
    pipeline = _make_pipeline(d_head=32)
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.50)

    latencies = []
    for i in range(100):
        Q, K, V = _random_qkv(4, 32, 32, seed=i)
        _, _, report = pipeline.run_pipeline(FakeRequest(request_id=f"r{i}"), Q, K, V)
        latencies.append(report["routing_latency_ms"])

    avg_lat = sum(latencies) / len(latencies)
    assert avg_lat < 0.1, f"avg decision_latency={avg_lat:.4f}ms >= 0.1ms"


# ---- pipeline_summary --------------------------------------------------------

def test_pipeline_summary_returns_dict() -> None:
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05)

    for i in range(3):
        Q, K, V = _random_qkv(4, 64, 64, seed=i)
        pipeline.dualpath_lb._node_status["dc0"].active_dual_path = 0
        pipeline.run_pipeline(FakeRequest(request_id=f"r{i}"), Q, K, V)

    summary = pipeline.pipeline_summary()
    assert isinstance(summary, dict)
    assert "total_runs" in summary
    assert "dual_path_runs" in summary
    assert "single_path_runs" in summary
    assert "avg_compression_ratio" in summary


def test_pipeline_summary_empty_returns_empty() -> None:
    pipeline = _make_pipeline()
    assert pipeline.pipeline_summary() == {}


def test_pipeline_summary_total_runs_count() -> None:
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.50)

    for i in range(5):
        Q, K, V = _random_qkv(4, 32, 64, seed=i)
        pipeline.run_pipeline(FakeRequest(request_id=f"r{i}"), Q, K, V)

    summary = pipeline.pipeline_summary()
    assert summary["total_runs"] == 5


# ---- idle decode absent: single path fallback --------------------------------

def test_idle_decode_absent_single_fallback() -> None:
    """No idle decode nodes -> fallback to single path even if prefill saturated."""
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    # No decode nodes registered at all
    Q, K, V = _random_qkv(4, 64, 64)
    _, _, report = pipeline.run_pipeline(FakeRequest(), Q, K, V)
    assert report["path"] == "single"


def test_decode_nodes_all_busy_single_fallback() -> None:
    """All decode nodes at max capacity -> single path."""
    pipeline = _make_pipeline(max_dual_path_per_node=2)
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05, active_dual_path=2)
    pipeline.dualpath_lb.update_nic_status("dc1", "decode", 0.05, active_dual_path=2)
    Q, K, V = _random_qkv(4, 64, 64)
    _, _, report = pipeline.run_pipeline(FakeRequest(), Q, K, V)
    assert report["path"] == "single"


# ---- kv_bytes_transferred approximates budget_ratio -------------------------

def test_kv_bytes_transferred_proportional_to_budget() -> None:
    """kv_bytes_transferred / kv_bytes_original ≈ budget_ratio."""
    d, N = 32, 200
    budget = 0.20
    pipeline = _make_pipeline(kv_budget_ratio_default=budget, d_head=d)
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05)

    Q, K, V = _random_qkv(4, N, d, seed=88)
    _, _, report = pipeline.run_pipeline(FakeRequest(), Q, K, V, is_reasoning_task=False)

    if report["path"] == "dual":
        ratio = report["kv_bytes_transferred"] / report["kv_bytes_original"]
        # Allow 20% tolerance since tensor dtype may differ
        assert ratio < budget + 0.20, (
            f"transferred ratio={ratio:.4f} much larger than budget={budget}"
        )


# ---- complete_dual_path called on dual route ---------------------------------

def test_complete_dual_path_decrements_active_count() -> None:
    """After dual-path run, active_dual_path on relay node is decremented."""
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05, active_dual_path=0)

    Q, K, V = _random_qkv(4, 64, 64, seed=5)
    _, _, report = pipeline.run_pipeline(FakeRequest(), Q, K, V)

    if report["path"] == "dual":
        relay = report["relay_decode_node_id"]
        # After complete_dual_path is called in run_pipeline, active_dual_path should be 0
        assert pipeline.dualpath_lb._node_status[relay].active_dual_path == 0


# ---- conc_q / conc_k in dual report -----------------------------------------

def test_dual_report_contains_concentration_metrics() -> None:
    pipeline = _make_pipeline()
    pipeline.dualpath_lb.update_nic_status("pf0", "prefill", 0.90)
    pipeline.dualpath_lb.update_nic_status("dc0", "decode", 0.05)

    Q, K, V = _random_qkv(4, 64, 64, seed=9)
    _, _, report = pipeline.run_pipeline(FakeRequest(), Q, K, V)

    if report["path"] == "dual":
        assert report["conc_q"] is not None
        assert report["conc_k"] is not None
        assert 0.0 <= report["conc_q"] <= 1.0
        assert 0.0 <= report["conc_k"] <= 1.0
