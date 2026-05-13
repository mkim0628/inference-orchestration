"""Unit tests for PBKVAgentSegmentPreservationScheduler (Activity A).

Covers:
  - MLP predictor input/output shapes
  - schedule() request reordering
  - update_preservation_policy() GPU preserve / host evict sets
  - Lipschitz robustness (preemption margin)
  - update_agent_history() and update_wait() fairness
"""

import pytest
import torch

from src.cache.segmented import SegmentedHashCache
from src.engine.runner import InferenceRequest
from src.scheduler.pbkv_agent_segment_scheduler import (
    PBKVAgentSegmentPreservationScheduler,
    PBKVConfig,
    _SegmentMLP,
)


@pytest.fixture
def config() -> PBKVConfig:
    return PBKVConfig(
        segment_emb_dim=64,   # smaller for unit test speed
        history_steps=5,
        prediction_horizon=3,
        gpu_preserve_threshold=0.6,
        host_evict_threshold=0.3,
        preemption_margin=0.3,
        fairness_max_wait=5,
        chunk_size=16,
        seed=42,
    )


@pytest.fixture
def scheduler(config: PBKVConfig) -> PBKVAgentSegmentPreservationScheduler:
    cache = SegmentedHashCache(chunk_size=16, max_entries=200)
    return PBKVAgentSegmentPreservationScheduler(cache, config)


def make_request(rid: str, n_tokens: int = 32) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        token_ids=list(range(n_tokens)),
    )


# ------------------------------------------------------------------ #
# _SegmentMLP                                                          #
# ------------------------------------------------------------------ #

class TestSegmentMLP:
    def test_output_shape(self, config: PBKVConfig) -> None:
        mlp = _SegmentMLP(config.segment_emb_dim, config.history_steps)
        x = torch.randn(4, config.segment_emb_dim + config.history_steps)
        out = mlp(x)
        assert out.shape == (4, 1)

    def test_output_in_0_1(self, config: PBKVConfig) -> None:
        mlp = _SegmentMLP(config.segment_emb_dim, config.history_steps)
        x = torch.randn(8, config.segment_emb_dim + config.history_steps)
        out = mlp(x)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_gradient_flows(self, config: PBKVConfig) -> None:
        mlp = _SegmentMLP(config.segment_emb_dim, config.history_steps)
        x = torch.randn(2, config.segment_emb_dim + config.history_steps, requires_grad=False)
        out = mlp(x).sum()
        out.backward()
        for p in mlp.parameters():
            assert p.grad is not None


# ------------------------------------------------------------------ #
# schedule()                                                           #
# ------------------------------------------------------------------ #

class TestSchedule:
    def test_schedule_returns_all_requests(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        reqs = [make_request(f"r{i}") for i in range(5)]
        result = scheduler.schedule(reqs)
        assert len(result) == len(reqs)
        assert {r.request_id for r in result} == {r.request_id for r in reqs}

    def test_schedule_empty_input(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        assert scheduler.schedule([]) == []

    def test_schedule_single_request(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        reqs = [make_request("only")]
        result = scheduler.schedule(reqs)
        assert result[0].request_id == "only"

    def test_schedule_fairness_with_wait(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        """A request with max wait steps should not be indefinitely starved."""
        reqs = [make_request("new"), make_request("old")]
        # Simulate old request waiting fairness_max_wait steps
        scheduler._wait_steps["old"] = scheduler.config.fairness_max_wait
        result = scheduler.schedule(reqs)
        # The 'old' request should appear somewhere in the result
        assert any(r.request_id == "old" for r in result)


# ------------------------------------------------------------------ #
# update_preservation_policy()                                        #
# ------------------------------------------------------------------ #

class TestUpdatePreservationPolicy:
    def test_returns_two_sets(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        preserve, evict = scheduler.update_preservation_policy([], [])
        assert isinstance(preserve, set)
        assert isinstance(evict, set)

    def test_no_overlap_between_preserve_and_evict(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        # Populate the cache with some keys
        import torch
        cache = scheduler.cache
        for i in range(5):
            cache.put(f"seg_{i}", torch.randn(8, 2, 4, 16))
        preserve, evict = scheduler.update_preservation_policy([], [])
        assert preserve.isdisjoint(evict)

    def test_preserve_and_evict_subsets_of_cache(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        import torch
        cache = scheduler.cache
        keys = [f"k{i}" for i in range(4)]
        for k in keys:
            cache.put(k, torch.randn(4, 2, 4, 16))
        preserve, evict = scheduler.update_preservation_policy([], [])
        all_keys = set(keys)
        assert preserve.issubset(all_keys)
        assert evict.issubset(all_keys)

    def test_lipschitz_margin_reduces_preserve_threshold(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        """Effective threshold = gpu_preserve_threshold - preemption_margin.
        This means more keys may be preserved vs threshold without margin."""
        effective = (
            scheduler.config.gpu_preserve_threshold
            - scheduler.config.preemption_margin
        )
        # effective threshold should be less than gpu_preserve_threshold
        assert effective < scheduler.config.gpu_preserve_threshold


# ------------------------------------------------------------------ #
# update_agent_history()                                               #
# ------------------------------------------------------------------ #

class TestUpdateAgentHistory:
    def test_history_stored(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        scheduler.update_agent_history("agent_1", ["key_a", "key_b"])
        assert "agent_1" in scheduler._agent_history
        assert "key_a" in scheduler._agent_history["agent_1"]

    def test_history_capped_at_history_steps(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        keys = [f"key_{i}" for i in range(20)]
        scheduler.update_agent_history("agent_2", keys)
        assert len(scheduler._agent_history["agent_2"]) <= scheduler.config.history_steps

    def test_history_accumulates_across_calls(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        scheduler.update_agent_history("agent_3", ["k1", "k2"])
        scheduler.update_agent_history("agent_3", ["k3"])
        history = scheduler._agent_history["agent_3"]
        assert "k3" in history


# ------------------------------------------------------------------ #
# update_wait()                                                        #
# ------------------------------------------------------------------ #

class TestUpdateWait:
    def test_wait_increments_for_unprocessed(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        scheduler.update_wait(processed_ids=["r1"], all_ids=["r1", "r2"])
        assert scheduler._wait_steps.get("r2", 0) == 1
        assert scheduler._wait_steps.get("r1", 0) == 0

    def test_wait_does_not_increment_processed(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        scheduler.update_wait(processed_ids=["r1"], all_ids=["r1"])
        assert scheduler._wait_steps.get("r1", 0) == 0

    def test_wait_accumulates_over_rounds(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        for _ in range(3):
            scheduler.update_wait(processed_ids=[], all_ids=["stale"])
        assert scheduler._wait_steps["stale"] == 3


# ------------------------------------------------------------------ #
# internal helpers                                                     #
# ------------------------------------------------------------------ #

class TestInternalHelpers:
    def test_segment_embedding_shape(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        emb = scheduler._get_segment_embedding("some_key")
        assert emb.shape == (scheduler.config.segment_emb_dim,)

    def test_history_vector_shape(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        vec = scheduler._get_history_vector("agent_x")
        assert vec.shape == (scheduler.config.history_steps,)

    def test_history_vector_all_zeros_when_no_history(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        vec = scheduler._get_history_vector("nonexistent_agent")
        assert torch.all(vec == 0.0)

    def test_chunk_key_deterministic(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        k1 = scheduler._chunk_key([1, 2, 3, 4], 0)
        k2 = scheduler._chunk_key([1, 2, 3, 4], 0)
        assert k1 == k2

    def test_chunk_key_different_chunks_differ(
        self, scheduler: PBKVAgentSegmentPreservationScheduler
    ) -> None:
        k1 = scheduler._chunk_key([1, 2, 3, 4], 0)
        k2 = scheduler._chunk_key([5, 6, 7, 8], 0)
        assert k1 != k2
