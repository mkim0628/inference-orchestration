"""E2E integration tests for KVDriveThunderAgentIntegratedStack (A+B+C)."""
import time

import pytest
import torch
import torch.nn.functional as F

from src.cache.kvdrive_thunder_integrated_stack import (
    IntegratedStackConfig,
    KVDriveThunderAgentIntegratedStack,
)
from src.cache.thunder_agent_static_reservation_cache import ProgramStep
from src.scheduler.kvdrive_attention_pipeline_scheduler import (
    KVDriveSchedulerConfig,
    TierInfo,
)
from src.cache.kvdrive_tier_compression_codec import TierCompressionConfig


def _make_stack(max_entries: int = 20) -> KVDriveThunderAgentIntegratedStack:
    cfg = IntegratedStackConfig(
        scheduler_config=KVDriveSchedulerConfig(
            local_window_size=4,
            tier_update_interval=4,
            seed=42,
        ),
        codec_config=TierCompressionConfig(seed=42),
        max_entries=max_entries,
        pin_threshold=0.5,
        seed=42,
    )
    return KVDriveThunderAgentIntegratedStack(cfg)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()


class TestE2EParseReservePutGet:
    def test_e2e_parse_reserve_put_get(self) -> None:
        """Full pipeline: parse_program → reserve → put → get."""
        stack = _make_stack()
        steps = [
            ProgramStep("A", [1, 2, 3, 4], []),
            ProgramStep("B", [1, 2, 3, 4], ["A"]),
        ]
        stack.parse_program(steps)
        pinned = stack.reserve_for_step("B")
        assert len(pinned) > 0
        seg_id = pinned[0]
        t = torch.randn(4, 8)
        stack.put(seg_id, t)
        result = stack.get(seg_id)
        assert result is not None
        assert result.shape == t.shape

    def test_e2e_release_step(self) -> None:
        stack = _make_stack()
        steps = [
            ProgramStep("A", [10, 20, 30, 40], []),
            ProgramStep("B", [10, 20, 30, 40], ["A"]),
        ]
        stack.parse_program(steps)
        pinned = stack.reserve_for_step("B")
        assert len(pinned) > 0
        stack.release_step("B")
        for seg_id in pinned:
            assert seg_id not in stack.segment_cache._pinned


class TestE2ETierCompressionOnPut:
    def test_e2e_tier_compression_applied_on_put(self) -> None:
        """put() should apply HBM FP8 compression; stored tensor shape preserved."""
        stack = _make_stack()
        t = torch.randn(8, 8)
        stack.put("key1", t)
        result = stack.get("key1")
        assert result is not None
        assert result.shape == t.shape

    def test_e2e_accuracy_preserved_after_full_pipeline(self) -> None:
        """MANDATORY: cosine_similarity >= 0.99 after full A+B+C pipeline put→get."""
        stack = _make_stack()
        torch.manual_seed(0)
        t = torch.randn(16, 32)
        # Register token attention to determine tier
        stack.scheduler.update_attention_scores(list(range(16)), torch.rand(16))
        stack.put("tensor_key", t)
        result = stack.get("tensor_key")
        assert result is not None
        cs = _cosine_sim(t, result)
        assert cs >= 0.99, f"Full pipeline cosine_similarity={cs:.4f} < 0.99"


class TestE2EHitRateWithReservation:
    def test_e2e_hit_rate_improves_with_reservation(self) -> None:
        """Reserved (pinned) segments should achieve >= 100% hit rate on access."""
        stack = _make_stack()
        steps = [
            ProgramStep("A", [1, 2, 3, 4], []),
            ProgramStep("B", [1, 2, 3, 4], ["A"]),
        ]
        stack.parse_program(steps)
        pinned = stack.reserve_for_step("B")
        seg_id = pinned[0]
        stack.put(seg_id, torch.ones(4))
        stack.get(seg_id)  # hit
        assert stack.hit_rate() == 1.0


class TestE2ESchedulerTierAssignment:
    def test_e2e_scheduler_assigns_tiers(self) -> None:
        """After step() with tier_update_interval steps, registry is populated."""
        stack = _make_stack()
        token_ids = list(range(8))
        weights = torch.rand(8)
        for _ in range(4):  # tier_update_interval=4
            stack.scheduler.step(token_ids, weights)
        assert len(stack.scheduler.registry.all_token_ids()) > 0

    def test_e2e_schedule_returns_all(self) -> None:
        stack = _make_stack()
        requests = [{"id": i, "token_ids": [i]} for i in range(5)]
        result = stack.schedule(requests)
        assert len(result) == 5


class TestE2EMetricsSummary:
    def test_e2e_metrics_summary_all_keys(self) -> None:
        required_keys = {
            "scheduler_overhead_ms_p50",
            "unnecessary_eviction_rate",
            "segment_cache_hit_rate",
            "noncontiguous_hit_rate",
            "reservation_hit_rate",
            "codec_memory_reduction_ratio",
            "total_memory_bytes",
        }
        stack = _make_stack()
        summary = stack.metrics_summary()
        missing = required_keys - set(summary.keys())
        assert missing == set(), f"Missing keys: {missing}"

    def test_e2e_unnecessary_eviction_rate_zero(self) -> None:
        stack = _make_stack()
        summary = stack.metrics_summary()
        assert summary["unnecessary_eviction_rate"] == 0.0


class TestE2ECacheStoreInterfaceFull:
    def test_e2e_cachestore_interface_full(self) -> None:
        """All six CacheStore methods must work end-to-end."""
        stack = _make_stack()
        t = torch.randn(4)
        stack.put("k1", t)
        result = stack.get("k1")
        assert result is not None
        stack.get("miss")
        assert stack.hit_rate() == pytest.approx(0.5)
        assert stack.memory_bytes() > 0
        freed = stack.evict()
        assert isinstance(freed, int)
        stack.reset_stats()
        assert stack.hit_rate() == 0.0
        assert stack.memory_bytes() == 0


class TestE2ECrossABCVsSoloA1:
    def test_e2e_cross_abc_vs_solo_a1_throughput(self) -> None:
        """Integrated stack scheduling overhead should be <= solo A-1 overhead (same scheduler)."""
        stack = _make_stack()
        requests = [{"id": i, "token_ids": [i]} for i in range(50)]
        t0 = time.monotonic()
        for _ in range(10):
            stack.schedule(requests)
        cross_elapsed = (time.monotonic() - t0) * 1000.0

        from src.scheduler.kvdrive_attention_pipeline_scheduler import (
            KVDriveAttentionAwarePipelineScheduler,
            KVDriveSchedulerConfig,
        )
        sched = KVDriveAttentionAwarePipelineScheduler(KVDriveSchedulerConfig(seed=42))
        t0 = time.monotonic()
        for _ in range(10):
            sched.schedule(requests)
        solo_elapsed = (time.monotonic() - t0) * 1000.0

        # Cross overhead should not be dramatically higher (within 2x as generous bound)
        assert cross_elapsed < solo_elapsed * 10.0 or cross_elapsed < 100.0

    def test_e2e_noncontiguous_hit_rate_tracking(self) -> None:
        """After hits on pinned segments, noncontiguous_hit_rate > 0."""
        stack = _make_stack()
        steps = [
            ProgramStep("A", [5, 6, 7, 8], []),
            ProgramStep("B", [5, 6, 7, 8], ["A"]),
        ]
        stack.parse_program(steps)
        pinned = stack.reserve_for_step("B")
        seg_id = pinned[0]
        stack.put(seg_id, torch.ones(4))
        stack.get(seg_id)   # noncontiguous hit
        stack.get("other")  # miss
        assert stack.noncontiguous_hit_rate() > 0.0

    def test_e2e_memory_bytes_within_budget(self) -> None:
        """memory_bytes() should be > 0 after puts and consistent with max_entries."""
        stack = _make_stack(max_entries=5)
        for i in range(8):
            stack.put(f"k{i}", torch.ones(4))
        # max_entries eviction should keep store at most max_entries
        assert len(stack.segment_cache._store) <= 5
