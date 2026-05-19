"""Unit tests for KVDriveAttentionAwarePipelineScheduler (Activity A)."""
import time

import pytest
import torch

from src.scheduler.kvdrive_attention_pipeline_scheduler import (
    KVDriveAttentionAwarePipelineScheduler,
    KVDriveSchedulerConfig,
    KVTierRegistry,
    TierInfo,
    make_kvdrive_scheduler_class,
)


def _make_scheduler(
    local_window_size: int = 8,
    tier_update_interval: int = 4,
    attn_hbm_threshold: float = 0.8,
    attn_dram_threshold: float = 0.3,
    enable_multinode: bool = False,
    seed: int = 42,
) -> KVDriveAttentionAwarePipelineScheduler:
    cfg = KVDriveSchedulerConfig(
        local_window_size=local_window_size,
        tier_update_interval=tier_update_interval,
        attn_hbm_threshold=attn_hbm_threshold,
        attn_dram_threshold=attn_dram_threshold,
        enable_multinode=enable_multinode,
        seed=seed,
    )
    return KVDriveAttentionAwarePipelineScheduler(cfg)


class TestKVTierRegistry:
    def test_set_and_get(self) -> None:
        reg = KVTierRegistry()
        info = TierInfo(tier="HBM", physical_location="hbm:0", approx_size=128)
        reg.set_tier(1, info)
        assert reg.get_tier(1) is info

    def test_missing_returns_none(self) -> None:
        reg = KVTierRegistry()
        assert reg.get_tier(999) is None

    def test_all_token_ids(self) -> None:
        reg = KVTierRegistry()
        reg.set_tier(10, TierInfo("HBM", "hbm:0", 64))
        reg.set_tier(20, TierInfo("DRAM", "dram:0", 64))
        ids = reg.all_token_ids()
        assert set(ids) == {10, 20}

    def test_clear(self) -> None:
        reg = KVTierRegistry()
        reg.set_tier(1, TierInfo("SSD", "ssd:/tmp", 64))
        reg.clear()
        assert reg.get_tier(1) is None


class TestTierAssignment:
    def test_window_always_hbm(self) -> None:
        """The most recent local_window_size tokens should all be HBM."""
        sched = _make_scheduler(local_window_size=4)
        token_ids = list(range(10))
        # Give early tokens zero attention, recent ones some attention
        for i in range(10):
            sched._cumul_attn[i] = 0.0  # all zero → would map to SSD without window
        sched.assign_tiers(token_ids)
        for tid in token_ids[-4:]:
            info = sched.registry.get_tier(tid)
            assert info is not None and info.tier == "HBM", f"token {tid} should be HBM"

    def test_low_attn_gets_ssd(self) -> None:
        """Tokens with cumul_attn == 0 (and outside the window) → SSD."""
        sched = _make_scheduler(local_window_size=2)
        token_ids = list(range(10))
        for tid in token_ids:
            sched._cumul_attn[tid] = 0.0
        sched.assign_tiers(token_ids)
        # Tokens outside the window of 2 at the end should be SSD
        for tid in token_ids[:-2]:
            info = sched.registry.get_tier(tid)
            assert info is not None and info.tier == "SSD", f"token {tid} should be SSD"

    def test_high_attn_stays_hbm(self) -> None:
        """Token with highest cumul_attn (outside window) → HBM."""
        sched = _make_scheduler(local_window_size=1)
        token_ids = list(range(5))
        for tid in token_ids:
            sched._cumul_attn[tid] = 0.0
        sched._cumul_attn[0] = 1.0  # highest; outside window (window covers token 4)
        sched.assign_tiers(token_ids)
        info = sched.registry.get_tier(0)
        assert info is not None and info.tier == "HBM"

    def test_mid_attn_gets_dram(self) -> None:
        """Token with normalized score between thresholds → DRAM."""
        sched = _make_scheduler(local_window_size=1)
        # token_ids: [0,1,2], window covers only token 2
        # token 0: score=0.0 → SSD; token 1: score=0.5 (mid) → DRAM; token 2 in window
        sched._cumul_attn[0] = 0.0
        sched._cumul_attn[1] = 0.5
        sched._cumul_attn[2] = 1.0  # max
        sched.assign_tiers([0, 1, 2])
        info = sched.registry.get_tier(1)
        assert info is not None and info.tier == "DRAM"


class TestStepAndTierUpdateInterval:
    def test_step_triggers_tier_update_at_interval(self) -> None:
        """Tier registry should be populated after tier_update_interval steps."""
        sched = _make_scheduler(tier_update_interval=4, local_window_size=2)
        token_ids = [10, 11, 12, 13]
        weights = torch.tensor([0.1, 0.5, 0.3, 0.9])
        # Run 4 steps → triggers assign_tiers
        for _ in range(4):
            sched.step(token_ids, weights)
        # At least one token should be registered
        assert len(sched.registry.all_token_ids()) > 0

    def test_step_no_tier_update_before_interval(self) -> None:
        """Before interval, registry stays empty."""
        sched = _make_scheduler(tier_update_interval=4)
        token_ids = [1, 2]
        weights = torch.ones(2)
        for _ in range(3):
            sched.step(token_ids, weights)
        # No assign_tiers yet
        assert len(sched.registry.all_token_ids()) == 0


class TestScheduleMethod:
    def test_schedule_returns_all_requests(self) -> None:
        sched = _make_scheduler()
        requests = [{"id": i, "token_ids": [i]} for i in range(10)]
        result = sched.schedule(requests)
        assert len(result) == len(requests)

    def test_scheduling_overhead_below_5ms(self) -> None:
        """schedule() for 100 requests must complete in < 5ms."""
        sched = _make_scheduler()
        requests = [{"id": i, "token_ids": [i]} for i in range(100)]
        t0 = time.monotonic()
        sched.schedule(requests)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        assert elapsed_ms < 5.0, f"schedule overhead {elapsed_ms:.2f}ms >= 5ms"

    def test_reset_stats_clears_times(self) -> None:
        sched = _make_scheduler()
        sched.schedule([{"id": 0}])
        sched.reset_stats()
        assert sched.scheduling_overhead_ms_p50() == 0.0

    def test_unnecessary_eviction_rate_always_zero(self) -> None:
        sched = _make_scheduler()
        assert sched.unnecessary_eviction_rate() == 0.0


class TestAttentionScoreUpdate:
    def test_update_attention_scores_ema(self) -> None:
        """EMA should keep values in [0, max_input] and converge."""
        sched = _make_scheduler()
        token_ids = [1]
        # Feed the same weight repeatedly
        for _ in range(20):
            sched.update_attention_scores(token_ids, torch.tensor([1.0]))
        v = sched._cumul_attn[1]
        assert 0.0 < v <= 1.0

    def test_register_token_attention_mixin(self) -> None:
        """Mixin method register_token_attention via string token_id."""
        sched = _make_scheduler()
        sched.register_token_attention("5", 0.7)
        tid = 5
        assert sched._cumul_attn[tid] == pytest.approx(0.05 * 0.7, abs=1e-6)


class TestMultinodeFlag:
    def test_enable_multinode_flag_no_error(self) -> None:
        """enable_multinode=True must not raise on CPU."""
        sched = _make_scheduler(enable_multinode=True)
        requests = [{"id": 0, "token_ids": [0]}]
        result = sched.schedule(requests)
        assert len(result) == 1


class TestFactory:
    def test_make_kvdrive_scheduler_class_returns_type(self) -> None:
        cls = make_kvdrive_scheduler_class()
        assert isinstance(cls, type)

    def test_make_kvdrive_scheduler_class_instantiable(self) -> None:
        cls = make_kvdrive_scheduler_class()
        cfg = KVDriveSchedulerConfig(seed=0)
        obj = cls(cfg)
        assert hasattr(obj, "schedule")
        result = obj.schedule([{"id": 1}])
        assert len(result) == 1
