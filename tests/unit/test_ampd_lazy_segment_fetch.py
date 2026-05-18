"""Unit tests for AMPDLazySegmentFetchScheduler (Activity A).

Covers:
  - SegmentMetadataRegistry: register, cancel, ratio
  - AMPDLazySegmentFetchScheduler: pre_resolve, confirm, fetch, schedule, metrics
"""

import asyncio
import time

import pytest
import torch

from src.engine.runner import InferenceRequest
from src.scheduler.ampd_lazy_segment_fetch import (
    AMPDLazySchedulerConfig,
    AMPDLazySegmentFetchScheduler,
    KVSegment,
    SegmentMeta,
    SegmentMetadataRegistry,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_meta(seg_id: str, tier: str = "HBM") -> SegmentMeta:
    return SegmentMeta(
        segment_id=seg_id,
        source_node_id="local",
        tier=tier,  # type: ignore[arg-type]
        approx_size_bytes=128 * 64 * 2,
        position_range=(0, 128),
    )


def _make_request(req_id: str = "req0") -> InferenceRequest:
    return InferenceRequest(request_id=req_id, token_ids=list(range(128)))


def _default_scheduler() -> AMPDLazySegmentFetchScheduler:
    cfg = AMPDLazySchedulerConfig(seed=42)
    return AMPDLazySegmentFetchScheduler(cfg)


# ------------------------------------------------------------------ #
# SegmentMetadataRegistry tests                                        #
# ------------------------------------------------------------------ #

def test_segment_metadata_registry_register() -> None:
    reg = SegmentMetadataRegistry()
    meta = _make_meta("seg_a")
    reg.register(meta)
    result = reg.get("seg_a")
    assert result is not None
    assert result.segment_id == "seg_a"


def test_segment_metadata_registry_cancel() -> None:
    reg = SegmentMetadataRegistry()
    meta = _make_meta("seg_b")
    reg.register(meta)
    assert reg.unnecessary_transfer_ratio() == 0.0
    reg.cancel("seg_b")
    assert reg.unnecessary_transfer_ratio() == 1.0


def test_segment_metadata_registry_ratio() -> None:
    reg = SegmentMetadataRegistry()
    for i in range(5):
        reg.register(_make_meta(f"seg_{i}"))
    reg.cancel("seg_0")
    reg.cancel("seg_1")
    ratio = reg.unnecessary_transfer_ratio()
    assert abs(ratio - 0.4) < 1e-9


def test_pre_resolve_segments_no_kv_transfer() -> None:
    sched = _default_scheduler()
    req = _make_request()
    metas = [_make_meta(f"seg_{i}") for i in range(3)]
    candidate_ids = [m.segment_id for m in metas]
    sched.pre_resolve_segments(req, candidate_ids, metas)
    # Metadata registered — no KV data: registry entries have no kv_tensor attribute
    for seg_id in candidate_ids:
        stored = sched.registry.get(seg_id)
        assert stored is not None
        assert not hasattr(stored, "kv_tensor")


def test_confirm_segments_cancels_non_confirmed() -> None:
    sched = _default_scheduler()
    req = _make_request()
    candidates = ["a", "b", "c"]
    metas = [_make_meta(s) for s in candidates]
    sched.pre_resolve_segments(req, candidates, metas)
    sched.confirm_segments(candidates, ["a"])
    # 2 of 3 cancelled → ratio = 2/3
    ratio = sched.registry.unnecessary_transfer_ratio()
    assert abs(ratio - 2 / 3) < 1e-9


def test_fetch_segments_lazy_yields_kvsegment() -> None:
    sched = _default_scheduler()
    req = _make_request()
    meta = _make_meta("seg1", tier="HBM")
    sched.pre_resolve_segments(req, ["seg1"], [meta])
    sched.confirm_segments(["seg1"], ["seg1"])

    async def _run():
        segments = []
        async for seg in sched.fetch_segments_lazy(["seg1"]):
            segments.append(seg)
        return segments

    segments = asyncio.get_event_loop().run_until_complete(_run())
    assert len(segments) == 1
    assert isinstance(segments[0], KVSegment)
    assert segments[0].segment_id == "seg1"


def test_fetch_segments_lazy_hbm_latency() -> None:
    cfg = AMPDLazySchedulerConfig(hbm_fetch_latency_ms=0.01, seed=42)
    sched = AMPDLazySegmentFetchScheduler(cfg)
    req = _make_request()
    meta = _make_meta("seg_hbm", tier="HBM")
    sched.pre_resolve_segments(req, ["seg_hbm"], [meta])

    async def _run():
        segs = []
        async for s in sched.fetch_segments_lazy(["seg_hbm"]):
            segs.append(s)
        return segs

    segs = asyncio.get_event_loop().run_until_complete(_run())
    assert len(segs) == 1
    # load_latency_ms should match configured HBM latency
    assert abs(segs[0].load_latency_ms - cfg.hbm_fetch_latency_ms) < 0.01


def test_fetch_segments_lazy_ddr_latency() -> None:
    cfg = AMPDLazySchedulerConfig(ddr_fetch_latency_ms=0.5, seed=42)
    sched = AMPDLazySegmentFetchScheduler(cfg)
    req = _make_request()
    meta = _make_meta("seg_ddr", tier="DDR")
    sched.pre_resolve_segments(req, ["seg_ddr"], [meta])

    async def _run():
        segs = []
        async for s in sched.fetch_segments_lazy(["seg_ddr"]):
            segs.append(s)
        return segs

    segs = asyncio.get_event_loop().run_until_complete(_run())
    assert len(segs) == 1
    assert abs(segs[0].load_latency_ms - cfg.ddr_fetch_latency_ms) < 0.05


def test_scheduling_overhead_below_01ms() -> None:
    sched = _default_scheduler()
    requests = [_make_request(f"r{i}") for i in range(10)]
    t0 = time.monotonic()
    sched.schedule(requests)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert elapsed_ms < 0.1, f"schedule() took {elapsed_ms:.3f} ms, expected < 0.1 ms"


def test_schedule_returns_all_requests() -> None:
    sched = _default_scheduler()
    requests = [_make_request(f"r{i}") for i in range(5)]
    result = sched.schedule(requests)
    assert len(result) == len(requests)


def test_unnecessary_transfer_ratio_zero_initial() -> None:
    sched = _default_scheduler()
    assert sched.unnecessary_transfer_ratio() == 0.0


def test_reset_stats_clears_counts() -> None:
    sched = _default_scheduler()
    req = _make_request()
    metas = [_make_meta(f"seg_{i}") for i in range(3)]
    candidate_ids = [m.segment_id for m in metas]
    sched.pre_resolve_segments(req, candidate_ids, metas)
    sched.confirm_segments(candidate_ids, [candidate_ids[0]])
    assert sched.unnecessary_transfer_ratio() > 0.0
    sched.reset_stats()
    assert sched.unnecessary_transfer_ratio() == 0.0
