"""Unit tests for ThunderAgentStaticSegmentReservationCache and LLMProgramDAG (Activity B)."""
import pytest
import torch

from src.cache.thunder_agent_static_reservation_cache import (
    LLMProgramDAG,
    LLMProgramStep,
    ProgramStep,
    ReusableSegment,
    ThunderAgentStaticSegmentReservationCache,
)


def _make_cache(max_entries: int = 10, pin_threshold: float = 0.5) -> ThunderAgentStaticSegmentReservationCache:
    return ThunderAgentStaticSegmentReservationCache(
        max_entries=max_entries,
        pin_threshold=pin_threshold,
        seed=42,
    )


def _make_tensor(n: int = 4) -> torch.Tensor:
    return torch.randn(n)


class TestLLMProgramDAG:
    def test_content_hash_deterministic(self) -> None:
        tokens = [1, 2, 3, 4]
        h1 = LLMProgramDAG.content_hash(tokens)
        h2 = LLMProgramDAG.content_hash(tokens)
        assert h1 == h2

    def test_content_hash_position_independent(self) -> None:
        """Sorted tokens → same hash regardless of input order."""
        h1 = LLMProgramDAG.content_hash([1, 2, 3])
        h2 = LLMProgramDAG.content_hash([3, 1, 2])
        assert h1 == h2

    def test_token_overlap_ratio_full_match(self) -> None:
        assert LLMProgramDAG.token_overlap_ratio([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_token_overlap_ratio_no_match(self) -> None:
        assert LLMProgramDAG.token_overlap_ratio([1, 2], [3, 4]) == pytest.approx(0.0)

    def test_token_overlap_ratio_half(self) -> None:
        # |{1,2} ∩ {2,3}| / |{1,2,3}| = 1/3
        ratio = LLMProgramDAG.token_overlap_ratio([1, 2], [2, 3])
        assert 0.0 < ratio < 1.0

    def test_token_overlap_ratio_empty(self) -> None:
        assert LLMProgramDAG.token_overlap_ratio([], []) == pytest.approx(1.0)

    def test_add_step_and_compute_reuse_edges_above_threshold(self) -> None:
        dag = LLMProgramDAG(reuse_threshold=0.6)
        dag.add_step(ProgramStep("A", [1, 2, 3, 4], []))
        dag.add_step(ProgramStep("B", [1, 2, 3, 5], []))
        edges = dag.compute_reuse_edges()
        # overlap([1,2,3,4],[1,2,3,5]) = 3/5 = 0.6 >= threshold
        assert len(edges) >= 1

    def test_compute_reuse_edges_below_threshold(self) -> None:
        dag = LLMProgramDAG(reuse_threshold=0.6)
        dag.add_step(ProgramStep("A", [1, 2, 3], []))
        dag.add_step(ProgramStep("B", [5, 6, 7], []))
        edges = dag.compute_reuse_edges()
        assert len(edges) == 0

    def test_get_pinned_segments(self) -> None:
        dag = LLMProgramDAG(reuse_threshold=0.5)
        dag.add_step(ProgramStep("A", [1, 2, 3, 4], []))
        dag.add_step(ProgramStep("B", [1, 2, 3, 4], []))  # identical → overlap 1.0
        pinned = dag.get_pinned_segments()
        assert len(pinned) > 0

    def test_build_reservation_map_identifies_reusable(self) -> None:
        dag = LLMProgramDAG(reuse_threshold=0.6)
        dag.add_step(ProgramStep("A", [1, 2, 3, 4], []))
        dag.add_step(ProgramStep("B", [1, 2, 3, 5], ["A"]))
        rmap = dag.build_reservation_map()
        assert len(rmap["B"]) > 0

    def test_build_reservation_map_excludes_low_overlap(self) -> None:
        dag = LLMProgramDAG(reuse_threshold=0.6)
        dag.add_step(ProgramStep("A", [1, 2], []))
        dag.add_step(ProgramStep("B", [9, 10], ["A"]))
        rmap = dag.build_reservation_map()
        assert rmap["B"] == []


class TestThunderAgentCacheInterface:
    def test_put_get(self) -> None:
        cache = _make_cache()
        t = _make_tensor()
        cache.put("k1", t)
        result = cache.get("k1")
        assert result is not None
        assert result.shape == t.shape

    def test_hit_rate_accuracy(self) -> None:
        cache = _make_cache()
        cache.put("k1", _make_tensor())
        cache.get("k1")   # hit
        cache.get("k2")   # miss
        assert cache.hit_rate() == pytest.approx(0.5)

    def test_evict_returns_bytes(self) -> None:
        cache = _make_cache()
        cache.put("k1", torch.ones(4))
        freed = cache.evict()
        assert freed > 0

    def test_memory_bytes(self) -> None:
        cache = _make_cache()
        t = torch.ones(8, dtype=torch.float32)
        cache.put("k1", t)
        assert cache.memory_bytes() == t.nbytes

    def test_reset_stats(self) -> None:
        cache = _make_cache()
        cache.put("k1", _make_tensor())
        cache.get("k1")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0
        assert cache.memory_bytes() == 0


class TestReservationAndPinning:
    def _setup_cache_with_dag(
        self,
    ) -> tuple:
        """Returns (cache, seg_id) where seg_id is the pinned segment hash."""
        cache = _make_cache(max_entries=5, pin_threshold=0.5)
        steps = [
            ProgramStep("A", [1, 2, 3, 4], []),
            ProgramStep("B", [1, 2, 3, 4], ["A"]),  # identical → overlap 1.0
        ]
        cache.parse_program(steps)
        pinned_ids = cache.reserve_segments("B")
        return cache, pinned_ids

    def test_reserve_segments_pins_high_probability(self) -> None:
        cache, pinned_ids = self._setup_cache_with_dag()
        assert len(pinned_ids) > 0
        for seg_id in pinned_ids:
            assert seg_id in cache._pinned

    def test_reserve_segments_skips_low_probability(self) -> None:
        cache = _make_cache(pin_threshold=0.9)
        steps = [
            ProgramStep("A", [1, 2, 3, 4], []),
            ProgramStep("B", [1, 2, 3, 5], ["A"]),  # overlap = 0.6 < 0.9
        ]
        cache.parse_program(steps)
        pinned = cache.reserve_segments("B")
        assert pinned == []

    def test_pinned_segment_not_evicted(self) -> None:
        cache = _make_cache(max_entries=2, pin_threshold=0.5)
        steps = [
            ProgramStep("A", [1, 2, 3, 4], []),
            ProgramStep("B", [1, 2, 3, 4], ["A"]),
        ]
        cache.parse_program(steps)
        pinned_ids = cache.reserve_segments("B")
        assert len(pinned_ids) > 0
        seg_id = pinned_ids[0]
        # Store the pinned segment
        cache.put(seg_id, torch.ones(4))
        # Fill cache to trigger eviction
        cache.put("other1", torch.ones(4))
        cache.put("other2", torch.ones(4))  # triggers evict
        # Pinned segment must still be in store
        assert seg_id in cache._store

    def test_release_reservations_unpins(self) -> None:
        cache, pinned_ids = self._setup_cache_with_dag()
        assert len(pinned_ids) > 0
        cache.release_reservations("B")
        for seg_id in pinned_ids:
            assert seg_id not in cache._pinned

    def test_noncontiguous_hit_rate_above_threshold(self) -> None:
        cache = _make_cache(max_entries=10, pin_threshold=0.5)
        steps = [
            ProgramStep("A", [1, 2, 3, 4], []),
            ProgramStep("B", [1, 2, 3, 4], ["A"]),
        ]
        cache.parse_program(steps)
        pinned_ids = cache.reserve_segments("B")
        seg_id = pinned_ids[0]
        # Store and retrieve the pinned segment
        cache.put(seg_id, torch.ones(4))
        cache.get(seg_id)   # noncontiguous hit
        cache.get("miss")   # miss
        assert cache.noncontiguous_hit_rate() > 0.0

    def test_reservation_hit_rate_tracks_pinned_hits(self) -> None:
        cache = _make_cache(max_entries=10, pin_threshold=0.5)
        steps = [
            ProgramStep("A", [1, 2, 3, 4], []),
            ProgramStep("B", [1, 2, 3, 4], ["A"]),
        ]
        cache.parse_program(steps)
        pinned_ids = cache.reserve_segments("B")
        seg_id = pinned_ids[0]
        cache.put(seg_id, torch.ones(4))
        cache.get(seg_id)   # reservation hit
        assert cache.reservation_hit_rate() > 0.0


class TestLRUEviction:
    def test_lru_eviction_respects_pinned(self) -> None:
        cache = _make_cache(max_entries=3, pin_threshold=0.5)
        steps = [
            ProgramStep("A", [10, 20, 30, 40], []),
            ProgramStep("B", [10, 20, 30, 40], ["A"]),
        ]
        cache.parse_program(steps)
        pinned_ids = cache.reserve_segments("B")
        seg_id = pinned_ids[0]

        cache.put(seg_id, torch.ones(4))  # pinned
        cache.put("x", torch.ones(4))
        cache.put("y", torch.ones(4))
        # Adding one more should evict "x" (LRU non-pinned), not seg_id
        cache.put("z", torch.ones(4))
        assert seg_id in cache._store
        assert "z" in cache._store

    def test_max_entries_enforced(self) -> None:
        cache = _make_cache(max_entries=3)
        for i in range(5):
            cache.put(f"k{i}", torch.ones(2))
        assert len(cache._store) <= 3


class TestLLMProgramStepDataclass:
    def test_llmprogramstep_fields(self) -> None:
        step = LLMProgramStep(step_id="s1", token_hash="abc", dependencies=["s0"])
        assert step.step_id == "s1"
        assert step.token_hash == "abc"
        assert step.dependencies == ["s0"]
