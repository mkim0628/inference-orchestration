import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from src.cache.base import CacheStore


@dataclass
class LLMProgramStep:
    step_id: str
    token_hash: str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ProgramStep:
    step_id: str
    input_tokens: List[int]
    can_reuse_from: List[str]
    estimated_kv_size: int = 0


@dataclass
class ReusableSegment:
    segment_id: str
    source_step_id: str
    reuse_probability: float
    pinned: bool = False


class LLMProgramDAG:
    """Parses an agentic workflow into a KV-reuse DAG.

    For each step_j that lists step_i in can_reuse_from, computes the
    Jaccard overlap of their input_tokens. If overlap >= reuse_threshold,
    a ReusableSegment is recorded with reuse_probability = overlap.
    """

    def __init__(self, reuse_threshold: float = 0.6) -> None:
        self.reuse_threshold = reuse_threshold
        self._steps: Dict[str, ProgramStep] = {}

    def add_step(self, step: ProgramStep) -> None:
        self._steps[step.step_id] = step

    @staticmethod
    def content_hash(token_ids: List[int]) -> str:
        """SHA-256 position-independent content hash (truncated to 16 hex chars)."""
        data = b"".join(t.to_bytes(4, "little") for t in sorted(token_ids))
        return hashlib.sha256(data).hexdigest()[:16]

    @staticmethod
    def token_overlap_ratio(a: List[int], b: List[int]) -> float:
        """Jaccard overlap of two token lists treated as sets."""
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def compute_reuse_edges(self) -> Dict[str, float]:
        """Return {segment_hash: reuse_probability} for all pairs above threshold."""
        result: Dict[str, float] = {}
        steps = list(self._steps.values())
        for i, step_i in enumerate(steps):
            for step_j in steps[i + 1:]:
                overlap = self.token_overlap_ratio(step_i.input_tokens, step_j.input_tokens)
                if overlap >= self.reuse_threshold:
                    seg_id = self.content_hash(step_i.input_tokens)
                    result[seg_id] = max(result.get(seg_id, 0.0), overlap)
        return result

    def get_pinned_segments(self) -> Set[str]:
        """Return segment hashes with reuse_probability >= 0.5."""
        edges = self.compute_reuse_edges()
        return {seg_id for seg_id, prob in edges.items() if prob >= 0.5}

    def build_reservation_map(self) -> Dict[str, List[ReusableSegment]]:
        """Build {step_id → List[ReusableSegment]} from DAG edges."""
        reservation_map: Dict[str, List[ReusableSegment]] = {}
        for step_j_id, step_j in self._steps.items():
            segs: List[ReusableSegment] = []
            for step_i_id in step_j.can_reuse_from:
                step_i = self._steps.get(step_i_id)
                if step_i is None:
                    continue
                overlap = self.token_overlap_ratio(step_i.input_tokens, step_j.input_tokens)
                if overlap >= self.reuse_threshold:
                    seg_id = self.content_hash(step_i.input_tokens)
                    segs.append(
                        ReusableSegment(
                            segment_id=seg_id,
                            source_step_id=step_i_id,
                            reuse_probability=overlap,
                        )
                    )
            reservation_map[step_j_id] = segs
        return reservation_map


class ThunderAgentStaticSegmentReservationCache(CacheStore):
    """Static segment reservation cache using LLMProgramDAG analysis (Activity B).

    Before execution, parses the agentic workflow DAG to identify high-reuse
    segments and pins them so they are never evicted during the execution window.
    Non-contiguous hits are tracked separately from ordinary prefix hits.

    CacheStore interface fully implemented.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        pin_threshold: float = 0.5,
        max_reservation_budget: float = 0.20,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.max_entries = max_entries
        self.pin_threshold = pin_threshold
        self.max_reservation_budget = max_reservation_budget
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._pinned: Set[str] = set()
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0
        self._reservation_hits: int = 0
        self._reservation_total: int = 0
        self._dag: Optional[LLMProgramDAG] = None
        self._reservation_map: Dict[str, List[ReusableSegment]] = {}

    def register_program(self, dag: LLMProgramDAG) -> None:
        """Pre-reserve pinned segments identified by the DAG."""
        self._dag = dag
        pinned = dag.get_pinned_segments()
        self._pinned.update(pinned)

    def parse_program(self, steps: List[ProgramStep]) -> None:
        """Parse workflow steps into the reservation DAG."""
        self._dag = LLMProgramDAG()
        for step in steps:
            self._dag.add_step(step)
        self._reservation_map = self._dag.build_reservation_map()

    def reserve_segments(self, step_id: str) -> List[str]:
        """Pin high-probability reuse segments for step_id."""
        segs = self._reservation_map.get(step_id, [])
        pinned_ids: List[str] = []
        for seg in segs:
            if seg.reuse_probability >= self.pin_threshold:
                self._pinned.add(seg.segment_id)
                self._reservation_total += 1
                pinned_ids.append(seg.segment_id)
        return pinned_ids

    def release_reservations(self, step_id: str) -> None:
        """Unpin segments after step_id completes."""
        segs = self._reservation_map.get(step_id, [])
        for seg in segs:
            self._pinned.discard(seg.segment_id)

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of all accesses that were non-contiguous (pinned segment) hits."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._noncontiguous_hits / total

    def reservation_hit_rate(self) -> float:
        """Fraction of reserved segments that were actually hit."""
        if self._reservation_total == 0:
            return 0.0
        return self._reservation_hits / self._reservation_total

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        compressed = self.compression_hook(key, value)
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_entries:
                self.evict()
        self._store[key] = compressed.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            if key in self._pinned:
                self._reservation_hits += 1
                self._noncontiguous_hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        """LRU eviction, skipping pinned entries. Returns bytes freed."""
        for key in list(self._store.keys()):
            if key not in self._pinned:
                v = self._store.pop(key)
                return v.nbytes
        return 0

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._reservation_hits = 0
        self._reservation_total = 0
        self._store.clear()
        self._pinned.clear()
        self._reservation_map.clear()
