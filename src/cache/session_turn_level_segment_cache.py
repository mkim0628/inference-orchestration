"""SessionAwareTurnLevelSegmentCache — Activity B: Non-Contiguous KV Cache Reuse.

Session-aware, turn-level non-contiguous segment cache with
(content_hash, session_id, turn_id) 3-tuple key and position-aware reuse scoring.
"""

import hashlib
import math
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.cache.base import CacheStore


@dataclass
class TurnSegmentEntry:
    turn_id: int
    segment_id: str           # key derived from (content_hash, session_id, turn_id)
    kv_pointer: str           # actual KV storage key (key into _store)
    token_range: Tuple[int, int]   # (start_pos, end_pos)
    center_position: float    # (start_pos + end_pos) / 2
    timestamp: float          # monotonic time at storage
    ttl: Optional[float]      # seconds; None = no expiry


@dataclass
class SessionTurnLevelConfig:
    chunk_size: int = 128           # tokens per chunk
    max_entries: int = 1000         # total maximum entries
    max_turns_per_session: int = 10  # maximum turns preserved per session
    session_lru_penalty: float = 0.5  # eviction priority for session entries (lower = preserved longer)
    seed: int = 42


class SessionAwareTurnLevelSegmentCache(CacheStore):
    """Session-aware turn-level non-contiguous segment cache (Activity B).

    Fully implements CacheStore interface.

    Key data structures:
      Segment key: SHA-256 hash of (content_hash, session_id, turn_id, layer_idx).
      TurnSegmentIndex: session_id → List[TurnSegmentEntry].
      Session-priority LRU: cross-session entries evicted before session-local entries.

    Position-aware reuse scoring (DapQ principle):
      position_reuse_score(segment, current_pos) = exp(-|center - pos| / decay_factor)
      Segments closer to the current decode position have higher reuse probability.

    Evaluation targets (evaluation_criteria.md §3):
      - Overall cache hit rate +5%p baseline (high priority)
      - Non-contiguous segment hit rate >= 30% of total hits (high priority)
      - KV memory footprint <= baseline +20% (high priority)
    """

    def __init__(self, config: SessionTurnLevelConfig) -> None:
        if _TORCH_AVAILABLE:
            torch.manual_seed(config.seed)
        self.config = config
        # Actual KV storage: kv_key → torch.Tensor
        self._store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Session index: session_id → List[TurnSegmentEntry]
        self._session_index: Dict[str, List[TurnSegmentEntry]] = {}
        # Reverse mapping: kv_key → session_id
        self._key_to_session: Dict[str, str] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0

    @staticmethod
    def _make_segment_key(
        content_hash: str,
        session_id: str,
        turn_id: int,
        layer_idx: int = 0,
    ) -> str:
        """Generate 3-tuple based segment key from (content_hash, session_id, turn_id, layer_idx)."""
        raw = f"{content_hash}|{session_id}|{turn_id}|{layer_idx}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _content_hash(token_ids: List[int], chunk_idx: int, chunk_size: int) -> str:
        """Deterministic content hash for a token chunk."""
        start = chunk_idx * chunk_size
        chunk = token_ids[start:start + chunk_size]
        if not chunk:
            return hashlib.sha256(b"").hexdigest()
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        return hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------------ #
    # Session-turn API                                                     #
    # ------------------------------------------------------------------ #

    def put_turn_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: "torch.Tensor",
        session_id: str,
        turn_id: int,
        layer_idx: int = 0,
        ttl: Optional[float] = None,
    ) -> str:
        """Store a turn-level segment and register it in the session index.

        Algorithm:
          content_hash = _content_hash(token_ids, chunk_idx, chunk_size)
          key = _make_segment_key(content_hash, session_id, turn_id, layer_idx)
          start_pos = chunk_idx * chunk_size
          end_pos = min(start_pos + chunk_size, len(token_ids))
          Register TurnSegmentEntry in _session_index[session_id].
          put(key, kv)
          return key
        """
        content_hash = self._content_hash(token_ids, chunk_idx, self.config.chunk_size)
        key = self._make_segment_key(content_hash, session_id, turn_id, layer_idx)
        start_pos = chunk_idx * self.config.chunk_size
        end_pos = min(start_pos + self.config.chunk_size, len(token_ids))
        entry = TurnSegmentEntry(
            turn_id=turn_id,
            segment_id=key,
            kv_pointer=key,
            token_range=(start_pos, end_pos),
            center_position=(start_pos + end_pos) / 2.0,
            timestamp=time.monotonic(),
            ttl=ttl,
        )
        if session_id not in self._session_index:
            self._session_index[session_id] = []
        self._session_index[session_id].append(entry)
        self._key_to_session[key] = session_id
        self.put(key, kv)
        return key

    def get_session_segments(
        self,
        session_id: str,
        turn_range: Optional[Tuple[int, int]] = None,
    ) -> List[TurnSegmentEntry]:
        """Return list of stored segments for a session.

        Args:
            session_id: target session.
            turn_range: (min_turn, max_turn_inclusive) filter, or None for all.

        Returns:
            List of TurnSegmentEntry for segments that exist in _store.
        """
        entries = self._session_index.get(session_id, [])
        if turn_range is not None:
            lo, hi = turn_range
            entries = [e for e in entries if lo <= e.turn_id <= hi]
        return [e for e in entries if e.kv_pointer in self._store]

    def position_reuse_score(
        self,
        entry: TurnSegmentEntry,
        current_decode_pos: float,
        decay_factor: float = 512.0,
    ) -> float:
        """DapQ position-aware segment reuse probability score.

        score = exp(-|center_position - current_decode_pos| / decay_factor)
        Segments closer to current decode position receive higher scores.
        """
        dist = abs(entry.center_position - current_decode_pos)
        return math.exp(-dist / decay_factor)

    def get_top_segments_by_position(
        self,
        session_id: str,
        current_decode_pos: float,
        keep_ratio: float = 0.50,
        decay_factor: float = 512.0,
    ) -> List[TurnSegmentEntry]:
        """Return top keep_ratio segments ranked by position-aware score.

        Used by DapQSessionSegmentDualReductionPipeline Step 2.
        """
        entries = self.get_session_segments(session_id)
        if not entries:
            return []
        scored = [
            (e, self.position_reuse_score(e, current_decode_pos, decay_factor))
            for e in entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(scored) * keep_ratio))
        return [e for e, _ in scored[:k]]

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: "torch.Tensor") -> None:
        """Store KV tensor. Updates LRU position if key already exists."""
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        self._store[key] = value.detach().clone()

    def get(self, key: str) -> "Optional[torch.Tensor]":
        """Retrieve KV tensor; increments hit/miss counter."""
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            # Non-contiguous hit: segment from a prior turn (turn_id > 0)
            session_id = self._key_to_session.get(key)
            if session_id:
                entries = self._session_index.get(session_id, [])
                hit_entry = next((e for e in entries if e.kv_pointer == key), None)
                if hit_entry and hit_entry.turn_id > 0:
                    self._noncontiguous_hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        """Session-aware LRU eviction.

        Cross-session entries are evicted before session-local entries,
        preserving in-session segments for reuse across turns.
        """
        if not self._store:
            return 0
        # Cross-session keys (no session affiliation) evicted first
        for key in list(self._store.keys()):
            if key not in self._key_to_session:
                v = self._store.pop(key)
                return v.nbytes
        # All entries are session-local: evict LRU (first in OrderedDict)
        key, v = self._store.popitem(last=False)
        session_id = self._key_to_session.pop(key, None)
        if session_id and session_id in self._session_index:
            self._session_index[session_id] = [
                e for e in self._session_index[session_id]
                if e.kv_pointer != key
            ]
        return v.nbytes

    def hit_rate(self) -> float:
        """Cumulative cache hit rate (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous (from prior turns)."""
        return self._noncontiguous_hits / max(1, self._hits)

    def memory_bytes(self) -> int:
        """Current memory footprint in bytes."""
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        """Reset all hit/miss/non-contiguous counters."""
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
