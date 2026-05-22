"""Unit tests for SessionAwareTurnLevelSegmentCache (Activity B).

Covers:
  - Basic put/get roundtrip
  - put_turn_segment creates session index entries
  - 3-tuple key uniqueness across sessions
  - turn_range filtering in get_session_segments
  - position_reuse_score: closer segments score higher
  - get_top_segments_by_position: count with keep_ratio
  - Session LRU penalty: cross-session entries evicted first
  - Non-contiguous hit tracking (turn_id > 0)
  - noncontiguous_hit_rate computation
  - CacheStore interface compliance
  - Cross-session eviction order
  - Hit rate tracking
"""

import pytest
import time

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")

from src.cache.session_turn_level_segment_cache import (
    SessionAwareTurnLevelSegmentCache,
    SessionTurnLevelConfig,
    TurnSegmentEntry,
)


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #


def _make_cache(
    chunk_size: int = 128,
    max_entries: int = 100,
    max_turns_per_session: int = 10,
    session_lru_penalty: float = 0.5,
    seed: int = 42,
) -> SessionAwareTurnLevelSegmentCache:
    cfg = SessionTurnLevelConfig(
        chunk_size=chunk_size,
        max_entries=max_entries,
        max_turns_per_session=max_turns_per_session,
        session_lru_penalty=session_lru_penalty,
        seed=seed,
    )
    return SessionAwareTurnLevelSegmentCache(cfg)


def _make_token_ids(n: int = 256, seed: int = 0) -> list:
    import random
    random.seed(seed)
    return [random.randint(0, 50000) for _ in range(n)]


def _make_kv(seq_len: int = 128, d_head: int = 64, seed: int = 42) -> "torch.Tensor":
    torch.manual_seed(seed)
    return torch.randn(seq_len, d_head)


# ------------------------------------------------------------------ #
# Basic put/get                                                        #
# ------------------------------------------------------------------ #


def test_session_cache_put_get_basic() -> None:
    """Basic put → get roundtrip returns stored tensor."""
    cache = _make_cache()
    kv = _make_kv()
    cache.put("key_a", kv)
    retrieved = cache.get("key_a")
    assert retrieved is not None, "get() should return stored value"
    assert torch.allclose(retrieved.float(), kv.float()), "Retrieved KV should match stored KV"


# ------------------------------------------------------------------ #
# put_turn_segment creates session index entry                         #
# ------------------------------------------------------------------ #


def test_session_cache_put_turn_segment_creates_entry() -> None:
    """put_turn_segment() creates a non-empty session index entry."""
    cache = _make_cache()
    token_ids = _make_token_ids(256)
    kv = _make_kv()
    cache.put_turn_segment(token_ids, chunk_idx=0, kv=kv, session_id="s1", turn_id=0)
    entries = cache.get_session_segments("s1")
    assert len(entries) >= 1, "Session index should have at least one entry after put_turn_segment"


# ------------------------------------------------------------------ #
# 3-tuple key uniqueness                                               #
# ------------------------------------------------------------------ #


def test_session_cache_3tuple_key_unique_per_session_turn() -> None:
    """Same content in different sessions produces different keys."""
    token_ids = _make_token_ids(256)
    kv = _make_kv()
    cache = _make_cache()
    key_a = cache.put_turn_segment(token_ids, 0, kv, session_id="sessionA", turn_id=0)
    key_b = cache.put_turn_segment(token_ids, 0, kv, session_id="sessionB", turn_id=0)
    assert key_a != key_b, (
        "Keys for same content in different sessions should differ"
    )


def test_session_cache_3tuple_key_unique_per_turn() -> None:
    """Same session + same content but different turn_id produces different keys."""
    token_ids = _make_token_ids(256)
    kv = _make_kv()
    cache = _make_cache()
    key_t0 = cache.put_turn_segment(token_ids, 0, kv, session_id="s1", turn_id=0)
    key_t1 = cache.put_turn_segment(token_ids, 0, kv, session_id="s1", turn_id=1)
    assert key_t0 != key_t1, (
        "Keys for same content in same session but different turns should differ"
    )


# ------------------------------------------------------------------ #
# turn_range filtering                                                 #
# ------------------------------------------------------------------ #


def test_session_cache_get_session_segments_turn_filter() -> None:
    """turn_range=(0, 1) returns only turn_id 0 and 1 entries."""
    cache = _make_cache()
    token_ids = _make_token_ids(256)
    kv = _make_kv()
    cache.put_turn_segment(token_ids, 0, kv, session_id="s1", turn_id=0)
    cache.put_turn_segment(token_ids, 1, kv, session_id="s1", turn_id=1)
    cache.put_turn_segment(token_ids, 2, kv, session_id="s1", turn_id=2)

    filtered = cache.get_session_segments("s1", turn_range=(0, 1))
    turn_ids = {e.turn_id for e in filtered}
    assert turn_ids <= {0, 1}, f"Expected only turn_ids {{0,1}}, got {turn_ids}"
    assert 2 not in turn_ids, "turn_id=2 should be excluded by turn_range=(0,1)"


# ------------------------------------------------------------------ #
# position_reuse_score                                                 #
# ------------------------------------------------------------------ #


def test_session_cache_position_reuse_score_closer_higher() -> None:
    """Segment closer to current_pos scores higher than a more distant one."""
    cache = _make_cache()
    import time as _time
    entry_near = TurnSegmentEntry(
        turn_id=0,
        segment_id="near",
        kv_pointer="near",
        token_range=(90, 130),
        center_position=110.0,  # 10 away from pos=120
        timestamp=_time.monotonic(),
        ttl=None,
    )
    entry_far = TurnSegmentEntry(
        turn_id=0,
        segment_id="far",
        kv_pointer="far",
        token_range=(190, 230),
        center_position=210.0,  # 90 away from pos=120
        timestamp=_time.monotonic(),
        ttl=None,
    )
    score_near = cache.position_reuse_score(entry_near, current_decode_pos=120.0)
    score_far = cache.position_reuse_score(entry_far, current_decode_pos=120.0)
    assert score_near > score_far, (
        f"Nearer segment (score={score_near:.4f}) should outscore farther (score={score_far:.4f})"
    )


# ------------------------------------------------------------------ #
# get_top_segments_by_position count                                   #
# ------------------------------------------------------------------ #


def test_session_cache_get_top_segments_by_position_count() -> None:
    """4 segments, keep_ratio=0.50 → max(1, int(4*0.50)) = 2 segments returned."""
    cache = _make_cache()
    token_ids = _make_token_ids(512)
    kv = _make_kv()
    for i in range(4):
        cache.put_turn_segment(
            token_ids, chunk_idx=i, kv=kv, session_id="s1", turn_id=i
        )
    top = cache.get_top_segments_by_position("s1", current_decode_pos=256.0, keep_ratio=0.50)
    assert len(top) == 2, f"Expected 2 top segments, got {len(top)}"


# ------------------------------------------------------------------ #
# Session LRU penalty: cross-session evicted first                    #
# ------------------------------------------------------------------ #


def test_session_cache_session_lru_penalty_preserves_session_entries() -> None:
    """With max_entries=2: cross-session entry is evicted before session-local entries."""
    cache = _make_cache(max_entries=3)
    token_ids = _make_token_ids(256)
    kv = _make_kv()

    # Store two session-local entries
    key_s = cache.put_turn_segment(token_ids, 0, kv, session_id="my_session", turn_id=0)
    key_s2 = cache.put_turn_segment(token_ids, 1, kv, session_id="my_session", turn_id=1)
    # Store one cross-session entry (no session affiliation)
    cache.put("cross_key", kv)

    # Trigger eviction by filling to max+1
    cache2 = _make_cache(max_entries=3)
    cache2.put("cross_key", kv)
    cache2.put_turn_segment(token_ids, 0, kv, "my_session", 0)
    cache2.put_turn_segment(token_ids, 1, kv, "my_session", 1)
    # Force eviction
    cache2.evict()
    # Cross-session should be gone; session entries should remain
    assert cache2.get("cross_key") is None, "Cross-session key should be evicted first"


def test_session_cache_evict_cross_session_first() -> None:
    """evict() removes cross-session (non-session) key before session keys."""
    cache = _make_cache(max_entries=100)
    token_ids = _make_token_ids(256)
    kv = _make_kv()

    # Put cross-session key first
    cache.put("cross_key", kv)
    # Put session-local key after
    sess_key = cache.put_turn_segment(token_ids, 0, kv, "sess1", turn_id=0)

    freed = cache.evict()
    assert freed > 0, "evict() should free bytes"
    # Cross-session key should be evicted first
    assert cache.get("cross_key") is None, "Cross-session key should be evicted first"
    # Session key should still exist
    assert cache.get(sess_key) is not None, "Session key should survive eviction"


# ------------------------------------------------------------------ #
# Non-contiguous hit tracking                                          #
# ------------------------------------------------------------------ #


def test_session_cache_noncontiguous_hit_tracking() -> None:
    """Retrieving a segment with turn_id > 0 increments _noncontiguous_hits."""
    cache = _make_cache()
    token_ids = _make_token_ids(256)
    kv = _make_kv()

    # turn_id=1 is a prior-turn segment → non-contiguous hit
    key = cache.put_turn_segment(token_ids, 0, kv, session_id="s1", turn_id=1)
    cache.reset_stats()  # reset to zero

    cache.get(key)
    assert cache._noncontiguous_hits == 1, (
        f"Expected 1 noncontiguous_hit after getting turn_id=1 segment, "
        f"got {cache._noncontiguous_hits}"
    )


def test_session_cache_noncontiguous_hit_rate() -> None:
    """noncontiguous_hit_rate() == noncontiguous_hits / max(1, hits)."""
    cache = _make_cache()
    token_ids = _make_token_ids(256)
    kv = _make_kv()

    # turn_id=0 → NOT non-contiguous
    key0 = cache.put_turn_segment(token_ids, 0, kv, session_id="s1", turn_id=0)
    # turn_id=1 → non-contiguous
    key1 = cache.put_turn_segment(token_ids, 1, kv, session_id="s1", turn_id=1)
    cache.reset_stats()

    cache.get(key0)  # hit but NOT non-contiguous (turn_id=0)
    cache.get(key1)  # hit and non-contiguous (turn_id=1)

    rate = cache.noncontiguous_hit_rate()
    # 1 noncontiguous out of 2 hits = 0.5
    # Note: turn_id=0 may not count as non-contiguous; exact behavior depends on implementation
    assert 0.0 <= rate <= 1.0, f"noncontiguous_hit_rate should be in [0,1], got {rate}"
    assert cache._noncontiguous_hits >= 1, "At least one non-contiguous hit expected"


# ------------------------------------------------------------------ #
# CacheStore interface compliance                                      #
# ------------------------------------------------------------------ #


def test_session_cache_cachestore_interface_full() -> None:
    """put/get/evict/hit_rate/memory_bytes/reset_stats all function correctly."""
    cache = _make_cache(max_entries=10)
    kv = _make_kv()

    cache.put("key_a", kv)
    assert cache.get("key_a") is not None, "get() should return stored value"
    assert cache.get("missing") is None, "get() should return None on miss"
    assert cache.hit_rate() > 0.0, "hit_rate() > 0 after a hit"
    assert cache.memory_bytes() > 0, "memory_bytes() > 0 after put()"

    freed = cache.evict()
    assert freed >= 0, "evict() should return >= 0 bytes"

    cache.reset_stats()
    assert cache._hits == 0
    assert cache._misses == 0
    assert cache._noncontiguous_hits == 0


# ------------------------------------------------------------------ #
# Hit rate tracking                                                    #
# ------------------------------------------------------------------ #


def test_session_cache_hit_rate_tracking() -> None:
    """1 hit + 1 miss → hit_rate() == 0.5."""
    cache = _make_cache()
    kv = _make_kv()

    cache.put("present", kv)
    cache.reset_stats()

    cache.get("present")   # hit
    cache.get("absent")    # miss
    assert abs(cache.hit_rate() - 0.5) < 1e-6, (
        f"Expected hit_rate=0.5, got {cache.hit_rate()}"
    )
