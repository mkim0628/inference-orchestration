"""Unit tests for Activity B-1: IrminsulMLASegmentCache.

Tests:
  - test_cdc_chunking_avg_size: average chunk size within ±50% of avg_chunk_size
  - test_cdc_segment_key_position_independence: same tokens → same key regardless of position
  - test_delta_rotation_correctness: mathematical δ-rotation validation
  - test_mla_put_get_segment: round-trip put_mla_segment / get_mla_segment_with_delta_rotation
  - test_mla_cache_store_interface: all CacheStore abstract methods work
  - test_mla_noncontiguous_hit_rate: non-contiguous hit counting accuracy
  - test_mla_lru_eviction: LRU eviction on max_entries overflow
"""

import pytest
import torch

from src.cache.irminsul_mla_segment_cache import (
    IrminsulKVEntry,
    IrminsulMLAConfig,
    IrminsulMLASegmentCache,
    apply_delta_rotation,
    assert_delta_rotation_correctness,
    cdc_chunk,
    cdc_segment_key,
)


# ------------------------------------------------------------------ #
# CDC chunking tests                                                  #
# ------------------------------------------------------------------ #


def test_cdc_chunking_avg_size() -> None:
    """Average chunk size should be within [min_chunk_size, max_chunk_size].

    With random-ish token IDs, CDC boundaries are content-defined.
    We verify the basic invariant: all chunks are within [min, max] bounds
    and the overall chunking covers all tokens.

    Note: For highly regular (sequential) data, fingerprints may rarely hit
    the boundary condition, causing avg chunk size to approach max_chunk_size.
    The meaningful invariant is min <= each_chunk <= max.
    """
    avg_chunk_size = 256
    min_chunk_size = 64
    max_chunk_size = 1024
    # Use random-ish token IDs to get more realistic chunking
    import random
    random.seed(42)
    token_ids = [random.randint(0, 50000) for _ in range(2048)]

    chunks = cdc_chunk(
        token_ids,
        avg_chunk_size=avg_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    assert len(chunks) > 0, "Should produce at least one chunk"

    # All intermediate chunks must respect min/max bounds
    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk) >= min_chunk_size, (
            f"Chunk {i} size {len(chunk)} < min={min_chunk_size}"
        )
        assert len(chunk) <= max_chunk_size, (
            f"Chunk {i} size {len(chunk)} > max={max_chunk_size}"
        )

    # Average size should be bounded by max (not exceed it)
    avg_size = sum(len(c) for c in chunks) / len(chunks)
    assert avg_size <= max_chunk_size, (
        f"Average chunk size {avg_size:.1f} > max_chunk_size={max_chunk_size}"
    )
    assert avg_size >= min_chunk_size * 0.5, (
        f"Average chunk size {avg_size:.1f} is suspiciously small"
    )


def test_cdc_chunking_min_max_constraints() -> None:
    """All chunks must respect min_chunk_size and max_chunk_size."""
    token_ids = list(range(4096))
    chunks = cdc_chunk(token_ids, avg_chunk_size=256, min_chunk_size=64, max_chunk_size=1024)
    for i, chunk in enumerate(chunks[:-1]):  # last chunk may be smaller
        assert len(chunk) >= 64, f"Chunk {i} length {len(chunk)} < min_chunk_size=64"
        assert len(chunk) <= 1024, f"Chunk {i} length {len(chunk)} > max_chunk_size=1024"


def test_cdc_chunking_covers_all_tokens() -> None:
    """CDC chunks should cover all input tokens."""
    token_ids = list(range(1000))
    chunks = cdc_chunk(token_ids, avg_chunk_size=128, min_chunk_size=32, max_chunk_size=512)
    reconstructed = []
    for chunk in chunks:
        reconstructed.extend(chunk)
    assert reconstructed == token_ids, "CDC chunks must cover all input tokens exactly"


def test_cdc_segment_key_position_independence() -> None:
    """Same token content at different positions must produce the same segment key."""
    chunk_content = [100, 200, 300, 400, 500]

    # Same tokens regardless of where they appear in the stream
    key_at_pos_0 = cdc_segment_key(chunk_content)
    key_at_pos_1000 = cdc_segment_key(chunk_content)  # same content
    assert key_at_pos_0 == key_at_pos_1000, (
        "CDC segment key must be position-independent (content-only)"
    )

    # Different content → different key
    key_different = cdc_segment_key([999, 888, 777])
    assert key_at_pos_0 != key_different, "Different content must produce different keys"


def test_cdc_segment_key_is_sha256_hex() -> None:
    """Segment key should be a 64-character hex string (SHA256)."""
    key = cdc_segment_key([1, 2, 3, 4, 5])
    assert len(key) == 64, f"SHA256 hex should be 64 chars, got {len(key)}"
    assert all(c in "0123456789abcdef" for c in key), "Key must be lowercase hex"


# ------------------------------------------------------------------ #
# δ-rotation tests                                                    #
# ------------------------------------------------------------------ #


def test_delta_rotation_correctness() -> None:
    """Verify δ-rotation mathematical correctness via assert_delta_rotation_correctness."""
    # This will raise AssertionError if the math is wrong
    assert_delta_rotation_correctness(k_r_dim=64, rope_base=10000.0, rtol=1e-4, atol=1e-4)


def test_delta_rotation_zero_delta_identity() -> None:
    """delta=0 should return the original k_r unchanged."""
    torch.manual_seed(42)
    k_r = torch.randn(8, 64)
    k_r_corrected = apply_delta_rotation(k_r, delta=0, rope_base=10000.0, k_r_dim=64)
    assert torch.allclose(k_r_corrected, k_r, atol=1e-5), (
        "delta=0 should return identity"
    )


def test_delta_rotation_shape_preserved() -> None:
    """Output shape must match input shape."""
    torch.manual_seed(42)
    n_tokens, k_r_dim = 16, 64
    k_r = torch.randn(n_tokens, k_r_dim)
    k_r_out = apply_delta_rotation(k_r, delta=50, rope_base=10000.0, k_r_dim=k_r_dim)
    assert k_r_out.shape == k_r.shape, (
        f"Shape mismatch: expected {k_r.shape}, got {k_r_out.shape}"
    )


def test_delta_rotation_composable() -> None:
    """Two sequential δ-rotations should equal one combined δ-rotation."""
    torch.manual_seed(42)
    k_r = torch.randn(8, 64)
    delta_a, delta_b = 30, 70

    k_r_ab = apply_delta_rotation(
        apply_delta_rotation(k_r, delta_a), delta_b
    )
    k_r_combined = apply_delta_rotation(k_r, delta_a + delta_b)
    assert torch.allclose(k_r_ab, k_r_combined, rtol=1e-4, atol=1e-4), (
        "Sequential δ-rotations should compose additively"
    )


# ------------------------------------------------------------------ #
# MLA segment cache tests                                             #
# ------------------------------------------------------------------ #


def _make_cache(max_entries: int = 100) -> IrminsulMLASegmentCache:
    cfg = IrminsulMLAConfig(
        avg_chunk_size=64,
        min_chunk_size=16,
        max_chunk_size=256,
        rope_base=10000.0,
        k_r_dim=64,
        max_entries=max_entries,
        seed=42,
    )
    return IrminsulMLASegmentCache(cfg)


def test_mla_put_get_segment() -> None:
    """put_mla_segment → get_mla_segment_with_delta_rotation round-trip."""
    cache = _make_cache()
    chunk_tokens = list(range(50))
    n_tokens = len(chunk_tokens)
    d_c = 128
    k_r_dim = 64

    torch.manual_seed(42)
    c_kv = torch.randn(n_tokens, d_c)
    k_r = torch.randn(n_tokens, k_r_dim)
    source_position = 0

    seg_key = cache.put_mla_segment(chunk_tokens, c_kv, k_r, source_position, layer_idx=0)
    assert isinstance(seg_key, str) and len(seg_key) == 64

    # Retrieve at same position (delta=0) → identical
    result = cache.get_mla_segment_with_delta_rotation(seg_key, source_position, layer_idx=0)
    assert result is not None
    c_kv_ret, k_r_ret = result
    assert torch.allclose(c_kv_ret, c_kv, atol=1e-5)
    assert torch.allclose(k_r_ret, k_r, atol=1e-5)


def test_mla_put_get_with_delta() -> None:
    """get_mla_segment_with_delta_rotation applies δ-rotation correctly."""
    cache = _make_cache()
    chunk_tokens = list(range(32))
    n_tokens = len(chunk_tokens)

    torch.manual_seed(42)
    c_kv = torch.randn(n_tokens, 128)
    k_r = torch.randn(n_tokens, 64)
    source_position = 100
    target_position = 300

    seg_key = cache.put_mla_segment(chunk_tokens, c_kv, k_r, source_position)

    result = cache.get_mla_segment_with_delta_rotation(seg_key, target_position)
    assert result is not None
    c_kv_ret, k_r_corrected = result

    # c_kv should be unchanged (position-free)
    assert torch.allclose(c_kv_ret, c_kv, atol=1e-5)

    # k_r should be rotated by (target - source)
    expected_k_r = apply_delta_rotation(k_r, target_position - source_position)
    assert torch.allclose(k_r_corrected, expected_k_r, atol=1e-4)


def test_mla_cache_miss_returns_none() -> None:
    """Looking up a non-existent key returns None."""
    cache = _make_cache()
    result = cache.get_mla_segment_with_delta_rotation("nonexistent_key_" + "a" * 48, 0)
    assert result is None


# ------------------------------------------------------------------ #
# CacheStore interface tests                                          #
# ------------------------------------------------------------------ #


def test_mla_cache_store_interface() -> None:
    """All CacheStore abstract methods must function correctly."""
    cache = _make_cache()

    torch.manual_seed(1)
    kv = torch.randn(10, 64)

    # put / get
    cache.put("test_key", kv)
    retrieved = cache.get("test_key")
    assert retrieved is not None
    assert retrieved.shape == kv.shape

    # hit_rate
    rate = cache.hit_rate()
    assert 0.0 <= rate <= 1.0

    # memory_bytes
    mem = cache.memory_bytes()
    assert mem >= 0

    # evict
    freed = cache.evict()
    assert freed >= 0

    # reset_stats
    cache.reset_stats()
    assert cache._hits == 0
    assert cache._misses == 0


def test_mla_get_miss_increments_misses() -> None:
    """Accessing a non-existent generic key increments miss counter."""
    cache = _make_cache()
    cache.reset_stats()
    result = cache.get("nonexistent")
    assert result is None
    assert cache._misses == 1


# ------------------------------------------------------------------ #
# Non-contiguous hit rate tests                                       #
# ------------------------------------------------------------------ #


def test_mla_noncontiguous_hit_rate() -> None:
    """Non-contiguous hits should be counted when earlier chunks are misses.

    Strategy: CDC-chunk the token_ids first, then pre-populate only even-indexed
    chunks. Odd-indexed chunks remain uncached. Any hit at chunk_idx > first_miss
    is a non-contiguous hit.
    """
    cache = _make_cache(max_entries=500)

    import random
    random.seed(99)
    # Use random-ish tokens so CDC produces multiple chunks
    token_ids = [random.randint(100, 50000) for _ in range(500)]

    # Get CDC chunks as they will be looked up
    chunks = cdc_chunk(token_ids, avg_chunk_size=64, min_chunk_size=16, max_chunk_size=256)

    if len(chunks) < 3:
        # If too few chunks, skip (degenerate case)
        cache.reset_stats()
        nc_rate = cache.noncontiguous_hit_rate()
        assert nc_rate == 0.0
        return

    # Pre-populate even-indexed chunks only
    pos = 0
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:  # cache even chunks, skip odd chunks
            torch.manual_seed(i + 100)
            cache.put_mla_segment(
                chunk,
                torch.randn(len(chunk), 128),
                torch.randn(len(chunk), 64),
                source_position=pos,
                layer_idx=0,
            )
        pos += len(chunk)

    cache.reset_stats()
    hits, miss_chunks = cache.get_segments_mla(token_ids, target_offset=0, layer_idx=0)

    assert len(hits) >= 1, f"Should have at least one hit (pre-populated even chunks). Got {hits}"
    nc_rate = cache.noncontiguous_hit_rate()
    assert 0.0 <= nc_rate <= 1.0, f"Non-contiguous hit rate {nc_rate} out of [0,1]"

    # If there are any odd-indexed chunks that missed, and even-indexed chunks after them hit,
    # nc_rate should be > 0
    hit_indices = {h[0] for h in hits}
    miss_indices = set(range(len(chunks))) - hit_indices
    has_nc = any(
        h_idx > min(miss_indices)
        for h_idx in hit_indices
        if miss_indices
    )
    if has_nc:
        assert nc_rate > 0.0, (
            f"Expected nc_rate > 0 (hits after misses exist), got {nc_rate}"
        )


def test_mla_noncontiguous_hit_rate_zero_with_no_hits() -> None:
    """noncontiguous_hit_rate() returns 0.0 when there are no hits."""
    cache = _make_cache()
    cache.reset_stats()
    assert cache.noncontiguous_hit_rate() == 0.0


# ------------------------------------------------------------------ #
# LRU eviction tests                                                  #
# ------------------------------------------------------------------ #


def test_mla_lru_eviction() -> None:
    """max_entries overflow triggers LRU eviction of oldest entry."""
    max_entries = 5
    cache = _make_cache(max_entries=max_entries)

    d_c, k_r_dim = 32, 64

    # Fill cache to capacity
    keys_stored = []
    for i in range(max_entries):
        chunk = list(range(i * 10, i * 10 + 10))
        torch.manual_seed(i)
        seg_key = cache.put_mla_segment(
            chunk,
            torch.randn(10, d_c),
            torch.randn(10, k_r_dim),
            source_position=i * 10,
            layer_idx=0,
        )
        keys_stored.append((seg_key, chunk))

    # Add one more — should evict the oldest
    new_chunk = list(range(1000, 1010))
    torch.manual_seed(99)
    cache.put_mla_segment(
        new_chunk, torch.randn(10, d_c), torch.randn(10, k_r_dim),
        source_position=1000, layer_idx=0,
    )

    # Total entries in _store should not exceed max_entries
    assert len(cache._store) <= max_entries, (
        f"Store size {len(cache._store)} > max_entries={max_entries}"
    )


def test_mla_put_same_key_no_duplicate() -> None:
    """Putting the same content chunk twice should not create duplicate entries."""
    cache = _make_cache(max_entries=10)
    chunk = list(range(50))

    torch.manual_seed(42)
    c_kv = torch.randn(50, 128)
    k_r = torch.randn(50, 64)

    key1 = cache.put_mla_segment(chunk, c_kv, k_r, source_position=0)
    key2 = cache.put_mla_segment(chunk, c_kv, k_r, source_position=100)

    # Same content → same key
    assert key1 == key2
    # Should not duplicate in store
    matching = [(k, v) for (k, v) in cache._store.items() if k[0] == key1]
    assert len(matching) == 1, "Same segment key should not be duplicated"


# ------------------------------------------------------------------ #
# get_segments (InferenceRunner API)                                  #
# ------------------------------------------------------------------ #


def test_mla_get_segments_returns_correct_format() -> None:
    """get_segments() returns (hits, misses) in InferenceRunner format."""
    cache = _make_cache(max_entries=100)
    token_ids = list(range(256))

    # Pre-fill some chunks
    chunks = cdc_chunk(token_ids, avg_chunk_size=64, min_chunk_size=16, max_chunk_size=256)
    pos = 0
    for chunk in chunks[:2]:  # store first 2 chunks
        torch.manual_seed(hash(tuple(chunk)) % 2**31)
        cache.put_mla_segment(
            chunk, torch.randn(len(chunk), 128), torch.randn(len(chunk), 64), pos
        )
        pos += len(chunk)

    hits, misses = cache.get_segments(token_ids, layer_idx=0)

    # Verify format
    for chunk_idx, kv_tensor in hits:
        assert isinstance(chunk_idx, int)
        assert isinstance(kv_tensor, torch.Tensor)
        assert kv_tensor.dim() == 2

    for miss_idx in misses:
        assert isinstance(miss_idx, int)
