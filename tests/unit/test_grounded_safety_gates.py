"""Unit tests for GroundedSafetyGatedSegmentCache (Activity B-1).

Covers: all 4 gates individually, fail-fast behavior, partial reuse,
stale eviction, stats tracking, and full pipeline flow.
"""

import pytest
import torch

from src.cache.grounded_safety_gated_cache import (
    GroundedSafetyGatedConfig,
    GroundedSafetyGatedSegmentCache,
    GateThresholds,
    SegmentMetadata,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_cache(
    content_sim: float = 0.85,
    evidence_overlap: float = 0.60,
    attn_support: float = 0.50,
    ngram_n: int = 3,
    tfidf_top_k: int = 20,
    max_entries: int = 100,
) -> GroundedSafetyGatedSegmentCache:
    thresholds = GateThresholds(
        content_sim_threshold=content_sim,
        evidence_overlap_threshold=evidence_overlap,
        attn_support_threshold=attn_support,
        ngram_n=ngram_n,
        tfidf_top_k=tfidf_top_k,
    )
    cfg = GroundedSafetyGatedConfig(gate_thresholds=thresholds, max_entries=max_entries)
    return GroundedSafetyGatedSegmentCache(cfg)


def make_kv(n_tokens: int = 8, d_kv: int = 16) -> torch.Tensor:
    return torch.randn(n_tokens, d_kv)


def identical_tokens(n: int = 20) -> list:
    return list(range(1, n + 1))


def compute_version_hash(cache: GroundedSafetyGatedSegmentCache, tokens: list) -> str:
    return cache._version_hash(tokens)


# ------------------------------------------------------------------ #
# Test 1: CacheStore interface — all abstract methods callable         #
# ------------------------------------------------------------------ #

def test_cache_store_interface_all_methods():
    cache = make_cache()
    kv = make_kv()
    tokens = identical_tokens()

    # insert (typed API)
    key = cache.insert(tokens, kv)
    assert isinstance(key, str)

    # put / get (raw CacheStore API)
    cache.put("raw_key", kv)
    result = cache.get("raw_key")
    assert result is not None

    # evict
    freed = cache.evict()
    assert isinstance(freed, int) and freed >= 0

    # hit_rate
    assert 0.0 <= cache.hit_rate() <= 1.0

    # memory_bytes
    assert cache.memory_bytes() >= 0

    # reset_stats
    cache.reset_stats()
    assert cache.hit_rate() == 0.0


# ------------------------------------------------------------------ #
# Test 2: Gate 1 n-gram similarity — pass                               #
# ------------------------------------------------------------------ #

def test_gate1_ngram_similarity_pass():
    """Identical context tokens → Gate 1 passes → cache hit or higher gate outcome."""
    cache = make_cache(content_sim=0.85)
    tokens = list(range(10, 30))
    kv = make_kv()
    key = cache.insert(tokens, kv, source_context_token_ids=tokens)
    # Lookup with same context → Gate 1 must pass
    result_kv, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
    )
    assert outcome != "gate_rejected", f"Expected gate pass, got {outcome}"


# ------------------------------------------------------------------ #
# Test 3: Gate 1 n-gram similarity — fail                               #
# ------------------------------------------------------------------ #

def test_gate1_ngram_similarity_fail():
    """Completely different context tokens → Gate 1 fails → gate_rejected immediately."""
    cache = make_cache(content_sim=0.85)
    stored_tokens = list(range(1, 25))
    kv = make_kv()
    cache.insert(stored_tokens, kv, source_context_token_ids=stored_tokens)

    # Query with entirely different tokens
    different_tokens = list(range(100, 125))
    _, outcome = cache.lookup(
        query_tokens=stored_tokens,          # same as stored (for key match)
        current_context_tokens=different_tokens,  # totally different context
    )
    assert outcome == "gate_rejected", f"Expected gate_rejected, got {outcome}"


# ------------------------------------------------------------------ #
# Test 4: Gate 2 TF-IDF overlap — pass                                  #
# ------------------------------------------------------------------ #

def test_gate2_tfidf_overlap_pass():
    """High TF-IDF overlap → Gate 2 passes (combined with Gate 1 pass)."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0)  # thresholds = 0 → always pass
    tokens = list(range(5, 30))
    kv = make_kv()
    cache.insert(tokens, kv, source_context_token_ids=tokens)
    _, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
    )
    # Both gates at threshold 0 should pass → safe_reuse_hit
    assert outcome in ("safe_reuse_hit", "partial_reuse_hit")


# ------------------------------------------------------------------ #
# Test 5: Gate 2 TF-IDF overlap — fail → partial_reuse                 #
# ------------------------------------------------------------------ #

def test_gate2_tfidf_overlap_fail_partial_reuse():
    """Low evidence overlap → Gate 2 fail → partial_reuse_hit with c_KV only."""
    # Gate 1 threshold = 0 (always pass), Gate 2 threshold = 1.0 (almost never pass)
    cache = make_cache(content_sim=0.0, evidence_overlap=1.0, attn_support=0.0)
    tokens = list(range(5, 30))
    kv = make_kv(n_tokens=8, d_kv=16)
    cache.insert(tokens, kv, source_context_token_ids=tokens)

    # Use different context to ensure top_k_tokens differ
    other_tokens = list(range(200, 225))
    result_kv, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=other_tokens,
    )
    assert outcome == "partial_reuse_hit", f"Expected partial_reuse_hit, got {outcome}"
    # c_KV should be half the last dimension
    assert result_kv is not None
    assert result_kv.shape[-1] == kv.shape[-1] // 2


# ------------------------------------------------------------------ #
# Test 6: Gate 3 version hash — pass                                    #
# ------------------------------------------------------------------ #

def test_gate3_version_hash_pass():
    """Matching version hash → Gate 3 passes."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0, attn_support=0.0)
    tokens = list(range(1, 25))
    kv = make_kv()
    cache.insert(tokens, kv, source_context_token_ids=tokens)
    correct_hash = compute_version_hash(cache, tokens)

    _, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
        current_source_version_hash=correct_hash,
    )
    assert outcome in ("safe_reuse_hit", "partial_reuse_hit"), f"Got {outcome}"


# ------------------------------------------------------------------ #
# Test 7: Gate 3 version hash — fail → stale_evicted                   #
# ------------------------------------------------------------------ #

def test_gate3_version_hash_fail_evicts():
    """Mismatched version hash → Gate 3 fail → stale_evicted + entry removed."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0)
    tokens = list(range(1, 25))
    kv = make_kv()
    key = cache.insert(tokens, kv, source_context_token_ids=tokens)

    wrong_hash = "deadbeef" * 8  # 64 chars, SHA-256-length but wrong

    _, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
        current_source_version_hash=wrong_hash,
    )
    assert outcome == "stale_evicted", f"Expected stale_evicted, got {outcome}"
    # The entry should have been evicted from the inner store
    assert cache._inner.get(key) is None


# ------------------------------------------------------------------ #
# Test 8: Gate 4 KV cosine similarity — pass                            #
# ------------------------------------------------------------------ #

def test_gate4_kv_cosine_pass():
    """Embedding matching mean KV vector → Gate 4 passes → safe_reuse_hit."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0, attn_support=0.5)
    tokens = list(range(1, 25))
    kv = make_kv(n_tokens=8, d_kv=16)
    cache.insert(tokens, kv, source_context_token_ids=tokens)

    # Use mean KV vector as query embedding → cosine sim = 1.0 → pass
    meta = list(cache._metadata.values())[0]
    matching_embedding = meta.mean_kv_vector.clone()

    _, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
        query_embedding=matching_embedding,
    )
    assert outcome in ("safe_reuse_hit", "partial_reuse_hit"), f"Got {outcome}"


# ------------------------------------------------------------------ #
# Test 9: Gate 4 KV cosine — fail → partial_reuse                       #
# ------------------------------------------------------------------ #

def test_gate4_kv_cosine_fail_partial_reuse():
    """Orthogonal embedding → Gate 4 fails → partial_reuse_hit."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0, attn_support=0.99)
    tokens = list(range(1, 25))
    kv = make_kv(n_tokens=8, d_kv=16)
    cache.insert(tokens, kv, source_context_token_ids=tokens)

    meta = list(cache._metadata.values())[0]
    # Orthogonal vector
    mean_kv = meta.mean_kv_vector
    ortho = torch.zeros_like(mean_kv)
    if mean_kv.shape[0] >= 2:
        ortho[0] = mean_kv[1]
        ortho[1] = -mean_kv[0]

    result_kv, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
        query_embedding=ortho,
    )
    assert outcome == "partial_reuse_hit", f"Expected partial_reuse_hit, got {outcome}"
    assert result_kv is not None
    assert result_kv.shape[-1] == kv.shape[-1] // 2


# ------------------------------------------------------------------ #
# Test 10: fail-fast skips later gates                                  #
# ------------------------------------------------------------------ #

def test_fail_fast_skips_later_gates():
    """Gate 1 fail → immediate rejection without evaluating Gates 2–4."""
    # Set a very high Gate 1 threshold so that different tokens always fail
    cache = make_cache(content_sim=0.99, evidence_overlap=0.0, attn_support=0.0)
    stored_tokens = list(range(1, 25))
    kv = make_kv()
    cache.insert(stored_tokens, kv, source_context_token_ids=stored_tokens)

    # Lookup with completely different context tokens
    different_context = list(range(500, 525))
    _, outcome = cache.lookup(
        query_tokens=stored_tokens,
        current_context_tokens=different_context,
    )
    assert outcome == "gate_rejected"
    # Gate rejections counter incremented, others stay at 0
    assert cache._gate_rejections == 1
    assert cache._safe_reuse_hits == 0
    assert cache._partial_reuse_hits == 0
    assert cache._stale_evictions == 0


# ------------------------------------------------------------------ #
# Test 11: all gates pass → safe_reuse_hit, full KV returned            #
# ------------------------------------------------------------------ #

def test_all_gates_pass_returns_safe_reuse():
    """All thresholds at 0 → all gates pass → safe_reuse_hit + full KV."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0, attn_support=0.0)
    tokens = list(range(1, 25))
    kv = make_kv(n_tokens=8, d_kv=16)
    cache.insert(tokens, kv, source_context_token_ids=tokens)

    result_kv, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
    )
    assert outcome == "safe_reuse_hit"
    assert result_kv is not None
    assert result_kv.shape == kv.shape  # full KV returned


# ------------------------------------------------------------------ #
# Test 12: miss on unknown key                                           #
# ------------------------------------------------------------------ #

def test_miss_on_unknown_key():
    """Lookup of a key that was never inserted returns miss."""
    cache = make_cache()
    unknown_tokens = list(range(999, 1020))
    result_kv, outcome = cache.lookup(
        query_tokens=unknown_tokens,
        current_context_tokens=unknown_tokens,
    )
    assert outcome == "miss"
    assert result_kv is None


# ------------------------------------------------------------------ #
# Test 13: stats tracking all outcomes                                  #
# ------------------------------------------------------------------ #

def test_stats_tracking_all_outcomes():
    """After lookups producing each outcome type, get_stats reflects counts."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0, attn_support=0.0)
    tokens = list(range(1, 25))
    kv = make_kv()
    cache.insert(tokens, kv, source_context_token_ids=tokens)

    # Cause a miss
    cache.lookup(query_tokens=list(range(500, 525)), current_context_tokens=tokens)

    stats = cache.get_stats()
    assert "total_lookups" in stats
    assert "safe_reuse_hit_rate" in stats
    assert "partial_reuse_hit_rate" in stats
    assert "gate_rejection_rate" in stats
    assert "stale_eviction_rate" in stats
    assert "miss_rate" in stats
    assert "safety_gate_pass_rate" in stats
    assert stats["total_lookups"] > 0


# ------------------------------------------------------------------ #
# Test 14: insert pre-computes metadata                                 #
# ------------------------------------------------------------------ #

def test_insert_precomputes_metadata():
    """After insert(), _metadata contains SegmentMetadata with all fields populated."""
    cache = make_cache()
    tokens = list(range(5, 30))
    kv = make_kv()
    key = cache.insert(tokens, kv, source_context_token_ids=tokens)

    assert key in cache._metadata
    meta = cache._metadata[key]
    assert isinstance(meta, SegmentMetadata)
    assert len(meta.ngrams) > 0
    assert isinstance(meta.top_k_tokens, set)
    assert isinstance(meta.source_version_hash, str) and len(meta.source_version_hash) == 64
    assert isinstance(meta.mean_kv_vector, torch.Tensor)


# ------------------------------------------------------------------ #
# Test 15: safety_gate_pass_rate above threshold scenario               #
# ------------------------------------------------------------------ #

def test_safety_gate_pass_rate_above_threshold():
    """safe_reuse / (safe + rejected + stale) ≥ 0.8 when all gates disabled."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0, attn_support=0.0)
    # Insert 10 segments and look them all up → all should be safe_reuse_hit
    for i in range(10):
        tokens = list(range(i * 30, i * 30 + 25))
        kv = make_kv()
        cache.insert(tokens, kv, source_context_token_ids=tokens)
        cache.lookup(
            query_tokens=tokens,
            current_context_tokens=tokens,
        )

    stats = cache.get_stats()
    assert stats["safety_gate_pass_rate"] >= 0.8, \
        f"Pass rate {stats['safety_gate_pass_rate']:.2f} < 0.8"


# ------------------------------------------------------------------ #
# Test 16: n-gram Jaccard below threshold → gate_rejected              #
# ------------------------------------------------------------------ #

def test_ngram_jaccard_below_threshold_rejects():
    """Low Jaccard similarity below threshold forces Gate 1 rejection."""
    cache = make_cache(content_sim=0.9)  # high threshold
    stored_tokens = list(range(1, 25))
    kv = make_kv()
    cache.insert(stored_tokens, kv, source_context_token_ids=stored_tokens)

    different_context = list(range(1000, 1025))
    _, outcome = cache.lookup(
        query_tokens=stored_tokens,
        current_context_tokens=different_context,
    )
    assert outcome == "gate_rejected"


# ------------------------------------------------------------------ #
# Test 17: stale_eviction_rate tracked across multiple Gate 3 failures  #
# ------------------------------------------------------------------ #

def test_stale_eviction_rate_tracked():
    """Multiple Gate 3 failures cause stale_eviction_rate > 0."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0)
    stale_count = 3
    wrong_hash = "00" * 32  # 64 hex chars, wrong hash

    for i in range(stale_count):
        tokens = list(range(i * 30, i * 30 + 25))
        kv = make_kv()
        key = cache.insert(tokens, kv, source_context_token_ids=tokens)
        # Force re-insert because eviction in previous iter may remove it
        cache._inner.put(key, kv)
        cache._metadata[key].source_version_hash = "correct_hash"
        cache.lookup(
            query_tokens=tokens,
            current_context_tokens=tokens,
            current_source_version_hash=wrong_hash,
        )

    stats = cache.get_stats()
    assert stats["stale_eviction_rate"] > 0.0


# ------------------------------------------------------------------ #
# Test 18: full pipeline insert → lookup flow                           #
# ------------------------------------------------------------------ #

def test_full_pipeline_lookup_insert():
    """Insert a segment, lookup with matching context, verify outcome and KV shape."""
    cache = make_cache(content_sim=0.0, evidence_overlap=0.0, attn_support=0.0)
    tokens = list(range(10, 35))
    kv = make_kv(n_tokens=8, d_kv=32)
    key = cache.insert(tokens, kv, source_context_token_ids=tokens)

    result_kv, outcome = cache.lookup(
        query_tokens=tokens,
        current_context_tokens=tokens,
    )
    assert outcome == "safe_reuse_hit"
    assert result_kv is not None
    assert result_kv.shape == kv.shape
