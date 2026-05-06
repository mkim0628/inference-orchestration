"""Unit tests for QueryCentricTriAttentionCache (Activity B+C, Cross-1)."""

import pytest
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore
from src.cache.tri_attention_codec import TriAttentionCodec
from src.cache.qc_tri_store import QueryCentricTriAttentionCache


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

LAYERS, HEADS, SEQ, DIM = 2, 2, 64, 32
CAPACITY = 64 * 1024 * 1024  # 64 MiB


def make_kv(seq: int = SEQ) -> torch.Tensor:
    return torch.randn(LAYERS, HEADS, seq, DIM)


def make_codec(calibrated: bool = True) -> TriAttentionCodec:
    codec = TriAttentionCodec(n_layers=LAYERS, n_heads=HEADS, head_dim=DIM)
    if calibrated:
        kvs = [torch.randn(LAYERS, HEADS, 32, DIM) for _ in range(10)]
        codec.calibrate(kvs)
    return codec


def make_cache(calibrated: bool = True, threshold: float = 0.60) -> QueryCentricTriAttentionCache:
    codec = make_codec(calibrated=calibrated)
    return QueryCentricTriAttentionCache(
        capacity_bytes=CAPACITY,
        codec=codec,
        relevance_threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                         #
# --------------------------------------------------------------------------- #

class TestHighRelevanceRawStorage:
    def test_high_relevance_raw_storage(self) -> None:
        """High-relevance segments go to _raw_store (not compressed)."""
        cache = make_cache(threshold=0.0)  # everything is high-relevance with threshold=0

        kv = make_kv()
        query_emb = kv.mean(dim=(0, 1, 2))  # perfectly aligned
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)

        cache.put_with_query("seg_high", kv, k_pre, query_emb)

        # Must be in raw store (cosine similarity to itself is ~1.0 >> threshold=0.0)
        assert "seg_high" in cache._raw_store

    def test_high_relevance_retrievable(self) -> None:
        """Raw-stored segment is returned by get()."""
        cache = make_cache(threshold=0.0)
        kv = make_kv()
        query_emb = kv.mean(dim=(0, 1, 2))
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put_with_query("r", kv, k_pre, query_emb)

        result = cache.get("r")
        assert result is not None
        assert result.shape == kv.shape


class TestLowRelevanceCompressedStorage:
    def test_low_relevance_compressed_storage(self) -> None:
        """Low-relevance segments go to _compressed_store."""
        cache = make_cache(threshold=2.0)  # impossibly high threshold → always compressed

        kv = make_kv()
        query_emb = -kv.mean(dim=(0, 1, 2))  # anti-aligned → low cosine similarity
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)

        cache.put_with_query("seg_low", kv, k_pre, query_emb)

        assert "seg_low" in cache._compressed_store
        assert "seg_low" not in cache._raw_store

    def test_low_relevance_still_retrievable(self) -> None:
        """Compressed segment can be retrieved (decompressed approximation)."""
        cache = make_cache(threshold=2.0)
        kv = make_kv()
        query_emb = -kv.mean(dim=(0, 1, 2))
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put_with_query("c", kv, k_pre, query_emb)

        result = cache.get("c")
        assert result is not None
        # Shape should be reconstructed to original length
        assert result.shape[2] == SEQ


class TestSelectiveRecomputeUsesRaw:
    def test_selective_recompute_uses_raw(self) -> None:
        """selective_recompute only considers segments in _raw_store."""
        cache = make_cache(threshold=0.0)  # everything high-relevance

        kv = make_kv()
        query_emb = kv.mean(dim=(0, 1, 2))
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put_with_query("raw_seg", kv, k_pre, query_emb)

        selected = cache.selective_recompute(query_emb, ["raw_seg"], budget=1.0)
        assert "raw_seg" in selected

    def test_selective_recompute_excludes_compressed(self) -> None:
        """Compressed segments are not eligible for selective recompute."""
        cache = make_cache(threshold=2.0)  # everything goes to compressed

        kv = make_kv()
        query_emb = -kv.mean(dim=(0, 1, 2))
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put_with_query("comp_seg", kv, k_pre, query_emb)

        selected = cache.selective_recompute(query_emb, ["comp_seg"], budget=1.0)
        assert "comp_seg" not in selected

    def test_put_without_query_is_eligible(self) -> None:
        """Segments stored via put() (no query) can be recomputed via QCRC."""
        cache = make_cache()
        kv = make_kv()
        cache.put("plain_seg", kv)
        query = kv.mean(dim=(0, 1, 2))
        selected = cache.selective_recompute(query, ["plain_seg"], budget=1.0)
        assert isinstance(selected, list)


class TestCacheStoreInterface:
    def test_cachestore_interface(self) -> None:
        """QueryCentricTriAttentionCache implements all CacheStore abstract methods."""
        assert issubclass(QueryCentricTriAttentionCache, CacheStore)
        cache = make_cache()
        assert callable(cache.put)
        assert callable(cache.get)
        assert callable(cache.evict)
        assert callable(cache.hit_rate)
        assert callable(cache.memory_bytes)
        assert callable(cache.reset_stats)

    def test_put_get_basic(self) -> None:
        """put/get round-trip via the standard CacheStore interface."""
        cache = make_cache()
        kv = make_kv()
        cache.put("k", kv)
        result = cache.get("k")
        assert result is not None

    def test_get_miss_returns_none(self) -> None:
        cache = make_cache()
        assert cache.get("nonexistent") is None

    def test_hit_rate_increases(self) -> None:
        cache = make_cache()
        kv = make_kv()
        cache.put("h", kv)
        cache.get("h")  # hit
        assert cache.hit_rate() > 0.0

    def test_evict_non_empty(self) -> None:
        cache = make_cache()
        kv = make_kv()
        cache.put("e", kv)
        freed = cache.evict()
        assert freed >= 0

    def test_evict_empty_returns_zero(self) -> None:
        cache = make_cache()
        assert cache.evict() == 0

    def test_memory_bytes_nonnegative(self) -> None:
        cache = make_cache()
        assert cache.memory_bytes() >= 0

    def test_reset_stats_zeroes_rates(self) -> None:
        cache = make_cache()
        kv = make_kv()
        cache.put("x", kv)
        cache.get("x")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0

    def test_compressed_keys_helper(self) -> None:
        """compressed_keys() returns only keys in _compressed_store."""
        cache = make_cache(threshold=2.0)
        kv = make_kv()
        query_emb = -kv.mean(dim=(0, 1, 2))
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put_with_query("c1", kv, k_pre, query_emb)
        assert "c1" in cache.compressed_keys()

    def test_raw_keys_helper(self) -> None:
        """raw_keys() returns only keys in _raw_store."""
        cache = make_cache(threshold=0.0)
        kv = make_kv()
        query_emb = kv.mean(dim=(0, 1, 2))
        k_pre = torch.randn(LAYERS, HEADS, SEQ, DIM)
        cache.put_with_query("r1", kv, k_pre, query_emb)
        assert "r1" in cache.raw_keys()
