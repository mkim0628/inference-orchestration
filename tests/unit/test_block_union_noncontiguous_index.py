"""Unit tests for Activity B: BlockUnionNonContiguousReuseIndex.

block-union 변환 정확성, GQA per-group 테이블, 비연속 히트율, 하위 호환성 검증.
evaluation_criteria.md §3 Activity B 항목 커버.
"""

import pytest
import torch

from src.cache.block_union_noncontiguous_index import (
    BlockUnionConfig,
    BlockUnionNonContiguousReuseIndex,
    KVSelectionBlockTable,
)
from src.cache.segmented import SegmentedHashCache


# --------------------------------------------------------------------------- #
# KVSelectionBlockTable 테스트                                                  #
# --------------------------------------------------------------------------- #


def test_kv_selection_block_table_to_tensor_shape() -> None:
    """KVSelectionBlockTable.to_block_table_tensor() shape 검증."""
    n_groups = 4
    max_blocks = 5
    group_blocks = {g: list(range(max_blocks)) for g in range(n_groups)}
    table = KVSelectionBlockTable(
        group_blocks=group_blocks,
        n_groups=n_groups,
        total_blocks=n_groups * max_blocks,
        selected_blocks=n_groups * max_blocks,
    )
    tensor = table.to_block_table_tensor()
    assert tensor.shape == (n_groups, max_blocks)
    assert tensor.dtype == torch.int64


def test_kv_selection_block_table_selection_ratio() -> None:
    """selected_blocks=4, total_blocks=10 → selection_ratio == 0.4."""
    table = KVSelectionBlockTable(
        group_blocks={0: [0, 1, 2, 3]},
        n_groups=1,
        total_blocks=10,
        selected_blocks=4,
    )
    assert abs(table.selection_ratio - 0.4) < 1e-9


def test_kv_selection_block_table_to_tensor_empty_groups() -> None:
    """빈 group_blocks → 빈 텐서 또는 패딩 텐서 반환."""
    table = KVSelectionBlockTable(
        group_blocks={},
        n_groups=0,
        total_blocks=0,
        selected_blocks=0,
    )
    tensor = table.to_block_table_tensor()
    assert tensor.numel() == 0


def test_kv_selection_block_table_padding_minus_one() -> None:
    """불균등한 블록 수 → 짧은 그룹에 -1 패딩."""
    group_blocks = {0: [1, 2, 3], 1: [4, 5]}
    table = KVSelectionBlockTable(
        group_blocks=group_blocks,
        n_groups=2,
        total_blocks=5,
        selected_blocks=5,
    )
    tensor = table.to_block_table_tensor()
    assert tensor.shape == (2, 3)
    assert tensor[1, 2].item() == -1


# --------------------------------------------------------------------------- #
# BlockUnionNonContiguousReuseIndex 테스트                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def index() -> BlockUnionNonContiguousReuseIndex:
    cfg = BlockUnionConfig(
        block_size=16,
        n_kv_heads=8,
        n_gqa_groups=4,
        max_entries=100,
        seed=42,
    )
    return BlockUnionNonContiguousReuseIndex(cfg)


def test_block_union_index_put_creates_block_ptrs(index: BlockUnionNonContiguousReuseIndex) -> None:
    """put(key, value) 후 _segment_blocks[key] 존재 확인."""
    kv = torch.randn(32, 64)
    index.put("seg_a", kv)
    assert "seg_a" in index._segment_blocks
    assert len(index._segment_blocks["seg_a"]) >= 1


def test_block_union_index_get_returns_stored_value(index: BlockUnionNonContiguousReuseIndex) -> None:
    """put → get 왕복 확인."""
    kv = torch.randn(32, 64)
    index.put("seg_a", kv)
    retrieved = index.get("seg_a")
    assert retrieved is not None
    assert retrieved.shape == kv.shape
    assert torch.allclose(retrieved, kv)


def test_block_union_index_get_block_union_single_segment(index: BlockUnionNonContiguousReuseIndex) -> None:
    """단일 세그먼트 히트 → KVSelectionBlockTable 반환."""
    kv = torch.randn(32, 64)
    index.put("seg_a", kv)
    table = index.get_block_union(["seg_a"])
    assert table is not None
    assert isinstance(table, KVSelectionBlockTable)
    assert table.total_blocks > 0


def test_block_union_index_get_block_union_multiple_segments(index: BlockUnionNonContiguousReuseIndex) -> None:
    """3개 세그먼트 히트 → group_blocks에 union 블록 포인터 포함."""
    for i in range(3):
        kv = torch.randn(32, 64)
        index.put(f"seg_{i}", kv)
    table = index.get_block_union(["seg_0", "seg_1", "seg_2"])
    assert table is not None
    # 모든 그룹에 블록 포인터가 있어야 함
    for g_id in range(index.config.n_gqa_groups):
        assert g_id in table.group_blocks
        assert len(table.group_blocks[g_id]) > 0


def test_block_union_index_get_block_union_all_miss(index: BlockUnionNonContiguousReuseIndex) -> None:
    """모든 세그먼트 미스 → None 반환."""
    result = index.get_block_union(["no_such_key_1", "no_such_key_2"])
    assert result is None


def test_block_union_index_gqa_group_count(index: BlockUnionNonContiguousReuseIndex) -> None:
    """n_gqa_groups=4 → group_blocks에 4개 그룹 키 존재."""
    kv = torch.randn(32, 64)
    index.put("seg_a", kv)
    table = index.get_block_union(["seg_a"])
    assert table is not None
    assert table.n_groups == 4
    assert len(table.group_blocks) == 4


def test_block_union_index_noncontiguous_hit_detection(index: BlockUnionNonContiguousReuseIndex) -> None:
    """[hit, miss, hit] 패턴 → noncontiguous_hits 증가 확인."""
    kv = torch.randn(32, 64)
    index.put("seg_0", kv)
    index.put("seg_2", kv)
    # seg_1은 미스 — seg_2 히트 앞에 미스가 있으므로 비연속 히트
    table = index.get_block_union(["seg_0", "seg_1", "seg_2"])
    assert table is not None
    assert index._noncontiguous_hits > 0


def test_block_union_index_noncontiguous_hit_rate(index: BlockUnionNonContiguousReuseIndex) -> None:
    """비연속 히트율 = noncontiguous_hits / total_hits (>= 0)."""
    kv = torch.randn(32, 64)
    index.put("seg_0", kv)
    index.put("seg_2", kv)
    index.get_block_union(["seg_0", "seg_1", "seg_2"])
    rate = index.noncontiguous_hit_rate()
    assert 0.0 <= rate <= 1.0


def test_block_union_index_build_block_union_table_empty(index: BlockUnionNonContiguousReuseIndex) -> None:
    """모든 키 미스 → total_blocks=0 테이블 반환."""
    table = index.build_block_union_table(["missing_1", "missing_2"])
    assert table.total_blocks == 0
    assert table.selected_blocks == 0


def test_block_union_index_evict_lru() -> None:
    """max_entries=2, 3개 put → 첫 번째 항목 퇴거 확인."""
    cfg = BlockUnionConfig(block_size=16, n_kv_heads=8, n_gqa_groups=4, max_entries=2, seed=42)
    idx = BlockUnionNonContiguousReuseIndex(cfg)
    kv = torch.randn(16, 64)
    idx.put("a", kv)
    idx.put("b", kv)
    idx.put("c", kv)  # "a"가 퇴거되어야 함
    assert idx.get("a") is None
    assert idx.get("b") is not None or idx.get("c") is not None


def test_block_union_index_hit_rate_tracking(index: BlockUnionNonContiguousReuseIndex) -> None:
    """put 2개 후 get 1회 히트 + 1회 미스 → hit_rate() == 0.5."""
    kv = torch.randn(32, 64)
    index.put("seg_a", kv)
    index.put("seg_b", kv)
    index.get("seg_a")    # hit
    index.get("missing")  # miss
    assert abs(index.hit_rate() - 0.5) < 1e-9


def test_block_union_index_cachestore_interface_full(index: BlockUnionNonContiguousReuseIndex) -> None:
    """put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작."""
    kv = torch.randn(32, 64)
    index.put("seg_a", kv)
    assert index.get("seg_a") is not None
    assert index.memory_bytes() > 0
    assert index.hit_rate() > 0.0
    freed = index.evict()
    assert freed > 0
    index.reset_stats()
    assert index._hits == 0
    assert index._misses == 0


def test_block_union_index_seed_reproducibility() -> None:
    """동일 put 순서 + 동일 seed → 동일 block_ptrs."""
    def _build(seed: int) -> BlockUnionNonContiguousReuseIndex:
        cfg = BlockUnionConfig(block_size=16, n_kv_heads=8, n_gqa_groups=4, max_entries=100, seed=seed)
        idx = BlockUnionNonContiguousReuseIndex(cfg)
        torch.manual_seed(seed)
        kv = torch.randn(32, 64)
        idx.put("seg_a", kv)
        return idx

    idx1 = _build(42)
    idx2 = _build(42)
    assert idx1._segment_blocks["seg_a"] == idx2._segment_blocks["seg_a"]


def test_block_union_index_block_table_tensor_injectable(index: BlockUnionNonContiguousReuseIndex) -> None:
    """to_block_table_tensor()가 유효한 int64 텐서 반환."""
    kv = torch.randn(32, 64)
    index.put("seg_a", kv)
    table = index.get_block_union(["seg_a"])
    assert table is not None
    tensor = table.to_block_table_tensor()
    assert tensor.dtype == torch.int64
    assert tensor.dim() == 2
    assert tensor.shape[0] == index.config.n_gqa_groups


# --------------------------------------------------------------------------- #
# SegmentedHashCache use_block_union 플래그 테스트                              #
# --------------------------------------------------------------------------- #


def test_segmented_cache_use_block_union_flag() -> None:
    """SegmentedHashCache.get_segments(use_block_union=True) → block-union 인덱스 경로 호출."""
    cache = SegmentedHashCache(chunk_size=16, max_entries=100)
    cfg = BlockUnionConfig(block_size=16, n_kv_heads=8, n_gqa_groups=4, max_entries=100, seed=42)
    bu_index = BlockUnionNonContiguousReuseIndex(cfg)
    cache.set_block_union_index(bu_index)

    # 데이터 준비: 토큰 32개 (청크 2개)
    token_ids = list(range(32))
    kv = torch.randn(16, 64)
    # 첫 번째 청크를 캐시에 넣음
    cache.put_segment(token_ids, 0, kv, layer_idx=0)

    hits, misses = cache.get_segments(token_ids, layer_idx=0, use_block_union=True)
    assert len(hits) >= 1  # 첫 번째 청크 히트


def test_segmented_cache_use_block_union_false_unchanged() -> None:
    """SegmentedHashCache.get_segments(use_block_union=False) → 기존 memcpy 경로 동작 유지 (하위 호환)."""
    cache = SegmentedHashCache(chunk_size=16, max_entries=100)
    token_ids = list(range(32))
    kv = torch.randn(16, 64)
    cache.put_segment(token_ids, 0, kv, layer_idx=0)

    # use_block_union=False (기본값): 기존 동작과 동일
    hits, misses = cache.get_segments(token_ids, layer_idx=0, use_block_union=False)
    hits_default, misses_default = cache.get_segments(token_ids, layer_idx=0)
    # 두 호출 결과 일치 (히트/미스 수 동일)
    assert len(hits) == len(hits_default)
    assert misses == misses_default


def test_segmented_cache_block_union_index_not_called_without_flag() -> None:
    """use_block_union=False 시 block-union 인덱스의 통계가 변하지 않음."""
    cache = SegmentedHashCache(chunk_size=16, max_entries=100)
    cfg = BlockUnionConfig(block_size=16, n_kv_heads=8, n_gqa_groups=4, max_entries=100, seed=42)
    bu_index = BlockUnionNonContiguousReuseIndex(cfg)
    cache.set_block_union_index(bu_index)

    token_ids = list(range(32))
    kv = torch.randn(16, 64)
    cache.put_segment(token_ids, 0, kv, layer_idx=0)

    # use_block_union=False → bu_index 호출 없음
    cache.get_segments(token_ids, layer_idx=0, use_block_union=False)
    assert bu_index._hits == 0
    assert bu_index._misses == 0
