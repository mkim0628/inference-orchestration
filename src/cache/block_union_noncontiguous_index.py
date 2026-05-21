"""Activity B: BlockUnion Non-Contiguous KV Cache Reuse Index.

CompactAttention (2605.16839) Block-Union 기반 비연속 KV 재사용.
memcpy 없이 GQA-aware per-group 블록 테이블로 변환해 표준 PA 커널에 직접 투입.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch

from src.cache.base import CacheStore

# BlockPtr: PA 블록 ID (int)
BlockPtr = int


@dataclass
class KVSelectionBlockTable:
    """GQA-aware per-group KV 선택 블록 테이블.

    group_blocks[group_id][position] = block_ptr 형태로
    비연속 PA 블록들을 GQA 그룹 단위로 통합 표현.

    표준 PagedAttention 커널의 block_tables 인자로 직접 투입 가능.
    배치 내에서만 유효 (배치 간 재사용 없음).
    """

    group_blocks: Dict[int, List[BlockPtr]] = field(default_factory=dict)
    # group_blocks[group_id] = 순서대로 나열된 블록 포인터 리스트
    n_groups: int = 0
    total_blocks: int = 0
    selected_blocks: int = 0  # 선택된 블록 수 (C-1에서 설정)

    @property
    def selection_ratio(self) -> float:
        if self.total_blocks == 0:
            return 1.0
        return self.selected_blocks / self.total_blocks

    def to_block_table_tensor(self) -> torch.Tensor:
        """표준 PagedAttention 커널용 block_table 텐서 변환.

        Returns: [n_groups, max_blocks_per_group] int64 텐서.
        -1은 패딩(빈 슬롯)을 나타낸다.
        """
        if not self.group_blocks:
            return torch.empty(0, 0, dtype=torch.int64)
        max_len = max(len(v) for v in self.group_blocks.values())
        if max_len == 0:
            return torch.full(
                (len(self.group_blocks), 1), -1, dtype=torch.int64
            )
        n_groups = len(self.group_blocks)
        table = torch.full((n_groups, max_len), -1, dtype=torch.int64)
        for g_idx, (g_id, blk_list) in enumerate(sorted(self.group_blocks.items())):
            if blk_list:
                table[g_idx, : len(blk_list)] = torch.tensor(
                    blk_list, dtype=torch.int64
                )
        return table


@dataclass
class BlockUnionConfig:
    block_size: int = 16          # 블록당 토큰 수 (PagedAttention block_size)
    n_kv_heads: int = 8           # 총 KV 헤드 수
    n_gqa_groups: int = 4         # GQA 그룹 수 (n_kv_heads / n_q_heads_per_group)
    max_entries: int = 1000       # 캐시 최대 엔트리 수
    rope_reencoding_enabled: bool = True  # 위치 불일치 시 AdapShot RoPE 재인코딩
    use_block_union: bool = True   # block-union 경로 활성화 (False면 memcpy fallback)
    seed: int = 42


class BlockUnionNonContiguousReuseIndex(CacheStore):
    """CompactAttention Block-Union 기반 비연속 KV 재사용 인덱스.

    Activity B: Non-Contiguous KV Cache Reuse
    CacheStore 인터페이스 완전 구현.

    핵심 알고리즘:
      비연속 PA 블록들을 GQA-aware per-group 블록 테이블(KVSelectionBlockTable)로 변환.
      변환 비용: O(k × seg_len / block_size) 포인터 연산 (memcpy 없음).
      표준 PagedAttention 커널의 block_tables 인자로 직접 주입.

    GQA-aware per-group 블록 테이블 변환 원리:
      GQA 그룹 내 모든 헤드가 동일 K/V 블록을 공유하므로,
      그룹 단위로 블록 포인터를 통합 (중복 저장 제거).
      group_id = head_id // (n_kv_heads // n_gqa_groups)
    """

    def __init__(self, config: BlockUnionConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        # 세그먼트 ID → PA 블록 포인터 리스트 매핑
        self._segment_blocks: OrderedDict[str, List[BlockPtr]] = OrderedDict()
        # 세그먼트 ID → 원본 KV 텐서 (fallback용)
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._block_union_hits = 0   # block-union 경로 히트 수
        self._memcpy_hits = 0        # memcpy fallback 경로 히트 수

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """KV 텐서를 세그먼트 키로 저장.

        PA 블록 포인터 목록을 단조 증가 ID로 생성해 _segment_blocks에 저장.
        실제 환경에서는 PA 블록 할당기에서 받아옴 — 여기서는 시뮬레이션.
        """
        if key in self._store:
            self._store.move_to_end(key)
            self._segment_blocks.move_to_end(key)
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        n_seq = value.shape[0] if value.dim() >= 1 else 1
        n_blocks = max(1, (n_seq + self.config.block_size - 1) // self.config.block_size)
        # 단조 증가 블록 ID 부여 (시뮬레이션)
        base_id = len(self._segment_blocks) * n_blocks
        block_ptrs = list(range(base_id, base_id + n_blocks))
        self._segment_blocks[key] = block_ptrs
        self._store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """일반 get: fallback memcpy 경로."""
        if key in self._store:
            self._store.move_to_end(key)
            self._segment_blocks.move_to_end(key)
            self._hits += 1
            self._memcpy_hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._store:
            return 0
        key, v = self._store.popitem(last=False)
        self._segment_blocks.pop(key, None)
        return v.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._block_union_hits = 0
        self._memcpy_hits = 0

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """bool 마스크 반환 — 기본 구현은 저장된 모든 위치를 True로 반환."""
        kv = self._store.get(key)
        if kv is None:
            return None
        n_kv = kv.shape[0] if kv.dim() >= 1 else 1
        return torch.ones(n_kv, dtype=torch.bool)

    # ------------------------------------------------------------------ #
    # Block-Union API                                                      #
    # ------------------------------------------------------------------ #

    def get_block_union(
        self,
        segment_keys: List[str],
    ) -> Optional["KVSelectionBlockTable"]:
        """비연속 세그먼트들의 GQA-aware block-union 테이블 생성.

        히트된 세그먼트의 블록 포인터를 GQA 그룹 단위로 통합 (순서 보존 union).
        모든 세그먼트가 미스이면 None 반환.
        """
        hit_keys = [k for k in segment_keys if k in self._segment_blocks]
        if not hit_keys:
            self._misses += len(segment_keys)
            return None

        # 히트/미스 계수
        self._hits += len(hit_keys)
        self._misses += len(segment_keys) - len(hit_keys)
        self._block_union_hits += len(hit_keys)

        # 비연속 히트 감지: 앞에 미스가 존재하는 히트 위치
        all_keys_list = list(segment_keys)
        miss_set: Set[int] = set(
            i for i, k in enumerate(all_keys_list) if k not in self._segment_blocks
        )
        for i, k in enumerate(all_keys_list):
            if k in self._segment_blocks and any(m < i for m in miss_set):
                self._noncontiguous_hits += 1
                break

        # GQA-aware per-group 블록 테이블 구성
        n_groups = max(1, self.config.n_gqa_groups)
        group_blocks: Dict[int, List[BlockPtr]] = {g: [] for g in range(n_groups)}
        seen: Dict[int, Set[int]] = {g: set() for g in range(n_groups)}

        for seg_key in hit_keys:
            blk_ptrs = self._segment_blocks[seg_key]
            for blk_ptr in blk_ptrs:
                for g_id in range(n_groups):
                    if blk_ptr not in seen[g_id]:
                        group_blocks[g_id].append(blk_ptr)
                        seen[g_id].add(blk_ptr)

        total = sum(len(v) for v in group_blocks.values())
        return KVSelectionBlockTable(
            group_blocks=group_blocks,
            n_groups=n_groups,
            total_blocks=total,
            selected_blocks=total,
        )

    def build_block_union_table(
        self,
        segment_ids: List[str],
    ) -> "KVSelectionBlockTable":
        """segment_ids에 대한 GQA-aware 블록 테이블 구성 (히트 미스 무관).

        미스 세그먼트는 건너뛰고, 히트 세그먼트만으로 테이블 구성.
        반환 테이블이 비어 있으면 total_blocks=0.
        """
        result = self.get_block_union(segment_ids)
        if result is None:
            n_groups = max(1, self.config.n_gqa_groups)
            return KVSelectionBlockTable(
                group_blocks={g: [] for g in range(n_groups)},
                n_groups=n_groups,
                total_blocks=0,
                selected_blocks=0,
            )
        return result

    def noncontiguous_hit_rate(self) -> float:
        """비연속 히트 비율 (전체 히트 중)."""
        return self._noncontiguous_hits / max(1, self._hits)

    def block_union_hit_rate(self) -> float:
        """block-union 경로 히트 비율."""
        return self._block_union_hits / max(1, self._hits + self._misses)
