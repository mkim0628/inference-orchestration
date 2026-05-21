"""Cross Activity B+C: BlockUnion 비연속 재사용 × KV 선택 압축 통합 파이프라인.

B-1 BlockUnionNonContiguousReuseIndex + C-1 CompactAttentionBlockUnionCodec.
공통 KVSelectionBlockTable 자료구조로 두 Activity를 통합.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.cache.base import CacheStore
from src.cache.block_union_noncontiguous_index import (
    BlockUnionConfig,
    BlockUnionNonContiguousReuseIndex,
    KVSelectionBlockTable,
)
from src.cache.compact_attention_block_union_codec import (
    BlockUnionCodecConfig,
    CompactAttentionBlockUnionCodec,
)


@dataclass
class BCPipelineConfig:
    b_config: Optional[BlockUnionConfig] = None
    c_config: Optional[BlockUnionCodecConfig] = None
    # B+C 결합 감소 효과 설정
    apply_selection_to_union: bool = True  # True: 비연속 히트 블록에 추가로 KV 선택 적용
    seed: int = 42


class BlockUnionBCPipeline(CacheStore):
    """Block-Union 비연속 재사용 × KV 선택 압축 통합 파이프라인.

    Cross Activity B+C:
      Step 1 (B): BlockUnionNonContiguousReuseIndex.get_block_union() →
                  비연속 세그먼트 히트의 GQA-aware 블록 테이블 구성.
      Step 2 (C): CompactAttentionBlockUnionCodec.build_kv_selection_block_table() →
                  히트 블록에서 중요 블록만 추가 선택.
      Step 3: 통합 KVSelectionBlockTable을 표준 PA 커널에 투입.

    B+C 결합 감소 효과:
      비연속 히트 블록 수 × KV 선택률 = 곱연산적 메모리 절감.
      예: 50% 비연속 히트 × 40% 선택률 → 전체 KV의 20% 사용.

    CacheStore 인터페이스 완전 구현.
    put/get은 c_codec(C-1)으로 위임 (압축 저장 기본 경로).
    """

    def __init__(self, config: BCPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config

        b_cfg = config.b_config or BlockUnionConfig(seed=config.seed)
        c_cfg = config.c_config or BlockUnionCodecConfig(seed=config.seed)

        self.b_index = BlockUnionNonContiguousReuseIndex(b_cfg)
        self.c_codec = CompactAttentionBlockUnionCodec(c_cfg)

    # ------------------------------------------------------------------ #
    # 통합 B+C 처리 API                                                    #
    # ------------------------------------------------------------------ #

    def process_noncontiguous_segments(
        self,
        segment_keys: List[str],
        attn_scores: Optional[torch.Tensor] = None,  # [n_heads, n_q_chunk, n_kv_total]
    ) -> Optional[KVSelectionBlockTable]:
        """B+C 4단 파이프라인 실행.

        Step 1: B-1 block-union으로 비연속 세그먼트 히트의 GQA-aware 블록 테이블 구성.
        Step 2: 어텐션 점수가 주어지면 C-1 중요도 마스크 업데이트.
        Step 3: C-1 KV 선택으로 히트 블록에서 추가 압축.
        Returns: 최종 KVSelectionBlockTable 또는 None (모든 세그먼트 미스).
        """
        # Step 1: B-1 block-union 테이블 구성
        union_table = self.b_index.get_block_union(segment_keys)
        if union_table is None:
            return None

        # Step 2: C-1 중요도 마스크 업데이트 (어텐션 점수 있을 때만)
        if attn_scores is not None:
            self.c_codec.update_chunk_attention(attn_scores)

        # Step 3: C-1 KV 선택 적용 (B 히트 블록 집합에서 추가 압축)
        if self.config.apply_selection_to_union and union_table.total_blocks > 0:
            n_hit_total = union_table.total_blocks * self.b_index.config.block_size
            existing_ptrs = list(union_table.group_blocks.get(0, []))
            sel_table = self.c_codec.build_kv_selection_block_table(
                n_kv_total=n_hit_total,
                existing_block_ptrs=existing_ptrs,
            )
            return sel_table
        return union_table

    def combined_reduction_ratio(self) -> float:
        """B+C 결합 메모리 감소율 추정.

        = 1 - (b_block_union_hit_rate × c_kv_selection_ratio)
        실제 측정은 memory_bytes() 비교로 수행.
        """
        nc_hit = self.b_index.block_union_hit_rate()
        sel_ratio = self.c_codec.config.kv_selection_ratio
        return 1.0 - (nc_hit * sel_ratio)

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 (c_codec에 위임)                               #
    # ------------------------------------------------------------------ #

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return self.c_codec.compression_hook(key, value)

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        return self.c_codec.get_importance_mask(key)

    def put(self, key: str, value: torch.Tensor) -> None:
        self.c_codec.put(key, value)
        self.b_index.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.c_codec.get(key)

    def evict(self) -> int:
        freed_c = self.c_codec.evict()
        freed_b = self.b_index.evict()
        return freed_c + freed_b

    def hit_rate(self) -> float:
        return self.c_codec.hit_rate()

    def memory_bytes(self) -> int:
        return self.c_codec.memory_bytes()

    def reset_stats(self) -> None:
        self.c_codec.reset_stats()
        self.b_index.reset_stats()

    def metrics_summary(self) -> Dict:
        """B+C 복합 효과 측정용 통합 메트릭."""
        return {
            "b_noncontiguous_hit_rate": self.b_index.noncontiguous_hit_rate(),
            "b_block_union_hit_rate": self.b_index.block_union_hit_rate(),
            "c_memory_reduction_ratio": self.c_codec.memory_reduction_ratio(),
            "c_kv_selection_ratio": self.c_codec.config.kv_selection_ratio,
            "bc_combined_reduction_estimate": self.combined_reduction_ratio(),
            "total_memory_bytes": self.memory_bytes(),
            "hit_rate": self.hit_rate(),
        }
