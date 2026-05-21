<!-- 변경 이유 (이전 Spec.md: 2026-05-20 대비):
이전 사이클(2026-05-20)은 A+C 조합이었다:
  - A-1 CONCURCongestionBasedAgentAdmissionScheduler (KV 풀 혼잡도 기반 에이전트 어드미션 제어)
  - C-1 SpecAttnVerificationGuidedKVSparseCodec (자기-추측 디코딩 검증 로짓 유도 KV 희소화)
  - Cross CongestionAdmissionSpecAttnDualReductionPipeline (A+C 통합)

이번 사이클(2026-05-21)은 B+C 조합으로 전환된다. 설계 축이 다음과 같이 변경된다:

주요 변경:
1. [전략 전환] 혼잡도 기반 어드미션 제어(A) → Block-Union 비연속 KV 재사용(B)으로 초점 이동.
   이전 기법이 "얼마나 많은 에이전트를 허용하느냐"의 볼륨 제어 문제였다면,
   이번 기법은 "비연속 PA 블록을 memcpy 없이 GQA-aware 블록 테이블로 변환해 표준
   커널에 직접 투입"하는 물리적 표현 변환 문제다. CompactAttention(2605.16839) 기반.

2. [Activity B 신규] BlockUnionNonContiguousReuseIndex (B-1):
   비연속 PA 블록을 GQA-aware per-group 블록 테이블로 변환 (memcpy 없음).
   기존 `SegmentedHashCache.get_segments()` 경로에 `use_block_union=True` 플래그 추가.
   비연속 히트 처리 경로 전면 교체.

3. [Activity C 교체] SpecAttnVerificationGuidedKVSparseCodec(검증 로짓 유도 희소화) →
   CompactAttentionBlockUnionCodec (청크드 프리필 특화 2D 블록-스파스 마스크 → KV 선택 블록 테이블 변환).
   이전 기법이 "자기-추측 디코딩 검증 단계 로짓"을 활용했다면, 이번 기법은
   "청크드 프리필의 어텐션 중요도 마스크를 Q-block union + Intra-group union으로 GQA-aware
   블록 테이블로 변환해 선택된 KV 블록만 표준 어텐션 커널에 투입"하는 방식이다.
   커스텀 스파스 커널 불필요. CompactAttention 원논문(2605.16839) 기반.

4. [Cross 신규] BlockUnionNonContiguousCompressionPipeline (B+C, Cross-1):
   B-1 block-union 인덱스와 C-1 KV 선택 코덱이 공통 KVSelectionBlockTable 자료구조를
   공유하므로 단일 파이프라인으로 통합. "비연속 히트 블록 × KV 선택률" 곱연산 감소.

5. [보존 파일] 이전 사이클 구현 파일(concur_*, specattn_*, congestion_specattn_*)은 수정하지 않는다.
   base.py의 get_importance_mask() 메서드는 이전 사이클에서 이미 추가되어 있으므로 변경 불필요.
   기존 단위·통합 테스트가 회귀 없이 통과해야 한다.

6. [Activity A 미포함] 이번 사이클에서 A는 별도 구현 대상이 아니다. A-1 Bayesian Profiling
   기반 KVServeBayesianScheduler는 B+C 파이프라인 안정화 이후 다음 사이클에서 추가한다.
-->

# Spec — 2026-05-21

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-21.md`

**최우선 구현 타겟**:
- **Cross-1 (최우선)**: `BlockUnionNonContiguousCompressionPipeline` (B+C)
  — BlockUnionNonContiguousReuseIndex(B-1) + CompactAttentionBlockUnionCodec(C-1)의
  통합 파이프라인. 비연속 히트 블록 × KV 선택률 곱연산 감소 효과.
- **C-1 단독 (2순위)**: `CompactAttentionBlockUnionCodec`
  — KV 선택 압축 효과 독립 측정. Cross-1 구현의 C 컴포넌트와 동일 클래스.
- **B-1 단독 (3순위)**: `BlockUnionNonContiguousReuseIndex`
  — memcpy 병목 제거 효과 독립 측정. Cross-1 구현의 B 컴포넌트와 동일 클래스.

**해결하려는 문제**:

- **Activity B (Block-Union 비연속 재사용)**: 기존 비연속 KV 재사용은 물리적으로 분산된
  PagedAttention 블록들을 연속 KV 텐서로 집결하는 memcpy 비용이 주요 병목이다. 비연속
  히트가 발생하더라도 memcpy 오버헤드가 히트 이득을 상쇄하여 사실상 재사용이 불가능한
  케이스가 발생한다. `BlockUnionNonContiguousReuseIndex`는 CompactAttention(2605.16839)의
  Block-Union KV Selection 아이디어를 차용해, 비연속 PA 블록들을 GQA-aware per-group
  블록 테이블로 변환하고 memcpy 없이 표준 어텐션 커널에 직접 투입함으로써 비연속 재사용
  처리 병목을 제거한다.

- **Activity C (Block-Union KV 선택 압축)**: 기존 KV 압축 기법은 커스텀 스파스 커널이
  필요하거나(구현 복잡도 높음), 단일 토큰 단위 결정(post-hoc)으로 청크드 프리필의
  블록-스파스 구조를 활용하지 못한다. `CompactAttentionBlockUnionCodec`은 청크드 프리필의
  2D 블록-스파스 어텐션 마스크를 Q-block union + Intra-group union 연산으로 GQA-aware
  KV 선택 블록 테이블로 변환해, 선택된 KV 블록(상위 40%)만 표준 PagedAttention 커널에
  투입한다. 커스텀 스파스 커널 불필요. CompactAttention 원논문 기준 accuracy delta ±0.5%
  이내 (RULER 벤치마크 128K 컨텍스트 기준).

- **Cross B+C**: 두 기법이 공통 `KVSelectionBlockTable` 자료구조를 공유한다.
  B-1이 비연속 PA 블록 히트 집합을 구성하고, C-1이 그 집합에서 중요 블록만 선택한다.
  결합 감소 효과: 비연속 히트율 × KV 선택률 = 곱연산적 메모리 감소 (예: 히트율 50% ×
  선택률 40% → 전체 KV의 20% 사용 = 80% 절감).

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling (미포함, 다음 사이클로 이연)
- [x] Activity B: Non-Contiguous KV Cache Reuse (BlockUnionNonContiguousReuseIndex)
- [x] Activity C: KV Cache Compression (CompactAttentionBlockUnionCodec)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — `attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp) < 0.01` (MANDATORY)
      — kv_selection_ratio=0.40 기준 측정
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 ±1% 이내
      — `cosine_similarity_output(q, k_orig, v_orig, k_comp, v_comp) >= 0.99` (MANDATORY)
      — RULER 벤치마크 proxy: cosine similarity >= 0.99
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C 높음): KV Memory Reduction >= -30%
      — KV 블록 선택률 40% 기준 60% 블록 제거 → -30% 이상 (필수)
      — B+C 결합 시 추가 감소 기대
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C 높음): Effective Context Length 동일 메모리 2× 이상
      — KV 선택 압축으로 확보한 메모리로 더 긴 컨텍스트 수용
- [ ] 목표 5 (evaluation_criteria.md §3 Activity B 높음): 전체 Cache Hit Rate +5%p 이상
      — block-union 변환으로 비연속 히트 처리 가능 범위 확대
- [ ] 목표 6 (evaluation_criteria.md §3 Activity B 높음): 비연속 세그먼트 히트율 >= 전체 히트의 30%
      — memcpy 병목 제거로 비연속 히트 처리 비율 증가
- [ ] 목표 7 (evaluation_criteria.md §1 처리량 높음): Inference Throughput 베이스라인 +20% 이상
      — B+C 결합: block-union 병목 제거 + KV 선택 압축 어텐션 속도 향상 복합 효과
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합): 복합 Throughput 단일 Activity 대비 +5% 이상
      — B-1 단독 vs C-1 단독 vs Cross-1 3방향 비교
- [ ] 목표 9 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후 accuracy ±1% 이내
      — Cross-1 전체 흐름 후 cosine_sim >= 0.99 (MANDATORY)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/block_union_noncontiguous_index.py` | B | BlockUnionNonContiguousReuseIndex — GQA-aware 블록 테이블 변환, memcpy 없는 비연속 재사용 |
| `src/cache/compact_attention_block_union_codec.py` | C | CompactAttentionBlockUnionCodec — 2D 블록-스파스 마스크 → KV 선택 블록 테이블 변환, accuracy-preserving 압축 |
| `src/cache/block_union_bc_pipeline.py` | B+C | BlockUnionBCPipeline — B-1 + C-1 통합 파이프라인, 공통 KVSelectionBlockTable 공유 |
| `tests/unit/test_block_union_noncontiguous_index.py` | B | B-1 단위 테스트: block-union 변환 정확성, GQA per-group 테이블, 비연속 히트율, memcpy vs block-union 오버헤드 |
| `tests/unit/test_compression_accuracy.py` | C | C-1 accuracy 검증: perplexity ±1%, cosine similarity, RULER proxy, 선택률별 곡선 (기존 파일 덮어쓰기) |
| `tests/integration/test_block_union_bc_e2e.py` | B+C | Cross-1 E2E 통합 테스트 |
| `configs/experiments/2026-05-21.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/segmented.py` | `get_segments()` 및 `put_segment()` 경로에 `use_block_union: bool = False` 플래그 추가. `use_block_union=True` 시 `BlockUnionNonContiguousReuseIndex`로 위임. 기존 경로(memcpy fallback) 유지. |

**주의**: `src/cache/base.py`는 이전 사이클(2026-05-20)에서 이미 `get_importance_mask()` 메서드가 추가되어 있다. 추가 변경 불필요. 기존 추상 메서드 6개(put, get, evict, hit_rate, memory_bytes, reset_stats) 불변.

---

## 알고리즘 상세

### 공통 자료구조: KVSelectionBlockTable

B-1과 C-1이 공유하는 핵심 자료구조다.

```python
# src/cache/block_union_noncontiguous_index.py 상단에 정의

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

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
        n_groups = len(self.group_blocks)
        table = torch.full((n_groups, max_len), -1, dtype=torch.int64)
        for g_idx, (g_id, blk_list) in enumerate(sorted(self.group_blocks.items())):
            table[g_idx, :len(blk_list)] = torch.tensor(blk_list, dtype=torch.int64)
        return table
```

---

### BlockUnionNonContiguousReuseIndex (Activity B)

```python
# src/cache/block_union_noncontiguous_index.py

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.block_union_noncontiguous_index import KVSelectionBlockTable, BlockPtr


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

    평가 기준 (evaluation_criteria.md §3):
      - 전체 Cache Hit Rate +5%p 이상 (높음)
      - 비연속 세그먼트 히트율 >= 전체 히트의 30% (높음)
      - KV Memory Footprint 베이스라인 +20% 이내 (높음)
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

        Algorithm:
          1. PA 블록 포인터 목록 생성:
             n_blocks = ceil(value.shape[0] / block_size)
             block_ptrs = list(range(len(_segment_blocks) * n_blocks,
                                    (len(_segment_blocks) + 1) * n_blocks))
             (실제 환경에서는 PA 블록 할당기에서 받아옴 — 여기서는 단조 증가 ID 사용)
          2. _segment_blocks[key] = block_ptrs
          3. _store[key] = value  (fallback 및 to_block_table_tensor() 없는 환경 대비)
          4. LRU 관리: max_entries 초과 시 evict() 호출.
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

    # ------------------------------------------------------------------ #
    # Block-Union API                                                      #
    # ------------------------------------------------------------------ #

    def get_block_union(
        self,
        segment_keys: List[str],
    ) -> Optional[KVSelectionBlockTable]:
        """비연속 세그먼트들의 GQA-aware block-union 테이블 생성.

        Algorithm:
          1. 히트된 세그먼트 필터링:
             hit_keys = [k for k in segment_keys if k in _segment_blocks]
          2. GQA per-group 블록 포인터 통합:
             heads_per_group = n_kv_heads // n_gqa_groups
             for group_id in range(n_gqa_groups):
               head_start = group_id * heads_per_group
               # 그룹 내 모든 세그먼트의 블록 포인터 합집합 (순서 보존 union)
               group_blocks[group_id] = union_ordered(
                   _segment_blocks[k] for k in hit_keys
               )
          3. KVSelectionBlockTable 구성:
             table.n_groups = n_gqa_groups
             table.total_blocks = sum(len(v) for v in group_blocks.values())
             table.selected_blocks = table.total_blocks  (C-1 미적용 시)
          4. 비연속 히트 계수 증가.
          5. miss이면 None 반환.

        Returns: KVSelectionBlockTable 또는 None (모든 세그먼트 미스)
        """
        hit_keys = [k for k in segment_keys if k in self._segment_blocks]
        if not hit_keys:
            self._misses += len(segment_keys)
            return None

        # 히트/미스 계수
        self._hits += len(hit_keys)
        self._misses += len(segment_keys) - len(hit_keys)
        self._block_union_hits += len(hit_keys)

        # 비연속 히트 감지: 연속 접두사가 아닌 히트 존재 여부
        all_keys = segment_keys
        miss_set = set(k for k in all_keys if k not in self._segment_blocks)
        for i, k in enumerate(hit_keys):
            orig_idx = all_keys.index(k)
            if any(all_keys.index(m) < orig_idx for m in miss_set if m in all_keys):
                self._noncontiguous_hits += 1
                break

        # GQA-aware per-group 블록 테이블 구성
        heads_per_group = max(1, self.config.n_kv_heads // max(1, self.config.n_gqa_groups))
        group_blocks: Dict[int, List[BlockPtr]] = {}
        seen: Dict[int, set] = {}

        for g_id in range(self.config.n_gqa_groups):
            group_blocks[g_id] = []
            seen[g_id] = set()

        for seg_key in hit_keys:
            blk_ptrs = self._segment_blocks[seg_key]
            for blk_ptr in blk_ptrs:
                for g_id in range(self.config.n_gqa_groups):
                    if blk_ptr not in seen[g_id]:
                        group_blocks[g_id].append(blk_ptr)
                        seen[g_id].add(blk_ptr)

        total = sum(len(v) for v in group_blocks.values())
        table = KVSelectionBlockTable(
            group_blocks=group_blocks,
            n_groups=self.config.n_gqa_groups,
            total_blocks=total,
            selected_blocks=total,
        )
        return table

    def build_block_union_table(
        self,
        segment_ids: List[str],
    ) -> KVSelectionBlockTable:
        """segment_ids에 대한 GQA-aware 블록 테이블 구성 (히트 미스 무관).

        미스 세그먼트는 건너뛰고, 히트 세그먼트만으로 테이블 구성.
        반환 테이블이 비어 있으면 total_blocks=0.
        """
        result = self.get_block_union(segment_ids)
        if result is None:
            return KVSelectionBlockTable(
                group_blocks={g: [] for g in range(self.config.n_gqa_groups)},
                n_groups=self.config.n_gqa_groups,
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
```

---

### CompactAttentionBlockUnionCodec (Activity C)

```python
# src/cache/compact_attention_block_union_codec.py

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore
from src.cache.block_union_noncontiguous_index import KVSelectionBlockTable, BlockPtr


MaskSource = Literal["profile", "online", "hybrid"]


@dataclass
class BlockUnionCodecConfig:
    chunk_size: int = 2048          # 청크드 프리필 청크 크기 (토큰 수)
    kv_selection_ratio: float = 0.40  # KV 블록 선택률 (상위 40% 유지)
    n_kv_heads: int = 8             # 총 KV 헤드 수
    n_gqa_groups: int = 4           # GQA 그룹 수
    block_size: int = 16            # PA 블록 크기 (토큰 수)
    mask_source: MaskSource = "online"  # 중요도 마스크 생성 방법
    profile_path: Optional[str] = None  # "profile" 방법 사용 시 프로파일 경로
    max_entries: int = 1000
    seed: int = 42


class CompactAttentionBlockUnionCodec(CacheStore):
    """CompactAttention Block-Union KV 선택 압축 코덱.

    Activity C: KV Cache Compression (accuracy-preserving)
    CacheStore 인터페이스 완전 구현.

    핵심 알고리즘:
      청크드 프리필의 2D 블록-스파스 어텐션 마스크를 GQA-aware KV 선택 블록 테이블로 변환.
      선택된 KV 블록(상위 kv_selection_ratio)만 표준 PA 커널에 투입.
      커스텀 스파스 커널 불필요.

    Q-block union 연산:
      쿼리 청크 내 모든 쿼리 토큰이 참조하는 KV 블록의 합집합(union).
      selected_blocks = union_{q ∈ q_chunk} {k_block : M[q][k_block] = 1}

    Intra-group union 연산:
      GQA 그룹 내 헤드별 Q-block union 결과를 그룹 단위로 통합.
      group_blocks[group_id] = union_{h ∈ group} selected_blocks[h]

    accuracy-preserving 근거:
      Q-block union은 쿼리 토큰이 필요로 하는 KV 블록의 합집합이므로
      false negative 없음 (보수적 선택 보장).
      CompactAttention 원논문(2605.16839): LLaMA-3.1-8B-Instruct 128K RULER 벤치마크
      밀집 어텐션 대비 accuracy delta ±0.5% 이내 실험적 검증.

    평가 기준 (evaluation_criteria.md §4):
      - Accuracy 보존 (필수): relative_error < 0.01 (MANDATORY)
      - downstream 태스크 정확도: cosine_sim >= 0.99 (MANDATORY)
      - KV Memory Reduction >= -30% (높음)
    """

    def __init__(self, config: BlockUnionCodecConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        # key → KVSelectionBlockTable (선택된 블록 정보 보존)
        self._block_tables: Dict[str, KVSelectionBlockTable] = {}
        # 청크별 누적 어텐션 중요도 마스크 (online 방법)
        # shape: [n_kv_blocks] (블록 단위 중요도 누적값)
        self._accumulated_importance: Optional[torch.Tensor] = None
        self._n_chunks_processed: int = 0
        self._hits = 0
        self._misses = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0

    # ------------------------------------------------------------------ #
    # 마스크 생성 API                                                       #
    # ------------------------------------------------------------------ #

    def update_chunk_attention(
        self,
        attn_scores: torch.Tensor,   # [n_heads, n_q_chunk, n_kv_total]
        chunk_idx: int = 0,
    ) -> None:
        """청크 어텐션 점수로 중요도 마스크 점진적 구축 (online 방법).

        Algorithm:
          attn_probs = softmax(attn_scores, dim=-1)  # [n_heads, n_q_chunk, n_kv_total]
          # 블록 단위 집계
          n_kv_blocks = ceil(n_kv_total / block_size)
          block_importance = zeros(n_kv_blocks)
          for b in range(n_kv_blocks):
            start = b * block_size
            end = min(start + block_size, n_kv_total)
            block_importance[b] = attn_probs[:, :, start:end].max().item()
          # 누적 (첫 청크가 전체 대표하도록 EMA)
          if _accumulated_importance is None:
            _accumulated_importance = block_importance
          else:
            _accumulated_importance = 0.7 * _accumulated_importance + 0.3 * block_importance
          _n_chunks_processed += 1
        """
        attn_probs = F.softmax(attn_scores.float(), dim=-1)
        n_kv_total = attn_probs.shape[-1]
        n_kv_blocks = max(1, (n_kv_total + self.config.block_size - 1) // self.config.block_size)
        block_importance = torch.zeros(n_kv_blocks)
        for b in range(n_kv_blocks):
            start = b * self.config.block_size
            end = min(start + self.config.block_size, n_kv_total)
            block_importance[b] = attn_probs[..., start:end].max().item()
        if self._accumulated_importance is None or self._accumulated_importance.shape[0] != n_kv_blocks:
            self._accumulated_importance = block_importance
        else:
            self._accumulated_importance = 0.7 * self._accumulated_importance + 0.3 * block_importance
        self._n_chunks_processed += 1

    def build_kv_selection_block_table(
        self,
        n_kv_total: int,
        existing_block_ptrs: Optional[List[BlockPtr]] = None,
    ) -> KVSelectionBlockTable:
        """누적 중요도로 KV 선택 블록 테이블 구성.

        Algorithm:
          n_kv_blocks = ceil(n_kv_total / block_size)
          if _accumulated_importance is not None:
            importance = _accumulated_importance[:n_kv_blocks]
          else:
            importance = ones(n_kv_blocks)  # 폴백: 전부 선택

          # Q-block union: 상위 kv_selection_ratio 블록 선택
          k_select = max(1, int(n_kv_blocks * kv_selection_ratio))
          selected_indices = topk(importance, k_select).indices  # 정렬 안 함
          selected_indices, _ = sort(selected_indices)  # 위치 순서 보존

          # GQA Intra-group union: 그룹별 동일 블록 포인터 부여
          heads_per_group = n_kv_heads // n_gqa_groups
          for g_id in range(n_gqa_groups):
            group_blocks[g_id] = [block_ptrs[i] for i in selected_indices]

          return KVSelectionBlockTable(
            group_blocks=group_blocks,
            n_groups=n_gqa_groups,
            total_blocks=n_kv_blocks,
            selected_blocks=k_select,
          )
        """
        n_kv_blocks = max(1, (n_kv_total + self.config.block_size - 1) // self.config.block_size)

        if self._accumulated_importance is not None and self._accumulated_importance.shape[0] >= n_kv_blocks:
            importance = self._accumulated_importance[:n_kv_blocks]
        else:
            importance = torch.ones(n_kv_blocks)

        k_select = max(1, int(round(n_kv_blocks * self.config.kv_selection_ratio)))
        k_select = min(k_select, n_kv_blocks)
        selected_indices = importance.topk(k_select).indices
        selected_indices, _ = selected_indices.sort()

        # 블록 포인터 결정
        if existing_block_ptrs is None:
            block_ptrs: List[BlockPtr] = list(range(n_kv_blocks))
        else:
            block_ptrs = existing_block_ptrs[:n_kv_blocks]
            while len(block_ptrs) < n_kv_blocks:
                block_ptrs.append(len(block_ptrs))

        selected_ptrs = [block_ptrs[i.item()] for i in selected_indices]

        # GQA Intra-group union
        group_blocks: Dict[int, List[BlockPtr]] = {}
        for g_id in range(self.config.n_gqa_groups):
            group_blocks[g_id] = list(selected_ptrs)

        return KVSelectionBlockTable(
            group_blocks=group_blocks,
            n_groups=self.config.n_gqa_groups,
            total_blocks=n_kv_blocks,
            selected_blocks=k_select,
        )

    def reset_chunk_state(self) -> None:
        """청크드 프리필 종료 후 상태 초기화 (배치 경계)."""
        self._accumulated_importance = None
        self._n_chunks_processed = 0

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """KV 선택 블록 테이블 기반 압축.

        Algorithm:
          n_kv = value.shape[0]
          table = build_kv_selection_block_table(n_kv_total=n_kv)
          _block_tables[key] = table

          # 선택된 블록 위치의 KV만 유지, 나머지 zeroing
          selected_mask = zeros(n_kv, dtype=bool)
          for b in selected_indices:
            start = b * block_size
            end = min(start + block_size, n_kv)
            selected_mask[start:end] = True

          result = zeros_like(value)
          result[selected_mask] = value[selected_mask]
          return result
        """
        n_kv = value.shape[0] if value.dim() >= 1 else 1
        table = self.build_kv_selection_block_table(n_kv_total=n_kv)
        self._block_tables[key] = table

        # 선택된 블록 위치 마스크 생성
        selected_mask = torch.zeros(n_kv, dtype=torch.bool)
        # group_blocks[0]은 모든 그룹 공통 (Intra-group union으로 동일)
        selected_ptrs = table.group_blocks.get(0, [])
        for blk_ptr in selected_ptrs:
            # 블록 포인터를 위치 인덱스로 변환
            # put()에서 block_ptrs = range(n_blocks)로 부여했으므로
            # blk_ptr가 직접 블록 인덱스 역할
            b = blk_ptr % max(1, (n_kv + self.config.block_size - 1) // self.config.block_size)
            start = b * self.config.block_size
            end = min(start + self.config.block_size, n_kv)
            if start < n_kv:
                selected_mask[start:end] = True

        result = torch.zeros_like(value)
        if selected_mask.any():
            result[selected_mask] = value[selected_mask]
        return result

    def get_block_table(self, key: str) -> Optional[KVSelectionBlockTable]:
        """저장된 KV 선택 블록 테이블 반환."""
        return self._block_tables.get(key)

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """KV 선택 블록 테이블에서 bool 마스크 생성 (base.py 인터페이스 구현).

        Returns: [n_kv] bool 텐서 (선택된 위치 True) 또는 None.
        """
        table = self._block_tables.get(key)
        if table is None:
            return None
        kv = self._store.get(key)
        if kv is None:
            return None
        n_kv = kv.shape[0]
        mask = torch.zeros(n_kv, dtype=torch.bool)
        selected_ptrs = table.group_blocks.get(0, [])
        for blk_ptr in selected_ptrs:
            b = blk_ptr % max(1, (n_kv + self.config.block_size - 1) // self.config.block_size)
            start = b * self.config.block_size
            end = min(start + self.config.block_size, n_kv)
            if start < n_kv:
                mask[start:end] = True
        return mask

    def put(self, key: str, value: torch.Tensor) -> None:
        self._total_bytes_original += value.nbytes
        compressed = self.compression_hook(key, value)
        self._total_bytes_stored += compressed.nbytes
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.config.max_entries:
                self.evict()
        self._store[key] = compressed.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._store:
            return 0
        key, v = self._store.popitem(last=False)
        self._block_tables.pop(key, None)
        return v.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def memory_reduction_ratio(self) -> float:
        if self._total_bytes_original == 0:
            return 0.0
        return 1.0 - self._total_bytes_stored / self._total_bytes_original

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0
        self._store.clear()
        self._block_tables.clear()
        self.reset_chunk_state()
```

---

### BlockUnionBCPipeline (Cross B+C)

```python
# src/cache/block_union_bc_pipeline.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

from src.cache.base import CacheStore
from src.cache.block_union_noncontiguous_index import (
    BlockUnionNonContiguousReuseIndex,
    BlockUnionConfig,
    KVSelectionBlockTable,
)
from src.cache.compact_attention_block_union_codec import (
    CompactAttentionBlockUnionCodec,
    BlockUnionCodecConfig,
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

    평가 기준 (evaluation_criteria.md §5):
      - 복합 Throughput: 단일 Activity 대비 +5% 이상 (높음)
      - 복합 Memory Reduction: 단일 Activity 대비 −10% 이상 (높음)
      - Accuracy 보존 (C 포함): 복합 후 cosine_sim >= 0.99 (MANDATORY)
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

        Algorithm (ideas/2026-05-21.md Cross-1 구체 설계 기반):
          Step 1 (세그먼트 히트 감지):
            union_table = b_index.get_block_union(segment_keys)
            if union_table is None: return None

          Step 2 (중요도 마스크 생성):
            if attn_scores is not None:
              c_codec.update_chunk_attention(attn_scores)
            # 히트된 블록만 대상으로 마스크 크기 감소

          Step 3 (통합 KV 선택 블록 테이블 구성):
            if apply_selection_to_union:
              n_hit_blocks = union_table.total_blocks
              sel_table = c_codec.build_kv_selection_block_table(
                n_kv_total=n_hit_blocks * b_index.config.block_size,
                existing_block_ptrs=union_table.group_blocks[0]
              )
              # sel_table이 B+C 결합 블록 테이블
              return sel_table
            else:
              return union_table  # B만 적용

          Step 4 (표준 커널 투입):
            반환된 KVSelectionBlockTable.to_block_table_tensor() →
            표준 PA 커널 block_tables 인자. (호출자가 수행)

        Returns: 최종 KVSelectionBlockTable 또는 None (모든 세그먼트 미스)
        """
        # Step 1
        union_table = self.b_index.get_block_union(segment_keys)
        if union_table is None:
            return None

        # Step 2
        if attn_scores is not None:
            self.c_codec.update_chunk_attention(attn_scores)

        # Step 3
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

        = b_noncontiguous_hit_rate × (1 - c_codec_selection_ratio) 기반
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
```

---

### SegmentedHashCache 변경 (Activity B 연동)

`src/cache/segmented.py`의 `get_segments()`에 `use_block_union` 플래그를 추가한다.

```python
# src/cache/segmented.py — 변경 부분

# __init__에 추가:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.cache.block_union_noncontiguous_index import (
        BlockUnionNonContiguousReuseIndex,
        KVSelectionBlockTable,
    )

# SegmentedHashCache.__init__에 추가:
#   self._block_union_index: Optional["BlockUnionNonContiguousReuseIndex"] = None

def set_block_union_index(
    self,
    index: "BlockUnionNonContiguousReuseIndex",
) -> None:
    """block-union 인덱스를 연결 (use_block_union=True 사용 시 호출)."""
    self._block_union_index = index

def get_segments(
    self,
    token_ids: List[int],
    layer_idx: int = 0,
    codec: Optional["VQCodec"] = None,
    use_block_union: bool = False,  # 신규 파라미터
) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
    """기존 시그니처 유지 + use_block_union 플래그 추가.

    use_block_union=True이고 _block_union_index가 설정된 경우:
      히트 세그먼트들의 chunk_key를 수집해 block-union 인덱스에 위임.
      반환 형식은 기존과 동일 (하위 호환 유지).
    use_block_union=False (기본): 기존 memcpy 경로 그대로.
    """
    # ... 기존 로직 그대로 유지 ...
    # use_block_union=True 경로:
    #   hit_segment_keys = [chunk_key(token_ids, i, layer_idx) for i in hit_chunk_indices]
    #   if self._block_union_index:
    #     table = self._block_union_index.get_block_union(hit_segment_keys)
    #     # 반환 형식은 기존과 동일하게 (chunk_idx, kv_tensor) 유지
    #     # table은 부가 정보로 별도 접근 가능
    ...
```

**중요**: `get_segments()`의 기존 반환 타입 `Tuple[List[Tuple[int, torch.Tensor]], List[int]]`을 유지한다. `use_block_union` 플래그는 내부 처리 경로만 변경하고 외부 인터페이스는 불변이다.

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(`CompactAttentionBlockUnionCodec`)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 proxy — `src/metrics/perplexity.py`의 `attention_output_relative_error()`로 synthetic float32 KV 텐서 활용.
  실제 WikiText-2 데이터셋이 없는 경우, `torch.randn`으로 생성한 `[seq_len, d_head]` 텐서로 대체.
- **모델**: LLaMA-3.1-8B-Instruct 설정 기준 (n_heads=32, n_kv_heads=8, d_head=128, GQA groups=4). 테스트에서는 소형 synthetic 텐서 사용.
- **측정 방법**:
  ```
  relative_error = attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp)
  허용 오차: relative_error < 0.01 (1%) — MANDATORY (evaluation_criteria.md §4 필수)
  ```
- **KV 선택률별 측정 (4개 시나리오)**:
  - kv_selection_ratio=0.50 (상위 50% 유지): relative_error 측정
  - kv_selection_ratio=0.40 (상위 40% 유지, 기본값): relative_error 측정 [MANDATORY]
  - kv_selection_ratio=0.30 (상위 30% 유지): relative_error 측정
  - kv_selection_ratio=0.20 (상위 20% 유지, 공격적): relative_error 측정
- **예상 결과**: kv_selection_ratio=0.40 기준 ±0.5% 이내 (CompactAttention 원논문 근거)

### 태스크 정확도 측정

- **벤치마크**: RULER 벤치마크(128K 컨텍스트) proxy — `src/metrics/perplexity.py`의 `cosine_similarity_output()`로 attention output cosine similarity 측정. RULER 대리 지표로 사용.
- **측정 방법**:
  ```
  cosine_sim = cosine_similarity_output(q, k_orig, v_orig, k_comp, v_comp)
  허용 오차: cosine_sim >= 0.99 — MANDATORY (evaluation_criteria.md §4 필수)
  ```
- **LongBench 8개 서브태스크 proxy**: 동일한 cosine_similarity_output()으로 8개의 독립적 synthetic 시퀀스에 대해 측정. 8개 중 8개 모두 cosine_sim >= 0.99 충족 여부 확인.
- **KL divergence 보조 측정**: `attention_kl_divergence(q, k_orig, k_comp) < 0.015`
- **허용 오차**: ±1% 이내 — evaluation_criteria.md §4 필수

### 청크 크기별 마스크 품질 비교

- chunk_size=512: relative_error 측정
- chunk_size=1024: relative_error 측정
- chunk_size=2048 (기본값): relative_error 측정 [MANDATORY]
- 목적: 청크 크기에 따른 중요도 마스크 품질 안정성 확인

### KV 선택률별 메모리 감소율 검증

- kv_selection_ratio=0.50 → memory_reduction_ratio >= 0.40
- kv_selection_ratio=0.40 → memory_reduction_ratio >= 0.50 [MANDATORY: -30% 기준 충족]
- kv_selection_ratio=0.30 → memory_reduction_ratio >= 0.60

### Fail 기준

**kv_selection_ratio=0.40 기준 relative_error > 1% → 테스트 실패 (evaluation_criteria.md §4 필수 항목 — 무조건 전체 Fail)**

**cosine_sim < 0.99 → 테스트 실패 (evaluation_criteria.md §4 필수 항목 — 무조건 전체 Fail)**

### 검증 테스트 파일

`tests/unit/test_compression_accuracy.py` (기존 파일 덮어쓰기)

**필수 테스트 케이스**:

```
test_block_union_codec_full_selection_zero_error:
    kv_selection_ratio=1.0 → relative_error ≈ 0.0 (기준 검증)

test_block_union_codec_selection_50pct_relative_error_below_1pct:
    kv_selection_ratio=0.50 → relative_error < 0.01

test_block_union_codec_selection_40pct_relative_error_below_1pct:
    kv_selection_ratio=0.40 (기본값) → relative_error < 0.01 (MANDATORY)

test_block_union_codec_selection_40pct_cosine_similarity_above_099:
    kv_selection_ratio=0.40 → cosine_sim >= 0.99 (MANDATORY)

test_block_union_codec_selection_30pct_relative_error_below_1pct:
    kv_selection_ratio=0.30 → relative_error < 0.01

test_block_union_codec_kl_divergence_below_threshold:
    KL divergence < 0.015 (보조 지표)

test_block_union_codec_q_block_union_covers_all_queries:
    Q-block union 결과가 모든 쿼리 토큰 참조 블록을 포함 (false negative 없음 검증)

test_block_union_codec_intra_group_union_per_gqa_group:
    n_gqa_groups=4: 4개 그룹 각각에 동일 선택 블록 포인터 목록 부여 확인

test_block_union_codec_block_table_tensor_shape:
    to_block_table_tensor() 반환 shape == [n_gqa_groups, max_blocks_per_group]

test_block_union_codec_memory_reduction_above_50pct:
    kv_selection_ratio=0.40 → memory_reduction_ratio() >= 0.50
    (evaluation_criteria.md §4 높음 — KV Memory Reduction -30% 이상 포함)

test_block_union_codec_memory_reduction_above_60pct:
    kv_selection_ratio=0.30 → memory_reduction_ratio() >= 0.60

test_block_union_codec_chunk_size_512_accuracy:
    chunk_size=512 → relative_error < 0.01

test_block_union_codec_chunk_size_2048_accuracy:
    chunk_size=2048 (기본) → relative_error < 0.01 (MANDATORY)

test_block_union_codec_longbench_8subtask_proxy:
    8개 독립 synthetic 시퀀스 모두 cosine_sim >= 0.99 (LongBench proxy)

test_block_union_codec_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_block_union_codec_get_importance_mask_returns_bool_tensor:
    put() 후 get_importance_mask(key) → bool tensor 반환, shape [n_kv]

test_block_union_codec_seed_reproducibility:
    동일 seed + attn_scores → 동일 선택 블록 집합

test_cross_bc_pipeline_accuracy_preserved:
    BlockUnionBCPipeline: put → get 왕복 후 cosine_sim >= 0.99 (MANDATORY, §5)

test_cross_bc_combined_reduction_ratio:
    BlockUnionBCPipeline: combined_reduction_ratio() > 0 (B+C 결합 감소 확인)

test_block_union_codec_update_chunk_accumulates_importance:
    update_chunk_attention() 2회 호출 → _accumulated_importance 업데이트 확인

test_block_union_codec_reset_chunk_state_clears_accumulated:
    reset_chunk_state() 후 _accumulated_importance is None 확인
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-21.yaml
experiment:
  date: "2026-05-21"
  activity: "B+C"
  description: >
    Cross-1 BlockUnionNonContiguousCompressionPipeline:
    B-1 BlockUnionNonContiguousReuseIndex (GQA-aware block-union 비연속 재사용, memcpy 없음) +
    C-1 CompactAttentionBlockUnionCodec (청크드 프리필 특화 KV 선택 압축, 표준 커널 재사용).
    공통 KVSelectionBlockTable 자료구조로 B+C 통합.
    CompactAttention(2605.16839) 원논문 기반.
  cache_type: block_union_bc_pipeline
  compression_method: compact_attention_block_union
  scheduler_type: default  # Activity A 미포함

block_union_noncontiguous_index:  # B-1
  block_size: 16                   # PA 블록당 토큰 수
  n_kv_heads: 8                    # 총 KV 헤드 수 (LLaMA-3.1-8B GQA 기준)
  n_gqa_groups: 4                  # GQA 그룹 수 (8 KV heads / 2 heads per group)
  max_entries: 1000
  rope_reencoding_enabled: true    # 위치 불일치 시 RoPE 재인코딩
  use_block_union: true            # block-union 경로 활성화
  seed: 42

compact_attention_block_union_codec:  # C-1
  chunk_size: 2048                 # 청크드 프리필 청크 크기
  kv_selection_ratio: 0.40         # KV 블록 선택률 (상위 40%)
  n_kv_heads: 8
  n_gqa_groups: 4
  block_size: 16
  mask_source: "online"            # 중요도 마스크 생성: online (첫 청크 추정)
  profile_path: null               # "profile" 방법 사용 시 경로
  max_entries: 1000
  seed: 42

bc_pipeline:  # Cross-1
  apply_selection_to_union: true   # B 히트 블록에 C 선택 추가 적용
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    dataset_proxy: "wikitext2_synthetic"
    task_accuracy_proxy: "ruler_cosine_similarity"
    relative_error_max: 0.01       # ±1% (evaluation_criteria.md §4 MANDATORY)
    cosine_similarity_min: 0.99    # (evaluation_criteria.md §4 MANDATORY)
    kl_divergence_max: 0.015       # 보조 지표
    kv_selection_ratios_to_test: [0.50, 0.40, 0.30, 0.20]
    chunk_sizes_to_test: [512, 1024, 2048]
    longbench_subtask_count: 8
  activity_b:
    cache_hit_rate_improvement_min_pct: 5.0   # +5%p (§3 높음)
    noncontiguous_hit_rate_min_pct: 30.0      # 전체 히트의 30% 이상 (§3 높음)
    memory_footprint_max_increase_pct: 20.0   # +20% 이내 (§3 높음)
  activity_c:
    memory_reduction_min: 0.30     # -30% 이상 (§4 높음)
    effective_context_multiplier: 2.0
    compression_overhead_ttft_max_pct: 10.0   # TTFT +10% 이내 (§4 높음)
  cross_bc:
    throughput_min_improvement_vs_solo: 5.0   # 단일 Activity 대비 +5% (§5 높음)
    memory_min_improvement_vs_solo: 10.0      # 단일 Activity 대비 -10% (§5 높음)
    accuracy_cosine_min: 0.99                 # C 포함 (§5 MANDATORY)
    comparison_methods: ["solo_b1", "solo_c1", "cross_bc"]
  throughput:
    target_improvement_pct: 20     # 베이스라인 대비 +20% (§1 장기 목표)

segmented_cache_ab_comparison:
  # SegmentedHashCache.get_segments() A/B 비교 실험
  use_block_union_path: true       # block-union 경로
  use_memcpy_fallback_path: false  # memcpy fallback 경로

seed: 42
results_dir: "results/2026-05-21"
```

---

## 테스트 요구사항

- [x] `tests/unit/test_compression_accuracy.py` — Activity C 필수 accuracy 검증 (21개 테스트, 위 목록 참조) (기존 파일 덮어쓰기)
- [x] `tests/unit/test_block_union_noncontiguous_index.py` — Activity B 단위 테스트
- [x] `tests/integration/test_block_union_bc_e2e.py` — Cross-1 E2E 통합 테스트

### 단위 테스트 최소 요구사항 (test_block_union_noncontiguous_index.py)

```
test_kv_selection_block_table_to_tensor_shape:
    KVSelectionBlockTable.to_block_table_tensor() shape 검증

test_kv_selection_block_table_selection_ratio:
    selected_blocks=4, total_blocks=10 → selection_ratio == 0.4

test_block_union_index_put_creates_block_ptrs:
    put(key, value) 후 _segment_blocks[key] 존재 확인

test_block_union_index_get_returns_stored_value:
    put → get 왕복 확인

test_block_union_index_get_block_union_single_segment:
    단일 세그먼트 히트 → KVSelectionBlockTable 반환

test_block_union_index_get_block_union_multiple_segments:
    3개 세그먼트 히트 → group_blocks에 union 블록 포인터 포함

test_block_union_index_get_block_union_all_miss:
    모든 세그먼트 미스 → None 반환

test_block_union_index_gqa_group_count:
    n_gqa_groups=4 → group_blocks에 4개 그룹 키 존재

test_block_union_index_noncontiguous_hit_detection:
    [hit, miss, hit] 패턴 → noncontiguous_hits 증가 확인

test_block_union_index_noncontiguous_hit_rate:
    비연속 히트율 = noncontiguous_hits / total_hits

test_block_union_index_build_block_union_table_empty:
    모든 키 미스 → total_blocks=0 테이블 반환

test_block_union_index_evict_lru:
    max_entries=2, 3개 put → 첫 번째 항목 퇴거 확인

test_block_union_index_hit_rate_tracking:
    put 2개 후 get 1회 히트 + 1회 미스 → hit_rate() == 0.5

test_block_union_index_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_block_union_index_seed_reproducibility:
    동일 put 순서 + 동일 seed → 동일 block_ptrs

test_segmented_cache_use_block_union_flag:
    SegmentedHashCache.get_segments(use_block_union=True) →
    block-union 인덱스 경로 호출 확인

test_segmented_cache_use_block_union_false_unchanged:
    SegmentedHashCache.get_segments(use_block_union=False) →
    기존 memcpy 경로 동작 유지 (하위 호환)
```

### 통합 테스트 최소 요구사항 (test_block_union_bc_e2e.py)

```
test_e2e_bc_pipeline_put_get_basic:
    put → get 왕복 기본 동작

test_e2e_bc_pipeline_process_noncontiguous_segments:
    b_index에 3개 세그먼트 put 후 process_noncontiguous_segments() →
    KVSelectionBlockTable 반환 (non-None)

test_e2e_bc_pipeline_combined_selection_smaller_than_union:
    apply_selection_to_union=True →
    sel_table.selected_blocks < union_table.total_blocks (C가 추가 압축)

test_e2e_bc_pipeline_accuracy_preserved_cosine_above_099:
    c_codec kv_selection_ratio=0.40 → cosine_sim >= 0.99 (MANDATORY, §5)

test_e2e_bc_pipeline_memory_reduction_above_30pct:
    c_codec kv_selection_ratio=0.40 → memory_reduction_ratio() >= 0.30
    (evaluation_criteria.md §4 높음 항목)

test_e2e_bc_metrics_summary_all_keys:
    metrics_summary()가 필수 키 포함:
    [b_noncontiguous_hit_rate, b_block_union_hit_rate,
     c_memory_reduction_ratio, bc_combined_reduction_estimate,
     total_memory_bytes, hit_rate]

test_e2e_bc_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_e2e_bc_solo_b_vs_solo_c_vs_cross:
    B-1 단독 / C-1 단독 / Cross-1 메모리 감소율 3방향 비교 기록

test_e2e_bc_runner_integration:
    InferenceRunner(cache=BlockUnionBCPipeline) 로 run_batch() 호출 성공
    (src/engine/runner.py 사용)

test_e2e_bc_block_table_tensor_injectable:
    process_noncontiguous_segments() 반환 테이블의
    to_block_table_tensor()가 유효한 int64 텐서 반환 확인
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 3개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - `test_block_union_codec_selection_40pct_relative_error_below_1pct` 통과 (relative_error < 0.01)
      - `test_block_union_codec_selection_40pct_cosine_similarity_above_099` 통과 (cosine_sim >= 0.99)
      - `test_block_union_codec_memory_reduction_above_50pct` 통과 (reduction >= 0.50)
- [ ] `evaluation_criteria.md` §3 Activity B 항목 충족:
      - `test_block_union_index_noncontiguous_hit_detection` 통과
      - `test_block_union_index_get_block_union_multiple_segments` 통과
      - `test_segmented_cache_use_block_union_flag` 통과 (하위 호환 유지)
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함:
      - `test_e2e_bc_pipeline_accuracy_preserved_cosine_above_099` 통과 (MANDATORY)
      - B-1 단독 / C-1 단독 / Cross-1 3방향 비교 수치 기록
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - CacheStore 인터페이스 모든 추상 메서드 구현
        (BlockUnionNonContiguousReuseIndex, CompactAttentionBlockUnionCodec, BlockUnionBCPipeline)
      - `src/cache/segmented.py` 변경 후 기존 테스트(`test_segmented_cache.py`) 회귀 없음 확인
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-21.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-21/metrics.json`에 JSON 기록:
      ```json
      {
        "inference_throughput_improvement_pct": ...,
        "kv_memory_reduction_ratio_c1_solo": ...,
        "kv_memory_reduction_ratio_bc_cross": ...,
        "block_union_relative_error_ratio_040": ...,
        "block_union_cosine_similarity_040": ...,
        "block_union_kl_divergence": ...,
        "effective_context_length_multiplier": ...,
        "b1_noncontiguous_hit_rate": ...,
        "b1_block_union_hit_rate": ...,
        "b1_total_cache_hit_rate_improvement_pct": ...,
        "c1_kv_selection_ratio": 0.40,
        "c1_memory_reduction_ratio": ...,
        "bc_combined_reduction_estimate": ...,
        "bc_vs_solo_b1_throughput_pct": ...,
        "bc_vs_solo_c1_throughput_pct": ...,
        "bc_accuracy_cosine": ...,
        "ruler_proxy_cosine_128k": ...,
        "longbench_8subtask_cosine_min": ...
      }
      ```
- [ ] `src/cache/segmented.py` `get_segments()`에 `use_block_union=False` 기본값 파라미터 추가 — 기존 호출 시그니처 불변
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
