<!-- 변경 이유 (이전 Spec.md: 2026-05-04 대비):
이전 사이클(2026-05-04)은 A+B+C (DAGTopologyScheduler + WorkloadAwareTTLCache + RedundancyAwareEvictionPolicy) 조합이었다.
이번 사이클은 B+C (DiffAwareSegmentStore + NQKVCodec + CompressedDiffStore + FireQCodec) 조합으로 전환한다.

주요 변경:
1. [Activity A 제외] DAGTopologyScheduler/DAGAwareTTLAdjuster가 이미 2026-05-04에 구현 완료되었으므로
   이번 사이클에서는 Activity A를 포함하지 않는다. 스케줄러 관련 파일은 수정하지 않는다.

2. [Activity B 교체] WorkloadAwareTTLCache(카테고리별 TTL 세그먼트 보존) →
   DiffAwareSegmentStore(마스터+블록-희소 차분 저장). 알고리즘 패러다임이 "시간 패턴 기반 TTL 보존"
   에서 "에이전트 간 공통 KV 마스터+차분 저장"으로 전환된다. FAISS N>10K 병목을
   검색 공간을 마스터 수(그룹 수)로 구조적으로 축소해 우회한다.

3. [Activity C 강화] RedundancyAwareEvictionPolicy(중요도×중복성 이중 스코어 퇴거, 보조 레이어) →
   NQKVCodec(정규분포 블록-분위수 Normal Float INT4 양자화) + FireQCodec(RoPE-인식 2단계
   아웃라이어 평활화 INT4+FP8 혼합). Activity C가 보조에서 메인 스코프로 격상된다.
   실제 GPU 처리량 검증 경로와 실제 perplexity 측정이 이번 사이클 필수 산출물이다.

4. [Cross-1 신규] CompressedDiffStore: DiffAwareSegmentStore(B) + NQKVCodec(C) 통합.
   마스터 KV를 INT4로 압축하고 차분 블록은 FP16 원본으로 유지한다.
   이전 사이클의 DAGAwareTTLAdjuster(A+B 교차 연결)와는 완전히 다른 설계다.

5. [SUMMARY.md 미해결 항목 직접 해소]
   - 미해결 1 (실제 GPU 처리량 미검증): FireQCodec torch.cuda.Event TTFT 실측 스크립트 포함
   - 미해결 2 (FAISS N>10K 병목): DiffAwareSegmentStore 마스터 수 기반 검색 공간 축소로 우회
   - 미해결 5 (실제 perplexity 미측정): NQKVCodec GPT-2+WikiText-2 perplexity 직접 측정 스크립트 포함

기존 파일(workload_ttl_cache.py, redundancy_eviction.py, dag_topology_scheduler.py, dag_ttl_adjuster.py,
turbo_quant.py, dhd_segment_cache.py, speculative_fetcher.py, sign_vq_segment.py, leverage_compressor.py,
compression.py, segmented.py, contiguous.py, tri_state_compressor.py, compressed_segment.py,
segment_adapter.py, cache_aware_scheduler.py, dual_map_scheduler.py, multi_node_scheduler.py)은
이번 사이클에서 수정하지 않는다.
기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-05

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-05.md`
**최우선 구현 타겟**: Cross-1 (B+C) — DiffAwareSegmentStore(B-1) + NQKVCodec(C-1) + CompressedDiffStore 통합, 보조로 FireQCodec(C-2)

**해결하려는 문제**:
- 기존 세그먼트 캐시(`SegmentedHashCache`, `WorkloadAwareTTLCache` 등)는 에이전트 그룹 내 공통 KV를 각 에이전트마다 독립적으로 저장해 중복이 심하다. 멀티에이전트 All-Gather 패턴에서 17.5× 저장 낭비가 발생한다.
- FAISS N>10K 유사도 검색 병목이 SUMMARY.md의 미해결 항목으로 남아 있다. 검색 공간을 마스터 복사본 수(그룹 수)로 제한하면 이 병목을 구조적으로 우회할 수 있다.
- 이전 사이클까지 Activity C 압축은 실제 perplexity를 직접 측정하지 않고 proxy 수치만 사용했다. 이번 사이클에서는 GPT-2 + WikiText-2 perplexity를 직접 측정해 SUMMARY.md 미해결 항목 5를 해소한다.
- 기존 Hadamard/PolarQuant 계열 양자화는 행렬 변환 오버헤드가 있었다. NQKVCodec은 변환 없이 정규분포 블록-분위수만으로 INT4를 달성한다.
- RoPE-인식 이상치 처리가 이전 사이클의 모든 양자화 기법에서 명시적으로 다루지 않았다. FireQCodec이 RoPE-INT4 충돌을 2단계 평활화로 명시적으로 해소한다.

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling (이번 사이클 제외 — 2026-05-04에 구현 완료)
- [x] Activity B: Non-Contiguous KV Cache Reuse — DiffAwareSegmentStore (마스터+블록-희소 차분 저장)
- [x] Activity C: KV Cache Compression — NQKVCodec (Normal Float INT4) + FireQCodec (RoPE-인식 INT4+FP8)

---

## 목표

- [ ] 목표 1 (§3 Non-Contiguous Hit Rate): 전체 히트 중 비연속 히트 비율 ≥ 30% (diff hit + master hit)
- [ ] 목표 2 (§4 Accuracy 필수): perplexity 변화 ±1% 이내 (GPT-2 + WikiText-2 직접 실측)
- [ ] 목표 3 (§4 Accuracy 필수): downstream 태스크 정확도 변화 ±1% 이내
- [ ] 목표 4 (§4 KV Memory Reduction): 베이스라인 대비 −30% 이상 (NQKVCodec INT4 기여)
- [ ] 목표 5 (§3 KV Memory Footprint): 에이전트 그룹 시나리오에서 독립 저장 대비 −85% 이상 (차분 저장 기여)
- [ ] 목표 6 (§1 Throughput): tokens/sec 베이스라인 대비 +20% 이상 (Cross-1 복합 효과)
- [ ] 목표 7 (§4 Compression Overhead): Encode/Decode 추가 지연 TTFT +10% 이내
- [ ] 목표 8 (§5 Cross): 복합 메모리 감소 단일 Activity 대비 추가 −10% 이상 (마스터 INT4 압축 + 차분 희소 저장 승수 효과)
- [ ] 목표 9 (§5 Cross): 복합 처리량 향상 단일 Activity 대비 추가 +5% 이상
- [ ] 목표 10 (SUMMARY.md 미해결 항목): FAISS 검색 공간 — 마스터 수(그룹 수) vs 기존 전체 세그먼트 수 비교 측정

---

## 아키텍처 개요

```
요청 도착 (에이전트 그룹 컨텍스트 포함)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  CompressedDiffStore (Cross-1 통합 — src/cache/compressed_diff_store.py)  │
│                                                                  │
│  register_master(master_kv, group_id)                           │
│    └─ NQKVCodec.encode(master_kv) → INT4 압축 마스터 저장        │
│                                                                  │
│  put_agent_kv(agent_id, group_id, agent_kv)                     │
│    └─ L2 거리 > diff_threshold: diff 블록 저장 (FP16 원본)       │
│    └─ L2 거리 ≤ diff_threshold: 마스터 참조 포인터만 저장        │
│                                                                  │
│  get_agent_kv(agent_id, group_id)                               │
│    └─ NQKVCodec.decode(master_int4) + diff 블록 병합             │
│       → 에이전트 전체 KV 재구성 (FP16)                           │
└──────────────────┬───────────────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌─────────────────┐  ┌──────────────────────────┐
│  NQKVCodec (C-1)│  │  DiffAwareSegmentStore   │
│  src/cache/     │  │  (B-1)                   │
│  nqkv_codec.py  │  │  src/cache/              │
│                 │  │  diff_aware_store.py      │
│  Normal Float   │  │                          │
│  INT4 분위점    │  │  마스터 KV (FP16/INT4)   │
│  테이블 (상수)  │  │  diff 블록 (FP16 희소)   │
│  블록 크기: 64  │  │  그룹 LRU 관리           │
│  (μ, σ) FP16   │  │  FAISS 미사용             │
│  블록당 저장    │  │  (검색 공간 = 그룹 수)   │
└─────────────────┘  └──────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  FireQCodec (C-2 보조 — src/cache/fireq_codec.py)               │
│                                                                  │
│  calibrate(calib_kvs, layer_idx) → pre_rope_scales 저장         │
│  encode(kv, layer_idx, rope_applied) → Key INT4 + Value FP8     │
│  decode(compressed, layer_idx) → FP16 복원                      │
│                                                                  │
│  Stage 1: pre-RoPE 채널 쌍 스케일 팩터                          │
│  Stage 2: post-RoPE 이상치 채널 마스크 + 표적 스케일링           │
└──────────────────────────────────────────────────────────────────┘

측정 산출물:
  experiments/run_perplexity_nqkv.py   → results/nqkv_perplexity/metrics.json
  experiments/run_gpu_throughput.py    → results/fireq_throughput/metrics.json
```

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/nqkv_codec.py` | C-1 | `NQKVCodec` — 정규분포 블록-분위수 Normal Float INT4 양자화. `CompressionCodec` 인터페이스 구현 |
| `src/cache/diff_aware_store.py` | B-1 | `DiffAwareSegmentStore` — 마스터+블록-희소 차분 저장. `CacheStore` 인터페이스 구현 |
| `src/cache/compressed_diff_store.py` | Cross-1 | `CompressedDiffStore` — `DiffAwareSegmentStore` 상속. 마스터 INT4 압축 + 차분 FP16 통합 |
| `src/cache/fireq_codec.py` | C-2 | `FireQCodec` — RoPE-인식 2단계 아웃라이어 평활화 + INT4+FP8 혼합. `CompressionCodec` 구현 |
| `experiments/run_perplexity_nqkv.py` | C-1 | GPT-2 + WikiText-2 perplexity 직접 실측 스크립트. NQKVCodec ON/OFF 비교 |
| `experiments/run_gpu_throughput.py` | C-2 | torch.cuda.Event TTFT/TBT 실측 스크립트. CPU 환경에서는 NumPy 기반 정확도 검증 |
| `configs/experiments/2026-05-05.yaml` | 공통 | 실험 설정 YAML |
| `tests/unit/test_nqkv_accuracy.py` | C-1 | NQKVCodec encode/decode 왕복 + perplexity 직접 측정 테스트 |
| `tests/unit/test_diff_aware_store.py` | B-1 | DiffAwareSegmentStore 마스터+차분 저장/복원 왕복 테스트 |
| `tests/unit/test_fireq_codec.py` | C-2 | FireQCodec RoPE 이상치 처리 검증 테스트 |
| `tests/integration/test_cross_bc_diff_compressed.py` | Cross-1 | CompressedDiffStore 통합 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | 변경 없음 — 기존 CacheStore 인터페이스 그대로 준수 |

**주의**: 기존 파일(`workload_ttl_cache.py`, `redundancy_eviction.py`, `dag_topology_scheduler.py`,
`dag_ttl_adjuster.py`, `turbo_quant.py`, `dhd_segment_cache.py`, `speculative_fetcher.py`,
`sign_vq_segment.py`, `leverage_compressor.py`, `compression.py`, `segmented.py`,
`contiguous.py`, `tri_state_compressor.py`, `compressed_segment.py`, `segment_adapter.py`,
`cache_aware_scheduler.py`, `dual_map_scheduler.py`, `multi_node_scheduler.py`)은
이번 사이클에서 수정하지 않는다.

---

## 알고리즘 상세

### 1. NQKVCodec — Normal Float INT4 분위수 양자화 (Activity C-1)

**파일**: `src/cache/nqkv_codec.py`

**핵심 아이디어**:
- KV 캐시 원소가 정규분포를 따른다는 관찰에 기반해, 정규분포의 등면적 분위점 15개로 16개 구간을 정의한다.
- 각 블록(64 원소)에 대해 (μ, σ) 2개 FP16 스칼라만 추가 저장해 역양자화를 수행한다.
- 행렬 변환(Hadamard, PolarQuant, SVD 등) 오버헤드가 전혀 없다.
- `CompressionCodec` 인터페이스로 기존 `CompressedSegmentCache`에 drop-in 교체 가능.

**Normal Float 분위점 테이블** (상수, 재계산 불필요):
```
NF4_QUANTILES = [
    -1.7408,  # Φ^{-1}(0.5/16) 기준 등면적 구간 경계
    -1.2315,
    -0.9004,
    -0.6340,
    -0.4001,
    -0.1876,
     0.0000,
     0.1876,
     0.4001,
     0.6340,
     0.9004,
     1.2315,
     1.7408,
]
# 16개 대표값 (각 구간의 중앙값):
NF4_VALUES = [-1.9951, -1.4731, -1.0607, -0.7589, -0.5165, -0.2892,
              -0.0941,  0.0941,  0.2892,  0.5165,  0.7589,  1.0607,
               1.4731,  1.9951]
# 4비트 → 16개 구간 → 각 구간 대표값으로 역양자화
```

**의사코드**:

```python
import torch
import numpy as np
from typing import Optional, Tuple

# CompressionCodec 인터페이스 (src/cache/compression.py에서 임포트)
from src.cache.compression import CompressionCodec


NF4_VALUES: list[float] = [...]  # 위 14개 대표값 (상수 정의)


class NQKVCodec(CompressionCodec):
    """Normal Float INT4 블록-분위수 양자화 코덱 (Activity C-1).

    변환 오버헤드 없음. 블록당 (mu, sigma) FP16 2개 스칼라만 추가 저장.
    CacheStore가 아닌 CompressionCodec 인터페이스 구현.
    """

    def __init__(
        self,
        block_size: int = 64,        # 양자화 블록 크기 (원소 수)
        nf4_values: Optional[list] = None,  # 커스텀 분위점 테이블 (None → 기본값)
    ) -> None:
        self.block_size = block_size
        self.nf4_values = torch.tensor(nf4_values or NF4_VALUES, dtype=torch.float32)

    def encode(self, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FP16/FP32 KV 텐서 → (indices_int4, mu_fp16, sigma_fp16) 반환.

        Args:
            kv: (*, D) 형태의 KV 텐서. D는 블록 크기의 배수여야 함 (패딩 허용)
        Returns:
            indices: torch.uint8 텐서, shape=(num_blocks, block_size) — 4비트 인덱스 (0~13)
            mu:      torch.float16 텐서, shape=(num_blocks,) — 블록별 평균
            sigma:   torch.float16 텐서, shape=(num_blocks,) — 블록별 표준편차

        의사코드:
            flat = kv.float().reshape(-1)
            num_blocks = (len(flat) + block_size - 1) // block_size
            pad_flat = F.pad(flat, (0, num_blocks * block_size - len(flat)))
            blocks = pad_flat.reshape(num_blocks, block_size)
            mu = blocks.mean(dim=-1)                           # (num_blocks,)
            sigma = blocks.std(dim=-1).clamp(min=1e-8)        # (num_blocks,)
            normalized = (blocks - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
            # 각 원소를 NF4_VALUES에서 가장 가까운 인덱스로 매핑
            # normalized: (num_blocks, block_size)
            # nf4: (14,) → broadcast 비교 → argmin(|normalized - nf4|)
            diff = (normalized.unsqueeze(-1) - nf4_values.view(1, 1, -1)).abs()
            indices = diff.argmin(dim=-1).to(torch.uint8)      # (num_blocks, block_size)
            return indices, mu.half(), sigma.half()
        """
        ...

    def decode(
        self,
        indices: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        original_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        """(indices_int4, mu_fp16, sigma_fp16) → FP16 KV 텐서 복원.

        의사코드:
            # nf4_values[indices] → 정규화된 값
            reconstructed = nf4_values[indices.long()]  # (num_blocks, block_size)
            # 역정규화: x = reconstructed * sigma + mu
            restored = reconstructed * sigma.unsqueeze(-1) + mu.unsqueeze(-1)
            flat = restored.reshape(-1)
            if original_shape is not None:
                flat = flat[:original_shape.numel()]
                return flat.reshape(original_shape).half()
            return flat.half()
        """
        ...

    def compression_ratio(self, kv: torch.Tensor) -> float:
        """FP16 대비 압축률 반환. INT4 = 0.5 bytes/elem + 2 FP16/block overhead."""
        n_elem = kv.numel()
        n_blocks = (n_elem + self.block_size - 1) // self.block_size
        compressed_bytes = n_blocks * self.block_size * 0.5 + n_blocks * 4  # 4 = 2×FP16
        original_bytes = n_elem * 2  # FP16
        return original_bytes / compressed_bytes
```

---

### 2. DiffAwareSegmentStore — 마스터+블록-희소 차분 저장 (Activity B-1)

**파일**: `src/cache/diff_aware_store.py`

**핵심 아이디어**:
- 에이전트 그룹의 공통 KV를 단일 마스터로 저장하고, 개별 에이전트의 차이만 블록-희소 차분으로 저장한다.
- 블록 단위(기본 64 토큰): 마스터 블록과 에이전트 블록 간 L2 거리 > diff_threshold이면 차분 블록 저장, 이하이면 마스터 참조 포인터만 저장.
- 그룹 LRU 관리: 그룹 마스터 퇴거 시 해당 그룹의 모든 에이전트 차분도 함께 퇴거.
- FAISS 미사용: 검색 공간을 마스터 수(그룹 수)로 제한해 N>10K 병목 구조적 우회.

**CacheStore 인터페이스 구현**:
- `put(key, value)`: key를 `"master:{group_id}"` 또는 `"agent:{group_id}:{agent_id}"` 형식으로 자동 파싱해 라우팅
- `get(key)`: 에이전트 키이면 마스터 + 차분 병합 후 반환
- `evict()`: 가장 오래된 LRU 그룹 퇴거 (그룹 마스터 + 모든 에이전트 차분)
- `hit_rate()`, `memory_bytes()`, `reset_stats()`: 표준 구현

**의사코드**:

```python
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import torch

from src.cache.base import CacheStore


@dataclass
class MasterEntry:
    kv: torch.Tensor          # 마스터 KV (FP16 또는 상위 클래스에서 INT4)
    group_id: str


@dataclass
class DiffEntry:
    """에이전트 차분 저장 엔트리."""
    group_id: str
    agent_id: str
    # diff_blocks[block_idx] = diff_tensor (차분이 있는 블록만 저장)
    diff_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    # master_ref_blocks: 차분 없는 블록 인덱스 집합 (마스터 참조)
    master_ref_blocks: Set[int] = field(default_factory=set)


class DiffAwareSegmentStore(CacheStore):
    """에이전트 간 마스터+블록-희소 차분 저장 KV 캐시 (Activity B-1).

    # NOTE: FAISS 미사용. 검색 공간 = 그룹(마스터) 수
    # → N>10K 병목 구조적 우회 (SUMMARY.md 미해결 항목 2 직접 해소)
    """

    def __init__(
        self,
        block_size: int = 64,           # 차분 계산 블록 크기 (토큰 단위)
        diff_threshold: float = 0.1,    # 블록 L2 거리 임계값 (이 이하이면 마스터 참조)
        max_groups: int = 100,          # 최대 그룹(마스터) 수 — LRU 퇴거
    ) -> None:
        self.block_size = block_size
        self.diff_threshold = diff_threshold
        self.max_groups = max_groups
        # 그룹 순서 기반 LRU: OrderedDict[group_id, MasterEntry]
        self._masters: OrderedDict[str, MasterEntry] = OrderedDict()
        # 에이전트 차분: {group_id: {agent_id: DiffEntry}}
        self._diffs: Dict[str, Dict[str, DiffEntry]] = {}
        self._hits = 0
        self._misses = 0
        self._diff_hits = 0    # 차분 블록에서 발생한 히트
        self._master_hits = 0  # 마스터 참조에서 발생한 히트

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """key 포맷:
        - "master:{group_id}" → register_master(value, group_id) 호출
        - "agent:{group_id}:{agent_id}" → put_agent_kv(agent_id, group_id, value) 호출
        - 그 외 → group_id=key, agent_id="default"로 처리
        """
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        """key 포맷:
        - "master:{group_id}" → 마스터 KV 반환
        - "agent:{group_id}:{agent_id}" → get_agent_kv(agent_id, group_id) 반환
        """
        ...

    def evict(self) -> int:
        """가장 오래된 LRU 그룹 퇴거. 그룹 마스터 + 해당 그룹 모든 에이전트 차분 삭제.
        반환: 해제된 bytes 수
        """
        if not self._masters:
            return 0
        oldest_group_id = next(iter(self._masters))
        return self._evict_group(oldest_group_id)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """마스터 KV bytes + 모든 에이전트 차분 bytes 합산."""
        total = sum(e.kv.nbytes for e in self._masters.values())
        for agent_diffs in self._diffs.values():
            for diff_entry in agent_diffs.values():
                total += sum(b.nbytes for b in diff_entry.diff_blocks.values())
        return total

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._diff_hits = 0
        self._master_hits = 0

    # ------------------------------------------------------------------ #
    # 확장 API                                                             #
    # ------------------------------------------------------------------ #

    def register_master(self, master_kv: torch.Tensor, group_id: str) -> None:
        """에이전트 그룹의 공통 KV를 마스터로 등록.

        max_groups 초과 시 LRU 그룹 퇴거 후 저장.
        """
        if len(self._masters) >= self.max_groups and group_id not in self._masters:
            self.evict()
        self._masters[group_id] = MasterEntry(
            kv=master_kv.detach().clone(),
            group_id=group_id,
        )
        self._masters.move_to_end(group_id)
        if group_id not in self._diffs:
            self._diffs[group_id] = {}

    def put_agent_kv(
        self,
        agent_id: str,
        group_id: str,
        agent_kv: torch.Tensor,
    ) -> None:
        """에이전트 KV를 마스터와 비교해 차분 블록만 저장.

        블록 단위 L2 거리 계산:
            n_blocks = ceil(seq_len / block_size)
            for block_idx in range(n_blocks):
                master_block = master_kv[..., start:end, :]
                agent_block  = agent_kv[..., start:end, :]
                l2_dist = (master_block - agent_block).norm()
                if l2_dist > diff_threshold:
                    diff_entry.diff_blocks[block_idx] = agent_block - master_block
                else:
                    diff_entry.master_ref_blocks.add(block_idx)
        """
        ...

    def get_agent_kv(
        self,
        agent_id: str,
        group_id: str,
    ) -> Optional[torch.Tensor]:
        """마스터 KV + 차분 블록 병합 → 에이전트 전체 KV 재구성.

        의사코드:
            master_kv = _get_master_kv(group_id)  # INT4이면 decode 수행 (CompressedDiffStore)
            result = master_kv.clone()
            for block_idx, diff_block in diff_entry.diff_blocks.items():
                result[..., start:end, :] = master_kv[..., start:end, :] + diff_block
            return result
        """
        ...

    def diff_hit_stats(self) -> dict:
        """비연속 히트 세분화 통계.

        Returns:
            {
              "diff_hit_rate":   diff_hits / total_hits,   # 차분 블록 히트
              "master_hit_rate": master_hits / total_hits, # 마스터 참조 포인터 히트
              "overall_hit_rate": (hits) / (hits + misses),
              "n_groups": int,          # 현재 등록된 그룹(마스터) 수
              "search_space_reduction": baseline_segments / n_groups,  # FAISS 우회 효과
            }
        """
        ...

    def _get_master_kv(self, group_id: str) -> Optional[torch.Tensor]:
        """마스터 KV 반환. CompressedDiffStore에서 override해 INT4 역양자화."""
        entry = self._masters.get(group_id)
        if entry is None:
            return None
        self._masters.move_to_end(group_id)
        return entry.kv

    def _evict_group(self, group_id: str) -> int:
        """특정 그룹 퇴거. 마스터 + 모든 에이전트 차분 삭제."""
        freed = 0
        if group_id in self._masters:
            freed += self._masters.pop(group_id).kv.nbytes
        if group_id in self._diffs:
            for diff_entry in self._diffs.pop(group_id).values():
                freed += sum(b.nbytes for b in diff_entry.diff_blocks.values())
        return freed
```

---

### 3. CompressedDiffStore — NQKVCodec + DiffAwareSegmentStore 통합 (Cross-1)

**파일**: `src/cache/compressed_diff_store.py`

**핵심 아이디어**:
- `DiffAwareSegmentStore`를 상속하고 마스터 저장 시 `NQKVCodec`으로 INT4 자동 압축한다.
- 차분 블록은 FP16 원본으로 유지(규모가 작으므로 압축 오버헤드 무시 가능).
- `_get_master_kv()` override: INT4 역양자화 후 반환.
- 결과: 마스터 4× INT4 압축 × 에이전트 차분 희소 저장 = 승수적 메모리 절감.

**의사코드**:

```python
from typing import Optional, Tuple
import torch

from src.cache.diff_aware_store import DiffAwareSegmentStore, MasterEntry
from src.cache.nqkv_codec import NQKVCodec


@dataclass
class CompressedMasterEntry:
    """INT4 압축된 마스터 KV 저장 엔트리."""
    indices: torch.Tensor      # (num_blocks, block_size) uint8
    mu: torch.Tensor           # (num_blocks,) float16
    sigma: torch.Tensor        # (num_blocks,) float16
    original_shape: torch.Size
    group_id: str


class CompressedDiffStore(DiffAwareSegmentStore):
    """DiffAwareSegmentStore + NQKVCodec 통합 (Cross-1).

    마스터 KV → INT4 압축 저장
    차분 블록 → FP16 원본 저장 (희소)
    복원 → 마스터 INT4 역양자화 + 차분 병합
    """

    def __init__(
        self,
        block_size: int = 64,
        diff_threshold: float = 0.1,
        max_groups: int = 100,
        codec_block_size: int = 64,     # NQKVCodec 내부 양자화 블록 크기
    ) -> None:
        super().__init__(block_size=block_size, diff_threshold=diff_threshold, max_groups=max_groups)
        self.codec = NQKVCodec(block_size=codec_block_size)
        # 압축된 마스터 저장소 (DiffAwareSegmentStore._masters는 MasterEntry용이므로 별도)
        self._compressed_masters: dict = {}  # group_id → CompressedMasterEntry

    def register_master(self, master_kv: torch.Tensor, group_id: str) -> None:
        """마스터 KV를 INT4로 압축해 저장.

        의사코드:
            indices, mu, sigma = self.codec.encode(master_kv)
            self._compressed_masters[group_id] = CompressedMasterEntry(
                indices=indices, mu=mu, sigma=sigma,
                original_shape=master_kv.shape, group_id=group_id
            )
            # LRU 관리는 _masters를 더미 엔트리로 유지
            super()._register_lru_only(group_id)
        """
        ...

    def _get_master_kv(self, group_id: str) -> Optional[torch.Tensor]:
        """INT4 역양자화 후 FP16 마스터 KV 반환 (override).

        의사코드:
            entry = self._compressed_masters.get(group_id)
            if entry is None:
                return None
            return self.codec.decode(entry.indices, entry.mu, entry.sigma, entry.original_shape)
        """
        ...

    def evict(self) -> int:
        """가장 오래된 LRU 그룹 퇴거. 압축 마스터 포함."""
        ...

    def memory_bytes(self) -> int:
        """압축 마스터 bytes + 차분 bytes 합산."""
        ...
```

---

### 4. FireQCodec — RoPE-인식 2단계 아웃라이어 평활화 (Activity C-2)

**파일**: `src/cache/fireq_codec.py`

**핵심 아이디어**:
- Stage 1 (pre-RoPE): RoPE 회전 쌍 (채널 i, 채널 i+d/2)의 분산을 동등하게 스케일링해 회전 전 이상치 균형을 맞춘다. 스케일 팩터는 캘리브레이션 배치에서 1회 추정한다.
- Stage 2 (post-RoPE): RoPE 적용 후 여전히 이상치로 남은 채널(|K_rotated| > 3σ)을 표적 스케일링한다.
- Key INT4 + Value FP8 혼합 정밀도 (FireQ 원논문 설계).
- CPU 환경: NumPy 기반 참조 구현으로 정확도 검증. GPU 환경: Triton/CUDA 커널 확장 가능.

**의사코드**:

```python
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import numpy as np

from src.cache.compression import CompressionCodec


class FireQCodec(CompressionCodec):
    """RoPE-인식 2단계 아웃라이어 평활화 + INT4+FP8 혼합 양자화 코덱 (Activity C-2).

    Key: INT4 (pre-RoPE 채널 쌍 정규화 + post-RoPE 이상치 마스크 적용)
    Value: FP8 (torch.float8_e4m3fn 또는 CPU fallback float16)
    """

    def __init__(
        self,
        n_heads: int = 12,
        d_head: int = 64,
        outlier_threshold_sigma: float = 3.0,  # post-RoPE 이상치 기준 (σ 배수)
        calib_scale_dir: Optional[str] = None,  # 사전 캘리브레이션 스케일 저장 경로
    ) -> None:
        self.n_heads = n_heads
        self.d_head = d_head
        self.outlier_threshold_sigma = outlier_threshold_sigma
        self.calib_scale_dir = Path(calib_scale_dir) if calib_scale_dir else None
        # 레이어별 pre-RoPE 채널 쌍 스케일 팩터: {layer_idx: Tensor(n_heads, d_head//2)}
        self._pre_rope_scales: Dict[int, torch.Tensor] = {}
        # 레이어별 post-RoPE 이상치 채널 마스크: {layer_idx: Tensor(n_heads, d_head) bool}
        self._outlier_masks: Dict[int, torch.Tensor] = {}

    def calibrate(
        self,
        calib_kvs: list,   # List[Tuple[kv_tensor, layer_idx]]: 캘리브레이션 KV 샘플
        min_samples: int = 10,
    ) -> None:
        """캘리브레이션 배치에서 레이어별 pre-RoPE 스케일 팩터와 post-RoPE 이상치 마스크 추정.

        Stage 1 스케일 팩터 계산:
            for layer_idx, key_tensors in grouped_by_layer(calib_kvs):
                K: Tensor(batch, n_heads, seq_len, d_head)
                # 채널 쌍 (i, i + d_head//2)의 분산 비율
                var_first  = K[..., :d_head//2].var(dim=(0,2))   # (n_heads, d_head//2)
                var_second = K[..., d_head//2:].var(dim=(0,2))   # (n_heads, d_head//2)
                s = (var_first / var_second.clamp(min=1e-8)).sqrt()  # (n_heads, d_head//2)
                _pre_rope_scales[layer_idx] = s

        Stage 2 이상치 마스크 추정 (RoPE 적용 후 샘플에서):
            for layer_idx, key_rope_tensors in grouped_by_layer(calib_kvs):
                K_rope: Tensor(batch, n_heads, seq_len, d_head)
                channel_std = K_rope.std(dim=(0,2))          # (n_heads, d_head)
                channel_max = K_rope.abs().max(dim=(0,2))[0] # (n_heads, d_head)
                outlier_mask = channel_max > outlier_threshold_sigma × channel_std
                _outlier_masks[layer_idx] = outlier_mask

        calib_scale_dir가 지정된 경우 레이어별 스케일을 JSON/PT 파일로 저장.
        """
        ...

    def encode(
        self,
        kv: torch.Tensor,          # (n_heads, seq_len, d_head) 또는 (batch, n_heads, seq_len, d_head)
        layer_idx: int,
        rope_applied: bool = True,  # True: post-RoPE KV 입력 (일반 경우)
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """KV 텐서 → (key_int4, value_fp8, meta) 반환.

        Stage 1 (rope_applied=False일 때): pre-RoPE 채널 쌍 스케일 적용
            scales = _pre_rope_scales.get(layer_idx)
            if scales is not None:
                K[..., :d//2] /= scales           # 채널 i 스케일 다운
                K[..., d//2:] *= scales           # 채널 i+d/2 스케일 업

        Stage 2 (rope_applied=True일 때): post-RoPE 이상치 채널 표적 스케일링
            mask = _outlier_masks.get(layer_idx)
            if mask is not None:
                channel_max = K.abs().max(dim=-2)[0]  # (n_heads, d_head)
                target_scale = channel_max.clamp(min=1.0)
                K = K / target_scale.unsqueeze(-2)    # 이상치 채널만 스케일 다운
                meta["outlier_scale"] = target_scale  # 역양자화용

        Key INT4 양자화:
            key_int4 = block_quantize_int4(K, block_size=64)

        Value FP8 양자화:
            value_fp8 = V.to(torch.float8_e4m3fn) if cuda else V.half()  # CPU fallback

        return key_int4, value_fp8, meta
        """
        ...

    def decode(
        self,
        key_int4: torch.Tensor,
        value_fp8: torch.Tensor,
        meta: dict,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(key_int4, value_fp8, meta) → (K_fp16, V_fp16) 복원.

        Key 역양자화:
            K = dequantize_int4(key_int4)               # (n_heads, seq_len, d_head)
            # Stage 2 역변환: 이상치 스케일 복원
            if "outlier_scale" in meta:
                K = K × meta["outlier_scale"].unsqueeze(-2)
            # Stage 1 역변환: 채널 쌍 스케일 복원
            scales = _pre_rope_scales.get(layer_idx)
            if scales is not None:
                K[..., :d//2] *= scales
                K[..., d//2:] /= scales

        Value 역양자화:
            V = value_fp8.float() if cuda else value_fp8.float()

        return K.half(), V.half()
        """
        ...

    def load_calibration(self, layer_idx: int) -> bool:
        """저장된 캘리브레이션 스케일/마스크 로드. 성공 시 True."""
        ...
```

---

## Activity C — Accuracy Preservation 검증 계획

**이 섹션은 Activity C 포함으로 인해 반드시 완성한다.
검증 계획 없이 Spec.md를 완성하지 않는다.**

### 설계상 accuracy-preserving 근거

**NQKVCodec (C-1)**:
- KV 캐시 K/V 벡터는 LayerNorm 직후 생성되므로 정규분포에 근사한다는 경험적 관찰이 성립한다.
- 블록-분위수 방식이 정보이론적으로 최적 양자화 오류(최솟값)를 달성한다. 동일 비트폭에서 균등 양자화보다 오류가 적다.
- 블록 내 분포 파라미터(μ, σ)가 채널별 이상치를 흡수하므로 별도 이상치 처리가 불필요하다.

**FireQCodec (C-2)**:
- pre-RoPE 채널 쌍 정규화가 RoPE 회전으로 증폭된 이상치를 회전 전에 제거한다.
- post-RoPE 표적 스케일링이 잔여 이상치를 채널 단위로 처리해 INT4 양자화 오류 분포를 정규분포에 근접시킨다.
- 2단계 평활화 후 INT4 양자화 오류 곡선이 정규화 전 대비 현저히 감소함이 FireQ 논문에서 Llama2-7B/Llama3-8B 실증되었다.

**CompressedDiffStore (Cross-1)**:
- 차분 블록을 FP16 원본으로 유지해 에이전트별 특수성 정보 손실이 없다.
- 마스터 INT4 압축 오류가 모든 에이전트에 공통으로 적용되므로, 차분이 에이전트 고유 정보를 완전히 보존한다.

### perplexity 측정 계획

- **데이터셋**: WikiText-2 (wikitext-2-raw-v1, 표준 test 분할)
- **모델**: GPT-2 (소형, 12레이어, 117M) — CPU 실행 가능 표준 모델
- **측정 방법**: stride=512, max_length=1024 슬라이딩 윈도우 perplexity
- **비교 조건**: NQKVCodec OFF (FP16 baseline) vs ON (INT4 압축)
- **허용 오차**: `|PPL_nqkv - PPL_baseline| / PPL_baseline ≤ 0.01` (±1% 이내)
- **실측 스크립트**: `experiments/run_perplexity_nqkv.py`
  - GPT-2 모델 로드 (transformers 라이브러리)
  - WikiText-2 토크나이즈
  - KV 캐시 후킹: NQKVCodec encode/decode를 GPT-2 어텐션 레이어에 적용
  - perplexity 계산 및 `results/nqkv_perplexity/metrics.json` 저장
  - ±1% 기준 통과 여부 자동 판정 출력

```python
# experiments/run_perplexity_nqkv.py 핵심 구조
def compute_perplexity(model, tokenizer, dataset, codec=None) -> float:
    """stride 슬라이딩 윈도우 perplexity 계산.
    codec이 None이면 FP16 baseline, 아니면 NQKVCodec encode/decode 적용.
    """
    ...

def main():
    codec = NQKVCodec(block_size=64)
    ppl_baseline = compute_perplexity(model, tokenizer, dataset, codec=None)
    ppl_nqkv     = compute_perplexity(model, tokenizer, dataset, codec=codec)
    delta_pct = abs(ppl_nqkv - ppl_baseline) / ppl_baseline * 100
    result = {
        "ppl_baseline": ppl_baseline,
        "ppl_nqkv": ppl_nqkv,
        "delta_pct": delta_pct,
        "pass": delta_pct <= 1.0,
    }
    json.dump(result, open("results/nqkv_perplexity/metrics.json", "w"))
    print(f"PASS={result['pass']}  delta={delta_pct:.3f}%")
```

### 태스크 정확도 측정 계획

- **벤치마크 1**: WikiText-2 perplexity (위 실측 스크립트로 직접 측정, 허용 오차 ±1%)
- **벤치마크 2 (단위 테스트 proxy)**: encode→decode 왕복 오류(RMSE/MAE) — 랜덤 정규분포 KV 텐서 기준 ≤ 0.05
- **벤치마크 3 (단위 테스트 proxy)**: 고중요도 채널 상위 10% 보존율 — encode→decode 왕복 후 채널 순위 상관계수(Spearman ρ) ≥ 0.95
- **FireQCodec 검증**: WikiText-2 perplexity ±1% (NQKVCodec 동일 스크립트 활용, 모델 후킹 방식 동일)
- **허용 오차**: 각 벤치마크 절대 정확도 변화 ±1% 이내

### 검증 테스트 파일: `tests/unit/test_nqkv_accuracy.py`

```python
def test_encode_decode_roundtrip_rmse():
    # torch.manual_seed(42)
    # kv = torch.randn(4, 64, 64)  # (n_heads, seq_len, d_head) 정규분포
    # codec = NQKVCodec(block_size=64)
    # indices, mu, sigma = codec.encode(kv)
    # kv_restored = codec.decode(indices, mu, sigma, kv.shape)
    # rmse = (kv - kv_restored).pow(2).mean().sqrt()
    # assert rmse.item() <= 0.05

def test_encode_decode_preserves_shape():
    # 원본과 복원 텐서의 shape 일치

def test_nf4_values_count():
    # len(NQKVCodec().nf4_values) == 14

def test_block_size_64_default():
    # NQKVCodec().block_size == 64

def test_compression_ratio_approx_4x():
    # FP16 대비 압축률 ≥ 3.5× (INT4 + (μ,σ) 오버헤드 포함)

def test_mu_sigma_stored_fp16():
    # encode() 반환 mu, sigma dtype == torch.float16

def test_channel_rank_preserved():
    # 채널별 norm 상위 10% 순위가 encode→decode 후 Spearman ρ ≥ 0.95

def test_perplexity_delta_within_1pct():
    # experiments/run_perplexity_nqkv.py 로직을 인라인으로 실행
    # GPT-2 small + WikiText-2 일부 샘플(10개 문장)로 proxy perplexity 계산
    # |PPL_nqkv - PPL_baseline| / PPL_baseline <= 0.01
    # (실제 전체 WikiText-2 측정은 experiments/run_perplexity_nqkv.py에서 수행)

def test_non_normal_distribution_still_bounded():
    # 균등분포 KV 텐서로 encode/decode → RMSE가 0.1 이내 (최악 케이스 상한)

def test_encode_non_multiple_block_size():
    # D=100 (블록 크기 64의 배수 아님) → 패딩 후 정상 encode, decode 시 원래 shape 복원

def test_cpu_only_no_cuda_required():
    # torch.cuda.is_available() == False인 환경에서 encode/decode 정상 동작
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-05.yaml
experiment: "2026-05-05-cross1-bc-diff-compressed"
date: "2026-05-05"
activities: [B, C]

cache:
  type: CompressedDiffStore      # Cross-1 통합: DiffAwareSegmentStore + NQKVCodec
  block_size: 64                 # 차분 계산 및 양자화 공통 블록 크기
  diff_threshold: 0.1            # 블록 L2 거리 임계값 (이하이면 마스터 참조)
  max_groups: 100                # 최대 그룹(마스터) 수

compression:
  nqkv:
    type: NQKVCodec
    block_size: 64               # 양자화 블록 크기 (원소 수)
    # nf4_values: null           # null이면 기본 Normal Float 분위점 테이블 사용
  fireq:
    type: FireQCodec
    n_heads: 12                  # 모델 헤드 수 (GPT-2 기준)
    d_head: 64                   # 헤드 차원
    outlier_threshold_sigma: 3.0 # post-RoPE 이상치 기준 σ 배수
    calib_scale_dir: "results/fireq_calib"  # 캘리브레이션 스케일 저장 경로

diff_store:
  type: DiffAwareSegmentStore    # 독립 사용 시 (CompressedDiffStore 미사용)
  block_size: 64
  diff_threshold: 0.1
  max_groups: 100

metrics:
  target_noncontiguous_hit_rate: 0.30     # 전체 히트의 30% 이상 비연속 (§3)
  target_kv_memory_reduction: 0.30        # 베이스라인 대비 −30% 이상 (§4)
  target_agent_memory_reduction: 0.85     # 독립 저장 대비 −85% 이상 (차분 저장)
  target_throughput_gain: 0.20            # 베이스라인 대비 +20% (§1)
  max_perplexity_delta_pct: 1.0           # ±1% perplexity (§4 필수)
  max_task_accuracy_delta_pct: 1.0        # ±1% 태스크 정확도 (§4 필수)
  max_encode_decode_ttft_overhead_pct: 10.0  # +10% 이내 압축 오버헤드 (§4)
  cross_throughput_gain_vs_single: 0.05   # 복합 처리량 단일 대비 +5% (§5)
  cross_memory_reduction_vs_single: 0.10  # 복합 메모리 단일 대비 −10% (§5)

accuracy_benchmarks:
  - name: nqkv_perplexity_wikitext2
    script: experiments/run_perplexity_nqkv.py
    metric: delta_pct
    threshold: 1.0            # ≤ 1.0%
    output: results/nqkv_perplexity/metrics.json
  - name: nqkv_roundtrip_rmse
    metric: encode_decode_rmse
    threshold: 0.05           # ≤ 0.05 (정규분포 KV 기준)
  - name: channel_rank_spearman
    metric: spearman_rho
    threshold: 0.95           # ≥ 0.95
  - name: fireq_outlier_reduction
    metric: post_smoothing_outlier_pct
    threshold: 0.05           # 이상치 채널 5% 이내 (post-smoothing)

perplexity_measurement:
  model: gpt2
  dataset: wikitext-2-raw-v1
  split: test
  stride: 512
  max_length: 1024
  output_dir: results/nqkv_perplexity
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_nqkv_accuracy.py` — NQKVCodec encode/decode 왕복 + perplexity 직접 측정 (Activity C accuracy 필수)
- [ ] `tests/unit/test_diff_aware_store.py` — DiffAwareSegmentStore 마스터+차분 저장/복원 왕복 테스트
- [ ] `tests/unit/test_fireq_codec.py` — FireQCodec RoPE 이상치 처리 검증
- [ ] `tests/integration/test_cross_bc_diff_compressed.py` — CompressedDiffStore 통합 테스트
- [ ] 기존 단위·통합 테스트 전부 회귀 없이 통과

### test_diff_aware_store.py 필수 테스트 케이스

```python
def test_register_master_and_get():
    # register_master(master_kv, "group1") → get("master:group1") 반환 동일 tensor

def test_put_agent_kv_identical_returns_no_diff():
    # agent_kv == master_kv → diff_blocks 비어 있음, master_ref_blocks에 모든 블록 포함

def test_put_agent_kv_different_stores_diff():
    # agent_kv != master_kv (L2 > threshold) → diff_blocks에 해당 블록 저장

def test_get_agent_kv_restores_original():
    # register_master + put_agent_kv → get_agent_kv → torch.allclose(original, restored)

def test_evict_removes_group_and_all_agents():
    # 그룹 LRU 퇴거 시 마스터 + 모든 에이전트 차분 삭제

def test_diff_threshold_boundary():
    # L2 거리 == diff_threshold → 마스터 참조 (미만 포함)
    # L2 거리 > diff_threshold → 차분 저장

def test_max_groups_triggers_eviction():
    # max_groups=2, 3번째 그룹 등록 → 가장 오래된 그룹 퇴거

def test_lru_order_on_access():
    # 오래된 그룹에 get() → LRU 순서 갱신 → 다음 evict()에서 두 번째 오래된 그룹 퇴거

def test_diff_hit_stats_structure():
    # diff_hit_stats() 반환값에 diff_hit_rate, master_hit_rate, n_groups 키 포함

def test_faiss_not_imported():
    # src/cache/diff_aware_store.py에 "faiss" 임포트 없음 (검색 공간 축소 방식)

def test_put_get_cachestore_interface():
    # put("agent:g1:a1", kv) → get("agent:g1:a1") 인터페이스 준수

def test_memory_bytes_accounts_diff_blocks():
    # 차분 블록 저장 후 memory_bytes() > 마스터 kv.nbytes

def test_reset_stats():
    # reset_stats() 후 hit_rate() == 0.0, _diff_hits == 0, _master_hits == 0

def test_cachestore_interface_all_methods():
    # put, get, evict, hit_rate, memory_bytes, reset_stats 모두 호출 가능 (회귀 없음)
```

### test_fireq_codec.py 필수 테스트 케이스

```python
def test_calibrate_produces_pre_rope_scales():
    # calibrate() 후 _pre_rope_scales[layer_idx].shape == (n_heads, d_head//2)

def test_calibrate_produces_outlier_masks():
    # calibrate() 후 _outlier_masks[layer_idx].dtype == torch.bool

def test_encode_returns_key_int8_value_fp16():
    # CPU 환경: encode() 반환 key_int4는 uint8 텐서, value 반환 가능

def test_encode_decode_roundtrip_bounded_error():
    # torch.manual_seed(42)
    # kv = torch.randn(12, 32, 64)  # 정규분포 KV
    # codec = FireQCodec(n_heads=12, d_head=64)
    # key_int4, val_fp8, meta = codec.encode(kv, layer_idx=0)
    # K_restored, V_restored = codec.decode(key_int4, val_fp8, meta, layer_idx=0)
    # rmse = (kv - K_restored).pow(2).mean().sqrt()
    # assert rmse.item() <= 0.1  # FireQCodec은 RoPE 평활화 목적, 약간 느슨한 기준

def test_stage1_scale_reduces_channel_variance_imbalance():
    # 불균형 분산을 가진 합성 KV → calibrate → encode 후 채널 쌍 분산 비율이 1.0에 근접

def test_stage2_outlier_channel_smoothed():
    # 특정 채널에 인위적 이상치 삽입 → encode 후 해당 채널 최대값이 threshold 이하

def test_outlier_mask_bool_type():
    # _outlier_masks 값이 모두 bool dtype

def test_pre_rope_scales_positive():
    # _pre_rope_scales 값이 모두 양수

def test_encode_without_calibration_still_works():
    # 캘리브레이션 없이 encode() → _pre_rope_scales, _outlier_masks가 비어 있어도 오류 없음

def test_cpu_fallback_no_cuda():
    # torch.cuda.is_available() == False → encode() 정상 동작 (float16 fallback)

def test_load_calibration_from_file():
    # calibrate() → save → load_calibration() → 스케일 복원
```

### test_cross_bc_diff_compressed.py 필수 통합 테스트 케이스

```python
def test_compressed_diff_store_register_and_get():
    # CompressedDiffStore.register_master(kv, "g1") → get("agent:g1:a1") 호출 전
    # 마스터 INT4 저장 확인 (memory_bytes < 원본 FP16 nbytes)

def test_master_stored_as_int4():
    # register_master 후 _compressed_masters["g1"] 존재
    # indices dtype == torch.uint8

def test_agent_kv_roundtrip_with_compression():
    # register_master → put_agent_kv → get_agent_kv
    # torch.allclose(original_agent_kv, restored, atol=0.2)  # INT4 오류 허용

def test_diff_blocks_remain_fp16():
    # put_agent_kv 후 diff_entry.diff_blocks 값들의 dtype == torch.float16 또는 float32

def test_memory_reduction_vs_independent_storage():
    # 5개 에이전트, 공통 마스터 KV 1000 토큰
    # CompressedDiffStore memory_bytes < 5 × 1000 × d_head × 2(FP16) × 0.70
    # (최소 30% 절감 검증)

def test_noncontiguous_hit_rate_from_diff_store():
    # 여러 에이전트 put/get 후 diff_hit_stats()["diff_hit_rate"] + ["master_hit_rate"] ≥ 0.30

def test_evict_group_frees_compressed_master():
    # evict() 후 _compressed_masters에 해당 group_id 없음

def test_cachestore_interface_compliance():
    # put, get, evict, hit_rate, memory_bytes, reset_stats 모두 정상 동작

def test_end_to_end_pipeline():
    # NQKVCodec → CompressedDiffStore → DiffAwareSegmentStore 전체 파이프라인
    # register_master, put_agent_kv, get_agent_kv → 예외 없이 완료

def test_accuracy_preservation_after_compression():
    # 100개 랜덤 KV 텐서로 encode→store→retrieve→decode
    # 전체 RMSE ≤ 0.1 (INT4 압축 오류 허용 범위)
    # 고중요도 채널 보존율 검증: 상위 10% 채널 L2 norm 감소율 ≤ 20%
```

---

## 구현 시 주의사항

1. **CacheStore 인터페이스 준수**: `DiffAwareSegmentStore`와 `CompressedDiffStore`는 `CacheStore`를 직접 상속하며 6개 추상 메서드(`put`, `get`, `evict`, `hit_rate`, `memory_bytes`, `reset_stats`)를 모두 구현해야 한다.

2. **NQKVCodec과 FireQCodec은 CacheStore 미상속**: 순수 코덱 레이어이므로 `CompressionCodec` 인터페이스(또는 독립 클래스)로 구현한다. `CacheStore`를 상속하지 않는다.

3. **FAISS 금지**: `DiffAwareSegmentStore`와 `CompressedDiffStore`는 FAISS 또는 유사 근사 최근접 이웃 라이브러리를 임포트하거나 사용하지 않는다. 검색 공간을 마스터 수(그룹 수)로 제한하는 것이 이 설계의 핵심이다.

4. **차분 블록 FP16 원본 유지**: `put_agent_kv()`에서 계산된 차분 블록은 INT4 압축 없이 FP16/FP32 원본으로 저장한다. 마스터만 INT4 압축 대상이다.

5. **블록 크기 통일**: `DiffAwareSegmentStore.block_size`(차분 계산 단위)와 `NQKVCodec.block_size`(양자화 단위)를 동일 값(기본 64)으로 설정해 블록 경계가 일치하도록 한다.

6. **모든 단위 테스트는 CPU 텐서 기준**: `torch.device("cpu")`. GPU 불필요. GPU 환경에서는 FireQCodec의 FP8 경로가 활성화되며 CPU에서는 FP16 fallback을 사용한다.

7. **시드 고정**: 모든 테스트에서 `torch.manual_seed(42)` 사용. 재현성 보장.

8. **훈련-무료 제약**: `NQKVCodec`, `DiffAwareSegmentStore`, `CompressedDiffStore`, `FireQCodec` 모두 학습 파라미터(`nn.Parameter`, `nn.Module`) 미포함. `FireQCodec.calibrate()`는 통계 추정이며 학습이 아니다.

9. **기존 파일 수정 금지**: 아래 파일들은 이번 사이클에서 절대 수정하지 않는다:
   - `src/cache/workload_ttl_cache.py`
   - `src/cache/redundancy_eviction.py`
   - `src/scheduler/dag_topology_scheduler.py`
   - `src/scheduler/dag_ttl_adjuster.py`
   - `src/cache/turbo_quant.py`
   - `src/cache/dhd_segment_cache.py`
   - `src/cache/speculative_fetcher.py`
   - `src/cache/sign_vq_segment.py`
   - `src/cache/leverage_compressor.py`
   - `src/cache/compression.py`
   - `src/cache/segmented.py`
   - `src/cache/contiguous.py`
   - `src/cache/tri_state_compressor.py`
   - `src/cache/compressed_segment.py`
   - `src/cache/segment_adapter.py`
   - `src/scheduler/cache_aware_scheduler.py`
   - `src/scheduler/dual_map_scheduler.py`
   - `src/scheduler/multi_node_scheduler.py`

10. **CompressionCodec 인터페이스 확인**: `NQKVCodec`과 `FireQCodec`이 구현하는 `CompressionCodec` 인터페이스가 `src/cache/compression.py`에 정의되어 있는지 먼저 확인한다. 미정의 시 동일 파일에 독립 클래스로 구현한다(기존 `compression.py` 수정 없이 `nqkv_codec.py`와 `fireq_codec.py` 내부에 인터페이스 정의 포함).

11. **experiments/ 스크립트 실행 가능성**: `run_perplexity_nqkv.py`와 `run_gpu_throughput.py`는 `python experiments/run_perplexity_nqkv.py` 형태로 직접 실행 가능해야 한다. 의존성(`transformers`, `datasets`)이 없는 CPU 환경에서는 graceful fallback(경고 출력 후 종료)을 제공한다.

12. **결과 파일 경로**: 
    - `results/nqkv_perplexity/metrics.json` — NQKVCodec perplexity 실측 결과
    - `results/fireq_throughput/metrics.json` — FireQCodec TTFT/TBT 실측 결과
    - `results/`는 git-ignored이므로 디렉토리 자동 생성 코드 포함(`os.makedirs(..., exist_ok=True)`).

---

## 완료 기준 (Definition of Done)

- [ ] `tests/unit/test_nqkv_accuracy.py` — 10개 테스트 케이스 전부 통과 (§4 Accuracy 필수)
- [ ] `tests/unit/test_diff_aware_store.py` — 14개 테스트 케이스 전부 통과
- [ ] `tests/unit/test_fireq_codec.py` — 10개 테스트 케이스 전부 통과
- [ ] `tests/integration/test_cross_bc_diff_compressed.py` — 10개 테스트 케이스 전부 통과
- [ ] 기존 단위·통합 테스트 전부 회귀 없이 통과
- [ ] `configs/experiments/2026-05-05.yaml` 존재 (§0 설정 YAML 필수)
- [ ] `experiments/run_perplexity_nqkv.py` 존재 및 실행 가능 (SUMMARY.md 미해결 항목 5 해소)
- [ ] `experiments/run_gpu_throughput.py` 존재 및 실행 가능 (SUMMARY.md 미해결 항목 1 해소)
- [ ] `evaluation_criteria.md` §4 Activity C (필수):
  - NQKVCodec perplexity ±1% 이내 (GPT-2 + WikiText-2 직접 실측)
  - encode/decode 왕복 RMSE ≤ 0.05 (정규분포 KV 기준)
  - KV Memory Reduction −30% 이상 (INT4 4× 압축)
  - Encode/Decode 추가 지연 TTFT +10% 이내
- [ ] `evaluation_criteria.md` §3 Activity B:
  - 비연속 히트율 ≥ 전체 히트의 30% (diff hit + master hit)
  - KV Memory Footprint 에이전트 그룹 시나리오 −85% 이상
- [ ] `evaluation_criteria.md` §5 Cross:
  - 복합 메모리 감소: 단일 Activity 대비 추가 −10% 이상
  - 복합 처리량 향상: 단일 Activity 대비 추가 +5% 이상
  - Accuracy 보존 복합 적용 후 ±1% 이내 (필수)
- [ ] FAISS 미사용 확인: diff_aware_store.py, compressed_diff_store.py에 faiss 임포트 없음
- [ ] 타입 힌트 모든 공개 함수·메서드에 존재 (§0 중간)
- [ ] 불필요한 추상화 없음: 이 Spec에 없는 클래스·인터페이스 도입 금지 (§0 낮음)
