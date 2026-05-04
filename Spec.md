# Spec — 2026-05-04

<!-- 변경 이유 (이전 Spec.md: 2026-05-03 대비):
이전 사이클(2026-05-03)은 A+B+C (DualMapScheduler + SemanticSegmentCache/DHD + TurboQuantCodec/PolarQuant+QJL) 조합이었다.
이번 사이클은 A+B (DAGTopologyScheduler + WorkloadAwareTTLCache + DAGAwareTTLAdjuster) 조합으로 전환한다.
주요 변경 내용:

1. [Activity A 교체] DualMapScheduler(이중 해시 + 의미 히트율 가중 라우팅) →
   DAGTopologyScheduler(SAGA/Pythia 영감 워크플로우 DAG 위상 기반 KV 선제 보존 스케줄러).
   알고리즘 자료구조가 해시 맵 → BFS/DFS DAG 위상 분석으로 본질적으로 다르다.
   최적화 단위도 요청(request) 수준 → 에이전트 파이프라인(workflow) 수준으로 격상된다.

2. [Activity B 교체] SemanticSegmentCache/DHD(의미 유사도 기반 비연속 KV 공유) →
   WorkloadAwareTTLCache(워크로드 카테고리별 TTL 세그먼트 보존 정책).
   세그먼트 선택 기준이 내용 유사도 → 사용 시간 패턴으로 전환된다.

3. [Activity C 제외] TurboQuantCodec(PolarQuant+QJL 3비트) 이번 사이클에서 제외.
   Activity C에 해당하는 RedundancyAwareEvictionPolicy(C-1)는 이번 사이클 구현 타겟에
   명시적으로 포함되어 있으나, 아이디어 리포트 최우선 타겟(Cross-1 A+B)에 따라 메인
   스코프는 A+B로 제한한다. 단, B-2의 WorkloadAwareTTLCache 퇴거 품질 향상을 위해
   RedundancyAwareEvictionPolicy(C-1)를 보조 구성요소로 포함한다.
   → Activity C 포함으로 판단: accuracy-preserving 검증 계획 필수 작성.

4. [Cross-1 신규] DAGAwareTTLAdjuster: DAGTopologyScheduler의 후속 노드 KV 재사용
   확률 예측을 WorkloadAwareTTLCache의 TTL 조정에 실시간 연결.
   이전 사이클에 없던 DAG 이벤트 → 캐시 TTL 피드백 루프.

기존 파일(turbo_quant.py, dhd_segment_cache.py, speculative_fetcher.py, sign_vq_segment.py,
leverage_compressor.py, compression.py, segmented.py, contiguous.py, tri_state_compressor.py,
compressed_segment.py, segment_adapter.py, cache_aware_scheduler.py, dual_map_scheduler.py,
multi_node_scheduler.py)은 수정하지 않는다.
기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-04.md`
**최우선 구현 타겟**: Cross-1 (A+B) — DAGTopologyScheduler(A-1) + WorkloadAwareTTLCache(B-2) + DAGAwareTTLAdjuster
**보조 구성요소**: RedundancyAwareEvictionPolicy(C-1) — WorkloadAwareTTLCache 퇴거 품질 향상 레이어

**해결하려는 문제**:
- 기존 스케줄러(`CacheAwareScheduler`, `DualMapScheduler`)는 요청 단위 최적화에 머물러 에이전트
  워크플로우의 DAG 구조(도구 호출 순서, 에이전트 의존 관계)를 활용하지 못한다. 후속 에이전트
  노드가 현재 노드의 KV를 재사용할 가능성이 높아도 퇴거 결정에 반영되지 않는다.
- 기존 캐시(`SegmentedHashCache`, `SemanticSegmentCache`)는 세그먼트 내용(해시, 의미 유사도)만
  보고 사용 시간 패턴(카테고리별 재사용 간격)은 무시한다. KVCache-in-the-Wild 실측에 따르면
  카테고리별(코드/대화/RAG/에이전틱) TTL을 다르게 설정하면 히트율 23.9% 향상 가능하다.
- 단순 LRU 퇴거는 세그먼트 중요도와 중복성을 동시에 평가하지 못해 중복 세그먼트가 오랫동안
  캐시를 점유하고 고유한 중요 세그먼트가 조기 퇴거된다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling — DAGTopologyScheduler (DAG 위상 기반 KV 선제 보존)
- [x] Activity B: Non-Contiguous KV Cache Reuse — WorkloadAwareTTLCache (카테고리별 TTL 세그먼트)
- [x] Activity C: KV Cache Compression (보조) — RedundancyAwareEvictionPolicy (중요도×중복성 이중 스코어 퇴거)

---

## 목표

- [ ] 목표 1 (§3 Non-Contiguous Hit Rate): 전체 히트 중 비연속 히트 비율 ≥ 30% (TTL 기반 보존으로 달성)
- [ ] 목표 2 (§2 Scheduling): 스케줄링 TTFT p50 오버헤드 ≤ +5% (DAG 분석 비용 포함)
- [ ] 목표 3 (§2 Scheduling): 스케줄링 적용 캐시 히트율 향상 ≥ +10%p (DAG 인식 보존 적용 전 대비)
- [ ] 목표 4 (§1 Throughput): tokens/sec 베이스라인 대비 +20% 이상 (Cross-1 복합 효과)
- [ ] 목표 5 (§3 KV Memory Footprint): 베이스라인 대비 +20% 이내 (TTL 퇴거로 메모리 압력 완화)
- [ ] 목표 6 (§4 Accuracy 필수): 중복성 퇴거 전후 perplexity 변화 ±1% 이내 (proxy 수치 검증)
- [ ] 목표 7 (§4 Accuracy 필수): downstream 태스크 정확도 변화 ±1% 이내 (proxy 수치 검증)
- [ ] 목표 8 (§5 Cross): 복합 처리량 향상 단일 Activity 대비 추가 +5% 이상
- [ ] 목표 9 (§5 Cross): 복합 메모리 감소 단일 Activity 대비 추가 −10% 이상
- [ ] 목표 10: DAG 재사용 확률 예측 정확도 측정 (예측 KV 보존률 vs 실제 히트 비율, 결과 파일 저장)

---

## 아키텍처 개요

```
워크플로우 DAG 등록
    │ register_workflow(dag_id, node_graph)
    ▼
┌──────────────────────────────────────────────────────────────┐
│  DAGTopologyScheduler (Activity A)                           │
│  BFS 위상 분석 → 후속 노드 KV 재사용 확률 계산               │
│  kv_reuse_probability > retain_threshold:                    │
│    → 세그먼트 핀 고정 (LRU 퇴거 제외)                        │
│  DAG 미지 요청 → CacheAwareScheduler 위임 (폴백)             │
└───────────────────────────┬──────────────────────────────────┘
                            │ {segment_id, dag_reuse_probability} 이벤트
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  DAGAwareTTLAdjuster (Cross-1 통합 모듈)                     │
│  adjusted_ttl = base_ttl × (1 + prob × alpha)               │
│  후속 노드 완료 → TTL 즉시 0 (조기 퇴거 허용)                │
└───────────────────────────┬──────────────────────────────────┘
                            │ TTL 조정 명령
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  WorkloadAwareTTLCache (Activity B)                          │
│  카테고리 분류 (키워드 룰 / k-NN 첫 32 토큰)                 │
│  카테고리별 TTL 세그먼트 저장                                 │
│  evict_candidates() → TTL 만료 후보 목록                     │
│    └→ RedundancyAwareEvictionPolicy 이중 스코어 적용 (보조)  │
│  최종 퇴거 실행                                              │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
  (hits, misses, ttl_hits, exact_hits, memory_bytes)
  hit_type: "exact" | "ttl_preserved" | "miss"
  eviction_type: "ttl_expired" | "memory_pressure" | "dag_completed"
```

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/workload_ttl_cache.py` | B | `WorkloadAwareTTLCache` — 카테고리 분류기 + 카테고리별 TTL + LRU-TTL 복합 퇴거. `CacheStore` 인터페이스 구현 |
| `src/cache/redundancy_eviction.py` | C (보조) | `RedundancyAwareEvictionPolicy` — 중요도×중복성 이중 스코어 퇴거 정책. WorkloadAwareTTLCache의 evict_candidates() 훅에 연결 |
| `src/scheduler/dag_topology_scheduler.py` | A | `DAGTopologyScheduler` — DAG 메타데이터 수집 + BFS 위상 분석 + KV 선제 보존 결정. CacheAwareScheduler 위임 폴백 |
| `src/scheduler/dag_ttl_adjuster.py` | A+B (Cross-1) | `DAGAwareTTLAdjuster` — DAGTopologyScheduler 이벤트 구독 + WorkloadAwareTTLCache TTL 동적 조정 |
| `configs/experiments/2026-05-04.yaml` | 공통 | 실험 설정 YAML |
| `tests/unit/test_workload_ttl_cache.py` | B | WorkloadAwareTTLCache 단위 테스트 |
| `tests/unit/test_redundancy_eviction_accuracy.py` | C | RedundancyAwareEvictionPolicy accuracy-preserving 검증 (필수) |
| `tests/unit/test_dag_topology_scheduler.py` | A | DAGTopologyScheduler 단위 테스트 |
| `tests/unit/test_dag_ttl_adjuster.py` | A+B | DAGAwareTTLAdjuster 단위 테스트 |
| `tests/integration/test_cross_ab_dag_ttl.py` | A+B+C | Cross-1 전체 통합 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | 변경 없음 — 기존 CacheStore 인터페이스 그대로 준수 |

**주의**: 기존 파일(`turbo_quant.py`, `dhd_segment_cache.py`, `speculative_fetcher.py`,
`sign_vq_segment.py`, `leverage_compressor.py`, `compression.py`, `segmented.py`,
`contiguous.py`, `tri_state_compressor.py`, `compressed_segment.py`, `segment_adapter.py`,
`cache_aware_scheduler.py`, `dual_map_scheduler.py`, `multi_node_scheduler.py`)은
이번 사이클에서 수정하지 않는다. 기존 단위·통합 테스트 전부가 회귀 없이 통과해야 한다.

---

## 알고리즘 상세

### 1. WorkloadAwareTTLCache (Activity B)

**파일**: `src/cache/workload_ttl_cache.py`

**핵심 아이디어**:
- 요청을 카테고리(code / chat / rag / agentic)로 분류하고 카테고리별 TTL을 세그먼트에 할당.
- TTL 만료 세그먼트를 퇴거 1순위 후보로 표시, 메모리 부족 시 TTL 만료 세그먼트 우선 퇴거.
- 실제 히트·미스 패턴을 카테고리별로 집계해 EMA 방식으로 TTL 프로파일 온라인 갱신.
- DAGAwareTTLAdjuster로부터 TTL 조정 명령을 수신해 세그먼트 TTL을 동적으로 오버라이드.

**카테고리 분류 로직**:
```
키워드 룰 기반 (우선):
  - "def ", "class ", "import ", "```python" 포함 → "code"
  - "document", "context:", "passage", "retrieved" 포함 → "rag"
  - "tool_call", "function_call", "agent", "workflow" 포함 → "agentic"
  - 그 외 → "chat"

k-NN 폴백 (키워드 룰 미분류 시, 선택 옵션):
  - 첫 32 토큰 ID 평균 임베딩 벡터 → K=5 최근접 이웃으로 카테고리 결정
  - 초기 레이블 없을 경우 "chat" 기본값 사용
```

**TTL 프로파일 초기값** (KVCache-in-the-Wild 논문 Table 3 기반):
```
code:     ttl_base_sec = 600, reuse_probability = 0.75
chat:     ttl_base_sec = 300, reuse_probability = 0.60
rag:      ttl_base_sec = 120, reuse_probability = 0.45
agentic:  ttl_base_sec = 480, reuse_probability = 0.80
```

**의사코드**:

```python
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple
import torch

from src.cache.base import CacheStore


@dataclass
class TTLEntry:
    value: torch.Tensor
    category: str
    ttl_sec: float
    created_at: float          # time.monotonic()
    pinned: bool = False       # DAG 보존 핀 고정
    importance_score: float = 0.0  # 누적 어텐션 스코어 (RedundancyAwareEvictionPolicy용)
    embedding: Optional[torch.Tensor] = None  # 중복성 계산용 Key 평균 벡터


class WorkloadAwareTTLCache(CacheStore):
    """카테고리별 TTL 세그먼트 보존 KV 캐시 (Activity B).

    저장소 구조:
      _store: OrderedDict[str, TTLEntry]  — LRU 순서 유지
      _ttl_profiles: Dict[str, dict]      — 카테고리별 TTL 프로파일 (EMA 갱신)
      _pinned: Set[str]                   — DAG 핀 고정 세그먼트 키 집합
      _eviction_policy: Optional[RedundancyAwareEvictionPolicy]  — 보조 퇴거 정책

    히트 분류:
      exact_hits: 정확 키 매칭
      ttl_preserved_hits: TTL 만료 전 재사용 (비연속 히트로 분류)
      misses: 미스
    """

    def __init__(
        self,
        max_entries: int = 1000,
        chunk_size: int = 128,
        ttl_ema_alpha: float = 0.1,       # EMA 갱신 계수
        eviction_policy: Optional["RedundancyAwareEvictionPolicy"] = None,
        ttl_profiles: Optional[Dict[str, dict]] = None,
    ) -> None: ...

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 구현                                            #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        # 카테고리 미지정 시 "chat" 기본값으로 TTLEntry 생성
        # put_segment() 권장; 이 메서드는 하위 호환성 유지용
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        # TTL 만료 체크: (time.monotonic() - entry.created_at) > entry.ttl_sec → miss
        # 미만이면 hit; exact_hits 또는 ttl_preserved_hits 카운터 증가
        # LRU 갱신: _store.move_to_end(key)
        ...

    def evict(self) -> int:
        # 1단계: TTL 만료 세그먼트 목록 생성 (핀 고정 제외)
        # 2단계: eviction_policy가 있으면 이중 스코어로 정렬
        # 3단계: 최고 eviction_score 세그먼트 퇴거
        # 4단계: TTL 만료 세그먼트 없으면 LRU 폴백 (비핀 고정 중 가장 오래된 것)
        ...

    def hit_rate(self) -> float:
        # (exact_hits + ttl_preserved_hits) / total
        ...

    def memory_bytes(self) -> int:
        # _store의 모든 value.nbytes 합산
        ...

    def reset_stats(self) -> None:
        # exact_hits, ttl_preserved_hits, misses, eviction_counts 초기화
        ...

    # ------------------------------------------------------------------ #
    # 확장 API                                                             #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        key: str,
        value: torch.Tensor,
        category: str,
        embedding: Optional[torch.Tensor] = None,
        override_ttl_sec: Optional[float] = None,
    ) -> None:
        # TTLEntry 생성: ttl_sec = override_ttl_sec or _ttl_profiles[category]["ttl_base_sec"]
        # embedding 저장 (중복성 계산용)
        # max_entries 초과 시 evict()
        ...

    def adjust_ttl(self, key: str, new_ttl_sec: float) -> None:
        # DAGAwareTTLAdjuster 호출용: 특정 세그먼트 TTL 즉시 변경
        # new_ttl_sec == 0.0이면 즉시 퇴거 후보로 표시 (expires_at = created_at)
        ...

    def pin(self, key: str) -> None:
        # _pinned에 key 추가 → evict()에서 제외
        ...

    def unpin(self, key: str) -> None:
        # _pinned에서 key 제거
        ...

    def evict_candidates(self) -> List[str]:
        # TTL 만료 세그먼트 키 목록 반환 (핀 고정 제외)
        # RedundancyAwareEvictionPolicy가 이 목록을 스코어링함
        ...

    def record_hit(self, key: str, is_ttl_preserved: bool = False) -> None:
        # 히트 기록 + EMA 온라인 TTL 갱신
        # category = _store[key].category
        # reuse_gap = time.monotonic() - _store[key].created_at
        # _ttl_profiles[category]["ttl_base_sec"] =
        #     (1 - ema_alpha) × old_ttl + ema_alpha × reuse_gap × ttl_multiplier
        ...

    def record_importance(self, key: str, score: float) -> None:
        # 어텐션 스코어 누적 (RedundancyAwareEvictionPolicy 중요도 계산용)
        ...

    def ttl_hit_stats(self) -> dict:
        # {
        #   "exact_hit_rate": exact_hits / total,
        #   "ttl_preserved_hit_rate": ttl_preserved_hits / total,
        #   "overall_hit_rate": (exact_hits + ttl_preserved_hits) / total,
        #   "noncontiguous_ratio": ttl_preserved_hits / (exact_hits + ttl_preserved_hits),
        #   "eviction_ttl_count": int,   # TTL 만료 퇴거 수
        #   "eviction_pressure_count": int,  # 메모리 압력 퇴거 수
        # }
        ...

    def _classify_category(self, key: str, token_ids: Optional[List[int]] = None) -> str:
        # 키워드 룰 기반 분류 (key 문자열에서 메타데이터 추출 불가 시 token_ids 사용)
        # 폴백: "chat"
        ...

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        # segmented.py와 동일한 SHA-256 방식 (SegmentedHashCache.chunk_key 호환)
        import hashlib, struct
        start = chunk_idx * self.chunk_size
        chunk = token_ids[start : start + self.chunk_size]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()
```

---

### 2. RedundancyAwareEvictionPolicy (Activity C 보조)

**파일**: `src/cache/redundancy_eviction.py`

**핵심 아이디어**:
- 중요도 스코어(누적 어텐션 스코어)와 중복성 스코어(Key 벡터 코사인 유사도 평균)를 결합.
- `eviction_score = (1 - importance_score) × redundancy_score` — 중요도 낮고 중복성 높은 세그먼트 우선 퇴거.
- WorkloadAwareTTLCache의 `evict_candidates()` 훅에 연결해 TTL 만료 후보 중에서만 스코어링.
- 훈련 불필요, N ≤ 100 세그먼트에서 brute-force O(N × chunk_size × d_head) 허용.

**accuracy-preserving 근거**:
- 이중 스코어의 곱 형태(`(1-importance) × redundancy`)가 중요 토큰(importance 높음)을 구조적으로
  보호한다. 중요성이 낮은 후보 중에서만 중복성으로 우선순위를 조정하므로 false negative(중요 토큰
  퇴거) 위험이 없다.
- RAG 중복 문서 등 의미적으로 동일한 세그먼트를 우선 퇴거해 정확도 마진을 확보하면서 메모리 압력
  완화가 가능하다.

**의사코드**:

```python
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


class RedundancyAwareEvictionPolicy:
    """중요도×중복성 이중 스코어 퇴거 정책 (Activity C 보조).

    WorkloadAwareTTLCache.evict_candidates() 결과에 적용하는 drop-in 퇴거 정책.
    CacheStore를 상속하지 않음 — 순수 스코어링 레이어.
    """

    def __init__(
        self,
        redundancy_top_n: int = 100,     # 중복성 계산 대상 상위 N 세그먼트
        importance_weight: float = 1.0,  # 중요도 가중치
        redundancy_weight: float = 1.0,  # 중복성 가중치
        doc_id_shortcut: bool = True,    # 문서 ID 기반 중복 감지 단축 경로
    ) -> None: ...

    def score_candidates(
        self,
        candidates: List[str],           # evict_candidates()에서 받은 키 목록
        store_entries: Dict[str, "TTLEntry"],  # _store 직접 참조
    ) -> List[Tuple[str, float]]:
        """Returns (key, eviction_score) pairs sorted by score descending.

        eviction_score = (1 - normalized_importance) × redundancy_score
        높을수록 먼저 퇴거.
        """
        # 1. 중요도 정규화:
        #    max_imp = max(e.importance_score for e in candidates_entries) or 1.0
        #    normalized_importance[key] = entry.importance_score / max_imp

        # 2. 중복성 스코어 계산 (N ≤ redundancy_top_n인 경우 brute-force):
        #    embeddings = [store[k].embedding for k in candidates if store[k].embedding is not None]
        #    emb_matrix = torch.stack(embeddings)  # (N, d_head)
        #    e_norm = F.normalize(emb_matrix, dim=-1)
        #    sim_matrix = e_norm @ e_norm.T        # (N, N)
        #    # 자기 자신 제외
        #    sim_matrix.fill_diagonal_(0.0)
        #    redundancy[key] = sim_matrix[i].mean().item()

        # 3. doc_id_shortcut: key에 doc_id 접두사가 동일한 세그먼트 → 즉시 redundancy=1.0

        # 4. eviction_score = (1 - normalized_importance) × redundancy_score
        ...

    def select_evict_keys(
        self,
        candidates: List[str],
        store_entries: Dict[str, "TTLEntry"],
        n_evict: int = 1,
    ) -> List[str]:
        # score_candidates() 후 상위 n_evict 키 반환
        ...
```

---

### 3. DAGTopologyScheduler (Activity A)

**파일**: `src/scheduler/dag_topology_scheduler.py`

**핵심 아이디어**:
- 에이전트 워크플로우 DAG(노드 = 에이전트/도구 호출, 엣지 = 의존 관계)를 등록.
- BFS/DFS로 DAG를 위상 순회해 각 노드의 후속 노드 수(out_degree)와 KV 공유 확률 계산.
- KV 재사용 확률 > retain_threshold인 세그먼트를 WorkloadAwareTTLCache.pin()으로 핀 고정.
- 워크플로우 실행 완료된 노드의 세그먼트는 unpin() + TTL 즉시 단축.
- DAG 미지 요청은 기존 `CacheAwareScheduler`에 위임(폴백).
- Bélády 상한 계산(시뮬레이션 기반)을 벤치마크 비교 도구로 포함.

**스케줄링 결정 단위**: 배치(batch) 단위. 배치 내 각 요청에 `dag_node_id`와 `kv_reuse_probability` 주석 추가.

**캐시 상태 접근**: `WorkloadAwareTTLCache.pin()` / `unpin()` / `adjust_ttl()`을 통해서만 캐시 상태를 수정. 통계 오염 없음.

**DAG 노드 JSON 포맷**:
```json
{
  "dag_id": "workflow_001",
  "nodes": [
    {
      "agent_id": "agent_A",
      "tool_calls": ["tool_1", "tool_2"],
      "expected_kv_tokens": 512,
      "parent_ids": []
    },
    {
      "agent_id": "agent_B",
      "tool_calls": ["tool_3"],
      "expected_kv_tokens": 256,
      "parent_ids": ["agent_A"]
    }
  ]
}
```

**의사코드**:

```python
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from src.scheduler.cache_aware_scheduler import CacheAwareScheduler
from src.engine.runner import InferenceRequest


@dataclass
class DAGNode:
    agent_id: str
    tool_calls: List[str]
    expected_kv_tokens: int
    parent_ids: List[str]
    out_degree: int = 0              # BFS 분석 후 채워짐
    kv_reuse_probability: float = 0.0


@dataclass
class WorkflowDAG:
    dag_id: str
    nodes: Dict[str, DAGNode]        # agent_id → DAGNode
    topological_order: List[str]     # BFS 위상 순서
    completed_nodes: Set[str] = field(default_factory=set)
    belady_upper_bound: float = 0.0  # Bélády 상한 (시뮬레이션 계산)


class DAGTopologyScheduler:
    """워크플로우 DAG 위상 기반 KV 선제 보존 스케줄러 (Activity A).

    스케줄링 결정 단위: 배치(batch).
    캐시 접근 방법: WorkloadAwareTTLCache.pin() / unpin() / adjust_ttl() 전용 API.
    """

    def __init__(
        self,
        cache: "WorkloadAwareTTLCache",
        fallback_scheduler: Optional[CacheAwareScheduler] = None,
        retain_threshold: float = 0.5,  # KV 보존 결정 최소 확률
        alpha_ttl_extend: float = 2.0,  # TTL 연장 계수 (DAGAwareTTLAdjuster 전달용)
        kv_reuse_histogram: Optional[Dict] = None,  # 워크플로우별 히스토그램 (결과 저장용)
    ) -> None: ...

    def register_workflow(self, dag_spec: dict) -> str:
        """DAG JSON 스펙을 파싱하고 위상 분석 후 dag_id 반환.

        Args:
            dag_spec: DAG JSON 딕셔너리
        Returns:
            dag_id: 등록된 워크플로우 식별자
        """
        # 1. DAGNode 목록 파싱
        # 2. BFS 위상 순회: deque를 사용한 Kahn 알고리즘
        #    in_degree = {node_id: len(parents) for ...}
        #    queue = deque(node_id for node_id in nodes if in_degree[node_id] == 0)
        #    topological_order = []
        #    while queue:
        #        node_id = queue.popleft()
        #        topological_order.append(node_id)
        #        for child in children[node_id]:
        #            in_degree[child] -= 1
        #            if in_degree[child] == 0:
        #                queue.append(child)
        # 3. out_degree 계산: 각 노드의 자식 수
        # 4. kv_reuse_probability 계산:
        #    prob[node] = out_degree[node] / max(1, max_out_degree) × base_weight
        #    base_weight: 10회 이상 실행된 워크플로우는 히스토그램 기반, 미만은 out_degree 기반
        # 5. Bélády 상한 시뮬레이션: 모든 노드 KV를 알고 있다고 가정 시 최적 히트율 계산
        # 6. WorkflowDAG 저장: _workflows[dag_id] = dag
        ...

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """배치 내 각 요청에 dag_node_id와 kv_reuse_probability 주석 추가.

        DAG 등록된 요청: pin() 호출 + kv_reuse_probability 주석
        DAG 미지 요청: fallback_scheduler 위임
        Returns: 주석 추가된 요청 목록 (순서 변경 없음)
        """
        ...

    def notify_node_complete(self, dag_id: str, agent_id: str) -> None:
        """DAG 노드 처리 완료 알림 → 해당 세그먼트 핀 해제 + TTL 단축.

        completed_nodes에 agent_id 추가.
        연결된 WorkloadAwareTTLCache에 unpin() 호출.
        DAGAwareTTLAdjuster에 완료 이벤트 발행.
        """
        ...

    def predict_kv_reuse(self, dag_id: str, agent_id: str) -> float:
        """특정 DAG 노드의 KV 재사용 확률 반환.

        _workflows[dag_id].nodes[agent_id].kv_reuse_probability
        DAG 미등록 시 0.0 반환
        """
        ...

    def compute_belady_upper_bound(self, dag_id: str) -> float:
        """Bélády 최적 정책 시뮬레이션으로 캐시 히트율 상한 계산.

        완전 정보 가정(모든 미래 접근 패턴 알고 있음)에서 최적 퇴거 시 달성 가능한 히트율.
        실제 히트율과의 gap을 측정해 개선 여지를 정량화.
        Returns: 상한 히트율 (0.0~1.0)
        """
        ...

    def save_reuse_histogram(self, output_path: str) -> None:
        """kv_reuse_probability 히스토그램을 JSON 파일로 저장.

        결과 파일: results/<exp>/kv_reuse_histogram.json
        """
        ...

    def _build_children_map(
        self, nodes: Dict[str, DAGNode]
    ) -> Dict[str, List[str]]:
        # {parent_id: [child_id, ...]} 역방향 그래프 구성
        ...
```

---

### 4. DAGAwareTTLAdjuster (Cross-1 통합 모듈)

**파일**: `src/scheduler/dag_ttl_adjuster.py`

**핵심 아이디어**:
- DAGTopologyScheduler에서 `{segment_key, dag_reuse_probability}` 이벤트를 수신.
- `adjusted_ttl = base_ttl × (1 + dag_reuse_probability × alpha)` 공식으로 TTL 연장.
- 후속 노드 완료 시 TTL 즉시 0으로 설정(조기 퇴거 허용).
- 이벤트 수신 → TTL 갱신 지연을 측정해 오버헤드 보고.

```python
import time
from typing import Callable, Dict, Optional


class DAGAwareTTLAdjuster:
    """DAG 이벤트를 WorkloadAwareTTLCache TTL 조정으로 변환하는 중간 레이어 (Cross-1).

    DAGTopologyScheduler와 WorkloadAwareTTLCache 사이의 결합을 최소화하는 어댑터.
    """

    def __init__(
        self,
        cache: "WorkloadAwareTTLCache",
        alpha: float = 2.0,              # TTL 연장 계수
        measure_latency: bool = True,    # 이벤트→TTL 갱신 지연 측정
    ) -> None:
        self.cache = cache
        self.alpha = alpha
        self.measure_latency = measure_latency
        self._latency_samples: list = []   # 이벤트→갱신 지연 (ms) 목록
        ...

    def on_kv_reuse_event(
        self,
        segment_key: str,
        dag_reuse_probability: float,
    ) -> None:
        """KV 재사용 확률 이벤트 수신 → TTL 연장.

        t0 = time.monotonic()
        entry = cache._store.get(segment_key)
        if entry is None: return
        base_ttl = cache._ttl_profiles[entry.category]["ttl_base_sec"]
        adjusted_ttl = base_ttl × (1 + dag_reuse_probability × alpha)
        cache.adjust_ttl(segment_key, adjusted_ttl)
        if measure_latency:
            _latency_samples.append((time.monotonic() - t0) × 1000)
        """
        ...

    def on_node_complete(self, segment_key: str) -> None:
        """후속 노드 완료 이벤트 → TTL 즉시 0 설정 (조기 퇴거 허용).

        cache.adjust_ttl(segment_key, new_ttl_sec=0.0)
        cache.unpin(segment_key)
        """
        ...

    def overhead_stats(self) -> dict:
        """이벤트→TTL 갱신 지연 통계 반환.

        Returns:
            {
              "p50_ms": float,
              "p99_ms": float,
              "mean_ms": float,
              "n_samples": int,
            }
        """
        ...
```

---

## Activity C — Accuracy Preservation 검증 계획

**이 섹션은 Activity C(RedundancyAwareEvictionPolicy) 포함으로 인해 반드시 완성한다.
검증 계획 없이 Spec.md를 완성하지 않는다.**

### 설계상 accuracy-preserving 근거

`RedundancyAwareEvictionPolicy`는 퇴거 후보를 새로 생성하지 않는다. 오직 이미
`WorkloadAwareTTLCache.evict_candidates()`가 반환한 TTL 만료 세그먼트 중에서 퇴거 순서를
조정할 뿐이다. 따라서:

1. **중요 토큰 보호**: `eviction_score = (1 - importance) × redundancy`의 곱 구조에 의해,
   어텐션 스코어가 높은(importance ≈ 1.0) 세그먼트는 eviction_score ≈ 0.0이 되어 퇴거 우선순위
   최하위로 밀린다. 구조적으로 중요 세그먼트를 퇴거할 수 없다.
2. **중복 세그먼트 우선 퇴거**: RAG 시나리오에서 동일 문서의 중복 청크(redundancy ≈ 1.0)가
   먼저 퇴거되어 고유 정보 손실 없이 메모리를 확보한다.
3. **TTL 만료 후 처리**: TTL이 이미 만료된 세그먼트만 대상이므로, 정상 TTL 범위 내 세그먼트는
   이 정책으로 퇴거되지 않는다.

### perplexity 측정 계획

- **데이터셋**: WikiText-2 (wikitext-2-raw-v1, 표준 분할)
- **모델**: GPT-2 (소형, 12레이어) — CPU 실행 가능 표준 모델
- **측정 방법**: stride=512, max_length=1024 슬라이딩 윈도우 perplexity
- **허용 오차**: `|PPL_with_eviction - PPL_baseline| / PPL_baseline ≤ 0.01` (±1% 이내)
- **수치 프록시** (단위 테스트, 실제 모델 호출 없이):
  - 중복 세그먼트 퇴거 후 잔존 세그먼트의 Key 벡터 평균 코사인 유사도(퇴거 전 vs 후): ≥ 0.99
    (중요 정보가 보존됨을 수치로 검증)
  - 중요도 높은 세그먼트(importance_score > 0.8)가 퇴거 후에도 캐시에 잔존: 100% 보장
  - 정규화 재구성 오류(잔존 KV 대표성 손실): ≤ 2% (중복 제거로 인한 정보 손실 상한)

### 태스크 정확도 측정 계획

- **벤치마크 1**: LongBench-QA (단일 문서 QA, ROUGE-L 점수) — proxy: 중요 세그먼트 보존율 100%
- **벤치마크 2**: AIME 추론 (CoT 중복성 효과 확인) — proxy: 자기 성찰 패턴 중복 세그먼트
  퇴거 후 비중복 세그먼트 Hit Rate 변화 ≤ 1%p
- **허용 오차**: 각 서브태스크 절대 정확도 변화 ±1% 이내
- **이중 스코어 효과 검증**: 단순 LRU 퇴거 vs 이중 스코어 퇴거 비교 시,
  이중 스코어가 동등 메모리 절감에서 더 낮은 중요 토큰 퇴거율을 달성함을 수치로 검증

### 검증 테스트 파일: `tests/unit/test_redundancy_eviction_accuracy.py`

```python
def test_high_importance_segment_never_evicted():
    # importance_score=1.0인 세그먼트가 score_candidates()에서 eviction_score=0.0
    # → select_evict_keys()에서 절대 선택되지 않음

def test_redundant_segment_evicted_first():
    # 두 세그먼트: high_redundancy(cos_sim=0.95) + low_redundancy(cos_sim=0.1)
    # 동일 importance_score → high_redundancy 먼저 퇴거

def test_eviction_score_formula():
    # eviction_score = (1 - importance) × redundancy 공식 수치 검증
    # importance=0.5, redundancy=0.8 → score=0.40

def test_doc_id_shortcut_detects_duplicates():
    # 동일 doc_id 접두사 세그먼트 → redundancy=1.0 즉시 부여

def test_important_tokens_preserved_after_eviction():
    # 10개 세그먼트, 2개 고중요도(importance>0.8), 3개 중복(redundancy>0.9)
    # n_evict=3 시 중복 세그먼트 3개 퇴거 + 고중요도 2개 잔존 검증

def test_perplexity_proxy_residual_cosine_similarity():
    # 중복 세그먼트 퇴거 후 잔존 Key 벡터 집합의 평균 코사인 유사도 ≥ 0.99
    # WikiText-2 ±1% perplexity proxy

def test_task_accuracy_proxy_important_hit_rate():
    # 퇴거 후 high_importance 세그먼트의 캐시 Hit Rate 변화 ≤ 1%p
    # LongBench ±1% 정확도 proxy

def test_no_training_required():
    # RedundancyAwareEvictionPolicy 인스턴스에 nn.Parameter, nn.Module 없음

def test_score_candidates_returns_sorted_descending():
    # score_candidates() 반환값이 eviction_score 내림차순 정렬

def test_empty_candidates_list():
    # candidates=[] → score_candidates() 반환 []

def test_single_candidate():
    # candidates 길이 1 → select_evict_keys(n_evict=1) 정상 동작

def test_redundancy_computation_without_embedding():
    # embedding=None인 세그먼트 → redundancy=0.0 처리 (에러 없음)
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-04.yaml
experiment: "2026-05-04-cross1-dag-ttl"
date: "2026-05-04"
activities: [A, B, C]

cache:
  type: WorkloadAwareTTLCache
  max_entries: 1000
  chunk_size: 128
  ttl_ema_alpha: 0.1               # EMA 온라인 TTL 갱신 계수

  ttl_profiles:
    code:
      ttl_base_sec: 600
      reuse_probability: 0.75
    chat:
      ttl_base_sec: 300
      reuse_probability: 0.60
    rag:
      ttl_base_sec: 120
      reuse_probability: 0.45
    agentic:
      ttl_base_sec: 480
      reuse_probability: 0.80

eviction_policy:
  type: RedundancyAwareEvictionPolicy
  redundancy_top_n: 100
  importance_weight: 1.0
  redundancy_weight: 1.0
  doc_id_shortcut: true

scheduler:
  type: DAGTopologyScheduler
  retain_threshold: 0.5            # KV 보존 결정 최소 확률
  alpha_ttl_extend: 2.0            # DAGAwareTTLAdjuster TTL 연장 계수
  fallback: CacheAwareScheduler    # DAG 미지 요청 폴백

dag_ttl_adjuster:
  alpha: 2.0
  measure_latency: true

metrics:
  target_throughput_gain: 0.20          # +20% tokens/sec (§1)
  target_noncontiguous_hit_rate: 0.30   # 전체 히트의 30% 이상 비연속 (§3)
  max_scheduling_overhead_ttft_pct: 5.0 # 스케줄링 TTFT 오버헤드 +5% 이내 (§2)
  min_scheduling_hit_rate_gain: 0.10    # 스케줄링 캐시 히트율 +10%p (§2)
  max_perplexity_delta_pct: 1.0         # ±1% perplexity (§4)
  max_task_accuracy_delta_pct: 1.0      # ±1% 태스크 정확도 (§4)
  cross_throughput_gain_vs_single: 0.05 # 복합 처리량 단일 대비 +5% (§5)
  cross_memory_reduction_vs_single: 0.10 # 복합 메모리 단일 대비 −10% (§5)

accuracy_benchmarks:
  - name: important_segment_preservation
    metric: high_importance_segment_retention_rate
    threshold: 1.00                # 100% 보존 (±1% perplexity proxy)
  - name: residual_cosine_similarity
    metric: cosine_sim_after_eviction
    threshold: 0.99                # ≥ 0.99
  - name: information_loss_proxy
    metric: normalized_representation_loss
    threshold: 0.02                # ≤ 2%
  - name: longbench_qa
    metric: important_hit_rate_delta
    threshold: 0.01                # ≤ 1%p 변화
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_workload_ttl_cache.py` (신규 — WorkloadAwareTTLCache 단위 테스트)
- [ ] `tests/unit/test_redundancy_eviction_accuracy.py` (신규 — Activity C accuracy-preserving 필수)
- [ ] `tests/unit/test_dag_topology_scheduler.py` (신규 — DAGTopologyScheduler 단위 테스트)
- [ ] `tests/unit/test_dag_ttl_adjuster.py` (신규 — DAGAwareTTLAdjuster 단위 테스트)
- [ ] `tests/integration/test_cross_ab_dag_ttl.py` (신규 — Cross-1 전체 통합 테스트)
- [ ] 기존 단위·통합 테스트 전부 회귀 없이 통과

### test_workload_ttl_cache.py 필수 테스트 케이스

```python
def test_put_get_exact_hit():
    # put_segment() → get() 동일 key → hit 반환 (TTL 만료 전)

def test_ttl_expiry_returns_miss():
    # put_segment(ttl_override=0.001초) → 1ms 대기 → get() → None (TTL 만료)

def test_category_classification_code():
    # "def foo():" 포함 키/토큰 → category == "code"

def test_category_classification_rag():
    # "retrieved document:" 포함 → category == "rag"

def test_category_classification_agentic():
    # "tool_call:" 포함 → category == "agentic"

def test_category_default_chat():
    # 키워드 미포함 → category == "chat"

def test_ttl_profiles_different_per_category():
    # code TTL(600s) > chat TTL(300s) > agentic TTL(480s) 순서 검증

def test_pin_prevents_eviction():
    # pin(key) 후 evict() 시 해당 키 퇴거되지 않음

def test_unpin_allows_eviction():
    # pin() → unpin() → evict() → 키 퇴거됨

def test_adjust_ttl_extends():
    # adjust_ttl(key, 9999s) 후 TTL 만료 없이 get() 성공

def test_adjust_ttl_to_zero_immediate_eviction_candidate():
    # adjust_ttl(key, 0.0) → evict_candidates()에 포함

def test_noncontiguous_hit_rate_ttl_based():
    # TTL 만료 전 재사용 세그먼트가 ttl_hit_stats()["noncontiguous_ratio"] ≥ 0.30 기여

def test_ema_ttl_update_on_hit():
    # record_hit() 후 _ttl_profiles[category]["ttl_base_sec"] 변화 확인

def test_evict_ttl_expired_first():
    # TTL 만료 세그먼트와 미만료 세그먼트 공존 시 만료 세그먼트 우선 퇴거

def test_lru_fallback_when_no_expired():
    # TTL 만료 없을 때 evict() → LRU (가장 오래된 비핀 세그먼트) 퇴거

def test_cachestore_interface_compliance():
    # put(), get(), evict(), hit_rate(), memory_bytes(), reset_stats() 모두 구현

def test_reset_stats():
    # reset_stats() 후 exact_hits=0, ttl_preserved_hits=0, misses=0

def test_chunk_key_deterministic():
    # 동일 token_ids, chunk_idx, layer_idx → 동일 키 (segmented.py 호환)
```

### test_dag_topology_scheduler.py 필수 테스트 케이스

```python
def test_register_workflow_returns_dag_id():
    # register_workflow(valid_dag_spec) → dag_id 문자열 반환

def test_topological_order_correct():
    # A→B→C DAG → topological_order == ["A", "B", "C"]

def test_out_degree_computed():
    # 중간 노드 B가 자식 C를 가질 때 out_degree[B] == 1

def test_kv_reuse_probability_nonzero_for_parent():
    # 자식이 있는 노드의 kv_reuse_probability > 0.0

def test_kv_reuse_probability_zero_for_leaf():
    # 자식이 없는 리프 노드의 kv_reuse_probability == 0.0

def test_schedule_annotates_dag_node_id():
    # schedule() 후 DAG 등록 요청에 dag_node_id 속성 존재

def test_schedule_annotates_kv_reuse_probability():
    # schedule() 후 요청에 kv_reuse_probability 속성 존재

def test_pin_called_for_high_probability_node():
    # kv_reuse_probability > retain_threshold 노드 → cache.pin() 호출됨

def test_notify_node_complete_unpins():
    # notify_node_complete() 호출 → cache.unpin() 호출됨

def test_unknown_dag_falls_back_to_cache_aware():
    # dag_id 미등록 요청 → fallback_scheduler.schedule() 위임

def test_belady_upper_bound_gte_actual_hit_rate():
    # compute_belady_upper_bound() ≥ 실제 히트율 (상한 성질 검증)

def test_save_reuse_histogram_creates_file():
    # save_reuse_histogram(path) → JSON 파일 생성 확인

def test_scheduling_overhead_below_threshold():
    # schedule() 실행 시간이 배치당 5ms 이내 (100 요청 기준)

def test_cyclic_dag_raises_error():
    # 순환 DAG(A→B→A) 등록 시 ValueError 발생
```

### test_dag_ttl_adjuster.py 필수 테스트 케이스

```python
def test_on_kv_reuse_event_extends_ttl():
    # dag_reuse_probability=0.8, base_ttl=300, alpha=2.0
    # → adjusted_ttl = 300 × (1 + 0.8 × 2.0) = 780s 적용 확인

def test_on_node_complete_sets_ttl_zero():
    # on_node_complete(segment_key) → cache.adjust_ttl(key, 0.0) 호출

def test_on_node_complete_calls_unpin():
    # on_node_complete(segment_key) → cache.unpin(key) 호출

def test_latency_measured():
    # measure_latency=True → overhead_stats()["n_samples"] > 0

def test_overhead_p50_below_1ms():
    # 100번 on_kv_reuse_event() 호출 → p50 < 1.0ms (경량 연산 검증)

def test_zero_probability_does_not_extend_ttl():
    # dag_reuse_probability=0.0 → adjusted_ttl = base_ttl × 1.0 (변화 없음)
```

### test_cross_ab_dag_ttl.py 필수 통합 테스트 케이스

```python
def test_end_to_end_dag_ttl_pipeline():
    # DAGTopologyScheduler → DAGAwareTTLAdjuster → WorkloadAwareTTLCache 전체 파이프라인
    # DAG 등록 → 스케줄 → TTL 조정 → 조회 → 예외 없이 완료

def test_dag_kv_reuse_extends_ttl():
    # 후속 노드가 있는 DAG 노드의 세그먼트 TTL이 기본값보다 길어짐 확인

def test_dag_completion_enables_early_eviction():
    # notify_node_complete() 후 해당 세그먼트가 evict_candidates()에 포함

def test_redundancy_eviction_in_pipeline():
    # WorkloadAwareTTLCache + RedundancyAwareEvictionPolicy 통합
    # TTL 만료 후보 중 중복 세그먼트 우선 퇴거 확인

def test_noncontiguous_hit_rate_above_30pct():
    # 다수 TTL 보존 히트 발생 후 ttl_hit_stats()["noncontiguous_ratio"] ≥ 0.30

def test_pinned_segments_not_evicted_during_pipeline():
    # DAG 핀 고정 세그먼트가 evict() 호출에도 캐시에 잔존

def test_scheduling_overhead_within_5pct():
    # DAGTopologyScheduler.schedule() 오버헤드 측정 → TTFT +5% 이내 proxy

def test_hit_rate_improvement_with_dag_scheduling():
    # DAGTopologyScheduler 적용 시 CacheAwareScheduler 대비 캐시 히트율 +10%p 이상

def test_accuracy_preservation_in_full_pipeline():
    # 중요도 높은 세그먼트가 전체 파이프라인 실행 후에도 캐시에 잔존
    # test_redundancy_eviction_accuracy.py의 핵심 조건을 통합 환경에서 재확인
```

---

## 구현 시 주의사항

1. **CacheStore 인터페이스 준수**: `WorkloadAwareTTLCache`는 `CacheStore`를 직접 상속하며,
   6개 추상 메서드(`put`, `get`, `evict`, `hit_rate`, `memory_bytes`, `reset_stats`)를
   모두 구현해야 한다. `put_segment()`, `adjust_ttl()`, `pin()` 등은 추가 메서드다.

2. **RedundancyAwareEvictionPolicy는 CacheStore 미상속**: 순수 스코어링 레이어이므로
   `CacheStore`를 상속하지 않는다. `WorkloadAwareTTLCache` 생성자에서 선택적으로 주입.

3. **DAGTopologyScheduler는 스케줄러 역할**: `CacheStore` 미상속. 캐시를 소유하지 않고
   `WorkloadAwareTTLCache`의 `pin()` / `unpin()` / `adjust_ttl()` API만 호출.

4. **TTL 만료 체크는 get() 시점**: `time.monotonic()` 기반. 저장된 `created_at`과 비교.
   만료된 세그먼트는 통계에서 miss로 기록하고 즉시 또는 다음 evict() 시 제거.

5. **순환 DAG 감지**: `register_workflow()` 에서 Kahn 알고리즘 완료 후
   `len(topological_order) != len(nodes)` 이면 순환 DAG → `ValueError` 발생.

6. **모든 단위 테스트는 CPU 텐서 기준**: `torch.device("cpu")`. GPU 불필요.

7. **시드 고정**: 모든 테스트에서 `torch.manual_seed(42)` 사용. 재현성 보장.

8. **훈련-무료 제약**: `WorkloadAwareTTLCache`, `RedundancyAwareEvictionPolicy`,
   `DAGTopologyScheduler`, `DAGAwareTTLAdjuster` 모두 학습 파라미터(`nn.Parameter`,
   `nn.Module`) 미포함.

9. **thread 안전성**: `WorkloadAwareTTLCache._store` 접근은 단일 스레드 가정 (멀티스레드 환경
   필요 시 `threading.Lock` 추가 가능하지만 이번 사이클 필수 아님).

10. **기존 테스트 회귀 없이 통과**: 새로운 파일만 추가하므로 기존 코드가 변경되지 않는다.
    `test_turbo_quant_accuracy.py`, `test_dhd_segment_cache.py` 등 이전 사이클 테스트는
    그대로 유지.

11. **chunk_key() 호환성**: `WorkloadAwareTTLCache.chunk_key()`는 `SegmentedHashCache.chunk_key()`
    와 동일한 SHA-256 알고리즘을 사용해 기존 캐시와 키 체계 호환.

12. **결과 파일 저장**: `DAGTopologyScheduler.save_reuse_histogram()`이 생성하는 파일은
    `results/<exp_name>/kv_reuse_histogram.json` 경로에 저장. `results/`는 git-ignored.

---

## 완료 기준 (Definition of Done)

- [ ] `tests/unit/test_workload_ttl_cache.py` — 17개 테스트 케이스 전부 통과
- [ ] `tests/unit/test_redundancy_eviction_accuracy.py` — 12개 테스트 케이스 전부 통과 (§4 Accuracy 필수)
- [ ] `tests/unit/test_dag_topology_scheduler.py` — 14개 테스트 케이스 전부 통과
- [ ] `tests/unit/test_dag_ttl_adjuster.py` — 6개 테스트 케이스 전부 통과
- [ ] `tests/integration/test_cross_ab_dag_ttl.py` — 9개 테스트 케이스 전부 통과
- [ ] 기존 단위·통합 테스트 전부 회귀 없이 통과
- [ ] `configs/experiments/2026-05-04.yaml` 존재 (§0 설정 YAML 필수)
- [ ] `evaluation_criteria.md` §4 Activity C (보조, 필수):
  - 중요 세그먼트 보존율 100% (eviction_score 구조적 보장)
  - 잔존 Key 코사인 유사도 ≥ 0.99 (perplexity ±1% proxy)
  - 정보 손실 proxy ≤ 2% (LongBench ±1% proxy)
- [ ] `evaluation_criteria.md` §3 Activity B:
  - 비연속 히트율(TTL 보존 히트) ≥ 전체 히트의 30%
  - KV Memory Footprint: 베이스라인 대비 +20% 이내
- [ ] `evaluation_criteria.md` §2 Activity A:
  - 스케줄링 오버헤드 TTFT p50 +5% 이내
  - 스케줄링 적용 캐시 히트율 향상 ≥ +10%p
- [ ] `evaluation_criteria.md` §5 Cross:
  - 복합 Throughput 향상: 단일 Activity 대비 추가 +5%
  - 복합 Memory Reduction: 단일 Activity 대비 추가 −10%
  - Accuracy 보존 복합 적용 후 ±1% 이내 (필수)
- [ ] 타입 힌트 모든 공개 함수·메서드에 존재 (§0 중간)
- [ ] 불필요한 추상화 없음: 이 Spec에 없는 클래스·인터페이스 도입 금지 (§0 낮음)
- [ ] `results/<exp>/kv_reuse_histogram.json` 저장 확인 (목표 10 — DAG 재사용 확률 측정)
