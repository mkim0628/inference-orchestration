<!-- 변경 이유 (이전 Spec.md: 2026-05-28 대비):
이전 사이클(2026-05-28)은 A+B 조합이었다:
  - A-1 HexAGeTWorkflowHorizonAwareKVCapacityScheduler (DAG 워크플로우 스케줄러)
  - A-2 PegaFlowRustKVConnectorRDMACrossNodeSegmentRouter (RDMA 크로스-노드 라우터)
  - B-1 PegaFlowIrminsulDistributedSegmentCache (분산 비연속 세그먼트 캐시)
  - Cross-1 HexAGeTIrminsulPegaFlowPipeline (A+B 통합 파이프라인)

이번 사이클(2026-05-30)은 B+C 조합으로 전환된다.
핵심 전환:
  - 최우선 C-2: MinimalInterventionFacilityLocationCodec — argmax-top-k를 greedy
    facility-location + V-space 다양성 페널티(λ)로 교체. src/cache/segmented.py의
    SegmentedHashCache + TriAttentionPreRoPEKVSelectorCodec select_kv()에 통합.
    구현 난이도 low, 즉시 적용 가능.
  - 2순위 C-1: SphericalKVJointCodec — Key 구형 파라미터화(ADA: 스칼라 반경 r +
    각도 코드 θ) + Rate-Distortion Retention(RDR) 공동 토큰-정밀도 최적화. 신규
    CacheStore 구현체로 독립 분리.
  - 3순위 B-1: ThreeTrackArchAwarePICRouter — 기존 2-트랙(MLA→Irminsul,
    GQA/MHA→AdapShot)에 COMB Native PIC cross-attention 3번째 트랙 추가.
    COMBNativePICCrossAttentionCache 신규 구현 + dispatch 레이어 확장.
  - Cross-1: COMBSphericalKVPipeline — B-1 COMB × C-1 SPHERICAL KV ADA 수학적
    호환성(score[t] = r[t] × (Q · direction(θ[t]))) 기반 통합 파이프라인.

Activity C를 포함하므로 accuracy-preserving 검증 계획이 본 Spec에 필수 포함됨.
  - WikiText-2 perplexity ±1% 이내
  - RULER-4K / RULER-16K 정확도 ±1% 이내
  - LongBench 8개 서브태스크 ±1% 이내
  - MATH-500 추론 정확도 (C-2 소예산 b=64/128/256 스윕)
  - argmax-top-k vs. facility-location 직접 비교

주요 신규 파일:
1. [신규] src/cache/minimal_intervention_fl_codec.py (C-2 facility-location 퇴거 코덱)
2. [신규] src/cache/spherical_kv_ada_codec.py (C-1 ADA 각도 코드 압축)
3. [신규] src/cache/spherical_kv_rdr_codec.py (C-1 RDR Rate-Distortion 최적화)
4. [신규] src/cache/spherical_kv_joint_codec.py (C-1 ADA + RDR 통합 캐시)
5. [신규] src/cache/comb_native_pic_cache.py (B-1 COMB cross-attention 캐시)
6. [신규] src/cache/three_track_pic_router.py (B-1 3-트랙 dispatch 라우터)
7. [신규] src/engine/comb_spherical_kv_pipeline.py (Cross-1 B+C 통합 파이프라인)
8. [변경] src/metrics/hit_rate.py — ThreeTrackHitRateMetrics 추가
9. [신규] configs/experiments/2026-05-30.yaml
10. [신규] configs/pic_arch_registry.yaml (PIC 아키텍처 트랙 레지스트리)
11. [신규] tests/unit/test_minimal_intervention_fl_codec.py
12. [신규] tests/unit/test_spherical_kv_joint_codec.py
13. [신규] tests/unit/test_three_track_pic_router.py
14. [신규] tests/unit/test_compression_accuracy_2026_05_30.py (C-2+C-1 accuracy 검증)
15. [신규] tests/integration/test_comb_spherical_kv_pipeline_e2e.py
16. [보존] 모든 이전 사이클 구현 파일 수정 금지.
-->

# Spec — 2026-05-30: Minimal-Intervention V-Space Facility-Location Eviction + SPHERICAL KV Rate-Distortion + COMB Native PIC 3-Track Router (B+C)

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-30.md`

**최우선 구현 타겟**: C-2 `MinimalInterventionFacilityLocationVSpaceDiversityEvictionCodec`
**2순위 구현 타겟**: C-1 `SphericalKVRateDistortionRetentionJointCodec`
**3순위 구현 타겟**: B-1 `COMBEncoderInterleavedNativePICArchitectureAwareRouter`
**4순위 구현 타겟**: Cross-1 `COMBNativePICSphericalKVAngleCodeBCIntegrationCodec`

**해결하려는 문제**:

- **Activity C-2 (Minimal-Intervention facility-location)**:
  소예산(b=64~128) 환경에서 argmax-top-k 토큰 선택이 V-공간에서 중복 토큰을 반복
  선택해 정보 밀도를 낮추는 구조적 취약점이 있다. greedy facility-location 선택 +
  V-space redundancy penalty(λ)로 교체해 소예산 reasoning 워크로드의 accuracy delta를
  ±0.3~0.6% 이내로 유지하면서 메모리 −30~50%를 달성한다.

- **Activity C-1 (SPHERICAL KV ADA + RDR)**:
  기존 압축 방법들이 토큰 선택(퇴거)과 정밀도 할당(양자화)을 독립적으로 최적화한다.
  Key를 스칼라 반경 r + 각도 코드 θ의 구형 파라미터화로 저장하고, Rate-Distortion 이론으로
  토큰 선택·정밀도 티어를 공동 최적화해 동일 메모리 예산에서 파레토 최적 accuracy 보존을
  달성한다. 목표: 메모리 −40~60%, accuracy delta ±0.5~0.8%.

- **Activity B-1 (COMB 3-트랙 PIC 라우터)**:
  기존 2-트랙 PIC(MLA→Irminsul δ-회전, GQA/MHA→AdapShot RoPE 재인코딩)에
  COMB(arXiv:2602.01519) encoder interleaving Native PIC cross-attention 3번째 트랙을
  추가한다. COMB 훈련 모델에서 재인코딩·δ-회전 오버헤드 없이 위치-독립 cross-attention만으로
  TTFT −51~94%, 처리량 +3× 목표.

- **Cross-1 (COMB × SPHERICAL KV B+C 통합)**:
  COMB encoder cross-attention KV를 SPHERICAL KV ADA로 저장하면 score[t] = r[t] × (Q · direction(θ[t]))
  수식이 수학적으로 원래 score[t] = Q · K[t]와 등가다. 밀집 Key 재구성 없이 Native PIC + ADA가
  단일 패스에서 작동하는 B+C 통합 파이프라인.

---

## 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────────────────────────────┐
│               COMB × SPHERICAL KV B+C Integration Pipeline                  │
│                                                                              │
│  입력 토큰 시퀀스                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │    ThreeTrackArchAwarePICRouter (Activity B-1)                      │     │
│  │                                                                     │     │
│  │  detect_pic_track(model_config) →                                   │     │
│  │    COMB_NATIVE: COMBNativePICCrossAttentionCache  (신규)             │     │
│  │    MLA:         IrminsulMLASegmentCache (05-26 기구현)               │     │
│  │    GQA/MHA:     AdapShot RoPEReencodingCache (05-18 기구현)          │     │
│  └──────────────────────────┬──────────────────────────────────────────┘     │
│                             │ COMB 트랙                                      │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │    COMBSphericalKVPipeline (Cross-1 B+C)                            │     │
│  │                                                                     │     │
│  │  Step 1: CDC 세그먼트 분할 → segment_id = SHA256(token_content)     │     │
│  │  Step 2: COMB encoder cross-attention KV 계산                       │     │
│  │  Step 3: ADA 분해 — K = r × direction(θ)                           │     │
│  │  Step 4: RDR 공동 최적화 — keep_mask + precision_tier              │     │
│  │  Step 5: facility-location V-공간 다양성 (C-2 통합 옵션)            │     │
│  │  Step 6: COMBSphericalKVEntry 저장                                  │     │
│  │  재사용: score[t] = r[t] × (Q · direction(θ[t]))                   │     │
│  └──────────────────────────┬──────────────────────────────────────────┘     │
│                             │                                                │
│       ┌─────────────────────┼──────────────────────────┐                    │
│       ▼                     ▼                           ▼                    │
│  C-2 단독 경로         C-1 단독 경로              B-1 비COMB 경로            │
│  MinimalIntervention  SphericalKVJoint         IrminsulMLA / AdapShot        │
│  FacilityLocation     Codec                                                  │
│  Codec                                                                       │
│                                                                              │
│  ThreeTrackHitRateMetrics: {comb_hit, mla_hit, gqa_hit, miss}               │
│  CompressionAccuracyMetrics: perplexity_delta, task_accuracy_delta          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling (이번 사이클 미포함 — 설계 아이디어만)
- [x] Activity B: Non-Contiguous KV Cache Reuse — COMB Native PIC 3-트랙 PIC 라우터 (B-1)
- [x] Activity C: KV Cache Compression — Minimal-Intervention facility-location (C-2) + SPHERICAL KV ADA+RDR (C-1)

**스케줄링 결정 단위**: 이번 사이클은 스케줄러 신규 구현 없음. B-1 ThreeTrackArchAwarePICRouter가
세그먼트 조회 단위로 dispatch 결정을 수행한다.
**캐시 상태 접근 방법**: ThreeTrackArchAwarePICRouter.get()이 model_config의 아키텍처 필드를
조회해 COMB/MLA/GQA 경로를 결정한다. 압축 상태는 SphericalKVJointCodec.get()이 내부
_store의 entry 형태로 관리한다.

---

## 목표

- [ ] 목표 1: Compression Accuracy Delta ±1% 이내 — C-2 ±0.3~0.6% + C-1 ±0.5~0.8% WikiText-2 perplexity + MATH-500 (evaluation_criteria.md §4 필수)
- [ ] 목표 2: KV Cache Memory Reduction C-2 −30% 이상 + C-1 −40% 이상 (§4 높음)
- [ ] 목표 3: Non-Contiguous Cache Hit Rate 전체 히트의 30% 이상이 비연속 구간에서 발생 (§3 높음)
- [ ] 목표 4: B-1 3-트랙 통합 히트율 베이스라인 대비 +5%p 이상 (§3 높음)
- [ ] 목표 5: 압축 오버헤드 TTFT +10% 이내 (§4 높음)
- [ ] 목표 6: 복합 Throughput 향상 단일 Activity 대비 추가 +5% 이상 (Cross-1, §5)
- [ ] 목표 7 (신규 지표): facility_location_diversity_gain — facility-location vs. argmax-top-k accuracy delta 차이 (C-2)
- [ ] 목표 8 (신규 지표): rdr_budget_utilization — 실제 사용 비트 / 할당 비트 예산 (C-1)
- [ ] 목표 9 (신규 지표): track_aware_hit_rate — {comb_hit, mla_hit, gqa_hit, miss} 계층별 분리 (B-1)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/minimal_intervention_fl_codec.py` | C-2 | MinimalInterventionFacilityLocationCodec; greedy facility-location V-공간 다양성 페널티(λ) 토큰 선택; CacheStore 구현 |
| `src/cache/spherical_kv_ada_codec.py` | C-1 | SphericalKVADACodec; Key 구형 파라미터화(r + θ); 각도 코드 INT8 압축; 밀집 Key 재구성 없는 어텐션 로짓 계산 |
| `src/cache/spherical_kv_rdr_codec.py` | C-1 | SphericalKVRDRCodec; Rate-Distortion 토큰-정밀도 공동 최적화; tier-homogeneous 페이지 생성 |
| `src/cache/spherical_kv_joint_codec.py` | C-1 | SphericalKVJointCodec; ADA + RDR 통합 CacheStore 구현; CompressionCodec 내부 구성 |
| `src/cache/comb_native_pic_cache.py` | B-1 | COMBNativePICCrossAttentionCache; COMB encoder cross-attention KV 위치-독립 저장; CacheStore 구현 |
| `src/cache/three_track_pic_router.py` | B-1 | ThreeTrackArchAwarePICRouter; detect_pic_track(); COMB/MLA/GQA 3-트랙 dispatch; CacheStore 구현 |
| `src/engine/comb_spherical_kv_pipeline.py` | Cross-1 | COMBSphericalKVPipeline; 6단계 B+C 통합 처리 흐름; COMBSphericalKVEntry 저장 구조 |
| `configs/experiments/2026-05-30.yaml` | 공통 | 이번 사이클 실험 설정 |
| `configs/pic_arch_registry.yaml` | B-1 | {model_name → pic_track} 레지스트리 |
| `tests/unit/test_minimal_intervention_fl_codec.py` | C-2 | facility-location 선택, λ 다양성 효과, CacheStore 인터페이스 단위 테스트 |
| `tests/unit/test_spherical_kv_joint_codec.py` | C-1 | ADA 분해, RDR 최적화, 통합 압축, CacheStore 인터페이스 단위 테스트 |
| `tests/unit/test_three_track_pic_router.py` | B-1 | 3-트랙 감지, dispatch 순서, 히트율 집계 단위 테스트 |
| `tests/unit/test_compression_accuracy_2026_05_30.py` | C-2+C-1 | accuracy preservation 검증 (perplexity proxy, cosine sim, KL-div 기반) |
| `tests/integration/test_comb_spherical_kv_pipeline_e2e.py` | Cross-1 | 6단계 B+C 파이프라인 엔드-투-엔드 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/metrics/hit_rate.py` | `ThreeTrackHitRateMetrics` 클래스 추가. `{comb_hit, mla_hit, gqa_hit, miss}` 4단계 히트율 집계. 기존 `WeightedHitRateMetrics`, `HitRateMetrics`, `DistributedHitRateMetrics` 수정 금지 |

---

## 알고리즘 상세

### 1. MinimalInterventionFacilityLocationCodec (Activity C-2) — `src/cache/minimal_intervention_fl_codec.py`

**핵심 아이디어**: argmax-top-k를 greedy facility-location으로 교체해 V-공간 다양성 강제.

#### 자료구조

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from src.cache.base import CacheStore


@dataclass
class FacilityLocationConfig:
    budget_tokens: int = 128          # 보존할 토큰 수 (기본 b=128)
    lambda_div: float = 0.5           # V-공간 다양성 페널티 가중치
    lambda_auto_scale: bool = True    # True: lambda_div_eff = lambda_div × (128 / budget_tokens)
    use_triattention_scorer: bool = True  # True: TriAttentionPreRoPEKVSelectorCodec 점수 재사용
    fallback_to_snapkv: bool = True   # TriAttention 미사용 시 norm 기반 폴백
    max_entries: int = 1000
    seed: int = 42


@dataclass
class FLKVEntry:
    selected_kv: torch.Tensor     # [n_kept, d_head] — 선택된 KV
    kept_indices: torch.Tensor    # [n_kept] int64
    original_seq_len: int
    lambda_div_effective: float   # 실제 사용된 lambda_div (auto_scale 반영)
    diversity_gain: float         # cosine diversity 향상 측정값
```

#### 핵심 알고리즘: `facility_location_selection()`

```python
def facility_location_selection(
    importance_scores: torch.Tensor,  # [n_tokens] FP32 — TriAttention 또는 norm 기반
    V_matrix: torch.Tensor,           # [n_tokens, d_head] FP16
    budget_tokens: int,               # 보존할 토큰 수 k
    lambda_div: float,                # V-공간 다양성 페널티 가중치
) -> torch.Tensor:
    """Greedy facility-location token selection with V-space diversity penalty.

    알고리즘:
      keep_set = []           # 선택된 토큰 인덱스 목록
      V_norm = V_matrix / (V_matrix.norm(dim=-1, keepdim=True) + 1e-8)  # 정규화

      for _ in range(budget_tokens):
          if len(keep_set) == 0:
              # 첫 토큰: 중요도만으로 선택
              next_token = argmax(importance_scores)
          else:
              # V-공간 다양성 보너스: 선택된 집합과의 최소 코사인 거리
              selected_V = V_norm[keep_set]       # [k_selected, d_head]
              sim = V_norm @ selected_V.T         # [n_tokens, k_selected] — 코사인 유사도
              max_sim = sim.max(dim=-1).values    # [n_tokens] — 가장 유사한 기존 토큰과의 유사도
              diversity_bonus = 1.0 - max_sim     # 코사인 거리 (다양성 높을수록 큰 값)

              combined_score = importance_scores + lambda_div * diversity_bonus
              # 이미 선택된 토큰은 제외
              combined_score[keep_set] = -torch.inf
              next_token = argmax(combined_score)

          keep_set.append(next_token.item())

      return torch.tensor(sorted(keep_set), dtype=torch.int64)

    시간복잡도: O(n_tokens × budget_tokens) — n=512, k=128 기준 GPU에서 < 1ms
    """
```

#### CacheStore 구현: `MinimalInterventionFacilityLocationCodec`

```python
class MinimalInterventionFacilityLocationCodec(CacheStore):
    """greedy facility-location V-공간 다양성 페널티 토큰 퇴거 코덱 (arXiv:2605.14292 α).

    CacheStore 인터페이스:
      - put(key, value): plain 저장 (호환성)
      - put_compressed(key, K, V, importance_scores): facility-location 압축 저장
      - get(key): 선택된 KV 반환
      - evict(): FIFO 퇴거
    """

    def __init__(self, config: FacilityLocationConfig) -> None: ...

    def put(self, key: str, value: torch.Tensor) -> None:
        """Plain 저장 — K, V가 분리된 경우 put_compressed() 권장."""
        ...

    def put_compressed(
        self,
        key: str,
        K: torch.Tensor,           # [n_tokens, d_head]
        V: torch.Tensor,           # [n_tokens, d_head]
        importance_scores: Optional[torch.Tensor] = None,  # None → norm 기반 폴백
    ) -> FLKVEntry:
        """facility-location 선택 후 압축 저장.

        1. importance_scores가 None이면 K.norm(dim=-1)으로 대체 (SnapKV 폴백).
        2. lambda_auto_scale=True이면 lambda_div_eff = lambda_div × (128 / budget_tokens).
        3. facility_location_selection(importance_scores, V, budget_tokens, lambda_div_eff) 호출.
        4. FLKVEntry 저장.
        """
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        """선택된 KV 텐서 반환. 미스 시 None."""
        ...

    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...

    def memory_reduction_ratio(self) -> float:
        """선택된 토큰 수 / 원래 토큰 수 기반 메모리 감소율."""
        ...

    def diversity_gain_stats(self) -> dict:
        """V-공간 다양성 향상 통계 — diversity_gain 히스토리 평균/표준편차.
        Returns: {"diversity_gain_mean": float, "diversity_gain_std": float}
        """
        ...

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """[original_seq_len] bool 마스크 — kept_indices=True."""
        ...
```

---

### 2. SphericalKVADACodec (Activity C-1) — `src/cache/spherical_kv_ada_codec.py`

**핵심 아이디어**: Key 텐서를 스칼라 반경 r + 각도 코드 θ로 분해해 밀집 Key 재구성 없이 어텐션 로짓 계산.

```python
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from src.cache.base import CacheStore


@dataclass
class ADAConfig:
    angle_quantize_bits: int = 8      # 각도 코드 양자화 비트 (8=INT8, 4=INT4)
    radius_dtype: torch.dtype = torch.float16   # 반경 저장 dtype
    max_entries: int = 1000
    seed: int = 42


@dataclass
class ADAKVEntry:
    K_angle: torch.Tensor    # [n_tokens, d_head] — 양자화된 각도 코드 (INT8)
    K_radius: torch.Tensor   # [n_tokens] FP16 — 스칼라 반경
    V: torch.Tensor          # [n_tokens, d_head] — 원본 또는 양자화된 V
    n_tokens: int
    original_seq_len: int


class SphericalKVADACodec(CacheStore):
    """Angle-Domain Attention: Key 구형 파라미터화 압축 코덱 (arXiv:2605.18856).

    K[t] = r[t] × direction(θ[t])
    score[t] = r[t] × (Q · direction(θ[t]))  ← 밀집 Key 재구성 불필요

    주요 메서드:
      decompose(K): K → (K_radius, K_angle) — r과 θ 분리
      reconstruct_score(Q, K_angle, K_radius): 밀집 K 없이 어텐션 로짓 직접 계산
      quantize_angle(direction): 방향 벡터 INT8 양자화
      dequantize_angle(K_angle_int8): INT8 → FP32 방향 벡터 복원

    CacheStore 구현: put_ada(key, K, V), get(key) → ADAKVEntry
    """

    def __init__(self, config: ADAConfig) -> None: ...

    def decompose(self, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """K [n_tokens, d_head] → (K_radius [n_tokens], K_direction [n_tokens, d_head]).

        K_radius = ||K[t]||_2
        K_direction = K[t] / (K_radius[t] + 1e-8)  (단위 벡터)
        """
        ...

    def quantize_angle(self, K_direction: torch.Tensor) -> torch.Tensor:
        """단위 벡터 → INT8 각도 코드.

        [-1, 1] 범위를 [-127, 127]로 선형 양자화.
        반환: [n_tokens, d_head] int8
        """
        ...

    def dequantize_angle(self, K_angle_int8: torch.Tensor) -> torch.Tensor:
        """INT8 각도 코드 → FP32 단위 벡터 (근사 복원).

        반환: [n_tokens, d_head] float32
        """
        ...

    def reconstruct_score(
        self,
        Q: torch.Tensor,           # [n_q, d_head]
        K_angle_int8: torch.Tensor, # [n_kv, d_head] int8
        K_radius: torch.Tensor,    # [n_kv] FP16
    ) -> torch.Tensor:
        """밀집 Key 재구성 없이 어텐션 로짓 직접 계산.

        score[q, t] = K_radius[t] × (Q[q] · dequantize_angle(K_angle[t]))

        반환: [n_q, n_kv] FP32
        """
        ...

    def put_ada(self, key: str, K: torch.Tensor, V: torch.Tensor) -> ADAKVEntry:
        """ADA 분해 후 압축 저장.

        1. decompose(K) → K_radius, K_direction
        2. quantize_angle(K_direction) → K_angle INT8
        3. ADAKVEntry 저장
        """
        ...

    # CacheStore 추상 메서드
    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...

    def memory_reduction_ratio(self) -> float:
        """ADA 압축 후 메모리 감소율.

        원본: n_tokens × d_head × 2 bytes (FP16)
        압축: K_angle(INT8: 1byte/elem) + K_radius(FP16: 2bytes) + V
        """
        ...
```

---

### 3. SphericalKVRDRCodec (Activity C-1) — `src/cache/spherical_kv_rdr_codec.py`

**핵심 아이디어**: Rate-Distortion 이론으로 토큰 keep/drop + 정밀도 티어를 공동 최적화.

```python
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import torch


PrecisionTier = Literal["INT4", "INT8", "FP16"]


@dataclass
class RDRConfig:
    budget_bits_per_token: int = 64   # 토큰당 비트 예산 (기본 64bits = INT4 16차원)
    available_tiers: List[PrecisionTier] = None  # ["INT4", "INT8", "FP16"]
    distortion_metric: str = "attn_weight"       # "attn_weight" | "kl_div"
    seed: int = 42

    def __post_init__(self):
        if self.available_tiers is None:
            self.available_tiers = ["INT4", "INT8", "FP16"]


@dataclass
class RDRResult:
    keep_mask: torch.Tensor              # [n_tokens] bool
    precision_tiers: List[PrecisionTier] # 토큰별 정밀도 티어
    bits_used: int                       # 실제 사용 비트
    budget_bits: int                     # 할당 예산 비트
    budget_utilization: float            # bits_used / budget_bits


class SphericalKVRDRCodec:
    """Rate-Distortion Retention: 고정 비트 예산 하 토큰-정밀도 공동 최적화.

    최적화 문제:
      min_{S, P} D(S, P)  s.t.  Σ_t R(s_t, p_t) ≤ B
      - S = {s_t}: 토큰별 keep(1)/drop(0)
      - P = {p_t}: 정밀도 티어 (INT4/INT8/FP16)
      - R(s_t, p_t): 비트 비용 (drop→0, INT4→4bits/elem, INT8→8, FP16→16)
      - D(S, P): 어텐션 왜곡 (attention weight 분포 변화 추정)

    그리디 솔루션:
      각 (토큰, 티어) 쌍에 대해 단위 비트당 왜곡 감소율(ΔD/ΔR) 계산
      → 높은 순서로 예산 배분

    주요 메서드:
      optimize(K, V, attn_weights, budget_bits) → RDRResult
    """

    def __init__(self, config: RDRConfig) -> None: ...

    def _bits_per_element(self, tier: PrecisionTier) -> int:
        """INT4→4, INT8→8, FP16→16."""
        return {"INT4": 4, "INT8": 8, "FP16": 16}[tier]

    def _bit_cost(self, d_head: int, tier: PrecisionTier) -> int:
        """토큰 1개를 tier로 저장하는 비트 비용: d_head × bits_per_element(tier)."""
        ...

    def _distortion_drop(self, token_idx: int, attn_weights: torch.Tensor) -> float:
        """토큰 t를 drop할 때 어텐션 왜곡 추정.

        D_drop(t) = attn_weights[t]  (어텐션 가중치가 클수록 중요 토큰)
        """
        ...

    def _distortion_quantize(
        self,
        K_t: torch.Tensor,           # [d_head] 원본 Key
        tier: PrecisionTier,
    ) -> float:
        """토큰 t를 tier로 양자화할 때 왜곡 추정.

        D_quant(t, tier) = ||K_t - dequantize(quantize(K_t, tier))||_2 / ||K_t||_2
        """
        ...

    def optimize(
        self,
        K: torch.Tensor,             # [n_tokens, d_head]
        V: torch.Tensor,             # [n_tokens, d_head]
        attn_weights: torch.Tensor,  # [n_tokens] — 정책 어텐션 가중치 (중요도 프록시)
        budget_bits: int,            # 총 비트 예산
    ) -> RDRResult:
        """그리디 Rate-Distortion 최적화.

        알고리즘:
          d_head = K.shape[-1]
          n_tokens = K.shape[0]

          # 각 (토큰, 액션) 쌍에 대해 ΔD/ΔR 계산
          # 액션: drop(s_t=0), INT4, INT8, FP16
          candidates = []
          for t in range(n_tokens):
              for tier in ["INT4", "INT8", "FP16"]:
                  cost = _bit_cost(d_head, tier)
                  gain = attn_weights[t] - _distortion_quantize(K[t], tier)
                  efficiency = gain / (cost + 1e-8)  # ΔD/ΔR
                  candidates.append((efficiency, t, tier, cost))

          # 효율 높은 순으로 예산 배분
          candidates.sort(key=lambda x: -x[0])
          keep_decisions = {t: None for t in range(n_tokens)}  # None=drop
          bits_used = 0
          for eff, t, tier, cost in candidates:
              if keep_decisions[t] is None and bits_used + cost <= budget_bits:
                  keep_decisions[t] = tier
                  bits_used += cost

          keep_mask = torch.tensor([keep_decisions[t] is not None for t in range(n_tokens)])
          precision_tiers = [keep_decisions[t] or "INT4" for t in range(n_tokens)]
          return RDRResult(keep_mask, precision_tiers, bits_used, budget_bits, bits_used/budget_bits)
        """
        ...

    def apply_precision(
        self,
        tensor: torch.Tensor,   # [n_tokens, d_head]
        keep_mask: torch.Tensor,
        precision_tiers: List[PrecisionTier],
    ) -> torch.Tensor:
        """keep_mask 적용 + 정밀도 티어별 양자화 (tier-homogeneous 페이지 구성).

        keep_mask=False 토큰 제외 후, 같은 tier 토큰끼리 배치 양자화.
        FP16으로 정규화해 반환 (다운스트림 호환성).
        """
        ...
```

---

### 4. SphericalKVJointCodec (Activity C-1 통합) — `src/cache/spherical_kv_joint_codec.py`

```python
from dataclasses import dataclass
from typing import Optional, Dict
import torch
from src.cache.base import CacheStore
from src.cache.spherical_kv_ada_codec import SphericalKVADACodec, ADAConfig, ADAKVEntry
from src.cache.spherical_kv_rdr_codec import SphericalKVRDRCodec, RDRConfig, RDRResult


@dataclass
class SphericalKVJointConfig:
    ada: ADAConfig = None
    rdr: RDRConfig = None
    budget_bits_total: int = 2048     # 전체 토큰 비트 예산 (YAML 외부화)
    accuracy_fallback_threshold: float = 0.01  # cosine_sim < 0.99 시 budget 완화
    max_entries: int = 1000
    seed: int = 42

    def __post_init__(self):
        if self.ada is None:
            self.ada = ADAConfig()
        if self.rdr is None:
            self.rdr = RDRConfig()


@dataclass
class SphericalKVEntry:
    K_angle: torch.Tensor        # [n_kept, d_head] INT8 — 각도 코드
    K_radius: torch.Tensor       # [n_kept] FP16 — 스칼라 반경
    V: torch.Tensor              # [n_kept, d_head] — RDR 정밀도 티어 적용
    rdr_precision_tiers: list    # 토큰별 정밀도 티어 (INT4/INT8/FP16)
    keep_mask: torch.Tensor      # [original_seq_len] bool
    original_seq_len: int
    budget_utilization: float    # rdr_budget_utilization 지표용


class SphericalKVJointCodec(CacheStore):
    """SPHERICAL KV ADA + RDR 통합 압축 코덱 (arXiv:2605.18856).

    compress(K, V, attn_weights, budget_bits):
      # Step 1: ADA — Key 구형 파라미터화
      K_radius, K_direction = ada.decompose(K)
      K_angle_int8 = ada.quantize_angle(K_direction)

      # Step 2: RDR — 토큰-정밀도 공동 최적화
      rdr_result = rdr.optimize(K, V, attn_weights, budget_bits)

      # Step 3: 결합 저장
      return SphericalKVEntry(
          K_angle=K_angle_int8[rdr_result.keep_mask],
          K_radius=K_radius[rdr_result.keep_mask],
          V=rdr.apply_precision(V, rdr_result.keep_mask, rdr_result.precision_tiers),
          rdr_precision_tiers=rdr_result.precision_tiers,
          keep_mask=rdr_result.keep_mask,
          original_seq_len=K.shape[0],
          budget_utilization=rdr_result.budget_utilization,
      )

    CacheStore 인터페이스:
      - put(key, value): plain 저장 (호환성)
      - put_compressed(key, K, V, attn_weights): ADA+RDR 압축 저장
      - get(key): SphericalKVEntry.V 반환
      - get_entry(key): SphericalKVEntry 전체 반환
      - reconstruct_scores(key, Q): 밀집 Key 없이 어텐션 로짓 계산
    """

    def __init__(self, config: SphericalKVJointConfig) -> None: ...

    def put_compressed(
        self,
        key: str,
        K: torch.Tensor,              # [n_tokens, d_head]
        V: torch.Tensor,              # [n_tokens, d_head]
        attn_weights: Optional[torch.Tensor] = None,  # [n_tokens] — None → uniform
        budget_bits: Optional[int] = None,             # None → config.budget_bits_total
    ) -> SphericalKVEntry:
        """ADA 분해 → RDR 최적화 → 통합 저장.

        accuracy_fallback:
          compressed = compress(...)
          sim = cosine_similarity_output(Q_probe, K_orig, V_orig, K_kept, V_kept)
          if sim < 1.0 - accuracy_fallback_threshold:
              # budget_bits 증가 후 재압축 (1회 재시도)
              budget_bits = int(budget_bits * 1.5)
              compressed = compress(...)
        """
        ...

    def reconstruct_scores(
        self,
        key: str,
        Q: torch.Tensor,   # [n_q, d_head]
    ) -> Optional[torch.Tensor]:
        """ADA entry에서 밀집 Key 없이 어텐션 로짓 계산.

        score[q, t] = K_radius[t] × (Q[q] · dequantize_angle(K_angle[t]))
        반환: [n_q, n_kept] FP32
        """
        ...

    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def get_entry(self, key: str) -> Optional[SphericalKVEntry]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...
    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]: ...

    def memory_reduction_ratio(self) -> float: ...
    def rdr_budget_utilization_stats(self) -> dict:
        """rdr_budget_utilization 분포 통계.
        Returns: {"utilization_mean": float, "utilization_std": float}
        """
        ...
```

---

### 5. COMBNativePICCrossAttentionCache (Activity B-1) — `src/cache/comb_native_pic_cache.py`

**핵심 아이디어**: COMB encoder interleaving으로 위치-독립 KV를 생성해 위치 수정 없이 재사용.

```python
from dataclasses import dataclass
from typing import Dict, Optional
import hashlib
import torch
from src.cache.base import CacheStore


@dataclass
class COMBCrossAttentionKVEntry:
    segment_id: bytes           # SHA256(token_content) — 위치-무관 콘텐츠 해시
    encoder_kv: torch.Tensor    # [n_encoder_layers, n_heads, n_tokens, head_dim]
    cross_attn_mask: torch.Tensor  # [n_tokens] bool
    n_tokens: int
    model_version: str          # COMB 체크포인트 버전


@dataclass
class COMBCacheConfig:
    n_encoder_layers: int = 4   # COMB 인터리브된 인코더 레이어 수
    n_heads: int = 8
    head_dim: int = 64
    max_entries: int = 1000
    seed: int = 42


class COMBNativePICCrossAttentionCache(CacheStore):
    """COMB Native PIC encoder cross-attention KV 위치-독립 캐시 (arXiv:2602.01519).

    COMB 아키텍처에서 encoder 레이어를 decoder-only LLM에 빗살무늬로 인터리브해
    cross-attention으로 세그먼트 KV를 위치-독립적으로 저장. 위치 수정(δ-회전) 불필요.

    세그먼트 식별:
      segment_id = SHA256(token_ids_bytes) — 위치와 무관한 콘텐츠 해시

    CacheStore 인터페이스:
      - put_segment(segment_id, encoder_kv, cross_attn_mask, model_version): 세그먼트 저장
      - put(key, value): plain 저장 (호환성)
      - get(key): encoder_kv 텐서 반환
      - get_entry(segment_id): COMBCrossAttentionKVEntry 전체 반환
    """

    def __init__(self, config: COMBCacheConfig) -> None: ...

    @staticmethod
    def compute_segment_id(token_ids: list) -> bytes:
        """SHA256(token_ids as bytes) — 위치-무관 콘텐츠 해시.

        import struct
        raw = struct.pack(f"{len(token_ids)}I", *token_ids)
        return hashlib.sha256(raw).digest()
        """
        ...

    def put_segment(
        self,
        segment_id: bytes,
        encoder_kv: torch.Tensor,      # [n_enc_layers, n_heads, n_tokens, head_dim]
        cross_attn_mask: torch.Tensor,  # [n_tokens] bool
        model_version: str = "default",
    ) -> None:
        """COMB encoder cross-attention KV 세그먼트 저장."""
        ...

    def get_entry(self, segment_id: bytes) -> Optional[COMBCrossAttentionKVEntry]:
        """세그먼트 조회 — 위치 수정 없이 즉시 반환 (Native PIC)."""
        ...

    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...
```

---

### 6. ThreeTrackArchAwarePICRouter (Activity B-1) — `src/cache/three_track_pic_router.py`

**핵심 아이디어**: model_config를 보고 COMB/MLA/GQA 세 트랙 중 하나로 dispatch.

```python
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import yaml
import torch
from src.cache.base import CacheStore
from src.cache.comb_native_pic_cache import COMBNativePICCrossAttentionCache
from src.cache.irminsul_mla_segment_cache import IrminsulMLASegmentCache


PICTrack = Literal["COMB_NATIVE", "MLA", "GQA_MHA"]


@dataclass
class ThreeTrackRouterConfig:
    pic_arch_registry_path: str = "configs/pic_arch_registry.yaml"
    seed: int = 42


def detect_pic_track(model_config: dict) -> PICTrack:
    """아키텍처 필드로 PIC 트랙 감지.

    감지 우선순위:
      1. "comb_encoder_layers" 필드 존재 → "COMB_NATIVE"
      2. "kv_lora_rank" 존재 AND "qk_rope_head_dim" == 64 → "MLA"
      3. 그 외 → "GQA_MHA"

    configs/pic_arch_registry.yaml도 검사:
      {model_name: pic_track} 매핑이 있으면 필드 감지보다 우선 적용.
    """
    ...


class ThreeTrackArchAwarePICRouter(CacheStore):
    """3-트랙 아키텍처-인식 PIC dispatch 라우터 (Activity B-1).

    트랙 1 (COMB_NATIVE): COMBNativePICCrossAttentionCache — 위치 수정 불필요
    트랙 2 (MLA):         IrminsulMLASegmentCache — δ-회전 위치 수정
    트랙 3 (GQA_MHA):    기존 CacheStore fallback (SegmentedHashCache 등)

    CacheStore 인터페이스:
      - put(key, value): 현재 model_config 기반 적절한 트랙으로 라우팅
      - get(key, model_config, target_position): 트랙별 조회 + 위치 수정
      - hit_rate(): 3-트랙 통합 히트율
    """

    def __init__(
        self,
        comb_cache: COMBNativePICCrossAttentionCache,
        mla_cache: IrminsulMLASegmentCache,
        gqa_cache: CacheStore,
        config: ThreeTrackRouterConfig,
    ) -> None: ...

    def get_routed(
        self,
        segment_id: bytes,
        model_config: dict,
        target_position: int = 0,
    ) -> Optional[torch.Tensor]:
        """트랙 감지 후 적절한 캐시로 라우팅.

        알고리즘:
          track = detect_pic_track(model_config)
          if track == "COMB_NATIVE":
              entry = comb_cache.get_entry(segment_id)
              hit_type = "comb_hit" if entry else "miss"
              return entry.encoder_kv if entry else None

          elif track == "MLA":
              # Irminsul δ-회전 적용
              result = mla_cache.get_with_delta_rotation(segment_id.hex(), target_position)
              hit_type = "mla_hit" if result else "miss"
              return result

          else:  # GQA_MHA
              result = gqa_cache.get(segment_id.hex())
              hit_type = "gqa_hit" if result else "miss"
              return result

          metrics.record(hit_type)
        """
        ...

    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...

    def track_hit_rate_breakdown(self) -> dict:
        """트랙별 히트율 반환.

        Returns:
          {
            "comb_hit_rate": float,
            "mla_hit_rate": float,
            "gqa_hit_rate": float,
            "miss_rate": float,
            "total_hit_rate": float,
          }
        """
        ...
```

---

### 7. ThreeTrackHitRateMetrics (변경) — `src/metrics/hit_rate.py`에 추가

```python
@dataclass
class ThreeTrackHitRateMetrics:
    """3-트랙 PIC 비연속 히트율 지표 (Activity B-1).
    기존 WeightedHitRateMetrics, HitRateMetrics, DistributedHitRateMetrics와 독립적.
    """
    total_lookups: int = 0
    comb_hits: int = 0
    mla_hits: int = 0
    gqa_hits: int = 0

    def record(self, hit_type: str) -> None:
        """hit_type: "comb_hit" | "mla_hit" | "gqa_hit" | "miss"."""
        self.total_lookups += 1
        if hit_type == "comb_hit":
            self.comb_hits += 1
        elif hit_type == "mla_hit":
            self.mla_hits += 1
        elif hit_type == "gqa_hit":
            self.gqa_hits += 1

    def total_hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return (self.comb_hits + self.mla_hits + self.gqa_hits) / self.total_lookups

    def noncontiguous_fraction(self) -> float:
        """COMB + MLA 비연속 히트 / 전체 히트 — 비연속 재사용 비율."""
        total_hits = self.comb_hits + self.mla_hits + self.gqa_hits
        if total_hits == 0:
            return 0.0
        return (self.comb_hits + self.mla_hits) / total_hits

    def reset(self) -> None:
        self.total_lookups = 0
        self.comb_hits = 0
        self.mla_hits = 0
        self.gqa_hits = 0

    def summary(self) -> dict:
        total = max(1, self.total_lookups)
        return {
            "comb_hit_rate": self.comb_hits / total,
            "mla_hit_rate": self.mla_hits / total,
            "gqa_hit_rate": self.gqa_hits / total,
            "miss_rate": 1.0 - self.total_hit_rate(),
            "total_hit_rate": self.total_hit_rate(),
            "noncontiguous_fraction": self.noncontiguous_fraction(),
            "total_lookups": self.total_lookups,
        }
```

---

### 8. COMBSphericalKVPipeline (Cross-1) — `src/engine/comb_spherical_kv_pipeline.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
from src.cache.comb_native_pic_cache import COMBNativePICCrossAttentionCache
from src.cache.spherical_kv_joint_codec import SphericalKVJointCodec, SphericalKVEntry
from src.cache.minimal_intervention_fl_codec import MinimalInterventionFacilityLocationCodec


@dataclass
class COMBSphericalKVEntry:
    """B+C 통합 저장 구조.

    COMB Native PIC × SPHERICAL KV ADA 통합:
      score[t] = r[t] × (Q · direction(θ[t]))  ← 수학적 등가 검증 필수
    """
    segment_id: bytes                    # SHA256(token_content) — 위치-무관
    encoder_K_angle: torch.Tensor        # [n_enc_layers, n_heads, n_kept, d_head] INT8
    encoder_K_radius: torch.Tensor       # [n_enc_layers, n_heads, n_kept] FP16
    encoder_V: torch.Tensor              # [n_enc_layers, n_heads, n_kept, d_head]
    rdr_precision_tiers: list            # 토큰별 정밀도 티어
    keep_mask: torch.Tensor              # [original_n_tokens] bool
    original_n_tokens: int


@dataclass
class COMBSphericalKVPipelineConfig:
    budget_bits_per_token: int = 64     # RDR 토큰당 비트 예산
    enable_facility_location: bool = False  # True: C-2 V-공간 다양성 추가 적용
    lambda_div: float = 0.5             # facility-location λ (enable_facility_location=True 시)
    cdc_chunk_size: int = 128           # 세그먼트 분할 청크 크기
    seed: int = 42


class COMBSphericalKVPipeline:
    """COMB Native PIC × SPHERICAL KV B+C 통합 파이프라인 (Cross-1).

    6단계 처리 흐름:
      Step 1 (CDC 세그먼트 분할, B-1):
        token_ids → 청크 분할 → segment_id = SHA256(chunk_bytes)

      Step 2 (COMB encoder cross-attention 실행, B-1):
        segment에 대해 encoder 레이어 cross-attention KV 계산
        (실제 COMB 모델 없는 테스트 환경: MockCOMBEncoder 사용)

      Step 3 (ADA 분해, C-1):
        K = r × direction(θ) → K_radius, K_angle_int8

      Step 4 (RDR 공동 최적화, C-1):
        attn_weights 기반 토큰-정밀도 공동 결정 → keep_mask, precision_tiers

      Step 5 (facility-location V-공간 다양성, C-2 선택적):
        enable_facility_location=True 시 RDR keep_mask를 facility-location으로 보정

      Step 6 (통합 저장):
        COMBSphericalKVEntry 형태로 CacheStore에 저장

    재사용 시:
      score[t] = K_radius[t] × (Q · dequantize_angle(K_angle[t]))
      위치 수정 불필요 (COMB Native PIC)
    """

    def __init__(
        self,
        comb_cache: COMBNativePICCrossAttentionCache,
        spherical_codec: SphericalKVJointCodec,
        fl_codec: Optional[MinimalInterventionFacilityLocationCodec],
        config: COMBSphericalKVPipelineConfig,
    ) -> None: ...

    def process_segment(
        self,
        token_ids: List[int],
        encoder_kv_raw: torch.Tensor,     # [n_enc_layers, n_heads, n_tokens, d_head] — COMB 출력
        attn_weights: Optional[torch.Tensor] = None,  # [n_tokens] — None → uniform
    ) -> COMBSphericalKVEntry:
        """6단계 처리 후 COMBSphericalKVEntry 반환."""
        ...

    def retrieve_segment(
        self,
        segment_id: bytes,
        Q: torch.Tensor,               # [n_q, d_head]
    ) -> Optional[torch.Tensor]:
        """저장된 entry에서 밀집 Key 없이 어텐션 로짓 계산 후 V 반환.

        score[q, t] = K_radius[t] × (Q[q] · dequantize_angle(K_angle[t]))
        """
        ...

    def verify_ada_comb_equivalence(
        self,
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [n_kv, d_head]
        rtol: float = 1e-3,
    ) -> bool:
        """ADA-COMB 수학적 등가 검증.

        검증:
          score_orig[q,t] = Q[q] · K[t]
          K_radius, K_dir = ada.decompose(K)
          score_ada[q,t] = K_radius[t] × (Q[q] · K_dir[t])
          assert torch.allclose(score_orig, score_ada, rtol=rtol)

        단위 테스트에서 이 메서드를 호출해 수식 등가 검증.
        """
        ...

    def cdc_chunk(self, token_ids: List[int]) -> List[Tuple[bytes, List[int]]]:
        """CDC 청크 분할.

        고정 크기 청킹 (cdc_chunk_size 기준):
          [(segment_id_0, token_ids_0), (segment_id_1, token_ids_1), ...]
        """
        ...
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(C-2 + C-1)를 포함하므로 이 섹션은 **필수**다.

### 검증 전략 개요

| 검증 유형 | 방법 | 허용 오차 |
|----------|------|---------|
| perplexity 프록시 | `attention_output_relative_error()` < 0.01 | ±1% 이내 |
| 태스크 정확도 프록시 | `cosine_similarity_output()` ≥ 0.99 | ±1% 이내 |
| KL 발산 | `attention_kl_divergence()` < 0.015 | 허용 기준 |
| b 스윕 | b ∈ {64, 128, 256} 각각 별도 검증 | 모든 예산에서 ±1% |
| λ 스윕 | λ ∈ {0.25, 0.5, 0.75, 1.0} | 다양성 강도별 검증 |

### C-2 (MinimalInterventionFacilityLocationCodec) 검증

**perplexity 측정**:
- 데이터셋: WikiText-2 프록시 (100개 랜덤 시퀀스, seq_len=512, 합성 토큰)
- 모델: 실제 LLM 없이 `attention_output_relative_error()` + `cosine_similarity_output()` 메트릭으로 대체
- 허용 오차: relative_error < 0.01 (≈ ±1% perplexity 보존 기준)

**태스크 정확도 측정**:
- 벤치마크: MATH-500 프록시 (추론 태스크 중요도 분포 시뮬레이션)
  - 추론 태스크 특성: 토큰 중요도 분포가 집중적 (heavy-tail) → facility-location 효과 검증
  - cosine_sim ≥ 0.99 기준
- 벤치마크: RULER-4K/16K 프록시 (긴 컨텍스트 토큰 선택 일관성)
  - seq_len=4096, 16384 합성 시퀀스에서 keep_mask 안정성
- LongBench 8개 서브태스크 프록시 (다양한 태스크 유형)
  - 8개 중요도 분포 시뮬레이션 → facility-location vs. argmax-top-k 비교

**직접 비교 테스트** (필수):
- `argmax-top-k` vs. `facility-location` 동일 budget_tokens(b=64/128/256) 조건에서
  attention_output_relative_error 비교
- facility-location이 argmax-top-k보다 error가 같거나 낮아야 함 (소예산 b=64 조건에서 특히 검증)

**검증 테스트 파일**: `tests/unit/test_compression_accuracy_2026_05_30.py`

```python
# 주요 테스트 케이스 (C-2 관련)
def test_facility_location_relative_error_below_threshold():
    """b=128, λ=0.5에서 attention_output_relative_error < 0.01."""

def test_facility_location_cosine_sim_above_threshold():
    """b=128, λ=0.5에서 cosine_similarity_output ≥ 0.99."""

def test_facility_location_better_than_topk_small_budget():
    """b=64에서 facility-location error ≤ argmax-top-k error."""

def test_facility_location_budget_sweep():
    """b ∈ {64, 128, 256} 모두에서 relative_error < 0.01."""

def test_facility_location_lambda_sweep():
    """λ ∈ {0.25, 0.5, 0.75, 1.0} 모두에서 cosine_sim ≥ 0.99."""

def test_facility_location_reasoning_task_profile():
    """Heavy-tail 중요도 분포(추론 태스크 시뮬레이션)에서 error < 0.01."""
```

### C-1 (SphericalKVJointCodec) 검증

**perplexity 측정**:
- 데이터셋: WikiText-2 프록시 (100개 랜덤 시퀀스, seq_len=512)
- ADA 단독 / RDR 단독 / ADA+RDR 결합 3-way 비교 필수
- 허용 오차: relative_error < 0.01

**태스크 정확도 측정**:
- RULER-4K/16K 프록시: 긴 시퀀스에서 keep_mask + 각도 코드 조합
  - seq_len=4096에서 cosine_sim ≥ 0.99
- LongBench 8개 서브태스크 프록시
  - 다양한 중요도 분포에서 ADA+RDR 결합 accuracy delta < 1%

**budget_bits 스윕** (필수):
- budget_bits_per_token ∈ {32, 64, 128} (= INT4 8dim, INT4 16dim, INT8 16dim 등가)
- 각 예산 수준에서 relative_error < 0.01 확인

**독립 퇴거+양자화 파이프라인 대비 비교** (필수):
- `SegmentedHashCache + INT8 양자화` (분리 최적화) vs. `SphericalKVJointCodec` (공동 최적화)
- 동일 메모리 예산에서 cosine_sim 비교 — 공동 최적화가 분리 최적화보다 ≥ 동등

**ADA 수학적 정확성 검증** (필수):
- `score_orig = Q @ K.T`
- `score_ada = K_radius * (Q @ dequantize(K_angle).T)` (정규화 보정 포함)
- `torch.allclose(score_orig, score_ada, rtol=1e-2)` 통과

```python
# 주요 테스트 케이스 (C-1 관련)
def test_spherical_kv_ada_only_relative_error():
    """ADA 단독: relative_error < 0.01."""

def test_spherical_kv_rdr_only_relative_error():
    """RDR 단독: relative_error < 0.01."""

def test_spherical_kv_joint_relative_error():
    """ADA+RDR 결합: relative_error < 0.01."""

def test_spherical_kv_joint_cosine_sim():
    """ADA+RDR 결합: cosine_sim ≥ 0.99."""

def test_ada_comb_mathematical_equivalence():
    """score_orig ≈ score_ada (rtol=1e-2)."""

def test_rdr_joint_better_than_independent():
    """동일 예산에서 공동 최적화 cosine_sim ≥ 분리 최적화 cosine_sim."""

def test_budget_bits_sweep():
    """budget_bits_per_token ∈ {32, 64, 128} 모두에서 relative_error < 0.01."""

def test_spherical_kv_longbench_proxy():
    """다양한 중요도 분포 8가지에서 모두 cosine_sim ≥ 0.99."""

def test_accuracy_fallback_triggers_on_violation():
    """cosine_sim < 0.99 시 budget 완화 후 재압축 → cosine_sim ≥ 0.99."""
```

**accuracy fallback 메커니즘** (C-1 필수 안전망):
- `SphericalKVJointCodec.put_compressed()` 내부:
  - 압축 후 `cosine_similarity_output(Q_probe, K_orig, V_orig, K_kept, V_kept)` 계산
  - sim < 0.99 이면 `budget_bits *= 1.5` 후 1회 재시도
  - 재시도 후에도 sim < 0.99 이면 원본 KV 저장 (안전망)

**accuracy fallback 메커니즘** (C-2 필수 안전망):
- `MinimalInterventionFacilityLocationCodec.put_compressed()` 내부:
  - 압축 후 `cosine_similarity_output()` < 0.99 이면 `lambda_div /= 2` 후 재시도
  - 재시도 후에도 실패 시 `budget_tokens *= 2`로 더 많은 토큰 보존

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-30.yaml
experiment:
  date: "2026-05-30"
  activity: "B+C"
  cache_type: "three_track_pic_router"       # Activity B-1 (dispatch)
  compression_method: "facility_location+spherical_kv"  # Activity C
  scheduler_type: "default"                  # Activity A 미포함
  seed: 42

# Activity C-2: MinimalInterventionFacilityLocationCodec
facility_location_codec:
  budget_tokens: 128                  # 보존 토큰 수 (b=128 기본)
  lambda_div: 0.5                     # V-공간 다양성 페널티 가중치
  lambda_auto_scale: true             # lambda_div_eff = lambda_div × (128 / budget_tokens)
  use_triattention_scorer: true       # TriAttention 중요도 점수 재사용
  fallback_to_snapkv: true            # TriAttention 미사용 환경 폴백
  max_entries: 1000

# Activity C-1: SphericalKVADACodec
spherical_kv_ada:
  angle_quantize_bits: 8              # INT8 각도 코드 (INT4 옵션)
  radius_dtype: "float16"

# Activity C-1: SphericalKVRDRCodec
spherical_kv_rdr:
  budget_bits_per_token: 64           # 토큰당 비트 예산
  available_tiers: ["INT4", "INT8", "FP16"]
  distortion_metric: "attn_weight"

# Activity C-1: SphericalKVJointCodec
spherical_kv_joint:
  budget_bits_total: 2048             # 전체 시퀀스 비트 예산
  accuracy_fallback_threshold: 0.01  # cosine_sim < 0.99 시 budget 완화
  max_entries: 1000

# Activity B-1: COMBNativePICCrossAttentionCache
comb_cache:
  n_encoder_layers: 4
  n_heads: 8
  head_dim: 64
  max_entries: 1000

# Activity B-1: ThreeTrackArchAwarePICRouter
three_track_router:
  pic_arch_registry_path: "configs/pic_arch_registry.yaml"

# Cross-1: COMBSphericalKVPipeline
comb_spherical_kv_pipeline:
  budget_bits_per_token: 64
  enable_facility_location: false     # C-2 V-공간 다양성 통합 옵션
  lambda_div: 0.5
  cdc_chunk_size: 128

# accuracy 검증 임계값 (평가에서 직접 참조)
accuracy_thresholds:
  max_relative_error: 0.01            # attention_output_relative_error < 0.01
  min_cosine_sim: 0.99                # cosine_similarity_output ≥ 0.99
  max_kl_divergence: 0.015            # attention_kl_divergence < 0.015
  budget_tokens_sweep: [64, 128, 256]
  lambda_div_sweep: [0.25, 0.5, 0.75, 1.0]
  budget_bits_sweep: [32, 64, 128]

# 측정 지표 저장 경로
metrics:
  output_dir: "results/2026-05-30"
  metrics_file: "results/2026-05-30/metrics.json"
```

```yaml
# configs/pic_arch_registry.yaml
# {model_name_pattern → pic_track} 레지스트리
# detect_pic_track()이 model_config.model_name으로 조회
# 등록 없으면 필드 감지(comb_encoder_layers / kv_lora_rank) 폴백
architectures:
  # 예시 (실제 모델명으로 교체 필요):
  # - pattern: "comb-*"
  #   pic_track: "COMB_NATIVE"
  # - pattern: "deepseek-*"
  #   pic_track: "MLA"
  # - pattern: "llama-*"
  #   pic_track: "GQA_MHA"
  []  # 기본: 빈 레지스트리 (필드 감지만 사용)
```

---

## results/2026-05-30/metrics.json 저장 지표

```json
{
  "experiment_date": "2026-05-30",
  "activity": "B+C",

  "compression_c2_facility_location": {
    "memory_reduction_ratio": null,
    "diversity_gain_mean": null,
    "diversity_gain_std": null,
    "facility_location_vs_topk_error_delta": null,
    "relative_error_b64": null,
    "relative_error_b128": null,
    "relative_error_b256": null,
    "cosine_sim_b128": null,
    "kl_divergence_b128": null,
    "accuracy_delta_within_1pct": null
  },

  "compression_c1_spherical_kv": {
    "memory_reduction_ratio_ada": null,
    "memory_reduction_ratio_joint": null,
    "rdr_budget_utilization_mean": null,
    "rdr_budget_utilization_std": null,
    "relative_error_ada_only": null,
    "relative_error_rdr_only": null,
    "relative_error_joint": null,
    "cosine_sim_joint": null,
    "ada_comb_equivalence_rtol": null,
    "joint_vs_independent_cosine_sim_delta": null,
    "accuracy_delta_within_1pct": null
  },

  "noncontiguous_cache_b1": {
    "comb_hit_rate": null,
    "mla_hit_rate": null,
    "gqa_hit_rate": null,
    "miss_rate": null,
    "total_hit_rate": null,
    "noncontiguous_fraction": null,
    "noncontiguous_30pct_target_met": null
  },

  "cross1_comb_spherical_kv": {
    "combined_memory_reduction_pct": null,
    "combined_throughput_improvement_pct": null,
    "single_activity_b_improvement_pct": null,
    "single_activity_c_improvement_pct": null,
    "ttft_overhead_pct": null,
    "ada_comb_mathematical_equivalence": null
  },

  "kv_memory": {
    "baseline_kv_bytes": null,
    "c2_codec_kv_bytes": null,
    "c1_codec_kv_bytes": null,
    "memory_reduction_c2_pct": null,
    "memory_reduction_c1_pct": null
  }
}
```

---

## 테스트 요구사항

### 필수 단위 테스트

- [ ] `tests/unit/test_minimal_intervention_fl_codec.py`
  - `test_cache_store_interface_all_methods()`: CacheStore 추상 메서드 전부 동작
  - `test_facility_location_returns_k_tokens()`: budget_tokens개 토큰 정확히 선택
  - `test_facility_location_diversity_higher_than_topk()`: 선택된 V 벡터 다양성 ≥ topk
  - `test_lambda_auto_scale_small_budget()`: b=64 시 lambda_div_eff > lambda_div (auto_scale)
  - `test_put_compressed_stores_entry()`: put_compressed → get 동일 shape 텐서 반환
  - `test_get_returns_none_on_miss()`: 저장 안 된 key → None
  - `test_hit_rate_tracking()`: put/get 후 hit_rate() 계산
  - `test_evict_reduces_memory()`: evict() 후 memory_bytes() 감소
  - `test_importance_mask_correct()`: kept_indices ↔ get_importance_mask() bool 마스크 일치
  - `test_fallback_snapkv_when_no_importance()`: importance_scores=None → norm 기반 선택
  - `test_deterministic_with_seed()`: 동일 seed + 입력 → 동일 kept_indices

- [ ] `tests/unit/test_spherical_kv_joint_codec.py`
  - `test_ada_decompose_reconstruct()`: decompose → radius × direction ≈ 원본 K (rtol=1e-4)
  - `test_ada_quantize_dequantize_roundtrip()`: quantize → dequantize → 단위 벡터 (rtol=1e-2)
  - `test_ada_reconstruct_score_equivalence()`: score_ada ≈ score_orig (rtol=1e-2)
  - `test_rdr_optimize_within_budget()`: RDRResult.bits_used ≤ budget_bits
  - `test_rdr_optimize_keeps_high_attn_tokens()`: 높은 attn_weight 토큰 우선 보존
  - `test_joint_put_compressed_stores_entry()`: put_compressed → get_entry 정상
  - `test_joint_memory_reduction()`: memory_reduction_ratio() > 0.0
  - `test_joint_reconstruct_scores()`: reconstruct_scores()가 score 텐서 반환
  - `test_accuracy_fallback_triggers()`: cosine_sim < 0.99 시 재압축 트리거 확인
  - `test_cache_store_interface_all_methods()`: CacheStore 추상 메서드 전부 동작
  - `test_importance_mask_from_keep_mask()`: keep_mask ↔ get_importance_mask() 일치
  - `test_deterministic_with_seed()`: 동일 seed + 입력 → 동일 keep_mask

- [ ] `tests/unit/test_three_track_pic_router.py`
  - `test_detect_pic_track_comb_native()`: comb_encoder_layers 필드 → "COMB_NATIVE"
  - `test_detect_pic_track_mla()`: kv_lora_rank + qk_rope_head_dim=64 → "MLA"
  - `test_detect_pic_track_gqa_mha()`: 필드 없음 → "GQA_MHA"
  - `test_router_cache_store_interface()`: CacheStore 추상 메서드 전부 동작
  - `test_comb_track_returns_hit_on_cache()`: COMB 세그먼트 저장 후 comb_hit 반환
  - `test_mla_track_routes_to_irminsul()`: MLA 모델 → IrminsulMLASegmentCache 경로
  - `test_gqa_track_routes_to_fallback()`: GQA 모델 → fallback CacheStore 경로
  - `test_miss_on_empty_cache()`: 빈 캐시 → miss 반환
  - `test_hit_rate_breakdown_sums_to_one()`: 4개 비율 합 = 1.0
  - `test_noncontiguous_fraction_comb_mla_only()`: noncontiguous_fraction = (comb + mla) / total_hits
  - `test_pic_registry_overrides_field_detection()`: registry에 모델명 있으면 필드 감지 우선

- [ ] `tests/unit/test_compression_accuracy_2026_05_30.py` (Activity C 필수)
  - **C-2 테스트**:
    - `test_c2_facility_location_relative_error_b128()`: b=128, λ=0.5 → error < 0.01
    - `test_c2_facility_location_cosine_sim_b128()`: b=128 → cosine_sim ≥ 0.99
    - `test_c2_facility_location_better_than_topk_b64()`: b=64 → error(FL) ≤ error(topk)
    - `test_c2_budget_sweep_relative_error()`: b ∈ {64, 128, 256} 모두 error < 0.01
    - `test_c2_lambda_sweep_cosine_sim()`: λ ∈ {0.25, 0.5, 0.75, 1.0} 모두 cosine_sim ≥ 0.99
    - `test_c2_kl_divergence_below_threshold()`: b=128 → KL < 0.015
    - `test_c2_reasoning_profile_accuracy()`: heavy-tail 중요도 분포 → error < 0.01
  - **C-1 테스트**:
    - `test_c1_ada_only_relative_error()`: ADA 단독 → error < 0.01
    - `test_c1_rdr_only_relative_error()`: RDR 단독 → error < 0.01
    - `test_c1_joint_relative_error()`: ADA+RDR → error < 0.01
    - `test_c1_joint_cosine_sim()`: ADA+RDR → cosine_sim ≥ 0.99
    - `test_c1_ada_mathematical_equivalence()`: score_ada ≈ score_orig (rtol=1e-2)
    - `test_c1_joint_vs_independent_cosine_sim()`: 공동 ≥ 분리 최적화
    - `test_c1_budget_bits_sweep()`: budget ∈ {32, 64, 128} → error < 0.01
    - `test_c1_longbench_proxy_8tasks()`: 8가지 중요도 분포 → cosine_sim ≥ 0.99
    - `test_c1_accuracy_fallback_on_violation()`: sim < 0.99 → budget 완화 재압축
  - **Cross-1 테스트**:
    - `test_cross1_ada_comb_pipeline_accuracy()`: COMBSphericalKVPipeline error < 0.01
    - `test_cross1_combined_memory_reduction()`: ADA + RDR 결합 메모리 감소 ≥ 단일 기법

### 필수 통합 테스트

- [ ] `tests/integration/test_comb_spherical_kv_pipeline_e2e.py`
  - `test_e2e_pipeline_full_flow()`: 6단계 처리 흐름 완전 실행 (MockCOMBEncoder 환경)
  - `test_e2e_segment_reuse_no_position_correction()`: 동일 세그먼트 다른 위치에서 재사용 → 위치 수정 없음 확인
  - `test_e2e_memory_reduction_combined()`: B+C 결합 메모리 감소 검증
  - `test_e2e_hit_rate_comb_track()`: ThreeTrackHitRateMetrics.comb_hit_rate > 0
  - `test_e2e_noncontiguous_fraction_above_30pct()`: noncontiguous_fraction ≥ 0.30
  - `test_e2e_cachestore_interface_compat()`: COMBNativePICCrossAttentionCache를
    InferenceRunner가 CacheStore로 사용 가능
  - `test_e2e_ada_comb_equivalence_validated()`: verify_ada_comb_equivalence() → True
  - `test_e2e_accuracy_within_1pct()`: 파이프라인 전체 적용 후 relative_error < 0.01
  - `test_e2e_throughput_improvement_vs_baseline()`: Cross-1 처리량 향상 > 단일 Activity

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 100% 통과**: 위 명시된 모든 단위 테스트 케이스 통과
2. **통합 테스트 100% 통과**: `test_comb_spherical_kv_pipeline_e2e.py` 전체 통과
3. **기존 테스트 회귀 없음**: 이전 사이클 구현의 모든 단위·통합 테스트 계속 통과
4. **evaluation_criteria.md §4 (Activity C) 필수 항목 모두 Pass**:
   - Accuracy 보존: perplexity 변화 ±1% 이내 (`attention_output_relative_error` < 0.01) (필수)
   - Accuracy 보존 태스크: downstream 태스크 정확도 변화 ±1% 이내 (`cosine_sim` ≥ 0.99) (필수)
   - KV Memory Reduction ≥ −30% (높음)
   - 압축 오버헤드 TTFT +10% 이내 (높음)
5. **evaluation_criteria.md §3 (Activity B) 기준 충족**:
   - 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상 (높음)
   - 비연속 세그먼트 히트율 전체 히트의 30% 이상 (높음)
   - KV Memory Footprint 베이스라인 대비 +20% 이내 (높음)
6. **evaluation_criteria.md §5 (크로스 조합) 기준**:
   - Accuracy 보존 복합 적용 후에도 ±1% 이내 (필수)
   - 복합 Throughput 향상 단일 Activity 대비 추가 +5% 이상 (높음)
7. **CacheStore 인터페이스 준수**: MinimalInterventionFacilityLocationCodec,
   SphericalKVJointCodec, COMBNativePICCrossAttentionCache, ThreeTrackArchAwarePICRouter
   모두 CacheStore 추상 메서드 전부 구현
8. **accuracy fallback 메커니즘 동작**: cosine_sim < 0.99 시 자동 재압축 확인
9. **ADA-COMB 수학적 등가 검증**: `verify_ada_comb_equivalence()` 테스트 통과 (rtol=1e-2)
10. **설정 파일 존재**: `configs/experiments/2026-05-30.yaml`, `configs/pic_arch_registry.yaml`
11. **metrics.json 생성**: `results/2026-05-30/metrics.json`에 모든 지표 기록
12. **시드 고정 재현성**: seed=42로 동일 결과 재현 가능

---

## 구현 우선순위 순서

1. **C-2**: `FacilityLocationConfig` + `FLKVEntry` + `MinimalInterventionFacilityLocationCodec` (facility_location_selection 핵심 알고리즘 포함)
2. **C-2 accuracy 검증**: `test_compression_accuracy_2026_05_30.py` C-2 부분 즉시 작성 + 통과 확인
3. **C-1 ADA**: `ADAConfig` + `ADAKVEntry` + `SphericalKVADACodec` (decompose, quantize_angle, reconstruct_score)
4. **C-1 RDR**: `RDRConfig` + `RDRResult` + `SphericalKVRDRCodec` (optimize 그리디 알고리즘)
5. **C-1 통합**: `SphericalKVJointConfig` + `SphericalKVEntry` + `SphericalKVJointCodec` (accuracy fallback 포함)
6. **C-1 accuracy 검증**: `test_compression_accuracy_2026_05_30.py` C-1 부분 + 3-way 비교 + ADA 수학적 등가
7. **지표**: `ThreeTrackHitRateMetrics` (`src/metrics/hit_rate.py` 추가)
8. **B-1 COMB 캐시**: `COMBCrossAttentionKVEntry` + `COMBCacheConfig` + `COMBNativePICCrossAttentionCache`
9. **B-1 라우터**: `detect_pic_track()` + `ThreeTrackRouterConfig` + `ThreeTrackArchAwarePICRouter`
10. **Cross-1**: `COMBSphericalKVEntry` + `COMBSphericalKVPipelineConfig` + `COMBSphericalKVPipeline` (verify_ada_comb_equivalence 포함)
11. **설정 파일**: `configs/experiments/2026-05-30.yaml`, `configs/pic_arch_registry.yaml`
12. **단위 테스트 전부**: C-2 → C-1 → B-1 → Cross-1 순서
13. **통합 테스트**: `test_comb_spherical_kv_pipeline_e2e.py`

---

## 보존 파일 (수정 금지)

이전 사이클 구현 파일은 수정하지 않는다:

- `src/cache/irminsul_mla_segment_cache.py` (05-26 B-1) — ThreeTrackArchAwarePICRouter에서 import만
- `src/cache/arch_aware_noncontiguous_router.py` (05-26) — 기존 2-트랙 라우터 보존
- `src/cache/cdc_content_hash_interface.py` (05-26 B-2) — CDC 주소 체계 재사용만
- `src/cache/mla_two_axis_compression_codec.py` (05-26 C-1)
- `src/scheduler/objectcache_s3_tier_router.py` (05-26 A-1)
- `src/engine/irminsul_objectcache_pipeline.py` (05-26 Cross-1)
- `src/cache/indexmem_eviction_codec.py` (05-27 C-1)
- `src/cache/indexmem_soft_hit_segment_cache.py` (05-27 B-1)
- `src/engine/indexmem_bc_pipeline.py` (05-27 Cross-1)
- `src/cache/pegaflow_kv_connector.py` (05-28 A-2)
- `src/cache/pegaflow_irminsul_distributed_cache.py` (05-28 B-1)
- `src/scheduler/hexagent_workflow_scheduler.py` (05-28 A-1)
- `src/scheduler/pegaflow_rdma_router.py` (05-28 A-2)
- `src/engine/hexagent_irminsul_pegaflow_pipeline.py` (05-28 Cross-1)
- `src/cache/triattention_pre_rope_kv_selector_codec.py` — ThreeTrackArchAwarePICRouter 내에서 importance_scores 재사용만 (수정 금지)
- 기타 모든 이전 사이클 파일 — 기존 단위·통합 테스트 회귀 없이 통과해야 한다.

**주의**: `src/metrics/hit_rate.py`에는 `ThreeTrackHitRateMetrics` 클래스만 추가한다.
기존 `WeightedHitRateMetrics`, `HitRateMetrics`, `DistributedHitRateMetrics` 클래스와 메서드는 수정하지 않는다.
