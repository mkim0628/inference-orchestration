<!-- 변경 이유 (이전 Spec.md: 2026-05-28 대비):
이전 사이클(2026-05-28)은 A+B 조합이었다:
  - A-1 HexAGeTWorkflowHorizonAwareKVCapacityScheduler (DAG 워크플로우 스케줄러)
  - A-2 PegaFlowKVConnector + PegaFlowRDMACrossNodeRouter (GIL-free Rust KV + RDMA 라우터)
  - B-1 PegaFlowIrminsulDistributedSegmentCache (분산 비연속 세그먼트 캐시)
  - Cross-1 HexAGeTIrminsulPegaFlowPipeline (A+B 통합 파이프라인)

이번 사이클(2026-05-29)은 B+C 조합으로 전환된다.
핵심 전환:
  - Activity C 최우선: WynerZivAdaptiveWindowEviction — Wyner-Ziv Bounds(2605.25085)의
    다항식 절단-민감도 함수를 런타임 온라인 추정하고 이론 하한 W*(ε)로 슬라이딩 윈도우
    크기를 동적으로 결정. 경험적 고정 윈도우를 이론 근거 기반 적응형 윈도우로 대체하는
    최초의 C 기법. accuracy-preserving ±1% 이론 보장 포함.
  - Activity B 2순위: GroundedSafetyGatedSegmentCache — GroundedCache(2605.27494)의
    4-게이트(콘텐츠 유사도·증거 중첩·소스 버전 유효성·어텐션 기여 지지) 안전성 프레임워크를
    비연속 KV 세그먼트 재사용에 적용. 의미적으로 안전한 세그먼트만 재사용 허용.
  - Cross-1 B+C 통합 파이프라인: 4-게이트 안전 재사용 + Wyner-Ziv 이론 윈도우의 결합으로
    "의미 안전 재사용 + 이론 하한 유지"의 2-레이어 정확도 보존 스택.

주요 신규 파일:
1. [신규] src/cache/wyner_ziv_adaptive_window_eviction.py (C-1 Wyner-Ziv 이론 윈도우)
2. [신규] src/cache/grounded_safety_gated_cache.py (B-1 4-게이트 안전 필터)
3. [신규] src/engine/grounded_wyner_ziv_bc_pipeline.py (Cross-1 B+C 통합 파이프라인)
4. [신규] configs/experiments/2026-05-29.yaml
5. [신규] configs/wyner_ziv_window_config.yaml (레이어별 α_l/C_l 초기값)
6. [신규] tests/unit/test_wyner_ziv_eviction.py
7. [신규] tests/unit/test_grounded_safety_gates.py
8. [신규] tests/unit/test_compression_accuracy.py (Activity C 필수)
9. [신규] tests/integration/test_grounded_wyner_ziv_bc_pipeline_e2e.py
10. [변경] src/metrics/hit_rate.py — SafetyGatedHitRateMetrics 클래스 추가
11. [변경] src/metrics/memory.py — WynerZivWindowMetrics 클래스 추가
12. [보존] 모든 이전 사이클 구현 파일 수정 금지.
-->

# Spec — 2026-05-29: GroundedSafetyGatedSegmentCache (B-1) + WynerZivAdaptiveWindowEviction (C-1) (B+C)

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-29.md`

**최우선 구현 타겟**: C-1 `WynerZivAdaptiveWindowEviction`
**2순위 구현 타겟**: B-1 `GroundedSafetyGatedSegmentCache`
**3순위 구현 타겟**: Cross-1 `GroundedWynerZivBCPipeline` (B+C 통합)

**해결하려는 문제**:

- **Activity C (Wyner-Ziv 이론 하한 기반 적응형 슬라이딩 윈도우)**:
  기존 슬라이딩 윈도우 KV 퇴거(StreamingLLM, H2O, SnapKV, IndexMem 등)는 윈도우 크기를
  경험적으로 고정하며 정확도 예산 달성 이론적 근거가 없었다. Wyner-Ziv Bounds(2605.25085)의
  핵심 발견 — 컨텍스트 절단 민감도가 다항식 감쇠 `S(d) = C × d^{-α}`를 따른다 — 을
  런타임에 온라인 적합하고, 이 함수에서 도출한 이론 하한 `W*(ε) = (C/ε)^(1/α)`로 윈도우
  크기를 동적으로 결정한다. 경험적 고정 윈도우 대비 메모리를 추가로 30~50% 절감하면서
  accuracy delta ±1% 이내를 이론적으로 보장한다.

- **Activity B (GroundedCache 4-게이트 비연속 재사용 안전 필터)**:
  기존 비연속 KV 세그먼트 재사용 기법(Irminsul, KV Packet, IndexMem Soft Hit 등)은
  콘텐츠 해시 일치 → 재사용의 이진 판단을 사용하며 의미적 정합성 검증이 없었다.
  GroundedCache(2605.27494)의 4-게이트 안전성 프레임워크를 세그먼트 레벨로 적용해
  4개 게이트 모두 통과한 경우에만 재사용을 허용한다. 비연속 재사용의 accuracy-preserving
  조건을 구조적으로 강화하는 최초의 B 기법이다.

- **Cross-1 (B+C 통합 파이프라인)**:
  4-게이트 안전 재사용 + Wyner-Ziv 이론 윈도우를 "의미 안전 재사용 → 이론 하한 유지"의
  2-레이어 정확도 보존 스택으로 통합한다. 동일 ε 예산 공유 모니터링 포함.

---

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│            GroundedWynerZiv B+C Integration Pipeline            │
│                                                                 │
│  요청 수신 (토큰 시퀀스)                                          │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   GroundedSafetyGatedSegmentCache (Activity B-1)         │   │
│  │                                                          │   │
│  │  재사용 후보 세그먼트 조회                                  │   │
│  │       │                                                  │   │
│  │  Gate 1 — Content Similarity (n-gram Jaccard ≥ 0.85)    │   │
│  │       │ 실패 → 즉시 거부 (재계산)                          │   │
│  │  Gate 2 — Evidence Overlap (TF-IDF Jaccard ≥ 0.60)      │   │
│  │       │ 실패 → partial_reuse (c_KV만 재사용)              │   │
│  │  Gate 3 — Source Version Validity (버전 해시 일치)         │   │
│  │       │ 실패 → stale_segment (즉시 퇴거)                  │   │
│  │  Gate 4 — Attention Contribution Support (cosine ≥ 0.50) │   │
│  │       │ 실패 → partial_reuse (c_KV만 재사용)              │   │
│  │       ↓ 전체 통과                                         │   │
│  │  safe_reuse → KV 반환                                    │   │
│  └──────────────────────────────┬───────────────────────────┘   │
│                                 │                               │
│                                 ▼                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   WynerZivAdaptiveWindowEviction (Activity C-1)          │   │
│  │                                                          │   │
│  │  Online Polynomial Sensitivity Fitting                   │   │
│  │    S_l(d) ~ C_l × d^{-α_l}  (EMA 온라인 갱신)            │   │
│  │       ↓                                                  │   │
│  │  W*(ε) = (C_l / ε)^(1/α_l) per layer                   │   │
│  │       ↓                                                  │   │
│  │  Sliding Window Eviction (W*(ε) 외부 토큰 퇴거)           │   │
│  │  Layer-wise differential window sizes                    │   │
│  │       ↓                                                  │   │
│  │  Fallback: 실측 delta > ε → W* 2× 증가                   │   │
│  └──────────────────────────────┬───────────────────────────┘   │
│                                 │                               │
│                                 ▼                               │
│  SafetyGatedHitRateMetrics + WynerZivWindowMetrics              │
│  {safe_reuse_hit, partial_reuse_hit, gate_rejected,             │
│   stale_evicted, miss} + {w_star_per_layer, alpha_ema}          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling (이번 사이클 미포함)
- [x] Activity B: Non-Contiguous KV Cache Reuse — GroundedSafetyGatedSegmentCache (B-1)
- [x] Activity C: KV Cache Compression — WynerZivAdaptiveWindowEviction (C-1)

---

## 목표

- [ ] 목표 1: KV Cache Memory Reduction 고정 윈도우(W=512) 대비 −35% 이상 (evaluation_criteria.md §4)
- [ ] 목표 2: Compression Accuracy Delta perplexity ±1% 이내 (evaluation_criteria.md §4 필수)
- [ ] 목표 3: Non-Contiguous Safety Gate Pass Rate ≥ 80% — safe_reuse_hit / (safe_reuse_hit + gate_rejected + stale_evicted) (evaluation_criteria.md §3)
- [ ] 목표 4: Non-Contiguous Cache Hit Rate 전체 히트의 30% 이상이 비연속 구간에서 발생 (evaluation_criteria.md §3)
- [ ] 목표 5: W*(ε) 이론 하한 정확성 — S_l(W*) ≤ ε 조건 시뮬레이션 검증 (evaluation_criteria.md §4)
- [ ] 목표 6: Cross-1 복합 Memory Reduction 단일 C-1 대비 추가 −10% 이상 (evaluation_criteria.md §5)
- [ ] 목표 7: TTFT p50 베이스라인 대비 +5% 이내 — 게이트 검사 오버헤드 포함 (evaluation_criteria.md §1)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/wyner_ziv_adaptive_window_eviction.py` | C-1 | CacheStore 구현; 온라인 다항식 민감도 적합(EMA); W*(ε) 이론 하한 계산; 레이어별 슬라이딩 윈도우; Fallback |
| `src/cache/grounded_safety_gated_cache.py` | B-1 | CacheStore 구현; SegmentedHashCache 래핑; 4-게이트 lazy 안전성 검증; 5단계 히트율 집계; partial_reuse / stale_evicted 분기 |
| `src/engine/grounded_wyner_ziv_bc_pipeline.py` | Cross-1 | B+C 통합 파이프라인; 공유 ε 예산 모니터링; 샘플 배치 perplexity 측정; Fallback 자동 트리거 |
| `configs/experiments/2026-05-29.yaml` | 공통 | 이번 사이클 실험 설정 |
| `configs/wyner_ziv_window_config.yaml` | C-1 | 레이어별 (C_l_init, alpha_l_init, window_size) 초기값 |
| `tests/unit/test_wyner_ziv_eviction.py` | C-1 | 다항식 적합, W* 계산, ε 예산 준수, 레이어별 윈도우 단위 테스트 (≥7개) |
| `tests/unit/test_grounded_safety_gates.py` | B-1 | 각 게이트 독립 테스트, fail-fast 순서, 전체 파이프라인 테스트 (≥7개) |
| `tests/unit/test_compression_accuracy.py` | C-1 | perplexity ±1% 시뮬레이션, W* fallback 동작, accuracy budget 준수 (≥5개) |
| `tests/integration/test_grounded_wyner_ziv_bc_pipeline_e2e.py` | Cross-1 | B+C 통합 파이프라인 엔드-투-엔드 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/metrics/hit_rate.py` | `SafetyGatedHitRateMetrics` 클래스 추가. 5단계 히트 유형 집계. 기존 클래스 수정 금지 |
| `src/metrics/memory.py` | `WynerZivWindowMetrics` 클래스 추가. 레이어별 W*_l 시계열 + α_l_ema 추적. 기존 클래스 수정 금지 |

---

## 알고리즘 상세

### 1. WynerZivAdaptiveWindowEviction (Activity C-1) — `src/cache/wyner_ziv_adaptive_window_eviction.py`

#### 자료구조 및 설정

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from collections import OrderedDict
from src.cache.base import CacheStore


@dataclass
class WynerZivConfig:
    accuracy_budget: float = 0.01          # ε: perplexity delta 허용 상한 (1%)
    min_window_size: int = 64              # W* 하한 (최소 유지 토큰 수)
    max_window_size: int = 4096            # W* 상한 (메모리 예산 상한)
    fitting_interval: int = 1000           # W* 재계산 주기 (배치 수)
    sensitivity_sample_ratio: float = 0.05 # 민감도 측정 배치 비율 (5%)
    ema_gamma: float = 0.1                 # α_l EMA 갱신 계수 (슬로우 갱신)
    warmup_batches: int = 100              # α_l/C_l 초기화 워밍업 배치 수
    kv_pressure_threshold: float = 0.8    # KV 예산 압박 임계값
    epsilon_relaxation_factor: float = 2.0 # 압박 시 ε 완화 배율
    n_layers: int = 32                     # 모델 레이어 수
    truncation_distances: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )                                      # 민감도 측정 절단 거리 샘플
    window_config_path: str = "configs/wyner_ziv_window_config.yaml"
    seed: int = 42
```

#### CacheStore 구현 및 핵심 메서드

```python
class WynerZivAdaptiveWindowEviction(CacheStore):
    """Wyner-Ziv 다항식 민감도 기반 이론-인도 적응형 슬라이딩 윈도우 KV 퇴거 캐시.

    온라인 적합: 각 레이어 l의 S_l(d) = C_l × d^{-α_l} 파워 법칙 EMA 갱신.
    이론 하한:  W*_l(ε) = (C_l / ε)^(1/α_l)  →  전체 W* = max_l W*_l
    슬라이딩 윈도우: 각 레이어의 토큰 KV를 OrderedDict(시간순)로 유지.
                    len > window_size[l] 이면 가장 오래된 토큰 퇴거.
    Fallback: 샘플 배치에서 실측 perplexity_delta > ε → W* 2× 증가.
    """

    def __init__(self, config: WynerZivConfig) -> None:
        # 레이어별 상태
        self._alpha_ema: Dict[int, float] = {}    # α_l EMA
        self._c_ema: Dict[int, float] = {}         # C_l EMA
        self._window_size: Dict[int, int] = {}     # 현재 W*_l
        self._kv_store: Dict[int, OrderedDict] = {} # layer → {key → tensor}
        # 통계
        self._hits: int = 0
        self._misses: int = 0
        self._batch_count: int = 0
        self._warmup_done: bool = False
        self._config = config
        self._init_from_config()

    def _init_from_config(self) -> None:
        """wyner_ziv_window_config.yaml에서 (C_l_init, alpha_l_init) 로드."""
        ...

    # ---- CacheStore 추상 메서드 구현 ----

    def put(
        self,
        key: str,
        value: torch.Tensor,
        layer_idx: int = 0,
        perplexity_delta: Optional[float] = None,
    ) -> None:
        """레이어 l의 슬라이딩 윈도우에 KV 저장.

        알고리즘:
          window = self._kv_store[layer_idx]
          window[key] = value  (시간순 OrderedDict 끝에 추가)
          if perplexity_delta is not None:
              self.update_sensitivity_model(perplexity_delta, len(window))
          while len(window) > self._window_size[layer_idx]:
              self._evict_oldest(layer_idx)
        """
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        """모든 레이어 윈도우에서 key 검색 (첫 번째 히트 반환)."""
        ...

    def evict(self) -> int:
        """가장 큰 레이어의 가장 오래된 토큰 퇴거. 해제된 bytes 반환."""
        ...

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(
            v.nbytes
            for layer_store in self._kv_store.values()
            for v in layer_store.values()
        )

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    # ---- Wyner-Ziv 핵심 API ----

    def update_sensitivity_model(
        self,
        perplexity_delta: float,
        truncation_length: int,
        layer_idx: int = 0,
    ) -> None:
        """파워 법칙 EMA 갱신.

        알고리즘:
          # 측정된 S_l(d) = perplexity_delta (KL 발산 근사)
          # d = truncation_length
          # 파워 법칙: log S = log C - α × log d
          # 단일 측정으로부터 α, C 추정:
          #   α_measured = -log(perplexity_delta) / log(truncation_length)  (C=1 가정 초기화)
          #   C_measured  = perplexity_delta × truncation_length^α_measured
          alpha_measured = max(0.1, -math.log(max(perplexity_delta, 1e-9)) /
                               math.log(max(truncation_length, 1)))
          c_measured = perplexity_delta * (truncation_length ** alpha_measured)
          # EMA 갱신 (슬로우 갱신으로 분산 억제)
          γ = self._config.ema_gamma
          self._alpha_ema[layer_idx] = (
              γ * alpha_measured + (1 - γ) * self._alpha_ema.get(layer_idx, 1.5)
          )
          self._c_ema[layer_idx] = (
              γ * c_measured + (1 - γ) * self._c_ema.get(layer_idx, 1.0)
          )
        """
        ...

    def compute_optimal_window(
        self,
        layer_idx: int,
        epsilon: Optional[float] = None,
    ) -> int:
        """W*_l(ε) = (C_l / ε)^(1/α_l) 계산.

        알고리즘:
          ε = epsilon or self._config.accuracy_budget
          α = self._alpha_ema.get(layer_idx, 1.5)
          C = self._c_ema.get(layer_idx, 1.0)
          if α <= 0:
              return self._config.max_window_size
          w_star = int((C / ε) ** (1.0 / α))
          return max(self._config.min_window_size,
                     min(w_star, self._config.max_window_size))

        전체 W* = max_l W*_l(ε) — 가장 민감한 레이어 기준 보수적 결정.
        레이어별 독립 적용 시 window_size[l] = W*_l(ε).
        """
        ...

    def recompute_all_windows(
        self,
        epsilon: Optional[float] = None,
    ) -> Dict[int, int]:
        """모든 레이어의 W*_l 재계산 후 self._window_size 갱신.

        fitting_interval 배치마다 자동 호출.
        KV 압박(kv_pressure > threshold) 시 ε_relaxed = ε × epsilon_relaxation_factor 적용.
        """
        ...

    def trigger_fallback_if_needed(
        self,
        observed_perplexity_delta: float,
    ) -> bool:
        """실측 perplexity_delta > ε 이면 W* 2× 증가 Fallback 실행.

        Returns:
          True if fallback was triggered, False otherwise.
        알고리즘:
          if observed_perplexity_delta > self._config.accuracy_budget:
              for layer_idx in self._window_size:
                  self._window_size[layer_idx] = min(
                      self._window_size[layer_idx] * 2,
                      self._config.max_window_size
                  )
              return True
          return False
        """
        ...

    def get_stats(self) -> dict:
        """현재 상태 요약 반환.

        Returns:
          {
            "window_sizes": {layer_idx: w_star},
            "alpha_ema": {layer_idx: alpha},
            "c_ema": {layer_idx: c},
            "hit_rate": float,
            "memory_bytes": int,
            "batch_count": int,
            "warmup_done": bool,
          }
        """
        ...
```

---

### 2. GroundedSafetyGatedSegmentCache (Activity B-1) — `src/cache/grounded_safety_gated_cache.py`

#### 자료구조 및 설정

```python
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple
import torch
import hashlib
from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


HitOutcome = Literal[
    "safe_reuse_hit",     # 4-게이트 전부 통과
    "partial_reuse_hit",  # Gate 2 또는 Gate 4 부분 실패 → c_KV만 재사용
    "gate_rejected",      # Gate 1 실패 → 완전 거부 (재계산)
    "stale_evicted",      # Gate 3 실패 → 버전 불일치, 세그먼트 퇴거
    "miss",               # 캐시에 없음
]


@dataclass
class GateThresholds:
    content_sim_threshold: float = 0.85    # Gate 1: n-gram Jaccard 최소값
    evidence_overlap_threshold: float = 0.60  # Gate 2: 키워드 Jaccard 최소값
    # Gate 3: 버전 해시 일치 여부 (binary, 임계값 없음)
    attn_support_threshold: float = 0.50   # Gate 4: cosine similarity 최소값
    ngram_n: int = 3                        # Gate 1에 사용할 n-gram 크기
    tfidf_top_k: int = 20                  # Gate 2: TF-IDF 상위 k 토큰 추출


@dataclass
class SegmentMetadata:
    """캐시 저장 시 사전 계산된 메타데이터 (Gate 검사 오버헤드 최소화)."""
    segment_id: str
    token_ids: List[int]
    ngrams: Set[Tuple[int, ...]]           # Gate 1용: 사전 계산
    top_k_tokens: Set[int]                 # Gate 2용: TF-IDF 상위 k 토큰
    source_version_hash: str               # Gate 3용: sha256(original_context_token_ids)
    mean_kv_vector: torch.Tensor           # Gate 4용: K 벡터 평균 (저장 시 1회 계산)


@dataclass
class GroundedSafetyGatedConfig:
    gate_thresholds: GateThresholds = field(default_factory=GateThresholds)
    chunk_size: int = 128
    max_entries: int = 1000
    seed: int = 42
```

#### CacheStore 구현 및 핵심 메서드

```python
class GroundedSafetyGatedSegmentCache(CacheStore):
    """GroundedCache 4-게이트 안전성 검증 기반 비연속 KV 세그먼트 재사용 캐시.

    SegmentedHashCache를 내부 스토리지로 래핑.
    4-게이트 fail-fast 검사: Gate 1 → Gate 2 → Gate 3 → Gate 4 순서.
    Gate 1 실패 시 즉시 거부 (나머지 게이트 미실행).
    Gate 2/4 실패 시 partial_reuse (c_KV 성분만 반환).
    Gate 3 실패 시 stale_segment 처리 → 즉시 퇴거.
    메타데이터(SegmentMetadata)는 insert 시 1회 계산, 조회 시 재계산 없음.
    """

    def __init__(self, config: GroundedSafetyGatedConfig) -> None:
        self._inner = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._metadata: Dict[str, SegmentMetadata] = {}
        self._config = config
        # 5단계 히트율 카운터
        self._safe_reuse_hits: int = 0
        self._partial_reuse_hits: int = 0
        self._gate_rejections: int = 0
        self._stale_evictions: int = 0
        self._misses: int = 0
        self._total_lookups: int = 0

    # ---- CacheStore 추상 메서드 구현 ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """내부 SegmentedHashCache에 위임. 메타데이터는 insert()로 별도 등록."""
        self._inner.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """내부 SegmentedHashCache에 위임 (메타데이터 없는 단순 조회)."""
        return self._inner.get(key)

    def evict(self) -> int:
        return self._inner.evict()

    def hit_rate(self) -> float:
        """safe_reuse + partial_reuse를 히트로 계산."""
        if self._total_lookups == 0:
            return 0.0
        return (self._safe_reuse_hits + self._partial_reuse_hits) / self._total_lookups

    def memory_bytes(self) -> int:
        return self._inner.memory_bytes()

    def reset_stats(self) -> None:
        self._inner.reset_stats()
        self._safe_reuse_hits = 0
        self._partial_reuse_hits = 0
        self._gate_rejections = 0
        self._stale_evictions = 0
        self._misses = 0
        self._total_lookups = 0

    # ---- 4-게이트 안전성 검증 핵심 API ----

    def insert(
        self,
        token_ids: List[int],
        kv_data: torch.Tensor,
        source_context_token_ids: Optional[List[int]] = None,
        layer_idx: int = 0,
    ) -> str:
        """세그먼트 저장 + 메타데이터 사전 계산.

        알고리즘:
          key = self._inner.chunk_key(token_ids, chunk_idx=0, layer_idx=layer_idx)
          # Gate 1용 n-gram 사전 계산
          ngrams = compute_ngrams(token_ids, n=config.gate_thresholds.ngram_n)
          # Gate 2용 TF-IDF 상위 k 토큰
          top_k = tfidf_top_k_tokens(token_ids, k=config.gate_thresholds.tfidf_top_k)
          # Gate 3용 버전 해시
          ctx = source_context_token_ids or token_ids
          version_hash = sha256(bytes(ctx)).hexdigest()
          # Gate 4용 K 벡터 평균 (kv_data shape: [n_tokens, n_heads, d_kv] 가정)
          mean_kv = kv_data.mean(dim=0).mean(dim=0)  # [d_kv]
          metadata = SegmentMetadata(
              segment_id=key,
              token_ids=token_ids,
              ngrams=ngrams,
              top_k_tokens=top_k,
              source_version_hash=version_hash,
              mean_kv_vector=mean_kv.detach().clone(),
          )
          self._metadata[key] = metadata
          self._inner.put(key, kv_data)
          return key
        """
        ...

    def lookup(
        self,
        query_tokens: List[int],
        current_context_tokens: List[int],
        query_embedding: Optional[torch.Tensor] = None,
        current_source_version_hash: Optional[str] = None,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], HitOutcome]:
        """4-게이트 안전성 검증 후 KV 반환.

        알고리즘 (fail-fast):
          key = self._inner.chunk_key(query_tokens, chunk_idx=0, layer_idx=layer_idx)
          self._total_lookups += 1

          # 캐시 미스 확인
          kv = self._inner.get(key)
          if kv is None:
              self._misses += 1
              return None, "miss"

          meta = self._metadata.get(key)
          if meta is None:
              self._misses += 1
              return None, "miss"

          # --- Gate 1: Content Similarity (fail-fast) ---
          query_ngrams = compute_ngrams(query_tokens, n=config.gate_thresholds.ngram_n)
          ctx_ngrams = compute_ngrams(current_context_tokens, n=config.gate_thresholds.ngram_n)
          content_sim = jaccard(meta.ngrams, ctx_ngrams)
          if content_sim < config.gate_thresholds.content_sim_threshold:
              self._gate_rejections += 1
              return None, "gate_rejected"  # 즉시 거부, Gate 2~4 미실행

          # --- Gate 2: Evidence Overlap ---
          ctx_top_k = tfidf_top_k_tokens(current_context_tokens, k=config.gate_thresholds.tfidf_top_k)
          if len(meta.top_k_tokens) > 0:
              evidence_overlap = len(meta.top_k_tokens & ctx_top_k) / len(meta.top_k_tokens)
          else:
              evidence_overlap = 0.0
          gate2_pass = evidence_overlap >= config.gate_thresholds.evidence_overlap_threshold

          # --- Gate 3: Source Version Validity ---
          if current_source_version_hash is not None:
              gate3_pass = (meta.source_version_hash == current_source_version_hash)
          else:
              gate3_pass = True  # 버전 정보 없으면 통과 (보수적 허용)
          if not gate3_pass:
              self._stale_evictions += 1
              self._inner.evict()  # 해당 세그먼트 즉시 퇴거
              # 실제로는 key 특정 퇴거 필요 → _evict_key(key) 헬퍼 사용
              return None, "stale_evicted"

          # --- Gate 4: Attention Contribution Support ---
          gate4_pass = True
          if query_embedding is not None and meta.mean_kv_vector is not None:
              attn_support = cosine_similarity(meta.mean_kv_vector, query_embedding)
              gate4_pass = attn_support >= config.gate_thresholds.attn_support_threshold

          # --- 결과 분기 ---
          if gate2_pass and gate4_pass:
              self._safe_reuse_hits += 1
              return kv, "safe_reuse_hit"
          else:
              # partial_reuse: c_KV 성분만 반환 (앞 절반 = content KV)
              self._partial_reuse_hits += 1
              c_kv = kv[..., :kv.shape[-1] // 2]  # 위치-자유 성분
              return c_kv, "partial_reuse_hit"
        """
        ...

    def get_stats(self) -> dict:
        """5단계 히트율 통계 반환.

        Returns:
          {
            "total_lookups": int,
            "safe_reuse_hit_rate": float,
            "partial_reuse_hit_rate": float,
            "gate_rejection_rate": float,
            "stale_eviction_rate": float,
            "miss_rate": float,
            "safety_gate_pass_rate": float,  # safe_reuse / (safe_reuse + rejected + stale)
          }
        """
        ...

    # ---- 보조 헬퍼 (private) ----

    def _compute_ngrams(self, tokens: List[int], n: int) -> Set[Tuple[int, ...]]:
        """토큰 ID 시퀀스에서 n-gram 집합 생성.

        알고리즘:
          return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}
        오버헤드: O(len(tokens) × n) ≈ 0.1ms (n=3, len≤256).
        """
        ...

    def _tfidf_top_k(self, tokens: List[int], k: int) -> Set[int]:
        """경량 TF-IDF 상위 k 토큰 추출 (모델 호출 불필요).

        알고리즘:
          tf = Counter(tokens)  # 토큰 빈도
          # IDF 근사: log(total_tokens / tf[token]) (단일 문서, 코퍼스 IDF 생략)
          tfidf_scores = {t: tf[t] * math.log(len(tokens) / tf[t]) for t in tf}
          return set(sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)[:k])
        """
        ...

    def _jaccard(self, a: Set, b: Set) -> float:
        """Jaccard 유사도 |a ∩ b| / |a ∪ b|."""
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    def _cosine_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """1D 텐서 cosine similarity. 정규화 후 dot product."""
        a_norm = torch.nn.functional.normalize(a.float().unsqueeze(0), dim=-1)
        b_norm = torch.nn.functional.normalize(b.float().unsqueeze(0), dim=-1)
        return float((a_norm * b_norm).sum())
```

---

### 3. SafetyGatedHitRateMetrics — `src/metrics/hit_rate.py` 추가

```python
@dataclass
class SafetyGatedHitRateMetrics:
    """5단계 안전성 게이트 히트율 지표. 기존 클래스와 독립적."""
    total_lookups: int = 0
    safe_reuse_hits: int = 0
    partial_reuse_hits: int = 0
    gate_rejections: int = 0
    stale_evictions: int = 0
    misses: int = 0

    def record(self, outcome: str) -> None:
        """outcome: HitOutcome 리터럴 값 중 하나."""
        self.total_lookups += 1
        if outcome == "safe_reuse_hit":
            self.safe_reuse_hits += 1
        elif outcome == "partial_reuse_hit":
            self.partial_reuse_hits += 1
        elif outcome == "gate_rejected":
            self.gate_rejections += 1
        elif outcome == "stale_evicted":
            self.stale_evictions += 1
        else:
            self.misses += 1

    def safety_gate_pass_rate(self) -> float:
        """safe_reuse / (safe_reuse + gate_rejected + stale_evicted). 목표: ≥80%."""
        denom = self.safe_reuse_hits + self.gate_rejections + self.stale_evictions
        return self.safe_reuse_hits / denom if denom > 0 else 0.0

    def effective_hit_rate(self) -> float:
        """(safe_reuse + partial_reuse) / total_lookups."""
        if self.total_lookups == 0:
            return 0.0
        return (self.safe_reuse_hits + self.partial_reuse_hits) / self.total_lookups

    def stale_eviction_rate(self) -> float:
        return self.stale_evictions / self.total_lookups if self.total_lookups > 0 else 0.0

    def summary(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "safe_reuse_hit_rate": self.safe_reuse_hits / max(1, self.total_lookups),
            "partial_reuse_hit_rate": self.partial_reuse_hits / max(1, self.total_lookups),
            "gate_rejection_rate": self.gate_rejections / max(1, self.total_lookups),
            "stale_eviction_rate": self.stale_eviction_rate(),
            "miss_rate": self.misses / max(1, self.total_lookups),
            "safety_gate_pass_rate": self.safety_gate_pass_rate(),
            "effective_hit_rate": self.effective_hit_rate(),
        }

    def reset(self) -> None:
        self.total_lookups = 0
        self.safe_reuse_hits = 0
        self.partial_reuse_hits = 0
        self.gate_rejections = 0
        self.stale_evictions = 0
        self.misses = 0
```

---

### 4. WynerZivWindowMetrics — `src/metrics/memory.py` 추가

```python
@dataclass
class WynerZivWindowMetrics:
    """레이어별 W*_l 시계열 + α_l EMA 추적. 기존 클래스와 독립적."""
    _window_history: Dict[int, List[int]] = field(default_factory=dict)
    _alpha_history: Dict[int, List[float]] = field(default_factory=dict)
    _batch_indices: List[int] = field(default_factory=list)

    def record(
        self,
        batch_idx: int,
        window_sizes: Dict[int, int],
        alpha_ema: Dict[int, float],
    ) -> None:
        """window_sizes: {layer_idx → W*_l}, alpha_ema: {layer_idx → α_l}."""
        self._batch_indices.append(batch_idx)
        for layer_idx, w in window_sizes.items():
            self._window_history.setdefault(layer_idx, []).append(w)
        for layer_idx, a in alpha_ema.items():
            self._alpha_history.setdefault(layer_idx, []).append(a)

    def latest_window_sizes(self) -> Dict[int, int]:
        return {l: hist[-1] for l, hist in self._window_history.items() if hist}

    def latest_alpha_ema(self) -> Dict[int, float]:
        return {l: hist[-1] for l, hist in self._alpha_history.items() if hist}

    def memory_reduction_vs_fixed(self, fixed_window: int = 512) -> float:
        """최신 W* 합 vs 고정 윈도우 합 비교. 양수 = 절감."""
        total_w_star = sum(self.latest_window_sizes().values())
        n_layers = len(self._window_history)
        if n_layers == 0:
            return 0.0
        total_fixed = fixed_window * n_layers
        return (total_fixed - total_w_star) / total_fixed if total_fixed > 0 else 0.0

    def summary(self) -> dict:
        return {
            "latest_window_sizes": self.latest_window_sizes(),
            "latest_alpha_ema": self.latest_alpha_ema(),
            "memory_reduction_vs_fixed_512": self.memory_reduction_vs_fixed(512),
            "total_batches_recorded": len(self._batch_indices),
        }
```

---

### 5. GroundedWynerZivBCPipeline (Cross-1) — `src/engine/grounded_wyner_ziv_bc_pipeline.py`

```python
@dataclass
class BCPipelineConfig:
    shared_accuracy_budget: float = 0.01   # 공유 ε 예산
    perplexity_sample_ratio: float = 0.01  # 샘플 배치 비율 (1%)
    auto_fallback: bool = True             # 실측 delta > ε 시 자동 Fallback
    seed: int = 42


class GroundedWynerZivBCPipeline:
    """GroundedSafetyGatedSegmentCache(B-1) + WynerZivAdaptiveWindowEviction(C-1)
    B+C 통합 파이프라인.

    처리 흐름:
      Step 1 (B-1): 비연속 재사용 요청 → 4-게이트 안전성 검증
        → safe_reuse_hit: KV 반환, SafetyGatedHitRateMetrics 기록
        → partial_reuse_hit: c_KV만 반환
        → gate_rejected / stale_evicted / miss: Step 2로 KV 계산 후 저장

      Step 2 (C-1): 새로 계산된 KV → WynerZivAdaptiveWindowEviction 슬라이딩 윈도우 저장
        → window_size[l] 초과 토큰 자동 퇴거
        → perplexity_delta 측정 시 sensitivity model 갱신

      Step 3 (ε 예산 모니터링): sample_ratio 비율의 배치에서 perplexity_delta 측정
        → trigger_fallback_if_needed() 호출
        → WynerZivWindowMetrics 기록

    공유 ε 예산: B-1 gate_thresholds와 C-1 accuracy_budget이 동일 ε 값 참조.
    """

    def __init__(
        self,
        safety_cache: GroundedSafetyGatedSegmentCache,
        window_eviction: WynerZivAdaptiveWindowEviction,
        safety_metrics: SafetyGatedHitRateMetrics,
        window_metrics: WynerZivWindowMetrics,
        config: BCPipelineConfig,
    ) -> None: ...

    def process_request(
        self,
        query_tokens: List[int],
        current_context_tokens: List[int],
        kv_computed: Optional[torch.Tensor] = None,
        query_embedding: Optional[torch.Tensor] = None,
        current_source_version_hash: Optional[str] = None,
        layer_idx: int = 0,
        perplexity_delta: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], HitOutcome]:
        """단일 요청 B+C 파이프라인 처리."""
        ...

    def get_combined_stats(self) -> dict:
        """B+C 통합 통계.

        Returns:
          {
            "safety_gate": SafetyGatedHitRateMetrics.summary(),
            "window_eviction": WynerZivAdaptiveWindowEviction.get_stats(),
            "window_metrics": WynerZivWindowMetrics.summary(),
            "shared_accuracy_budget": float,
          }
        """
        ...
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C를 포함하므로 아래 검증 계획을 반드시 구현에 포함한다.

### perplexity 측정

- **데이터셋**: WikiText-2 (표준 LM 평가셋, 토큰 수 ~2M)
- **모델**: `src/engine/runner.py`의 기존 InferenceRunner 활용 (실제 모델 없을 시 Mock 허용)
- **허용 오차**: perplexity delta ±1% 이내 (evaluation_criteria.md §4 필수)
- **측정 방법**:
  1. 고정 윈도우 W=512 (베이스라인)
  2. 적응형 W*(ε=0.01) (이번 구현)
  3. 적응형 W*(ε=0.005) (더 보수적 예산)
  4. 세 경우의 perplexity 비교 — 적응형이 고정 대비 ±1% 이내여야 한다
- **샘플 배치 perplexity 모니터링**: `sensitivity_sample_ratio=0.05` (5%) 배치에서
  실제 KL 발산 측정 → `update_sensitivity_model()` 입력
- **Fallback 검증**: 인위적으로 높은 perplexity_delta 주입 → W* 2× 증가 확인

### 태스크 정확도 측정

- **벤치마크**: RULER (Long-Range Understanding and Evaluation)
  - RULER-4K, RULER-16K, RULER-32K 세 길이 설정
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)
- **비교 기준**: 고정 윈도우 W=256, 고정 윈도우 W=512, 적응형 W*(ε=0.01)
- **추가**: LongBench 서브셋 (단일 문서 QA) — 컨텍스트 길이별 정확도 추적

### 검증 테스트 파일: `tests/unit/test_compression_accuracy.py`

필수 테스트 케이스 (≥5개):

```python
def test_w_star_satisfies_accuracy_budget():
    """compute_optimal_window(ε) 반환값에서 S_l(W*) ≤ ε 조건 성립."""

def test_w_star_increases_when_fallback_triggered():
    """observed_delta > ε 입력 시 trigger_fallback_if_needed() → W* ≥ 2 × 이전값."""

def test_perplexity_delta_within_budget_simulation():
    """Mock 민감도 함수로 W*(ε=0.01) 윈도우 사용 시 simulated perplexity delta ≤ 0.01."""

def test_accuracy_budget_not_violated_under_kv_pressure():
    """kv_pressure > threshold 시 ε_relaxed = ε × 2 적용 → W* 축소 but delta ≤ ε_relaxed."""

def test_warmup_batches_stabilize_alpha_ema():
    """100회 update_sensitivity_model() 호출 후 alpha_ema 분산 < 0.1."""

def test_fixed_vs_adaptive_window_memory_reduction():
    """적응형 W*(ε=0.01) 메모리 합 < 고정 W=512 × n_layers × 기준_tensor_size."""
```

### α_l 추정 안정성 검증

- 워밍업 100 배치 완료 후 `_warmup_done = True` 설정
- EMA γ=0.1 (슬로우 갱신) → α_l 과소 추정으로 인한 과도한 퇴거 방지
- 레이어별 α_l 분포: 0.5 ~ 3.0 범위가 정상 (벗어나면 경고 로그)

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-29.yaml
experiment:
  date: "2026-05-29"
  activity: "B+C"
  cache_type: "grounded_safety_gated"      # Activity B-1
  compression_method: "wyner_ziv_adaptive_window"  # Activity C-1
  scheduler_type: "default"               # Activity A 미포함
  seed: 42

# Activity C-1: WynerZiv Adaptive Window Eviction
wyner_ziv:
  accuracy_budget: 0.01                   # ε: perplexity delta 허용 상한 (1%)
  min_window_size: 64
  max_window_size: 4096
  fitting_interval: 1000                  # W* 재계산 주기 (배치 수)
  sensitivity_sample_ratio: 0.05          # 민감도 측정 배치 비율 (5%)
  ema_gamma: 0.1                          # α_l EMA 갱신 계수
  warmup_batches: 100
  kv_pressure_threshold: 0.8
  epsilon_relaxation_factor: 2.0
  n_layers: 32
  truncation_distances: [32, 64, 128, 256, 512]
  window_config_path: "configs/wyner_ziv_window_config.yaml"

# Activity B-1: GroundedSafetyGated Segment Cache
grounded_safety_gated:
  gate_thresholds:
    content_sim_threshold: 0.85           # Gate 1: n-gram Jaccard
    evidence_overlap_threshold: 0.60      # Gate 2: TF-IDF Jaccard
    attn_support_threshold: 0.50          # Gate 4: cosine similarity
    ngram_n: 3
    tfidf_top_k: 20
  chunk_size: 128
  max_entries: 1000

# Cross-1 B+C Pipeline
bc_pipeline:
  shared_accuracy_budget: 0.01           # B-1 + C-1 공유 ε 예산
  perplexity_sample_ratio: 0.01          # 샘플 배치 비율 (1%)
  auto_fallback: true

# 측정 지표 저장 경로
metrics:
  output_dir: "results/2026-05-29"
  metrics_file: "results/2026-05-29/metrics.json"
```

```yaml
# configs/wyner_ziv_window_config.yaml
# 레이어별 초기값 (온라인 갱신으로 수렴)
# C_l_init, alpha_l_init: Wyner-Ziv 논문 실험 기반 추정 초기값
# window_size: 초기 슬라이딩 윈도우 크기 (W* 계산 전 사용)
layer_defaults:
  C_l_init: 1.0
  alpha_l_init: 1.5    # 다항식 지수 초기값 (α ~ 1~3 범위)
  window_size: 512     # 초기 윈도우 크기 (워밍업 전)

# 레이어별 커스텀 초기값 (옵션, 비어 있으면 layer_defaults 사용)
layer_overrides: {}
# 예시:
# layer_overrides:
#   0: {C_l_init: 1.2, alpha_l_init: 1.0, window_size: 1024}
#   31: {C_l_init: 0.8, alpha_l_init: 2.0, window_size: 256}
```

---

## results/2026-05-29/metrics.json 저장 지표

```json
{
  "experiment_date": "2026-05-29",
  "activity": "B+C",

  "compression_c1": {
    "fixed_window_512_perplexity": null,
    "adaptive_window_epsilon_001_perplexity": null,
    "perplexity_delta_pct": null,
    "memory_reduction_vs_fixed_512_pct": null,
    "wyner_ziv_w_star_mean_across_layers": null,
    "alpha_ema_mean_across_layers": null,
    "fallback_trigger_count": null,
    "warmup_batch_count": null,
    "ruler_4k_accuracy_baseline": null,
    "ruler_4k_accuracy_adaptive": null,
    "ruler_16k_accuracy_baseline": null,
    "ruler_16k_accuracy_adaptive": null,
    "ruler_32k_accuracy_baseline": null,
    "ruler_32k_accuracy_adaptive": null
  },

  "safety_gate_b1": {
    "total_lookups": null,
    "safe_reuse_hit_rate": null,
    "partial_reuse_hit_rate": null,
    "gate_rejection_rate": null,
    "stale_eviction_rate": null,
    "miss_rate": null,
    "safety_gate_pass_rate": null,
    "noncontiguous_hit_fraction": null,
    "gate1_rejection_rate": null,
    "gate2_partial_rate": null,
    "gate3_stale_rate": null,
    "gate4_partial_rate": null,
    "gate_overhead_ms_per_segment_p50": null,
    "gate_overhead_ms_per_segment_p99": null
  },

  "cross_bc_pipeline": {
    "combined_memory_reduction_pct": null,
    "combined_perplexity_delta_pct": null,
    "effective_hit_rate": null,
    "epsilon_budget_violations": null,
    "throughput_tokens_per_sec_baseline": null,
    "throughput_tokens_per_sec_bc_pipeline": null,
    "ttft_p50_baseline_ms": null,
    "ttft_p50_bc_pipeline_ms": null,
    "ttft_p50_delta_pct": null
  }
}
```

---

## 테스트 요구사항

### 필수 단위 테스트

- [ ] `tests/unit/test_wyner_ziv_eviction.py` (≥7개 테스트)
  - `test_cache_store_interface_all_methods()`: CacheStore 추상 메서드 전부 동작
  - `test_put_get_round_trip_single_layer()`: put → get 동일 텐서 반환
  - `test_window_eviction_respects_max_size()`: len(window) > window_size 이면 자동 퇴거
  - `test_compute_optimal_window_formula()`: W* = (C/ε)^(1/α) 수치 검증 (α=1.5, C=1.0, ε=0.01 → W* ≈ 464)
  - `test_update_sensitivity_model_ema_convergence()`: 반복 호출 시 EMA 수렴
  - `test_trigger_fallback_doubles_window()`: observed_delta > ε → W* 2× 증가
  - `test_fallback_not_triggered_within_budget()`: observed_delta ≤ ε → W* 불변
  - `test_kv_pressure_relaxes_epsilon()`: kv_pressure > threshold → ε_relaxed = 2ε → W* 축소
  - `test_layer_wise_independent_windows()`: 레이어별 window_size 독립적으로 설정
  - `test_recompute_all_windows_updates_state()`: recompute_all_windows() 후 window_size 갱신됨
  - `test_warmup_flag_set_after_100_batches()`: _warmup_done = True (warmup_batches 이후)

- [ ] `tests/unit/test_grounded_safety_gates.py` (≥7개 테스트)
  - `test_cache_store_interface_all_methods()`: CacheStore 추상 메서드 전부 동작
  - `test_gate1_fail_fast_rejects_immediately()`: Gate 1 실패 → "gate_rejected" 즉시 반환, Gate 2~4 미호출
  - `test_gate3_stale_triggers_eviction()`: Gate 3 실패 → "stale_evicted" + 내부 퇴거 확인
  - `test_gate2_partial_returns_c_kv_only()`: Gate 2 실패 → "partial_reuse_hit" + 반환 텐서 크기 = 원본 절반
  - `test_gate4_partial_returns_c_kv_only()`: Gate 4 실패 → "partial_reuse_hit"
  - `test_all_gates_pass_returns_safe_reuse()`: 모든 게이트 통과 → "safe_reuse_hit", 전체 KV 반환
  - `test_miss_on_unknown_key()`: 저장 안 된 key → "miss"
  - `test_safety_gate_pass_rate_above_threshold()`: safe_reuse / (safe + rejected + stale) ≥ 0.8 시나리오
  - `test_insert_precomputes_metadata()`: insert() 후 _metadata에 SegmentMetadata 존재
  - `test_ngram_jaccard_below_threshold_rejects()`: 낮은 content_sim → gate_rejected
  - `test_stale_eviction_rate_tracked()`: Gate 3 실패 다수 시 stale_eviction_rate > 0

- [ ] `tests/unit/test_compression_accuracy.py` (≥5개 테스트)
  - `test_w_star_satisfies_accuracy_budget()`: W* 계산 후 S_l(W*) ≤ ε 조건 시뮬레이션
  - `test_w_star_increases_when_fallback_triggered()`: trigger_fallback_if_needed(delta > ε) → W* 2×
  - `test_perplexity_delta_within_budget_simulation()`: Mock 민감도로 simulated delta ≤ ε
  - `test_accuracy_budget_not_violated_under_kv_pressure()`: 압박 시 W* 축소 + delta ≤ ε_relaxed
  - `test_warmup_batches_stabilize_alpha_ema()`: 100회 update 후 alpha_ema 분산 < 0.1
  - `test_fixed_vs_adaptive_window_memory_reduction()`: 적응형 메모리 < 고정 윈도우 메모리

### 필수 통합 테스트

- [ ] `tests/integration/test_grounded_wyner_ziv_bc_pipeline_e2e.py`
  - `test_bc_pipeline_full_request_flow()`: B+C 파이프라인 완전 흐름 (Mock 환경)
  - `test_bc_safe_reuse_hit_uses_gated_cache()`: safe_reuse_hit 시 WynerZiv 저장 호출 안 됨
  - `test_bc_miss_inserts_into_window_eviction()`: miss 시 WynerZiv 슬라이딩 윈도우에 저장
  - `test_bc_accuracy_budget_shared()`: B-1 gate_thresholds.ε == C-1 accuracy_budget
  - `test_bc_fallback_auto_triggers_on_high_delta()`: auto_fallback=True + 높은 delta → W* 2×
  - `test_bc_cachestore_interface_compat()`: 두 클래스 모두 CacheStore로 사용 가능
  - `test_bc_memory_reduction_exceeds_single_c()`: 복합 메모리 절감 > 단독 C-1 절감

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 100% 통과**: 위 명시된 모든 단위 테스트 케이스 통과
2. **통합 테스트 100% 통과**: `test_grounded_wyner_ziv_bc_pipeline_e2e.py` 전체 통과
3. **기존 테스트 회귀 없음**: 이전 사이클 구현의 모든 단위·통합 테스트 계속 통과
4. **evaluation_criteria.md §4 (Activity C) 필수 항목 모두 Pass**:
   - perplexity 변화 ±1% 이내 (필수) — `test_compression_accuracy.py`로 검증
   - downstream 태스크 정확도 변화 ±1% 이내 (필수)
   - KV Memory Reduction 베이스라인 대비 −30% 이상 (높음)
5. **evaluation_criteria.md §3 (Activity B) 기준 충족**:
   - 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상 (높음)
   - 비연속 세그먼트 히트율 전체 히트의 30% 이상 (높음)
   - safety_gate_pass_rate ≥ 80% (신규 지표)
6. **evaluation_criteria.md §5 (크로스 조합) 기준**:
   - 복합 Memory Reduction 단일 C-1 대비 추가 −10% 이상 (높음)
   - 복합 적용 후에도 accuracy ±1% 이내 (필수)
7. **CacheStore 인터페이스 준수**: `GroundedSafetyGatedSegmentCache`, `WynerZivAdaptiveWindowEviction` 추상 메서드 전부 구현
8. **설정 파일 존재**: `configs/experiments/2026-05-29.yaml`, `configs/wyner_ziv_window_config.yaml`
9. **metrics.json 생성**: `results/2026-05-29/metrics.json`에 모든 지표 기록
10. **시드 고정 재현성**: seed=42로 동일 결과 재현 가능
11. **게이트 오버헤드**: 4-게이트 검사 총 오버헤드 < 1ms/세그먼트 (p99 기준)
    - Gate 1 n-gram: insert 시 사전 계산 (조회 시 O(1))
    - Gate 2 TF-IDF: insert 시 사전 계산 (조회 시 O(k))
    - Gate 3 버전 해시: O(1) 문자열 비교
    - Gate 4 cosine sim: 저장 시 mean_kv_vector 1회 계산, 조회 시 O(d_kv)

---

## 구현 우선순위 순서

1. C-1: `WynerZivConfig` + `WynerZivAdaptiveWindowEviction` (온라인 적합, W* 계산, 슬라이딩 윈도우, Fallback)
2. C-1: `configs/wyner_ziv_window_config.yaml` 초기값 설정
3. B-1: `GateThresholds`, `SegmentMetadata`, `GroundedSafetyGatedConfig` + `GroundedSafetyGatedSegmentCache` (4-게이트, insert, lookup)
4. 지표: `SafetyGatedHitRateMetrics` (src/metrics/hit_rate.py 추가)
5. 지표: `WynerZivWindowMetrics` (src/metrics/memory.py 추가)
6. Cross-1: `BCPipelineConfig` + `GroundedWynerZivBCPipeline` (공유 ε 예산, 샘플 perplexity 측정)
7. 설정 파일: `configs/experiments/2026-05-29.yaml`
8. 단위 테스트: `test_wyner_ziv_eviction.py` → `test_grounded_safety_gates.py` → `test_compression_accuracy.py`
9. 통합 테스트: `test_grounded_wyner_ziv_bc_pipeline_e2e.py`
10. `results/2026-05-29/metrics.json` 생성

---

## 보존 파일 (수정 금지)

이전 사이클 구현 파일은 수정하지 않는다:

- `src/cache/irminsul_mla_segment_cache.py` (05-26 B-1)
- `src/cache/cdc_content_hash_interface.py` (05-26 B-2)
- `src/cache/arch_aware_noncontiguous_router.py` (05-26)
- `src/cache/mla_two_axis_compression_codec.py` (05-26 C-1)
- `src/cache/indexmem_eviction_codec.py` (05-27 C-1)
- `src/cache/indexmem_latent_memory_module.py` (05-27 C-1)
- `src/cache/indexmem_learnable_indexer.py` (05-27 C-1)
- `src/cache/indexmem_soft_hit_segment_cache.py` (05-27 B-1)
- `src/engine/indexmem_bc_pipeline.py` (05-27 Cross-1)
- `src/cache/pegaflow_kv_connector.py` (05-28 A-2)
- `src/cache/pegaflow_irminsul_distributed_cache.py` (05-28 B-1)
- `src/scheduler/hexagent_workflow_scheduler.py` (05-28 A-1)
- `src/scheduler/pegaflow_rdma_router.py` (05-28 A-2)
- `src/engine/hexagent_irminsul_pegaflow_pipeline.py` (05-28 Cross-1)
- 기타 모든 이전 사이클 파일 — 기존 단위·통합 테스트 회귀 없이 통과해야 한다.

**주의**:
- `src/metrics/hit_rate.py`에는 `SafetyGatedHitRateMetrics` 클래스만 추가한다.
  기존 `WeightedHitRateMetrics`, `HitRateMetrics`, `DistributedHitRateMetrics` 수정 금지.
- `src/metrics/memory.py`에는 `WynerZivWindowMetrics` 클래스만 추가한다.
  기존 클래스 수정 금지.
