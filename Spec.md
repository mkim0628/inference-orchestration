<!-- 변경 이유 (이전 Spec.md: 2026-05-19 대비):
이전 사이클(2026-05-19)은 A+B+C 조합이었다:
  - A-1 KVDriveAttentionAwarePipelineScheduler (어텐션 점수 기반 3계층 HBM/DRAM/SSD 배치)
  - B-1 ThunderAgentStaticSegmentReservationCache (LLMProgramDAG 정적 분석 세그먼트 예약)
  - C-1 KVDriveTierDifferentiatedCompressionCodec (계층별 차등 압축: FP8/VQ/INT4)
  - Cross KVDriveThunderAgentIntegratedStack (A+B+C 통합)

이번 사이클(2026-05-20)은 A+C 조합으로 전환된다. 설계 축이 다음과 같이 변경된다:

주요 변경:
1. [전략 전환] 3계층 정적 배치(HBM/DRAM/SSD) → 런타임 KV 풀 혼잡도 기반 동적 어드미션 제어.
   이전 기법이 "KV를 어느 계층에 두느냐"의 정적 배치 문제였다면, 이번 기법은
   "언제 새 에이전트를 허용하느냐"의 동적 어드미션 게이트 문제다. CONCUR 논문 기반.

2. [Activity A 교체] KVDriveAttentionAwarePipelineScheduler(3계층 tier 배치 + KVTierRegistry) →
   CONCURCongestionBasedAgentAdmissionScheduler(KV 풀 점유율 실시간 모니터링 → 3단계 혼잡
   상태 분류 → 에이전트 어드미션 게이트 + 온라인 임계값 적응).
   스케줄링 결정 단위: 에이전트 스텝(step) 단위 (요청/배치 단위 아님).
   캐시 상태 접근: KVPoolMonitor.get_occupancy() O(1) 실시간 조회.

3. [Activity C 교체] KVDriveTierDifferentiatedCompressionCodec(계층별 압축: FP8/VQ/INT4) →
   SpecAttnVerificationGuidedKVSparseCodec(자기-추측 디코딩 검증 로짓 유도 KV 희소화).
   이전 기법이 "KV가 저장된 위치 기반 압축 강도 결정"이었다면, 이번 기법은
   "자기-추측 디코딩의 검증 단계 full-attention 로짓을 무비용 사이드채널로 활용하여
   토큰별 중요도 마스크를 추출하고 중요도 낮은 KV를 즉시 희소화"하는 방식이다.
   Training-free: 검증 로짓은 추측 디코딩 파이프라인의 기존 연산 부산물.

4. [Cross 신규] CongestionAdmissionSpecAttnDualReductionPipeline:
   A(에이전트 수 제한) × C(에이전트당 KV 접근 희소화)의 곱연산적 감소 효과.
   KV 풀 혼잡도를 C-1 임계값 동적 조정에도 재활용(통합 피드백 루프).

5. [CacheStore 인터페이스 확장] base.py에 get_importance_mask() 메서드를 선택적(optional)
   메서드로 추가. 기존 추상 메서드(put/get/evict/hit_rate/memory_bytes/reset_stats) 유지.
   기존 모든 구현체는 base의 기본 구현(NotImplementedError raise)을 상속하므로 깨지지 않음.

6. [보존 파일] 이전 사이클 구현 파일(kvdrive_*, thunder_agent_*)은 수정하지 않는다.
   기존 단위·통합 테스트가 회귀 없이 통과해야 한다.

7. [Activity B 미포함] 이번 사이클에서 B는 구현 대상이 아니다. B-1 아이디어
   (CongestionAwareNonContiguousSegmentEvictionPolicy)는 다음 사이클로 이연한다.
   단, A-1의 혼잡도 신호(KVPoolMonitor)는 B 확장 시 재사용 가능하도록 독립 모듈로 분리한다.
-->

# Spec — 2026-05-20

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-20.md`

**최우선 구현 타겟**:
- **Cross-1 (최우선)**: `CongestionAdmissionSpecAttnDualReductionPipeline` (A+C)
  — CONCURCongestionBasedAgentAdmissionScheduler(A-1) + SpecAttnVerificationGuidedKVSparseCodec(C-1)
  의 통합 파이프라인. 에이전트 수 제한(볼륨 제어) × 에이전트당 KV 희소화(접근 희소화)의
  곱연산적 감소 효과.
- **A-1 단독 (2순위)**: `CONCURCongestionBasedAgentAdmissionScheduler`
  — 혼잡도 제어 단독 처리량 측정용. Cross-1 구현의 A 컴포넌트와 동일 클래스.

**해결하려는 문제**:

- **Activity A (CONCUR 혼잡 어드미션)**: 기존 스케줄러들은 KV 풀이 포화 상태가 되어도
  신규 에이전트 스텝을 계속 수락하여 "중간 단계 쓰래싱(middle-phase thrashing)"이 발생한다.
  CONCURCongestionBasedAgentAdmissionScheduler는 KV 풀 점유율을 실시간으로 모니터링하여
  3단계 혼잡 상태(여유/경계/혼잡)로 분류하고, 혼잡 단계에서 신규 어드미션을 일시 중단하여
  쓰래싱 없이 처리량을 극대화한다.

- **Activity C (SpecAttn 검증 유도 희소화)**: 기존 KV 압축 기법(H2O, SnapKV 등)은 별도
  캘리브레이션 데이터나 heuristic 통계에 의존하여 중요 KV를 오분류할 위험이 있다.
  SpecAttnVerificationGuidedKVSparseCodec은 자기-추측 디코딩의 검증 단계에서 이미 계산되는
  full-attention 로짓을 "무비용 사이드채널"로 활용(Collect-2-Query 메커니즘)하여 현재 컨텍스트
  에서 실제로 중요한 KV를 정확히 식별한다. Training-free이며 추가 연산 비용이 없다.

- **Cross A+C**: 두 기법의 감소 효과는 곱연산적이다. A-1이 KV 풀에 진입하는 에이전트 수를
  제한(볼륨 제어)하고, C-1이 진입한 에이전트의 KV 접근 자체를 희소화(접근 희소화)한다.
  KV 풀 점유율 신호는 C-1의 희소화 임계값 동적 조정에도 재활용되어 통합 피드백 루프를 형성한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling (CONCURCongestionBasedAgentAdmissionScheduler)
- [ ] Activity B: Non-Contiguous KV Cache Reuse (미포함, 다음 사이클로 이연)
- [x] Activity C: KV Cache Compression (SpecAttnVerificationGuidedKVSparseCodec)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — WikiText-103 proxy: attention_output_relative_error < 0.01 (MANDATORY)
      — src/metrics/perplexity.py의 attention_output_relative_error() 활용
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 ±1% 이내
      — MMLU proxy: cosine_similarity_output() ≥ 0.99 (MANDATORY)
      — 희소화 임계값 보수적 설정(상위 70~85% KV 유지)으로 달성
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C 높음): KV Memory Reduction ≥ −30%
      — 중요도 낮은 KV 퇴거(eviction) + INT4 압축으로 −30~40% 달성
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C 높음): Effective Context Length 동일 메모리 2× 이상
      — 희소화로 확보한 HBM 슬랙으로 더 긴 컨텍스트 수용
- [ ] 목표 5 (evaluation_criteria.md §2 Activity A 필수): 스케줄링 오버헤드 TTFT p50 +5% 이내
      — KVPoolMonitor.get_occupancy() O(1) + 어드미션 게이트 판정 < 0.1ms/스텝
- [ ] 목표 6 (evaluation_criteria.md §2 Activity A 높음): 캐시 히트율 향상 +10%p
      — 쓰래싱 제거로 진행 중인 에이전트의 KV 퇴거 방지 → 히트율 향상
- [ ] 목표 7 (evaluation_criteria.md §1 처리량 높음): Inference Throughput 베이스라인 +20% 이상
      — Cross-1 A×C 곱연산적 효과: +50~60% 예상
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합): 복합 Throughput 단일 Activity 대비 +5% 이상
      — A-1 단독 vs C-1 단독 vs Cross-1 3방향 비교
- [ ] 목표 9 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후 accuracy ±1% 이내
      — Cross-1 전체 흐름 후 cosine_similarity ≥ 0.99 (MANDATORY)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/scheduler/concur_congestion_admission_scheduler.py` | A | CONCURCongestionBasedAgentAdmissionScheduler + KVPoolMonitor |
| `src/cache/specattn_sparse_codec.py` | C | SpecAttnVerificationGuidedKVSparseCodec — 검증 로짓 유도 KV 희소화 + CacheStore 구현 |
| `src/cache/congestion_specattn_pipeline.py` | A+C | CongestionAdmissionSpecAttnDualReductionPipeline — Cross-1 통합 파이프라인 (CacheStore 구현) |
| `tests/unit/test_concur_congestion_scheduler.py` | A | 어드미션 게이트·혼잡 단계 전환·오버헤드 검증 |
| `tests/unit/test_compression_accuracy.py` | C | SpecAttn 희소화 accuracy 검증 (MANDATORY, 기존 파일 덮어쓰기 또는 신규 생성) |
| `tests/integration/test_congestion_specattn_e2e.py` | A+C | Cross-1 E2E 통합 테스트 |
| `configs/experiments/2026-05-20.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/base.py` | `get_importance_mask()` 선택적 메서드 추가 (기본 구현: `raise NotImplementedError`). 기존 추상 메서드 6개 불변. |

---

## 알고리즘 상세

### KVPoolMonitor (Activity A 내부 유틸리티)

```python
# src/scheduler/concur_congestion_admission_scheduler.py 상단

from dataclasses import dataclass, field
from typing import Deque, List, Literal
from collections import deque
import time
import threading

CongestionLevel = Literal["FREE", "BOUNDARY", "CONGESTED"]

@dataclass
class KVPoolMonitor:
    """KV 풀 점유율 실시간 모니터.

    캐시 상태 접근 방법: get_occupancy() O(1).
    내부적으로 현재 사용 바이트와 최대 용량을 추적.
    """
    capacity_bytes: int         # KV 풀 최대 용량
    alpha_low: float = 0.60    # 혼잡 → 여유 복구 임계값
    alpha_high: float = 0.85   # 여유/경계 → 혼잡 진입 임계값
    _current_bytes: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def update(self, used_bytes: int) -> None:
        with self._lock:
            self._current_bytes = used_bytes

    def get_occupancy(self) -> float:
        """현재 KV 풀 점유율 (0.0~1.0). O(1)."""
        if self.capacity_bytes == 0:
            return 0.0
        return self._current_bytes / self.capacity_bytes

    def congestion_level(self) -> CongestionLevel:
        occ = self.get_occupancy()
        if occ >= self.alpha_high:
            return "CONGESTED"
        elif occ >= self.alpha_low:
            return "BOUNDARY"
        else:
            return "FREE"
```

---

### CONCURCongestionBasedAgentAdmissionScheduler (Activity A)

스케줄링 결정 단위: **에이전트 스텝(step) 단위**.
캐시 상태 접근 방법: `KVPoolMonitor.get_occupancy()` O(1) 실시간 조회.

```python
# src/scheduler/concur_congestion_admission_scheduler.py

@dataclass
class CongestionAdmissionConfig:
    capacity_bytes: int = 1_000_000_000  # KV 풀 최대 용량 (1 GB 기본)
    alpha_low: float = 0.60              # 복구 임계값
    alpha_high: float = 0.85            # 혼잡 진입 임계값
    priority_weights: dict = field(default_factory=dict)  # agent_id → float
    online_adapt_window: int = 100       # 온라인 임계값 적응 윈도우 크기 (스텝 수)
    enable_multinode: bool = False       # 멀티노드: 글로벌 occupancy 집계
    seed: int = 42


class CONCURCongestionBasedAgentAdmissionScheduler(BaseScheduler):
    """CONCUR 기반 KV 풀 혼잡도 어드미션 스케줄러.

    Activity A: KV Cache-aware Scheduling
    스케줄링 결정 단위: 에이전트 스텝(step) 단위.
    캐시 상태 접근: KVPoolMonitor.get_occupancy() O(1) 조회.

    혼잡 상태 3단계:
      FREE (occupancy < alpha_low): 표준 FIFO + 우선순위 재개. 신규 어드미션 허용.
      BOUNDARY (alpha_low <= occupancy < alpha_high): 신규 어드미션 신중 허용.
        우선순위 상위 에이전트만 허용, 나머지 대기열.
      CONGESTED (occupancy >= alpha_high): 신규 에이전트 스텝 어드미션 일시 중단.
        진행 중인 에이전트의 KV는 선제적으로 퇴거하지 않음 (CONCUR 핵심 원칙).

    온라인 임계값 적응:
      매 online_adapt_window 스텝마다 어드미션 게이트 처리량 vs 대기 지연 비율을 계산하여
      alpha_high/alpha_low를 ±0.02 범위 내에서 조정.

    멀티노드 고려사항:
      enable_multinode=True 시 update_remote_occupancy(node_id, occupancy) 로
      원격 노드 KV 풀 점유율을 집계하여 글로벌 혼잡 신호 구성.

    평가 기준 (evaluation_criteria.md §2):
      - 스케줄링 오버헤드: TTFT p50 +5% 이내 (MANDATORY)
      - 캐시 히트율: 스케줄링 미적용 대비 +10%p 이상
      - 최대 대기 시간: 2× 초과하지 않음
    """

    def __init__(self, config: CongestionAdmissionConfig) -> None:
        ...
        self.monitor = KVPoolMonitor(
            capacity_bytes=config.capacity_bytes,
            alpha_low=config.alpha_low,
            alpha_high=config.alpha_high,
        )
        self._wait_queue: Deque = deque()       # 대기 중인 에이전트 스텝
        self._admitted: List = []               # 현재 진행 중인 에이전트
        self._scheduling_times: List[float] = []
        self._step_count: int = 0
        self._gate_throughput: List[float] = []  # 어드미션 처리량 추적
        self._remote_occupancies: dict = {}      # 멀티노드: {node_id: float}

    def admit(self, agent_step: object, priority: float = 1.0) -> bool:
        """에이전트 스텝 어드미션 시도.

        Algorithm:
          1. monitor.congestion_level() 조회.
          2. CONGESTED: 대기열에 추가, return False.
          3. BOUNDARY: priority >= median(priority_weights) → 허용, else 대기열.
          4. FREE: 허용 (FIFO + 우선순위 가중치 정렬).
          5. _step_count 증가, 처리량 기록.
          Returns: True(허용) / False(대기).
        """
        ...

    def release(self, agent_step: object, freed_bytes: int) -> None:
        """에이전트 완료 시 KV 풀 공간 반환 + 대기열 재개.

        Algorithm:
          1. monitor.update(monitor._current_bytes - freed_bytes).
          2. congestion_level() 재확인.
          3. FREE이면 wait_queue에서 FIFO + 우선순위 순으로 대기 에이전트 재개.
        """
        ...

    def update_kv_pool(self, used_bytes: int) -> None:
        """KV 풀 상태 갱신 (매 스텝 호출).
        Algorithm: monitor.update(used_bytes).
        """
        self.monitor.update(used_bytes)

    def update_remote_occupancy(self, node_id: str, occupancy: float) -> None:
        """멀티노드: 원격 노드 점유율 집계."""
        self._remote_occupancies[node_id] = occupancy

    def global_occupancy(self) -> float:
        """로컬 + 원격 점유율 평균 (멀티노드 혼잡 신호)."""
        all_occ = [self.monitor.get_occupancy()] + list(self._remote_occupancies.values())
        return sum(all_occ) / len(all_occ)

    def _adapt_thresholds(self) -> None:
        """온라인 임계값 적응.

        Algorithm:
          throughput = admitted_in_window / online_adapt_window
          wait_ratio = len(wait_queue) / max(1, admitted_in_window)
          if wait_ratio > 0.5: alpha_high -= 0.02  # 더 공격적으로 막음
          if wait_ratio < 0.1: alpha_high += 0.02  # 더 허용
          alpha_high = clamp(alpha_high, 0.70, 0.95)
          alpha_low = alpha_high - 0.25 (항상 차이 유지)
        """
        ...

    def schedule(self, requests: List) -> List:
        """BaseScheduler 호환 schedule() 인터페이스.

        batch 단위 호출 시: congestion_level()로 전체 배치 허용/대기 결정.
        Returns: 허용된 requests 부분 목록.
        """
        t0 = time.monotonic()
        level = self.monitor.congestion_level()
        if level == "CONGESTED":
            result = []  # 혼잡: 신규 배치 전부 대기
        elif level == "BOUNDARY":
            # 우선순위 상위 절반만 허용
            sorted_reqs = sorted(
                requests,
                key=lambda r: self.config.priority_weights.get(getattr(r, "request_id", ""), 1.0),
                reverse=True,
            )
            result = sorted_reqs[: max(1, len(sorted_reqs) // 2)]
        else:
            result = list(requests)
        self._scheduling_times.append((time.monotonic() - t0) * 1000.0)
        self._step_count += 1
        if self._step_count % self.config.online_adapt_window == 0:
            self._adapt_thresholds()
        return result

    def scheduling_overhead_ms_p50(self) -> float:
        """스케줄링 오버헤드 중앙값 (ms). 평가 기준: < 5ms (MANDATORY)."""
        if not self._scheduling_times:
            return 0.0
        s = sorted(self._scheduling_times)
        return s[len(s) // 2]

    def reset_stats(self) -> None:
        self._scheduling_times.clear()
        self._step_count = 0
        self._gate_throughput.clear()
        self._remote_occupancies.clear()
```

---

### SpecAttnVerificationGuidedKVSparseCodec (Activity C)

```python
# src/cache/specattn_sparse_codec.py

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class SpecAttnCodecConfig:
    # 레이어별 KV 유지 비율 (상위 중요도 기준)
    retention_ratio_by_layer: List[float] = field(
        default_factory=lambda: [0.85] * 12
    )  # 하위 레이어: 0.85, 상위 레이어: 0.70 (레이어 인덱스로 조정)
    global_retention_ratio: float = 0.80   # 레이어별 설정 없을 때 기본값
    low_importance_quant_int4: bool = True  # 중요도 낮은 KV: INT4 양자화 vs 퇴거
    int4_threshold: float = 0.01           # INT4 sparsification zero_threshold
    max_entries: int = 1000
    seed: int = 42


class SpecAttnVerificationGuidedKVSparseCodec(CacheStore):
    """SpecAttn Collect-2-Query 메커니즘 기반 KV 희소화 코덱.

    Activity C: KV Cache Compression (Training-free)
    CacheStore 인터페이스 완전 구현.

    핵심 알고리즘 (Collect-2-Query):
      1. 자기-추측 디코딩 검증 단계의 full-attention 로짓(attn_logits: [n_heads, n_q, n_kv])을
         set_verification_logits()로 주입. 이 로짓은 어차피 검증 단계에서 계산되므로 추가 비용 없음.
      2. extract_importance_mask(): 각 KV 위치의 최대 어텐션 가중치를 헤드 전체에서 집계 →
         상위 retention_ratio KV를 "중요" 마스크로 표시.
      3. put()에서 compression_hook()을 통해 중요도 마스크 적용:
         - 중요 KV: 원본 정밀도 유지.
         - 비중요 KV: INT4 양자화(low_importance_quant_int4=True) 또는 즉시 퇴거.
      4. get_importance_mask(key): 해당 KV의 중요도 마스크 반환 (base.py 신규 메서드).

    레이어별 retention_ratio:
      layer_idx < n_layers // 2 → retention_ratio_by_layer[layer_idx] (보수적, 높음)
      layer_idx >= n_layers // 2 → 더 낮은 비율 (상위 레이어는 덜 중요)

    accuracy-preserving 근거 (evaluation_criteria.md §4):
      SpecAttn 논문: Collect-2-Query로 선택된 KV 집합은 full-attention과 출력이 수학적으로
      동등함을 증명. 검증 단계 전체 어텐션 로짓에서 추출한 마스크는 현재 컨텍스트에서
      실제로 중요한 KV를 정확히 식별. 상위 70~85% KV 유지 → perplexity 변화 ±0.5% 이내.

    평가 기준 (evaluation_criteria.md §4):
      - Accuracy 보존 (필수): perplexity 변화 ±1% 이내 — relative_error < 0.01 (MANDATORY)
      - KV Memory Reduction ≥ −30%
      - 압축 오버헤드: TTFT +10% 이내
    """

    def __init__(self, config: SpecAttnCodecConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._importance_masks: Dict[str, torch.Tensor] = {}  # key → bool mask
        self._current_logits: Optional[torch.Tensor] = None   # [n_heads, n_q, n_kv]
        self._current_layer_idx: int = 0
        self._hits = 0
        self._misses = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0

    def set_verification_logits(
        self,
        attn_logits: torch.Tensor,   # [n_heads, n_q, n_kv] — 검증 단계 full-attention 로짓
        layer_idx: int = 0,
    ) -> None:
        """검증 단계 로짓 주입. 다음 put() 호출 전에 설정.

        Algorithm:
          self._current_logits = attn_logits
          self._current_layer_idx = layer_idx
        """
        self._current_logits = attn_logits
        self._current_layer_idx = layer_idx

    def extract_importance_mask(
        self,
        n_kv: int,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Collect-2-Query: 검증 로짓에서 KV 중요도 마스크 추출.

        Algorithm:
          if self._current_logits is None:
            return torch.ones(n_kv, dtype=torch.bool)  # 로짓 없으면 전부 중요
          attn_probs = softmax(self._current_logits, dim=-1)   # [n_heads, n_q, n_kv]
          max_attn_per_kv = attn_probs.max(dim=1).values       # [n_heads, n_kv]
          importance = max_attn_per_kv.mean(dim=0)             # [n_kv]
          # 레이어별 retention_ratio 조회
          if layer_idx < len(config.retention_ratio_by_layer):
            ratio = config.retention_ratio_by_layer[layer_idx]
          else:
            ratio = config.global_retention_ratio
          threshold = importance.quantile(1.0 - ratio)
          mask = importance >= threshold                        # [n_kv] bool
          return mask
        """
        if self._current_logits is None:
            return torch.ones(n_kv, dtype=torch.bool)
        logits = self._current_logits
        if logits.dim() == 3:
            attn_probs = F.softmax(logits.float(), dim=-1)        # [n_heads, n_q, n_kv]
            max_attn = attn_probs.max(dim=1).values               # [n_heads, n_kv]
            importance = max_attn.mean(dim=0)                     # [n_kv]
        else:
            importance = torch.ones(n_kv)
        # 레이어별 ratio
        if layer_idx < len(self.config.retention_ratio_by_layer):
            ratio = self.config.retention_ratio_by_layer[layer_idx]
        else:
            ratio = self.config.global_retention_ratio
        if n_kv == 0:
            return torch.ones(0, dtype=torch.bool)
        k = max(1, int(round(n_kv * ratio)))
        # 상위 k개 중요 KV 선택
        topk_indices = importance.topk(min(k, importance.numel())).indices
        mask = torch.zeros(n_kv, dtype=torch.bool)
        mask[topk_indices] = True
        return mask

    def _compress_low_importance(self, value: torch.Tensor) -> torch.Tensor:
        """비중요 KV: INT4 양자화 (low_importance_quant_int4=True) 또는 zeroing.

        Algorithm:
          if not low_importance_quant_int4:
            return torch.zeros_like(value)  # 퇴거 대신 zeroing (메모리 측정 포함)
          scale = value.abs().max().clamp(min=1e-8) / 7.0
          sparse = value.clone()
          sparse[sparse.abs() < int4_threshold] = 0.0
          q = (sparse / scale).round().clamp(-7, 7)
          return (q * scale).to(value.dtype)
        """
        if not self.config.low_importance_quant_int4:
            return torch.zeros_like(value)
        sparse = value.clone().float()
        sparse[sparse.abs() < self.config.int4_threshold] = 0.0
        scale = sparse.abs().max().clamp(min=1e-8) / 7.0
        q = (sparse / scale).round().clamp(-7, 7)
        return (q * scale).to(value.dtype)

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """중요도 마스크 기반 선택적 KV 압축.

        Algorithm:
          mask = extract_importance_mask(n_kv=value.shape[0], layer_idx=_current_layer_idx)
          self._importance_masks[key] = mask
          result = value.clone()
          if mask.shape[0] == value.shape[0]:
            result[~mask] = _compress_low_importance(value[~mask])
          return result
        """
        n_kv = value.shape[0] if value.dim() >= 1 else 1
        mask = self.extract_importance_mask(n_kv, self._current_layer_idx)
        self._importance_masks[key] = mask
        result = value.clone()
        if mask.shape[0] == value.shape[0] and (~mask).any():
            result[~mask] = self._compress_low_importance(value[~mask])
        return result

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """해당 key의 중요도 마스크 반환 (base.py 신규 메서드 구현).

        Returns: bool tensor [n_kv] 또는 None (미등록 key).
        """
        return self._importance_masks.get(key)

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
        if self._store:
            _, v = self._store.popitem(last=False)
            return v.nbytes
        return 0

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
        self._importance_masks.clear()
        self._current_logits = None
```

---

### CongestionAdmissionSpecAttnDualReductionPipeline (Cross A+C)

```python
# src/cache/congestion_specattn_pipeline.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

from src.cache.base import CacheStore
from src.cache.specattn_sparse_codec import (
    SpecAttnVerificationGuidedKVSparseCodec,
    SpecAttnCodecConfig,
)
from src.scheduler.concur_congestion_admission_scheduler import (
    CONCURCongestionBasedAgentAdmissionScheduler,
    CongestionAdmissionConfig,
)


@dataclass
class DualReductionConfig:
    scheduler_config: Optional[CongestionAdmissionConfig] = None
    codec_config: Optional[SpecAttnCodecConfig] = None
    # 통합 피드백 루프 설정
    codec_adapt_on_congestion: bool = True  # 혼잡 시 희소화 임계값 강화
    retention_reduction_on_congestion: float = 0.10  # 혼잡 시 retention_ratio -= 0.10
    seed: int = 42


class CongestionAdmissionSpecAttnDualReductionPipeline(CacheStore):
    """CONCUR 혼잡 어드미션 × SpecAttn KV 희소화 통합 파이프라인.

    Cross Activity A+C:
      레이어 1 (A): CONCURCongestionBasedAgentAdmissionScheduler — KV 풀 혼잡도 기반 어드미션.
      레이어 2 (C): SpecAttnVerificationGuidedKVSparseCodec — 검증 로짓 유도 KV 희소화.

    통합 피드백 루프:
      KV 풀 점유율이 alpha_high 초과 → C-1 retention_ratio -= retention_reduction_on_congestion
        (더 공격적 희소화로 풀 압박 경감).
      KV 풀 점유율이 alpha_low 이하 → C-1 retention_ratio를 원래 값으로 복원
        (더 많은 KV 보존).

    에이전트별 KV 사용 패턴 추적:
      _agent_kv_usage: {agent_id: total_bytes_used}
      "불균형 에이전트"(usage > mean + 2*std): CONGESTED 단계에서 선별 차단.

    CacheStore 인터페이스 완전 구현.
    codec의 SpecAttn 희소화가 모든 put() 호출에 적용됨.

    평가 기준 (evaluation_criteria.md §5):
      - 복합 Throughput: 단일 Activity 대비 +5% 이상
      - 복합 Memory Reduction: 단일 Activity 대비 −10% 이상
      - Accuracy 보존 (C 포함): 복합 후 cosine ≥ 0.99 (MANDATORY)
    """

    def __init__(self, config: DualReductionConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config

        sched_cfg = config.scheduler_config or CongestionAdmissionConfig(seed=config.seed)
        codec_cfg = config.codec_config or SpecAttnCodecConfig(seed=config.seed)

        self.scheduler = CONCURCongestionBasedAgentAdmissionScheduler(sched_cfg)
        self.codec = SpecAttnVerificationGuidedKVSparseCodec(codec_cfg)
        self._base_retention_ratios = list(codec_cfg.retention_ratio_by_layer)
        self._agent_kv_usage: Dict[str, int] = {}

    def set_verification_logits(
        self, attn_logits: torch.Tensor, layer_idx: int = 0
    ) -> None:
        """검증 로짓을 codec에 전달."""
        self.codec.set_verification_logits(attn_logits, layer_idx)

    def update_kv_pool(self, used_bytes: int) -> None:
        """KV 풀 상태 갱신 + 피드백 루프 트리거.

        Algorithm:
          scheduler.update_kv_pool(used_bytes)
          level = scheduler.monitor.congestion_level()
          if codec_adapt_on_congestion:
            if level == "CONGESTED":
              for i in range(len(retention_ratios)):
                codec.config.retention_ratio_by_layer[i] = max(
                  0.50, _base_retention_ratios[i] - retention_reduction_on_congestion
                )
            elif level == "FREE":
              codec.config.retention_ratio_by_layer = list(_base_retention_ratios)
        """
        self.scheduler.update_kv_pool(used_bytes)
        if self.config.codec_adapt_on_congestion:
            level = self.scheduler.monitor.congestion_level()
            if level == "CONGESTED":
                delta = self.config.retention_reduction_on_congestion
                for i in range(len(self.codec.config.retention_ratio_by_layer)):
                    self.codec.config.retention_ratio_by_layer[i] = max(
                        0.50,
                        self._base_retention_ratios[i] - delta,
                    )
            elif level == "FREE":
                self.codec.config.retention_ratio_by_layer = list(self._base_retention_ratios)

    def schedule(self, requests: List) -> List:
        """스케줄러 위임."""
        return self.scheduler.schedule(requests)

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return self.codec.compression_hook(key, value)

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        return self.codec.get_importance_mask(key)

    def put(self, key: str, value: torch.Tensor) -> None:
        self.codec.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.codec.get(key)

    def evict(self) -> int:
        return self.codec.evict()

    def hit_rate(self) -> float:
        return self.codec.hit_rate()

    def memory_bytes(self) -> int:
        return self.codec.memory_bytes()

    def reset_stats(self) -> None:
        self.codec.reset_stats()
        self.scheduler.reset_stats()
        self._agent_kv_usage.clear()

    def metrics_summary(self) -> Dict:
        """복합 효과 측정용 통합 메트릭."""
        return {
            "scheduler_overhead_ms_p50": self.scheduler.scheduling_overhead_ms_p50(),
            "kv_pool_occupancy": self.scheduler.monitor.get_occupancy(),
            "congestion_level": self.scheduler.monitor.congestion_level(),
            "codec_hit_rate": self.codec.hit_rate(),
            "codec_memory_reduction_ratio": self.codec.memory_reduction_ratio(),
            "current_retention_ratios": list(self.codec.config.retention_ratio_by_layer),
            "total_memory_bytes": self.memory_bytes(),
        }
```

---

### base.py 변경: get_importance_mask() 선택적 메서드 추가

기존 `src/cache/base.py`에 아래 메서드를 `load_with_rope()` 다음에 추가한다.
기존 추상 메서드 6개(put, get, evict, hit_rate, memory_bytes, reset_stats)는 변경하지 않는다.

```python
    def get_importance_mask(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """Return importance mask for the KV entry stored under key.

        Used by SpecAttn-based codecs (Activity C) to expose the
        verification-logit-guided importance mask to upstream components.

        Returns: bool tensor [n_kv] marking important positions (True = important),
                 or None if the implementation does not support importance masking.
        Default implementation raises NotImplementedError — subclasses with
        Activity C SpecAttn support override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support importance masking."
        )
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(SpecAttnVerificationGuidedKVSparseCodec)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-103 proxy — `src/metrics/perplexity.py`의 `attention_output_relative_error()`로
  synthetic float32 KV 텐서를 활용한 attention output 상대 오차 측정.
  실제 WikiText-103 데이터셋이 없는 경우, `torch.randn`으로 생성한 [seq_len, d_head] 텐서를
  사용하는 synthetic proxy 방식으로 대체한다.
- **측정 방법**:
  ```
  relative_error = attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp)
  허용 오차: relative_error < 0.01 (1%) — MANDATORY (evaluation_criteria.md §4 필수)
  ```
- **시나리오별 측정**:
  - 상위 80% KV 유지 (retention_ratio=0.80): relative_error 측정
  - 상위 70% KV 유지 (retention_ratio=0.70): relative_error 측정
  - 혼잡 단계 강화 임계값 (retention_ratio=0.70): relative_error 측정
  - 예상 결과: ±0.5% 이내 (SpecAttn 논문 근거)

### 태스크 정확도 측정

- **벤치마크**: MMLU proxy — `src/metrics/perplexity.py`의 `cosine_similarity_output()`로
  attention output cosine similarity 측정 (downstream 태스크 정확도 대리 지표).
- **측정 방법**:
  ```
  cosine_sim = cosine_similarity_output(q, k_orig, v_orig, k_comp, v_comp)
  허용 오차: cosine_sim >= 0.99 — MANDATORY (evaluation_criteria.md §4 필수)
  ```
- **KL divergence 보조 측정**: `attention_kl_divergence(q, k_orig, k_comp) < 0.015`
- **허용 오차**: ±1% 이내 — evaluation_criteria.md §4 필수

### 희소화 단계별 독립 검증

1. **전체 중요 KV 유지 (ratio=1.0)**: relative_error ≈ 0.0 (기준 검증)
2. **상위 85% KV 유지 (ratio=0.85)**: relative_error < 0.005 (0.5%)
3. **상위 80% KV 유지 (ratio=0.80)**: relative_error < 0.008 (0.8%) — 기본 설정 [MANDATORY]
4. **상위 70% KV 유지 (ratio=0.70)**: relative_error < 0.01 (1%) — 혼잡 단계 [MANDATORY]
5. **INT4 압축 비중요 KV 유지 vs 즉시 퇴거 비교**: accuracy delta < ±0.5%
6. **Cross-1 통합 (A+C) 후 accuracy**: cosine_sim ≥ 0.99 (MANDATORY)

### Fail 기준

**ANY 케이스에서 retention_ratio=0.80 이상일 때 relative_error > 1% → 테스트 실패
(evaluation_criteria.md §4 필수 항목 — 무조건 전체 Fail)**

### 검증 테스트 파일

`tests/unit/test_compression_accuracy.py`

**필수 테스트 케이스**:

```
test_specattn_full_retention_zero_error:
    retention_ratio=1.0 → relative_error ≈ 0.0 (기준)

test_specattn_retention_80pct_relative_error_below_1pct:
    retention_ratio=0.80 → relative_error < 0.01 (MANDATORY)

test_specattn_retention_80pct_cosine_similarity_above_099:
    retention_ratio=0.80 → cosine_sim >= 0.99 (MANDATORY)

test_specattn_retention_70pct_relative_error_below_1pct:
    retention_ratio=0.70 (혼잡 단계) → relative_error < 0.01 (MANDATORY)

test_specattn_kl_divergence_below_threshold:
    KL divergence < 0.015 (보조 지표)

test_specattn_importance_mask_extracts_top_k:
    n_kv=100, ratio=0.80 → mask.sum() == 80

test_specattn_importance_mask_without_logits_returns_all_true:
    set_verification_logits 호출 없이 put → mask 전부 True (안전 폴백)

test_specattn_low_importance_int4_quant_preserves_shape:
    비중요 KV INT4 압축 후 shape 유지

test_specattn_memory_reduction_above_15pct:
    retention_ratio=0.80, int4_quant=True → memory_reduction_ratio() >= 0.15

test_specattn_memory_reduction_above_30pct:
    retention_ratio=0.70, int4_quant=True → memory_reduction_ratio() >= 0.30
    (evaluation_criteria.md §4 높음 항목)

test_specattn_put_get_evict_hit_rate:
    CacheStore 인터페이스 전체 동작 확인

test_specattn_get_importance_mask_returns_stored_mask:
    put() 후 get_importance_mask(key) → bool tensor 반환

test_specattn_seed_reproducibility:
    동일 seed + logits → 동일 mask, 동일 압축 결과

test_cross_pipeline_accuracy_preserved:
    CongestionAdmissionSpecAttnDualReductionPipeline:
    put → get 왕복 후 cosine_sim >= 0.99 (MANDATORY, evaluation_criteria.md §5)

test_congestion_feedback_reduces_retention_ratio:
    update_kv_pool(used=alpha_high 초과) → retention_ratio 감소 확인

test_congestion_free_restores_retention_ratio:
    update_kv_pool(used=alpha_low 이하) → retention_ratio 원래 값 복원
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-20.yaml
experiment:
  date: "2026-05-20"
  activity: "A+C"
  description: >
    Cross-1 CongestionAdmissionSpecAttnDualReductionPipeline:
    A-1 CONCURCongestionBasedAgentAdmissionScheduler (KV 풀 혼잡도 기반 에이전트 어드미션 제어) +
    C-1 SpecAttnVerificationGuidedKVSparseCodec (검증 로짓 유도 KV 희소화, Training-free).
    통합 피드백 루프: KV 풀 혼잡도 신호로 C-1 retention_ratio 동적 조정.
  cache_type: congestion_specattn_pipeline
  compression_method: specattn_verification_guided_sparse
  scheduler_type: concur_congestion_admission

concur_congestion_admission_scheduler:
  capacity_bytes: 1_000_000_000    # KV 풀 최대 용량 (1 GB)
  alpha_low: 0.60                  # 복구 임계값 (혼잡 → 여유)
  alpha_high: 0.85                 # 혼잡 진입 임계값
  online_adapt_window: 100         # 온라인 임계값 적응 윈도우 (스텝 수)
  enable_multinode: false          # 단일 노드 기본
  seed: 42

specattn_sparse_codec:
  retention_ratio_by_layer:         # 레이어별 KV 유지 비율 (12-layer 기준)
    - 0.85  # layer 0 (하위)
    - 0.85
    - 0.83
    - 0.83
    - 0.81
    - 0.80
    - 0.78
    - 0.77
    - 0.75
    - 0.73
    - 0.72
    - 0.70  # layer 11 (상위)
  global_retention_ratio: 0.80     # 레이어 설정 없을 때 기본값
  low_importance_quant_int4: true  # 비중요 KV: INT4 양자화
  int4_threshold: 0.01             # INT4 sparsification zero_threshold
  max_entries: 1000
  seed: 42

dual_reduction_pipeline:
  codec_adapt_on_congestion: true        # 혼잡 시 retention_ratio 동적 강화
  retention_reduction_on_congestion: 0.10  # 혼잡 시 retention_ratio -= 0.10
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"     # src/metrics/perplexity.py 활용
    dataset_proxy: "wikitext103_synthetic"
    task_accuracy_proxy: "mmlu_cosine_similarity"
    relative_error_max: 0.01             # ±1% (evaluation_criteria.md §4 MANDATORY)
    cosine_similarity_min: 0.99          # (evaluation_criteria.md §4 MANDATORY)
    kl_divergence_max: 0.015             # 보조 지표
    perplexity_tolerance_pct: 1.0        # ±1%
    task_accuracy_tolerance_pct: 1.0     # ±1%
    retention_ratios_to_test: [1.0, 0.85, 0.80, 0.70]
  activity_a:
    ttft_overhead_limit_pct: 5.0         # TTFT p50 +5% 이내 (MANDATORY)
    cache_hit_rate_improvement_min_pct: 10.0  # 히트율 +10%p 이상
    max_wait_time_multiplier: 2.0        # 최대 대기 2× 이하
    alpha_low: 0.60
    alpha_high: 0.85
  activity_c:
    memory_reduction_min: 0.30           # −30% 이상 (evaluation_criteria.md §4 높음)
    effective_context_multiplier: 2.0
    compression_overhead_ttft_max_pct: 10.0
  cross_ac:
    throughput_min_improvement_vs_solo: 5.0   # 단일 Activity 대비 +5% (§5 높음)
    memory_min_improvement_vs_solo: 10.0      # 단일 Activity 대비 −10% (§5 높음)
    accuracy_cosine_min: 0.99                 # C 포함 (§5 MANDATORY)
    comparison_methods: ["solo_a1", "solo_c1", "cross_ac"]
  throughput:
    target_improvement_pct: 20               # 베이스라인 대비 +20% (장기 목표)

seed: 42
results_dir: "results/2026-05-20"
```

---

## 테스트 요구사항

- [x] `tests/unit/test_compression_accuracy.py` — Activity C 필수 accuracy 검증 (16개 테스트, 위 목록 참조)
- [x] `tests/unit/test_concur_congestion_scheduler.py` — Activity A 단위 테스트
- [x] `tests/integration/test_congestion_specattn_e2e.py` — Cross-1 E2E 통합 테스트

### 단위 테스트 최소 요구사항 (test_concur_congestion_scheduler.py)

```
test_kv_pool_monitor_occupancy_calculation:
    update(used) → get_occupancy() == used / capacity

test_kv_pool_monitor_congestion_level_free:
    occupancy < alpha_low → "FREE"

test_kv_pool_monitor_congestion_level_boundary:
    alpha_low <= occupancy < alpha_high → "BOUNDARY"

test_kv_pool_monitor_congestion_level_congested:
    occupancy >= alpha_high → "CONGESTED"

test_admit_free_allows_all:
    FREE 상태에서 admit() → True 반환

test_admit_congested_blocks_all:
    CONGESTED 상태에서 admit() → False 반환

test_schedule_congested_returns_empty:
    CONGESTED 상태에서 schedule(requests) → [] 반환

test_schedule_boundary_returns_half:
    BOUNDARY 상태에서 schedule(10개 요청) → 5개 반환 (상위 절반)

test_schedule_free_returns_all:
    FREE 상태에서 schedule(requests) → 전부 반환

test_scheduling_overhead_below_5ms:
    schedule() TTFT 오버헤드 중앙값 < 5ms (evaluation_criteria.md §2 MANDATORY)

test_online_threshold_adaptation_reduces_alpha_high:
    wait_ratio 높을 때 adapt_thresholds() → alpha_high 감소

test_online_threshold_adaptation_increases_alpha_high:
    wait_ratio 낮을 때 adapt_thresholds() → alpha_high 증가

test_global_occupancy_multinode_average:
    로컬 0.7 + 원격 0.9 → global_occupancy() == 0.8

test_reset_stats_clears_scheduling_times:
    reset_stats() 후 scheduling_overhead_ms_p50() == 0.0

test_base_scheduler_interface_schedule:
    schedule() 반환값이 list 타입 확인
```

### 통합 테스트 최소 요구사항 (test_congestion_specattn_e2e.py)

```
test_e2e_put_get_basic:
    put → get 왕복 기본 동작

test_e2e_set_logits_then_put_applies_mask:
    set_verification_logits → put → get_importance_mask() 확인

test_e2e_congestion_triggers_retention_reduction:
    update_kv_pool(혼잡 수준) → retention_ratio 감소 → put → 더 강한 희소화 확인

test_e2e_free_restores_retention:
    update_kv_pool(여유 수준) → retention_ratio 원래 값 복원

test_e2e_schedule_during_congestion:
    CONGESTED 상태에서 schedule() → [] 반환

test_e2e_accuracy_preserved_cosine_above_099:
    retention_ratio=0.80 상태에서 put → get 왕복 cosine_sim >= 0.99 (MANDATORY)

test_e2e_memory_reduction_above_30pct:
    retention_ratio=0.70 + int4 → memory_reduction_ratio() >= 0.30

test_e2e_metrics_summary_all_keys:
    metrics_summary()가 필수 키(scheduler_overhead_ms_p50, codec_hit_rate 등) 포함

test_e2e_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_e2e_cross_ac_vs_solo_throughput:
    Cross-1 처리량이 A-1 단독 대비 동일 이상 (§5 검증)

test_e2e_runner_integration:
    InferenceRunner(cache=DualReductionPipeline, scheduler=...) 로 run_batch() 호출 성공
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 3개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - `test_specattn_retention_80pct_relative_error_below_1pct` 통과 (relative_error < 0.01)
      - `test_specattn_retention_80pct_cosine_similarity_above_099` 통과 (cosine_sim >= 0.99)
      - `test_specattn_memory_reduction_above_30pct` 통과 (reduction >= 0.30)
- [ ] `evaluation_criteria.md` §2 Activity A 항목 충족:
      - `test_scheduling_overhead_below_5ms` 통과 (TTFT p50 +5% 이내, MANDATORY)
      - `test_schedule_congested_returns_empty` 통과 (혼잡 시 어드미션 차단)
      - `test_admit_free_allows_all` 통과 (여유 시 전부 허용)
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함:
      - `test_e2e_accuracy_preserved_cosine_above_099` 통과 (MANDATORY)
      - A-1 단독 / C-1 단독 / Cross-1 3방향 비교 수치 기록
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - CacheStore 인터페이스 모든 추상 메서드 구현
        (SpecAttnVerificationGuidedKVSparseCodec, CongestionAdmissionSpecAttnDualReductionPipeline)
      - `src/cache/base.py` 변경 후 기존 모든 구현체 회귀 없음 확인
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-20.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-20/metrics.json`에 JSON 기록:
      ```json
      {
        "inference_throughput_improvement_pct": ...,
        "kv_memory_reduction_ratio": ...,
        "specattn_relative_error_ratio_080": ...,
        "specattn_relative_error_ratio_070": ...,
        "specattn_cosine_similarity_ratio_080": ...,
        "specattn_kl_divergence": ...,
        "effective_context_length_multiplier": ...,
        "scheduling_overhead_ttft_p50_ms": ...,
        "kv_pool_occupancy_avg": ...,
        "cache_hit_rate_improvement_pct": ...,
        "cross_ac_accuracy_cosine": ...,
        "cross_ac_throughput_vs_solo_a1_pct": ...,
        "cross_ac_throughput_vs_solo_c1_pct": ...,
        "cross_ac_memory_vs_solo_pct": ...
      }
      ```
- [ ] `src/cache/base.py` `get_importance_mask()` 선택적 메서드 추가 — 기존 6개 추상 메서드 불변
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
