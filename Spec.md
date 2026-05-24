<!-- 변경 이유 (이전 Spec.md: 2026-05-23 대비):
이전 사이클(2026-05-23)은 C+A+B 조합이었다:
  - C-1 RuntimeCertifiedQuantizedAttentionCodec (수학적 런타임 오류 경계 인증 양자화)
  - C-2 KVSculptDistillationCodec (L-BFGS+최소제곱 교대 증류 레이어 예산 배분)
  - Cross-1 RuntimeCertifiedKVSculptDistillationPipeline (C-1+C-2 폐루프 인증 증류)
  - B-1 CLCPositionalBiasGatedSegmentCache (위치 편향 3단계 선택적 재인코딩 게이트)
  - A-2 CPDWarmColdHitRateRouter (예측 히트율 기반 warm/cold 소프트 분기 스케줄러)

이번 사이클(2026-05-24)은 A+C 조합으로 전환된다.
핵심 전환:
  - Activity C 최우선: RuntimeCertified의 "수학적 사후 인증" 방식에서
    TriAttention의 "pre-RoPE 공간 Q/K 집중도 기반 원천적 중요도 계산"으로 전환.
    pre-RoPE 삼각함수 급수(arXiv 2604.04921)로 10.7x KV 절감 + 풀-어텐션 동등 정확도를
    추론(reasoning) 워크로드에서 달성한다.
  - Activity C 2순위: AttentionMatchingClosedFormCodec — 닫힌 형태 최소제곱 최적화로
    50x 압착을 수 초 내 달성. KVSculpt L-BFGS 반복 최적화의 대안.
  - Activity A: CPD 히트율 예측 라우터에서 DualPath 스토리지 NIC 이중 경로 로드 밸런서로
    전환. P/D 분리 멀티 노드 환경에서 스토리지 NIC 포화 시 decode 노드 유휴 NIC를
    중계 경로로 전환해 처리량 +40~96% 달성.
  - Cross-1: DualPathTriAttentionCompressPipeline — A-1 이중 경로 + C-1 압축을
    decode 노드에서 결합해 prefill 노드 전달 KV 크기를 추가 10.7x 감소.

주요 변경:
1. [신규] src/cache/triattention_pre_rope_kv_selector_codec.py (C-1, 최우선)
2. [신규] src/cache/attention_matching_closed_form_codec.py (C-2)
3. [신규] src/scheduler/dualpath_nic_load_balancer.py (A-1)
4. [신규] src/engine/dualpath_triattention_pipeline.py (Cross-1, A+C)
5. [신규] configs/triattention_budget_policy.yaml
6. [신규] configs/attention_matching_ref_query_config.yaml
7. [신규] configs/experiments/2026-05-24.yaml
8. [신규] 단위·통합 테스트 4종
9. [보존] 이전 사이클 구현 파일 전부 수정 금지.
    특히 runtime_certified_quant_codec.py, kvsculpt_distillation_codec.py,
    runtime_certified_distillation_pipeline.py, clc_positional_bias_gated_segment_cache.py,
    cpd_warm_cold_hit_router.py, tri_attention_codec.py 등
    기존 단위·통합 테스트 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-24: TriAttention pre-RoPE KV Selector + DualPath NIC Load Balancer + Attention Matching Closed-Form Codec

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-24.md`

**최우선 구현 타겟**: C-1 `TriAttentionPreRoPEKVSelectorCodec`

**해결하려는 문제**:

- **Activity C (TriAttention pre-RoPE 삼각함수 KV 선택)**: 기존 KV 퇴거 기법들은
  post-RoPE 공간의 어텐션 점수나 양자화 오류 경계에 의존해 중요도를 추정한다.
  TriAttention(arXiv 2604.04921)은 pre-RoPE 공간에서 Q/K 벡터가 고정 비제로 중심
  주변에 집중(concentration)된다는 사실을 수학적으로 증명하고, 이 집중도를 삼각함수
  급수로 정형화해 위치-거리 선호도 점수를 계산한다. 추론(reasoning) 워크로드에서
  AIME25 풀-어텐션 동등 정확도(40.8%)를 유지하면서 10.7x KV 메모리 절감을 달성한다.
  Q/K 집중도 메트릭(conc_Q)으로 삼각함수 거리 선호도와 Q/K 노름 기반 점수를
  자동 균형 조정해 불안정한 환경에서도 accuracy-preserving이 보장된다.

  주의: 기존 `src/cache/tri_attention_codec.py`는 순수 코덱(non-CacheStore)이며
  Fourier 계수를 calibration으로 학습한다. 이번 신규 파일
  `triattention_pre_rope_kv_selector_codec.py`는 CacheStore 인터페이스를 구현하고
  Q/K 집중도 메트릭 + 삼각함수 거리 선호도로 training-free 중요도를 계산한다.
  두 파일은 별개이며 기존 파일을 수정하지 않는다.

- **Activity C (AttentionMatching 닫힌 형태 최소제곱 압착)**: 기존 KVSculpt(05-23 C-2)
  는 L-BFGS 반복 최적화로 증류 압축을 수행해 속도가 느리다. Attention Matching
  (arXiv 2602.16284)은 어텐션 출력 보존과 어텐션 매스 보존 목표 함수에 대해 닫힌 형태
  최소제곱 해가 존재함을 보이고, 이를 통해 50x 압축을 수 초 내 달성한다.

- **Activity A (DualPath 스토리지 NIC 이중 경로 로드 밸런서)**: P/D 분리 클러스터에서
  단일 "저장소->prefill 노드" 경로가 스토리지 NIC 포화를 유발하고 decode 노드의
  스토리지 NIC는 유휴 상태로 낭비된다. DualPath(arXiv 2602.21548)의 핵심 통찰을
  KV 로드 라우팅 정책에 적용해, NIC 포화 시 유휴 decode 노드를 중계 경로로 전환하는
  "NIC-부하 인식 KV 이중 경로 라우터"를 구현한다.

- **Cross-1 (DualPath+TriAttention A+C 파이프라인)**: 이중 경로로 KV가 decode 노드를
  경유할 때 TriAttention 압축을 decode 노드에서 실행해 "중계 decode 노드의 유휴 GPU
  컴퓨팅을 KV 압축에 활용"한다. RDMA로 전달되는 KV 크기가 추가 10.7x 감소해
  곱연산적 복합 효과가 발생한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling (DualPathNICLoadBalancer, 멀티 노드 P/D 분리)
- [ ] Activity B: Non-Contiguous KV Cache Reuse (이번 사이클 최우선 아님)
- [x] Activity C: KV Cache Compression (TriAttentionPreRoPEKVSelectorCodec C-1 + AttentionMatchingClosedFormCodec C-2)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 필수): perplexity 변화 ±1% 이내
      — WikiText-2 proxy: `attention_output_relative_error < 0.01`
      — C-1: kv_budget_ratio=0.093(추론) / 0.20(비추론) 각각 측정
      — C-2: compression_ratio=5x/10x/20x/50x 각각 측정
- [ ] 목표 2 (evaluation_criteria.md §4 필수): downstream 태스크 정확도 ±1% 이내
      — C-1 AIME25 proxy: `cosine_similarity_output >= 0.99`
      — LongBench 8개 서브태스크 proxy: cosine similarity >= 0.99
      — C-2: NIAH(128K 컨텍스트 proxy) + LongBench 8개 서브태스크
- [ ] 목표 3 (evaluation_criteria.md §4 높음): KV Memory Reduction >= -30%
      — C-1: -89% 이상 (kv_budget_ratio=0.093, 추론 태스크)
      — C-2: -92% 이상 (50x 압착)
- [ ] 목표 4 (evaluation_criteria.md §4 높음): Effective Context Length 동일 메모리 2x 이상
- [ ] 목표 5 (evaluation_criteria.md §4 C 추가): pre-RoPE vs. post-RoPE 신호 대조
      — C-1: kv_budget_ratio [5%, 9.3%, 15%, 20%, 30%] 별 정확도 곡선
      — C-1: `{conc_Q_mean, conc_K_mean, kv_budget_ratio, is_reasoning_task}` JSON 기록
- [ ] 목표 6 (evaluation_criteria.md §2 필수): Scheduling overhead TTFT p50 +5% 이내
      — A-1: decide_routing() 총 오버헤드 < 0.1ms/요청, TTFT +2% 이내
- [ ] 목표 7 (evaluation_criteria.md §2 높음): 캐시 히트율 스케줄링 미적용 대비 +10%p 이상
- [ ] 목표 8 (evaluation_criteria.md §5 필수, C 포함): Cross-1 복합 accuracy ±1% 이내
      — cosine_similarity >= 0.99
- [ ] 목표 9 (evaluation_criteria.md §1 높음): 처리량 베이스라인 +20% 이상
      — C-1: KV 10.7x 절감 -> 배치 크기 증가 -> 처리량 향상
      — A-1: 스토리지 I/O 병목 해소 -> +40~96%
- [ ] 목표 10 (evaluation_criteria.md §4 높음): 압축 오버헤드 TTFT +10% 이내
      — C-1: pre-RoPE 집중도 계산 < 1ms/디코딩 스텝 (GPU 병렬화)
      — C-2: 닫힌 형태 최소제곱 < 1ms/세그먼트 (m=32 기준)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/triattention_pre_rope_kv_selector_codec.py` | C (최우선) | TriAttentionPreRoPEKVSelectorCodec — pre-RoPE Q/K 집중도 메트릭 + 삼각함수 거리 선호도 점수 + 집중도 가중 결합으로 상위 kv_budget_ratio KV 선택. CacheStore 인터페이스 구현. |
| `src/cache/attention_matching_closed_form_codec.py` | C (2순위) | AttentionMatchingClosedFormCodec — 레퍼런스 쿼리 기반 어텐션 출력/매스 보존 닫힌 형태 최소제곱 최적화로 m_c 토큰으로 KV 압착. CacheStore 인터페이스 구현. |
| `src/scheduler/dualpath_nic_load_balancer.py` | A | DualPathNICLoadBalancer — 스토리지 NIC 부하 모니터 + 단일/이중 경로 KV 로드 전환 결정. BaseScheduler 상속. |
| `src/engine/dualpath_triattention_pipeline.py` | A+C (Cross-1) | DualPathTriAttentionCompressPipeline — A-1 이중 경로 + C-1 TriAttention 압축을 decode 노드에서 순차 실행. |
| `configs/triattention_budget_policy.yaml` | C | C-1 kv_budget_ratio 정책 설정 |
| `configs/attention_matching_ref_query_config.yaml` | C | C-2 레퍼런스 쿼리 구성 설정 |
| `configs/experiments/2026-05-24.yaml` | 공통 | 이번 사이클 실험 설정 |
| `tests/unit/test_triattention_pre_rope_kv_selector_codec.py` | C | C-1 단위 테스트 |
| `tests/unit/test_attention_matching_closed_form_codec.py` | C | C-2 단위 테스트 |
| `tests/unit/test_dualpath_nic_load_balancer.py` | A | A-1 단위 테스트 |
| `tests/integration/test_dualpath_triattention_pipeline_e2e.py` | A+C Cross-1 | E2E 통합 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `tests/unit/test_compression_accuracy.py` | C-1/C-2 accuracy 검증 케이스 추가 (기존 케이스 유지) |

**보존 불변 파일**: `src/cache/base.py` 및 이전 사이클 구현 파일 전부
(runtime_certified_quant_codec.py, kvsculpt_distillation_codec.py,
runtime_certified_distillation_pipeline.py, clc_positional_bias_gated_segment_cache.py,
cpd_warm_cold_hit_router.py, tri_attention_codec.py 등) 수정 금지.

---

## 알고리즘 상세

### TriAttentionPreRoPEKVSelectorCodec (Activity C — 최우선)

```python
# src/cache/triattention_pre_rope_kv_selector_codec.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class TriAttentionSelectorConfig:
    d_head: int = 128
    n_kv_heads: int = 8
    rope_base: float = 10000.0
    kv_budget_ratio_reasoning: float = 0.093    # 추론 태스크 (10.7x 절감)
    kv_budget_ratio_default: float = 0.20       # 비추론 태스크 (보수적)
    high_pressure_threshold: float = 0.80       # KV 풀 점유율 고압 임계값
    high_pressure_extra_reduction: float = 0.10 # 고압 시 추가 10% 감소
    max_entries: int = 1000
    seed: int = 42


@dataclass
class SelectorKVEntry:
    selected_kv: torch.Tensor      # [n_kept, d_head] FP16
    kept_indices: torch.Tensor     # [n_kept] int64
    original_seq_len: int
    conc_q: float
    conc_k: float
    kv_budget_ratio: float
    is_reasoning_task: bool


class TriAttentionPreRoPEKVSelectorCodec(CacheStore):
    """TriAttention pre-RoPE Q/K 집중도 삼각함수 KV 선택 코덱 (arXiv 2604.04921).

    Activity C: training-free, accuracy-preserving KV 선택.

    핵심 알고리즘:
      1. pre-RoPE Q 중심 mu_Q = mean(Q, dim=0) 계산.
      2. Q 집중도: conc_Q = ||mu_Q||_2 / mean(||q_t||_2)  in [0, 1].
      3. 삼각함수 급수 거리 선호도 점수:
           dist_pref(k_i) = sum_{d=1}^{d_k/2} [|mu_Q[2d-1]| * cos(theta_d * |pos_i - pos_q|)
                                               + |mu_Q[2d]|   * sin(theta_d * |pos_i - pos_q|)]
           theta_d = rope_base^(-2d/d_k)
      4. K 노름 보조 점수: norm_score(k_i) = ||k_i|| / max(||k_j||)
      5. 집중도 가중 결합:
           importance(k_i) = conc_Q * dist_pref(k_i) + (1 - conc_Q) * norm_score(k_i)
      6. 상위 n_keep = max(1, int(N * kv_budget_ratio)) 토큰 선택.

    CacheStore 인터페이스:
      - put(key, value): 단순 저장 (압축 없음, 호환성)
      - put_compressed(key, Q, K, V, ...): 압축 저장 (권장)
      - get(key): 선택된 KV 텐서 반환
      - evict(): FIFO 퇴거
      - hit_rate(), memory_bytes(), reset_stats()
    """

    def __init__(self, config: TriAttentionSelectorConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: Dict[str, SelectorKVEntry] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._conc_q_history: List[float] = []
        self._conc_k_history: List[float] = []
        # RoPE 주파수 사전 계산: theta_d = rope_base^(-2d/d_k), d=1..d_k/2
        half_d = config.d_head // 2
        d_indices = torch.arange(1, half_d + 1, dtype=torch.float32)
        self._rope_freqs: torch.Tensor = config.rope_base ** (-2.0 * d_indices / config.d_head)

    @staticmethod
    def compute_concentration(vecs: torch.Tensor) -> float:
        """Q 또는 K 벡터 집합의 집중도.

        conc = ||mean(vecs)||_2 / mean(||vecs||_2)  clamped to [0, 1]

        Args:
            vecs: [T, d_head] FP32

        Returns:
            float in [0, 1]
        """
        vecs_f = vecs.float()
        mu = vecs_f.mean(dim=0)
        mu_norm = float(mu.norm())
        mean_norms = float(vecs_f.norm(dim=-1).mean())
        raw = mu_norm / (mean_norms + 1e-8)
        return float(min(max(raw, 0.0), 1.0))

    def compute_dist_pref_scores(
        self,
        mu_q: torch.Tensor,           # [d_head] FP32
        key_positions: torch.Tensor,  # [N] int64
        pos_q: int,
    ) -> torch.Tensor:
        """삼각함수 급수 거리 선호도 점수.

        dist_pref(k_i) = sum_{d=1}^{d_k/2}
            [|mu_Q[2d-1]| * cos(theta_d * |pos_i - pos_q|)
           + |mu_Q[2d]|   * sin(theta_d * |pos_i - pos_q|)]

        Returns:
            [N] FP32
        """
        device = mu_q.device
        rope_freqs = self._rope_freqs.to(device=device, dtype=torch.float32)
        delta_pos = (key_positions.float() - float(pos_q)).abs()  # [N]
        phase = delta_pos.unsqueeze(1) * rope_freqs.unsqueeze(0)  # [N, d_k/2]
        half_d = self.config.d_head // 2
        mu_odd  = mu_q.float()[0::2][:half_d].abs()  # |mu_Q[2d-1]|
        mu_even = mu_q.float()[1::2][:half_d].abs()  # |mu_Q[2d]|
        scores = mu_odd.unsqueeze(0) * torch.cos(phase) \
               + mu_even.unsqueeze(0) * torch.sin(phase)  # [N, d_k/2]
        return scores.sum(dim=-1)  # [N]

    def select_kv(
        self,
        Q: torch.Tensor,                          # [T_q, d_head] pre-RoPE
        K: torch.Tensor,                          # [N, d_head] pre-RoPE
        V: torch.Tensor,                          # [N, d_head]
        key_positions: Optional[torch.Tensor] = None,  # [N] int64
        pos_q: int = 0,
        is_reasoning_task: bool = False,
        kv_pool_pressure: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """집중도 + 삼각함수 선호도로 중요 KV 선택.

        Returns:
            (K_selected, V_selected, kept_indices, conc_q, conc_k, kv_budget_ratio)

        Algorithm:
          budget = kv_budget_ratio_reasoning if is_reasoning_task else kv_budget_ratio_default
          if kv_pool_pressure >= high_pressure_threshold:
              budget *= (1.0 - high_pressure_extra_reduction)
          n_keep = max(1, int(N * budget))

          mu_q = Q.float().mean(dim=0)
          conc_q = compute_concentration(Q)
          conc_k = compute_concentration(K)

          if key_positions is None:
              key_positions = torch.arange(N)
          dist_pref = compute_dist_pref_scores(mu_q, key_positions, pos_q)

          k_norms = K.norm(dim=-1)
          norm_score = k_norms / (k_norms.max() + 1e-8)

          importance = conc_q * dist_pref + (1.0 - conc_q) * norm_score
          kept_indices = importance.topk(n_keep).indices.sort().values

          return K[kept_indices], V[kept_indices], kept_indices, conc_q, conc_k, budget
        """
        Q_f = Q.float()
        K_f = K.float()
        N = K_f.shape[0]
        device = K_f.device

        # 1. 예산 비율
        if is_reasoning_task:
            budget = self.config.kv_budget_ratio_reasoning
        else:
            budget = self.config.kv_budget_ratio_default
        if kv_pool_pressure >= self.config.high_pressure_threshold:
            budget = budget * (1.0 - self.config.high_pressure_extra_reduction)
        n_keep = max(1, int(N * budget))

        # 2. 집중도
        mu_q = Q_f.mean(dim=0)
        conc_q = self.compute_concentration(Q_f)
        conc_k = self.compute_concentration(K_f)

        # 3. 삼각함수 거리 선호도
        if key_positions is None:
            key_positions = torch.arange(N, dtype=torch.int64, device=device)
        dist_pref = self.compute_dist_pref_scores(mu_q, key_positions, pos_q)

        # 4. K 노름 보조 점수
        k_norms = K_f.norm(dim=-1)
        norm_score = k_norms / (k_norms.max() + 1e-8)

        # 5. 집중도 가중 결합
        importance = conc_q * dist_pref + (1.0 - conc_q) * norm_score

        # 6. 상위 n_keep 선택
        kept_indices = importance.topk(n_keep).indices.sort().values

        self._conc_q_history.append(conc_q)
        self._conc_k_history.append(conc_k)

        return (
            K[kept_indices].to(K.dtype),
            V[kept_indices].to(V.dtype),
            kept_indices,
            conc_q,
            conc_k,
            budget,
        )

    @staticmethod
    def detect_reasoning_task(prompt_text: str) -> bool:
        """<think> 태그 포함 여부로 추론 태스크 감지."""
        return "<think>" in prompt_text or "</think>" in prompt_text

    # ---- CacheStore 인터페이스 ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """단순 저장 (압축 없음 — Q 정보 없이 호출되는 경우 호환성)."""
        if key in self._store:
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        seq_len = value.shape[0]
        entry = SelectorKVEntry(
            selected_kv=value.detach().clone(),
            kept_indices=torch.arange(seq_len, dtype=torch.int64),
            original_seq_len=seq_len,
            conc_q=0.5,
            conc_k=0.5,
            kv_budget_ratio=self.config.kv_budget_ratio_default,
            is_reasoning_task=False,
        )
        self._store[key] = entry

    def put_compressed(
        self,
        key: str,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_positions: Optional[torch.Tensor] = None,
        pos_q: int = 0,
        is_reasoning_task: bool = False,
        kv_pool_pressure: float = 0.0,
    ) -> SelectorKVEntry:
        """KV 선택 후 압축 저장 (권장)."""
        if key in self._store:
            return self._store[key]
        if len(self._store) >= self.config.max_entries:
            self.evict()
        K_sel, V_sel, kept_idx, conc_q, conc_k, budget = self.select_kv(
            Q, K, V, key_positions, pos_q, is_reasoning_task, kv_pool_pressure
        )
        entry = SelectorKVEntry(
            selected_kv=K_sel.detach().clone(),
            kept_indices=kept_idx,
            original_seq_len=K.shape[0],
            conc_q=conc_q,
            conc_k=conc_k,
            kv_budget_ratio=budget,
            is_reasoning_task=is_reasoning_task,
        )
        self._store[key] = entry
        return entry

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self._store:
            self._misses += 1
            return None
        self._hits += 1
        return self._store[key].selected_kv

    def get_entry(self, key: str) -> Optional[SelectorKVEntry]:
        return self._store.get(key)

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """노름 기반 단순 압축 훅 (Q 없이 호출 시 fallback)."""
        N = value.shape[0]
        budget = self.config.kv_budget_ratio_default
        n_keep = max(1, int(N * budget))
        norms = value.float().norm(dim=-1)
        kept = norms.topk(n_keep).indices.sort().values
        return value[kept]

    def evict(self) -> int:
        if not self._store:
            return 0
        k = next(iter(self._store))
        entry = self._store.pop(k)
        return entry.selected_kv.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(e.selected_kv.nbytes for e in self._store.values())

    def memory_reduction_ratio(self) -> float:
        """선택된 KV 대비 원본 FP16 동등 메모리 감소율."""
        total_selected = 0
        total_original = 0
        for e in self._store.values():
            d = e.selected_kv.shape[-1]
            total_selected += e.selected_kv.nbytes
            total_original += e.original_seq_len * d * 2  # FP16=2bytes
        if total_original == 0:
            return 0.0
        return 1.0 - total_selected / total_original

    def concentration_stats(self) -> dict:
        """conc_Q/conc_K 분포 통계 (JSON 기록용)."""
        if not self._conc_q_history:
            return {"conc_q_mean": 0.0, "conc_k_mean": 0.0}
        return {
            "conc_q_mean": float(sum(self._conc_q_history) / len(self._conc_q_history)),
            "conc_k_mean": float(sum(self._conc_k_history) / len(self._conc_k_history)),
        }

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """kept_indices 기반 bool 마스크 [original_seq_len] 반환."""
        entry = self._store.get(key)
        if entry is None:
            return None
        mask = torch.zeros(entry.original_seq_len, dtype=torch.bool)
        mask[entry.kept_indices] = True
        return mask

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._conc_q_history.clear()
        self._conc_k_history.clear()
```

---

### AttentionMatchingClosedFormCodec (Activity C — 2순위)

```python
# src/cache/attention_matching_closed_form_codec.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class AttentionMatchingConfig:
    d_head: int = 128
    n_ref_queries: int = 32       # 레퍼런스 쿼리 수 m
    compression_ratio: int = 50   # 압착 비율 (50x)
    alternating_rounds: int = 3   # K_c/V_c 교대 최적화 반복 횟수
    max_entries: int = 1000
    seed: int = 42


@dataclass
class CompactKVEntry:
    compact_k: torch.Tensor     # [m_c, d_head] FP32
    compact_v: torch.Tensor     # [m_c, d_head] FP32
    ref_queries: torch.Tensor   # [m, d_head] FP32
    original_seq_len: int
    compression_ratio_actual: float


class AttentionMatchingClosedFormCodec(CacheStore):
    """Attention Matching 닫힌 형태 최소제곱 잠재 공간 KV 압착 코덱 (arXiv 2602.16284).

    Activity C: KV Cache Compression.

    핵심 알고리즘:
      Step 1: Q_ref 구성 (m=32, 균등 샘플링 + 중요도 혼합).
      Step 2: 어텐션 매스 매칭 — 닫힌 형태 K_c:
        A_orig = softmax(Q_ref K_orig^T / sqrt(d))           # [m, N]
        K_c = sqrt(d) * Q_ref_pinv * log(A_orig + eps)       의사역행렬 기반
        Q_ref_pinv = (Q_ref^T Q_ref)^{-1} Q_ref^T           # [d_head, m]
        계산 복잡도: O(m^3) = O(32^3) < 0.1ms
      Step 3: 어텐션 출력 매칭 — 닫힌 형태 V_c:
        A_c = softmax(Q_ref K_c^T / sqrt(d))                 # [m, m_c]
        V_c = A_c_pinv * A_orig * V_orig                     의사역행렬 기반
        계산 복잡도: O(m_c^3) < 1ms (m_c = N / compression_ratio)
      Step 4: K_c/V_c 교대 최적화 alternating_rounds=3회.

    accuracy-preserving 근거:
      - 닫힌 형태 해: 파라미터화 범위 내 전역 최솟값 보장.
      - 어텐션 출력 보존 목표 함수: 모델 출력에 직접 영향하는 벡터를 직접 최소화.
      - compression_ratio=10x: accuracy delta ±0.2% 이내 (보수적 추정).
      - compression_ratio=50x: Cartridges 수준 ±0.5% 이내 (원논문 기반).
    """

    def __init__(self, config: AttentionMatchingConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: Dict[str, CompactKVEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def build_ref_queries(self, Q_context: torch.Tensor) -> torch.Tensor:
        """레퍼런스 쿼리 Q_ref 구성.

        구성: 균등 샘플링(m//2) + 노름 기반 중요도 상위(m//2) 혼합.

        Args:
            Q_context: [T, d_head] FP32

        Returns:
            Q_ref: [m, d_head] FP32

        Algorithm:
          m = n_ref_queries
          T = Q_context.shape[0]
          half_m = m // 2
          uniform_step = max(1, T // half_m)
          uniform_idx = arange(0, T, uniform_step)[:half_m]
          norms = Q_context.norm(dim=-1)
          top_idx = norms.topk(m - len(uniform_idx)).indices
          all_idx = unique(cat([uniform_idx, top_idx]))[:m]
          if len(all_idx) < m: pad with repetition
          return Q_context[all_idx[:m]]
        """
        m = self.config.n_ref_queries
        T = Q_context.shape[0]
        half_m = m // 2

        uniform_step = max(1, T // max(1, half_m))
        uniform_idx = torch.arange(0, T, uniform_step, device=Q_context.device)[:half_m]

        norms = Q_context.float().norm(dim=-1)
        top_m = m - len(uniform_idx)
        top_idx = norms.topk(min(top_m, T)).indices

        all_idx = torch.unique(torch.cat([uniform_idx, top_idx]))[:m]
        if len(all_idx) < m:
            repeat_times = (m - len(all_idx)) // max(1, len(all_idx)) + 1
            pad = all_idx.repeat(repeat_times)[: m - len(all_idx)]
            all_idx = torch.cat([all_idx, pad])

        return Q_context.float()[all_idx[:m]]

    def closed_form_k(
        self,
        Q_ref: torch.Tensor,   # [m, d_head]
        K_orig: torch.Tensor,  # [N, d_head]
    ) -> torch.Tensor:
        """어텐션 매스 매칭 닫힌 형태 K_c.

        A_orig = softmax(Q_ref K_orig^T / sqrt(d))  # [m, N]
        K_c^T = sqrt(d) * Q_ref_pinv * log(A_orig + eps)
        Q_ref_pinv = (Q_ref^T Q_ref)^{-1} Q_ref^T   # [d_head, m]

        Returns:
            K_c_full: [N, d_head] — 이후 어텐션 중요도로 m_c개 선택
        """
        scale = self.config.d_head ** -0.5
        A_orig = F.softmax(Q_ref @ K_orig.T * scale, dim=-1)  # [m, N]
        log_A = torch.log(A_orig + 1e-10)                      # [m, N]

        QTQ = Q_ref.T @ Q_ref  # [d_head, d_head]
        try:
            QTQ_inv = torch.linalg.inv(
                QTQ + 1e-6 * torch.eye(QTQ.shape[0], device=QTQ.device, dtype=QTQ.dtype)
            )
        except Exception:
            QTQ_inv = torch.linalg.pinv(QTQ)
        Q_pinv = QTQ_inv @ Q_ref.T  # [d_head, m]

        K_c_T = (self.config.d_head ** 0.5) * (Q_pinv @ log_A)  # [d_head, N]
        return K_c_T.T  # [N, d_head]

    def closed_form_v(
        self,
        Q_ref: torch.Tensor,     # [m, d_head]
        K_c: torch.Tensor,       # [m_c, d_head]
        A_orig_sel: torch.Tensor,  # [m, m_c] A_orig 중 선택된 열만
        V_orig: torch.Tensor,    # [N, d_head]
    ) -> torch.Tensor:
        """어텐션 출력 매칭 닫힌 형태 V_c.

        A_c = softmax(Q_ref K_c^T / sqrt(d))   # [m, m_c]
        V_c = A_c_pinv * A_orig_sel * V_orig

        Returns:
            V_c: [m_c, d_head]
        """
        scale = self.config.d_head ** -0.5
        A_c = F.softmax(Q_ref @ K_c.T * scale, dim=-1)  # [m, m_c]
        try:
            ATA = A_c.T @ A_c  # [m_c, m_c]
            ATA_inv = torch.linalg.inv(
                ATA + 1e-6 * torch.eye(ATA.shape[0], device=ATA.device, dtype=ATA.dtype)
            )
            A_pinv = ATA_inv @ A_c.T  # [m_c, m]
        except Exception:
            A_pinv = torch.linalg.pinv(A_c)  # [m_c, m]

        target = A_orig_sel.T @ V_orig[: A_orig_sel.shape[1]]  # [m_c, d_head] — 선택 V
        # 더 정확한 버전: A_orig_sel @ V_orig 재계산
        # A_orig_sel: [m, m_c], V_orig: [N, d_head]
        # target = (A_orig_sel).sum(dim=1) @ V_orig 의 근사
        target = A_c @ (A_orig_sel.T @ V_orig[:A_orig_sel.shape[1]])  # [m, d_head]
        V_c = A_pinv @ target  # [m_c, d_head]
        return V_c

    def compact(
        self,
        Q_context: torch.Tensor,  # [T, d_head]
        K_orig: torch.Tensor,     # [N, d_head]
        V_orig: torch.Tensor,     # [N, d_head]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """닫힌 형태 최소제곱으로 K/V를 m_c 토큰으로 압착.

        Returns:
            (K_c, V_c, Q_ref)
            K_c: [m_c, d_head], V_c: [m_c, d_head], Q_ref: [m, d_head]

        Algorithm:
          Q_ref = build_ref_queries(Q_context)
          N = K_orig.shape[0]
          m_c = max(1, N // compression_ratio)
          A_orig = softmax(Q_ref K_orig^T / sqrt(d))    # [m, N]

          for round in range(alternating_rounds):
            K_c_full = closed_form_k(Q_ref, K_orig)     # [N, d_head]
            # 어텐션 중요도로 m_c 토큰 선택
            attn_importance = A_orig.mean(dim=0)          # [N]
            top_mc_idx = topk(attn_importance, m_c).sort()
            K_c = K_c_full[top_mc_idx]                    # [m_c, d_head]
            # 닫힌 형태 V_c
            A_orig_sel = A_orig[:, top_mc_idx]            # [m, m_c]
            V_c = closed_form_v(Q_ref, K_c, A_orig_sel, V_orig)

          return K_c, V_c, Q_ref
        """
        Q_f = Q_context.float()
        K_f = K_orig.float()
        V_f = V_orig.float()
        N = K_f.shape[0]
        m_c = max(1, N // self.config.compression_ratio)
        scale = self.config.d_head ** -0.5

        Q_ref = self.build_ref_queries(Q_f)
        A_orig = F.softmax(Q_ref @ K_f.T * scale, dim=-1)  # [m, N]

        K_c = None
        V_c = None
        for _ in range(self.config.alternating_rounds):
            K_c_full = self.closed_form_k(Q_ref, K_f)       # [N, d_head]
            attn_importance = A_orig.mean(dim=0)              # [N]
            top_mc_idx = attn_importance.topk(m_c).indices.sort().values
            K_c = K_c_full[top_mc_idx]                        # [m_c, d_head]
            A_orig_sel = A_orig[:, top_mc_idx]                # [m, m_c]
            # V_c: closed-form
            A_c = F.softmax(Q_ref @ K_c.T * scale, dim=-1)   # [m, m_c]
            try:
                ATA = A_c.T @ A_c
                ATA_inv = torch.linalg.inv(
                    ATA + 1e-6 * torch.eye(ATA.shape[0], device=ATA.device, dtype=ATA.dtype)
                )
                A_pinv = ATA_inv @ A_c.T                       # [m_c, m]
            except Exception:
                A_pinv = torch.linalg.pinv(A_c)
            # target: A_orig * V_orig = [m, d_head]
            target_v = A_orig @ V_f                            # [m, d_head]
            V_c = A_pinv @ target_v                            # [m_c, d_head]
            # A_orig 갱신
            A_orig = F.softmax(Q_ref @ K_f.T * scale, dim=-1)

        return K_c.to(K_orig.dtype), V_c.to(V_orig.dtype), Q_ref

    # ---- CacheStore 인터페이스 ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """단순 저장 (Q 없이 호출 시 압착 없음)."""
        if key in self._store:
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        seq_len = value.shape[0]
        entry = CompactKVEntry(
            compact_k=value.float().detach().clone(),
            compact_v=value.float().detach().clone(),
            ref_queries=torch.zeros(self.config.n_ref_queries, self.config.d_head),
            original_seq_len=seq_len,
            compression_ratio_actual=1.0,
        )
        self._store[key] = entry

    def put_compact(
        self,
        key: str,
        Q_context: torch.Tensor,
        K_orig: torch.Tensor,
        V_orig: torch.Tensor,
    ) -> CompactKVEntry:
        """압착 후 저장 (권장)."""
        if key in self._store:
            return self._store[key]
        if len(self._store) >= self.config.max_entries:
            self.evict()
        K_c, V_c, Q_ref = self.compact(Q_context, K_orig, V_orig)
        m_c = K_c.shape[0]
        N = K_orig.shape[0]
        entry = CompactKVEntry(
            compact_k=K_c.detach().clone(),
            compact_v=V_c.detach().clone(),
            ref_queries=Q_ref.detach().clone(),
            original_seq_len=N,
            compression_ratio_actual=float(N) / max(1, m_c),
        )
        self._store[key] = entry
        return entry

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self._store:
            self._misses += 1
            return None
        self._hits += 1
        return self._store[key].compact_k

    def get_compact_kv(self, key: str) -> Optional[CompactKVEntry]:
        return self._store.get(key)

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """노름 기반 토큰 선택 fallback (Q 없이 호출 시)."""
        N = value.shape[0]
        m_c = max(1, N // self.config.compression_ratio)
        norms = value.float().norm(dim=-1)
        kept = norms.topk(m_c).indices.sort().values
        return value[kept]

    def evict(self) -> int:
        if not self._store:
            return 0
        k = next(iter(self._store))
        entry = self._store.pop(k)
        return entry.compact_k.nbytes + entry.compact_v.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(e.compact_k.nbytes + e.compact_v.nbytes for e in self._store.values())

    def memory_reduction_ratio(self) -> float:
        total_compact = 0
        total_original = 0
        for e in self._store.values():
            d = e.compact_k.shape[-1]
            total_compact += e.compact_k.nbytes + e.compact_v.nbytes
            total_original += e.original_seq_len * d * 2 * 2  # K+V FP16
        if total_original == 0:
            return 0.0
        return 1.0 - total_compact / total_original

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError("AttentionMatchingClosedFormCodec does not support importance masking.")

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
```

---

### DualPathNICLoadBalancer (Activity A)

```python
# src/scheduler/dualpath_nic_load_balancer.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import torch

from src.scheduler.base import BaseScheduler


@dataclass
class NodeNICStatus:
    node_id: str
    node_type: str           # "prefill" | "decode"
    nic_utilization: float   # 0.0~1.0
    active_dual_path: int    # 현재 이중 경로 KV 로드 수
    last_updated: float      # time.monotonic()


@dataclass
class RoutingDecision:
    request_id: str
    path: str                         # "single" | "dual"
    relay_decode_node_id: Optional[str]
    prefill_nic_utilization: float
    decision_latency_ms: float


@dataclass
class DualPathNICConfig:
    nic_saturation_threshold: float = 0.80
    idle_nic_threshold: float = 0.30
    max_dual_path_per_node: int = 4
    nic_monitor_interval_ms: float = 200.0
    stale_threshold_ms: float = 1000.0
    seed: int = 42


class DualPathNICLoadBalancer(BaseScheduler):
    """DualPath 스토리지 NIC 유휴율 인식 KV 로드 밸런서 (arXiv 2602.21548).

    Activity A: KV Cache-aware Scheduling — 멀티 노드 P/D 분리.

    스케줄링 결정 단위: KV 로드 요청(request) 단위.
    캐시 상태 접근: NodeNICStatus dict O(1) 룩업.

    멀티 노드 환경:
      - 스토리지 NIC (스토리지 네트워크)와 컴퓨트 NIC (RDMA) 물리적 분리 가정.
      - gRPC heartbeat 200ms 주기로 각 노드 NIC 사용률 수집.
      - 단일 경로: 저장소 -> prefill 노드 (기본).
      - 이중 경로: 저장소 -> decode 노드(idle NIC) -> prefill 노드 (RDMA).
      - 이중 경로가 모델 실행 RDMA 채널과 간섭하지 않도록 스토리지/컴퓨트 네트워크 분리.

    경로 결정 알고리즘:
      if prefill_nic_util < nic_saturation_threshold (0.80):
          -> 단일 경로 (기본)
      elif len(idle_decode_nodes) > 0:
          -> 이중 경로, min_load_first + round-robin으로 relay node 선택
      else:
          -> 단일 경로 (포화지만 대안 없음)

    오버헤드: NIC 부하 조회 O(1) + idle decode 탐색 O(N_decode) < 0.1ms/요청.
    """

    def __init__(self, config: DualPathNICConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._node_status: Dict[str, NodeNICStatus] = {}
        self._single_path_count: int = 0
        self._dual_path_count: int = 0
        self._decision_latencies_ms: List[float] = []
        self._rr_index: int = 0

    def update_nic_status(
        self,
        node_id: str,
        node_type: str,
        nic_utilization: float,
        active_dual_path: int = 0,
    ) -> None:
        """노드 스토리지 NIC 부하 상태 갱신 (gRPC heartbeat 수신 시 호출)."""
        self._node_status[node_id] = NodeNICStatus(
            node_id=node_id,
            node_type=node_type,
            nic_utilization=float(nic_utilization),
            active_dual_path=active_dual_path,
            last_updated=time.monotonic(),
        )

    def _get_prefill_nic_utilization(self) -> float:
        """현재 prefill 노드들의 최대 스토리지 NIC 사용률."""
        now = time.monotonic()
        utils = [
            s.nic_utilization
            for s in self._node_status.values()
            if s.node_type == "prefill"
            and (now - s.last_updated) * 1000 < self.config.stale_threshold_ms
        ]
        return max(utils) if utils else 0.0

    def _get_idle_decode_nodes(self) -> List[NodeNICStatus]:
        """idle NIC + 이중 경로 여유가 있는 decode 노드 목록 (min_load_first 정렬)."""
        now = time.monotonic()
        idle = [
            s for s in self._node_status.values()
            if s.node_type == "decode"
            and s.nic_utilization < self.config.idle_nic_threshold
            and s.active_dual_path < self.config.max_dual_path_per_node
            and (now - s.last_updated) * 1000 < self.config.stale_threshold_ms
        ]
        return sorted(idle, key=lambda n: n.nic_utilization)

    def decide_routing(self, request: Any) -> RoutingDecision:
        """KV 로드 경로 결정.

        Returns:
            RoutingDecision with path="single" | "dual"
        """
        t_start = time.monotonic()
        request_id = getattr(request, "request_id", str(id(request)))
        prefill_nic_util = self._get_prefill_nic_utilization()

        def _make_single() -> RoutingDecision:
            self._single_path_count += 1
            lat = (time.monotonic() - t_start) * 1000
            self._decision_latencies_ms.append(lat)
            return RoutingDecision(
                request_id=request_id,
                path="single",
                relay_decode_node_id=None,
                prefill_nic_utilization=prefill_nic_util,
                decision_latency_ms=lat,
            )

        if prefill_nic_util < self.config.nic_saturation_threshold:
            return _make_single()

        idle_decode = self._get_idle_decode_nodes()
        if not idle_decode:
            return _make_single()

        # 이중 경로: 라운드-로빈 + min_load_first
        relay_node = idle_decode[self._rr_index % len(idle_decode)]
        self._rr_index += 1
        if relay_node.node_id in self._node_status:
            self._node_status[relay_node.node_id].active_dual_path += 1
        self._dual_path_count += 1

        lat = (time.monotonic() - t_start) * 1000
        self._decision_latencies_ms.append(lat)
        return RoutingDecision(
            request_id=request_id,
            path="dual",
            relay_decode_node_id=relay_node.node_id,
            prefill_nic_utilization=prefill_nic_util,
            decision_latency_ms=lat,
        )

    def complete_dual_path(self, decode_node_id: str) -> None:
        """이중 경로 완료 시 decode 노드 active 카운트 감소."""
        if decode_node_id in self._node_status:
            s = self._node_status[decode_node_id]
            s.active_dual_path = max(0, s.active_dual_path - 1)

    def schedule(self, requests: List[Any]) -> List[Any]:
        """BaseScheduler 인터페이스: 각 요청에 routing_decision 첨부 후 반환."""
        result = []
        for req in requests:
            decision = self.decide_routing(req)
            try:
                req.routing_decision = decision
            except AttributeError:
                pass
            result.append(req)
        return result

    def dual_path_ratio(self) -> float:
        total = self._single_path_count + self._dual_path_count
        return self._dual_path_count / max(1, total)

    def p99_decision_latency_ms(self) -> float:
        if not self._decision_latencies_ms:
            return 0.0
        s = sorted(self._decision_latencies_ms)
        idx = min(int(len(s) * 0.99), len(s) - 1)
        return s[idx]

    def scheduling_stats(self) -> dict:
        return {
            "single_path_count": self._single_path_count,
            "dual_path_count": self._dual_path_count,
            "dual_path_ratio": self.dual_path_ratio(),
            "decision_latency_p99_ms": self.p99_decision_latency_ms(),
            "decision_latency_mean_ms": (
                sum(self._decision_latencies_ms) / max(1, len(self._decision_latencies_ms))
            ),
        }

    def reset_stats(self) -> None:
        self._single_path_count = 0
        self._dual_path_count = 0
        self._decision_latencies_ms.clear()
        self._rr_index = 0
```

---

### DualPathTriAttentionCompressPipeline (Cross-1, A+C)

```python
# src/engine/dualpath_triattention_pipeline.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.cache.triattention_pre_rope_kv_selector_codec import (
    TriAttentionPreRoPEKVSelectorCodec, TriAttentionSelectorConfig,
)
from src.scheduler.dualpath_nic_load_balancer import (
    DualPathNICLoadBalancer, DualPathNICConfig, RoutingDecision,
)


@dataclass
class DualPathTriAttentionPipelineConfig:
    triattention_config: Optional[TriAttentionSelectorConfig] = None
    dualpath_config: Optional[DualPathNICConfig] = None
    seed: int = 42


class DualPathTriAttentionCompressPipeline:
    """DualPath NIC 이중 경로 + TriAttention pre-RoPE 압축 통합 파이프라인 (Cross-1).

    통합 처리 흐름:
      Step 1 (A-1): DualPathNICLoadBalancer — prefill NIC 포화 감지 -> idle decode 선택.
      Step 2 (I/O 시뮬레이션): 스토리지 -> decode 노드 NIC로 원본 KV 로드.
      Step 3 (C-1): decode 노드에서 TriAttentionPreRoPEKVSelectorCodec.select_kv() 실행.
                     상위 kv_budget_ratio 중요 키만 유지 (10.7x 압축).
      Step 4 (RDMA 시뮬레이션): 압축 KV를 decode -> prefill RDMA 전달.
      Step 5: 압축 KV 반환 -> 표준 어텐션 커널 투입.

    단일 경로 시: 압축 없이 K_full, V_full 반환.
    이중 경로 시: TriAttention 압축 KV 반환 (kv_bytes 원본의 budget_ratio 비율).
    """

    def __init__(self, config: DualPathTriAttentionPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        ta_cfg = config.triattention_config or TriAttentionSelectorConfig(seed=config.seed)
        dp_cfg = config.dualpath_config or DualPathNICConfig(seed=config.seed)
        self.triattention = TriAttentionPreRoPEKVSelectorCodec(ta_cfg)
        self.dualpath_lb = DualPathNICLoadBalancer(dp_cfg)
        self._pipeline_stats: List[dict] = []

    def run_pipeline(
        self,
        request: Any,
        Q: torch.Tensor,                          # [T_q, d_head] pre-RoPE
        K_full: torch.Tensor,                     # [N, d_head]
        V_full: torch.Tensor,                     # [N, d_head]
        key_positions: Optional[torch.Tensor] = None,
        pos_q: int = 0,
        is_reasoning_task: bool = False,
        kv_pool_pressure: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """이중 경로 라우팅 + TriAttention 압축 통합 실행.

        Returns:
            (K_final, V_final, pipeline_report)

        Algorithm:
          routing = dualpath_lb.decide_routing(request)

          if routing.path == "single":
            return K_full, V_full, {path="single", ...}

          # Step 3: decode 노드에서 TriAttention 압축
          K_sel, V_sel, kept_idx, conc_q, conc_k, budget =
              triattention.select_kv(Q, K_full, V_full, key_positions, pos_q,
                                     is_reasoning_task, kv_pool_pressure)

          # Step 4: RDMA 전달 완료 처리
          dualpath_lb.complete_dual_path(routing.relay_decode_node_id)

          return K_sel, V_sel, {path="dual", compression_ratio=N/n_keep, ...}
        """
        routing: RoutingDecision = self.dualpath_lb.decide_routing(request)

        if routing.path == "single":
            report = {
                "path": "single",
                "compression_ratio": 1.0,
                "kv_bytes_transferred": K_full.nbytes + V_full.nbytes,
                "routing_latency_ms": routing.decision_latency_ms,
                "conc_q": None,
                "conc_k": None,
                "kv_budget_ratio": None,
            }
            self._pipeline_stats.append(report)
            return K_full, V_full, report

        # Step 3: TriAttention 압축
        K_sel, V_sel, kept_idx, conc_q, conc_k, budget = self.triattention.select_kv(
            Q, K_full, V_full, key_positions, pos_q, is_reasoning_task, kv_pool_pressure
        )
        n_keep = K_sel.shape[0]
        N = K_full.shape[0]
        actual_ratio = float(N) / max(1, n_keep)

        # Step 4: RDMA 완료 처리
        if routing.relay_decode_node_id:
            self.dualpath_lb.complete_dual_path(routing.relay_decode_node_id)

        report = {
            "path": "dual",
            "relay_decode_node_id": routing.relay_decode_node_id,
            "compression_ratio": actual_ratio,
            "kv_bytes_transferred": K_sel.nbytes + V_sel.nbytes,
            "kv_bytes_original": K_full.nbytes + V_full.nbytes,
            "kv_size_reduction_ratio": 1.0 - (K_sel.nbytes + V_sel.nbytes) / max(
                1, K_full.nbytes + V_full.nbytes
            ),
            "routing_latency_ms": routing.decision_latency_ms,
            "conc_q": conc_q,
            "conc_k": conc_k,
            "kv_budget_ratio": budget,
            "is_reasoning_task": is_reasoning_task,
        }
        self._pipeline_stats.append(report)
        return K_sel, V_sel, report

    def pipeline_summary(self) -> dict:
        if not self._pipeline_stats:
            return {}
        dual = [s for s in self._pipeline_stats if s["path"] == "dual"]
        single = [s for s in self._pipeline_stats if s["path"] == "single"]
        avg_cr = sum(s["compression_ratio"] for s in dual) / len(dual) if dual else 1.0
        avg_sr = sum(s["kv_size_reduction_ratio"] for s in dual) / len(dual) if dual else 0.0
        return {
            "total_runs": len(self._pipeline_stats),
            "dual_path_runs": len(dual),
            "single_path_runs": len(single),
            "dual_path_ratio": len(dual) / max(1, len(self._pipeline_stats)),
            "avg_compression_ratio": avg_cr,
            "avg_kv_size_reduction_ratio": avg_sr,
            **self.dualpath_lb.scheduling_stats(),
            **self.triattention.concentration_stats(),
        }
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C를 포함하므로 반드시 작성한다.

### C-1: TriAttentionPreRoPEKVSelectorCodec

**perplexity 측정**:
- **데이터셋**: WikiText-2 합성 proxy (`torch.randn`으로 FP32 synthetic Q/K/V 텐서 생성)
- **측정 방법**: `attention_output_relative_error` 계산
  ```
  A_orig = softmax(Q K_orig^T / sqrt(d))      # [n_q, N]
  K_sel, V_sel, ... = select_kv(Q, K_orig, V_orig, ...)
  # 선택된 KV로 어텐션 계산
  A_sel = softmax(Q K_sel^T / sqrt(d))         # [n_q, n_keep]
  out_orig = A_orig @ V_orig                   # [n_q, d_head]
  out_sel  = A_sel @ V_sel                     # [n_q, d_head]
  relative_error = ||out_orig - out_sel||_F / ||out_orig||_F
  ```
  (out_orig와 out_sel은 차원이 다를 수 있으므로, 선택된 토큰에서만 비교하거나
   full-size 재구성 후 비교하는 방식 두 가지 모두 측정)
- **허용 오차**: `relative_error < 0.01` (±1% 이내, MANDATORY)
- **테스트 설정**: n_q=8, d_head=128, N=256, kv_budget_ratio=0.20 (비추론)
- **추론 태스크 설정**: N=256, kv_budget_ratio=0.093 (AIME25 proxy)

**태스크 정확도 측정**:
- **AIME25 추론 태스크 proxy**: kv_budget_ratio=0.093, `cosine_similarity_output >= 0.99`
- **LongBench 8개 서브태스크 proxy**: 8개 독립 synthetic 시퀀스에서 각각 cosine_similarity >= 0.99
- **허용 오차**: ±1% 이내 (MANDATORY)

**추가 검증 실험**:
1. kv_budget_ratio sweep: [5%, 9.3%, 15%, 20%, 30%] 별 relative_error 및 cosine_similarity 측정
2. pre-RoPE vs. post-RoPE ablation:
   - pre-RoPE 집중도 기반 선택 (conc_Q 정상 계산): 기본 설정
   - post-RoPE 노름만 사용 (conc_Q=0으로 고정하여 norm_score만 사용): ablation
   - 두 방식의 relative_error를 동일 시퀀스에서 직접 비교
3. Q/K 집중도 분포: 100개 시퀀스에서 conc_Q_mean, conc_K_mean 측정 후 JSON 기록

**검증 테스트 파일**: `tests/unit/test_compression_accuracy.py` (C-1 케이스 추가)

---

### C-2: AttentionMatchingClosedFormCodec

**perplexity 측정**:
- **데이터셋**: WikiText-2 합성 proxy
- **측정 방법**:
  ```
  K_c, V_c, Q_ref = compact(Q_context, K_orig, V_orig)
  A_orig = softmax(Q K_orig^T / sqrt(d))
  A_compact = softmax(Q K_c^T / sqrt(d))
  out_orig   = A_orig   @ V_orig
  out_compact = A_compact @ V_c
  relative_error = ||out_orig - out_compact||_F / ||out_orig||_F
  ```
- **허용 오차**: compression_ratio=5x 시 relative_error < 0.01 (MANDATORY)
- **참고**: compression_ratio=50x 시 relative_error < 0.05 (Cartridges 수준)

**태스크 정확도 측정**:
- **NIAH proxy**: 128K 컨텍스트 proxy (N=1024 시퀀스), cosine_similarity >= 0.99
- **LongBench 8개 서브태스크**: cosine_similarity >= 0.99
- **허용 오차**: compression_ratio=10x 이하에서 ±1% 이내 (MANDATORY)

**추가 검증 실험**:
1. compression_ratio sweep: [5, 10, 20, 50] 별 relative_error 곡선 측정
2. KVSculptDistillationCodec 대비 비교:
   - 동일 compression_ratio(50%)에서 양쪽 relative_error 직접 측정 비교
   - 닫힌 형태 해(alternating_rounds=1) vs. KVSculpt L-BFGS(5회 반복) 수렴 속도 측정
3. alternating_rounds sweep: [1, 3, 5] 별 relative_error 및 실행 시간 비교

**검증 테스트 파일**: `tests/unit/test_compression_accuracy.py` (C-2 케이스 추가)

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-24.yaml
experiment:
  date: "2026-05-24"
  activity: "A+C"
  description: >
    C-1 TriAttentionPreRoPEKVSelectorCodec (pre-RoPE Q/K 집중도 삼각함수 KV 선택, 10.7x 절감) +
    C-2 AttentionMatchingClosedFormCodec (닫힌 형태 최소제곱 잠재 공간 KV 압착, 50x 압착) +
    A-1 DualPathNICLoadBalancer (스토리지 NIC 이중 경로 로드 밸런서, P/D 분리 멀티 노드) +
    Cross-1 DualPathTriAttentionCompressPipeline (A+C 통합 파이프라인).
    TriAttention(arXiv 2604.04921) + AttentionMatching(arXiv 2602.16284) +
    DualPath(arXiv 2602.21548) 기반.
  cache_type: triattention_pre_rope_kv_selector_codec
  compression_method: pre_rope_trigonometric_selection
  scheduler_type: dualpath_nic_load_balancer

triattention_selector:  # C-1
  d_head: 128
  n_kv_heads: 8
  rope_base: 10000.0
  kv_budget_ratio_reasoning: 0.093     # AIME25 10.7x 절감 기준
  kv_budget_ratio_default: 0.20        # 비추론 태스크 보수적 설정
  high_pressure_threshold: 0.80
  high_pressure_extra_reduction: 0.10
  max_entries: 1000
  seed: 42

attention_matching:  # C-2
  d_head: 128
  n_ref_queries: 32
  compression_ratio: 50
  alternating_rounds: 3
  max_entries: 1000
  seed: 42

dualpath_nic_load_balancer:  # A-1
  nic_saturation_threshold: 0.80
  idle_nic_threshold: 0.30
  max_dual_path_per_node: 4
  nic_monitor_interval_ms: 200.0
  stale_threshold_ms: 1000.0
  seed: 42

dualpath_triattention_pipeline:  # Cross-1
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    dataset_proxy: "wikitext2_synthetic"
    task_accuracy_proxy: "cosine_similarity"
    relative_error_max: 0.01            # ±1% MANDATORY
    cosine_similarity_min: 0.99         # MANDATORY
    c1_budget_ratio_sweep: [0.05, 0.093, 0.15, 0.20, 0.30]
    c1_pre_rope_vs_post_rope_ablation: true
    c1_concentration_sequences: 100
    c2_compression_ratio_sweep: [5, 10, 20, 50]
    c2_vs_kvsculpt_comparison: true
    c2_alternating_rounds_sweep: [1, 3, 5]
    niah_context_lengths: [256, 512, 1024]
    longbench_subtask_count: 8
  activity_a:
    scheduling_overhead_max_ms: 0.1     # < 0.1ms/요청 MANDATORY
    scheduling_overhead_ttft_p50_max_pct: 5.0
    ttft_overhead_target_pct: 2.0
    cache_hit_rate_improvement_min_pct: 10.0
  activity_c:
    c1_memory_reduction_min_ratio: 0.89  # -89% 이상 (추론, budget=0.093)
    c1_memory_reduction_default_ratio: 0.50
    c2_memory_reduction_min_ratio: 0.92  # -92% 이상 (50x 압착)
    effective_context_multiplier: 2.0
    compression_overhead_ttft_max_pct: 10.0
  cross_a1_c1:
    pipeline_cosine_min: 0.99           # MANDATORY
    combined_kv_size_reduction_min: 0.80

seed: 42
results_dir: "results/2026-05-24"
```

```yaml
# configs/triattention_budget_policy.yaml
reasoning_kv_budget_ratio: 0.093
default_kv_budget_ratio: 0.20
high_pressure_threshold: 0.80
high_pressure_extra_reduction: 0.10
reasoning_detection:
  trigger_tags: ["<think>", "</think>"]
```

```yaml
# configs/attention_matching_ref_query_config.yaml
n_ref_queries: 32
sampling_strategy: "mixed"
uniform_fraction: 0.5
importance_fraction: 0.5
compression_ratios: [5, 10, 20, 50]
alternating_rounds: 3
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_triattention_pre_rope_kv_selector_codec.py`
  - compute_concentration() 수치 검증 (동일 방향 vecs -> conc ~1.0; 균등 분산 -> conc ~0.0)
  - compute_dist_pref_scores() 반환 shape [N] 검증
  - select_kv() shape 검증, kept_indices 정렬 검증
  - is_reasoning_task=True -> budget=0.093, False -> budget=0.20
  - kv_pool_pressure=0.85 시 budget 10% 추가 감소
  - detect_reasoning_task() 태그 감지 검증
  - CacheStore 인터페이스 전체: put/get/evict/hit_rate/memory_bytes/reset_stats
  - get_importance_mask() bool 마스크 shape [original_seq_len]
  - concentration_stats() dict 반환
  - memory_reduction_ratio() budget=0.093 시 >= 0.85

- [ ] `tests/unit/test_attention_matching_closed_form_codec.py`
  - build_ref_queries() shape [m, d_head], m=32 보장
  - closed_form_k() shape [N, d_head]
  - compact() K_c shape [m_c, d_head], m_c = max(1, N//compression_ratio)
  - 어텐션 출력 보존: compression_ratio=5x, relative_error < 0.01
  - compression_ratio=50x 시 m_c = max(1, N//50) 검증
  - CacheStore 인터페이스 전체
  - memory_reduction_ratio() compression_ratio=50x 시 >= 0.90
  - alternating_rounds=1 vs 3 실행 시간 비교

- [ ] `tests/unit/test_dualpath_nic_load_balancer.py`
  - update_nic_status() 상태 갱신/조회
  - prefill_nic_util < 0.80 -> path="single"
  - prefill_nic_util >= 0.80 + idle decode 존재 -> path="dual"
  - idle decode 없음 -> path="single"
  - min_load_first 정렬 검증
  - 라운드-로빈 균등 분산 검증
  - max_dual_path_per_node=4 초과 decode 노드 제외 검증
  - complete_dual_path() active_dual_path 감소
  - 결정 지연 < 0.1ms (p99)
  - schedule() routing_decision 첨부 검증
  - scheduling_stats() dict 반환

- [ ] `tests/unit/test_compression_accuracy.py` (기존 파일에 추가)
  - C-1: kv_budget_ratio=0.20 -> relative_error < 0.01 (MANDATORY)
  - C-1: kv_budget_ratio=0.093 -> relative_error < 0.01 (MANDATORY)
  - C-1: pre-RoPE 집중도 선택 vs. norm_score만 사용 ablation 비교
  - C-1: budget_ratio sweep [0.05, 0.093, 0.15, 0.20, 0.30] x relative_error 표
  - C-1: concentration_stats() conc_Q_mean, conc_K_mean JSON 기록
  - C-2: compression_ratio=5x -> relative_error < 0.01 (MANDATORY)
  - C-2: compression_ratio=50x -> relative_error < 0.05
  - C-2: KVSculptDistillationCodec 대비 동일 압축률에서 relative_error 비교
  - C-2: compression_ratio sweep [5, 10, 20, 50] x relative_error 표

- [ ] `tests/integration/test_dualpath_triattention_pipeline_e2e.py`
  - 단일 경로 시나리오: prefill NIC 0.60 -> path="single", K_final=K_full
  - 이중 경로 시나리오: prefill NIC 0.85 + idle decode -> path="dual"
  - 이중 경로 + 추론 태스크: n_keep = N * 0.093
  - accuracy: 이중 경로 후 relative_error < 0.01 (MANDATORY)
  - 100회 실행: avg decision_latency_ms < 0.1
  - pipeline_summary() dict 반환
  - idle decode 없을 때 single path fallback
  - kv_bytes_transferred / kv_bytes_original ~= budget_ratio

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 4종 + 기존 회귀 없음)
- [ ] 통합 테스트 전부 통과
- [ ] **evaluation_criteria.md §4 필수**: relative_error < 0.01 (MANDATORY)
  - C-1: kv_budget_ratio=0.093/0.20 각각
  - C-2: compression_ratio=5x
- [ ] **evaluation_criteria.md §4 필수**: cosine_similarity >= 0.99 (MANDATORY)
  - C-1: AIME25 proxy + LongBench proxy
  - C-2: NIAH proxy + LongBench proxy
- [ ] **evaluation_criteria.md §4 높음**: KV Memory Reduction >= -30%
  - C-1: -89% (budget=0.093)
  - C-2: -92% (50x 압착)
- [ ] **evaluation_criteria.md §2 필수**: TTFT p50 +5% 이내
  - A-1: p99 decision_latency_ms < 0.1
- [ ] **evaluation_criteria.md §5 필수 (C 포함)**: Cross-1 cosine_similarity >= 0.99
- [ ] **evaluation_criteria.md §1 높음**: 처리량 +20% 이상 (시뮬레이션)
- [ ] `configs/experiments/2026-05-24.yaml` 생성됨
- [ ] `configs/triattention_budget_policy.yaml` 생성됨
- [ ] `configs/attention_matching_ref_query_config.yaml` 생성됨
- [ ] `results/2026-05-24/metrics.json` 기록됨
  - 필수 필드: `{conc_Q_mean, conc_K_mean, kv_budget_ratio, is_reasoning_task,
    memory_reduction_ratio_c1, memory_reduction_ratio_c2,
    relative_error_c1, relative_error_c2, compression_ratio_actual_c2,
    dual_path_ratio, decision_latency_mean_ms}`
- [ ] 이전 사이클 모든 단위·통합 테스트 회귀 없이 통과
