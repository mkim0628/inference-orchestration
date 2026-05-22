<!-- 변경 이유 (이전 Spec.md: 2026-05-21 대비):
이전 사이클(2026-05-21)은 B+C 조합이었다:
  - B-1 BlockUnionNonContiguousReuseIndex (GQA-aware block-union 비연속 재사용, memcpy 없음)
  - C-1 CompactAttentionBlockUnionCodec (청크드 프리필 특화 KV 선택 블록 테이블 압축)
  - Cross BlockUnionBCPipeline (B+C 통합 파이프라인)

이번 사이클(2026-05-22)은 C+B+A 조합으로 전환된다.
이전 사이클이 "어텐션 마스크 기반 KV 블록 선택"에 초점을 뒀다면,
이번 사이클은 "위치 정보(RoPE 기반 유사 쿼리)가 의미 정보보다 퇴거 결정에 결정적"이라는
DapQ(arXiv 2603.11564)의 새로운 설계 원칙을 적용한다.

주요 변경:
1. [Activity C 교체] CompactAttentionBlockUnionCodec(청크드 프리필 블록 선택) →
   DapQPositionAwareEvictionCodec (위치-인식 유사 쿼리 기반 KV 퇴거).
   설계 원칙 자체가 전환된다: 이전 "어텐션 중요도 마스크 → 블록 선택"에서
   "RoPE 회전 위치-인식 유사 쿼리 → 토큰별 중요도 → 예산 기반 퇴거"로.
   DapQ 원논문에서 NIAH KV 예산 3%에서 99.5% 성능 보존 실험 검증.

2. [Cross B+C 신규] DapQSessionSegmentDualReductionPipeline:
   DapQ 위치-인식 원칙을 Activity B 세그먼트 선택에도 동일하게 적용.
   "어떤 세그먼트를 유지할지(B)"와 "유지된 세그먼트 내 어떤 KV를 퇴거할지(C)"를
   동일한 위치-인식 원칙으로 일관성 있게 결정.
   SessionAwareTurnLevelSegmentCache(B-1)와 DapQPositionAwareEvictionCodec(C-3) 결합.

3. [Activity A 보조 신규] PPDAppendFullPrefillClassifier:
   멀티-턴 append-prefill을 decode 노드 로컬로 처리해 KV 전송 우회.
   단, 기존 PPDAppendPrefillRouter(src/scheduler/ppd_append_prefill_router.py)가
   이미 유사 기능을 구현하고 있으므로 차별화된 세션 컨텍스트 해시 분류기로 확장.
   새 파일 src/scheduler/ppd_append_full_prefill_classifier.py 신규 생성.

4. [보존 파일] 이전 사이클 구현 파일
   (block_union_noncontiguous_index.py, compact_attention_block_union_codec.py,
   block_union_bc_pipeline.py, ppd_append_prefill_router.py 등)은 수정하지 않는다.
   기존 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-22: DapQ Position-Aware KV Eviction + Non-Contiguous Segment Dual Reduction

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-22.md`

**최우선 구현 타겟**:
- **C-3 (최우선)**: `DapQPositionAwareEvictionCodec`
  — DapQ(arXiv 2603.11564) 위치-인식 유사 쿼리(position-aware pseudo query) 기반 KV 퇴거.
  위치 임베딩(RoPE)을 현재 디코딩 위치에 맞게 회전시킨 유사 쿼리로 토큰 중요도를 추정해
  KV 예산 비율(budget_ratio)만큼만 KV를 유지하고 나머지를 퇴거.
  NIAH 99.5% 성능 보존, 구현 난이도 low.
- **Cross B+C (2순위)**: `DapQSessionSegmentDualReductionPipeline`
  — B-1 `SessionAwareTurnLevelSegmentCache` + C-3 `DapQPositionAwareEvictionCodec`의
  이중 감소 파이프라인. 동일 위치-인식 원칙으로 세그먼트 선택(B)과 KV 퇴거(C)를 일관성 있게 결정.
- **A (3순위)**: `PPDAppendFullPrefillClassifier`
  — 세션 컨텍스트 해시 기반 append-prefill / full-prefill 분류기. Turn 2+ TTFT 절감.

**해결하려는 문제**:

- **Activity C (DapQ 위치-인식 퇴거)**: 기존 KV 퇴거 방법들(LookaheadKV, SpecAttn 등)이
  프리필 단계 어텐션 패턴 또는 의미 정보로 토큰 중요도를 추정해 실제 디코딩 단계 쿼리와의
  불일치(query drift)가 발생한다. DapQ의 핵심 발견 — "위치 정보(positional information)가
  의미 정보(semantic content)보다 KV 퇴거 결정에 결정적" — 을 적용해 RoPE 기반 위치-인식
  유사 쿼리로 중요도를 추정하면 query drift 없이 정확도를 보존하면서 메모리를 절감한다.

- **Activity B+C (이중 감소)**: 세션 내 turn-level 비연속 세그먼트를 보존하는
  `SessionAwareTurnLevelSegmentCache`와 DapQ 위치-인식 원칙을 결합해,
  "어떤 세그먼트를 유지할지"와 "유지된 세그먼트 내 어떤 KV를 퇴거할지" 모두를
  동일한 위치 근접도 기준으로 결정한다. 설계 일관성 + 히트율 향상 + 메모리 절감 동시 달성.

- **Activity A (PPD 분류기)**: 멀티-턴 P/D 분리 환경에서 append-prefill 요청을
  decode 노드 로컬로 처리해 불필요한 P→D KV 전송을 우회한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling (PPDAppendFullPrefillClassifier, 보조)
- [x] Activity B: Non-Contiguous KV Cache Reuse (SessionAwareTurnLevelSegmentCache)
- [x] Activity C: KV Cache Compression (DapQPositionAwareEvictionCodec)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — WikiText-2 proxy: `attention_output_relative_error(q, k_orig, v_orig, k_evicted, v_evicted) < 0.01` (MANDATORY)
      — budget_ratio=0.30 기준 측정
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 ±1% 이내
      — NIAH proxy: `cosine_similarity_output(...) >= 0.99` (MANDATORY)
      — LongBench 8개 서브태스크 proxy 모두 통과
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C 높음): KV Memory Reduction >= −30%
      — budget_ratio=0.30: 70% 토큰 퇴거 → memory_reduction_ratio >= 0.70 (필수)
      — budget_ratio=0.50: 50% 퇴거 → memory_reduction_ratio >= 0.50
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C 높음): Effective Context Length 동일 메모리 2× 이상
      — KV 퇴거로 확보한 메모리로 더 긴 컨텍스트 수용
- [ ] 목표 5 (evaluation_criteria.md §3 Activity B 높음): 전체 Cache Hit Rate +5%p 이상
      — SessionAwareTurnLevelSegmentCache 세션-인식 LRU로 세션 내 세그먼트 보존 향상
- [ ] 목표 6 (evaluation_criteria.md §3 Activity B 높음): 비연속 세그먼트 히트율 >= 전체 히트의 30%
      — turn-level 세그먼트 비연속 재사용으로 비연속 히트 비율 증가
- [ ] 목표 7 (evaluation_criteria.md §2 Activity A 필수): Scheduling overhead TTFT p50 +5% 이내
      — PPDAppendFullPrefillClassifier O(1) 해시 비교 오버헤드
- [ ] 목표 8 (evaluation_criteria.md §1 처리량 높음): Inference Throughput 베이스라인 +20% 이상
      — DapQ KV 퇴거로 어텐션 계산 토큰 수 감소 + PPD append-prefill 전송 우회 복합 효과
- [ ] 목표 9 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후 accuracy ±1% 이내
      — DapQSessionSegmentDualReductionPipeline 전체 흐름 후 cosine_sim >= 0.99 (MANDATORY)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/dapq_position_aware_eviction_codec.py` | C | DapQPositionAwareEvictionCodec — RoPE 기반 위치-인식 유사 쿼리로 KV 중요도 추정 후 예산 기반 퇴거 |
| `src/cache/session_turn_level_segment_cache.py` | B | SessionAwareTurnLevelSegmentCache — (content_hash, session_id, turn_id) 3-tuple 키, 세션-인식 LRU |
| `src/cache/dapq_session_segment_dual_pipeline.py` | B+C | DapQSessionSegmentDualReductionPipeline — 위치-인식 세그먼트 선택(B) + KV 퇴거(C) 이중 감소 파이프라인 |
| `src/scheduler/ppd_append_full_prefill_classifier.py` | A | PPDAppendFullPrefillClassifier — 세션 컨텍스트 해시 기반 append/full-prefill 분류 + SLO-인식 라우팅 |
| `tests/unit/test_dapq_position_aware_eviction.py` | C | DapQPositionAwareEvictionCodec 단위 테스트 + accuracy 검증 |
| `tests/unit/test_compression_accuracy.py` | C | Accuracy-preservation 검증 (기존 파일 덮어쓰기) — DapQ 기반으로 전환 |
| `tests/unit/test_session_turn_level_segment_cache.py` | B | SessionAwareTurnLevelSegmentCache 단위 테스트 |
| `tests/unit/test_ppd_append_full_prefill_classifier.py` | A | PPDAppendFullPrefillClassifier 단위 테스트 |
| `tests/integration/test_dapq_session_segment_dual_e2e.py` | B+C | DapQSessionSegmentDualReductionPipeline E2E 통합 테스트 |
| `configs/experiments/2026-05-22.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/segmented.py` | 변경 없음 — 이미 `use_block_union` 플래그가 추가됨. `SessionAwareTurnLevelSegmentCache`는 `CacheStore` 직접 상속으로 독립 구현 |

**주의**: `src/cache/base.py`는 변경하지 않는다. 기존 추상 메서드 6개와 선택적 메서드(compression_hook, store_pre_rope, load_with_rope, get_importance_mask)는 불변. 이전 사이클 구현 파일들 모두 수정 금지.

---

## 알고리즘 상세

### DapQPositionAwareEvictionCodec (Activity C)

```python
# src/cache/dapq_position_aware_eviction_codec.py

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class DapQEvictionConfig:
    d_head: int = 128              # KV 헤드 차원 (RoPE 회전 대상 차원)
    n_kv_heads: int = 8            # KV 헤드 수
    n_layers: int = 12             # 모델 레이어 수
    budget_ratio: float = 0.30     # 유지할 KV 비율 (0.30 = 상위 30% 유지)
    high_pressure_threshold: float = 0.80   # KV 풀 점유율 > 이 값이면 공격적 퇴거
    low_pressure_threshold: float = 0.50    # KV 풀 점유율 < 이 값이면 보수적 퇴거
    high_pressure_budget_ratio: float = 0.15  # 공격적 모드: 상위 15% 유지
    low_pressure_budget_ratio: float = 0.50   # 보수적 모드: 상위 50% 유지
    recent_window: int = 32        # 최근 N 토큰은 항상 유지 (query drift 방지)
    use_unit_template: bool = True # True: q_template = 단위 벡터 (의미 정보 최소화)
    max_entries: int = 1000
    seed: int = 42


class DapQPositionAwareEvictionCodec(CacheStore):
    """DapQ (arXiv 2603.11564) Position-Aware Pseudo Query KV Eviction Codec.

    Activity C: KV Cache Compression (accuracy-preserving, training-free).
    CacheStore 인터페이스 완전 구현.

    핵심 알고리즘:
      DapQ의 핵심 발견 — "유사 쿼리 구성에서 위치 정보(RoPE)가 의미 정보보다 결정적" —
      을 적용해 현재 디코딩 위치 pos_decode를 기반으로 위치-인식 유사 쿼리를 생성하고,
      이로부터 KV 토큰별 중요도를 추정해 budget_ratio 상위 KV만 유지한다.

    위치-인식 유사 쿼리 생성:
      q_pseudo = RoPE(pos_decode, q_template)
      q_template: 레이어별 평균 쿼리 벡터 또는 단위 벡터 (use_unit_template=True 시)
      의미 정보를 제거한 단위 벡터가 query drift를 오히려 줄인다 (DapQ 이론적 근거).

    중요도 점수 계산:
      importance[i] = softmax(q_pseudo @ K.T)[i]   for each token i
      배치 처리: q_pseudo [d_head] @ K.T [d_head, seq_len] → scores [seq_len]

    예산 기반 KV 선택:
      top_k = max(recent_window, int(seq_len * effective_budget_ratio))
      selected = topk(importance, top_k).indices (정렬 후 반환)
      최근 recent_window 토큰은 indices에 항상 포함.

    accuracy-preserving 근거:
      (1) DapQ 원논문(2603.11564): NIAH 태스크 KV 예산 3%에서 99.5% 성능 보존 실험 검증.
      (2) RoPE 위치 임베딩이 어텐션 구조적 특성(어떤 위치가 중요한지)을 반영 →
          의미-독립적 위치-인식 쿼리가 실제 디코딩 쿼리를 더 잘 근사.
      (3) recent_window로 최근 토큰 보장 → local coherence 유지.
      (4) budget_ratio=0.30 (보통) / 0.50 (보수적): 대부분 요청에서 accuracy delta < ±1%.

    평가 기준 (evaluation_criteria.md §4):
      - Accuracy 보존 (필수): relative_error < 0.01 (MANDATORY)
      - downstream 태스크 정확도: cosine_sim >= 0.99 (MANDATORY)
      - KV Memory Reduction >= −30% (높음)
    """

    def __init__(self, config: DapQEvictionConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        # key → importance mask (bool tensor [seq_len])
        self._importance_masks: Dict[str, torch.Tensor] = {}
        # 레이어별 q_template 벡터 [n_layers, n_kv_heads, d_head]
        self._q_templates: Optional[torch.Tensor] = None
        # 현재 KV 풀 점유율 (외부에서 업데이트)
        self._pool_utilization: float = 0.0
        self._hits: int = 0
        self._misses: int = 0
        self._total_bytes_original: int = 0
        self._total_bytes_stored: int = 0

    # ------------------------------------------------------------------ #
    # RoPE 유틸리티                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_rope(
        x: torch.Tensor,   # [d_head] or [n_heads, d_head]
        pos: int,
        base: float = 10000.0,
    ) -> torch.Tensor:
        """RoPE 회전 적용.

        Algorithm:
          d = x.shape[-1]
          half_d = d // 2
          theta_i = base^(-2i/d) for i in range(half_d)  → [half_d]
          angle = pos * theta_i                            → [half_d]
          cos_v = cos(angle), sin_v = sin(angle)           → [half_d]
          x_r = x[..., :half_d], x_i = x[..., half_d:]
          result[..., :half_d] = x_r * cos_v - x_i * sin_v
          result[..., half_d:] = x_r * sin_v + x_i * cos_v
          return result
        """
        d = x.shape[-1]
        half_d = d // 2
        i = torch.arange(half_d, dtype=torch.float32, device=x.device)
        theta = base ** (-2.0 * i / d)
        angle = pos * theta
        cos_v = torch.cos(angle)
        sin_v = torch.sin(angle)
        x_float = x.float()
        x_r = x_float[..., :half_d]
        x_i = x_float[..., half_d:]
        out = torch.empty_like(x_float)
        out[..., :half_d] = x_r * cos_v - x_i * sin_v
        out[..., half_d:] = x_r * sin_v + x_i * cos_v
        return out.to(x.dtype)

    # ------------------------------------------------------------------ #
    # 유사 쿼리 생성 및 중요도 계산                                         #
    # ------------------------------------------------------------------ #

    def set_q_templates(self, q_templates: torch.Tensor) -> None:
        """레이어별 평균 쿼리 템플릿 설정 (오프라인 보정 단계에서 호출).

        Args:
            q_templates: [n_layers, n_kv_heads, d_head] float 텐서.
        """
        self._q_templates = q_templates.detach().clone()

    def _get_q_template(self, layer_idx: int = 0, head_idx: int = 0) -> torch.Tensor:
        """레이어·헤드별 쿼리 템플릿 반환.

        use_unit_template=True이거나 템플릿 미설정 시 단위 벡터 반환.
        """
        if self.config.use_unit_template or self._q_templates is None:
            return torch.ones(self.config.d_head, dtype=torch.float32) / (self.config.d_head ** 0.5)
        nl = self._q_templates.shape[0]
        nh = self._q_templates.shape[1]
        li = min(layer_idx, nl - 1)
        hi = min(head_idx, nh - 1)
        return self._q_templates[li, hi].float()

    def compute_importance(
        self,
        K: torch.Tensor,        # [seq_len, d_head] キー 텐서 (단일 헤드 or 평균)
        pos_decode: int,        # 현재 디코딩 위치
        layer_idx: int = 0,
        head_idx: int = 0,
    ) -> torch.Tensor:
        """위치-인식 유사 쿼리로 KV 토큰별 중요도 계산.

        Algorithm:
          q_template = _get_q_template(layer_idx, head_idx)   # [d_head]
          q_pseudo = _apply_rope(q_template, pos_decode)       # [d_head]
          scores = q_pseudo @ K.T                              # [seq_len]
          importance = softmax(scores / sqrt(d_head), dim=0)   # [seq_len]
          return importance

        Returns:
            importance: [seq_len] float 텐서 (합계 = 1.0)
        """
        q_template = self._get_q_template(layer_idx, head_idx)
        q_pseudo = self._apply_rope(q_template, pos_decode)
        scale = self.config.d_head ** 0.5
        K_float = K.float()
        scores = (q_pseudo.to(K_float.device) @ K_float.T) / scale  # [seq_len]
        importance = F.softmax(scores, dim=0)
        return importance

    def select_kv_indices(
        self,
        K: torch.Tensor,        # [seq_len, d_head]
        pos_decode: int,
        layer_idx: int = 0,
        head_idx: int = 0,
    ) -> torch.Tensor:
        """예산 기반 KV 인덱스 선택.

        Algorithm:
          effective_ratio = _get_effective_budget_ratio()
          top_k = max(recent_window, int(seq_len * effective_ratio))
          top_k = min(top_k, seq_len)
          importance = compute_importance(K, pos_decode, ...)
          selected_by_importance = topk(importance, top_k).indices
          # 최근 recent_window 토큰 보장
          recent_indices = arange(max(0, seq_len - recent_window), seq_len)
          selected = union(selected_by_importance, recent_indices)
          return sort(selected)

        Returns:
            selected: [n_selected] int64 인덱스 텐서 (위치 순 정렬)
        """
        seq_len = K.shape[0]
        effective_ratio = self._get_effective_budget_ratio()
        top_k = max(self.config.recent_window, int(seq_len * effective_ratio))
        top_k = min(top_k, seq_len)
        importance = self.compute_importance(K, pos_decode, layer_idx, head_idx)
        selected_indices = importance.topk(top_k).indices
        recent_start = max(0, seq_len - self.config.recent_window)
        recent_indices = torch.arange(recent_start, seq_len, device=K.device)
        all_indices = torch.cat([selected_indices, recent_indices])
        unique_indices = torch.unique(all_indices)
        return unique_indices.sort().values

    def _get_effective_budget_ratio(self) -> float:
        """현재 KV 풀 점유율에 따른 유효 budget_ratio 반환."""
        if self._pool_utilization > self.config.high_pressure_threshold:
            return self.config.high_pressure_budget_ratio
        elif self._pool_utilization < self.config.low_pressure_threshold:
            return self.config.low_pressure_budget_ratio
        return self.config.budget_ratio

    def update_pool_utilization(self, utilization: float) -> None:
        """KV 풀 점유율 업데이트 (외부 스케줄러에서 호출)."""
        self._pool_utilization = max(0.0, min(1.0, utilization))

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """위치-인식 유사 쿼리 기반 KV 퇴거 압축.

        Algorithm:
          K = value  (단순화: value가 K 텐서를 대표, [seq_len, d_head])
          pos_decode = value.shape[0]  (마지막 토큰 위치를 디코딩 위치로 근사)
          selected_indices = select_kv_indices(K, pos_decode)
          mask = zeros(seq_len, dtype=bool)
          mask[selected_indices] = True
          _importance_masks[key] = mask
          result = zeros_like(value)
          result[selected_indices] = value[selected_indices]
          return result

        주의: value가 다차원([seq_len, n_heads, d_head])이면 첫 번째 헤드로 중요도 추정.
        """
        if value.dim() == 1:
            # [d_head] 단일 벡터: 퇴거 불필요
            return value
        seq_len = value.shape[0]
        # 중요도 추정용 K: value가 [seq_len, d_head] 또는 [seq_len, n_heads, d_head]
        if value.dim() == 2:
            K = value  # [seq_len, d_head]
        else:
            K = value[:, 0, :]  # 첫 번째 헤드 사용 [seq_len, d_head]
        # 현재 디코딩 위치: 시퀀스 길이 (다음 생성 위치)
        pos_decode = seq_len
        selected_indices = self.select_kv_indices(K, pos_decode)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=value.device)
        mask[selected_indices] = True
        self._importance_masks[key] = mask.cpu()
        result = torch.zeros_like(value)
        result[selected_indices] = value[selected_indices]
        return result

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """저장된 bool 중요도 마스크 반환 (base.py get_importance_mask 구현).

        Returns: [seq_len] bool 텐서 (중요 위치 True) 또는 None.
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
        if not self._store:
            return 0
        key, v = self._store.popitem(last=False)
        self._importance_masks.pop(key, None)
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
        self._importance_masks.clear()
```

---

### SessionAwareTurnLevelSegmentCache (Activity B)

```python
# src/cache/session_turn_level_segment_cache.py

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore


@dataclass
class TurnSegmentEntry:
    turn_id: int
    segment_id: str          # (content_hash, session_id, turn_id) 기반 키
    kv_pointer: str          # 실제 KV 저장 키 (_store의 키)
    token_range: Tuple[int, int]   # (start_pos, end_pos)
    center_position: float   # (start_pos + end_pos) / 2
    timestamp: float         # 저장 시각 (time.monotonic())
    ttl: Optional[float]     # None이면 TTL 없음


@dataclass
class SessionTurnLevelConfig:
    chunk_size: int = 128          # 청크 단위 (토큰 수)
    max_entries: int = 1000        # 전체 최대 엔트리 수
    max_turns_per_session: int = 10  # 세션 내 보존 최대 턴 수
    session_lru_penalty: float = 0.5  # 세션 내 세그먼트 퇴거 우선순위 패널티
                                     # (낮을수록 세션 내 세그먼트가 더 오래 보존됨)
    seed: int = 42


class SessionAwareTurnLevelSegmentCache(CacheStore):
    """세션-인식 턴-레벨 비연속 세그먼트 캐시 (Activity B).

    CacheStore 인터페이스 완전 구현.

    핵심 자료구조:
      세그먼트 키: (content_hash, session_id, turn_id) 3-tuple의 해시
      TurnSegmentIndex: session_id → List[TurnSegmentEntry]
      session_priority_lru: 세션 내 세그먼트는 cross-session 세그먼트보다
                            퇴거 우선순위가 낮음 (더 오래 보존).

    위치-인식 재사용 점수:
      position_reuse_score(segment, current_pos) =
        exp(-|segment.center_position - current_pos| / decay_factor)
      DapQ 위치-인식 원칙: 현재 디코딩 위치에 가까운 세그먼트가 재사용 가능성 높음.

    평가 기준 (evaluation_criteria.md §3):
      - 전체 Cache Hit Rate +5%p 이상 (높음)
      - 비연속 세그먼트 히트율 >= 전체 히트의 30% (높음)
      - KV Memory Footprint 베이스라인 +20% 이내 (높음)
    """

    def __init__(self, config: SessionTurnLevelConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        # 실제 KV 저장소: kv_key → torch.Tensor
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        # 세션 인덱스: session_id → List[TurnSegmentEntry]
        self._session_index: Dict[str, List[TurnSegmentEntry]] = {}
        # 세그먼트 → 세션 역매핑 (빠른 세션 조회)
        self._key_to_session: Dict[str, str] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0

    @staticmethod
    def _make_segment_key(
        content_hash: str,
        session_id: str,
        turn_id: int,
        layer_idx: int = 0,
    ) -> str:
        """(content_hash, session_id, turn_id, layer_idx) 3-tuple 키 생성."""
        raw = f"{content_hash}|{session_id}|{turn_id}|{layer_idx}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _content_hash(token_ids: List[int], chunk_idx: int, chunk_size: int) -> str:
        start = chunk_idx * chunk_size
        chunk = token_ids[start:start + chunk_size]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        return hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------------ #
    # 세션-턴 API                                                          #
    # ------------------------------------------------------------------ #

    def put_turn_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        session_id: str,
        turn_id: int,
        layer_idx: int = 0,
        ttl: Optional[float] = None,
    ) -> str:
        """턴-레벨 세그먼트 저장.

        Algorithm:
          content_hash = _content_hash(token_ids, chunk_idx, chunk_size)
          key = _make_segment_key(content_hash, session_id, turn_id, layer_idx)
          start_pos = chunk_idx * chunk_size
          end_pos = min(start_pos + chunk_size, len(token_ids))
          entry = TurnSegmentEntry(turn_id, key, key, (start_pos, end_pos),
                                   center=(start_pos+end_pos)/2, ...)
          _session_index[session_id].append(entry)
          _key_to_session[key] = session_id
          put(key, kv)
          return key
        """
        import time
        content_hash = self._content_hash(token_ids, chunk_idx, self.config.chunk_size)
        key = self._make_segment_key(content_hash, session_id, turn_id, layer_idx)
        start_pos = chunk_idx * self.config.chunk_size
        end_pos = min(start_pos + self.config.chunk_size, len(token_ids))
        entry = TurnSegmentEntry(
            turn_id=turn_id,
            segment_id=key,
            kv_pointer=key,
            token_range=(start_pos, end_pos),
            center_position=(start_pos + end_pos) / 2.0,
            timestamp=time.monotonic(),
            ttl=ttl,
        )
        if session_id not in self._session_index:
            self._session_index[session_id] = []
        self._session_index[session_id].append(entry)
        self._key_to_session[key] = session_id
        self.put(key, kv)
        return key

    def get_session_segments(
        self,
        session_id: str,
        turn_range: Optional[Tuple[int, int]] = None,
    ) -> List[TurnSegmentEntry]:
        """세션 내 저장된 세그먼트 목록 반환.

        Args:
            session_id: 조회할 세션 ID
            turn_range: (min_turn, max_turn_inclusive) 또는 None (전체)

        Returns:
            해당 세션의 TurnSegmentEntry 리스트 (존재하는 KV만 포함)
        """
        entries = self._session_index.get(session_id, [])
        if turn_range is not None:
            lo, hi = turn_range
            entries = [e for e in entries if lo <= e.turn_id <= hi]
        # 실제로 _store에 존재하는 엔트리만 반환
        return [e for e in entries if e.kv_pointer in self._store]

    def position_reuse_score(
        self,
        entry: TurnSegmentEntry,
        current_decode_pos: float,
        decay_factor: float = 512.0,
    ) -> float:
        """DapQ 위치-인식 세그먼트 재사용 가능성 점수.

        score = exp(-|center_position - current_decode_pos| / decay_factor)
        현재 디코딩 위치에 가까울수록 점수가 높다.
        """
        import math
        dist = abs(entry.center_position - current_decode_pos)
        return math.exp(-dist / decay_factor)

    def get_top_segments_by_position(
        self,
        session_id: str,
        current_decode_pos: float,
        keep_ratio: float = 0.50,
        decay_factor: float = 512.0,
    ) -> List[TurnSegmentEntry]:
        """위치-인식 점수 상위 keep_ratio 세그먼트 반환.

        DapQSessionSegmentDualReductionPipeline Step 2에서 사용.
        """
        entries = self.get_session_segments(session_id)
        if not entries:
            return []
        scored = [(e, self.position_reuse_score(e, current_decode_pos, decay_factor))
                  for e in entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(scored) * keep_ratio))
        return [e for e, _ in scored[:k]]

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        self._store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            # 비연속 히트 감지: 세션 내 이전 턴 세그먼트를 참조
            session_id = self._key_to_session.get(key)
            if session_id:
                entries = self._session_index.get(session_id, [])
                hit_entry = next((e for e in entries if e.kv_pointer == key), None)
                if hit_entry and hit_entry.turn_id > 0:
                    self._noncontiguous_hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        """세션-인식 LRU: 세션 내 세그먼트를 더 오래 보존.

        Algorithm:
          candidates = list(_store.keys())
          for each key in candidates:
            session_id = _key_to_session.get(key)
            if session_id: score = session_lru_penalty  (낮은 점수 = 늦게 퇴거)
            else: score = 1.0
          evict_key = argmax(score) 중 가장 오래된(LRU first) 항목
          (동점 시 OrderedDict 순서로 LRU 선택)
        """
        if not self._store:
            return 0
        # cross-session 우선 퇴거: session에 속하지 않은 키 먼저
        for key in list(self._store.keys()):
            if key not in self._key_to_session:
                v = self._store.pop(key)
                return v.nbytes
        # 모두 세션 내 세그먼트면 LRU (OrderedDict 첫 번째)
        key, v = self._store.popitem(last=False)
        session_id = self._key_to_session.pop(key, None)
        if session_id and session_id in self._session_index:
            self._session_index[session_id] = [
                e for e in self._session_index[session_id]
                if e.kv_pointer != key
            ]
        return v.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def noncontiguous_hit_rate(self) -> float:
        return self._noncontiguous_hits / max(1, self._hits)

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
```

---

### DapQSessionSegmentDualReductionPipeline (Cross B+C)

```python
# src/cache/dapq_session_segment_dual_pipeline.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.session_turn_level_segment_cache import (
    SessionAwareTurnLevelSegmentCache,
    SessionTurnLevelConfig,
    TurnSegmentEntry,
)
from src.cache.dapq_position_aware_eviction_codec import (
    DapQPositionAwareEvictionCodec,
    DapQEvictionConfig,
)


@dataclass
class DualReductionPipelineConfig:
    b_config: Optional[SessionTurnLevelConfig] = None
    c_config: Optional[DapQEvictionConfig] = None
    segment_keep_ratio: float = 0.50    # Step 2: 상위 50% 세그먼트 선택
    kv_budget_ratio: float = 0.30       # Step 3: 선택 세그먼트 내 상위 30% KV 유지
    decay_factor: float = 512.0         # 위치-인식 점수 decay 파라미터
    seed: int = 42


class DapQSessionSegmentDualReductionPipeline(CacheStore):
    """DapQ 위치-인식 퇴거 + 세션 비연속 세그먼트 재사용 이중 감소 파이프라인 (B+C).

    CacheStore 인터페이스 완전 구현.

    통합 처리 흐름 (5단계):
      Step 1 (세션 세그먼트 조회, B):
        segment_cache.get_session_segments(session_id) → 이전 턴 세그먼트 목록.

      Step 2 (위치-인식 세그먼트 중요도 평가, C 원칙):
        각 세그먼트에 position_reuse_score(segment, current_decode_pos) 계산.
        상위 segment_keep_ratio 세그먼트 선택.

      Step 3 (선택 세그먼트 내 KV 중요도 평가, C):
        선택된 세그먼트의 KV에 DapQ 위치-인식 유사 쿼리로 importance 계산.
        상위 kv_budget_ratio KV 유지.

      Step 4 (비선택 세그먼트 + 비선택 KV 처리):
        비선택 KV: 즉시 퇴거 (압축된 값으로 덮어쓰기).

      Step 5 (최종 KV 집합 반환):
        선택된 세그먼트의 선택된 KV를 리스트로 반환.

    설계 일관성:
      세그먼트 선택(B)과 KV 선택(C) 모두 동일한 위치-인식 원칙 적용.
      → 세그먼트 내 중요 KV와 세그먼트 간 선택이 일관된 중요도 기준을 공유.

    평가 기준 (evaluation_criteria.md §5):
      - 복합 Throughput: 단일 Activity 대비 +5% 이상 (높음)
      - 복합 Memory Reduction: 단일 Activity 대비 −10% 이상 (높음)
      - Accuracy 보존 (C 포함): cosine_sim >= 0.99 (MANDATORY)
    """

    def __init__(self, config: DualReductionPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        b_cfg = config.b_config or SessionTurnLevelConfig(seed=config.seed)
        c_cfg = config.c_config or DapQEvictionConfig(
            budget_ratio=config.kv_budget_ratio,
            seed=config.seed,
        )
        self.segment_cache = SessionAwareTurnLevelSegmentCache(b_cfg)
        self.eviction_codec = DapQPositionAwareEvictionCodec(c_cfg)

    # ------------------------------------------------------------------ #
    # 이중 감소 파이프라인 API                                              #
    # ------------------------------------------------------------------ #

    def process_session(
        self,
        session_id: str,
        current_decode_pos: float,
        pos_decode_int: Optional[int] = None,
    ) -> List[Tuple[TurnSegmentEntry, torch.Tensor]]:
        """B+C 이중 감소 파이프라인 실행.

        Returns:
            선택된 (TurnSegmentEntry, evicted_kv_tensor) 리스트.
            빈 리스트 반환 시 세션 캐시 미스.
        """
        # Step 1: 세션 세그먼트 조회
        entries = self.segment_cache.get_session_segments(session_id)
        if not entries:
            return []

        # Step 2: 위치-인식 세그먼트 선택
        scored = [
            (e, self.segment_cache.position_reuse_score(
                e, current_decode_pos, self.config.decay_factor))
            for e in entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        k_seg = max(1, int(len(scored) * self.config.segment_keep_ratio))
        selected_entries = [e for e, _ in scored[:k_seg]]

        # Step 3: 선택된 세그먼트 내 KV에 DapQ 퇴거 적용
        pos = pos_decode_int if pos_decode_int is not None else int(current_decode_pos)
        result = []
        for entry in selected_entries:
            kv = self.segment_cache.get(entry.kv_pointer)
            if kv is None:
                continue
            compressed_kv = self.eviction_codec.compression_hook(entry.kv_pointer, kv)
            result.append((entry, compressed_kv))

        return result

    def dual_reduction_ratio(self) -> float:
        """B+C 이중 감소율 추정.

        = segment_keep_ratio × kv_budget_ratio 의 역수 기반 감소 추정.
        실제 측정은 memory_bytes() 비교로 수행.
        """
        return 1.0 - (self.config.segment_keep_ratio * self.config.kv_budget_ratio)

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스 (segment_cache에 위임)                         #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        self.segment_cache.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.segment_cache.get(key)

    def evict(self) -> int:
        return self.segment_cache.evict()

    def hit_rate(self) -> float:
        return self.segment_cache.hit_rate()

    def memory_bytes(self) -> int:
        return self.segment_cache.memory_bytes()

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return self.eviction_codec.compression_hook(key, value)

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        return self.eviction_codec.get_importance_mask(key)

    def reset_stats(self) -> None:
        self.segment_cache.reset_stats()
        self.eviction_codec.reset_stats()

    def metrics_summary(self) -> Dict:
        return {
            "session_cache_hit_rate": self.segment_cache.hit_rate(),
            "session_noncontiguous_hit_rate": self.segment_cache.noncontiguous_hit_rate(),
            "eviction_memory_reduction_ratio": self.eviction_codec.memory_reduction_ratio(),
            "dual_reduction_estimate": self.dual_reduction_ratio(),
            "total_memory_bytes": self.memory_bytes(),
        }
```

---

### PPDAppendFullPrefillClassifier (Activity A)

```python
# src/scheduler/ppd_append_full_prefill_classifier.py

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class PrefillTypeDecision:
    request_id: str
    session_id: str
    turn: int
    prefill_type: str    # "append" | "full"
    new_token_ratio: float    # 신규 토큰 비율
    routed_to: str            # "D_node" | "P_node"
    classifier_overhead_us: float   # 분류 오버헤드 (마이크로초)


@dataclass
class SessionContextEntry:
    last_context_hash: str
    last_total_tokens: int
    turn_count: int
    last_accessed: float


@dataclass
class PPDClassifierConfig:
    append_threshold: float = 0.15    # new_token_ratio < 이 값이면 append-prefill
    slo_headroom_threshold_ms: float = 30.0   # SLO 여유 < 이 값이면 P 노드 오프로드
    session_ttl_seconds: float = 3600.0       # 세션 만료 TTL
    seed: int = 42


class PPDAppendFullPrefillClassifier:
    """PPD (arXiv 2603.13358) 기반 append/full-prefill 분류기 + SLO-인식 라우팅.

    Activity A: KV Cache-aware Scheduling.
    스케줄링 결정 단위: 요청(request) 단위.
    캐시 상태 접근: PrefillTypeRegistry (세션 ID → 마지막 컨텍스트 해시) O(1) 딕셔너리 룩업.

    기존 PPDAppendPrefillRouter(ppd_append_prefill_router.py)와의 차이:
      PPDAppendPrefillRouter: TriangleInequalitySegmentIndex를 통한 hit_probability 추정 (O(log N)).
      PPDAppendFullPrefillClassifier: 세션 컨텍스트 해시 직접 비교로 append/full 분류 (O(1)).
                                      new_token_ratio를 명시적으로 계산해 분류 근거 제공.
                                      SLO-인식 D→P 강제 전환 로직 포함.

    분류 로직:
      append-prefill 조건:
        1. turn_count >= 2 (첫 번째 턴 이후)
        2. new_token_ratio = (total_tokens - last_total_tokens) / total_tokens
           new_token_ratio <= append_threshold (기본 0.15: 15% 이하가 신규 토큰)
        → 이전 KV를 D 노드에서 재사용하고 신규 토큰만 처리.

      full-prefill 조건:
        1. turn_count == 1 (첫 번째 턴, KV 캐시 cold start)
        2. new_token_ratio > append_threshold
        3. SLO headroom < slo_headroom_threshold_ms (D 노드 부하 높을 때 P 노드 오프로드)
        → 전체 프리필을 P 노드에서 처리.

    평가 기준 (evaluation_criteria.md §2):
      - Scheduling overhead TTFT p50 +5% 이내 (필수)
      - 캐시 히트율 향상: 스케줄링 미적용 대비 +10%p 이상 (높음)
    """

    def __init__(self, config: PPDClassifierConfig) -> None:
        self.config = config
        # PrefillTypeRegistry: session_id → SessionContextEntry
        self._registry: Dict[str, SessionContextEntry] = {}

    @staticmethod
    def _context_hash(token_ids: List[int]) -> str:
        """토큰 ID 시퀀스 해시 (컨텍스트 동일성 비교용)."""
        if not token_ids:
            return ""
        raw = b"".join(t.to_bytes(4, "little") for t in token_ids[:512])  # 최대 512 토큰
        return hashlib.sha256(raw).hexdigest()[:16]

    def classify(
        self,
        request_id: str,
        session_id: str,
        token_ids: List[int],
        remaining_slo_ms: Optional[float] = None,
    ) -> PrefillTypeDecision:
        """요청을 append-prefill 또는 full-prefill로 분류해 라우팅 결정.

        Algorithm:
          t_start = time.monotonic()
          entry = _registry.get(session_id)  # O(1)
          total_tokens = len(token_ids)
          ctx_hash = _context_hash(token_ids)

          if entry is None or entry.turn_count == 0:
            # 첫 번째 턴: full-prefill
            prefill_type = "full"
          else:
            new_tokens = total_tokens - entry.last_total_tokens
            new_token_ratio = max(0.0, new_tokens / max(1, total_tokens))
            if new_token_ratio <= append_threshold:
              # SLO 여유 확인
              if remaining_slo_ms is not None and remaining_slo_ms < slo_headroom_threshold_ms:
                prefill_type = "full"  # SLO 압박 시 P 노드 오프로드
              else:
                prefill_type = "append"
            else:
              prefill_type = "full"

          # 레지스트리 업데이트
          _registry[session_id] = SessionContextEntry(
            last_context_hash=ctx_hash,
            last_total_tokens=total_tokens,
            turn_count=(entry.turn_count if entry else 0) + 1,
            last_accessed=time.monotonic(),
          )
          overhead_us = (time.monotonic() - t_start) * 1e6
          routed_to = "D_node" if prefill_type == "append" else "P_node"
          return PrefillTypeDecision(...)
        """
        t_start = time.monotonic()
        total_tokens = len(token_ids)
        ctx_hash = self._context_hash(token_ids)
        entry = self._registry.get(session_id)

        if entry is None or entry.turn_count == 0:
            prefill_type = "full"
            new_token_ratio = 1.0
        else:
            new_tokens = max(0, total_tokens - entry.last_total_tokens)
            new_token_ratio = new_tokens / max(1, total_tokens)
            if new_token_ratio <= self.config.append_threshold:
                if (remaining_slo_ms is not None
                        and remaining_slo_ms < self.config.slo_headroom_threshold_ms):
                    prefill_type = "full"
                else:
                    prefill_type = "append"
            else:
                prefill_type = "full"

        turn = (entry.turn_count if entry else 0) + 1
        self._registry[session_id] = SessionContextEntry(
            last_context_hash=ctx_hash,
            last_total_tokens=total_tokens,
            turn_count=turn,
            last_accessed=time.monotonic(),
        )
        overhead_us = (time.monotonic() - t_start) * 1e6
        routed_to = "D_node" if prefill_type == "append" else "P_node"
        return PrefillTypeDecision(
            request_id=request_id,
            session_id=session_id,
            turn=turn,
            prefill_type=prefill_type,
            new_token_ratio=new_token_ratio,
            routed_to=routed_to,
            classifier_overhead_us=overhead_us,
        )

    def expire_sessions(self) -> int:
        """TTL 만료 세션 제거. 만료된 세션 수 반환."""
        now = time.monotonic()
        expired = [
            sid for sid, e in self._registry.items()
            if now - e.last_accessed > self.config.session_ttl_seconds
        ]
        for sid in expired:
            del self._registry[sid]
        return len(expired)

    def reset_session(self, session_id: str) -> None:
        """세션 레지스트리에서 세션 제거."""
        self._registry.pop(session_id, None)
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(`DapQPositionAwareEvictionCodec`)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 proxy — `src/metrics/perplexity.py`의 `attention_output_relative_error()`로
  synthetic float32 KV 텐서 활용 (실제 WikiText-2 없을 시 `torch.randn`으로 생성).
- **측정 설정**: n_heads=8, d_head=128, 시퀀스 길이=512, seq_len=256.
- **측정 방법**:
  ```
  # K_orig: 원본 KV [seq_len, d_head]
  # K_evicted: DapQ 퇴거 후 KV (비선택 위치 = 0으로 패딩)
  relative_error = attention_output_relative_error(q, K_orig, V_orig, K_evicted, V_evicted)
  허용 오차: relative_error < 0.01 (1%) — MANDATORY (evaluation_criteria.md §4 필수)
  ```
- **budget_ratio별 측정 (5개 시나리오)**:
  - budget_ratio=0.50: 50% KV 유지, relative_error 측정
  - budget_ratio=0.30: 30% KV 유지, relative_error 측정 [기본값 MANDATORY]
  - budget_ratio=0.15: 15% KV 유지 (공격적), relative_error 측정
  - budget_ratio=0.03: 3% KV 유지 (DapQ 논문 극단 설정), relative_error 측정
  - budget_ratio=1.00: 100% KV 유지 (기준 검증, relative_error ≈ 0.0)
- **예상 결과**: budget_ratio=0.30 기준 relative_error < 1% (DapQ 논문 근거: NIAH 99.5% 성능 보존).

### 태스크 정확도 측정

- **벤치마크 1 — NIAH proxy**: 
  - synthetic 시퀀스에서 "needle" 위치의 KV가 DapQ 퇴거 후에도 보존되는지 확인.
  - `needle_position_in_selected(selected_indices, needle_pos) → bool`
  - 32K/64K/128K 컨텍스트 길이 시뮬레이션: seq_len = 256/512/1024 (프록시 스케일).
  - 허용 기준: 각 컨텍스트 길이에서 needle 보존율 >= 99% (NIAH 99.5% 성능 보존 대응).

- **벤치마크 2 — LongBench proxy**:
  - 8개 독립 synthetic 시퀀스에 대해 `cosine_similarity_output(q, K_orig, V_orig, K_evicted, V_evicted)` 계산.
  - 허용 기준: 8개 모두 cosine_sim >= 0.99 — MANDATORY (evaluation_criteria.md §4 필수).

- **대조 실험 — 위치-인식 쿼리 vs. 의미 쿼리**:
  - `use_unit_template=True` (위치-인식, DapQ 원칙)
  - `use_unit_template=False` (평균 쿼리 벡터 = 의미 정보 포함)
  - 동일 seq_len, budget_ratio=0.30에서 두 설정의 relative_error, cosine_sim 비교.
  - 기대: 위치-인식 쿼리가 의미 쿼리 대비 accuracy 보존 측면에서 동등 이상.

### KV 메모리 감소율 검증

- budget_ratio=0.50 → memory_reduction_ratio >= 0.40 (50% 퇴거, but recent_window 보정)
- budget_ratio=0.30 → memory_reduction_ratio >= 0.60 [MANDATORY: -30% 기준 충족]
- budget_ratio=0.15 → memory_reduction_ratio >= 0.75

### Fail 기준

**budget_ratio=0.30 기준 relative_error > 1% → 테스트 실패 (evaluation_criteria.md §4 필수 항목 — 무조건 전체 Fail)**

**cosine_sim < 0.99 (LongBench proxy 8개 중 1개라도) → 테스트 실패 (MANDATORY)**

**DapQSessionSegmentDualReductionPipeline E2E 후 cosine_sim < 0.99 → 테스트 실패 (§5 MANDATORY)**

### 검증 테스트 파일

`tests/unit/test_compression_accuracy.py` (기존 파일 덮어쓰기)

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-22.yaml
experiment:
  date: "2026-05-22"
  activity: "A+B+C"
  description: >
    C-3 DapQPositionAwareEvictionCodec (위치-인식 유사 쿼리 기반 KV 퇴거) +
    B-1 SessionAwareTurnLevelSegmentCache (세션-인식 턴-레벨 비연속 세그먼트 캐시) +
    Cross-2 DapQSessionSegmentDualReductionPipeline (B+C 이중 감소 파이프라인) +
    A-1 PPDAppendFullPrefillClassifier (append/full-prefill 분류 라우터, 보조).
    DapQ(arXiv 2603.11564) 위치-인식 원칙 기반.
  cache_type: dapq_session_segment_dual_pipeline
  compression_method: dapq_position_aware_eviction
  scheduler_type: ppd_append_full_prefill_classifier

dapq_eviction_codec:  # C-3
  d_head: 128
  n_kv_heads: 8
  n_layers: 12
  budget_ratio: 0.30            # 기본값: 상위 30% KV 유지
  high_pressure_threshold: 0.80
  low_pressure_threshold: 0.50
  high_pressure_budget_ratio: 0.15
  low_pressure_budget_ratio: 0.50
  recent_window: 32             # 최근 32 토큰 항상 유지
  use_unit_template: true       # 의미 정보 최소화 (DapQ 원칙)
  max_entries: 1000
  seed: 42

session_turn_level_segment_cache:  # B-1
  chunk_size: 128
  max_entries: 1000
  max_turns_per_session: 10
  session_lru_penalty: 0.5
  seed: 42

dual_reduction_pipeline:  # Cross-2
  segment_keep_ratio: 0.50      # Step 2: 상위 50% 세그먼트 선택
  kv_budget_ratio: 0.30         # Step 3: 선택 세그먼트 내 상위 30% KV 유지
  decay_factor: 512.0

ppd_classifier:  # A-1
  append_threshold: 0.15        # new_token_ratio < 0.15이면 append-prefill
  slo_headroom_threshold_ms: 30.0
  session_ttl_seconds: 3600.0
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    dataset_proxy: "wikitext2_synthetic"
    task_accuracy_proxy: "niah_cosine_similarity"
    relative_error_max: 0.01        # ±1% (evaluation_criteria.md §4 MANDATORY)
    cosine_similarity_min: 0.99     # (evaluation_criteria.md §4 MANDATORY)
    kl_divergence_max: 0.015        # 보조 지표
    budget_ratios_to_test: [1.00, 0.50, 0.30, 0.15, 0.03]
    niah_context_lengths: [256, 512, 1024]  # proxy: 32K/64K/128K 스케일
    longbench_subtask_count: 8
    ablation_unit_vs_semantic_template: true  # 위치-인식 vs. 의미 쿼리 대조
  activity_b:
    cache_hit_rate_improvement_min_pct: 5.0   # +5%p (§3 높음)
    noncontiguous_hit_rate_min_pct: 30.0      # 전체 히트의 30% 이상 (§3 높음)
    memory_footprint_max_increase_pct: 20.0   # +20% 이내 (§3 높음)
  activity_a:
    scheduling_overhead_ttft_p50_max_pct: 5.0  # +5% 이내 (§2 필수)
    cache_hit_rate_improvement_min_pct: 10.0   # +10%p 이상 (§2 높음)
  activity_c:
    memory_reduction_min: 0.30      # -30% 이상 (§4 높음)
    effective_context_multiplier: 2.0
    compression_overhead_ttft_max_pct: 10.0
  cross_bc:
    throughput_min_improvement_vs_solo: 5.0   # 단일 Activity 대비 +5% (§5 높음)
    memory_min_improvement_vs_solo: 10.0      # 단일 Activity 대비 -10% (§5 높음)
    accuracy_cosine_min: 0.99                 # C 포함 (§5 MANDATORY)
    comparison_methods: ["solo_b1", "solo_c3", "cross_bc"]
  throughput:
    target_improvement_pct: 20      # 베이스라인 대비 +20% (§1 장기 목표)

seed: 42
results_dir: "results/2026-05-22"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_dapq_position_aware_eviction.py` — DapQPositionAwareEvictionCodec 단위 테스트
- [ ] `tests/unit/test_compression_accuracy.py` — Accuracy-preservation 검증 (기존 파일 덮어쓰기)
- [ ] `tests/unit/test_session_turn_level_segment_cache.py` — SessionAwareTurnLevelSegmentCache 단위 테스트
- [ ] `tests/unit/test_ppd_append_full_prefill_classifier.py` — PPDAppendFullPrefillClassifier 단위 테스트
- [ ] `tests/integration/test_dapq_session_segment_dual_e2e.py` — DapQSessionSegmentDualReductionPipeline E2E 통합 테스트

### 단위 테스트 명세 — test_dapq_position_aware_eviction.py

```
test_dapq_rope_apply_rotation_changes_vector:
    _apply_rope(x, pos=10) != x 확인 (RoPE 회전 적용 검증)

test_dapq_rope_apply_pos_zero_identity:
    _apply_rope(x, pos=0) ≈ x 확인 (pos=0이면 회전 없음)

test_dapq_compute_importance_sums_to_one:
    compute_importance(K, pos_decode) → softmax 합계 ≈ 1.0

test_dapq_compute_importance_shape:
    K: [seq_len, d_head] → importance: [seq_len]

test_dapq_select_kv_indices_count:
    seq_len=100, budget_ratio=0.30, recent_window=10 →
    len(selected_indices) >= max(10, 30) = 30

test_dapq_select_kv_indices_recent_always_included:
    최근 recent_window 토큰 인덱스가 selected_indices에 포함되는지 확인

test_dapq_select_kv_indices_sorted:
    select_kv_indices() 반환값 오름차순 정렬 확인

test_dapq_compression_hook_zeros_unselected:
    compression_hook 후 비선택 위치 값 = 0.0 확인

test_dapq_compression_hook_preserves_selected:
    선택된 위치의 값은 원본 유지 확인

test_dapq_compression_hook_stores_importance_mask:
    put() 후 get_importance_mask(key) → bool tensor [seq_len] 반환

test_dapq_memory_reduction_ratio_above_60pct:
    budget_ratio=0.30, seq_len=100, recent_window=10 →
    memory_reduction_ratio() >= 0.60 (30% + recent 보정)

test_dapq_high_pressure_uses_aggressive_budget:
    update_pool_utilization(0.90) → _get_effective_budget_ratio() == high_pressure_budget_ratio

test_dapq_low_pressure_uses_conservative_budget:
    update_pool_utilization(0.30) → _get_effective_budget_ratio() == low_pressure_budget_ratio

test_dapq_unit_template_vs_nonunit_same_dtype:
    use_unit_template=True/False 모두 동일 shape [d_head] 반환

test_dapq_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_dapq_seed_reproducibility:
    동일 seed + 동일 입력 → 동일 selected_indices

test_dapq_evict_lru_oldest_first:
    max_entries=2, 3번 put → 첫 번째 항목 퇴거 확인

test_dapq_hit_rate_tracking:
    put 2개 후 get 1회 히트 + 1회 미스 → hit_rate() == 0.5
```

### 단위 테스트 명세 — test_compression_accuracy.py (기존 파일 덮어쓰기)

```
test_dapq_full_budget_zero_relative_error:
    budget_ratio=1.00 → relative_error ≈ 0.0 (기준 검증)

test_dapq_budget_50pct_relative_error_below_1pct:
    budget_ratio=0.50 → relative_error < 0.01 (MANDATORY)

test_dapq_budget_30pct_relative_error_below_1pct:
    budget_ratio=0.30 (기본값) → relative_error < 0.01 (MANDATORY)

test_dapq_budget_30pct_cosine_similarity_above_099:
    budget_ratio=0.30 → cosine_sim >= 0.99 (MANDATORY)

test_dapq_budget_15pct_relative_error_below_1pct:
    budget_ratio=0.15 (공격적) → relative_error < 0.01

test_dapq_niah_proxy_needle_preserved_budget_30pct:
    seq_len=256, needle_pos=128, budget_ratio=0.30 →
    needle_pos가 selected_indices에 포함됨 (NIAH proxy)

test_dapq_niah_proxy_context_lengths:
    seq_len=[256, 512, 1024], budget_ratio=0.30 →
    각각에서 cosine_sim >= 0.99 (32K/64K/128K proxy)

test_dapq_longbench_8subtask_proxy:
    8개 독립 synthetic 시퀀스 모두 cosine_sim >= 0.99 (MANDATORY)

test_dapq_position_query_vs_semantic_query_accuracy:
    use_unit_template=True (위치-인식) vs. False (의미 포함),
    budget_ratio=0.30에서 두 설정의 relative_error 비교 기록.
    위치-인식 쿼리가 의미 쿼리 대비 동등 이상 accuracy 보존 확인.

test_dapq_kl_divergence_below_threshold:
    budget_ratio=0.30 → KL divergence < 0.015 (보조 지표)

test_dapq_memory_reduction_30pct_budget_above_60pct:
    budget_ratio=0.30 → memory_reduction_ratio() >= 0.60 (§4 높음: -30% 이상 충족)

test_dapq_dual_pipeline_accuracy_preserved:
    DapQSessionSegmentDualReductionPipeline: put → process_session →
    cosine_sim >= 0.99 (§5 MANDATORY, C 포함 크로스 조합)

test_dapq_dual_pipeline_dual_reduction_ratio:
    dual_reduction_ratio() > 0 (B+C 이중 감소 확인)
```

### 단위 테스트 명세 — test_session_turn_level_segment_cache.py

```
test_session_cache_put_get_basic:
    put → get 왕복 기본 동작

test_session_cache_put_turn_segment_creates_entry:
    put_turn_segment() 후 get_session_segments(session_id) 비어 있지 않음

test_session_cache_3tuple_key_unique_per_session_turn:
    (content_hash, session_A, turn_0) != (content_hash, session_B, turn_0) 키 확인

test_session_cache_get_session_segments_turn_filter:
    turn_range=(0, 1) → turn_id 0, 1만 반환

test_session_cache_position_reuse_score_closer_higher:
    center_position=100, current_pos=120 → score > center_position=200, current_pos=120

test_session_cache_get_top_segments_by_position_count:
    4개 세그먼트, keep_ratio=0.50 → 2개 반환

test_session_cache_session_lru_penalty_preserves_session_entries:
    max_entries=2: cross-session 1개 + 세션 내 2개 → 퇴거 시 cross-session 먼저 퇴거

test_session_cache_noncontiguous_hit_tracking:
    turn_id > 0인 get() 호출 시 noncontiguous_hits 증가 확인

test_session_cache_noncontiguous_hit_rate:
    noncontiguous_hit_rate() = noncontiguous_hits / max(1, hits)

test_session_cache_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_session_cache_evict_cross_session_first:
    cross-session 키 존재 시 evict()가 cross-session 키 먼저 제거

test_session_cache_hit_rate_tracking:
    put 2개 후 get 1회 히트 + 1회 미스 → hit_rate() == 0.5
```

### 단위 테스트 명세 — test_ppd_append_full_prefill_classifier.py

```
test_ppd_classifier_turn1_always_full_prefill:
    session의 첫 번째 요청 → prefill_type="full", routed_to="P_node"

test_ppd_classifier_turn2_small_new_tokens_append:
    turn 2, new_token_ratio < append_threshold → prefill_type="append", routed_to="D_node"

test_ppd_classifier_turn2_large_new_tokens_full:
    turn 2, new_token_ratio > append_threshold → prefill_type="full"

test_ppd_classifier_slo_pressure_forces_full:
    append 조건 + remaining_slo_ms < slo_headroom_threshold_ms → prefill_type="full"

test_ppd_classifier_overhead_below_1ms:
    classify() 오버헤드 < 1000μs (TTFT +5% 이내 준수 위한 O(1) 검증)

test_ppd_classifier_registry_updated_after_classify:
    classify() 후 _registry[session_id] 존재 + turn_count 증가 확인

test_ppd_classifier_session_ttl_expire:
    session_ttl_seconds=0.001 → 짧은 대기 후 expire_sessions() > 0

test_ppd_classifier_reset_session_removes_entry:
    classify() 후 reset_session() → _registry에서 제거 확인

test_ppd_classifier_multiple_sessions_independent:
    session_A와 session_B의 turn_count가 독립적으로 관리됨

test_ppd_classifier_new_token_ratio_calculation:
    prev_total=100, current_total=115 →
    new_token_ratio = 15/115 ≈ 0.130 < append_threshold(0.15) → "append"
```

### 통합 테스트 명세 — test_dapq_session_segment_dual_e2e.py

```
test_e2e_dual_pipeline_put_get_basic:
    put → get 왕복 기본 동작

test_e2e_dual_pipeline_process_session_returns_entries:
    put_turn_segment() × 3 후 process_session() →
    결과 리스트 비어 있지 않음

test_e2e_dual_pipeline_segment_selection_smaller_than_all:
    segment_keep_ratio=0.50 → 선택 세그먼트 수 < 전체 세그먼트 수

test_e2e_dual_pipeline_accuracy_preserved_cosine_above_099:
    process_session() 후 압축된 KV의 cosine_sim >= 0.99 (MANDATORY §5)

test_e2e_dual_pipeline_dual_reduction_ratio_above_60pct:
    segment_keep_ratio=0.50, kv_budget_ratio=0.30 →
    dual_reduction_ratio() >= 0.60 (0.50 × 0.30 = 0.85 감소 추정)

test_e2e_dual_pipeline_metrics_summary_all_keys:
    metrics_summary()에 필수 키 포함:
    [session_cache_hit_rate, session_noncontiguous_hit_rate,
     eviction_memory_reduction_ratio, dual_reduction_estimate,
     total_memory_bytes]

test_e2e_dual_pipeline_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_e2e_dual_pipeline_solo_b_vs_solo_c_vs_cross:
    SessionAwareTurnLevelSegmentCache 단독 /
    DapQPositionAwareEvictionCodec 단독 /
    DualReductionPipeline 3방향 메모리 감소율 비교 기록

test_e2e_dual_pipeline_runner_integration:
    InferenceRunner(cache=DapQSessionSegmentDualReductionPipeline)로
    run_batch() 호출 성공 (src/engine/runner.py 사용)

test_e2e_dual_pipeline_ppd_classifier_integration:
    PPDAppendFullPrefillClassifier.classify() 결과 "append"이면
    process_session()을 호출하고 D 노드 로컬 KV 재사용 경로 확인
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 4개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - `test_dapq_budget_30pct_relative_error_below_1pct` 통과 (relative_error < 0.01)
      - `test_dapq_budget_30pct_cosine_similarity_above_099` 통과 (cosine_sim >= 0.99)
      - `test_dapq_longbench_8subtask_proxy` 통과 (8개 모두 cosine_sim >= 0.99)
      - `test_dapq_memory_reduction_30pct_budget_above_60pct` 통과 (reduction >= 0.60)
- [ ] `evaluation_criteria.md` §3 Activity B 항목 충족:
      - `test_session_cache_noncontiguous_hit_tracking` 통과
      - `test_session_cache_get_top_segments_by_position_count` 통과
      - `test_session_cache_session_lru_penalty_preserves_session_entries` 통과
- [ ] `evaluation_criteria.md` §2 Activity A 항목 충족:
      - `test_ppd_classifier_overhead_below_1ms` 통과 (TTFT overhead 검증)
      - `test_ppd_classifier_turn2_small_new_tokens_append` 통과
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함:
      - `test_e2e_dual_pipeline_accuracy_preserved_cosine_above_099` 통과 (MANDATORY)
      - B 단독 / C 단독 / Cross B+C 3방향 비교 수치 기록
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - `DapQPositionAwareEvictionCodec`, `SessionAwareTurnLevelSegmentCache`,
        `DapQSessionSegmentDualReductionPipeline` 모두 CacheStore 인터페이스 구현
      - 기존 테스트(`test_segmented_cache.py`, `test_block_union_noncontiguous_index.py`,
        `test_ppd_router.py` 등) 회귀 없이 통과
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-22.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-22/metrics.json`에 JSON 기록:
      ```json
      {
        "inference_throughput_improvement_pct": ...,
        "kv_memory_reduction_ratio_c3_solo": ...,
        "kv_memory_reduction_ratio_bc_cross": ...,
        "dapq_relative_error_budget_030": ...,
        "dapq_cosine_similarity_budget_030": ...,
        "dapq_kl_divergence": ...,
        "dapq_niah_needle_preservation_rate": ...,
        "dapq_longbench_8subtask_cosine_min": ...,
        "dapq_position_vs_semantic_accuracy_delta": ...,
        "effective_context_length_multiplier": ...,
        "session_b1_noncontiguous_hit_rate": ...,
        "session_b1_total_cache_hit_rate_improvement_pct": ...,
        "ppd_classifier_overhead_mean_us": ...,
        "ppd_append_prefill_ratio": ...,
        "ppd_kv_transfer_reduction_estimate_pct": ...,
        "bc_dual_reduction_estimate": ...,
        "bc_combined_memory_reduction_ratio": ...,
        "bc_vs_solo_b1_throughput_pct": ...,
        "bc_vs_solo_c3_throughput_pct": ...,
        "bc_accuracy_cosine": ...
      }
      ```
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
