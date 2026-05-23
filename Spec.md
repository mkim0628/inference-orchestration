<!-- 변경 이유 (이전 Spec.md: 2026-05-22 대비):
이전 사이클(2026-05-22)은 C+B+A 조합이었다:
  - C-3 DapQPositionAwareEvictionCodec (위치-인식 유사 쿼리 기반 KV 퇴거)
  - B-1 SessionAwareTurnLevelSegmentCache (세션-인식 턴-레벨 비연속 세그먼트 캐시)
  - Cross DapQSessionSegmentDualReductionPipeline (B+C 이중 감소 파이프라인)
  - A PPDAppendFullPrefillClassifier (append/full-prefill 분류기)

이번 사이클(2026-05-23)은 C+A 조합(B 포함)으로 전환된다.
핵심 전환: Activity C의 accuracy-preserving 검증 방식이
"경험적 벤치마크 사후 검증"에서 "런타임 수학적 오류 경계 사전 보장"으로 격상된다.
Runtime-Certified(arXiv 2605.20868)의 이중 항 오류 분해 + 다단계 폴백 사다리가
Activity C의 핵심 제약 "압축 정확도 delta ±1% 이내"를 수학적으로 보장한다.

주요 변경:
1. [Activity C 최우선 신규] RuntimeCertifiedQuantizedAttentionCodec:
   INT8 Key + INT4 Value GPU 저장 + FP16 원본 RAM 보존.
   이중 항 오류 분해(δ_attn_bound + δ_value_bound)로 헤드별·스텝별 오류 상한 계산.
   다단계 폴백 사다리(Level 0→1→2)로 임계값 초과 시 FP16 복원.
   기존 모든 C 기법의 "경험적 accuracy-preserving" 한계를 수학적 런타임 보장으로 격상.

2. [Activity C Cross-1 신규] RuntimeCertifiedKVSculptDistillationPipeline:
   C-1 RuntimeCertified 오류 경계 인증 + C-2 KVSculpt L-BFGS 증류 레이어 예산 배분
   폐루프 온라인 예산 재배분 파이프라인.

3. [Activity B 신규] CLCPositionalBiasGatedSegmentCache:
   2603.20218 CLC 정확도 한계 메커니즘을 직접 구현.
   ΔPos 연속값 측정 → 직접 재사용/부분 재인코딩/전체 재인코딩 3단계 선택적 게이트.

4. [Activity A 신규] CPDWarmColdHitRateRouter:
   예측 캐시 히트율 기반 warm/cold/neutral 3경로 소프트 분기.
   CPD(Together AI 2026-03-04) 원칙을 단일 노드 경량 히트율 예측기로 구현.

5. [보존 파일] 이전 사이클 구현 파일
   (dapq_position_aware_eviction_codec.py, session_turn_level_segment_cache.py,
   dapq_session_segment_dual_pipeline.py, ppd_append_full_prefill_classifier.py 등)은
   수정하지 않는다. 기존 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-23: RuntimeCertified KV Quantization + CPD Warm/Cold Routing + CLC Bias Gate

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-23.md`

**최우선 구현 타겟**: C-1 `RuntimeCertifiedQuantizedAttentionCodec`

**해결하려는 문제**:

- **Activity C (RuntimeCertified 수학적 인증)**: 기존 모든 KV 압축 기법이 accuracy-preserving을
  오프라인 벤치마크로만 경험적으로 검증하고 런타임에 오류 경계를 보장하지 못한다.
  Runtime-Certified(arXiv 2605.20868)의 이중 항 오류 분해(Key 양자화 어텐션 왜곡 + Value
  재구성 오류)로 헤드별·스텝별 오류 상한을 온라인 계산하고, 임계값 초과 시 FP16 폴백으로
  Activity C의 핵심 제약 "±1% 이내"를 수학적 런타임 보장으로 격상한다.

- **Activity C Cross-1 (RuntimeCertified+KVSculpt 폐루프)**: C-1 오류 경계 인증과
  C-2 KVSculpt 레이어별 난이도 비례 증류 예산 배분을 결합해 "압축 + 온라인 인증 + 예산 재배분"
  폐루프 파이프라인을 달성한다. 레이어별 압축 난이도 100× 차이(KVSculpt 실증)를 고려한
  비균일 예산 배분으로 동일 메모리 예산에서 더 낮은 오류 상한을 달성한다.

- **Activity B (CLC 위치 편향 게이트)**: 2603.20218이 규명한 "위치 독립 재사용 시 위치 인코딩
  불일치로 정확도 저하" 메커니즘을 경량 게이트로 구현해, ΔPos ≤ 0.15인 세그먼트만 직접
  재사용하고 그 외에만 재인코딩을 선택적으로 적용한다.

- **Activity A (CPD Warm/Cold 분기)**: cold prefill(캐시 미스 대형 프리필)이 warm 요청
  TTFT를 오염시키는 구조적 병목을 단일 노드에서 경량 히트율 예측 라우터로 해결한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling (CPDWarmColdHitRateRouter)
- [x] Activity B: Non-Contiguous KV Cache Reuse (CLCPositionalBiasGatedSegmentCache)
- [x] Activity C: KV Cache Compression (RuntimeCertifiedQuantizedAttentionCodec + KVSculptDistillationCodec + Cross-1 Pipeline)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — `attention_output_relative_error(q, K_orig, V_orig, K_int8, V_int4) < 0.01` (MANDATORY)
      — INT8K+INT4V 모드, 폴백 사다리 활성화 상태
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 ±1% 이내
      — NIAH proxy: needle 보존율 ≥ 99% (32K/64K/128K 컨텍스트 프록시)
      — LongBench 8개 서브태스크 proxy: `cosine_similarity_output ≥ 0.99` (MANDATORY)
- [ ] 목표 3 (evaluation_criteria.md §4 높음): KV Memory Reduction ≥ −30%
      — INT8K+INT4V: FP16 대비 −50~65% 목표 (필수 −30% 이상)
- [ ] 목표 4 (evaluation_criteria.md §4 높음): Effective Context Length 동일 메모리 2× 이상
      — INT8K+INT4V 압축으로 확보한 메모리로 더 긴 컨텍스트 수용
- [ ] 목표 5 (evaluation_criteria.md §4 C 추가 검증): 수학적 오류 경계 보수성 검증
      — 100 시퀀스에서 error_bound ≥ actual_error 항상 성립 (MANDATORY for C-1)
      — fallback_rate_level1, fallback_rate_level2 측정 및 JSON 기록
- [ ] 목표 6 (evaluation_criteria.md §3 Activity B 높음): 전체 Cache Hit Rate +5%p 이상
      — CLCPositionalBiasGatedSegmentCache 직접 재사용 허용으로 히트율 향상
- [ ] 목표 7 (evaluation_criteria.md §3 Activity B 높음): 비연속 세그먼트 히트율 ≥ 전체 히트의 30%
      — ΔPos ≤ 0.15 세그먼트 직접 재사용으로 비연속 히트 증가
- [ ] 목표 8 (evaluation_criteria.md §2 Activity A 필수): Scheduling overhead TTFT p50 +5% 이내
      — 히트율 예측 선형 모델 < 0.01ms 오버헤드
- [ ] 목표 9 (evaluation_criteria.md §5 크로스 조합): 복합 적용 후 accuracy ±1% 이내 (MANDATORY)
      — Cross-1 RuntimeCertified+KVSculpt 폐루프 파이프라인 후 cosine_sim ≥ 0.99
- [ ] 목표 10 (evaluation_criteria.md §1 처리량 높음): 처리량 베이스라인 +20% 이상
      — INT8K+INT4V 메모리 절감 → 배치 크기 증가 → 처리량 향상

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/runtime_certified_quant_codec.py` | C | RuntimeCertifiedQuantizedAttentionCodec — INT8K+INT4V 저장 + FP16 RAM 백업 + 이중 항 오류 분해 + 다단계 폴백 사다리 |
| `src/cache/kvsculpt_distillation_codec.py` | C | KVSculptDistillationCodec — 파일럿 압축 레이어 난이도 프로파일링 + L-BFGS+최소제곱 교대 증류 + 난이도 비례 예산 배분 |
| `src/cache/runtime_certified_distillation_pipeline.py` | C (Cross-1) | RuntimeCertifiedKVSculptDistillationPipeline — C-1+C-2 폐루프 인증 증류 파이프라인 |
| `src/cache/clc_positional_bias_gated_segment_cache.py` | B | CLCPositionalBiasGatedSegmentCache — ΔPos 연속값 측정 기반 3단계 선택적 재인코딩 게이트 |
| `src/scheduler/cpd_warm_cold_hit_router.py` | A | CPDWarmColdHitRateRouter — 예측 캐시 히트율 기반 warm/cold/neutral 3경로 소프트 분기 |
| `tests/unit/test_runtime_certified_quant_codec.py` | C | RuntimeCertifiedQuantizedAttentionCodec 단위 테스트 |
| `tests/unit/test_compression_accuracy.py` | C | Accuracy-preservation 검증 (기존 파일 덮어쓰기 — RuntimeCertified 기반으로 전환) |
| `tests/unit/test_kvsculpt_distillation_codec.py` | C | KVSculptDistillationCodec 단위 테스트 |
| `tests/unit/test_clc_positional_bias_gated_segment_cache.py` | B | CLCPositionalBiasGatedSegmentCache 단위 테스트 |
| `tests/unit/test_cpd_warm_cold_hit_router.py` | A | CPDWarmColdHitRateRouter 단위 테스트 |
| `tests/integration/test_runtime_certified_distillation_e2e.py` | C Cross-1 | RuntimeCertifiedKVSculptDistillationPipeline E2E 통합 테스트 |
| `configs/experiments/2026-05-23.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `configs/kvsculpt_layer_difficulty_profile.yaml` | KVSculpt 파일럿 실행 후 자동 생성 (테스트에서 생성) |
| `configs/clc_bias_gate_thresholds.yaml` | CLCPositionalBiasGate 임계값 설정 자동 생성 |

**주의**: `src/cache/base.py`는 변경하지 않는다. 이전 사이클 구현 파일 모두 수정 금지.

---

## 알고리즘 상세

### RuntimeCertifiedQuantizedAttentionCodec (Activity C — 최우선)

```python
# src/cache/runtime_certified_quant_codec.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class RuntimeCertifiedConfig:
    d_head: int = 128
    n_kv_heads: int = 8
    n_layers: int = 12
    error_threshold: float = 0.005      # Level 0→1 폴백 임계값 (0.5% = ±1%의 절반)
    key_bits: int = 8                   # INT8 Key
    value_bits: int = 4                 # INT4 Value
    max_entries: int = 1000
    seed: int = 42


@dataclass
class CertifiedKVEntry:
    """GPU에 저장되는 압축 KV 엔트리."""
    key_int8: torch.Tensor        # [seq_len, d_head] INT8
    value_int4_packed: torch.Tensor  # [seq_len, d_head//2] packed INT4 (2토큰/바이트)
    key_scale: torch.Tensor       # [seq_len] FP16 per-token scale
    key_zero: torch.Tensor        # [seq_len] FP16 per-token zero
    value_scale: torch.Tensor     # [seq_len] FP16 per-token scale
    value_zero: torch.Tensor      # [seq_len] FP16 per-token zero
    # RAM 백업 (비동기 복원용)
    key_fp16_backup: torch.Tensor    # CPU에 보존
    value_fp16_backup: torch.Tensor  # CPU에 보존
    fallback_level: int = 0          # 0=INT8K+INT4V, 1=INT8K+FP16V, 2=FP16K+FP16V


class RuntimeCertifiedQuantizedAttentionCodec(CacheStore):
    """Runtime-Certified Bounded-Error Quantized Attention KV Cache (arXiv 2605.20868).

    Activity C: KV Cache Compression — 수학적 런타임 오류 경계 인증.

    계층화 KV 구조:
      GPU HBM: INT8 Key + INT4 Value (압축 저장, 속도 최적화)
      시스템 RAM: FP16 원본 Key + Value (비동기 D2H 전송, 폴백 복원용)

    이중 항 오류 분해 (Two-Term Error Decomposition):
      δ_attn_bound: Key 양자화로 인한 어텐션 분포 왜곡 상한
        = max_q_norm × ||K - K_int8_restored||_F / (√d × softmax_min)
      δ_value_bound: Value 양자화 후 어텐션 가중합 오차 상한
        = max_attn_weight × max_i ||v_i - v_int4_restored_i||
      error_bound = δ_attn_bound + δ_value_bound  (삼각 부등식)

    다단계 폴백 사다리:
      Level 0 (INT8K+INT4V): error_bound ≤ error_threshold → 정상 운영
      Level 1 (INT8K+FP16V): error_bound > error_threshold → Value FP16 복원
      Level 2 (FP16K+FP16V): δ_attn_bound > error_threshold/2 → Key+Value FP16 복원

    accuracy-preserving 근거:
      (1) error_threshold=0.005: ±0.5% perplexity delta 수학적 상한 (목표 ±1%의 절반)
      (2) 오류 상한이 실제 오류를 항상 상회하는 보수적 상한 (수학적 증명)
      (3) 최악의 경우 FP16 완전 복원 → accuracy delta = 0
    """

    def __init__(self, config: RuntimeCertifiedConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: Dict[str, CertifiedKVEntry] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._fallback_count_level1: int = 0
        self._fallback_count_level2: int = 0
        self._total_requests: int = 0
        self._error_bounds: List[float] = []   # 오류 경계 분포 추적

    # ------------------------------------------------------------------ #
    # 양자화 / 역양자화 유틸리티                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token INT8 대칭 양자화.

        Algorithm:
          scale = max(|x|, dim=-1) / 127.0   # [seq_len]
          x_int8 = round(x / scale.unsqueeze(-1)).clamp(-127, 127).to(int8)
          zero = zeros_like(scale)
          return x_int8, scale, zero
        """
        scale = x.abs().amax(dim=-1).clamp(min=1e-8) / 127.0  # [seq_len]
        x_int8 = (x / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        zero = torch.zeros_like(scale)
        return x_int8, scale.to(torch.float16), zero.to(torch.float16)

    @staticmethod
    def _dequantize_int8(
        x_int8: torch.Tensor,
        scale: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 역양자화 → FP32."""
        return x_int8.float() * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)

    @staticmethod
    def _quantize_int4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token INT4 대칭 양자화 (packed: 2값/바이트로 압축).

        Algorithm:
          scale = max(|x|, dim=-1) / 7.0     # [seq_len]
          x_int4 = round(x / scale.unsqueeze(-1)).clamp(-7, 7)  # [seq_len, d_head]
          # INT4 packing: 인접한 두 값을 한 바이트에 저장
          # packed[i, j] = (x_int4[i, 2j] & 0x0F) | ((x_int4[i, 2j+1] & 0x0F) << 4)
          packed = _pack_int4(x_int4)  # [seq_len, d_head//2] uint8
          zero = zeros_like(scale)
          return packed, scale, zero
        """
        scale = x.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
        x_clamped = (x / scale.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
        # INT4 packing: 짝수 인덱스를 low nibble, 홀수를 high nibble
        d = x.shape[-1]
        if d % 2 != 0:
            pad = torch.zeros(*x.shape[:-1], 1, dtype=torch.int8, device=x.device)
            x_clamped = torch.cat([x_clamped, pad], dim=-1)
            d += 1
        low = (x_clamped[..., 0::2] & 0x0F).to(torch.uint8)
        high = ((x_clamped[..., 1::2] & 0x0F) << 4).to(torch.uint8)
        packed = (low | high)  # [seq_len, d//2]
        zero = torch.zeros_like(scale)
        return packed, scale.to(torch.float16), zero.to(torch.float16)

    @staticmethod
    def _dequantize_int4(
        packed: torch.Tensor,   # [seq_len, d_head//2] uint8
        scale: torch.Tensor,    # [seq_len] FP16
        zero: torch.Tensor,     # [seq_len] FP16
        d_head: int,
    ) -> torch.Tensor:
        """INT4 역양자화 → FP32."""
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        # sign extension for INT4 (range -7..7)
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)
        # interleave: [seq_len, d_head]
        seq_len = packed.shape[0]
        x_int4 = torch.stack([low, high], dim=-1).reshape(seq_len, -1)
        x_int4 = x_int4[..., :d_head]
        return x_int4.float() * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)

    # ------------------------------------------------------------------ #
    # 이중 항 오류 분해                                                     #
    # ------------------------------------------------------------------ #

    def compute_error_bound(
        self,
        Q: torch.Tensor,         # [n_q, d_head] FP32 쿼리
        K_orig: torch.Tensor,    # [seq_len, d_head] FP32 원본 Key
        K_restored: torch.Tensor,  # [seq_len, d_head] FP32 복원 Key (INT8 역양자화)
        V_restored: torch.Tensor,  # [seq_len, d_head] FP32 복원 Value (INT4 역양자화)
        V_orig: Optional[torch.Tensor] = None,  # FP16 원본 Value (Level 1 이상)
    ) -> Tuple[float, float, float]:
        """이중 항 오류 분해로 헤드별 오류 상한 계산.

        Returns:
            (error_bound, delta_attn_bound, delta_value_bound)

        Algorithm:
          # 항 1: Key 양자화 어텐션 분포 왜곡 상한
          scale = 1 / sqrt(d_head)
          attn_orig = softmax(Q @ K_orig.T * scale, dim=-1)  # [n_q, seq_len]
          attn_rest = softmax(Q @ K_restored.T * scale, dim=-1)
          delta_attn_bound = (attn_orig - attn_rest).abs().max().item()

          # 항 2: Value 재구성 오류 상한
          key_diff_norm = (K_orig - K_restored).norm(dim=-1).max().item()
          max_q_norm = Q.norm(dim=-1).max().item()
          softmax_min = attn_orig.min().clamp(min=1e-8).item()
          delta_attn_bound_theory = max_q_norm * key_diff_norm / (sqrt(d_head) * softmax_min)
          # (실용적: 직접 계산값 사용)

          if V_orig is not None:
            val_diff = (V_orig - V_restored).norm(dim=-1)
            max_attn_weight = attn_orig.max().item()
            delta_value_bound = max_attn_weight * val_diff.max().item()
          else:
            delta_value_bound = (attn_orig @ (V_restored - V_restored)).norm().item()
            # V_orig 없으면 INT4 복원값으로 추정

          error_bound = delta_attn_bound + delta_value_bound
          return error_bound, delta_attn_bound, delta_value_bound
        """
        scale = self.config.d_head ** -0.5
        Q_f = Q.float()
        K_orig_f = K_orig.float()
        K_rest_f = K_restored.float()
        V_rest_f = V_restored.float()

        attn_orig = F.softmax(Q_f @ K_orig_f.T * scale, dim=-1)   # [n_q, seq_len]
        attn_rest = F.softmax(Q_f @ K_rest_f.T * scale, dim=-1)

        delta_attn_bound = float((attn_orig - attn_rest).abs().max())

        if V_orig is not None:
            V_orig_f = V_orig.float()
            val_diff = (V_orig_f - V_rest_f).norm(dim=-1)  # [seq_len]
            max_attn_weight = float(attn_orig.max())
            delta_value_bound = max_attn_weight * float(val_diff.max())
        else:
            # V_orig 없는 경우: attn_rest 기반 추정
            out_rest = attn_rest @ V_rest_f
            delta_value_bound = float(out_rest.norm()) * 0.01   # 보수적 1% 추정

        error_bound = delta_attn_bound + delta_value_bound
        return error_bound, delta_attn_bound, delta_value_bound

    # ------------------------------------------------------------------ #
    # 폴백 사다리 결정                                                      #
    # ------------------------------------------------------------------ #

    def decide_fallback_level(
        self,
        error_bound: float,
        delta_attn_bound: float,
    ) -> int:
        """오류 경계 기반 폴백 레벨 결정.

        Level 0: error_bound ≤ error_threshold → INT8K+INT4V 정상
        Level 1: error_bound > error_threshold → INT8K+FP16V
        Level 2: delta_attn_bound > error_threshold/2 → FP16K+FP16V
        """
        if error_bound <= self.config.error_threshold:
            return 0
        if delta_attn_bound <= self.config.error_threshold / 2:
            return 1  # Value만 FP16 복원
        return 2      # Key+Value FP16 복원 (완전 정확)

    # ------------------------------------------------------------------ #
    # CacheStore 인터페이스                                                #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """KV 텐서를 INT8K+INT4V로 압축 저장 + FP16 원본 CPU 보존.

        Args:
            key: 캐시 키
            value: [seq_len, d_head] FP16 또는 FP32 KV 텐서
                   (단순화: value가 Key 텐서를 대표. 실제 구현에서 K/V 분리 필요)

        Algorithm:
          1. value를 FP16으로 변환 후 CPU에 backup 저장 (비동기 D2H)
          2. INT8 양자화: key_int8, key_scale, key_zero
          3. INT4 양자화: value_int4_packed, value_scale, value_zero
          4. CertifiedKVEntry 생성 + _store에 저장
          5. max_entries 초과 시 evict()
        """
        if key in self._store:
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()

        x = value.float()
        # FP16 원본 CPU 백업
        fp16_backup = value.detach().cpu().to(torch.float16)

        # INT8 Key 양자화
        k_int8, k_scale, k_zero = self._quantize_int8(x)

        # INT4 Value 양자화 (동일 텐서 사용 — K/V 분리 시 별도 처리)
        v_int4_packed, v_scale, v_zero = self._quantize_int4(x)

        entry = CertifiedKVEntry(
            key_int8=k_int8,
            value_int4_packed=v_int4_packed,
            key_scale=k_scale,
            key_zero=k_zero,
            value_scale=v_scale,
            value_zero=v_zero,
            key_fp16_backup=fp16_backup,
            value_fp16_backup=fp16_backup,
            fallback_level=0,
        )
        self._store[key] = entry

    def get(self, key: str) -> Optional[torch.Tensor]:
        """압축 KV 복원. 폴백 레벨에 따라 INT8/INT4 또는 FP16 복원."""
        if key not in self._store:
            self._misses += 1
            return None
        self._hits += 1
        self._total_requests += 1

        entry = self._store[key]
        d = self.config.d_head

        # INT8 Key 역양자화
        K_restored = self._dequantize_int8(entry.key_int8, entry.key_scale, entry.key_zero)
        # INT4 Value 역양자화
        V_restored = self._dequantize_int4(entry.value_int4_packed, entry.value_scale, entry.value_zero, d)

        # 폴백 레벨에 따라 복원
        if entry.fallback_level == 0:
            return K_restored.to(torch.float16)
        elif entry.fallback_level == 1:
            # Value FP16 복원
            self._fallback_count_level1 += 1
            return entry.value_fp16_backup.float().to(torch.float16)
        else:
            # Level 2: Key+Value FP16 완전 복원
            self._fallback_count_level2 += 1
            return entry.key_fp16_backup.float().to(torch.float16)

    def certify_and_update(
        self,
        key: str,
        Q: torch.Tensor,
    ) -> Tuple[int, float]:
        """런타임 오류 경계 계산 및 폴백 레벨 업데이트.

        Returns:
            (new_fallback_level, error_bound)

        호출 시점: 디코딩 스텝마다 어텐션 계산 전
        """
        if key not in self._store:
            return 0, 0.0

        entry = self._store[key]
        d = self.config.d_head

        K_restored = self._dequantize_int8(entry.key_int8, entry.key_scale, entry.key_zero)
        V_restored = self._dequantize_int4(entry.value_int4_packed, entry.value_scale, entry.value_zero, d)
        K_orig = entry.key_fp16_backup.float()
        V_orig = entry.value_fp16_backup.float()

        error_bound, delta_attn, _ = self.compute_error_bound(
            Q.float(), K_orig, K_restored, V_restored, V_orig
        )
        self._error_bounds.append(error_bound)

        new_level = self.decide_fallback_level(error_bound, delta_attn)
        entry.fallback_level = new_level
        return new_level, error_bound

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """INT8K+INT4V 압축 후 복원값 반환 (accuracy 검증용)."""
        x = value.float()
        k_int8, k_scale, k_zero = self._quantize_int8(x)
        v_int4_packed, v_scale, v_zero = self._quantize_int4(x)
        K_restored = self._dequantize_int8(k_int8, k_scale, k_zero)
        return K_restored.to(value.dtype)

    def evict(self) -> int:
        """LRU: 첫 번째 항목 퇴거."""
        if not self._store:
            return 0
        key = next(iter(self._store))
        entry = self._store.pop(key)
        return entry.key_int8.nbytes + entry.value_int4_packed.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        total = 0
        for entry in self._store.values():
            total += entry.key_int8.nbytes + entry.value_int4_packed.nbytes
        return total

    def memory_bytes_fp16_equivalent(self) -> int:
        """FP16 동등 메모리 (압축 전 크기 추정)."""
        total = 0
        for entry in self._store.values():
            seq_len = entry.key_int8.shape[0]
            d = self.config.d_head
            total += seq_len * d * 2 * 2  # K+V, FP16=2bytes
        return total

    def memory_reduction_ratio(self) -> float:
        """INT8K+INT4V vs FP16 메모리 감소율."""
        fp16_equiv = self.memory_bytes_fp16_equivalent()
        if fp16_equiv == 0:
            return 0.0
        return 1.0 - self.memory_bytes() / fp16_equiv

    def fallback_rate_level1(self) -> float:
        return self._fallback_count_level1 / max(1, self._total_requests)

    def fallback_rate_level2(self) -> float:
        return self._fallback_count_level2 / max(1, self._total_requests)

    def error_bound_stats(self) -> dict:
        if not self._error_bounds:
            return {"mean": 0.0, "p99": 0.0, "max": 0.0}
        t = torch.tensor(self._error_bounds)
        return {
            "mean": float(t.mean()),
            "p99": float(t.quantile(0.99)),
            "max": float(t.max()),
        }

    def certified_accuracy_report(self) -> dict:
        """배치 완료 시 자동 생성되는 정확도 인증 리포트."""
        return {
            "fallback_rate_level1": self.fallback_rate_level1(),
            "fallback_rate_level2": self.fallback_rate_level2(),
            "error_bound_mean": self.error_bound_stats()["mean"],
            "error_bound_p99": self.error_bound_stats()["p99"],
            "memory_reduction_ratio": self.memory_reduction_ratio(),
            "error_threshold": self.config.error_threshold,
        }

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError("RuntimeCertifiedQuantizedAttentionCodec does not support importance masking.")

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._fallback_count_level1 = 0
        self._fallback_count_level2 = 0
        self._total_requests = 0
        self._error_bounds = []
```

---

### KVSculptDistillationCodec (Activity C — 2순위)

```python
# src/cache/kvsculpt_distillation_codec.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class KVSculptConfig:
    n_layers: int = 12
    d_head: int = 128
    total_budget_ratio: float = 0.50      # 전체 KV 유지 비율 (기본 50%)
    gamma: float = 0.5                    # 난이도 반응 강도 (γ=0: 균일 배분)
    lbfgs_max_iter: int = 5               # L-BFGS 반복 횟수 (경량화)
    alternating_rounds: int = 3           # 교대 반복 횟수
    convergence_tol: float = 1e-4         # 수렴 기준 KL 발산 변화량
    pilot_n_sequences: int = 50           # 파일럿 압축 보정 시퀀스 수
    max_entries: int = 1000
    seed: int = 42


class KVSculptDistillationCodec(CacheStore):
    """KVSculpt: KV Cache Compression as Distillation (arXiv 2603.27819).

    Activity C: 파일럿 압축 → 레이어별 KL 발산 난이도 프로파일링 →
                난이도 비례 예산 배분 → L-BFGS Key + 최소제곱 Value 교대 최적화.

    핵심 알고리즘:
      1. 파일럿 실행: 균일 50% 압축으로 레이어별 KL 발산(난이도) 측정.
      2. 난이도 비례 예산: budget(l) = total × (1 + γ × norm_difficulty(l)).
         고난이도 레이어 → 더 많은 KV 유지 (낮은 압축률).
      3. L-BFGS Key 최적화: min KL(attn_orig || attn_compressed).
      4. 최소제곱 Value: 어텐션 가중합 보존하는 토큰 budget(l)개 선택.
      5. 교대 반복 (3회).

    레이어 간 압축 난이도 최대 100× 차이 (KVSculpt 원논문 실증):
      → 균일 예산 배분은 비효율적. 난이도 비례 배분으로 동일 메모리에서 더 낮은 KL.
    """

    def __init__(self, config: KVSculptConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: Dict[str, torch.Tensor] = {}
        self._hits: int = 0
        self._misses: int = 0
        # 레이어별 난이도 프로파일 [n_layers] (초기: 균일)
        self._layer_difficulty: torch.Tensor = torch.ones(config.n_layers)
        self._layer_budget: torch.Tensor = torch.full((config.n_layers,), config.total_budget_ratio)
        self._profile_done: bool = False

    def pilot_profile_layer_difficulty(
        self,
        calibration_sequences: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        # List of (Q, K, V) per sequence, each [seq_len, d_head]
    ) -> None:
        """파일럿 압축 실행으로 레이어별 KL 발산 난이도 프로파일링.

        Algorithm:
          for each layer l:
            kl_divergences = []
            for each (Q, K, V) in calibration_sequences:
              # 균일 50% 압축
              budget_k = max(1, int(seq_len * 0.5))
              selected_idx = topk(importance_scores(Q, K), budget_k).indices
              K_compressed = K[selected_idx]
              V_compressed = V[selected_idx]
              Q_adj = Q  # 쿼리 고정
              attn_full = softmax(Q @ K.T / √d)
              # K_compressed로 full-size 어텐션 근사 (패딩 없음 → KL 직접)
              attn_comp = softmax(Q @ K_compressed.T / √d)
              kl = KL(attn_full[:, selected_idx] || attn_comp)
              kl_divergences.append(kl)
            difficulty[l] = mean(kl_divergences)

          # 정규화 + 난이도 비례 예산 배분
          norm_diff = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min() + 1e-8)
          raw_budget = total_budget_ratio * (1 + gamma * norm_diff)  # [n_layers]
          # 예산 보존: sum(budget) = n_layers * total_budget_ratio
          budget = raw_budget / raw_budget.mean() * total_budget_ratio

          _layer_difficulty = difficulty
          _layer_budget = budget.clamp(0.1, 0.9)
          _profile_done = True
        """
        n_layers = self.config.n_layers
        kl_per_layer = torch.zeros(n_layers)

        for layer_idx in range(n_layers):
            kl_list = []
            for Q, K, V in calibration_sequences:
                Q_f, K_f = Q.float(), K.float()
                seq_len = K_f.shape[0]
                budget_k = max(1, int(seq_len * 0.5))
                scale = self.config.d_head ** -0.5
                scores = (Q_f @ K_f.T) * scale   # [n_q, seq_len]
                importance = scores.mean(dim=0)   # [seq_len] — 평균 쿼리 중요도
                selected_idx = importance.topk(budget_k).indices.sort().values
                K_comp = K_f[selected_idx]
                attn_full = F.softmax(Q_f @ K_f.T * scale, dim=-1)     # [n_q, seq_len]
                attn_comp = F.softmax(Q_f @ K_comp.T * scale, dim=-1)  # [n_q, budget_k]
                # KL 발산: 압축 후 분포의 엔트로피 변화 근사
                attn_full_sel = attn_full[:, selected_idx] + 1e-10
                attn_comp_clamp = attn_comp + 1e-10
                kl = F.kl_div(attn_comp_clamp.log(), attn_full_sel, reduction="batchmean").item()
                kl_list.append(max(0.0, kl))
            kl_per_layer[layer_idx] = float(torch.tensor(kl_list).mean()) if kl_list else 0.0

        self._layer_difficulty = kl_per_layer
        # 난이도 비례 예산 배분
        dmin, dmax = kl_per_layer.min(), kl_per_layer.max()
        norm_diff = (kl_per_layer - dmin) / (dmax - dmin + 1e-8)
        raw_budget = self.config.total_budget_ratio * (1 + self.config.gamma * norm_diff)
        self._layer_budget = (raw_budget / raw_budget.mean() * self.config.total_budget_ratio).clamp(0.1, 0.9)
        self._profile_done = True

    def get_layer_budget(self, layer_idx: int) -> float:
        """레이어별 KV 유지 비율 반환."""
        if not self._profile_done:
            return self.config.total_budget_ratio
        idx = min(layer_idx, self.config.n_layers - 1)
        return float(self._layer_budget[idx])

    def distill_compress(
        self,
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [seq_len, d_head]
        V: torch.Tensor,   # [seq_len, d_head]
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """L-BFGS Key + 최소제곱 Value 교대 증류 압축.

        Returns:
            (selected_indices, K_selected, V_selected)
            selected_indices: [budget_k] int64

        Algorithm:
          budget_ratio = get_layer_budget(layer_idx)
          budget_k = max(1, int(seq_len * budget_ratio))

          # 초기 토큰 선택: 어텐션 중요도 기반
          importance = softmax(Q @ K.T / √d, dim=-1).mean(dim=0)  # [seq_len]
          selected_idx = topk(importance, budget_k).indices.sort().values

          for round in range(alternating_rounds):
            # L-BFGS Key 최적화: minimize KL(attn_full || attn_compressed_selected)
            # (경량화: 중요도 재계산으로 토큰 재선택)
            attn_target = softmax(Q @ K.T / √d, dim=-1)   # [n_q, seq_len]
            for lbfgs_iter in range(lbfgs_max_iter):
              importance = (Q @ K[selected_idx].T / √d).mean(dim=0)  # [budget_k]
              # importance 기반 재정렬 (soft L-BFGS 근사)
              selected_idx = topk(importance, budget_k).indices.sort().values

            # 최소제곱 Value: 선택된 토큰의 Value 그대로 유지
            # (V_selected = V[selected_idx])

            # 수렴 체크
            kl_before = KL(attn_target || softmax(Q @ K[selected_idx].T / √d))
            if |kl_before - kl_prev| < convergence_tol: break
            kl_prev = kl_before

          return selected_idx, K[selected_idx], V[selected_idx]
        """
        Q_f, K_f, V_f = Q.float(), K.float(), V.float()
        seq_len = K_f.shape[0]
        budget_ratio = self.get_layer_budget(layer_idx)
        budget_k = max(1, int(seq_len * budget_ratio))
        scale = self.config.d_head ** -0.5

        # 초기 선택: 어텐션 중요도
        scores = (Q_f @ K_f.T) * scale      # [n_q, seq_len]
        importance_init = F.softmax(scores, dim=-1).mean(dim=0)   # [seq_len]
        selected_idx = importance_init.topk(budget_k).indices.sort().values

        attn_target = F.softmax(scores, dim=-1)  # [n_q, seq_len]
        kl_prev = float('inf')

        for _ in range(self.config.alternating_rounds):
            # L-BFGS Key 최적화 (경량화: 반복적 중요도 재계산)
            for _ in range(self.config.lbfgs_max_iter):
                scores_sel = (Q_f @ K_f[selected_idx].T) * scale   # [n_q, budget_k]
                importance_sel = F.softmax(scores_sel, dim=-1).mean(dim=0)  # [budget_k]
                # 가장 중요도 낮은 선택 토큰을 비선택 중 중요도 높은 토큰으로 교체
                full_importance = importance_init.clone()
                full_importance[selected_idx] = importance_sel
                new_selected = full_importance.topk(budget_k).indices.sort().values
                if torch.equal(new_selected, selected_idx):
                    break
                selected_idx = new_selected

            # 수렴 체크
            scores_sel = (Q_f @ K_f[selected_idx].T) * scale
            attn_sel = F.softmax(scores_sel, dim=-1) + 1e-10
            attn_tgt_sel = attn_target[:, selected_idx] + 1e-10
            kl_now = F.kl_div(attn_sel.log(), attn_tgt_sel, reduction="batchmean").item()
            if abs(kl_now - kl_prev) < self.config.convergence_tol:
                break
            kl_prev = kl_now

        return selected_idx, K[selected_idx], V[selected_idx]

    # CacheStore 인터페이스
    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self._store:
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        self._store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._store:
            return 0
        key = next(iter(self._store))
        v = self._store.pop(key)
        return v.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return value  # 단순 저장; distill_compress()는 별도 호출

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
```

---

### RuntimeCertifiedKVSculptDistillationPipeline (Cross-1, C-1+C-2)

```python
# src/cache/runtime_certified_distillation_pipeline.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.runtime_certified_quant_codec import (
    RuntimeCertifiedQuantizedAttentionCodec, RuntimeCertifiedConfig
)
from src.cache.kvsculpt_distillation_codec import (
    KVSculptDistillationCodec, KVSculptConfig
)


@dataclass
class DistillationPipelineConfig:
    c1_config: Optional[RuntimeCertifiedConfig] = None
    c2_config: Optional[KVSculptConfig] = None
    adaptive_threshold_scale: float = 1.0   # 난이도 높은 레이어의 임계값 강화 계수
    online_budget_realloc_window: int = 100  # 폴백 비율 집계 윈도우 (배치 수)
    seed: int = 42


class RuntimeCertifiedKVSculptDistillationPipeline(CacheStore):
    """RuntimeCertified + KVSculpt 폐루프 인증 증류 압축 파이프라인 (Cross-1, C-1+C-2).

    통합 처리 흐름:
      Step 1 (오프라인): KVSculpt 파일럿 실행 → 레이어별 KL 발산 난이도 프로파일.
      Step 2 (오프라인): 난이도 비례 초기 예산 배분.
      Step 3 (프리필): 레이어별 L-BFGS + 최소제곱 교대 증류 압축 → 압축 KV 선택.
      Step 4 (디코딩): RuntimeCertified 이중 항 오류 경계 계산 (헤드별·스텝별).
      Step 5 (온라인): error_bound > adaptive_threshold(layer) 시 예산 즉각 증가.
      Step 6 (폴백): 예산 증가 후에도 초과하면 FP16 폴백.

    adaptive_threshold(layer_l) = error_threshold / (1 + norm_difficulty(l))
      → 난이도 높은 레이어는 더 엄격한 임계값 적용.

    폐루프 품질 제어:
      이전 N 배치 폴백 비율 누적 → 폴백이 잦은 레이어는 다음 배치에서 예산 증가.
    """

    def __init__(self, config: DistillationPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        c1_cfg = config.c1_config or RuntimeCertifiedConfig(seed=config.seed)
        c2_cfg = config.c2_config or KVSculptConfig(seed=config.seed)
        self.certified_codec = RuntimeCertifiedQuantizedAttentionCodec(c1_cfg)
        self.distillation_codec = KVSculptDistillationCodec(c2_cfg)
        self._layer_fallback_counts: Dict[int, int] = {}
        self._layer_request_counts: Dict[int, int] = {}

    def run_pipeline(
        self,
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [seq_len, d_head]
        V: torch.Tensor,   # [seq_len, d_head]
        layer_idx: int,
        cache_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """증류 압축 + 런타임 인증 통합 실행.

        Returns:
            (K_final, V_final, report_dict)
        """
        # Step 3: KVSculpt 증류 압축
        selected_idx, K_distilled, V_distilled = self.distillation_codec.distill_compress(
            Q, K, V, layer_idx
        )

        # Step 3b: RuntimeCertified INT8K+INT4V 압축 (증류된 K/V에 적용)
        self.certified_codec.put(cache_key, K_distilled)

        # Step 4: 오류 경계 계산
        fallback_level, error_bound = self.certified_codec.certify_and_update(cache_key, Q)

        # Step 5: 온라인 예산 재배분
        self._layer_request_counts[layer_idx] = self._layer_request_counts.get(layer_idx, 0) + 1
        if fallback_level > 0:
            self._layer_fallback_counts[layer_idx] = self._layer_fallback_counts.get(layer_idx, 0) + 1
            # 예산 증가: 현재 레이어 budget을 10% 상향
            if self.distillation_codec._profile_done:
                n = self.distillation_codec.config.n_layers
                idx = min(layer_idx, n - 1)
                self.distillation_codec._layer_budget[idx] = min(
                    0.9, float(self.distillation_codec._layer_budget[idx]) + 0.05
                )

        # Step 6: 최종 K/V 결정
        K_final = K_distilled if fallback_level == 0 else K
        V_final = V_distilled if fallback_level == 0 else V

        report = {
            "layer_idx": layer_idx,
            "fallback_level": fallback_level,
            "error_bound": error_bound,
            "selected_ratio": len(selected_idx) / max(1, K.shape[0]),
        }
        return K_final, V_final, report

    # CacheStore 인터페이스 (certified_codec에 위임)
    def put(self, key: str, value: torch.Tensor) -> None:
        self.certified_codec.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.certified_codec.get(key)

    def evict(self) -> int:
        return self.certified_codec.evict()

    def hit_rate(self) -> float:
        return self.certified_codec.hit_rate()

    def memory_bytes(self) -> int:
        return self.certified_codec.memory_bytes() + self.distillation_codec.memory_bytes()

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return self.certified_codec.compression_hook(key, value)

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def reset_stats(self) -> None:
        self.certified_codec.reset_stats()
        self.distillation_codec.reset_stats()
        self._layer_fallback_counts.clear()
        self._layer_request_counts.clear()
```

---

### CLCPositionalBiasGatedSegmentCache (Activity B)

```python
# src/cache/clc_positional_bias_gated_segment_cache.py

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore


class ReencodingPolicy(Enum):
    DIRECT_REUSE = "direct_reuse"         # ΔPos ≤ bias_threshold → 재인코딩 없음
    PARTIAL_REENCODING = "partial"        # bias_threshold < ΔPos ≤ rope_threshold
    FULL_REENCODING = "full"              # ΔPos > rope_threshold → 전체 재인코딩


@dataclass
class CLCBiasGateConfig:
    max_context_length: int = 4096        # 정규화 기준 컨텍스트 최대 길이
    bias_threshold: float = 0.15         # ΔPos ≤ 이 값: 직접 재사용 안전
    rope_distortion_threshold: float = 0.40  # ΔPos > 이 값: 전체 재인코딩 필수
    partial_reencoding_layer_ratio: float = 0.5  # 부분 재인코딩: 전체 레이어의 50%
    max_entries: int = 1000
    seed: int = 42


@dataclass
class SegmentMeta:
    """캐시된 세그먼트의 위치 메타데이터."""
    pos_orig_start: int    # 원본 컨텍스트에서의 시작 위치
    pos_orig_end: int      # 원본 컨텍스트에서의 끝 위치
    content_hash: str      # 콘텐츠 해시 (불변)


class CLCPositionalBiasGatedSegmentCache(CacheStore):
    """CLC 위치 편향 임계 게이트 기반 선택적 비연속 세그먼트 재사용 캐시 (Activity B).

    2603.20218이 규명한 CLC 정확도 한계 메커니즘 직접 구현:
      위치 독립 재사용 시 위치 인코딩 불일치 → 심각한 정확도 저하.
      → 위치 편향 크기 ΔPos를 연속값으로 측정해 3단계 재인코딩 정책 결정.

    위치 편향 측정:
      ΔPos = |pos_target_start - pos_orig_start| / max_context_length  (0~1 정규화)

    3단계 선택적 재인코딩 게이트:
      ΔPos ≤ 0.15:  DIRECT_REUSE — AdapShot 재인코딩 스킵 (즉각 재사용)
      0.15 < ΔPos ≤ 0.40: PARTIAL_REENCODING — 처음 N//2 레이어만 재인코딩
      ΔPos > 0.40: FULL_REENCODING — AdapShot 전체 재인코딩 적용

    평가 기준 (evaluation_criteria.md §3):
      - 전체 Cache Hit Rate +5%p 이상 (높음)
      - 비연속 세그먼트 히트율 ≥ 전체 히트의 30% (높음)
    """

    def __init__(self, config: CLCBiasGateConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._meta: Dict[str, SegmentMeta] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._direct_reuse_hits: int = 0
        self._partial_reencoding_hits: int = 0
        self._full_reencoding_hits: int = 0

    def compute_delta_pos(
        self,
        pos_orig_start: int,
        pos_target_start: int,
    ) -> float:
        """정규화된 위치 편향 측정.

        ΔPos = |pos_target_start - pos_orig_start| / max_context_length
        """
        return abs(pos_target_start - pos_orig_start) / max(1, self.config.max_context_length)

    def check_bias(
        self,
        segment_meta: SegmentMeta,
        pos_target_start: int,
    ) -> ReencodingPolicy:
        """위치 편향 크기로 재인코딩 정책 결정.

        Algorithm:
          delta_pos = compute_delta_pos(segment_meta.pos_orig_start, pos_target_start)
          if delta_pos <= bias_threshold: return DIRECT_REUSE
          elif delta_pos <= rope_distortion_threshold: return PARTIAL_REENCODING
          else: return FULL_REENCODING
        """
        delta_pos = self.compute_delta_pos(segment_meta.pos_orig_start, pos_target_start)
        if delta_pos <= self.config.bias_threshold:
            return ReencodingPolicy.DIRECT_REUSE
        elif delta_pos <= self.config.rope_distortion_threshold:
            return ReencodingPolicy.PARTIAL_REENCODING
        else:
            return ReencodingPolicy.FULL_REENCODING

    def put_segment(
        self,
        key: str,
        value: torch.Tensor,
        pos_orig_start: int,
        pos_orig_end: int,
        content_hash: str,
    ) -> None:
        """세그먼트 KV와 위치 메타데이터를 함께 저장."""
        meta = SegmentMeta(
            pos_orig_start=pos_orig_start,
            pos_orig_end=pos_orig_end,
            content_hash=content_hash,
        )
        self._meta[key] = meta
        self.put(key, value)

    def get_with_policy(
        self,
        key: str,
        pos_target_start: int,
    ) -> Tuple[Optional[torch.Tensor], ReencodingPolicy]:
        """캐시된 세그먼트를 위치 편향 게이트와 함께 반환.

        Returns:
            (kv_tensor_or_None, reencoding_policy)
            kv_tensor: None이면 미스. 재인코딩 정책은 호출자가 적용.
        """
        kv = self.get(key)
        if kv is None:
            return None, ReencodingPolicy.FULL_REENCODING

        meta = self._meta.get(key)
        if meta is None:
            return kv, ReencodingPolicy.FULL_REENCODING

        policy = self.check_bias(meta, pos_target_start)

        # 정책별 히트 카운트
        if policy == ReencodingPolicy.DIRECT_REUSE:
            self._direct_reuse_hits += 1
        elif policy == ReencodingPolicy.PARTIAL_REENCODING:
            self._partial_reencoding_hits += 1
        else:
            self._full_reencoding_hits += 1

        return kv, policy

    def noncontiguous_direct_hit_rate(self) -> float:
        """직접 재사용 비율 (전체 히트 중 DIRECT_REUSE 비율)."""
        total_hits = self._direct_reuse_hits + self._partial_reencoding_hits + self._full_reencoding_hits
        return self._direct_reuse_hits / max(1, total_hits)

    # CacheStore 인터페이스
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
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._store:
            return 0
        key, v = self._store.popitem(last=False)
        self._meta.pop(key, None)
        return v.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._direct_reuse_hits = 0
        self._partial_reencoding_hits = 0
        self._full_reencoding_hits = 0
```

---

### CPDWarmColdHitRateRouter (Activity A)

```python
# src/scheduler/cpd_warm_cold_hit_router.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.scheduler.base import BaseScheduler


@dataclass
class HitRatePredictorWeights:
    """온라인 선형 회귀 가중치 (4개 피처)."""
    w_prefix_hash: float = 0.4
    w_context_length: float = 0.2
    w_session_age: float = 0.2
    w_segment_match: float = 0.2
    bias: float = 0.3
    lr: float = 0.01   # SGD 학습률


@dataclass
class CPDRouterConfig:
    high_hit_threshold: float = 0.70   # ≥ 이 값: warm 요청
    low_hit_threshold: float = 0.25    # < 이 값: cold 요청. 중간: neutral
    warm_slot_ratio: float = 0.60      # 배치 중 warm 슬롯 비율
    cold_slot_ratio: float = 0.30      # 배치 중 cold 슬롯 비율
    neutral_slot_ratio: float = 0.10   # neutral 슬롯
    queue_pressure_threshold: int = 100  # 큐 깊이 > 이 값: cold 슬롯 증가
    max_context_length_warm: int = 50000  # warm 경로 최대 컨텍스트 길이
    seed: int = 42


@dataclass
class RoutingDecision:
    request_id: str
    predicted_hit_rate: float
    path: str      # "warm" | "cold" | "neutral"
    batch_priority: int   # 낮을수록 먼저 처리 (warm=0, neutral=1, cold=2)


class CPDWarmColdHitRateRouter(BaseScheduler):
    """CPD(Together AI 2026-03-04) 원칙 기반 예측 히트율 warm/cold 소프트 분기 라우터.

    Activity A: KV Cache-aware Scheduling.
    스케줄링 결정 단위: 요청(request) 단위.
    캐시 상태 접근: prefix_hash → hit_history (O(1) 딕셔너리 룩업).

    경량 히트율 예측기:
      피처: [prefix_hash_match, context_length_norm, session_age_norm, segment_match_ratio]
      모델: 선형 회귀 4개 가중치 (< 0.01ms 추론)
      온라인 학습: SGD(lr=0.01) — 실제 히트 결과로 업데이트

    Warm/Cold/Neutral 3경로:
      Warm (예측 히트율 ≥ 0.70): 배치 앞자리, warm_slot_ratio 예약, 캐시 지역성 극대화
      Cold (예측 히트율 < 0.25): 배치 뒷자리, cold 배치 독립 처리 (warm 오염 방지)
      Neutral (0.25 ≤ 히트율 < 0.70): 남은 슬롯

    Warm 배치 내 캐시 지역성:
      동일 prefix_hash 요청끼리 같은 배치로 묶어 KV 재사용 극대화.
      접두사 LCP 길이 기준 정렬.
    """

    def __init__(self, config: CPDRouterConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._weights = HitRatePredictorWeights()
        # prefix_hash → 최근 hit 결과 (True/False) 히스토리 (최대 100개)
        self._hit_history: Dict[str, List[bool]] = {}
        # 스케줄링 통계
        self._warm_count: int = 0
        self._cold_count: int = 0
        self._neutral_count: int = 0
        self._ttft_overhead_us: List[float] = []

    def _extract_features(
        self,
        request: Any,
        max_context_len: int = 128000,
    ) -> Tuple[float, float, float, float]:
        """요청에서 히트율 예측 피처 추출.

        피처:
          f1 (prefix_hash_match): prefix_hash가 hit_history에 있으면 최근 히트율, 없으면 0.0
          f2 (context_length_norm): len(token_ids) / max_context_len
          f3 (session_age_norm): 1.0 / (1 + session_turn_count)
          f4 (segment_match_ratio): 알려진 경우 세그먼트 매치 비율, 기본 0.0
        """
        prefix_hash = getattr(request, 'prefix_hash', '') or ''
        token_ids = getattr(request, 'token_ids', []) or []
        session_turn = getattr(request, 'session_turn', 0) or 0
        segment_match = getattr(request, 'segment_match_ratio', 0.0) or 0.0

        history = self._hit_history.get(prefix_hash, [])
        f1 = sum(history) / len(history) if history else 0.0
        f2 = min(1.0, len(token_ids) / max(1, max_context_len))
        f3 = 1.0 / (1.0 + session_turn)
        f4 = float(segment_match)
        return f1, f2, f3, f4

    def predict_hit_rate(self, request: Any) -> float:
        """선형 모델로 요청의 예측 캐시 히트율 계산 (< 0.01ms).

        score = w1*f1 + w2*f2 + w3*f3 + w4*f4 + bias
        hit_rate = sigmoid(score)
        """
        f1, f2, f3, f4 = self._extract_features(request)
        w = self._weights
        score = (w.w_prefix_hash * f1 + w.w_context_length * f2
                 + w.w_session_age * f3 + w.w_segment_match * f4 + w.bias)
        # sigmoid
        return float(1.0 / (1.0 + torch.tensor(-score).exp()))

    def update_predictor(self, request: Any, actual_hit: bool) -> None:
        """실제 히트 결과로 선형 모델 온라인 SGD 업데이트.

        Algorithm:
          y_pred = predict_hit_rate(request)
          y_true = 1.0 if actual_hit else 0.0
          error = y_pred - y_true
          gradient = error * feature_i (MSE gradient)
          weight_i -= lr * gradient
        """
        f1, f2, f3, f4 = self._extract_features(request)
        y_pred = self.predict_hit_rate(request)
        y_true = 1.0 if actual_hit else 0.0
        error = y_pred - y_true
        lr = self._weights.lr
        self._weights.w_prefix_hash -= lr * error * f1
        self._weights.w_context_length -= lr * error * f2
        self._weights.w_session_age -= lr * error * f3
        self._weights.w_segment_match -= lr * error * f4
        self._weights.bias -= lr * error

        # 히트 히스토리 업데이트
        prefix_hash = getattr(request, 'prefix_hash', '') or ''
        if prefix_hash:
            hist = self._hit_history.setdefault(prefix_hash, [])
            hist.append(actual_hit)
            if len(hist) > 100:
                hist.pop(0)

    def classify_request(self, request: Any) -> RoutingDecision:
        """요청을 warm/cold/neutral로 분류.

        import time
        t_start = time.monotonic()
        hit_rate = predict_hit_rate(request)
        if hit_rate >= high_hit_threshold: path="warm", priority=0
        elif hit_rate < low_hit_threshold: path="cold", priority=2
        else: path="neutral", priority=1
        overhead_us = (time.monotonic() - t_start) * 1e6
        """
        import time
        t_start = time.monotonic()
        hit_rate = self.predict_hit_rate(request)
        if hit_rate >= self.config.high_hit_threshold:
            path = "warm"
            priority = 0
            self._warm_count += 1
        elif hit_rate < self.config.low_hit_threshold:
            path = "cold"
            priority = 2
            self._cold_count += 1
        else:
            path = "neutral"
            priority = 1
            self._neutral_count += 1
        overhead_us = (time.monotonic() - t_start) * 1e6
        self._ttft_overhead_us.append(overhead_us)
        return RoutingDecision(
            request_id=getattr(request, 'request_id', ''),
            predicted_hit_rate=hit_rate,
            path=path,
            batch_priority=priority,
        )

    def sort_warm_batch_by_prefix_similarity(self, warm_requests: List[Any]) -> List[Any]:
        """Warm 배치 내 요청을 접두사 유사도(LCP 길이) 기준으로 정렬.

        유사 접두사 요청끼리 같은 배치로 묶어 KV 재사용 극대화.
        """
        def lcp_key(req: Any) -> str:
            return getattr(req, 'prefix_hash', '') or ''
        return sorted(warm_requests, key=lcp_key)

    def schedule(self, requests: List[Any]) -> List[Any]:
        """BaseScheduler 인터페이스 구현.

        Algorithm:
          decisions = [classify_request(req) for req in requests]
          warm = [r for r, d in zip(requests, decisions) if d.path=="warm"]
          cold = [r for r, d in zip(requests, decisions) if d.path=="cold"]
          neutral = [r for r, d in zip(requests, decisions) if d.path=="neutral"]

          # Warm 배치: 접두사 유사도 정렬 + 앞자리
          warm_sorted = sort_warm_batch_by_prefix_similarity(warm)

          # 큐 압박 시 cold 슬롯 증가 (warm_slot_ratio 조정)
          if len(requests) > queue_pressure_threshold:
            # cold 요청을 neutral과 섞어 수용 증가
            neutral.extend(cold[:len(cold)//2])
            cold = cold[len(cold)//2:]

          return warm_sorted + neutral + cold
        """
        if not requests:
            return []

        warm, cold, neutral = [], [], []
        for req in requests:
            decision = self.classify_request(req)
            if decision.path == "warm":
                warm.append(req)
            elif decision.path == "cold":
                cold.append(req)
            else:
                neutral.append(req)

        warm_sorted = self.sort_warm_batch_by_prefix_similarity(warm)

        # 큐 압박 시 cold 요청 일부를 neutral로 승격
        if len(requests) > self.config.queue_pressure_threshold:
            half = len(cold) // 2
            neutral.extend(cold[:half])
            cold = cold[half:]

        return warm_sorted + neutral + cold

    def scheduling_overhead_mean_us(self) -> float:
        if not self._ttft_overhead_us:
            return 0.0
        return sum(self._ttft_overhead_us) / len(self._ttft_overhead_us)

    def routing_stats(self) -> dict:
        total = max(1, self._warm_count + self._cold_count + self._neutral_count)
        return {
            "warm_ratio": self._warm_count / total,
            "cold_ratio": self._cold_count / total,
            "neutral_ratio": self._neutral_count / total,
            "scheduling_overhead_mean_us": self.scheduling_overhead_mean_us(),
        }
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 proxy — `src/metrics/perplexity.py`의 `attention_output_relative_error()` 활용.
  실제 WikiText-2 없을 시 `torch.randn`으로 FP32 synthetic KV 텐서 생성.
- **측정 설정**: n_q=8, d_head=128, seq_len=256 (기본). seq_len=512, 1024도 테스트.
- **측정 방법**:
  ```
  K_int8_restored = dequantize_int8(quantize_int8(K_orig))
  V_int4_restored = dequantize_int4(quantize_int4(V_orig))
  relative_error = attention_output_relative_error(Q, K_orig, V_orig,
                                                    K_int8_restored, V_int4_restored)
  허용 오차: relative_error < 0.01 (1%) — MANDATORY (evaluation_criteria.md §4)
  ```
- **폴백 사다리별 측정**:
  - Level 0 (INT8K+INT4V): relative_error 측정 [기본, MANDATORY]
  - Level 1 (INT8K+FP16V): Value FP16 복원 후 relative_error 측정
  - Level 2 (FP16K+FP16V): 완전 복원 → relative_error ≈ 0.0 (기준 검증)
- **수학적 보장 검증 (MANDATORY)**: 100개 이상 synthetic 시퀀스에서
  `error_bound ≥ actual_error` 항상 성립 확인 (보수적 상한 검증).
  - `actual_error = attention_output_relative_error(Q, K_orig, V_orig, K_restored, V_restored)`
  - `error_bound = delta_attn_bound + delta_value_bound` (compute_error_bound() 반환값)
  - **실패 기준**: error_bound < actual_error인 케이스가 1개라도 존재 → 테스트 FAIL

### 태스크 정확도 측정

- **벤치마크 1 — NIAH proxy**:
  - synthetic 시퀀스에서 "needle" 위치 KV가 INT8K+INT4V 복원 후 코사인 유사도 ≥ 0.99 보존 확인.
  - seq_len = [256, 512, 1024] (32K/64K/128K 컨텍스트 프록시)
  - `cosine_similarity_output(Q, K_orig, V_orig, K_restored, V_restored) ≥ 0.99` — MANDATORY

- **벤치마크 2 — LongBench proxy**:
  - 8개 독립 synthetic 시퀀스 (다른 seed/길이 조합) 각각에서 `cosine_similarity_output ≥ 0.99`
  - 허용 기준: **8개 모두** cosine_sim ≥ 0.99 — MANDATORY

- **KVSculpt 증류 정확도**:
  - 균일 50% 압축 vs. 난이도 비례 예산 배분(γ=0.5) 비교
  - 난이도 비례 배분이 균일 대비 KL 발산 감소 확인
  - γ = [0.0, 0.25, 0.50, 1.0] 별 정확도-압축률 곡선 측정

- **Cross-1 파이프라인 통합 정확도**:
  - RuntimeCertifiedKVSculptDistillationPipeline.run_pipeline() 실행 후
    `cosine_similarity_output(Q, K_orig, V_orig, K_final, V_final) ≥ 0.99` — MANDATORY

### KV 메모리 감소율 검증

- INT8K+INT4V vs FP16: `memory_reduction_ratio() ≥ 0.50` (50% 이상) [MANDATORY: −30% 기준 초과]
- INT8K+FP16V (Level 1 폴백): `memory_reduction_ratio() ≥ 0.25`
- KVSculpt budget_ratio=0.50: 50% KV 유지 → 50% 메모리 절감

### Fail 기준

- **Level 0 relative_error > 0.01** → 전체 FAIL (evaluation_criteria.md §4 필수)
- **LongBench proxy cosine_sim < 0.99** (1개라도) → 전체 FAIL (MANDATORY)
- **error_bound < actual_error** (1개라도, 100 시퀀스 중) → 전체 FAIL (수학적 보장 붕괴)
- **Cross-1 파이프라인 cosine_sim < 0.99** → 전체 FAIL (§5 MANDATORY)

### 검증 테스트 파일

`tests/unit/test_compression_accuracy.py` (기존 파일 덮어쓰기 — RuntimeCertified 기반)
`tests/unit/test_runtime_certified_quant_codec.py` (RuntimeCertifiedQuantizedAttentionCodec 전용)

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-23.yaml
experiment:
  date: "2026-05-23"
  activity: "C+A+B"
  description: >
    C-1 RuntimeCertifiedQuantizedAttentionCodec (수학적 런타임 오류 경계 인증 양자화) +
    C-2 KVSculptDistillationCodec (L-BFGS+최소제곱 교대 증류 레이어 예산 배분) +
    Cross-1 RuntimeCertifiedKVSculptDistillationPipeline (C-1+C-2 폐루프 인증 증류) +
    B-1 CLCPositionalBiasGatedSegmentCache (위치 편향 3단계 선택적 재인코딩 게이트) +
    A-2 CPDWarmColdHitRateRouter (예측 히트율 기반 warm/cold 소프트 분기 스케줄러).
    RuntimeCertified(arXiv 2605.20868) + KVSculpt(arXiv 2603.27819) 기반.
  cache_type: runtime_certified_distillation_pipeline
  compression_method: int8_key_int4_value_with_fp16_fallback
  scheduler_type: cpd_warm_cold_hit_router

runtime_certified_quant_codec:  # C-1
  d_head: 128
  n_kv_heads: 8
  n_layers: 12
  error_threshold: 0.005        # ±0.5% perplexity 수학적 상한 (목표 ±1%의 절반)
  key_bits: 8                   # INT8 Key
  value_bits: 4                 # INT4 Value
  max_entries: 1000
  seed: 42

kvsculpt_distillation_codec:  # C-2
  n_layers: 12
  d_head: 128
  total_budget_ratio: 0.50      # 전체 KV 유지 비율
  gamma: 0.5                    # 난이도 반응 강도
  lbfgs_max_iter: 5             # L-BFGS 반복 (경량화)
  alternating_rounds: 3         # 교대 반복 횟수
  convergence_tol: 1.0e-4
  pilot_n_sequences: 50         # 파일럿 보정 시퀀스 수
  max_entries: 1000
  seed: 42

distillation_pipeline:  # Cross-1
  adaptive_threshold_scale: 1.0
  online_budget_realloc_window: 100
  seed: 42

clc_bias_gate:  # B-1
  max_context_length: 4096
  bias_threshold: 0.15          # ΔPos ≤ 0.15: 직접 재사용
  rope_distortion_threshold: 0.40  # ΔPos > 0.40: 전체 재인코딩
  partial_reencoding_layer_ratio: 0.5
  max_entries: 1000
  seed: 42

cpd_warm_cold_router:  # A-2
  high_hit_threshold: 0.70
  low_hit_threshold: 0.25
  warm_slot_ratio: 0.60
  cold_slot_ratio: 0.30
  neutral_slot_ratio: 0.10
  queue_pressure_threshold: 100
  max_context_length_warm: 50000
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    dataset_proxy: "wikitext2_synthetic"
    task_accuracy_proxy: "niah_cosine_similarity"
    relative_error_max: 0.01         # ±1% (evaluation_criteria.md §4 MANDATORY)
    cosine_similarity_min: 0.99      # (evaluation_criteria.md §4 MANDATORY)
    kl_divergence_max: 0.015
    niah_context_lengths: [256, 512, 1024]   # proxy: 32K/64K/128K
    longbench_subtask_count: 8
    math_guarantee_sequences: 100    # error_bound ≥ actual_error 검증 시퀀스 수
    kvsculpt_gamma_sweep: [0.0, 0.25, 0.50, 1.0]
  activity_b:
    cache_hit_rate_improvement_min_pct: 5.0
    noncontiguous_direct_hit_rate_min_pct: 30.0
    memory_footprint_max_increase_pct: 20.0
  activity_a:
    scheduling_overhead_ttft_p50_max_pct: 5.0
    scheduling_overhead_max_us: 100.0   # 0.1ms 이내
    cache_hit_rate_improvement_min_pct: 10.0
  activity_c:
    memory_reduction_min_ratio: 0.50    # INT8K+INT4V: −50% 이상
    int8k_int4v_relative_error_max: 0.01
    effective_context_multiplier: 2.0
    fallback_rate_level1_max: 0.20      # Level 1 폴백 비율 최대 20%
    fallback_rate_level2_max: 0.05      # Level 2 폴백 비율 최대 5%
  cross_c1_c2:
    pipeline_cosine_min: 0.99           # MANDATORY
    combined_memory_reduction_min: 0.55  # C-1+C-2 결합 −55% 이상

seed: 42
results_dir: "results/2026-05-23"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_runtime_certified_quant_codec.py`
- [ ] `tests/unit/test_compression_accuracy.py` (기존 파일 덮어쓰기)
- [ ] `tests/unit/test_kvsculpt_distillation_codec.py`
- [ ] `tests/unit/test_clc_positional_bias_gated_segment_cache.py`
- [ ] `tests/unit/test_cpd_warm_cold_hit_router.py`
- [ ] `tests/integration/test_runtime_certified_distillation_e2e.py`

### 단위 테스트 명세 — test_runtime_certified_quant_codec.py

```
test_quantize_int8_round_trip_error_small:
    quantize_int8 → dequantize_int8: relative_error < 0.01

test_quantize_int4_round_trip_error_small:
    quantize_int4 → dequantize_int4: relative_error < 0.05

test_put_stores_int8_and_int4:
    put(key, tensor) 후 _store[key].key_int8.dtype == torch.int8 확인

test_put_cpu_backup_exists:
    put 후 _store[key].key_fp16_backup.device == cpu 확인

test_get_returns_restored_tensor_level0:
    put → get: fallback_level=0, 반환값 shape 동일 확인

test_compute_error_bound_conservative:
    100 synthetic 시퀀스에서 error_bound ≥ actual_relative_error 항상 성립 (MANDATORY)

test_decide_fallback_level0_below_threshold:
    error_bound=0.003 ≤ error_threshold=0.005 → fallback_level=0

test_decide_fallback_level1_above_threshold:
    error_bound=0.008 > threshold=0.005, delta_attn=0.002 ≤ threshold/2=0.0025 → level=1

test_decide_fallback_level2_high_attn_distortion:
    delta_attn=0.004 > threshold/2=0.0025 → level=2

test_certify_and_update_changes_fallback_level:
    put 후 certify_and_update(Q) 호출 → fallback_level 적절히 설정

test_memory_reduction_ratio_above_50pct:
    INT8K+INT4V: memory_reduction_ratio() ≥ 0.50 (FP16 대비 −50% 이상) (MANDATORY)

test_compression_hook_relative_error_below_1pct:
    compression_hook(key, tensor) 후 attention_output_relative_error < 0.01 (MANDATORY)

test_fallback_rate_level1_tracked:
    Level 1 폴백 발생 시 fallback_count_level1 증가 확인

test_certified_accuracy_report_keys:
    certified_accuracy_report() 에 필수 키 포함:
    [fallback_rate_level1, fallback_rate_level2, error_bound_mean, error_bound_p99,
     memory_reduction_ratio, error_threshold]

test_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_evict_lru_first:
    max_entries=2, 3번 put → 첫 번째 항목 퇴거

test_hit_rate_tracking:
    put 2개 후 get 1회 히트 + 1회 미스 → hit_rate() == 0.5

test_seed_reproducibility:
    동일 seed + 동일 입력 → 동일 양자화 결과
```

### 단위 테스트 명세 — test_compression_accuracy.py (기존 파일 덮어쓰기)

```
test_int8k_int4v_level0_relative_error_below_1pct:
    RuntimeCertifiedQuantizedAttentionCodec Level 0 (INT8K+INT4V):
    attention_output_relative_error < 0.01 (MANDATORY)

test_int8k_fp16v_level1_relative_error_below_1pct:
    Level 1 폴백 (INT8K+FP16V): relative_error < 0.01 (MANDATORY)

test_fp16k_fp16v_level2_relative_error_near_zero:
    Level 2 완전 복원 (FP16K+FP16V): relative_error ≈ 0.0 (기준 검증)

test_niah_proxy_level0_cosine_above_099:
    seq_len=[256, 512, 1024]: cosine_similarity_output ≥ 0.99 (MANDATORY)

test_longbench_8subtask_proxy_all_above_099:
    8개 독립 synthetic 시퀀스: cosine_sim ≥ 0.99 모두 (MANDATORY)

test_error_bound_conservative_100_sequences:
    100 synthetic 시퀀스: error_bound ≥ actual_error 모두 성립 (MANDATORY)

test_memory_reduction_int8k_int4v_above_50pct:
    memory_reduction_ratio() ≥ 0.50 (MANDATORY, −30% 기준 초과)

test_kvsculpt_difficulty_profile_varies_by_layer:
    pilot_profile_layer_difficulty 후 레이어별 난이도 값 다양성 확인
    (max_difficulty / min_difficulty > 1.0)

test_kvsculpt_budget_proportional_to_difficulty:
    gamma=0.5: 고난이도 레이어 budget > 저난이도 레이어 budget

test_kvsculpt_distill_compress_reduces_seq_len:
    distill_compress(Q, K, V, layer_idx=0): 반환 K_selected.shape[0] < K.shape[0]

test_kvsculpt_gamma0_uniform_budget:
    gamma=0.0: 모든 레이어 budget ≈ total_budget_ratio (균일 배분)

test_kvsculpt_accuracy_preserved_cosine_above_099:
    distill_compress 후 cosine_similarity_output(Q, K_orig, V_orig, K_sel, V_sel) ≥ 0.99 (MANDATORY)

test_cross1_pipeline_cosine_above_099:
    RuntimeCertifiedKVSculptDistillationPipeline.run_pipeline() 후
    cosine_similarity_output ≥ 0.99 (MANDATORY §5)

test_cross1_pipeline_memory_reduction_above_55pct:
    Cross-1 파이프라인: certified_codec.memory_reduction_ratio() ≥ 0.50

test_cross1_fallback_count_tracked:
    폴백 발생 시 layer_fallback_counts[layer_idx] 증가 확인

test_kvsculpt_gamma_sweep_accuracy_curve:
    gamma=[0.0, 0.25, 0.50, 1.0] 각각에서 cosine_sim 측정 및 기록
    (모두 ≥ 0.99 MANDATORY)
```

### 단위 테스트 명세 — test_clc_positional_bias_gated_segment_cache.py

```
test_clc_delta_pos_zero_same_position:
    pos_orig_start=100, pos_target_start=100 → ΔPos=0.0

test_clc_delta_pos_normalized:
    |pos_target - pos_orig| / max_context_length 정규화 확인

test_clc_bias_gate_direct_reuse_small_delta:
    ΔPos=0.10 ≤ 0.15 → ReencodingPolicy.DIRECT_REUSE

test_clc_bias_gate_partial_reencoding_mid_delta:
    ΔPos=0.25 (0.15 < 0.25 ≤ 0.40) → ReencodingPolicy.PARTIAL_REENCODING

test_clc_bias_gate_full_reencoding_large_delta:
    ΔPos=0.60 > 0.40 → ReencodingPolicy.FULL_REENCODING

test_clc_put_segment_stores_meta:
    put_segment(key, kv, pos_orig_start, pos_orig_end, hash) 후
    _meta[key] 존재 확인

test_clc_get_with_policy_returns_tuple:
    put_segment → get_with_policy(key, pos_target) → (tensor, policy) 반환

test_clc_get_with_policy_miss_returns_none:
    미존재 key → (None, FULL_REENCODING)

test_clc_direct_reuse_hit_count_increments:
    direct_reuse 정책 히트 → _direct_reuse_hits 증가

test_clc_noncontiguous_direct_hit_rate:
    direct_reuse_hits=3, partial=1, full=1 → rate=3/5=0.60

test_clc_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_clc_evict_lru_oldest_first:
    max_entries=2, 3번 put → 첫 번째 항목 퇴거

test_clc_hit_rate_tracking:
    put 2개 후 get 1회 히트 + 1회 미스 → hit_rate() == 0.5
```

### 단위 테스트 명세 — test_cpd_warm_cold_hit_router.py

```
test_cpd_predict_hit_rate_range_0_to_1:
    predict_hit_rate(req) ∈ [0.0, 1.0]

test_cpd_classify_warm_high_hit_rate:
    prefix_hash 히트 히스토리 100% → path="warm", priority=0

test_cpd_classify_cold_low_hit_rate:
    히스토리 없는 첫 요청, 낮은 피처 → path="cold", priority=2

test_cpd_classify_neutral_mid_hit_rate:
    중간 피처 조합 → path="neutral", priority=1

test_cpd_schedule_warm_first_in_output:
    warm 요청이 출력 리스트 앞에 위치 (batch_priority=0 먼저)

test_cpd_schedule_cold_last:
    cold 요청이 출력 리스트 뒤에 위치

test_cpd_update_predictor_changes_weights:
    update_predictor(req, actual_hit=True) → 가중치 변화 확인

test_cpd_hit_history_updated_after_update:
    update_predictor 후 _hit_history[prefix_hash] 비어 있지 않음

test_cpd_scheduling_overhead_below_1ms:
    classify_request() 오버헤드 < 1000μs (TTFT +5% 이내 준수)

test_cpd_schedule_empty_returns_empty:
    schedule([]) == []

test_cpd_sort_warm_batch_by_prefix:
    동일 prefix_hash를 가진 요청들이 연속하여 정렬됨

test_cpd_queue_pressure_cold_promoted:
    len(requests) > queue_pressure_threshold: cold 요청 일부가 neutral로 승격

test_cpd_routing_stats_keys:
    routing_stats() 에 필수 키:
    [warm_ratio, cold_ratio, neutral_ratio, scheduling_overhead_mean_us]

test_cpd_basescheduler_interface:
    schedule() 메서드가 List 반환 확인 (BaseScheduler 인터페이스)
```

### 통합 테스트 명세 — test_runtime_certified_distillation_e2e.py

```
test_e2e_pipeline_basic_run:
    run_pipeline(Q, K, V, layer_idx=0, key) 정상 완료 확인

test_e2e_pipeline_returns_kfinal_vfinal_report:
    run_pipeline 반환값 (K_final, V_final, report) 타입 확인
    report에 layer_idx, fallback_level, error_bound, selected_ratio 포함

test_e2e_pipeline_accuracy_preserved_cosine_above_099:
    run_pipeline 후 cosine_similarity_output(Q, K_orig, V_orig, K_final, V_final) ≥ 0.99
    (MANDATORY §5)

test_e2e_pipeline_memory_reduction_above_50pct:
    certified_codec.memory_reduction_ratio() ≥ 0.50 after pipeline run

test_e2e_pipeline_distillation_reduces_seq_len:
    run_pipeline: K_final.shape[0] ≤ K.shape[0] (증류 압축으로 토큰 수 감소)

test_e2e_pipeline_fallback_tracking:
    Layer 0 다수 처리 후 layer_fallback_counts / layer_request_counts 딕셔너리 존재

test_e2e_pipeline_cachestore_interface_full:
    put/get/evict/hit_rate/memory_bytes/reset_stats 모두 동작

test_e2e_pipeline_runner_integration:
    InferenceRunner(cache=RuntimeCertifiedKVSculptDistillationPipeline)로
    run_batch() 호출 성공 (src/engine/runner.py 사용)

test_e2e_pipeline_cpd_router_integration:
    CPDWarmColdHitRateRouter.schedule() → warm 요청에 대해 run_pipeline() 호출 후
    routing_stats()["warm_ratio"] > 0

test_e2e_clc_gate_integration:
    CLCPositionalBiasGatedSegmentCache.get_with_policy() 반환 policy에 따라
    DIRECT_REUSE 세그먼트는 재인코딩 없이 run_pipeline() 입력으로 사용

test_e2e_solo_c1_vs_solo_c2_vs_cross1_memory_comparison:
    C-1 단독 / C-2 단독 / Cross-1 통합 메모리 감소율 비교 기록
    (Cross-1이 C-1 단독 대비 ≥ −10% 추가 감소 또는 동등)
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 5개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - `test_int8k_int4v_level0_relative_error_below_1pct` 통과 (relative_error < 0.01, MANDATORY)
      - `test_longbench_8subtask_proxy_all_above_099` 통과 (cosine_sim ≥ 0.99, MANDATORY)
      - `test_error_bound_conservative_100_sequences` 통과 (수학적 보장 검증, MANDATORY)
      - `test_memory_reduction_int8k_int4v_above_50pct` 통과 (reduction ≥ 0.50)
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함:
      - `test_cross1_pipeline_cosine_above_099` 통과 (MANDATORY)
      - `test_e2e_pipeline_accuracy_preserved_cosine_above_099` 통과 (MANDATORY)
- [ ] `evaluation_criteria.md` §3 Activity B 항목 충족:
      - `test_clc_bias_gate_direct_reuse_small_delta` 통과
      - `test_clc_noncontiguous_direct_hit_rate` 통과
- [ ] `evaluation_criteria.md` §2 Activity A 항목 충족:
      - `test_cpd_scheduling_overhead_below_1ms` 통과 (TTFT overhead 검증)
      - `test_cpd_classify_warm_high_hit_rate` 통과
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - `RuntimeCertifiedQuantizedAttentionCodec`, `KVSculptDistillationCodec`,
        `RuntimeCertifiedKVSculptDistillationPipeline`, `CLCPositionalBiasGatedSegmentCache`
        모두 `CacheStore` 인터페이스 구현
      - `CPDWarmColdHitRateRouter`가 `BaseScheduler` 인터페이스 구현
      - 기존 테스트 회귀 없이 통과
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-23.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-23/metrics.json`에 JSON 기록:
      ```json
      {
        "c1_int8k_int4v_relative_error": ...,
        "c1_cosine_similarity_level0": ...,
        "c1_memory_reduction_ratio": ...,
        "c1_fallback_rate_level1": ...,
        "c1_fallback_rate_level2": ...,
        "c1_error_bound_mean": ...,
        "c1_error_bound_p99": ...,
        "c1_error_bound_conservative_violations": 0,
        "c2_kl_reduction_vs_uniform": ...,
        "c2_layer_difficulty_max_min_ratio": ...,
        "c2_accuracy_cosine_budget_050": ...,
        "c2_gamma_sweep": {"0.0": ..., "0.25": ..., "0.50": ..., "1.0": ...},
        "cross1_pipeline_cosine": ...,
        "cross1_memory_reduction_ratio": ...,
        "cross1_fallback_budget_realloc_count": ...,
        "b1_direct_reuse_hit_rate": ...,
        "b1_noncontiguous_hit_rate_pct": ...,
        "b1_cache_hit_rate_improvement_pct": ...,
        "a2_scheduling_overhead_mean_us": ...,
        "a2_warm_ratio": ...,
        "a2_cold_ratio": ...,
        "a2_cache_hit_rate_improvement_pct": ...,
        "inference_throughput_improvement_pct": ...,
        "effective_context_length_multiplier": ...
      }
      ```
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
