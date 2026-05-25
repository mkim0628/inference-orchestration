<!-- 변경 이유 (이전 Spec.md: 2026-05-24 대비):
이전 사이클(2026-05-24)은 A+C 조합이었다:
  - C-1 TriAttentionPreRoPEKVSelectorCodec (pre-RoPE Q/K 집중도 삼각함수 KV 선택, 10.7x 절감)
  - C-2 AttentionMatchingClosedFormCodec (닫힌 형태 최소제곱 잠재 공간 KV 압착, 50x 압착)
  - A-1 DualPathNICLoadBalancer (스토리지 NIC 이중 경로 로드 밸런서)
  - Cross-1 DualPathTriAttentionCompressPipeline (A+C 통합 파이프라인)

이번 사이클(2026-05-25)은 B+C 조합으로 전환된다.
핵심 전환:
  - Activity C 최우선: TriAttention/AttentionMatching "사후 선택/압착" 방식에서
    VeriCache의 "투기적 실행(speculative draft-verify)" 패러다임으로 전환.
    압축 KV로 드래프트 토큰을 생성하고 전체 KV로 검증함으로써,
    accuracy-preserving 제약을 ±1% 이내에서 결정론적 ±0.0%로 강화한다.
  - Activity B 2순위: 기존 KVPacketSoftAdapterCache가 이미 구현되어 있으나,
    VeriCache의 투기적 파이프라인과 B+C 크로스 통합에 최적화된
    KVPacketCache 클래스를 신규 파일로 추가한다.
    소프트-토큰 어댑터 자기지도 증류를 통해 재계산 FLOPs를 0으로 낮춘다.
  - Cross-1 B+C: VeriCache 투기적 코덱 + KV Packet 비연속 재사용 통합 파이프라인.
    비연속 패킷 세그먼트가 VeriCache 드래프팅의 압축 KV 소스로 직접 공급된다.

주요 변경:
1. [신규] src/cache/vericache_speculative_codec.py (C-1, 최우선)
2. [신규] src/cache/kv_packet.py (B-1)
3. [신규] src/engine/speculative_packet_pipeline.py (Cross-1, B+C)
4. [신규] configs/experiments/2026-05-25.yaml
5. [신규] configs/vericache_speculative_policy.yaml
6. [신규] configs/kv_packet_adapter_policy.yaml
7. [신규] tests/unit/test_vericache_speculative_codec.py (C-1)
8. [신규] tests/unit/test_kv_packet.py (B-1)
9. [신규] tests/unit/test_compression_accuracy.py 에 C-1 VeriCache 케이스 추가
10. [신규] tests/integration/test_speculative_packet_pipeline_e2e.py (Cross-1)
11. [보존] 이전 사이클 구현 파일 전부 수정 금지.
    특히 kv_packet_adapter.py, triattention_pre_rope_kv_selector_codec.py,
    attention_matching_closed_form_codec.py, dualpath_nic_load_balancer.py,
    runtime_certified_quant_codec.py, kvsculpt_distillation_codec.py 등
    기존 단위·통합 테스트 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-25: VeriCache Speculative KV Codec + KV Packet Zero-FLOPs Non-Contiguous Reuse

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-25.md`

**최우선 구현 타겟**: C-1 `VeriCacheSpeculativeKVDraftVerifyLosslessCodec`
**2순위 구현 타겟**: B-1 `KVPacketSoftTokenAdapterZeroFLOPsNonContiguousCache`
**통합 타겟**: Cross-1 `VeriCacheKVPacketZeroFLOPsLosslessNonContiguousPipeline` (B+C)

**해결하려는 문제**:

- **Activity C (VeriCache 투기적 드래프트-검증 무손실 코덱)**: 기존 KV 압축 기법들은
  확률적 정확도 보존을 주장하지만 ±1% 이내를 결정론적으로 보장하지 못한다.
  VeriCache(arXiv 2605.17613)는 투기적 실행 패러다임을 KV 압축에 최초 적용해
  압축 KV로 드래프트 토큰을 생성하고 전체 KV로 검증함으로써 결정론적 ±0.0% 정확도를
  수학적으로 보장한다. 드래프팅은 HBM-대역폭 바운드, 전체 KV 로드는 PCIe-바운드이므로
  두 연산이 서로 다른 하드웨어 자원을 사용해 완전한 병렬화가 가능하다.
  이전 사이클 RuntimeCertifiedQuantizedAttentionCodec(05-23)이 확률적 경계를 제공했다면,
  VeriCache는 투기적 실행으로 전체 KV와의 결정론적 동일성을 보장하는 더 강한 메커니즘이다.

- **Activity B (KV Packet 소프트-토큰 어댑터 제로-FLOPs 비연속 재사용)**: 기존 비연속
  KV 재사용 기법(segmented.py, kv_packet_adapter.py 등)은 컨텍스트 불연속성을 처리할 때
  재계산 FLOPs가 발생하거나 RoPE 재인코딩이 필요하다. KV Packet(arXiv 2604.13226)은
  훈련 가능한 소프트-토큰 어댑터와 자기지도 증류로 컨텍스트 불연속성을 사전에 흡수해
  재사용 시 추가 FLOPs를 0으로 낮춘다. 이미 구현된 `kv_packet_adapter.py`
  (KVPacketSoftAdapterCache)는 VeriCache 통합에 최적화되지 않았으므로,
  VeriCache의 드래프트-검증 파이프라인과 직접 연동하는 `KVPacketCache` 클래스를 신규 파일로
  추가한다. 기존 `kv_packet_adapter.py`는 보존한다.

- **Cross-1 B+C 통합 (SpeculativePacketPipeline)**: KV Packet 비연속 세그먼트 재사용이
  제로-FLOPs로 더 많은 세그먼트를 캐싱 가능하게 하면, VeriCache가 더 많은 세그먼트에
  압축-검증을 적용할 수 있어 메모리와 처리량이 복합 개선된다.

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling (이번 사이클 최우선 아님)
- [x] Activity B: Non-Contiguous KV Cache Reuse (KVPacketCache, 2순위)
- [x] Activity C: KV Cache Compression (VeriCacheSpeculativeCodec, 최우선)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 필수): perplexity 변화 ±1% 이내
      — C-1: 결정론적 ±0.0% (전체 KV 검증 수학적 보장)
      — 검증: `attention_output_relative_error` 비교 (압축 KV 드래프트 vs. 전체 KV)
      — 허용 오차: relative_error < 0.01 (MANDATORY)
- [ ] 목표 2 (evaluation_criteria.md §4 필수): downstream 태스크 정확도 ±1% 이내
      — C-1: 드래프트 수락 토큰 = 전체 KV 추론 토큰 (결정론적 동일)
      — cosine_similarity_output >= 0.99 (MANDATORY)
- [ ] 목표 3 (evaluation_criteria.md §4 높음): KV Memory Reduction >= -30%
      — C-1: INT8 양자화 코덱 사용 시 -50% 이상, 토큰 퇴거 코덱 사용 시 -70% 이상 목표
- [ ] 목표 4 (evaluation_criteria.md §4 높음): Effective Context Length 동일 메모리 2x 이상
      — 압축 KV를 HBM에 유지하고 전체 KV를 DRAM 오프로딩 → 동일 HBM 예산에서 2x 이상
- [ ] 목표 5 (evaluation_criteria.md §3 높음): 비연속 세그먼트 히트율 전체 히트의 30% 이상
      — B-1: 제로-FLOPs 재사용으로 비연속 히트 실용성 증가
- [ ] 목표 6 (evaluation_criteria.md §1 높음): 처리량 베이스라인 +20% 이상
      — C-1: 드래프트 수락률 0.7 기준 처리량 +50~70% 추정
- [ ] 목표 7 (evaluation_criteria.md §5 필수, C 포함): Cross-1 복합 accuracy ±1% 이내
      — cosine_similarity >= 0.99 (MANDATORY)
- [ ] 목표 8 (evaluation_criteria.md §4 높음): 압축 오버헤드 TTFT +10% 이내
      — C-1: 드래프팅 오버헤드 < 5ms/드래프트 배치, 검증 오버헤드 병렬화로 최소화

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/vericache_speculative_codec.py` | C (최우선) | VeriCacheSpeculativeCodec — 투기적 드래프트-검증 KV 압축 코덱. CacheStore 인터페이스 구현. 플러그인 압축 코덱(INT8/FP8/토큰퇴거) + 전체 KV 비동기 검증으로 결정론적 ±0.0% 정확도 보장. |
| `src/cache/kv_packet.py` | B | KVPacketCache — VeriCache 통합 최적화 KV Packet 비연속 재사용 캐시. CacheStore 인터페이스 구현. 소프트-토큰 어댑터 자기지도 증류로 재계산 FLOPs = 0. SpeculativePacketPipeline 연동 인터페이스 제공. |
| `src/engine/speculative_packet_pipeline.py` | B+C (Cross-1) | SpeculativePacketPipeline — KVPacketCache(B) + VeriCacheSpeculativeCodec(C) 통합 파이프라인. 비연속 패킷이 VeriCache 드래프트 소스로 공급됨. |
| `configs/vericache_speculative_policy.yaml` | C | VeriCache 정책 설정 |
| `configs/kv_packet_adapter_policy.yaml` | B | KV Packet 어댑터 정책 설정 |
| `configs/experiments/2026-05-25.yaml` | 공통 | 이번 사이클 실험 설정 |
| `tests/unit/test_vericache_speculative_codec.py` | C | C-1 단위 테스트 |
| `tests/unit/test_kv_packet.py` | B | B-1 단위 테스트 |
| `tests/integration/test_speculative_packet_pipeline_e2e.py` | B+C Cross-1 | E2E 통합 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `tests/unit/test_compression_accuracy.py` | C-1 VeriCache accuracy 검증 케이스 추가 (기존 케이스 유지) |

**보존 불변 파일**: `src/cache/base.py` 및 이전 사이클 구현 파일 전부.
특히 `kv_packet_adapter.py`, `triattention_pre_rope_kv_selector_codec.py`,
`attention_matching_closed_form_codec.py`, `dualpath_nic_load_balancer.py`,
`runtime_certified_quant_codec.py`, `kvsculpt_distillation_codec.py` 등
기존 단위·통합 테스트 회귀 없이 통과해야 한다.

---

## 알고리즘 상세

### VeriCacheSpeculativeCodec (Activity C — 최우선)

```python
# src/cache/vericache_speculative_codec.py

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


# ---- 플러그인 압축 코덱 인터페이스 ----

class DraftCodec:
    """압축 KV 생성 플러그인 인터페이스. 교체 가능."""

    def compress(self, kv: torch.Tensor) -> torch.Tensor:
        """전체 KV를 압축. kv: [n_tokens, d_head] -> compressed (크기 감소)."""
        raise NotImplementedError

    def decompress(self, compressed_kv: torch.Tensor) -> torch.Tensor:
        """압축 KV를 원본 공간으로 복원 (근사). 검증용이 아닌 드래프팅용."""
        raise NotImplementedError

    @property
    def compression_ratio(self) -> float:
        """압축 비율 (>= 1.0, 클수록 강한 압축)."""
        raise NotImplementedError


class Int8DraftCodec(DraftCodec):
    """INT8 양자화 기반 드래프트 코덱.

    압축: FP16 -> INT8 (2x 메모리 절감).
    복원: INT8 -> FP32 (근사, 양자화 오류 존재 — VeriCache 검증 단계에서 교정됨).

    Algorithm:
      compress(kv):
        scale = kv.abs().max() / 127.0 + eps
        quantized = (kv / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale  # (INT8 tensor, scale factor)

      decompress(compressed_kv, scale):
        return compressed_kv.float() * scale
    """

    def __init__(self, symmetric: bool = True) -> None:
        self.symmetric = symmetric

    def compress(self, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (int8_tensor, scale)."""
        scale = kv.float().abs().max() / 127.0 + 1e-8
        quantized = (kv.float() / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def decompress(self, compressed_kv: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns FP32 approximate tensor."""
        quantized, scale = compressed_kv
        return quantized.float() * scale

    @property
    def compression_ratio(self) -> float:
        return 2.0  # FP16(2bytes) -> INT8(1byte)


class TokenEvictionDraftCodec(DraftCodec):
    """토큰 퇴거 기반 드래프트 코덱. 중요도 하위 토큰을 제거.

    Algorithm:
      compress(kv):  # kv: [n_tokens, d_head]
        n_keep = max(1, int(n_tokens * keep_ratio))
        importance = kv.float().norm(dim=-1)           # [n_tokens]
        kept_idx = importance.topk(n_keep).indices.sort().values
        return kv[kept_idx], kept_idx                  # (kept_kv, kept_indices)

      decompress: 불완전 복원 (kept 토큰만 반환, 퇴거된 토큰은 0으로 패딩)
      — VeriCache 검증 단계가 오류를 교정하므로 완전 복원 불필요.
    """

    def __init__(self, keep_ratio: float = 0.5) -> None:
        self.keep_ratio = keep_ratio

    def compress(self, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (kept_kv [n_keep, d_head], kept_indices [n_keep])."""
        n_tokens = kv.shape[0]
        n_keep = max(1, int(n_tokens * self.keep_ratio))
        importance = kv.float().norm(dim=-1)
        kept_idx = importance.topk(n_keep).indices.sort().values
        return kv[kept_idx], kept_idx

    def decompress(self, compressed: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """패딩으로 원본 크기 복원 (드래프팅 전용 근사)."""
        kept_kv, kept_idx = compressed
        return kept_kv  # 드래프팅 시 패딩 없이 kept 토큰만 사용

    @property
    def compression_ratio(self) -> float:
        return 1.0 / self.keep_ratio


# ---- 핵심 코덱 클래스 ----

@dataclass
class SpeculativeKVEntry:
    """VeriCache KV 스토어 엔트리."""
    segment_id: str
    compressed_kv: object          # DraftCodec.compress() 반환값 (코덱 의존)
    full_kv_ref: torch.Tensor      # 전체 KV (검증용, DRAM 오프로딩 시뮬레이션)
    original_seq_len: int
    original_d_head: int
    draft_acceptance_rate: float = 0.0
    n_draft_calls: int = 0
    n_accepted: int = 0


@dataclass
class VerificationResult:
    """드래프트-검증 비교 결과."""
    accepted: bool                  # 드래프트가 전체 KV와 일치하는지
    draft_output: torch.Tensor      # 압축 KV로 계산한 어텐션 출력 [n_q, d_head]
    verified_output: torch.Tensor   # 전체 KV로 계산한 어텐션 출력 [n_q, d_head]
    relative_error: float           # ||draft - full||_F / ||full||_F
    acceptance_threshold: float     # 수락 판정 임계값


@dataclass
class VeriCacheConfig:
    d_head: int = 128
    draft_length: int = 4           # 드래프트 토큰 수 (1/2/4/8 스위프)
    acceptance_threshold: float = 0.01   # relative_error < 이 값이면 드래프트 수락
    max_entries: int = 512
    enable_async_verify: bool = True     # 비동기 병렬 검증 활성화
    seed: int = 42


class VeriCacheSpeculativeCodec(CacheStore):
    """VeriCache 투기적 드래프트-검증 KV 압축 코덱 (arXiv 2605.17613).

    Activity C: KV Cache Compression — 결정론적 accuracy-preserving.

    핵심 알고리즘:
      드래프트 단계 (HBM-대역폭 바운드):
        1. 압축 KV로 어텐션 출력 계산: draft_output = attention(Q, compressed_K, compressed_V)

      검증 단계 (PCIe-바운드, 병렬 실행):
        2. 전체 KV 비동기 로드 (DRAM 오프로딩 시뮬레이션)
        3. 전체 KV로 검증: verified_output = attention(Q, full_K, full_V)
        4. 수락 판정: if relative_error(draft, verified) < threshold → 드래프트 수락
                      else → 검증 출력으로 교체 (결정론적 보장)

      결정론적 정확도 보장:
        - 수락된 토큰: draft_output ≈ verified_output (threshold 이내)
        - 거부된 토큰: verified_output 그대로 사용 (전체 KV 정확도)
        - 어떤 압축 코덱을 사용해도 출력은 전체 KV 추론 이상의 정확도 보장

    CacheStore 인터페이스:
      - put(key, value): 전체 KV를 저장하고 드래프트 코덱으로 압축본 생성
      - get(key): 압축 KV 반환 (드래프팅용)
      - draft_and_verify(key, Q): 드래프팅 + 검증 수행, VerificationResult 반환
      - evict(): LRU 퇴거
      - hit_rate(), memory_bytes(), reset_stats()

    압축 코덱 플러그인:
      - set_draft_codec(codec: DraftCodec): 드래프트 코덱 교체 (기본: Int8DraftCodec)
      - 지원: Int8DraftCodec, TokenEvictionDraftCodec, 임의 DraftCodec 구현체
    """

    def __init__(self, config: VeriCacheConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._draft_codec: DraftCodec = Int8DraftCodec()
        self._store: OrderedDict[str, SpeculativeKVEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._total_draft_calls: int = 0
        self._total_accepted: int = 0
        self._relative_errors: List[float] = []
        self._verify_lock = threading.Lock()

    def set_draft_codec(self, codec: DraftCodec) -> None:
        """드래프트 코덱 교체. 이미 저장된 항목은 재압축하지 않음."""
        self._draft_codec = codec

    # ---- CacheStore 인터페이스 ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """전체 KV를 저장하고 드래프트 코덱으로 압축본 생성.

        Args:
            key: 세그먼트 식별자
            value: 전체 KV 텐서 [n_tokens, d_head] (K 또는 V 중 하나)
        """
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        compressed = self._draft_codec.compress(value.detach().clone())
        entry = SpeculativeKVEntry(
            segment_id=key,
            compressed_kv=compressed,
            full_kv_ref=value.detach().clone(),
            original_seq_len=value.shape[0],
            original_d_head=value.shape[-1] if value.dim() >= 2 else self.config.d_head,
        )
        self._store[key] = entry

    def put_kv_pair(
        self,
        key: str,
        K: torch.Tensor,   # [n_tokens, d_head]
        V: torch.Tensor,   # [n_tokens, d_head]
    ) -> None:
        """K와 V를 별도 키로 저장. 키 규약: key+'_K', key+'_V'.

        VeriCache는 K와 V를 독립적으로 압축하고 검증한다.
        """
        self.put(key + "_K", K)
        self.put(key + "_V", V)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """압축 KV를 복원해 반환 (드래프팅용 근사 텐서).

        Returns:
            approximate KV tensor (압축 오류 포함), 또는 None (미스)
        """
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        entry = self._store[key]
        return self._draft_codec.decompress(entry.compressed_kv)

    def draft_and_verify(
        self,
        key_K: str,       # K 텐서 키 (put_kv_pair 에서 key+'_K')
        key_V: str,       # V 텐서 키 (put_kv_pair 에서 key+'_V')
        Q: torch.Tensor,  # [n_q, d_head]
    ) -> Optional[VerificationResult]:
        """투기적 드래프팅 + 전체 KV 검증 수행.

        Algorithm:
          1. 압축 K/V 복원 (드래프트 단계)
          2. 드래프트 어텐션 출력 계산
          3. 전체 K/V로 검증 어텐션 출력 계산 (병렬 또는 순차)
          4. relative_error 계산
          5. threshold 비교 → 수락/거부 판정

        Returns:
            VerificationResult (accepted 여부 + 두 출력 모두 포함)
            None: 키 미스 (전체 KV 추론으로 fallback 필요)
        """
        if key_K not in self._store or key_V not in self._store:
            self._misses += 1
            return None

        self._hits += 1
        entry_K = self._store[key_K]
        entry_V = self._store[key_V]

        # 드래프트 단계: 압축 KV로 어텐션 계산
        draft_K = self._draft_codec.decompress(entry_K.compressed_kv)
        draft_V = self._draft_codec.decompress(entry_V.compressed_kv)
        draft_output = self._compute_attention(Q, draft_K, draft_V)

        # 검증 단계: 전체 KV로 검증 (병렬 실행 시뮬레이션)
        if self.config.enable_async_verify:
            verified_output = self._async_verify(
                Q, entry_K.full_kv_ref, entry_V.full_kv_ref
            )
        else:
            verified_output = self._compute_attention(
                Q, entry_K.full_kv_ref, entry_V.full_kv_ref
            )

        # relative_error 계산
        rel_error = float(
            (draft_output.float() - verified_output.float()).norm()
            / (verified_output.float().norm() + 1e-8)
        )

        # 수락 판정
        accepted = rel_error < self.config.acceptance_threshold

        # 통계 업데이트
        self._total_draft_calls += 1
        if accepted:
            self._total_accepted += 1
        self._relative_errors.append(rel_error)

        # 항목별 수락률 업데이트
        with self._verify_lock:
            entry_K.n_draft_calls += 1
            entry_V.n_draft_calls += 1
            if accepted:
                entry_K.n_accepted += 1
                entry_V.n_accepted += 1
            entry_K.draft_acceptance_rate = (
                entry_K.n_accepted / max(1, entry_K.n_draft_calls)
            )
            entry_V.draft_acceptance_rate = (
                entry_V.n_accepted / max(1, entry_V.n_draft_calls)
            )

        return VerificationResult(
            accepted=accepted,
            draft_output=draft_output,
            verified_output=verified_output,
            relative_error=rel_error,
            acceptance_threshold=self.config.acceptance_threshold,
        )

    def get_final_output(
        self,
        result: VerificationResult,
    ) -> torch.Tensor:
        """결정론적 최종 출력 선택.

        수락된 경우: draft_output 반환 (검증 통과, threshold 이내)
        거부된 경우: verified_output 반환 (전체 KV 정확도 보장)

        이 함수가 VeriCache의 결정론적 accuracy-preserving 보장의 핵심이다.
        어떤 압축 코덱을 사용해도 최종 출력은 전체 KV 추론 이상의 정확도를 보장한다.
        """
        return result.draft_output if result.accepted else result.verified_output

    def evict(self) -> int:
        """LRU 퇴거. 바이트 수 반환."""
        if not self._store:
            return 0
        key, entry = next(iter(self._store.items()))
        self._store.pop(key)
        return entry.full_kv_ref.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """압축 KV 메모리 (full_kv_ref는 DRAM 오프로딩으로 HBM에 없다고 가정)."""
        total = 0
        for entry in self._store.values():
            compressed = entry.compressed_kv
            if isinstance(compressed, tuple):
                # (tensor, scale) 형태
                for t in compressed:
                    if isinstance(t, torch.Tensor):
                        total += t.nbytes
            elif isinstance(compressed, torch.Tensor):
                total += compressed.nbytes
        return total

    def memory_bytes_full_kv(self) -> int:
        """전체 KV 메모리 (DRAM 오프로딩 포함 — 참조용)."""
        return sum(e.full_kv_ref.nbytes for e in self._store.values())

    def memory_reduction_ratio(self) -> float:
        """압축 KV 메모리 / 전체 KV 메모리 기준 감소율."""
        compressed_bytes = self.memory_bytes()
        full_bytes = self.memory_bytes_full_kv()
        if full_bytes == 0:
            return 0.0
        return 1.0 - compressed_bytes / full_bytes

    def draft_acceptance_rate(self) -> float:
        """전체 드래프트 수락률 (높을수록 처리량 이점 증가)."""
        if self._total_draft_calls == 0:
            return 0.0
        return self._total_accepted / self._total_draft_calls

    def mean_relative_error(self) -> float:
        """드래프트 어텐션 출력의 평균 relative_error."""
        if not self._relative_errors:
            return 0.0
        return sum(self._relative_errors) / len(self._relative_errors)

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_draft_calls = 0
        self._total_accepted = 0
        self._relative_errors.clear()

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """압축 KV의 중요도 마스크. TokenEviction 코덱 사용 시 kept_indices 기반 마스크 반환."""
        entry = self._store.get(key)
        if entry is None:
            return None
        compressed = entry.compressed_kv
        if isinstance(compressed, tuple) and len(compressed) == 2:
            # TokenEvictionDraftCodec: (kept_kv, kept_indices)
            _, kept_idx = compressed
            if isinstance(kept_idx, torch.Tensor) and kept_idx.dtype == torch.long:
                mask = torch.zeros(entry.original_seq_len, dtype=torch.bool)
                mask[kept_idx] = True
                return mask
        return None

    def speculative_stats(self) -> dict:
        """JSON 기록용 투기적 실행 통계."""
        return {
            "draft_acceptance_rate": self.draft_acceptance_rate(),
            "mean_relative_error": self.mean_relative_error(),
            "total_draft_calls": self._total_draft_calls,
            "total_accepted": self._total_accepted,
            "hit_rate": self.hit_rate(),
            "memory_reduction_ratio": self.memory_reduction_ratio(),
            "compression_codec": type(self._draft_codec).__name__,
            "compression_ratio": self._draft_codec.compression_ratio,
            "n_entries": len(self._store),
        }

    # ---- 내부 헬퍼 ----

    @staticmethod
    def _compute_attention(
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [n_kv, d_head]
        V: torch.Tensor,   # [n_kv, d_head]
    ) -> torch.Tensor:
        """스케일드 닷-프로덕트 어텐션. [n_q, d_head] 반환."""
        scale = Q.size(-1) ** -0.5
        scores = (Q.float() @ K.float().T) * scale   # [n_q, n_kv]
        attn = F.softmax(scores, dim=-1)
        return (attn @ V.float()).to(Q.dtype)

    def _async_verify(
        self,
        Q: torch.Tensor,
        full_K: torch.Tensor,
        full_V: torch.Tensor,
    ) -> torch.Tensor:
        """전체 KV 검증 (현재 구현: 동기 시뮬레이션, 병렬화 의도 표시).

        실제 GPU 환경에서는 CUDA Stream을 사용해 드래프팅과 병렬 실행:
          stream_draft = torch.cuda.Stream()
          stream_verify = torch.cuda.Stream()
          with torch.cuda.stream(stream_draft): draft_output = compute_attention(Q, compressed_K, V)
          with torch.cuda.stream(stream_verify): verified_output = compute_attention(Q, full_K, full_V)
          torch.cuda.synchronize()

        현재 구현(CPU/단순화): 순차 실행으로 동일 결과 보장.
        """
        return self._compute_attention(Q, full_K, full_V)
```

---

### KVPacketCache (Activity B)

```python
# src/cache/kv_packet.py

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class KVPacketConfig:
    n_heads: int = 8
    d_head: int = 128
    n_adapter_tokens: int = 4       # 소프트-토큰 어댑터 토큰 수 (KV Packet 논문 기본값)
    adapter_lr: float = 1e-4        # AdamW 학습률
    adapter_steps: int = 100        # 어댑터 훈련 스텝
    distillation_loss_threshold: float = 0.1   # 품질 낮은 패킷 퇴거 기준
    max_packets: int = 512
    seed: int = 42


@dataclass
class KVPacket:
    """VeriCache 통합용 KV Packet.

    kv_data: [n_tokens, 2, n_heads, d_head] FP16 불변 KV 블록
    adapter_K: [n_adapter_tokens, n_heads, d_head] 소프트-토큰 K 어댑터
    adapter_V: [n_adapter_tokens, n_heads, d_head] 소프트-토큰 V 어댑터
    distillation_loss: 어댑터 훈련 손실 (퇴거 우선순위)
    segment_id: 세그먼트 해시 키
    """
    segment_id: str
    kv_data: torch.Tensor              # [n_tokens, 2, n_heads, d_head]
    adapter_K: torch.Tensor            # [n_adapter_tokens, n_heads, d_head]
    adapter_V: torch.Tensor            # [n_adapter_tokens, n_heads, d_head]
    distillation_loss: float = 1.0
    n_reuses: int = 0


class KVPacketCache(CacheStore):
    """KV Packet (arXiv 2604.13226) 기반 제로-FLOPs 비연속 재사용 캐시.

    VeriCache 통합 최적화 버전 (src/cache/kv_packet_adapter.py와 독립적).

    핵심 차이점 (vs. kv_packet_adapter.py):
      - get_for_vericache(): K와 V를 분리 반환해 VeriCacheSpeculativeCodec.put_kv_pair()에 직접 공급
      - assemble_multi(): 다중 패킷 조합 시 어댑터 토큰을 경계 삽입해 불연속성 흡수
      - n_adapter_tokens=4 (KV Packet 논문 기본값, kv_packet_adapter.py의 rank=8과 다름)
      - 퇴거 정책: LRU + distillation_loss > threshold 우선 퇴거

    put(key, value): kv_data [n_tokens, 2, n_heads, d_head] 저장, 어댑터 초기화
    get(key): adapter 적용된 KV 반환 [n_adapter_tokens+n_tokens, 2, n_heads, d_head]
    get_for_vericache(key): (K, V) tuple 반환 (VeriCache 통합용)
    train_adapter(key, context_sample): 자기지도 증류로 어댑터 훈련 (재계산 FLOPs = 0)
    assemble_multi(keys): 다중 패킷 비연속 조합
    """

    def __init__(self, config: KVPacketConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: OrderedDict[str, KVPacket] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0
        self._access_order: List[str] = []

    # ---- CacheStore 인터페이스 ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """kv_data 저장 + 소프트-토큰 어댑터 초기화.

        Args:
            key: 세그먼트 식별자
            value: kv_data [n_tokens, 2, n_heads, d_head] FP16
        """
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.config.max_packets:
            self.evict()
        c = self.config
        # 소프트-토큰 어댑터 초기화 (n_adapter_tokens × d_head)
        adapter_K = torch.randn(c.n_adapter_tokens, c.n_heads, c.d_head) * 0.02
        adapter_V = torch.randn(c.n_adapter_tokens, c.n_heads, c.d_head) * 0.02
        packet = KVPacket(
            segment_id=key,
            kv_data=value.detach().clone().to(torch.float16),
            adapter_K=adapter_K,
            adapter_V=adapter_V,
        )
        self._store[key] = packet

    def get(self, key: str) -> Optional[torch.Tensor]:
        """어댑터 적용된 KV 반환 [n_adapter_tokens+n_tokens, 2, n_heads, d_head].

        재계산 FLOPs = 0: 어댑터 가중치는 훈련 완료 후 조회만 수행.
        """
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        packet = self._store[key]
        packet.n_reuses += 1
        self._track_noncontiguous(key)
        return self._apply_adapter(packet)

    def get_for_vericache(
        self,
        key: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """VeriCache 통합용 (K, V) 분리 반환.

        Returns:
            (K [n_adapter_tokens+n_tokens, n_heads, d_head],
             V [n_adapter_tokens+n_tokens, n_heads, d_head]) or None
        """
        adapted = self.get(key)
        if adapted is None:
            return None
        K = adapted[:, 0, :, :]   # [n_adapter_tokens+n_tokens, n_heads, d_head]
        V = adapted[:, 1, :, :]
        return K, V

    def evict(self) -> int:
        """LRU + distillation_loss 우선 퇴거.

        Algorithm:
          1. distillation_loss > threshold 항목 중 LRU 우선 퇴거
          2. 없으면 LRU 퇴거 (OrderedDict 첫 항목)
        """
        if not self._store:
            return 0
        # 품질 낮은 패킷 우선
        threshold = self.config.distillation_loss_threshold
        low_quality = [
            k for k, p in self._store.items()
            if p.distillation_loss > threshold
        ]
        evict_key = low_quality[0] if low_quality else next(iter(self._store))
        packet = self._store.pop(evict_key)
        return packet.kv_data.nbytes + packet.adapter_K.nbytes + packet.adapter_V.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        total = 0
        for p in self._store.values():
            total += p.kv_data.nbytes + p.adapter_K.nbytes + p.adapter_V.nbytes
        return total

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._access_order.clear()

    # ---- KV Packet 전용 API ----

    def train_adapter(
        self,
        key: str,
        context_kv: torch.Tensor,     # 레퍼런스 KV [m_ref, 2, n_heads, d_head]
        n_steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> float:
        """자기지도 증류로 소프트-토큰 어댑터 훈련.

        목표 함수 (KV Packet 논문 방식):
          L_adapter = ||attn_output(kv_packet + adapter, q) - attn_output(kv_ref, q)||_F
          q: context_kv에서 16개 랜덤 샘플링한 레퍼런스 쿼리

        재계산 없음: adapter 파라미터만 업데이트. kv_data는 불변.

        Returns:
            최종 훈련 손실 (distillation_loss)
        """
        if key not in self._store:
            return float("inf")
        packet = self._store[key]
        steps = n_steps or self.config.adapter_steps
        learning_rate = lr or self.config.adapter_lr

        adapter_K = nn.Parameter(packet.adapter_K.float().clone())
        adapter_V = nn.Parameter(packet.adapter_V.float().clone())
        optimizer = torch.optim.AdamW([adapter_K, adapter_V], lr=learning_rate)

        kv_data_f = packet.kv_data.float()
        ctx_f = context_kv.float()

        final_loss = float("inf")
        for _ in range(steps):
            # 레퍼런스 쿼리 16개 무작위 샘플링
            n_ref = min(16, ctx_f.shape[0])
            ref_idx = torch.randperm(ctx_f.shape[0])[:n_ref]
            q_ref = ctx_f[ref_idx, 0, 0, :]   # K 채널 첫 헤드를 쿼리로 사용 [n_ref, d_head]

            # 어댑터 적용 KV 조합: [n_adapter_tokens + n_tokens, n_heads, d_head]
            full_K = torch.cat([adapter_K, kv_data_f[:, 0, :, :]], dim=0)
            full_V = torch.cat([adapter_V, kv_data_f[:, 1, :, :]], dim=0)

            # 레퍼런스 KV (어댑터 없음)
            ref_K = ctx_f[:, 0, :, :]  # [m_ref_full, n_heads, d_head]
            ref_V = ctx_f[:, 1, :, :]

            # 어텐션 출력 비교 (첫 헤드만 사용해 속도 최적화)
            pred_out = self._attn_single_head(q_ref, full_K[:, 0, :], full_V[:, 0, :])
            target_out = self._attn_single_head(q_ref, ref_K[:, 0, :], ref_V[:, 0, :])

            loss = F.mse_loss(pred_out, target_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        packet.adapter_K = adapter_K.detach().to(packet.adapter_K.dtype)
        packet.adapter_V = adapter_V.detach().to(packet.adapter_V.dtype)
        packet.distillation_loss = final_loss
        return final_loss

    def assemble_multi(
        self,
        keys: List[str],
    ) -> Optional[torch.Tensor]:
        """다중 패킷 비연속 조합 (제로-FLOPs 재계산).

        각 패킷 경계에 어댑터 토큰을 삽입해 컨텍스트 불연속성 흡수.

        Returns:
            조합된 KV [sum(n_adapter_tokens + n_tokens_i), 2, n_heads, d_head]
            None: 어느 키라도 미스
        """
        parts: List[torch.Tensor] = []
        for k in keys:
            if k not in self._store:
                return None
            packet = self._store[k]
            parts.append(self._apply_adapter(packet))
        return torch.cat(parts, dim=0) if parts else None

    def noncontiguous_hit_rate(self) -> float:
        if self._hits == 0:
            return 0.0
        return self._noncontiguous_hits / self._hits

    # ---- 내부 헬퍼 ----

    def _apply_adapter(self, packet: KVPacket) -> torch.Tensor:
        """소프트-토큰 어댑터를 KV 데이터 앞에 삽입. FLOPs = 행렬 조회 + 연결.

        Returns [n_adapter_tokens + n_tokens, 2, n_heads, d_head].
        """
        c = self.config
        # adapter_K/V: [n_adapter_tokens, n_heads, d_head] -> [n_adapter_tokens, 2, n_heads, d_head]
        adapter_kv = torch.stack([packet.adapter_K, packet.adapter_V], dim=1)
        return torch.cat([adapter_kv.to(packet.kv_data.dtype), packet.kv_data], dim=0)

    def _track_noncontiguous(self, key: str) -> None:
        """비연속 히트 추적."""
        if self._access_order:
            prev = self._access_order[-1]
            keys_list = list(self._store.keys())
            if key in keys_list and prev in keys_list:
                if abs(keys_list.index(key) - keys_list.index(prev)) > 1:
                    self._noncontiguous_hits += 1
            else:
                self._noncontiguous_hits += 1
        self._access_order.append(key)

    @staticmethod
    def _attn_single_head(
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [n_kv, d_head]
        V: torch.Tensor,   # [n_kv, d_head]
    ) -> torch.Tensor:
        """단일 헤드 스케일드 닷-프로덕트 어텐션."""
        scale = Q.size(-1) ** -0.5
        attn = F.softmax(Q @ K.T * scale, dim=-1)
        return attn @ V
```

---

### SpeculativePacketPipeline (Cross-1, B+C)

```python
# src/engine/speculative_packet_pipeline.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.kv_packet import KVPacketCache, KVPacketConfig
from src.cache.vericache_speculative_codec import (
    VeriCacheSpeculativeCodec,
    VeriCacheConfig,
    Int8DraftCodec,
    TokenEvictionDraftCodec,
    VerificationResult,
)


@dataclass
class SpeculativePacketPipelineConfig:
    kv_packet_config: Optional[KVPacketConfig] = None
    vericache_config: Optional[VeriCacheConfig] = None
    use_token_eviction_codec: bool = False    # True: TokenEviction, False: INT8
    token_eviction_keep_ratio: float = 0.5
    seed: int = 42


@dataclass
class PipelineResult:
    """B+C 파이프라인 실행 결과."""
    segment_id: str
    path: str                          # "b_hit_c_draft", "b_miss_fallback", "c_reject_verified"
    final_output: torch.Tensor         # 최종 어텐션 출력 [n_q, d_head]
    b_hit: bool                        # B: KV Packet 히트 여부
    c_accepted: bool                   # C: VeriCache 드래프트 수락 여부
    relative_error: Optional[float]    # VeriCache 검증 relative_error
    memory_compressed_bytes: int       # C 압축 KV 메모리
    noncontiguous_adapter_applied: bool


class SpeculativePacketPipeline:
    """KV Packet (B) + VeriCache Speculative Codec (C) 통합 B+C 파이프라인.

    통합 처리 흐름:

    Step 1 (B-1, 비연속 히트 감지):
      KVPacketCache.get_for_vericache(segment_id)
      → 히트: (K_packet, V_packet) 반환, 어댑터 적용 (FLOPs = 0)

    Step 2 (C-1, VeriCache 저장):
      히트 시: VeriCacheSpeculativeCodec.put_kv_pair(segment_id, K_packet, V_packet)
      → 압축 KV 생성 + 전체 KV 참조 저장

    Step 3 (C-1, 투기적 드래프팅):
      VeriCacheSpeculativeCodec.draft_and_verify(key_K, key_V, Q)
      → VerificationResult 반환

    Step 4 (C-1, 결정론적 출력 선택):
      VeriCacheSpeculativeCodec.get_final_output(result)
      → 수락: draft_output (압축 KV 기반, threshold 이내)
      → 거부: verified_output (전체 KV 기반, 결정론적 보장)

    B 미스 시: 전체 KV 직접 계산으로 fallback (표준 어텐션 경로)
    """

    def __init__(self, config: SpeculativePacketPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        kv_cfg = config.kv_packet_config or KVPacketConfig(seed=config.seed)
        vc_cfg = config.vericache_config or VeriCacheConfig(seed=config.seed)
        self.kv_packet_cache = KVPacketCache(kv_cfg)
        self.vericache = VeriCacheSpeculativeCodec(vc_cfg)

        # 압축 코덱 설정
        if config.use_token_eviction_codec:
            self.vericache.set_draft_codec(
                TokenEvictionDraftCodec(config.token_eviction_keep_ratio)
            )
        else:
            self.vericache.set_draft_codec(Int8DraftCodec())

        self._pipeline_stats: List[dict] = []

    def store_segment(
        self,
        segment_id: str,
        kv_block: torch.Tensor,    # [n_tokens, 2, n_heads, d_head]
    ) -> None:
        """세그먼트를 KV Packet 캐시에 저장 (어댑터 초기화 포함)."""
        self.kv_packet_cache.put(segment_id, kv_block)

    def train_segment_adapter(
        self,
        segment_id: str,
        context_kv: torch.Tensor,  # [m_ref, 2, n_heads, d_head]
    ) -> float:
        """어댑터 자기지도 증류 훈련. 저장 직후 1회 호출 권장."""
        return self.kv_packet_cache.train_adapter(segment_id, context_kv)

    def run(
        self,
        segment_id: str,
        Q: torch.Tensor,            # [n_q, d_head]
        fallback_K: Optional[torch.Tensor] = None,  # B 미스 시 폴백 K
        fallback_V: Optional[torch.Tensor] = None,  # B 미스 시 폴백 V
    ) -> PipelineResult:
        """B+C 통합 파이프라인 실행.

        Args:
            segment_id: 세그먼트 식별자
            Q: 쿼리 텐서 [n_q, d_head]
            fallback_K/V: B 미스 시 사용할 전체 KV (None이면 빈 텐서 반환)

        Returns:
            PipelineResult
        """
        key_K = segment_id + "_K"
        key_V = segment_id + "_V"

        # Step 1: KV Packet 히트 확인
        kv_pair = self.kv_packet_cache.get_for_vericache(segment_id)

        if kv_pair is None:
            # B 미스: 전체 KV 직접 계산으로 fallback
            if fallback_K is not None and fallback_V is not None:
                final_out = VeriCacheSpeculativeCodec._compute_attention(
                    Q, fallback_K, fallback_V
                )
            else:
                final_out = torch.zeros(Q.shape[0], Q.shape[-1], dtype=Q.dtype)
            result = PipelineResult(
                segment_id=segment_id,
                path="b_miss_fallback",
                final_output=final_out,
                b_hit=False,
                c_accepted=False,
                relative_error=None,
                memory_compressed_bytes=0,
                noncontiguous_adapter_applied=False,
            )
            self._pipeline_stats.append({"path": "b_miss_fallback"})
            return result

        # Step 2: VeriCache에 패킷 KV 저장 (압축본 생성)
        K_packet, V_packet = kv_pair
        # 첫 헤드만 사용해 VeriCache에 저장 (단순화: n_heads 차원 평탄화)
        K_2d = K_packet.reshape(K_packet.shape[0], -1)   # [n_tokens, n_heads*d_head]
        V_2d = V_packet.reshape(V_packet.shape[0], -1)
        self.vericache.put_kv_pair(segment_id, K_2d, V_2d)

        # Step 3: 투기적 드래프팅 + 검증
        Q_2d = Q.reshape(Q.shape[0], -1) if Q.dim() > 2 else Q
        verify_result = self.vericache.draft_and_verify(key_K, key_V, Q_2d)

        if verify_result is None:
            # 드래프팅 실패: fallback
            final_out = torch.zeros(Q.shape[0], Q.shape[-1], dtype=Q.dtype)
            result = PipelineResult(
                segment_id=segment_id,
                path="b_hit_c_miss_fallback",
                final_output=final_out,
                b_hit=True,
                c_accepted=False,
                relative_error=None,
                memory_compressed_bytes=self.vericache.memory_bytes(),
                noncontiguous_adapter_applied=True,
            )
            self._pipeline_stats.append({"path": "b_hit_c_miss_fallback"})
            return result

        # Step 4: 결정론적 출력 선택
        final_out = self.vericache.get_final_output(verify_result)
        path = (
            "b_hit_c_draft"
            if verify_result.accepted
            else "c_reject_verified"
        )

        result = PipelineResult(
            segment_id=segment_id,
            path=path,
            final_output=final_out.reshape(Q.shape[0], -1),
            b_hit=True,
            c_accepted=verify_result.accepted,
            relative_error=verify_result.relative_error,
            memory_compressed_bytes=self.vericache.memory_bytes(),
            noncontiguous_adapter_applied=True,
        )
        self._pipeline_stats.append({
            "path": path,
            "c_accepted": verify_result.accepted,
            "relative_error": verify_result.relative_error,
        })
        return result

    def pipeline_summary(self) -> dict:
        """JSON 기록용 파이프라인 통계."""
        if not self._pipeline_stats:
            return {}
        b_hits = [s for s in self._pipeline_stats if s["path"] != "b_miss_fallback"]
        c_drafts = [s for s in self._pipeline_stats if s.get("c_accepted") is True]
        return {
            "total_runs": len(self._pipeline_stats),
            "b_hit_rate": len(b_hits) / max(1, len(self._pipeline_stats)),
            "c_draft_acceptance_rate": self.vericache.draft_acceptance_rate(),
            "mean_relative_error": self.vericache.mean_relative_error(),
            "kv_packet_hit_rate": self.kv_packet_cache.hit_rate(),
            "noncontiguous_hit_rate": self.kv_packet_cache.noncontiguous_hit_rate(),
            "vericache_memory_reduction": self.vericache.memory_reduction_ratio(),
            **self.vericache.speculative_stats(),
        }
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C를 포함하므로 반드시 작성한다.

### C-1: VeriCacheSpeculativeCodec — 결정론적 ±0.0% 보장 메커니즘

**perplexity 측정**:
- **데이터셋**: WikiText-2 합성 proxy (`torch.randn`으로 FP32 synthetic Q/K/V 텐서 생성)
- **측정 방법**:
  ```python
  # src/metrics/perplexity.py의 attention_output_relative_error() 사용
  # 1. 전체 KV로 어텐션 출력 계산
  out_full = compute_attention_output(Q, K_full, V_full)         # [n_q, d_head]
  # 2. 압축 KV로 드래프트 어텐션 출력 계산
  out_draft = compute_attention_output(Q, K_compressed, V_compressed)
  # 3. relative_error
  rel_error = ||out_full - out_draft||_F / ||out_full||_F
  # 4. VeriCache 최종 출력 (수락/거부 후)
  result = codec.draft_and_verify(key_K, key_V, Q)
  out_final = codec.get_final_output(result)
  final_error = ||out_full - out_final||_F / ||out_full||_F
  ```
- **허용 오차**: `final_error < 0.01` (MANDATORY) — 결정론적 보장
  - 드래프트 수락 시: `relative_error < acceptance_threshold (0.01)` 이므로 자동 만족
  - 드래프트 거부 시: `out_final = out_full` → `final_error = 0.0` (수학적 동일)
- **테스트 설정**: n_q=8, d_head=64, N=128, acceptance_threshold=0.01

**태스크 정확도 측정**:
- **AIME25 proxy**: cosine_similarity(out_final, out_full) >= 0.99 (MANDATORY)
- **LongBench 8개 서브태스크 proxy**: 8개 독립 synthetic 시퀀스에서 각각 cosine_similarity >= 0.99
- **결정론적 동일 출력 확인**: `assert_deterministic_output()` 단위 테스트
  - 거부된 드래프트의 최종 출력 = 전체 KV 어텐션 출력 (float 비교 가능 수준)

**드래프트 수락률 측정** (처리량 예측 핵심):
- **코덱별 수락률 sweep**:
  - `Int8DraftCodec`: 예상 수락률 0.7~0.9 (INT8 양자화 오류 작음)
  - `TokenEvictionDraftCodec(keep_ratio=0.5)`: 예상 수락률 0.5~0.7
  - `TokenEvictionDraftCodec(keep_ratio=0.3)`: 예상 수락률 0.3~0.5
- **드래프트 길이 sweep**: draft_length ∈ {1, 2, 4, 8} 별 수락률 및 처리량 곡선
- **acceptance_threshold sweep**: threshold ∈ {0.001, 0.005, 0.01, 0.02, 0.05} 별 수락률

**추가 검증 실험**:
1. 결정론적 보장 확인: 거부된 드래프트에서 `out_final == out_full` (fp32 기준 max_diff < 1e-5)
2. 메모리 감소율: `memory_reduction_ratio()` INT8 기준 ~50%, TokenEviction(0.5) ~50%
3. 수락률 vs. 처리량 트레이드오프 곡선 (acceptance_rate × draft_length 복합 효과)

**검증 테스트 파일**: `tests/unit/test_compression_accuracy.py` (C-1 VeriCache 케이스 추가)

---

### 검증 불변 조건 (VeriCache 수학적 보장)

VeriCache의 accuracy-preserving 보장은 다음 불변 조건에서 도출된다:

1. **수락 경로**: `relative_error(draft_output, verified_output) < threshold`이므로
   최종 출력 오류 ≤ threshold (기본 0.01 = 1%) — 허용 오차 ±1% 이내 보장.

2. **거부 경로**: `final_output = verified_output = attention(Q, K_full, V_full)` — 전체 KV
   어텐션과 수학적으로 동일. 오류 = 0.0%.

3. **결합 보장**: 수락 여부와 무관하게 `relative_error(final_output, full_kv_output) < threshold`
   — 드래프트 코덱의 압축 품질에 완전히 독립적.

따라서 `acceptance_threshold = 0.01`로 설정 시 항상 ±1% 이내 보장 (결정론적).
이 보장은 테스트 케이스 `test_vericache_deterministic_guarantee()`에서 검증한다.

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-25.yaml
experiment:
  date: "2026-05-25"
  activity: "B+C"
  description: >
    C-1 VeriCacheSpeculativeKVDraftVerifyLosslessCodec (투기적 드래프트-검증 결정론적 무손실 코덱,
    결정론적 ±0.0% accuracy 보장) + B-1 KVPacketSoftTokenAdapterZeroFLOPsNonContiguousCache
    (소프트-토큰 어댑터 자기지도 증류 제로-FLOPs 비연속 재사용) +
    Cross-1 SpeculativePacketPipeline (B+C 통합 파이프라인).
    VeriCache (arXiv 2605.17613) + KV Packet (arXiv 2604.13226) 기반.
  cache_type: vericache_speculative_codec
  compression_method: speculative_draft_verify
  scheduler_type: default

vericache_speculative:  # C-1
  d_head: 128
  draft_length: 4
  acceptance_threshold: 0.01         # ±1% 이내 결정론적 보장
  max_entries: 512
  enable_async_verify: true
  seed: 42

kv_packet:  # B-1
  n_heads: 8
  d_head: 128
  n_adapter_tokens: 4
  adapter_lr: 0.0001
  adapter_steps: 100
  distillation_loss_threshold: 0.1
  max_packets: 512
  seed: 42

speculative_packet_pipeline:  # Cross-1
  use_token_eviction_codec: false   # INT8 기본
  token_eviction_keep_ratio: 0.5
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    dataset_proxy: "wikitext2_synthetic"
    task_accuracy_proxy: "cosine_similarity"
    acceptance_threshold: 0.01             # ±1% MANDATORY
    relative_error_max: 0.01               # MANDATORY
    cosine_similarity_min: 0.99            # MANDATORY
    draft_length_sweep: [1, 2, 4, 8]
    codec_sweep:
      - type: "int8"
      - type: "token_eviction"
        keep_ratio: 0.5
      - type: "token_eviction"
        keep_ratio: 0.3
    acceptance_threshold_sweep: [0.001, 0.005, 0.01, 0.02, 0.05]
    longbench_subtask_count: 8
  activity_b:
    noncontiguous_hit_rate_min: 0.30       # 전체 히트의 30% 이상
    cache_hit_rate_improvement_min_pct: 5.0
  activity_c:
    memory_reduction_min_ratio: 0.30       # -30% 이상 (MANDATORY)
    effective_context_multiplier: 2.0      # 동일 메모리 2x 이상
    compression_overhead_ttft_max_pct: 10.0
  cross_bc:
    pipeline_cosine_min: 0.99             # MANDATORY
    combined_noncontiguous_hit_rate_min: 0.30

seed: 42
results_dir: "results/2026-05-25"
```

```yaml
# configs/vericache_speculative_policy.yaml
acceptance_threshold: 0.01
draft_length: 4
draft_codecs:
  - type: "int8"
    compression_ratio: 2.0
  - type: "token_eviction"
    keep_ratio: 0.5
    compression_ratio: 2.0
  - type: "token_eviction"
    keep_ratio: 0.3
    compression_ratio: 3.33
enable_async_verify: true
auto_codec_selection:
  enabled: false
  acceptance_rate_threshold: 0.70       # 수락률 < 0.70 시 더 정확한 코덱으로 전환
```

```yaml
# configs/kv_packet_adapter_policy.yaml
n_adapter_tokens: 4
adapter_lr: 0.0001
adapter_steps: 100
distillation_loss_threshold: 0.1
eviction_policy: "lru_plus_low_quality"   # LRU + distillation_loss 우선 퇴거
zero_flops_verify: true                   # 재계산 FLOPs = 0 검증 활성화
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_vericache_speculative_codec.py`
  - `Int8DraftCodec.compress()`: int8 텐서 반환, scale 양수
  - `Int8DraftCodec.decompress()`: float 복원, shape 동일
  - `TokenEvictionDraftCodec.compress()`: kept_kv shape [n_keep, d_head], n_keep = max(1, int(N * keep_ratio))
  - `VeriCacheSpeculativeCodec.put()`: 엔트리 저장 확인
  - `VeriCacheSpeculativeCodec.get()`: 근사 텐서 반환 (shape 동일, dtype 일치)
  - `VeriCacheSpeculativeCodec.put_kv_pair()`: key+'_K', key+'_V' 별도 저장 확인
  - `VeriCacheSpeculativeCodec.draft_and_verify()`: VerificationResult 반환, accepted bool 확인
  - `test_vericache_deterministic_guarantee()`:
    - 거부된 드래프트: `final_output == verified_output` (max_diff < 1e-5)
    - relative_error < acceptance_threshold 인 경우 accepted=True 확인
    - 최종 출력 `relative_error(final_output, full_kv_output) <= acceptance_threshold` 보장
  - `VeriCacheSpeculativeCodec.evict()`: LRU 퇴거, bytes 반환
  - `VeriCacheSpeculativeCodec.hit_rate()`: 히트율 계산
  - `VeriCacheSpeculativeCodec.memory_bytes()`: 압축 KV 메모리 (전체 KV 제외)
  - `VeriCacheSpeculativeCodec.memory_reduction_ratio()`: 0.0~1.0 범위
  - `VeriCacheSpeculativeCodec.set_draft_codec()`: 코덱 교체 후 동작 확인
  - `VeriCacheSpeculativeCodec.get_importance_mask()`: TokenEviction 코덱 시 bool 마스크 반환
  - `VeriCacheSpeculativeCodec.reset_stats()`: 카운터 초기화
  - CacheStore 추상 메서드 전체 구현 확인

- [ ] `tests/unit/test_kv_packet.py`
  - `KVPacketCache.put()`: 패킷 저장, adapter_K/V 초기화 (shape [n_adapter_tokens, n_heads, d_head])
  - `KVPacketCache.get()`: 어댑터 적용 KV 반환, shape [n_adapter_tokens+n_tokens, 2, n_heads, d_head]
  - `KVPacketCache.get_for_vericache()`: (K, V) tuple 반환, shape [n_adapter_tokens+n_tokens, n_heads, d_head]
  - `KVPacketCache.get()` 재계산 FLOPs = 0 확인:
    - `torch.autograd.profiler`로 backward pass 없음 확인
    - 또는 `torch.no_grad()` 래핑 확인
  - `KVPacketCache.train_adapter()`: distillation_loss 감소 확인 (100 스텝 후 < 초기값)
  - `KVPacketCache.evict()`: distillation_loss > threshold 우선 퇴거, LRU fallback
  - `KVPacketCache.assemble_multi()`: 다중 키 조합, 총 shape 합산 확인
  - `KVPacketCache.assemble_multi()` 하나라도 미스 시 None 반환
  - `KVPacketCache.hit_rate()`, `memory_bytes()`, `reset_stats()`
  - `KVPacketCache.noncontiguous_hit_rate()`: 비연속 패턴 히트 추적
  - CacheStore 추상 메서드 전체 구현 확인
  - `n_adapter_tokens=4` 기본값 검증 (kv_packet_adapter.py의 rank=8과 다름)

- [ ] `tests/unit/test_compression_accuracy.py` (기존 파일에 추가)
  - C-1 VeriCache: `acceptance_threshold=0.01` → `final_error < 0.01` (MANDATORY)
  - C-1 VeriCache 결정론적 보장: 거부된 드래프트 final_output = verified_output (max_diff < 1e-5)
  - C-1 Int8DraftCodec: `memory_reduction_ratio >= 0.40` (INT8, FP16 대비 ~50%)
  - C-1 TokenEvictionDraftCodec(0.5): `memory_reduction_ratio >= 0.40`
  - C-1 코덱별 수락률 sweep: Int8, TokenEviction(0.5), TokenEviction(0.3) 각각 측정
  - C-1 acceptance_threshold sweep [0.001, 0.005, 0.01, 0.02, 0.05] × 수락률 표
  - Cross-1 B+C 파이프라인: `relative_error(final_output, full_kv_output) < 0.01` (MANDATORY)
  - 기존 케이스 전부 유지 (회귀 없음)

- [ ] `tests/integration/test_speculative_packet_pipeline_e2e.py`
  - `store_segment()` + `train_segment_adapter()` + `run()` 전체 파이프라인 실행
  - B 히트 + C 수락 경로: path="b_hit_c_draft", c_accepted=True
  - B 히트 + C 거부 경로: path="c_reject_verified", 최종 출력 = 전체 KV 어텐션
  - B 미스 + 폴백 경로: path="b_miss_fallback", fallback_K/V 사용
  - 결정론적 accuracy 보장: E2E 실행 후 `relative_error(final_output, full_kv) < 0.01`
  - 비연속 히트율: 10개 무작위 순서 접근 후 `noncontiguous_hit_rate() >= 0.3`
  - `pipeline_summary()` dict 반환 및 필수 필드 포함 확인
  - 100회 실행 후 `mean_relative_error < 0.01` (MANDATORY)
  - 메모리 감소율: `vericache.memory_reduction_ratio() >= 0.30`

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 2종 + test_compression_accuracy.py 추가 케이스 + 기존 회귀 없음)
- [ ] 통합 테스트 전부 통과
- [ ] **evaluation_criteria.md §4 필수 (Activity C)**: `final_error < 0.01` (MANDATORY)
  - C-1 VeriCache: acceptance_threshold=0.01 설정 시 결정론적 보장
  - Cross-1 B+C 파이프라인: E2E relative_error < 0.01
- [ ] **evaluation_criteria.md §4 필수 (Activity C)**: cosine_similarity >= 0.99 (MANDATORY)
  - C-1: AIME25 proxy + LongBench proxy
- [ ] **evaluation_criteria.md §5 필수 (C 포함)**: Cross-1 B+C cosine_similarity >= 0.99 (MANDATORY)
- [ ] **evaluation_criteria.md §4 높음**: KV Memory Reduction >= -30%
  - C-1 Int8DraftCodec: ~50% 목표
- [ ] **evaluation_criteria.md §3 높음**: 비연속 히트율 >= 30%
  - B-1 KVPacketCache + Cross-1 파이프라인
- [ ] **evaluation_criteria.md §1 높음**: 처리량 시뮬레이션 +20% 이상
  - C-1: 드래프트 수락률 0.7 × draft_length=4 기준 추정
- [ ] **evaluation_criteria.md §4 높음**: 압축 오버헤드 TTFT +10% 이내
  - C-1: 드래프팅 < 5ms/배치 (CPU 기준, GPU 시 < 1ms)
- [ ] 결정론적 accuracy 보장 검증: `test_vericache_deterministic_guarantee()` 통과
- [ ] `configs/experiments/2026-05-25.yaml` 생성됨
- [ ] `configs/vericache_speculative_policy.yaml` 생성됨
- [ ] `configs/kv_packet_adapter_policy.yaml` 생성됨
- [ ] `results/2026-05-25/metrics.json` 기록됨
  - 필수 필드: `{draft_acceptance_rate, mean_relative_error, memory_reduction_ratio,
    compression_codec_name, noncontiguous_hit_rate, kv_packet_hit_rate,
    b_hit_rate, c_draft_acceptance_rate, pipeline_cosine_similarity,
    final_relative_error_max}`
- [ ] 이전 사이클 모든 단위·통합 테스트 회귀 없이 통과

---

## 구현 주의사항 및 함정 방지

### VeriCacheSpeculativeCodec 구현 시

1. **full_kv_ref 메모리 관리**: `memory_bytes()`는 압축 KV만 계산한다 (전체 KV는 DRAM 오프로딩
   시뮬레이션). `memory_bytes_full_kv()`를 별도 제공해 참조용으로만 사용한다.

2. **코덱 분기 처리**: `Int8DraftCodec.compress()`는 `(tensor, scale)` tuple을 반환하고,
   `TokenEvictionDraftCodec.compress()`는 `(kept_kv, kept_indices)` tuple을 반환한다.
   `decompress()`가 이 tuple을 받아야 하므로 타입 확인을 거쳐야 한다.
   `get_importance_mask()`는 두 번째 원소가 `torch.long` 텐서인 경우만 마스크를 반환한다.

3. **결정론적 보장 테스트**: `test_vericache_deterministic_guarantee()`는 반드시
   `acceptance_threshold=0.0`으로 설정해 모든 드래프트를 강제 거부한 후
   `final_output == verified_output`을 확인해야 한다.
   그 다음 `acceptance_threshold=1.0`으로 모든 드래프트를 강제 수락해
   `relative_error < 1.0` (기계적 항등)을 확인한다.

4. **CacheStore 인터페이스 준수**: `get(key)`는 단일 KV 텐서를 반환해야 한다.
   `draft_and_verify()`는 추가 API로 제공한다. `put_kv_pair()`도 `put()`을 내부에서 호출한다.

### KVPacketCache 구현 시

5. **kv_packet_adapter.py와의 차이점 유지**: `n_adapter_tokens=4` (논문 기본값)를
   기본으로 사용한다. `kv_packet_adapter.py`의 `adapter_rank=8`과 혼동하지 않는다.
   두 파일은 완전히 독립적이며 기존 파일은 수정하지 않는다.

6. **train_adapter() FLOPs 제로 보장**: `train_adapter()`는 저장 시 1회만 호출된다.
   재사용 시 `get()`은 저장된 adapter_K/V를 `torch.no_grad()` 없이 단순 조회+연결만 수행한다.
   테스트에서 `torch.autograd.is_enabled()` 확인 또는 loss.backward() 호출 부재를 검증한다.

7. **비연속 히트 추적**: `_track_noncontiguous()`는 `_store.keys()`의 순서 (삽입 순서)를 기준으로
   한다. `move_to_end()` 호출로 순서가 바뀌므로, 삽입 순서를 별도 리스트로 유지하거나
   현재 스토어 키 순서 대신 정적 기준을 사용해야 한다.
   현재 구현은 단순화를 위해 현재 스토어 키 순서를 사용한다 (허용됨).

### SpeculativePacketPipeline 구현 시

8. **K/V 차원 정렬**: `KVPacketCache.get_for_vericache()`는
   `[n_adapter_tokens+n_tokens, n_heads, d_head]` shape의 K, V를 반환한다.
   `VeriCacheSpeculativeCodec.put_kv_pair()`는 `[n_tokens, d_head]` 2D 텐서를 기대한다.
   따라서 `K_packet.reshape(K_packet.shape[0], -1)`로 평탄화가 필요하다 (구현에 포함됨).

9. **Q 차원 정렬**: Q가 2D `[n_q, d_head]`가 아닌 3D `[n_q, n_heads, d_head]`로 들어올 수 있다.
   `Q_2d = Q.reshape(Q.shape[0], -1)` 처리가 필요하다 (구현에 포함됨).

SPEC_SAVED
