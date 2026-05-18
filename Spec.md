<!-- 변경 이유 (이전 Spec.md: 2026-05-17 대비):
이전 사이클(2026-05-17)은 A+C 조합이었다:
  - A-1 HMAMultiConnectorCompressionPluginScheduler (HMA 멀티-커넥터 플러그인 메타-스케줄러)
  - C-1 RLAdaptivePrecisionQuantizer (RL 워크로드 온라인 적응 정밀도 양자화기)
  - Cross-1 HMAChainedACPipeline (A+C 통합 파이프라인)

이번 사이클(2026-05-18)은 A+B+C 조합으로 전환하며 설계 축이 완전히 교체된다.

주요 변경:
1. [Activity A 교체] HMAMultiConnectorCompressionPluginScheduler(플러그인 레지스트리) →
   AMPDLazySegmentFetchScheduler(AMPD pull-on-demand 지연 페치 스케줄러).
   세그먼트 재사용 집합 확정 전까지 KV 전송 자체를 보류하는 pull-on-demand 타이밍 제어로,
   이전 모든 A 기법(push/사전-오프로딩)과 근본 원리가 다르다.

2. [Activity B 신규 추가] 이전 사이클에 없던 Activity B를 포함:
   AMPDAdapShotLazyLoadPipeline — 세그먼트 집합 확정 후 로드와
   AdapShot RoPE 재인코딩을 비동기 오버랩하는 3단계 파이프라인.

3. [Activity C 교체] RLAdaptivePrecisionQuantizer(RL 리워드 피드백) →
   DPAttentionAwareCompressionSelector(DP Attention 상태 인식 환경별 압축 선택기).
   단일/멀티 GPU 환경에서 DP Attention KV 중복 제거 상태를 압축 정책 결정 입력으로 사용하는
   최초의 C 기법. 기구현 코덱 3종을 환경 인식 래퍼로 재활용.

4. [Cross 교체] HMAChainedACPipeline(A+C 플러그인 체이닝) →
   AMPDPrefillShareNonContiguousStack(A+B+C 완전 통합 스택).
   5단계 처리 흐름: 메타데이터 선행 전달 → 팬아웃 배포 → 지연 로드·재인코딩 →
   환경 인식 압축 → 비연속 어텐션 계산.

5. [보존 파일] 기존 모든 파일(rl_adaptive_precision_quantizer.py,
   hma_multi_connector_scheduler.py, hma_chained_ac_pipeline.py 등)은
   이번 사이클에서 수정하지 않는다. 기존 단위·통합 테스트가 회귀 없이 통과해야 한다.

6. [인터페이스 유지] src/cache/base.py는 이번 사이클에서 수정하지 않는다.
   CacheStore 6개 추상 메서드를 모든 신규 구현체가 완전 구현한다.

7. [Activity C 필수] DPAttentionAwareCompressionSelector는 accuracy-preserving
   검증 계획(WikiText-2 perplexity ±1% + LongBench 8개 서브태스크 + 환경별 교차 검증)
   없이 완성 불가.
-->

# Spec — 2026-05-18

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-18.md`

**최우선 구현 타겟**:
- **A-1 (주)**: AMPDLazySegmentFetchScheduler — AMPD pull-on-demand 원칙 기반
  세그먼트 집합 확정 전 KV 전송 보류 + 확정 후 비동기 pull 스케줄러
- **B-1 (주)**: AMPDAdapShotLazyLoadPipeline — 세그먼트 집합 확정 후
  로드-AdapShot RoPE 재인코딩 비동기 오버랩 3단계 파이프라인 (CacheStore 구현)
- **C-1 (주)**: DPAttentionAwareCompressionSelector — DP Attention 환경 상태 인식
  압축 정책 자동 선택기 (단일/멀티 GPU 환경별 최적 코덱 선택)
- **Cross-1 (주)**: AMPDPrefillShareNonContiguousStack — A-1+B-1+C-1 5단계
  완전 통합 스택

**해결하려는 문제**:

- **Activity A**: 기존 모든 A 기법이 스케줄러가 세그먼트 재사용 결정 확정 여부와
  관계없이 KV 블록을 선제 전송(push) 또는 사전 오프로딩하는 방식으로 불필요 전송이
  발생한다. AMPDLazySegmentFetchScheduler는 AMPD(2602.14516) "KV 지연 읽기" 원칙을
  비연속 세그먼트 경로에 적용해, 세그먼트 재사용 집합이 Louver 탐색으로 확정된 시점
  이후에만 pull-on-demand 방식으로 KV를 읽는다. 메타데이터만 먼저 전달하고 KV 데이터
  전송은 확정 후로 보류함으로써 불필요 전송 −40~65%를 달성한다.

- **Activity B**: 기존 모든 B 기법은 세그먼트가 이미 메모리에 있다고 가정하고 재사용
  결정을 내렸다. AMPDAdapShotLazyLoadPipeline은 Stage 1(세그먼트 탐색·확정) →
  Stage 2(비동기 로드) → Stage 3(AdapShot RoPE 재인코딩 오버랩)의 3단계 비동기
  파이프라인으로 로드와 재인코딩을 오버랩해 직렬 지연을 병렬 지연(이론적 최솟값)으로
  줄인다. CacheStore 인터페이스를 완전 구현한다.

- **Activity C**: 기존 모든 C 기법이 단일 GPU 환경을 암묵적으로 가정하거나 멀티 GPU
  환경에서의 DP Attention KV 중복 제거 상태를 고려하지 않았다.
  DPAttentionAwareCompressionSelector는 SGLang v0.5.11 DP Attention이 N-GPU 환경에서
  KV 복사본을 1/N으로 줄이는 상태를 압축 정책 결정 입력으로 사용해,
  단일 GPU(고압축 코덱 우선)와 멀티 GPU+DP Attention(한계 효용 기반 선택적 압축)에서
  각각 최적 코덱을 자동 선택한다. 기구현 코덱 3종을 래퍼로 재활용.

- **Cross A+B+C**: AMPDPrefillShareNonContiguousStack은 5단계 처리 흐름을 통해
  지연 스케줄링(A-1) → 지연 로드·재인코딩 파이프라인(B-1) → 환경 인식 압축(C-1)의
  복합 효과(처리량 +35~55%, 메모리 −70~95%, accuracy delta ±0.6% 이내)를 달성한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling (AMPDLazySegmentFetchScheduler)
- [x] Activity B: Non-Contiguous KV Cache Reuse (AMPDAdapShotLazyLoadPipeline)
- [x] Activity C: KV Cache Compression (DPAttentionAwareCompressionSelector)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — WikiText-2 proxy: attention_output_relative_error < 0.01
      — 단일 GPU / DP Attention 활성화 멀티 GPU 환경 각각 독립 측정
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      — LongBench 8개 서브태스크 proxy: KL divergence < 0.015, cosine >= 0.99
      — {DP Attention ON/OFF} × {압축 기법} 교차 행렬 검증
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C): KV Cache Memory Reduction >= −30%
      — 단일 GPU: 고압축 코덱 적용 시 −50~70% 목표
      — 멀티 GPU + DP Attention: 이중 절감 곱 효과 −70~95% 이론 검증
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C): Effective Context Length 동일 메모리 2× 이상
      — 압축 + DP Attention 결합 시 유효 KV 크기 측정
- [ ] 목표 5 (evaluation_criteria.md §2 Activity A): 스케줄링 오버헤드 TTFT p50 +5% 이내
      — 메타데이터 선행 전달 오버헤드 < 0.1ms/요청
      — unnecessary_transfer_ratio 지표 수집 및 검증
- [ ] 목표 6 (evaluation_criteria.md §2 Activity A): 캐시 히트율 향상 +10%p
      — 지연 페치로 확정된 세그먼트만 로드 → 불필요 로드 −35~55%
- [ ] 목표 7 (evaluation_criteria.md §3 Activity B): 비연속 세그먼트 히트율 전체 히트의 30% 이상
      — AMPDAdapShotLazyLoadPipeline의 noncontiguous_hit_rate() >= 0.30
- [ ] 목표 8 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +20% 이상
      — 지연 페치 + 로드-재인코딩 오버랩 + 환경 인식 압축 복합 효과
- [ ] 목표 9 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후 accuracy ±1% 이내
      — AMPDPrefillShareNonContiguousStack 5단계 통합 적용 후 cosine >= 0.99
      — 단독 A-1 / 단독 B-1 / 단독 C-1 / 결합 Cross-1 4방향 비교

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/scheduler/ampd_lazy_segment_fetch.py` | A | AMPDLazySegmentFetchScheduler: pull-on-demand 지연 페치 스케줄러 + SegmentMetadataRegistry |
| `src/cache/ampd_adapshot_lazy_pipeline.py` | B | AMPDAdapShotLazyLoadPipeline: 3단계 비동기 파이프라인 (resolve→load→reencode), CacheStore 구현 |
| `src/cache/dp_attention_aware_compression.py` | C | DPAttentionAwareCompressionSelector: DP Attention 상태 인식 환경별 압축 정책 선택기 |
| `src/engine/ampd_prefill_share_stack.py` | A+B+C | AMPDPrefillShareNonContiguousStack: 5단계 완전 통합 스택 |
| `tests/unit/test_ampd_lazy_segment_fetch.py` | A | Activity A 단위 테스트 |
| `tests/unit/test_ampd_adapshot_lazy_pipeline.py` | B | Activity B 단위 테스트 |
| `tests/unit/test_dp_attention_aware_compression.py` | C | Activity C 단위 테스트 (accuracy-preserving 검증 필수) |
| `tests/unit/test_ampd_prefill_share_stack.py` | A+B+C | Cross A+B+C 통합 단위 테스트 |
| `tests/integration/test_cross_abc_ampd_stack.py` | A+B+C | E2E 통합 테스트: 다중 요청 지연 페치 + 파이프라인 + 압축 |
| `configs/experiments/2026-05-18.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| (없음) | `src/cache/base.py` 및 기존 모든 파일 변경하지 않음 |

---

## 알고리즘 상세

### SegmentMeta 데이터클래스 (공통)

```python
# src/scheduler/ampd_lazy_segment_fetch.py 상단
from dataclasses import dataclass
from typing import Literal

SegmentTier = Literal["HBM", "DDR", "REMOTE"]

@dataclass
class SegmentMeta:
    segment_id: str          # 세그먼트 content hash (chunk_key와 동일 형식)
    source_node_id: str      # 세그먼트가 있는 노드 식별자 ("local" 또는 IP)
    tier: SegmentTier        # HBM / DDR / REMOTE
    approx_size_bytes: int   # KV 텐서 예상 크기 (bytes)
    position_range: tuple    # (start_token_idx, end_token_idx)
```

---

### SegmentMetadataRegistry (Activity A)

```python
# src/scheduler/ampd_lazy_segment_fetch.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional

class SegmentMetadataRegistry:
    """세그먼트 ID → SegmentMeta 매핑을 관리하는 경량 레지스트리.

    역할:
      - 요청 수신 시 KV 데이터 없이 메타데이터만 등록
      - 스케줄 확정 후 pull 대상 세그먼트 조회
      - 불필요 전송 비율(unnecessary_transfer_ratio) 추적

    단일 노드 환경: source_node_id = "local", gRPC 스트림 없이 in-process 동작.
    멀티 노드 환경: register_remote() 호출로 외부 노드 세그먼트 등록.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, SegmentMeta] = {}
        self._pre_resolved_count: int = 0   # 후보로 등록된 세그먼트 수
        self._cancelled_count: int = 0      # 확정 전 취소된 세그먼트 수

    def register(self, meta: SegmentMeta) -> None:
        """세그먼트 메타데이터 등록 (KV 데이터 없음)."""
        self._registry[meta.segment_id] = meta
        self._pre_resolved_count += 1

    def get(self, segment_id: str) -> Optional[SegmentMeta]:
        """세그먼트 메타데이터 조회. 없으면 None."""
        return self._registry.get(segment_id)

    def cancel(self, segment_id: str) -> None:
        """확정 전 취소 (pull하지 않기로 결정된 세그먼트)."""
        if segment_id in self._registry:
            self._cancelled_count += 1

    def unnecessary_transfer_ratio(self) -> float:
        """불필요 전송 비율 = 취소된 세그먼트 / 사전 등록된 후보 세그먼트."""
        if self._pre_resolved_count == 0:
            return 0.0
        return self._cancelled_count / self._pre_resolved_count

    def reset_stats(self) -> None:
        self._pre_resolved_count = 0
        self._cancelled_count = 0
```

---

### AMPDLazySegmentFetchScheduler (Activity A)

스케줄링 결정 단위: **요청(request) 단위**.
캐시 상태 접근: `SegmentMetadataRegistry`를 통해 세그먼트 메타데이터 조회 (KV 데이터 접근 없음).

```python
# src/scheduler/ampd_lazy_segment_fetch.py

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional
import torch

from src.cache.segmented import SegmentedHashCache
from src.engine.runner import InferenceRequest


@dataclass
class KVSegment:
    segment_id: str
    kv_tensor: torch.Tensor   # 실제 KV 데이터
    source_tier: str          # "HBM" | "DDR" | "REMOTE"
    load_latency_ms: float    # 로드 소요 시간


@dataclass
class AMPDLazySchedulerConfig:
    # 단일 노드 기본값 (멀티 노드 확장 시 remote_fetch_latency_ms 조정)
    hbm_fetch_latency_ms: float = 0.01
    ddr_fetch_latency_ms: float = 0.5
    remote_fetch_latency_ms: float = 5.0
    metadata_overhead_max_ms: float = 0.1   # 메타데이터 전달 오버헤드 상한
    max_concurrent_fetches: int = 8
    seed: int = 42


class AMPDLazySegmentFetchScheduler:
    """AMPD pull-on-demand 원칙 기반 비연속 세그먼트 지연 페치 스케줄러.

    Activity A: KV Cache-aware Scheduling
    스케줄링 결정 단위: 요청 단위 — 세그먼트 집합 확정 후에만 KV pull.
    캐시 상태 접근: SegmentMetadataRegistry (메타데이터 전용, KV 접근 없음).

    처리 흐름:
      1. pre_resolve_segments(): 요청에서 후보 세그먼트 메타데이터를 등록.
         KV 데이터 전송 없음.
      2. confirm_segments(): Louver 탐색 결과로 재사용 집합 S_reuse 확정.
         확정되지 않은 후보를 cancel() 처리.
      3. fetch_segments_lazy(): 확정된 세그먼트만 비동기 pull.

    평가 기준 (evaluation_criteria.md §2):
      - 스케줄링 오버헤드: TTFT p50 +5% 이내
      - 메타데이터 선행 전달 오버헤드: < 0.1ms/요청
      - unnecessary_transfer_ratio 지표: results/<exp-name>/metrics.json에 기록
    """

    def __init__(
        self,
        config: AMPDLazySchedulerConfig,
        registry: Optional[SegmentMetadataRegistry] = None,
        cache: Optional[SegmentedHashCache] = None,
    ) -> None:
        self.config = config
        self.registry = registry or SegmentMetadataRegistry()
        self._cache = cache
        self._scheduling_times: List[float] = []

    def pre_resolve_segments(
        self,
        request: InferenceRequest,
        candidate_segment_ids: List[str],
        metas: List[SegmentMeta],
    ) -> None:
        """Stage 0: 요청 수신 시 후보 세그먼트 메타데이터 등록.

        Algorithm:
          - 각 (segment_id, SegmentMeta) 쌍을 registry.register() 호출.
          - KV 데이터 전송 없음.
          - 오버헤드 측정: < metadata_overhead_max_ms/요청 (필수).
        """
        t0 = time.monotonic()
        for meta in metas:
            self.registry.register(meta)
        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)

    def confirm_segments(
        self,
        candidate_ids: List[str],
        confirmed_ids: List[str],
    ) -> None:
        """Stage 1 완료 후: 확정 세그먼트 결정, 나머지 cancel 처리.

        Algorithm:
          - confirmed_set = set(confirmed_ids)
          - candidate_ids 중 confirmed_set에 없는 항목 → registry.cancel()
          - 이 시점에서 불필요 전송이 차단됨.
        """
        confirmed_set = set(confirmed_ids)
        for seg_id in candidate_ids:
            if seg_id not in confirmed_set:
                self.registry.cancel(seg_id)

    async def fetch_segments_lazy(
        self,
        confirmed_segment_ids: List[str],
    ) -> AsyncIterator[KVSegment]:
        """Stage 2: 확정 세그먼트 비동기 pull.

        Algorithm:
          1. confirmed_segment_ids 각각에 대해 registry.get() → SegmentMeta 조회.
          2. tier별 fetch 시뮬레이션:
             - HBM: 즉시 반환 (hbm_fetch_latency_ms)
             - DDR: asyncio.sleep 기반 지연 후 반환 (ddr_fetch_latency_ms)
             - REMOTE: asyncio.sleep 기반 원격 지연 후 반환 (remote_fetch_latency_ms)
          3. 각 세그먼트 fetch 완료 시 KVSegment yield.
          4. cache가 있으면 cache.get(segment_id)로 실제 KV 로드 시도.

        Args:
            confirmed_segment_ids: 확정된 세그먼트 ID 목록

        Yields:
            KVSegment — 개별 세그먼트 KV 로드 완료 이벤트
        """
        for seg_id in confirmed_segment_ids:
            meta = self.registry.get(seg_id)
            latency = self.config.hbm_fetch_latency_ms
            if meta is not None:
                if meta.tier == "DDR":
                    latency = self.config.ddr_fetch_latency_ms
                elif meta.tier == "REMOTE":
                    latency = self.config.remote_fetch_latency_ms

            await asyncio.sleep(latency / 1000.0)

            # 실제 KV 텐서: 캐시에서 로드 또는 합성 텐서
            kv_tensor: Optional[torch.Tensor] = None
            if self._cache is not None:
                kv_tensor = self._cache.get(seg_id)
            if kv_tensor is None:
                kv_tensor = torch.zeros(1, 64)  # fallback: 합성 텐서

            yield KVSegment(
                segment_id=seg_id,
                kv_tensor=kv_tensor,
                source_tier=meta.tier if meta else "HBM",
                load_latency_ms=latency,
            )

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """InferenceRunner.run_batch() 호환 schedule() 인터페이스.

        Algorithm:
          - 기본 FIFO 순서 유지 (히트율 예측 정렬 확장 가능).
          - 각 요청의 메타데이터 선행 전달 오버헤드 측정.

        Returns:
            List[InferenceRequest] — 정렬된 요청 목록 (이번 구현: 입력 순서 유지)
        """
        t0 = time.monotonic()
        result = list(requests)
        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)
        return result

    def scheduling_overhead_ms_p50(self) -> float:
        """스케줄링 오버헤드 중앙값 (ms)."""
        if not self._scheduling_times:
            return 0.0
        sorted_t = sorted(self._scheduling_times)
        return sorted_t[len(sorted_t) // 2]

    def unnecessary_transfer_ratio(self) -> float:
        """불필요 전송 비율 (results/<exp>/metrics.json에 기록)."""
        return self.registry.unnecessary_transfer_ratio()

    def reset_stats(self) -> None:
        self._scheduling_times.clear()
        self.registry.reset_stats()
```

---

### AMPDAdapShotLazyLoadPipeline (Activity B)

CacheStore 인터페이스를 완전 구현하며 3단계 비동기 파이프라인을 제공한다.

```python
# src/cache/ampd_adapshot_lazy_pipeline.py

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class LazyPipelineConfig:
    chunk_size: int = 128
    max_entries: int = 1000
    rope_theta: float = 10000.0      # RoPE 기본 주파수 (AdapShot 재인코딩용)
    n_heads: int = 8
    d_head: int = 64
    # 예측 프리페치: 동반 세그먼트 히트 임계값
    companion_hit_threshold: int = 2
    seed: int = 42


class AMPDAdapShotLazyLoadPipeline(CacheStore):
    """AMPD 지연 로드 + AdapShot RoPE 재인코딩 비동기 오버랩 파이프라인.

    Activity B: Non-Contiguous KV Cache Reuse
    CacheStore 인터페이스 완전 구현.

    3단계 비동기 파이프라인:
      Stage 1 (Segment Resolution): resolve_segments()로 히트 보장 세그먼트 집합 확정.
        KV 데이터를 메모리로 읽지 않고 세그먼트 ID + source_tier만 결정.
      Stage 2 (Lazy Load): 확정 후 비동기 로드.
        asyncio.Event 기반 세그먼트별 완료 신호.
      Stage 3 (RoPE Reencoding Overlap): 세그먼트 로드 완료 이벤트 즉시 재인코딩 시작.
        Stage 2와 오버랩: 전체 지연 = max(로드 지연, 재인코딩 지연).

    AdapShot RoPE 재인코딩:
      원본 위치 [pos_start, pos_end] → 타겟 위치 [target_start, target_end]
      Δθ = target_position - source_position
      torch.einsum 기반 배치 회전 연산 (FP32 내부 계산 후 FP16 반환).

    평가 기준 (evaluation_criteria.md §3):
      - 비연속 세그먼트 히트율: 전체 히트의 30% 이상
      - KV Memory Footprint: 베이스라인 대비 +20% 이내
    """

    def __init__(self, config: LazyPipelineConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._store: SegmentedHashCache = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        # 동반 세그먼트 통계: {(seg_a, seg_b): co_hit_count}
        self._companion_stats: Dict[Tuple[str, str], int] = {}
        # 로드 완료 이벤트: {segment_id: asyncio.Event}
        self._load_events: Dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """표준 put — compression_hook 없이 저장 (Stage 2 lazy load 이후 호출됨)."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """표준 get — HBM 직접 조회."""
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self._store.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._store.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._store.reset_stats()
        self._load_events.clear()

    # ------------------------------------------------------------------ #
    # Stage 1: Segment Resolution                                          #
    # ------------------------------------------------------------------ #

    def resolve_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List["SegmentMeta"], List[int]]:
        """Stage 1: 히트 보장 세그먼트 메타데이터 수집 (KV 미로드).

        Algorithm:
          1. SegmentedHashCache.get_segments()로 모든 청크 히트/미스 판정.
          2. 히트 청크: SegmentMeta(segment_id, tier="HBM", ...) 반환.
          3. 미스 청크: miss_indices 반환.
          4. 이 단계에서 KV 텐서를 반환하지 않음 — 메타데이터만.

        Returns:
            (hit_metas, miss_chunk_indices)
        """
        # SegmentedHashCache는 get_segments에서 hit/miss 판정만 수행
        hits, misses = self._store.get_segments(token_ids, layer_idx)

        from src.scheduler.ampd_lazy_segment_fetch import SegmentMeta
        hit_metas = []
        chunk_size = self.config.chunk_size
        for chunk_idx, _kv in hits:
            seg_id = self._store.chunk_key(token_ids, chunk_idx, layer_idx)
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(token_ids))
            hit_metas.append(SegmentMeta(
                segment_id=seg_id,
                source_node_id="local",
                tier="HBM",
                approx_size_bytes=chunk_size * self.config.d_head * 2 * 2,  # FP16 2bytes
                position_range=(start, end),
            ))

        return hit_metas, misses

    # ------------------------------------------------------------------ #
    # Stage 2 + 3: Lazy Load + RoPE Reencoding Overlap                    #
    # ------------------------------------------------------------------ #

    async def load_and_reencode_segment(
        self,
        segment_id: str,
        source_position: int,
        target_position: int,
    ) -> Optional[torch.Tensor]:
        """Stage 2+3 오버랩: 세그먼트 로드 완료 즉시 AdapShot RoPE 재인코딩.

        Algorithm:
          1. self._store에서 segment_id KV 로드.
          2. 로드 완료 시 asyncio.Event 설정.
          3. AdapShot RoPE 재인코딩: Δθ = target_position - source_position.
          4. 재인코딩 완료 KV 반환.

        Args:
            segment_id: 캐시 키
            source_position: KV가 저장될 때의 시작 토큰 위치
            target_position: 현재 컨텍스트에서의 타겟 시작 위치

        Returns:
            RoPE 재인코딩된 KV 텐서 (FP16) 또는 None
        """
        kv = self._store.get(segment_id)
        if kv is None:
            return None

        # asyncio.Event 완료 신호
        event = self._load_events.setdefault(segment_id, asyncio.Event())
        event.set()

        # AdapShot RoPE 재인코딩 (Stage 3, Stage 2와 오버랩)
        if source_position != target_position:
            kv = self._adapshot_rope_reencode(kv, source_position, target_position)

        return kv

    def _adapshot_rope_reencode(
        self,
        kv: torch.Tensor,
        source_pos: int,
        target_pos: int,
    ) -> torch.Tensor:
        """AdapShot RoPE 위상 오프셋 재인코딩.

        Algorithm:
          1. delta = target_pos - source_pos
          2. d_head 차원에서 짝수/홀수 쌍으로 회전 행렬 구성.
          3. kv의 마지막 차원에 회전 적용.
          4. torch.einsum 기반 배치 회전 연산 (O(S × n_heads × d_head)).

        kv shape: [n_tokens, ...] — 마지막 dim = d_head (또는 d_head의 배수).
        반환: 동일 shape, dtype=float16.
        """
        d = kv.shape[-1]
        delta = float(target_pos - source_pos)
        # RoPE 주파수 계산
        half_d = d // 2
        inv_freq = 1.0 / (
            self.config.rope_theta ** (
                torch.arange(0, half_d, dtype=torch.float32) * 2.0 / d
            )
        )  # [half_d]
        angle = delta * inv_freq  # [half_d]
        cos_a = torch.cos(angle)  # [half_d]
        sin_a = torch.sin(angle)  # [half_d]

        kv_f = kv.detach().float()
        # 마지막 차원 분리 (짝수/홀수 쌍)
        flat = kv_f.reshape(-1, d)  # [N, d]
        x1 = flat[..., :half_d]    # [N, half_d]
        x2 = flat[..., half_d:]    # [N, half_d]

        # 회전: x1' = x1*cos - x2*sin, x2' = x1*sin + x2*cos
        rotated = torch.cat([
            x1 * cos_a - x2 * sin_a,
            x1 * sin_a + x2 * cos_a,
        ], dim=-1)  # [N, d]

        return rotated.reshape(kv.shape).half()

    # ------------------------------------------------------------------ #
    # 세그먼트-레벨 API (SegmentedHashCache 위임)                          #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """세그먼트 저장 (SegmentedHashCache 위임)."""
        self._store.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """비연속 세그먼트 히트 조회 (SegmentedHashCache 위임 + 비연속 히트 추적)."""
        hits, misses = self._store.get_segments(token_ids, layer_idx)
        miss_set = set(misses)
        for chunk_idx, _ in hits:
            if any(m < chunk_idx for m in miss_set):
                self._noncontiguous_hits += 1
            self._hits += 1
        self._misses += len(misses)
        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        """비연속 히트율 (전체 히트 대비)."""
        total_hits = self._store._hits + self._hits
        nc = self._store._noncontiguous_hits + self._noncontiguous_hits
        if total_hits == 0:
            return 0.0
        return nc / total_hits
```

---

### DPAttentionAwareCompressionSelector (Activity C)

기구현 코덱 3종(GlobalRetentionGateEvictionCodec, FibQuantVQCodec, SPKVWriteTimeCodec)을
환경 인식 래퍼로 재활용한다. CacheStore 인터페이스를 완전 구현한다.

```python
# src/cache/dp_attention_aware_compression.py

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import torch

from src.cache.base import CacheStore


@dataclass
class DPAttentionCompressionConfig:
    # DP Attention 환경 설정
    dp_attn_enabled: bool = False        # DP Attention 활성화 여부 (환경 변수로도 주입 가능)
    n_gpus: int = 1                       # GPU 수 (torch.cuda.device_count() 자동 감지로 덮어씀)
    auto_detect_gpus: bool = True         # True: 초기화 시 torch.cuda.device_count() 호출

    # 코덱 선택 정책
    single_gpu_codec: str = "global_retention"   # 단일 GPU / DP Attention 비활성 시
    dp_attn_codec: str = "global_retention"      # DP Attention 활성 시 (선택적 압축)
    dp_attn_compression_skip_threshold: float = 0.5  # 한계 효용 < 이 값이면 압축 스킵

    max_entries: int = 1000
    seed: int = 42


class DPAttentionAwareCompressionSelector(CacheStore):
    """DP Attention 상태 인식 환경별 압축 정책 선택기.

    Activity C: KV Cache Compression
    CacheStore 인터페이스 완전 구현.

    환경 감지:
      - n_gpus: auto_detect_gpus=True 시 torch.cuda.device_count() 자동 감지.
      - dp_attn_enabled: 환경 변수 DP_ATTN_ENABLED="1" 또는 config.dp_attn_enabled.
      - effective_kv_replicas = n_gpus (DP Attention 비활성) or 1 (DP Attention 활성).

    압축 정책 선택:
      - effective_kv_replicas > 1 (단일 GPU 또는 DP Attention 비활성):
          고압축 코덱 우선. 개별 KV 압축이 전체 메모리에 직접 기여.
      - effective_kv_replicas == 1 (DP Attention 활성):
          한계 효용 기반 선택적 압축. 한계 효용 < threshold → 압축 스킵.

    이중 절감 정량화:
      실효 메모리 절감율 = 1 - 1/(effective_kv_replicas * compression_ratio)
      결과를 results/<exp>/dp_attn_compression_matrix.json에 기록.

    정확도 보존:
      - DP Attention 활성 시 낮은 압축 강도 → accuracy delta 자동 감소.
      - 각 코덱의 accuracy-preserving 보장(±0.5%)을 상속.
      - 환경별 WikiText-2 perplexity ±1% 독립 검증 (필수).

    평가 기준 (evaluation_criteria.md §4):
      - Accuracy 보존: perplexity 변화 ±1% 이내 (필수)
      - KV Memory Reduction: −30% 이상
      - Effective Context Length: 2× 이상
    """

    def __init__(
        self,
        config: DPAttentionCompressionConfig,
        codec_registry: Optional[Dict[str, CacheStore]] = None,
        env_change_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        torch.manual_seed(config.seed)

        # DP Attention 환경 감지
        self._n_gpus = config.n_gpus
        if config.auto_detect_gpus:
            try:
                detected = torch.cuda.device_count()
                self._n_gpus = max(1, detected)
            except Exception:
                self._n_gpus = 1

        env_flag = os.environ.get("DP_ATTN_ENABLED", "")
        self._dp_attn_enabled = config.dp_attn_enabled or env_flag in ("1", "true", "True")

        # effective_kv_replicas 계산
        self._effective_kv_replicas = 1 if self._dp_attn_enabled else self._n_gpus

        # 코덱 레지스트리: 외부 주입 또는 기본값 (지연 초기화)
        self._codec_registry: Dict[str, CacheStore] = codec_registry or {}
        self._env_change_callback = env_change_callback

        # 내부 스토어 (코덱 없는 경우 fallback)
        self._fallback_store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0
        # 압축 기법별 compression_ratio 추적
        self._codec_compression_ratios: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # 환경 인식 코덱 선택                                                  #
    # ------------------------------------------------------------------ #

    def effective_kv_replicas(self) -> int:
        """현재 effective_kv_replicas 반환."""
        return self._effective_kv_replicas

    def select_codec(self) -> Optional[CacheStore]:
        """환경에 따라 압축 코덱 선택.

        Algorithm:
          1. effective_kv_replicas > 1 (단일/DP Attention 비활성):
               → config.single_gpu_codec 코덱 선택 (고압축 우선).
          2. effective_kv_replicas == 1 (DP Attention 활성):
               → 한계 효용 계산: marginal_utility = 1 - 1/compression_ratio.
               → marginal_utility < dp_attn_compression_skip_threshold
                   → None 반환 (압축 스킵).
               → 그 외 → config.dp_attn_codec 선택.
        """
        codec_name: str
        if self._effective_kv_replicas > 1:
            codec_name = self.config.single_gpu_codec
        else:
            # DP Attention 활성: 한계 효용 확인
            ratio = self._codec_compression_ratios.get(
                self.config.dp_attn_codec, 2.0
            )
            marginal_utility = 1.0 - 1.0 / max(ratio, 1.0)
            if marginal_utility < self.config.dp_attn_compression_skip_threshold:
                return None
            codec_name = self.config.dp_attn_codec

        return self._codec_registry.get(codec_name)

    def register_codec(self, name: str, codec: CacheStore, compression_ratio: float = 2.0) -> None:
        """코덱을 레지스트리에 등록."""
        self._codec_registry[name] = codec
        self._codec_compression_ratios[name] = compression_ratio

    def register_env_change_callback(self, callback: Callable[[], None]) -> None:
        """DP Attention 상태 변화 시 호출할 콜백 등록."""
        self._env_change_callback = callback

    def update_dp_attn_state(self, dp_attn_enabled: bool, n_gpus: Optional[int] = None) -> None:
        """런타임 DP Attention 상태 변화 시 정책 자동 전환.

        Algorithm:
          1. _dp_attn_enabled 및 _n_gpus 업데이트.
          2. _effective_kv_replicas 재계산.
          3. env_change_callback 호출 (있으면).
        """
        self._dp_attn_enabled = dp_attn_enabled
        if n_gpus is not None:
            self._n_gpus = n_gpus
        self._effective_kv_replicas = 1 if self._dp_attn_enabled else self._n_gpus
        if self._env_change_callback is not None:
            self._env_change_callback()

    def effective_memory_reduction_ratio(self, compression_ratio: float) -> float:
        """이중 절감 실효 메모리 절감율 계산.

        실효 절감 = 1 - 1 / (effective_kv_replicas * compression_ratio)
        DP Attention(N-GPU) + 압축(C×) → 1 - 1/(N*C).
        """
        return 1.0 - 1.0 / max(
            self._effective_kv_replicas * compression_ratio, 1.0
        )

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """환경 인식 압축 후 저장."""
        self._total_bytes_original += value.nbytes
        compressed = self.compression_hook(key, value)
        self._total_bytes_stored += compressed.nbytes

        codec = self.select_codec()
        if codec is not None:
            codec.put(key, compressed)
        else:
            # 압축 스킵: fallback store에 직접 저장
            if len(self._fallback_store) >= self.config.max_entries:
                self.evict()
            self._fallback_store[key] = compressed.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        codec = self.select_codec()
        result: Optional[torch.Tensor] = None
        if codec is not None:
            result = codec.get(key)
        if result is None:
            result = self._fallback_store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        codec = self.select_codec()
        if codec is not None:
            return codec.evict()
        if self._fallback_store:
            _, v = self._fallback_store.popitem(last=False)
            return v.nbytes
        return 0

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        codec = self.select_codec()
        if codec is not None:
            return codec.memory_bytes()
        return sum(v.nbytes for v in self._fallback_store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_bytes_original = 0
        self._total_bytes_stored = 0
        for codec in self._codec_registry.values():
            codec.reset_stats()
        self._fallback_store.clear()

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """선택된 코덱의 compression_hook 위임 또는 identity."""
        codec = self.select_codec()
        if codec is not None and hasattr(codec, "compression_hook"):
            return codec.compression_hook(key, value)
        return value

    # ------------------------------------------------------------------ #
    # 메트릭                                                               #
    # ------------------------------------------------------------------ #

    def memory_reduction_ratio(self) -> float:
        """실제 메모리 절감율 (bytes 기준)."""
        if self._total_bytes_original == 0:
            return 0.0
        return 1.0 - self._total_bytes_stored / self._total_bytes_original

    def dp_attn_compression_matrix_entry(
        self,
        codec_name: str,
        compression_ratio: float,
    ) -> Dict:
        """실험 행렬 단일 항목 반환 (results/<exp>/dp_attn_compression_matrix.json에 기록)."""
        return {
            "dp_attn_enabled": self._dp_attn_enabled,
            "n_gpus": self._n_gpus,
            "effective_kv_replicas": self._effective_kv_replicas,
            "codec_name": codec_name,
            "compression_ratio": compression_ratio,
            "effective_memory_reduction": self.effective_memory_reduction_ratio(compression_ratio),
            "actual_memory_reduction": self.memory_reduction_ratio(),
        }
```

---

### AMPDPrefillShareNonContiguousStack (Cross A+B+C)

5단계 처리 흐름을 통합하는 중앙 스택 모듈.

```python
# src/engine/ampd_prefill_share_stack.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from src.engine.runner import InferenceRequest, InferenceRunner
from src.cache.base import CacheStore
from src.scheduler.ampd_lazy_segment_fetch import (
    AMPDLazySegmentFetchScheduler,
    AMPDLazySchedulerConfig,
    SegmentMetadataRegistry,
)
from src.cache.ampd_adapshot_lazy_pipeline import (
    AMPDAdapShotLazyLoadPipeline,
    LazyPipelineConfig,
)
from src.cache.dp_attention_aware_compression import (
    DPAttentionAwareCompressionSelector,
    DPAttentionCompressionConfig,
)


@dataclass
class AMPDStackConfig:
    scheduler_config: Optional[AMPDLazySchedulerConfig] = None
    pipeline_config: Optional[LazyPipelineConfig] = None
    compression_config: Optional[DPAttentionCompressionConfig] = None
    seed: int = 42


class AMPDPrefillShareNonContiguousStack:
    """AMPD + AdapShot + DP Attention 인식 압축 A+B+C 완전 통합 스택.

    Cross Activity A+B+C:
      - A-1 AMPDLazySegmentFetchScheduler: 지연 페치 스케줄러
      - B-1 AMPDAdapShotLazyLoadPipeline: 지연 로드·재인코딩 파이프라인
      - C-1 DPAttentionAwareCompressionSelector: 환경 인식 압축 선택기

    5단계 처리 흐름:
      Step 1: 세그먼트 메타데이터 선행 전달 (KV 전송 없음).
      Step 2: 팬아웃 배포 시뮬레이션 (단일 노드: 로컬 put).
      Step 3: 스케줄 확정 후 지연 로드 + AdapShot 재인코딩 오버랩.
      Step 4: DP Attention 인식 압축 적용.
      Step 5: 비연속 어텐션 계산 투입 (InferenceRunner 호환).

    InferenceRunner 통합:
      runner = InferenceRunner(cache=stack.cache, scheduler=stack)
      runner.run_batch(requests) → stack.schedule(requests) 호출

    평가 기준 (evaluation_criteria.md §5):
      - 복합 처리량: 단일 Activity 대비 +5% 이상
      - 복합 메모리: 단일 Activity 대비 −10% 이상
      - Accuracy 보존 (C 포함): 복합 후 cosine >= 0.99 (필수)
    """

    def __init__(
        self,
        config: AMPDStackConfig,
        extra_codecs: Optional[Dict[str, CacheStore]] = None,
    ) -> None:
        self.config = config

        sched_cfg = config.scheduler_config or AMPDLazySchedulerConfig(seed=config.seed)
        pipeline_cfg = config.pipeline_config or LazyPipelineConfig(seed=config.seed)
        comp_cfg = config.compression_config or DPAttentionCompressionConfig(seed=config.seed)

        self.registry = SegmentMetadataRegistry()
        self.scheduler = AMPDLazySegmentFetchScheduler(sched_cfg, self.registry)
        self.pipeline = AMPDAdapShotLazyLoadPipeline(pipeline_cfg)
        self.compressor = DPAttentionAwareCompressionSelector(comp_cfg)

        # 추가 코덱 등록
        for name, codec in (extra_codecs or {}).items():
            self.compressor.register_codec(name, codec)

        # InferenceRunner가 사용하는 기본 캐시
        self.cache: CacheStore = self.pipeline

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """InferenceRunner.run_batch() 호환 schedule() 진입점."""
        return self.scheduler.schedule(requests)

    def process_request_step1_metadata(
        self,
        request: InferenceRequest,
        candidate_segment_ids: List[str],
    ) -> None:
        """Step 1: 세그먼트 메타데이터 선행 전달 (KV 전송 없음)."""
        from src.scheduler.ampd_lazy_segment_fetch import SegmentMeta
        metas = [
            SegmentMeta(
                segment_id=seg_id,
                source_node_id="local",
                tier="HBM",
                approx_size_bytes=128 * 64 * 2,
                position_range=(0, 128),
            )
            for seg_id in candidate_segment_ids
        ]
        self.scheduler.pre_resolve_segments(request, candidate_segment_ids, metas)

    def process_step3_lazy_load(
        self,
        confirmed_ids: List[str],
        source_positions: Optional[List[int]] = None,
        target_positions: Optional[List[int]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """Step 3: 지연 로드 + AdapShot 재인코딩 (동기 래퍼).

        AsyncIterator를 동기 호출로 래핑해 기존 InferenceRunner와 호환.
        """
        import asyncio
        src_pos = source_positions or [0] * len(confirmed_ids)
        tgt_pos = target_positions or [0] * len(confirmed_ids)

        async def _run():
            results = []
            for seg_id, sp, tp in zip(confirmed_ids, src_pos, tgt_pos):
                kv = await self.pipeline.load_and_reencode_segment(seg_id, sp, tp)
                results.append(kv)
            return results

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    def process_step4_compression(
        self,
        key: str,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """Step 4: DP Attention 인식 압축 적용."""
        return self.compressor.compression_hook(key, kv)

    def metrics_summary(self) -> Dict:
        """복합 효과 측정용 통합 메트릭."""
        return {
            "scheduler_overhead_ms_p50": self.scheduler.scheduling_overhead_ms_p50(),
            "unnecessary_transfer_ratio": self.scheduler.unnecessary_transfer_ratio(),
            "pipeline_hit_rate": self.pipeline.hit_rate(),
            "pipeline_noncontiguous_hit_rate": self.pipeline.noncontiguous_hit_rate(),
            "pipeline_memory_bytes": self.pipeline.memory_bytes(),
            "compressor_hit_rate": self.compressor.hit_rate(),
            "compressor_memory_reduction": self.compressor.memory_reduction_ratio(),
            "compressor_effective_replicas": self.compressor.effective_kv_replicas(),
        }
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(DPAttentionAwareCompressionSelector)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 proxy (실 데이터셋 없을 경우 synthetic 토큰 시퀀스)
- **측정 방법**: `src/metrics/perplexity.py`의 함수 사용
  - `attention_output_relative_error(q, k_orig, v_orig, k_comp, v_comp)` < 0.01 (1%)
  - 단일 GPU 환경(DP Attention 비활성): error < 0.01 (MANDATORY)
  - 멀티 GPU + DP Attention 활성 환경: error < 0.01 (MANDATORY)
  - 두 환경 간 error 차이: < 0.005 (DP Attention 상태가 정확도에 미치는 영향 정량화)
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)

### 태스크 정확도 측정 (LongBench proxy)

- **벤치마크**: LongBench 8개 서브태스크 proxy
  - 측정 방법: `attention_kl_divergence` < 0.015 (MANDATORY)
  - `cosine_similarity_output` >= 0.99 (MANDATORY)
- **실험 행렬**: {DP Attention ON/OFF} × {압축 기법: None/GlobalRetention/FibQuant/SPKVWriteTime} × {n_gpus: 1/2/4}
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)

### 이중 절감 이론 검증

- **검증 수식**: effective_reduction = 1 - 1/(N × C)
  - N=1, C=2: 50% (단일 GPU, GlobalRetention 2×)
  - N=4, C=3: 91.7% (4-GPU DP Attention + 3× 압축)
  - N=4, C=10: 97.5% (이론 최솟값, FibQuant 10×)
- **검증 테스트**: `test_effective_reduction_formula` — 수치가 이론값과 일치하는지 확인

### DP Attention ON/OFF 간 정확도 차이 정량화

- DP Attention 비활성 시 고압축(GlobalRetention) 적용 후 error
- DP Attention 활성 시 저압축(identity 또는 낮은 ratio) 적용 후 error
- 두 설정의 error 비교: DP Attention 활성 환경이 더 낮은 error를 가져야 함
- 이 비교가 "환경 인식 정책의 정확도 안전성"을 증명

### 환경 변화 내구성 검증

- DP Attention ON → OFF 런타임 전환 시 `update_dp_attn_state()` 호출
- 전환 후 코덱 선택이 정책에 맞게 자동 변경됨 확인
- 전환 후 첫 번째 put/get 사이클의 accuracy delta: < 0.01 (MANDATORY)

### 검증 테스트 파일

`tests/unit/test_dp_attention_aware_compression.py`

**테스트 케이스 목록**:

```
test_single_gpu_selects_high_compression:
    n_gpus=1, dp_attn_enabled=False → single_gpu_codec 선택 확인

test_dp_attn_enabled_selects_low_compression:
    dp_attn_enabled=True → dp_attn_codec 선택 또는 스킵 확인

test_dp_attn_skip_when_marginal_utility_low:
    marginal_utility < threshold → select_codec() == None 반환

test_effective_replicas_single_gpu:
    n_gpus=1, dp_attn=False → effective_kv_replicas == 1

test_effective_replicas_multi_gpu_no_dp:
    n_gpus=4, dp_attn=False → effective_kv_replicas == 4

test_effective_replicas_multi_gpu_with_dp:
    n_gpus=4, dp_attn=True → effective_kv_replicas == 1

test_effective_reduction_formula:
    N=4, C=10 → effective_memory_reduction_ratio == 0.975 (±0.001)

test_accuracy_single_gpu_within_1pct:
    단일 GPU 고압축 코덱 적용 후 attention error < 0.01 (MANDATORY)

test_accuracy_dp_attn_within_1pct:
    DP Attention 활성 환경 압축 후 attention error < 0.01 (MANDATORY)

test_kl_divergence_within_threshold:
    KL < 0.015 at global_retention codec (MANDATORY)

test_cosine_similarity_above_threshold:
    cosine >= 0.99 at global_retention codec (MANDATORY)

test_dp_attn_vs_single_gpu_error_difference:
    DP Attention 활성 error <= 단일 GPU error + 0.005

test_runtime_dp_attn_toggle:
    update_dp_attn_state(True) 후 effective_replicas == 1
    update_dp_attn_state(False) 후 effective_replicas == n_gpus

test_env_var_detection:
    DP_ATTN_ENABLED="1" 환경 변수 설정 시 dp_attn_enabled=True 자동 감지

test_cachestore_interface:
    put/get/evict/hit_rate/memory_bytes/reset_stats 동작 확인

test_memory_reduction_gt_30pct_single_gpu:
    단일 GPU + GlobalRetention: memory_reduction_ratio() >= 0.30 (MANDATORY)

test_compression_matrix_entry_keys:
    dp_attn_compression_matrix_entry() 반환 딕셔너리 키 구조 확인

test_register_env_change_callback:
    register_env_change_callback() 등록 후 update_dp_attn_state() 시 콜백 호출

test_compression_hook_identity_when_skip:
    select_codec() == None 시 compression_hook()이 identity 반환
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-18.yaml
experiment:
  date: "2026-05-18"
  activity: "A+B+C"
  description: >
    A-1 AMPDLazySegmentFetchScheduler (AMPD pull-on-demand 세그먼트 지연 페치 스케줄러) +
    B-1 AMPDAdapShotLazyLoadPipeline (지연 로드 + AdapShot RoPE 재인코딩 비동기 오버랩) +
    C-1 DPAttentionAwareCompressionSelector (DP Attention 상태 인식 환경별 압축 선택기) +
    Cross-1 AMPDPrefillShareNonContiguousStack (A+B+C 5단계 통합 스택)
  cache_type: ampd_adapshot_lazy_pipeline
  compression_method: dp_attention_aware
  scheduler_type: ampd_lazy_segment_fetch

ampd_lazy_segment_fetch_scheduler:
  hbm_fetch_latency_ms: 0.01
  ddr_fetch_latency_ms: 0.5
  remote_fetch_latency_ms: 5.0
  metadata_overhead_max_ms: 0.1     # 메타데이터 전달 오버헤드 상한 (MANDATORY 검증 기준)
  max_concurrent_fetches: 8
  seed: 42

ampd_adapshot_lazy_pipeline:
  chunk_size: 128
  max_entries: 1000
  rope_theta: 10000.0               # AdapShot RoPE 기본 주파수
  n_heads: 8
  d_head: 64
  companion_hit_threshold: 2        # 예측 프리페치 동반 세그먼트 히트 임계값
  seed: 42

dp_attention_aware_compression:
  dp_attn_enabled: false            # 환경 변수 DP_ATTN_ENABLED="1" 또는 이 값으로 설정
  n_gpus: 1                         # auto_detect_gpus=True 시 자동 감지
  auto_detect_gpus: true
  single_gpu_codec: "global_retention"   # 단일 GPU: GlobalRetentionGateEvictionCodec
  dp_attn_codec: "global_retention"      # DP Attention 활성: 동일 코덱 (낮은 강도)
  dp_attn_compression_skip_threshold: 0.5
  max_entries: 1000
  seed: 42

ampd_prefill_share_stack:
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    proxy_tolerance: 0.01            # 1% attention output error limit (MANDATORY)
    kl_tolerance: 0.015
    cosine_min: 0.99
    perplexity_dataset: "wikitext-2"
    perplexity_tolerance_pct: 1.0    # ±1% 이내 (MANDATORY)
    task_accuracy_tolerance_pct: 1.0
  dp_attn_experiment_matrix:         # 실험 행렬 설정
    dp_attn_states: [true, false]
    codec_names: ["none", "global_retention", "fibquant", "spkv"]
    n_gpus_list: [1, 2, 4]
    output_file: "results/2026-05-18/dp_attn_compression_matrix.json"
  activity_a:
    ttft_overhead_limit_pct: 5.0     # TTFT p50 +5% 이내 (MANDATORY)
    metadata_overhead_max_ms: 0.1    # < 0.1ms/요청 (MANDATORY)
    unnecessary_transfer_ratio_target: 0.40  # 40% 이상 불필요 전송 제거 목표
  activity_b:
    noncontiguous_hit_rate_min: 0.30  # 전체 히트의 30% 이상 (MANDATORY)
    kv_memory_footprint_increase_max: 0.20  # +20% 이내
  activity_c:
    memory_reduction_min: 0.30       # −30% 이상 (MANDATORY)
    effective_context_multiplier: 2.0
    compression_overhead_ttft_max_pct: 10.0
  throughput:
    target_improvement_pct: 20       # +20% 이상
  cross_abc_comparison:
    methods: ["solo_a1", "solo_b1", "solo_c1", "cross_combined"]
    throughput_min_improvement_vs_solo: 5.0   # +5% 이상 (§5)
    memory_min_improvement_vs_solo: 10.0      # −10% 이상 (§5)
    accuracy_cosine_min: 0.99                 # (§5 C 포함 필수)

seed: 42
results_dir: "results/2026-05-18"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_dp_attention_aware_compression.py` — Activity C 필수 accuracy 검증 (19개 테스트, 위 목록 참조)
- [ ] `tests/unit/test_ampd_lazy_segment_fetch.py` — Activity A 단위 테스트
- [ ] `tests/unit/test_ampd_adapshot_lazy_pipeline.py` — Activity B 단위 테스트
- [ ] `tests/unit/test_ampd_prefill_share_stack.py` — Cross A+B+C 통합 단위 테스트
- [ ] `tests/integration/test_cross_abc_ampd_stack.py` — E2E 통합 테스트

### 단위 테스트 최소 요구 사항 (test_ampd_lazy_segment_fetch.py)

```
test_segment_metadata_registry_register:
    register() 후 get()으로 메타데이터 조회 가능

test_segment_metadata_registry_cancel:
    cancel() 후 unnecessary_transfer_ratio 증가 확인

test_segment_metadata_registry_ratio:
    cancel 2 / pre_resolved 5 → ratio == 0.4

test_pre_resolve_segments_no_kv_transfer:
    pre_resolve_segments() 호출 후 KV 데이터 미전송 (registry에 메타데이터만 존재)

test_confirm_segments_cancels_non_confirmed:
    confirm_segments([a,b,c], [a]) → b, c cancel 처리됨

test_fetch_segments_lazy_yields_kvsegment:
    fetch_segments_lazy(["seg1"]) → KVSegment(segment_id="seg1") yield

test_fetch_segments_lazy_hbm_latency:
    tier=HBM fetch latency == hbm_fetch_latency_ms (±0.01ms)

test_fetch_segments_lazy_ddr_latency:
    tier=DDR fetch latency == ddr_fetch_latency_ms (±0.05ms)

test_scheduling_overhead_below_01ms:
    schedule() 단일 호출 오버헤드 < 0.1ms

test_schedule_returns_all_requests:
    schedule(requests) 반환 길이 == 입력 길이

test_unnecessary_transfer_ratio_zero_initial:
    초기 unnecessary_transfer_ratio() == 0.0

test_reset_stats_clears_counts:
    reset_stats() 후 ratio == 0.0
```

### 단위 테스트 최소 요구 사항 (test_ampd_adapshot_lazy_pipeline.py)

```
test_cachestore_interface:
    put/get/evict/hit_rate/memory_bytes/reset_stats 동작 확인

test_resolve_segments_returns_metas_no_kv:
    resolve_segments() 반환값이 SegmentMeta 목록 (KV 텐서 없음)

test_put_segment_and_get_segments:
    put_segment() 후 get_segments()에서 히트 반환

test_noncontiguous_hit_detection:
    chunk 0 miss + chunk 2 hit → noncontiguous_hit += 1

test_noncontiguous_hit_rate_above_threshold:
    비연속 패턴 10회 입력 후 noncontiguous_hit_rate() > 0.0

test_adapshot_reencode_changes_kv:
    source_pos != target_pos → 재인코딩 후 KV 값 변화 확인

test_adapshot_reencode_identity_same_pos:
    source_pos == target_pos → 재인코딩 후 원본과 거의 동일 (error < 1e-4)

test_adapshot_reencode_preserves_shape:
    재인코딩 전후 KV shape 동일

test_load_and_reencode_segment_async:
    asyncio로 load_and_reencode_segment() 호출 → KVSegment 반환

test_load_and_reencode_returns_none_on_miss:
    미저장 segment_id → None 반환

test_memory_bytes_increases_on_put:
    put() 후 memory_bytes() 증가
```

### 단위 테스트 최소 요구 사항 (test_ampd_prefill_share_stack.py)

```
test_stack_init_all_components:
    초기화 시 scheduler, pipeline, compressor 모두 존재

test_schedule_delegates_to_scheduler:
    stack.schedule() 호출 시 scheduler.schedule() 위임

test_process_step1_registers_metadata:
    process_request_step1_metadata() 후 registry에 세그먼트 등록됨

test_process_step3_lazy_load_returns_tensors:
    confirmed_ids 있으면 step3 결과가 텐서 목록

test_process_step4_compression_applies:
    compression_hook 활성 코덱 있으면 step4에서 텐서 변환됨

test_metrics_summary_keys:
    metrics_summary()가 scheduler_overhead, pipeline_hit_rate 등 모든 키 포함

test_cross_abc_accuracy_preserved:
    5단계 처리 후 cosine >= 0.99 (evaluation_criteria.md §5 C 포함 필수)

test_cross_abc_throughput_vs_solo_b1:
    단독 B-1 대비 Cross-1 처리량 +5% 이상 목표 (§5 검증)

test_cross_abc_memory_vs_solo_c1:
    단독 C-1 대비 Cross-1 메모리 −10% 이상 목표 (§5 검증)

test_cache_property_is_cachestore:
    stack.cache가 CacheStore 인스턴스
```

---

## vLLM 이식 경로 (vllm-porter 참조용)

```
vllm_integration/
├── scheduler_patch.py          # LazySegmentFetchSchedulerMixin 추가
├── attention_backend_patch.py  # LazyLoadReencodingAttentionHook 추가
└── dp_attn_compression_hook.py # DPAttentionAwareCompressionSelector vLLM 연동 (신규)
```

### Activity A 통합 포인트: `vllm/core/scheduler.py`

```python
# vllm_integration/scheduler_patch.py 추가 사항

class LazySegmentFetchSchedulerMixin:
    """vLLM Scheduler에 AMPD 지연 페치 기능을 추가하는 믹스인.

    vLLM 통합 포인트:
      - vllm.v1.core.sched.scheduler.Scheduler
      - schedule() 메서드를 오버라이드해 _ampd_pre_resolve() 훅 삽입.
      - 세그먼트 확정 이벤트를 KV 전송 타이밍 제어에 연결.

    make_lazy_fetch_scheduler_class() 팩토리:
      LazySegmentFetchSchedulerMixin + vLLM Scheduler 동적 조합.
    """
    def _ampd_pre_resolve(self, waiting_requests) -> None:
        """schedule() 전 후보 세그먼트 메타데이터 등록."""
        ...
```

### Activity B 통합 포인트: attention backend

```python
# vllm_integration/attention_backend_patch.py 추가 사항

class LazyLoadReencodingAttentionHook:
    """FlashAttentionImpl에 지연 로드 + AdapShot 재인코딩 훅 삽입.

    write_to_cache: 세그먼트 확정 후에만 캐시 기록.
    read_from_cache: load_and_reencode_segment()로 재인코딩된 KV 반환.
    """
    def write_to_cache(self, key_cache, value_cache, layer_idx: int) -> None: ...
    def read_from_cache(self, key_cache, value_cache, layer_idx: int): ...
```

### Activity C 통합 포인트: KV 압축 훅

```python
# vllm_integration/dp_attn_compression_hook.py (신규)

class DPAttentionCompressionHook:
    """vLLM의 DP Attention 상태를 DPAttentionAwareCompressionSelector에 주입.

    vLLM DP Attention 상태 읽기:
      - vllm.config.ParallelConfig.data_parallel_size
      - vllm.worker.WorkerBase.dp_rank 존재 여부로 DP Attention 활성 감지.
    """
    def __init__(self, selector: "DPAttentionAwareCompressionSelector") -> None: ...
    def inject_dp_state(self, vllm_engine) -> None:
        """vLLM 엔진에서 DP Attention 상태를 읽어 selector.update_dp_attn_state() 호출."""
        ...
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 4개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - perplexity 변화 ±1% 이내 (attention error < 0.01, 단일/멀티 GPU 양쪽)
      - downstream 태스크 정확도 ±1% 이내 (KL < 0.015, cosine >= 0.99)
      - {DP Attention ON/OFF} × {압축 기법} 교차 행렬 검증
      - DP Attention ON/OFF 간 정확도 차이 정량화
- [ ] `evaluation_criteria.md` §2 Activity A 항목 충족:
      - 스케줄링 오버헤드 TTFT p50 +5% 이내
      - 메타데이터 선행 전달 오버헤드 < 0.1ms/요청
      - unnecessary_transfer_ratio 지표 수집 및 기록
- [ ] `evaluation_criteria.md` §3 Activity B 항목 충족:
      - 비연속 세그먼트 히트율 전체 히트의 30% 이상
      - KV Memory Footprint +20% 이내
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함:
      - 복합 적용 후 accuracy ±1% 이내 (cosine >= 0.99)
      - 단독 A-1 / B-1 / C-1 / Cross-1 4방향 비교 수치 확인
      - 단독 Activity 대비 +5% 처리량, −10% 메모리 개선
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - CacheStore 인터페이스 모든 추상 메서드 구현 (AMPDAdapShotLazyLoadPipeline, DPAttentionAwareCompressionSelector)
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-18.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-18/metrics.json`에 JSON 기록:
      ```json
      {
        "inference_throughput_improvement_pct": ...,
        "kv_memory_reduction_ratio": ...,
        "compression_accuracy_delta_single_gpu": ...,
        "compression_accuracy_delta_dp_attn": ...,
        "effective_context_length_multiplier": ...,
        "scheduling_overhead_ttft_p50_pct": ...,
        "metadata_overhead_ms_p50": ...,
        "unnecessary_transfer_ratio": ...,
        "noncontiguous_hit_rate": ...,
        "cross_abc_throughput_vs_solo_pct": ...,
        "cross_abc_memory_vs_solo_pct": ...,
        "cross_abc_accuracy_cosine": ...,
        "dp_attn_compression_matrix": "results/2026-05-18/dp_attn_compression_matrix.json"
      }
      ```
- [ ] `src/cache/base.py` CacheStore 인터페이스 깨지지 않음 (수정 없음)
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
