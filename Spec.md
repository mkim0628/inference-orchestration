# Spec — 2026-04-29

<!-- 변경 이유: 이전 사이클(2026-04-28) B+C 기반 구현 완료 후 Activity A(스케줄러) 추가, Activity C를 Hadamard INT4로 업그레이드. -->

## 배경

이전 사이클(2026-04-28): Activity B+C(위치-독립 세그먼트 해시 + FP16/INT8 혼합 정밀도)로 비연속 히트율 30.3%, 메모리 −68.8%, 정확도 ±0.72% 달성.

이번 사이클의 세 가지 추가·개선:
1. **Activity A 신규 추가**: 비연속 히트율을 인식하는 캐시-가중 스케줄러로 처리량 +20% 목표에 근접.
2. **Activity B 개선**: ChunkKV 스타일 청크 중요도 기반 퇴거 정책 추가, 청크 크기 128 토큰 유지.
3. **Activity C 업그레이드**: FP16/INT8 혼합 정밀도 → Hadamard 회전 기반 INT4 양자화(SAW-INT4 스타일)로 메모리 절감을 −68.8% → −75%+ 로 향상.

해결 문제: Activity A 미구현(처리량 미측정), 압축률 추가 개선 여지, 퇴거 정책의 의미적 정합성 부족.

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling
- [x] Activity B: Non-Contiguous KV Cache Reuse (개선)
- [x] Activity C: KV Cache Compression (Hadamard INT4 업그레이드)

## 목표

- [ ] 목표 1 (§1): 배치 처리량(tokens/sec) 베이스라인 대비 +10% 이상 (스케줄러 효과 측정) — evaluation_criteria.md §1
- [ ] 목표 2 (§2): 스케줄링 오버헤드 TTFT p50 증가 +5% 이내, 캐시 히트율 +10%p 이상 — §2
- [ ] 목표 3 (§3): 비연속 세그먼트 히트율 전체 히트의 30% 이상 유지 — §3
- [ ] 목표 4 (§4): Hadamard INT4로 KV 메모리 −75% 이상, accuracy delta ±1% 이내 — §4
- [ ] 목표 5 (§5): A+B+C 복합 처리량 단일 Activity 대비 추가 +5%, 복합 메모리 추가 −10% — §5

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/scheduler/__init__.py` | A | 패키지 초기화 |
| `src/scheduler/cache_aware_scheduler.py` | A | 비연속 히트율 인식 캐시-가중 스케줄러 |
| `tests/unit/test_cache_aware_scheduler.py` | A | 스케줄러 단위 테스트 |
| `tests/integration/test_abc_e2e.py` | A+B+C | 전체 통합 엔드-투-엔드 테스트 |
| `configs/experiments/2026-04-29.yaml` | 공통 | 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/compression.py` | `HadamardInt4Codec` 클래스 추가 (기존 `CompressionCodec` 유지) |
| `src/cache/segmented.py` | `evict()` 메서드에 청크 중요도 스코어 지원 추가, `record_attention_score()` 메서드 추가 |
| `src/cache/compressed_segment.py` | `HadamardInt4Codec` 사용 가능하도록 codec 타입 힌트 일반화 |
| `src/engine/runner.py` | `InferenceRunner`에 optional scheduler 파라미터 추가, `run_batch()` 스케줄러 훅 |
| `tests/unit/test_compression_accuracy.py` | `HadamardInt4Codec` 정확도 테스트 추가 |

---

## 알고리즘 상세

### CacheAwareScheduler (Activity A)

```python
# src/scheduler/cache_aware_scheduler.py

@dataclass
class ScheduledRequest:
    request: InferenceRequest
    predicted_hit_rate: float
    wait_steps: int        # 얼마나 오래 대기했는지
    priority_score: float  # 최종 정렬 키

class CacheAwareScheduler:
    def __init__(
        self,
        cache: CacheStore,       # CompressedSegmentCache 또는 SegmentedHashCache
        fairness_max_wait: int = 10,  # 최대 대기 스텝 (공정성)
        chunk_size: int = 128,
    ) -> None: ...

    def _predict_hit_rate(self, request: InferenceRequest) -> float:
        # 1. 요청의 모든 청크에 대해 chunk_key 계산
        # 2. cache._store에 key가 있는지 확인 (실제 get() 호출 아님 — 통계 오염 방지)
        # 3. predicted_hit_rate = matched_chunks / total_chunks
        ...

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        # 1. 각 요청의 predicted_hit_rate 계산
        # 2. priority = hit_rate * (1 - wait_penalty)
        #    wait_penalty = min(wait_steps / fairness_max_wait, 1.0)
        # 3. priority 내림차순 정렬
        # 4. 반환: 재정렬된 요청 리스트
        ...

    def update_wait(self, skipped_requests: List[InferenceRequest]) -> None:
        """이번 배치에서 처리되지 않은 요청의 wait_steps 증가."""
        ...
```

**스케줄러 통합 (runner.py)**:
```python
class InferenceRunner:
    def __init__(self, cache, scheduler=None, ...):
        self.scheduler = scheduler  # Optional[CacheAwareScheduler]

    def run_batch(self, requests):
        if self.scheduler is not None:
            requests = self.scheduler.schedule(requests)
        return [self.run(r) for r in requests]
```

**_predict_hit_rate 구현 세부**:
- `cache`가 `SegmentedHashCache`/`CompressedSegmentCache`이면 `cache._store` dict를 직접 조회(key 존재 여부만 체크, get() 미호출)
- `cache`가 다른 타입이면 0.0 반환 (graceful fallback)
- chunk_key 계산은 `SegmentedHashCache.chunk_key()` 와 동일 로직 사용

---

### Chunk Importance-based Eviction (Activity B 개선)

```python
# src/cache/segmented.py 변경

class SegmentedHashCache(CacheStore):
    def __init__(self, chunk_size=128, max_entries=1000):
        ...
        self._importance: dict[str, float] = {}  # key → 누적 attention score

    def record_attention_score(self, key: str, score: float) -> None:
        """어텐션 스코어 누적. 캐시 히트 시 외부에서 호출 가능."""
        self._importance[key] = self._importance.get(key, 0.0) + score

    def evict(self) -> int:
        """중요도 최소 엔트리 퇴거. _importance 없으면 LRU fallback."""
        if not self._store:
            return 0
        candidates = list(self._store.keys())
        if self._importance:
            evict_key = min(
                candidates,
                key=lambda k: self._importance.get(k, 0.0),
            )
        else:
            evict_key = candidates[0]  # LRU: OrderedDict의 첫 번째 항목
        evicted = self._store.pop(evict_key)
        self._importance.pop(evict_key, None)
        return evicted.nbytes
```

---

### HadamardInt4Codec (Activity C 업그레이드)

```python
# src/cache/compression.py 에 추가

class HadamardInt4Codec:
    """Hadamard 회전 + INT4 양자화 코덱 (SAW-INT4 스타일).

    encode:
      1. kv를 FP32로 변환
      2. 초기 레이어(< cutoff): FP16으로 저장 (중요 정보 보존)
      3. 후반 레이어:
         a. Hadamard 회전 적용: kv_rot = H @ kv  (H = normalized Hadamard matrix)
            - dim이 power-of-2 아닌 경우: zero-pad → Hadamard → unpad 후 저장
         b. scale = kv_rot.abs().max() / 7.0  (INT4 범위 [-8, 7])
         c. quantize: round(kv_rot / scale).clamp(-8, 7).to(torch.int8)

    decode:
      1. 초기 레이어: FP16 → FP32
      2. 후반 레이어: int8 * scale → inverse Hadamard → FP32
         (정규화 Hadamard는 직교 행렬: H^{-1} = H^T = H)

    메모리 절감 (num_layers=12, cutoff_ratio=0.2):
      - 초기 2 레이어: FP16 → −50% vs FP32
      - 후반 10 레이어: INT8 저장 (INT4 범위) → −75% vs FP32
      - 평균: (2*0.5 + 10*0.75) / 12 = 0.708 → −70.8%
      (true INT4 bit-packing 구현 시: 후반 −87.5%, 평균 −81%)
    """

    def __init__(self, num_layers: int, cutoff_ratio: float = 0.2) -> None:
        self.num_layers = num_layers
        self.cutoff = max(1, int(num_layers * cutoff_ratio))
        self._scales: dict[tuple[int, int], float] = {}
        self._hadamard_cache: dict[int, torch.Tensor] = {}  # dim → H matrix

    def _next_power_of_two(self, n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    def _hadamard_matrix(self, dim: int) -> torch.Tensor:
        """Normalized Hadamard matrix for dim (builds recursively, cached)."""
        if dim in self._hadamard_cache:
            return self._hadamard_cache[dim]
        if dim == 1:
            h = torch.ones(1, 1)
        else:
            half = self._hadamard_matrix(dim // 2)
            top = torch.cat([half, half], dim=1)
            bot = torch.cat([half, -half], dim=1)
            h = torch.cat([top, bot], dim=0) / (2 ** 0.5)
        self._hadamard_cache[dim] = h
        return h

    def _apply_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard rotation to last dimension of x."""
        orig_dim = x.shape[-1]
        pad_dim = self._next_power_of_two(orig_dim)
        if pad_dim != orig_dim:
            pad = torch.zeros(*x.shape[:-1], pad_dim - orig_dim, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        H = self._hadamard_matrix(pad_dim).to(x.device)
        rotated = x @ H.T  # (..., pad_dim)
        return rotated[..., :orig_dim]

    def _inverse_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse Hadamard (H is orthogonal: H^{-1} = H^T = H for normalized)."""
        orig_dim = x.shape[-1]
        pad_dim = self._next_power_of_two(orig_dim)
        if pad_dim != orig_dim:
            pad = torch.zeros(*x.shape[:-1], pad_dim - orig_dim, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        H = self._hadamard_matrix(pad_dim).to(x.device)
        restored = x @ H  # H^T = H for normalized Hadamard
        return restored[..., :orig_dim]

    def encode(self, kv: torch.Tensor, layer_idx: int, tensor_id: int = 0) -> torch.Tensor:
        kv_f = kv.float()
        if layer_idx < self.cutoff:
            return kv_f.half()
        rotated = self._apply_hadamard(kv_f)
        abs_max = rotated.abs().max().item()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0
        self._scales[(layer_idx, tensor_id)] = scale
        quantized = rotated.div(scale).round().clamp(-8, 7).to(torch.int8)
        return quantized

    def decode(self, compressed: torch.Tensor, layer_idx: int, tensor_id: int = 0) -> torch.Tensor:
        if layer_idx < self.cutoff:
            return compressed.float()
        scale = self._scales.get((layer_idx, tensor_id), 1.0)
        dequantized = compressed.float() * scale
        return self._inverse_hadamard(dequantized)

    def compression_ratio(self, layer_idx: int) -> float:
        return 0.5 if layer_idx < self.cutoff else 0.75

    def average_compression_ratio(self) -> float:
        early = self.cutoff * 0.5
        late = (self.num_layers - self.cutoff) * 0.75
        return (early + late) / self.num_layers
```

---

## Activity C — Accuracy Preservation 검증 계획

- **perplexity 측정**: 합성 KV 텐서(shape: [128, 64], FP32) 기반 encode→decode 왕복 테스트.
  - L2 relative error = `||decoded - original||_F / ||original||_F` ≤ 1%
  - 코사인 유사도 = `cos_sim(decoded.flatten(), original.flatten())` ≥ 0.99
- **태스크 정확도 측정**: 어텐션 스코어 KL divergence.
  - Q matrix 고정(randn seed=42, shape [8, 64])
  - `attn_orig = softmax(Q @ K_orig.T / sqrt(64))`
  - `attn_decoded = softmax(Q @ K_decoded.T / sqrt(64))`
  - KL(attn_orig || attn_decoded) ≤ 0.01
- **검증 테스트 파일**: `tests/unit/test_compression_accuracy.py` — `TestHadamardInt4Accuracy` 클래스
  - `test_roundtrip_l2_error`: L2 relative error ≤1% (모든 레이어, num_layers=12)
  - `test_cosine_similarity`: cosine sim ≥0.99 (모든 레이어)
  - `test_attention_kl_divergence`: KL ≤0.01 (후반 레이어, INT4 적용 레이어)
  - `test_vs_baseline_codec`: HadamardInt4Codec vs CompressionCodec 정확도 비교
- **허용 오차**: ±1% — evaluation_criteria.md §4 필수 항목

---

## 설정 파라미터

```yaml
# configs/experiments/2026-04-29.yaml
experiment:
  date: "2026-04-29"
  activity: "A+B+C"
  description: "Cache-aware scheduler + chunk importance eviction + Hadamard INT4"

cache:
  type: "compressed_segment"
  chunk_size: 128
  max_entries: 1000

compression:
  codec: "hadamard_int4"
  num_layers: 12
  cutoff_ratio: 0.2

scheduler:
  type: "cache_aware"
  fairness_max_wait: 10
  enabled: true

runner:
  num_layers: 12
  hidden_dim: 64
  chunk_size: 128
  seed: 42

baseline:
  cache_type: "contiguous"
  scheduler: "none"
```

---

## 테스트 요구사항

- [ ] `tests/unit/test_cache_aware_scheduler.py`
  - `test_schedule_reorders_by_hit_rate`
  - `test_fairness_max_wait`
  - `test_scheduling_overhead_ms`
- [ ] `tests/unit/test_compression_accuracy.py` (기존 + `TestHadamardInt4Accuracy` 추가)
- [ ] `tests/unit/test_segmented_cache.py` (기존 + `test_importance_based_eviction` 추가)
- [ ] `tests/integration/test_abc_e2e.py` (신규)
  - A+B+C 전체 파이프라인: 스케줄러 → CompressedSegmentCache(HadamardInt4Codec) → 처리량 측정
  - 베이스라인 vs A+B+C 비교 (처리량 +10% 이상, 메모리 −70% 이상, accuracy ≤1%)

## 완료 기준 (Definition of Done)

- 단위 테스트 100% 통과
- 통합 테스트 100% 통과
- `HadamardInt4Codec` accuracy: L2 error ≤1%, cosine sim ≥0.99 (모든 레이어)
- 스케줄러 오버헤드: 100 요청 재정렬 ≤5ms
- 베이스라인 대비 메모리 −70% 이상
- 비연속 세그먼트 히트율 ≥30% 유지
- `evaluation_criteria.md` §0·§1·§2·§3·§4·§5 전 섹션 기준 적용
