# Spec — 2026-04-30

<!-- 변경 이유: 이전 사이클(2026-04-29) A+B+C 단일노드 통합 완료 후, 멀티노드 확장(A), KV Packet adapter(B), ARKV tri-state 압축(C)을 추가. -->

## 배경

이전 사이클(2026-04-29) 달성 결과:
- Activity A: `CacheAwareScheduler` 단일노드 완료, TTFT 오버헤드 ≤5%
- Activity B: `CompressedSegmentCache` 비연속 히트율 ≥30%, 중요도 기반 퇴거 완료
- Activity C: `HadamardInt4Codec` INT4 양자화 −70% 메모리, KL<0.007

미해결 항목 (SUMMARY.md 우선순위):
1. 멀티노드 KV 마이그레이션 라우팅 (Activity A)
2. KV Packet adapter 통합 (Activity B)
3. ARKV tri-state 압축 (Activity C)

## 이번 사이클 Activity

- [x] Activity A: MultiNodeScheduler (P/D 분리 시뮬레이션)
- [x] Activity B: SegmentAdapter (KV Packet 스타일 MLP adapter)
- [x] Activity C: TriStateCompressor (ARKV tri-state: retain/compress/evict)

## 목표

- [ ] §1: MultiNodeScheduler 처리량 단일노드 대비 +10% 이상
- [ ] §2: MultiNodeScheduler TTFT 오버헤드 +5% 이내 (멀티노드 포함)
- [ ] §3: SegmentAdapter 사용 후 비연속 히트 KL divergence ≤ 0.005
- [ ] §4: TriStateCompressor 메모리 −75% 이상, accuracy delta ±1% 이내
- [ ] §5: A+B+C 복합 통합 테스트 통과 (기존 55개 테스트 + 신규 테스트)

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/tri_state_compressor.py` | C | ARKV 스타일 tri-state KV 압축기 |
| `src/cache/segment_adapter.py` | B | KV Packet 스타일 MLP adapter |
| `src/scheduler/multi_node_scheduler.py` | A | P/D 분리 멀티노드 스케줄러 |
| `tests/unit/test_tri_state_compressor.py` | C | TriStateCompressor 단위 테스트 |
| `tests/unit/test_segment_adapter.py` | B | SegmentAdapter 단위 테스트 |
| `tests/unit/test_multi_node_scheduler.py` | A | MultiNodeScheduler 단위 테스트 |
| `configs/experiments/2026-04-30.yaml` | 공통 | 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/compressed_segment.py` | `adapter` 옵션 파라미터 추가, 비연속 히트 시 adapter 적용 |
| `tests/integration/test_abc_e2e.py` | 멀티노드 + tri-state + adapter 통합 테스트 추가 |

---

## 알고리즘 상세

### 1. TriStateCompressor (Activity C)

**파일**: `src/cache/tri_state_compressor.py`

클래스 설계:

```python
class TriStateCompressor:
    def __init__(
        self,
        codec: HadamardInt4Codec,
        retain_ratio: float = 0.20,  # top 20% by attn score → FP16
        evict_ratio: float = 0.40,   # bottom 40% → delete
        # middle 40% → INT4 compress
    ) -> None

    def classify(
        self,
        kv: torch.Tensor,           # (n_tokens, kv_dim)
        attn_weights: torch.Tensor, # (n_tokens,)
        layer_idx: int,
    ) -> dict:
        # returns: retain_kv, compress_kv, retain_indices, compress_indices, evict_indices

    def encode(
        self,
        kv: torch.Tensor,
        attn_weights: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> dict:
        # classify + compress compress_kv with codec
        # returns storage dict with retain_kv(FP16), compressed_kv(INT8), indices

    def decode(
        self,
        storage: dict,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        # reconstruct full kv; evicted positions filled with zeros
        # shape matches original (n_tokens, kv_dim)

    def compression_ratio(self, retain_ratio=0.20, evict_ratio=0.40) -> float:
        # retain: 0.20 * (2/4) = 0.10
        # compress: 0.40 * (1/4) = 0.10
        # evict: 0
        # total: 0.20 of FP32 baseline → 80% savings
```

**정확도 보존 검증 (accuracy-preservation plan)**:
- retain tier: FP16 보존이므로 KL divergence ≈ 0 (완전 보존)
- compress tier: HadamardInt4Codec 이미 KL<0.007 검증됨
- evict tier: 하위 40% low-attention 토큰 — attention에 기여 미미
- 통합 KL criterion: `KL(decode(encode(kv)), kv) < 0.01` (±1% perplexity proxy)
- cosine similarity ≥ 0.90 for non-evicted tokens

### 2. SegmentAdapter (Activity B)

**파일**: `src/cache/segment_adapter.py`

```python
import torch
import torch.nn as nn
from typing import List, Optional

class SegmentAdapter(nn.Module):
    def __init__(
        self,
        kv_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        # self.mlp = nn.Sequential(
        #     nn.Linear(kv_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, kv_dim),
        # )

    def forward(self, cached_kv: torch.Tensor) -> torch.Tensor:
        # residual connection: output = cached_kv + self.mlp(cached_kv)
        # shape preserved: (..., kv_dim) → (..., kv_dim)

    def train_step(
        self,
        cached_kv: torch.Tensor,
        target_kv: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        # L2 loss: F.mse_loss(self.forward(cached_kv), target_kv)
        # returns loss.item()

    def fit(
        self,
        cached_kvs: List[torch.Tensor],
        target_kvs: List[torch.Tensor],
        n_steps: int = 500,
        lr: float = 1e-3,
    ) -> List[float]:
        # Adam optimizer, returns loss history

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**CompressedSegmentCache 수정**:

`src/cache/compressed_segment.py`의 `__init__`에 파라미터 추가:
```python
def __init__(
    self,
    codec,          # HadamardInt4Codec or CompressionCodec
    chunk_size: int = 128,
    max_entries: int = 1000,
    adapter: Optional["SegmentAdapter"] = None,  # NEW
) -> None:
    super().__init__(chunk_size=chunk_size, max_entries=max_entries)
    self.codec = codec
    self.adapter = adapter
    self._key_layer: dict = {}
```

`get_segments()` 수정: 비연속 히트 발생 시 adapter 적용:
```python
# After decompressing kv:
is_noncontiguous = any(m < idx for m in miss_set)
if is_noncontiguous and self.adapter is not None:
    with torch.no_grad():
        kv = self.adapter(kv)
hits.append((i, kv))
```

### 3. MultiNodeScheduler (Activity A)

**파일**: `src/scheduler/multi_node_scheduler.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch

from src.scheduler.cache_aware_scheduler import CacheAwareScheduler
from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class NodeConfig:
    node_id: str
    node_type: str              # "prefill" or "decode"
    transfer_latency_ms: float = 10.0
    memory_capacity_gb: float = 80.0
    current_load: float = 0.0   # 0.0–1.0


class MultiNodeScheduler(CacheAwareScheduler):
    def __init__(
        self,
        cache: CacheStore,
        prefill_nodes: List[NodeConfig],
        decode_nodes: List[NodeConfig],
        codec=None,                             # HadamardInt4Codec for compress_before_transfer
        compress_threshold_bytes: int = 1024 * 1024,
        fairness_max_wait: int = 10,
        chunk_size: int = 128,
    ) -> None:
        super().__init__(cache, fairness_max_wait, chunk_size)
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes
        self.codec = codec
        self.compress_threshold_bytes = compress_threshold_bytes
        self._transfer_log: List[dict] = []

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        # Call parent schedule() for hit-rate ordering
        # Additionally annotate each request with routed (prefill_node, decode_node)
        # routing_score = hit_rate(req) / (1 + transfer_cost(p, d))

    def route(self, request: InferenceRequest) -> Tuple[NodeConfig, NodeConfig]:
        # Select prefill node with highest hit_rate for this request's chunks
        # Select decode node with lowest current_load
        # Return (prefill_node, decode_node)

    def simulate_transfer(
        self,
        kv: torch.Tensor,
        prefill_node: NodeConfig,
        decode_node: NodeConfig,
    ) -> Tuple[torch.Tensor, float]:
        # If kv.nbytes > compress_threshold_bytes and codec is not None:
        #   compressed = codec.encode(kv, layer_idx=5); latency *= 0.25
        #   kv_out = codec.decode(compressed, layer_idx=5)
        # Else: kv_out = kv
        # latency_ms = prefill_node.transfer_latency_ms * (1 + decode_node.current_load)
        # Return (kv_out, latency_ms)

    def node_load(self) -> dict:
        # Returns {node_id: current_load} for all nodes
```

**단일노드 fallback**: `prefill_nodes=[]`, `decode_nodes=[]`이면 `CacheAwareScheduler.schedule()`과 동일 동작.

---

## 테스트 명세

### tests/unit/test_tri_state_compressor.py

```python
def test_classify_ratios():
    # 100 tokens, verify len(retain)≈20, len(compress)≈40, len(evict)≈40

def test_encode_decode_roundtrip_kl():
    # KL(decode(encode(kv)), kv) < 0.01

def test_encode_decode_cosine():
    # cosine_similarity(decoded, original) >= 0.90 for non-evicted

def test_compression_ratio_above_75pct():
    # compressor.compression_ratio() <= 0.25 (i.e., ≥75% savings)

def test_accuracy_preservation_proxy():
    # ±1% perplexity proxy: KL < 0.01 for mixed retain+compress tiers
```

### tests/unit/test_segment_adapter.py

```python
def test_forward_shape_preserved():
    # adapter.forward(x).shape == x.shape

def test_residual_connection():
    # With zero-init mlp weights, forward(x) ≈ x

def test_training_reduces_loss():
    # fit() loss[-1] < loss[0] after 100 steps

def test_noncontiguous_correction():
    # perturbed_kv = kv + noise; fit(perturbed, kv)
    # KL(adapter(perturbed), kv) < KL(perturbed, kv)
```

### tests/unit/test_multi_node_scheduler.py

```python
def test_schedule_returns_all_requests():
    # len(scheduled) == len(input)

def test_high_hit_rate_scheduled_first():
    # warm request (matching chunks) before cold request

def test_fairness():
    # cold request scheduled within fairness_max_wait steps

def test_route_returns_valid_nodes():
    # route() returns (NodeConfig, NodeConfig) with correct node_types

def test_compress_before_transfer():
    # large kv tensor → simulate_transfer uses codec → shape preserved

def test_single_node_fallback():
    # prefill_nodes=[], decode_nodes=[] → same ordering as parent

def test_node_load_dict():
    # node_load() returns dict with all node IDs as keys
```

---

## 설정 파일

**`configs/experiments/2026-04-30.yaml`**:
```yaml
experiment: "2026-04-30-abc-multinode-tristate-adapter"
date: "2026-04-30"
activities: [A, B, C]

cache:
  type: CompressedSegmentCache
  codec: HadamardInt4Codec
  chunk_size: 128
  max_entries: 1000
  adapter:
    enabled: true
    kv_dim: 64
    hidden_dim: 64
    n_train_steps: 500
    lr: 0.001

compressor:
  type: TriStateCompressor
  retain_ratio: 0.20
  evict_ratio: 0.40

scheduler:
  type: MultiNodeScheduler
  prefill_nodes: 2
  decode_nodes: 2
  transfer_latency_ms: 10.0
  compress_threshold_bytes: 1048576
  fairness_max_wait: 10

metrics:
  target_throughput_gain: 0.20
  target_memory_reduction: 0.75
  target_noncontiguous_hit_rate: 0.30
  max_kl_divergence: 0.01
  max_scheduling_overhead_pct: 5.0
```

---

## 구현 시 주의사항

1. `TriStateCompressor`는 `CacheStore` 상속 불필요 — 변환기 역할로 `CompressedSegmentCache` 내부에서 선택적으로 사용.
2. `SegmentAdapter`는 `torch.nn.Module` 상속. 추론 시 반드시 `eval()` 모드. no_grad() 사용.
3. `MultiNodeScheduler`는 `CacheAwareScheduler` 상속으로 기존 단위 테스트(`test_cache_aware_scheduler.py`) 그대로 통과해야 함.
4. 모든 테스트는 CPU 텐서로 통과해야 함 (GPU 불필요).
5. 기존 55개 테스트가 그대로 통과해야 하며, 신규 파일 추가 시 `__init__.py`에 import 추가.
6. SegmentAdapter의 `fit()` 수렴 확인: loss[0] > loss[-1] (최소 감소)로 테스트.

SPEC_SAVED
