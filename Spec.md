<!-- 변경 이유 (이전 Spec.md: 2026-05-27 대비):
이전 사이클(2026-05-27)은 B+C 조합이었다:
  - C-1 IndexMemLearnableIndexerLatentMemoryEvictionCodec (퇴거 = 잠재 압축 + 조건부 복원)
  - B-1 IndexMemSoftHitSegmentCache (소프트 히트 비연속 재사용)
  - Cross-1 IndexMemBCIntegrationPipeline (B+C 통합)
  - Cross-2 VeriCache + IndexMem 드래프트 수락률 향상

이번 사이클(2026-05-28)은 A+B 조합으로 전환된다.
핵심 전환:
  - Activity A 최우선: HexAGenT(arXiv 2605.16637) 기반 온라인-공개 DAG 워크플로우 스케줄러.
    에이전틱 LLM 워크로드를 온라인-공개 DAG로 모델링하고, (1) 워크플로우 독립 완료 시간
    지평선(standalone completion horizon) 실시간 추정, (2) SLO 위험 가중 우선순위
    (SLO-risk-weighted priority)로 ready 호출 스케줄링, (3) KV 캐시 용량 제약 +
    이기종 GPU 전송 지연을 스케줄링 목적함수에 통합하는 3-way 최적화.
  - Activity A 2순위: PegaFlow GIL-free Rust 외부 KV 커넥터 래퍼 +
    Bloom Filter 기반 RDMA 크로스-노드 세그먼트 라우터.
  - Activity B 3순위: PegaFlowIrminsulDistributedSegmentCache —
    Irminsul(05-26 기구현) MLA δ-회전 비연속 재사용을 PegaFlow RDMA를 통해
    원격 노드까지 확장.
  - Cross-1 (A+B): HexAGenT DAG + PegaFlow RDMA + Irminsul δ-회전
    분산 비연속 재사용 통합 파이프라인.

주요 신규 파일:
1. [신규] src/scheduler/hexagent_workflow_scheduler.py (A-1 HexAGenT DAG 스케줄러)
2. [신규] src/cache/pegaflow_kv_connector.py (A-2 PegaFlow CacheStore 래퍼)
3. [신규] src/scheduler/pegaflow_rdma_router.py (A-2 RDMA 크로스-노드 라우터)
4. [신규] src/cache/pegaflow_irminsul_distributed_cache.py (B-1 분산 비연속 캐시)
5. [신규] src/engine/hexagent_irminsul_pegaflow_pipeline.py (Cross-1 A+B 파이프라인)
6. [변경] src/metrics/hit_rate.py — DistributedHitRateMetrics 클래스 추가
7. [신규] configs/experiments/2026-05-28.yaml
8. [신규] configs/gpu_rdma_bandwidth_table.yaml (A100/H100/H200 RDMA 대역폭 사전 측정값)
9. [신규] configs/pegaflow_peer_nodes.yaml (피어 노드 등록 목록)
10. [신규] tests/unit/test_hexagent_workflow_scheduler.py
11. [신규] tests/unit/test_pegaflow_kv_connector.py
12. [신규] tests/unit/test_pegaflow_rdma_router.py
13. [신규] tests/unit/test_pegaflow_irminsul_distributed_cache.py
14. [신규] tests/integration/test_hexagent_irminsul_pegaflow_pipeline_e2e.py
15. [보존] 모든 이전 사이클 구현 파일 수정 금지.
-->

# Spec — 2026-05-28: HexAGenT DAG 워크플로우 스케줄러 + PegaFlow RDMA + Irminsul 분산 비연속 재사용 (A+B)

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-28.md`

**최우선 구현 타겟**: A-1 `HexAGeTWorkflowHorizonAwareKVCapacityScheduler`
**2순위 구현 타겟**: A-2 `PegaFlowRustKVConnectorRDMACrossNodeSegmentRouter`
**3순위 구현 타겟**: B-1 `PegaFlowIrminsulDistributedSegmentCache`
**4순위 구현 타겟**: Cross-1 `HexAGeTDAGPegaFlowRDMAIrminsulABPipeline`

**해결하려는 문제**:

- **Activity A (HexAGenT 온라인-공개 DAG 3-way 최적화)**:
  기존 모든 A 스케줄러(CacheTTL, PBKV, CONCUR, ThunderAgent, SideQuest, DualMap,
  TokenDance, IndexMemTTLFallback 등 28개 사이클 전체)는 워크플로우 전체 완료 시간 지평선을
  직접 목적함수로 삼지 않았으며, KV 캐시 용량 제약 + 이기종 GPU 전송 지연을 통합한
  3-way 최적화가 없었다. HexAGenT(arXiv 2605.16637)의 원리를 구현해 에이전틱 워크로드의
  SLO 달성 스케일 팩터를 95% 기준 20.1%, 99% 기준 33.0% 개선하면서 Scheduling Overhead
  TTFT p50 +5% 이내를 달성한다.

- **Activity A (PegaFlow GIL-free Rust 외부 KV 커넥터)**:
  Python 기반 KV 오프로딩 경로의 GIL 병목을 Rust 외부 프로세스 IPC로 완전히 제거하고,
  Bloom Filter 기반 원격 세그먼트 인덱스 + RDMA 크로스-노드 조회를 단일 CacheStore 래퍼로 통합한다.

- **Activity B (PegaFlow RDMA + Irminsul δ-회전 분산 비연속 재사용)**:
  Irminsul(05-26 기구현)의 MLA c_KV/k_r 분리 + δ-회전 위치 수정이 로컬 HBM/DRAM에서만
  동작한다는 한계를 극복해, PegaFlow RDMA를 통해 원격 노드의 비연속 세그먼트까지 조회하고
  δ-회전을 로컬에서 적용한다. 분산 비연속 히트율 목표: 전체 히트의 30% 이상.

- **Cross-1 (A+B 통합 파이프라인)**:
  HexAGenT DAG 스케줄러가 워크플로우 다음 단계의 세그먼트 수요를 예측하고,
  PegaFlow RDMA가 전송 계층을 담당하며, Irminsul 분산 캐시가 δ-회전으로 위치 수정해
  재사용하는 5단계 비동기 파이프라인.

---

## 아키텍처 다이어그램

```
┌───────────────────────────────────────────────────────────────────────────┐
│              HexAGenT + PegaFlow + Irminsul A+B Integration Pipeline      │
│                                                                           │
│  에이전틱 세션 수신                                                        │
│       │                                                                   │
│       ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │    HexAGeTWorkflowScheduler (Activity A-1)                        │   │
│  │                                                                    │   │
│  │  WorkflowDAG 초기화 (task_type, dependency_ids)                   │   │
│  │       │                                                            │   │
│  │  standalone_completion_horizon() 실시간 추정                       │   │
│  │  SLO-risk-weighted priority 계산                                   │   │
│  │  KV 용량 제약 + 이기종 GPU 전송 지연 반영                            │   │
│  │       │                                                            │   │
│  │  배치 결정 (50ms 사이클)                                             │   │
│  └───────────────────────────┬────────────────────────────────────────┘   │
│                              │                                            │
│                              ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │    PegaFlowIrminsulDistributedSegmentCache (Activity B-1)         │   │
│  │                                                                    │   │
│  │  세그먼트 분산 검색 순서:                                             │   │
│  │    1. 로컬 HBM (IrminsulMLASegmentCache 기구현)                   │   │
│  │    2. PegaFlow 로컬 호스트/SSD (GIL-free IPC)                     │   │
│  │    3. PegaFlow RDMA 원격 노드 (Bloom Filter → RDMA)               │   │
│  │    4. Miss → 재계산                                                │   │
│  │                                                                    │   │
│  │  히트 시: δ-rotation(k_r, target_pos - source_pos) 로컬 적용       │   │
│  │  c_KV: 위치-자유, 원격 수신 즉시 재사용                              │   │
│  └───────────────────────────┬────────────────────────────────────────┘   │
│                              │                                            │
│                              ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │    PegaFlowKVConnector (Activity A-2) — CacheStore 래퍼           │   │
│  │                                                                    │   │
│  │  Unix 소켓 IPC → Rust PegaFlow 프로세스 (GIL-free)                │   │
│  │  계층: GPU HBM → 호스트 DRAM → SSD → RDMA 원격                    │   │
│  │  PegaFlowRDMACrossNodeRouter: Bloom Filter 원격 인덱스 유지        │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  DistributedHitRateMetrics: {local_hard_hit, pegaflow_local_hit,          │
│                               rdma_remote_hit, miss} 4단계 보고           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling — HexAGenT DAG 스케줄러 (A-1) + PegaFlow RDMA 라우터 (A-2)
- [x] Activity B: Non-Contiguous KV Cache Reuse — PegaFlow RDMA + Irminsul 분산 비연속 재사용 (B-1)
- [ ] Activity C: KV Cache Compression (이번 사이클 미포함)

**스케줄링 결정 단위**: 배치(batch) — 50ms 사이클마다 ready 태스크 전체를 재정렬해 배치 결정.
**캐시 상태 접근 방법**: `HexAGeTWorkflowScheduler`가 `kv_available_bytes()` 헬퍼를 통해
현재 GPU HBM 여유 KV 용량을 조회하고, 배치 구성 시 KV 수요 합계가 `kv_available`을 초과하지
않도록 제한. gRPC heartbeat(100ms 주기)로 클러스터 전체 GPU HBM 가용량 집계.

---

## 목표

- [ ] 목표 1: Inference Throughput 베이스라인 대비 +20% 이상 tokens/sec (evaluation_criteria.md §1, §2)
- [ ] 목표 2: Scheduling Overhead TTFT p50 +5% 이내 — DAG 갱신 + 우선순위 계산 < 1ms/사이클 (§2 필수)
- [ ] 목표 3: Non-Contiguous Cache Hit Rate 전체 히트의 30% 이상이 비연속 구간에서 발생 (§3 높음)
- [ ] 목표 4: 분산 비연속 히트율 (local + pegaflow_local + rdma_remote) 합산 베이스라인 대비 +5%p 이상 (§3 높음)
- [ ] 목표 5: KV Memory Footprint 베이스라인 대비 +20% 이내 (§3 높음)
- [ ] 목표 6: 캐시 히트율 — 스케줄링 미적용 대비 +10%p 이상 (§2 높음)
- [ ] 목표 7: GIL Contention Rate < 1% — PegaFlow 연동 후 메인 루프 GIL 보유 시간 (신규 지표)
- [ ] 목표 8: RDMA 크로스-노드 전송 지연 p50 — 재계산 비용 대비 이점 검증 (신규 지표)
- [ ] 목표 9: 복합 Throughput 향상 단일 Activity 대비 추가 +5% 이상 (Cross-1, §5)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/scheduler/hexagent_workflow_scheduler.py` | A-1 | HexAGenT 온라인-공개 DAG 스케줄러; WorkflowDAG + TaskNode 자료구조; standalone_completion_horizon 실시간 추정; SLO-risk 우선순위; KV 용량 인식 배치 결정; 이기종 GPU 전송 지연 반영 |
| `src/cache/pegaflow_kv_connector.py` | A-2 | CacheStore 구현; PegaFlow Rust 프로세스와 Unix 소켓 IPC; GIL-free put/get/delete; MockPegaFlowConnector fallback 포함 |
| `src/scheduler/pegaflow_rdma_router.py` | A-2 | BaseScheduler 상속; Bloom Filter 기반 피어 세그먼트 인덱스; RDMA 크로스-노드 세그먼트 조회 라우팅; PeerRegistry |
| `src/cache/pegaflow_irminsul_distributed_cache.py` | B-1 | CacheStore 구현; 로컬 HBM → PegaFlow 로컬 → PegaFlow RDMA 원격 4단계 조회; δ-rotation 원격 적용; 재사용 비용 결정 로직; 4단계 히트율 집계 |
| `src/engine/hexagent_irminsul_pegaflow_pipeline.py` | Cross-1 | 5단계 A+B 통합 파이프라인; DAG 구성 → 세그먼트 분산 검색 → δ-rotation → DAG 갱신 비동기 루프 |
| `configs/experiments/2026-05-28.yaml` | 공통 | 이번 사이클 실험 설정 |
| `configs/gpu_rdma_bandwidth_table.yaml` | A-1 | A100/H100/H200 간 RDMA 대역폭 사전 측정값 (GB/s) |
| `configs/pegaflow_peer_nodes.yaml` | A-2 | 피어 노드 등록 목록 `{node_id, rdma_address, port}` |
| `tests/unit/test_hexagent_workflow_scheduler.py` | A-1 | DAG 구성, horizon 추정, SLO-risk 우선순위, KV 용량 배치 결정 단위 테스트 |
| `tests/unit/test_pegaflow_kv_connector.py` | A-2 | MockPegaFlowConnector로 CacheStore 인터페이스, put/get/delete, 비동기 put 단위 테스트 |
| `tests/unit/test_pegaflow_rdma_router.py` | A-2 | Bloom Filter 인덱스, 로컬 → RDMA 원격 라우팅 순서, PeerRegistry 단위 테스트 |
| `tests/unit/test_pegaflow_irminsul_distributed_cache.py` | B-1 | 4단계 조회 순서, δ-rotation 적용, 재사용 결정 임계값, 분산 히트율 단위 테스트 |
| `tests/integration/test_hexagent_irminsul_pegaflow_pipeline_e2e.py` | Cross-1 | 5단계 파이프라인 엔드-투-엔드 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/metrics/hit_rate.py` | `DistributedHitRateMetrics` 클래스 추가. `{local_hard_hit, pegaflow_local_hit, rdma_remote_hit, miss}` 4단계 히트율 집계. 기존 `WeightedHitRateMetrics`, `HitRateMetrics` 수정 금지 |

---

## 알고리즘 상세

### 1. HexAGeTWorkflowScheduler (Activity A-1) — `src/scheduler/hexagent_workflow_scheduler.py`

#### 자료구조

```python
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple
import time

TaskStatus = Literal["pending", "ready", "running", "done"]

@dataclass
class TaskNode:
    task_id: str
    task_type: Literal["planning", "tool_call", "synthesis", "refine"]
    kv_demand_estimate: int          # 추정 KV 바이트 수
    gpu_type_preference: str         # "A100" | "H100" | "H200" | "any"
    dependency_ids: List[str]        # 이 태스크가 의존하는 선행 task_id 목록
    slo_deadline: float              # 절대 시간 (time.monotonic() 기반)
    status: TaskStatus = "pending"
    # 런타임 추정값 (EMA 갱신)
    t_prefill_ms: float = 0.0
    t_decode_ms: float = 0.0
    t_kv_transfer_ms: float = 0.0
    assigned_gpu: Optional[str] = None


@dataclass
class WorkflowDAG:
    workflow_id: str
    nodes: Dict[str, TaskNode]       # task_id → TaskNode
    edges: List[Tuple[str, str]]     # (upstream_id, downstream_id)
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class HexAGeTSchedulerConfig:
    schedule_cycle_ms: float = 50.0       # 배치 결정 주기 (ms)
    risk_weight: float = 2.0              # SLO 위험 가중치
    alpha_gpu_affinity: float = 0.5       # GPU 어피니티 스코어 가중치
    heartbeat_interval_ms: float = 100.0  # gRPC heartbeat 주기 (ms)
    kv_size_per_token_bytes: int = 512    # 토큰당 KV 바이트 (레이어 포함 추정값)
    n_layers: int = 32
    ema_decay: float = 0.9                # 실행 시간 EMA 감쇠 계수
    rdma_bandwidth_table_path: str = "configs/gpu_rdma_bandwidth_table.yaml"
    seed: int = 42
```

#### standalone_completion_horizon 추정

```python
def standalone_completion_horizon(
    self,
    task: TaskNode,
    current_time: float,
) -> float:
    """태스크 독립 완료 시간 지평선 추정.

    알고리즘:
      horizon = current_time
               + task.t_prefill_ms
               + task.t_decode_ms
               + task.t_kv_transfer_ms

    t_prefill_ms: EMA 갱신값. 초기값: input_tokens × kv_size_per_token_bytes / gpu_throughput
    t_decode_ms:  EMA 갱신값. 초기값: output_tokens_estimate × decode_ms_per_token
    t_kv_transfer_ms: kv_demand_estimate / rdma_bandwidth(src_gpu, tgt_gpu)
                      rdma_bandwidth_table_path에서 로드한 사전 측정 테이블 참조

    Returns:
      horizon: float (monotonic 타임스탬프)
    """
```

#### SLO-risk 우선순위 계산

```python
def slo_risk_score(self, task: TaskNode, current_time: float) -> float:
    """SLO 위험 점수.

    알고리즘:
      horizon = standalone_completion_horizon(task, current_time)
      risk = max(0.0, horizon - task.slo_deadline) / task.slo_deadline
      return risk

    0.0 = 여유 있음, 1.0+ = SLO 위반 임박
    """

def priority(self, task: TaskNode, current_time: float) -> float:
    """배치 정렬 우선순위 (높을수록 먼저 처리).

    priority = base_priority + risk_weight × slo_risk_score(task)
    base_priority: task_type별 사전 정의 (planning > synthesis > tool_call > refine)
    """
```

#### KV 용량 인식 배치 결정

```python
def build_batch(
    self,
    ready_tasks: List[TaskNode],
    kv_available_bytes: int,
) -> List[TaskNode]:
    """KV 용량 제약 내에서 최대 우선순위 태스크로 배치 구성.

    알고리즘:
      current_time = time.monotonic()
      sorted_tasks = sorted(ready_tasks,
                            key=lambda t: self.priority(t, current_time),
                            reverse=True)
      batch = []
      remaining_kv = kv_available_bytes
      for task in sorted_tasks:
          if task.kv_demand_estimate <= remaining_kv:
              batch.append(task)
              remaining_kv -= task.kv_demand_estimate
          else:
              # KV 용량 부족 → 저우선순위 기존 KV 퇴거 시도
              evicted = self._try_evict_to_fit(task, remaining_kv)
              if evicted >= task.kv_demand_estimate:
                  batch.append(task)
                  remaining_kv -= (task.kv_demand_estimate - evicted)
      return batch

    스케줄링 결정 단위: 배치 (schedule_cycle_ms마다 전체 ready_tasks 재평가)
    캐시 상태 접근: kv_available_bytes 파라미터로 주입 (외부에서 HBM 여유 조회 후 전달)
    """
```

#### GPU 어피니티 스코어

```python
def gpu_affinity_score(
    self,
    task: TaskNode,
    gpu_id: str,
) -> float:
    """이기종 GPU 전송 비용 기반 어피니티 스코어 (낮을수록 선호).

    score = -T_kv_transfer_to(gpu_id) - alpha × T_exec(task, gpu_id)
    T_kv_transfer_to: rdma_bandwidth_table에서 현재 GPU → gpu_id 대역폭으로 계산
    T_exec: kv_demand_estimate / gpu_throughput_table[gpu_id]
    alpha: config.alpha_gpu_affinity (기본 0.5)
    """
```

#### DAG 런타임 갱신

```python
def reveal_dag_edges(
    self,
    workflow_id: str,
    new_edges: List[Tuple[str, str]],
    new_nodes: Optional[Dict[str, TaskNode]] = None,
) -> None:
    """태스크 완료 시 새로 공개된 의존성 DAG에 추가 (온라인-공개 DAG).

    알고리즘:
      dag = self._dags[workflow_id]
      dag.edges.extend(new_edges)
      if new_nodes:
          dag.nodes.update(new_nodes)
      self._update_ready_status(dag)  # pending → ready 전환 재계산
    """

def schedule(self, requests: List) -> List:
    """BaseScheduler 인터페이스 구현. requests를 TaskNode로 변환 후 build_batch 적용."""
```

---

### 2. PegaFlowKVConnector (Activity A-2) — `src/cache/pegaflow_kv_connector.py`

```python
from enum import IntEnum
from typing import Optional
import torch
from src.cache.base import CacheStore


class OpCode(IntEnum):
    GET = 1
    PUT = 2
    DELETE = 3


@dataclass
class PegaFlowConnectorConfig:
    socket_path: str = "/tmp/pegaflow.sock"   # Unix 소켓 경로
    async_put: bool = True                     # True: put()이 즉시 반환 (비동기)
    timeout_ms: int = 100                      # get() 타임아웃 (ms)
    use_mock: bool = True                      # True: MockPegaFlowConnector 사용
    seed: int = 42


class PegaFlowKVConnector(CacheStore):
    """PegaFlow Rust 외부 KV 캐시 프로세스와 통신하는 CacheStore 래퍼.

    통신 방식: Unix 소켓 IPC (zero-copy 목표, Python GIL 경유 최소화)
    실제 PegaFlow 프로세스 없는 환경에서는 MockPegaFlowConnector로 대체.

    계층 순서 (PegaFlow 내부): GPU HBM → 호스트 DRAM → SSD (PegaFlow 내부 정책)
    Python 코드에서는 단일 get/put 인터페이스로 투명하게 접근.

    알고리즘:
      put(key, value):
        if async_put: socket.send_nonblocking(PUT, key, value)  → 즉시 반환
        else: socket.send_recv(PUT, key, value)
      get(key) → Optional[Tensor]:
        response = socket.send_recv(GET, key, timeout=timeout_ms)
        return response.tensor if response.found else None
      delete(key):
        socket.send_nonblocking(DELETE, key)
      evict() → int:
        # PegaFlow 내부 정책 위임: 로컬 통계 기반 추정값 반환
        return self._eviction_stats.last_freed_bytes
    """

    def __init__(self, config: PegaFlowConnectorConfig) -> None: ...

    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def delete(self, key: str) -> None: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...


class MockPegaFlowConnector(CacheStore):
    """PegaFlow 없는 테스트 환경용 in-memory fallback.

    동일 CacheStore 인터페이스를 Python dict로 모의 구현.
    GIL-free 동작 검증은 불가하지만 로직 단위 테스트에 사용.
    """

    def __init__(self, config: PegaFlowConnectorConfig) -> None:
        self._store: dict = {}
        self._hits: int = 0
        self._total: int = 0

    def put(self, key: str, value: torch.Tensor) -> None:
        self._store[key] = value

    def get(self, key: str) -> Optional[torch.Tensor]:
        self._total += 1
        v = self._store.get(key)
        if v is not None:
            self._hits += 1
        return v

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def evict(self) -> int:
        if not self._store:
            return 0
        key = next(iter(self._store))
        val = self._store.pop(key)
        return val.nelement() * val.element_size()

    def hit_rate(self) -> float:
        return self._hits / self._total if self._total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nelement() * v.element_size() for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._total = 0
```

---

### 3. PegaFlowRDMACrossNodeRouter (Activity A-2) — `src/scheduler/pegaflow_rdma_router.py`

```python
@dataclass
class PeerNodeEntry:
    node_id: str
    rdma_address: str
    port: int


@dataclass
class PegaFlowRDMARouterConfig:
    peer_nodes_config_path: str = "configs/pegaflow_peer_nodes.yaml"
    bloom_filter_capacity: int = 100_000    # Bloom Filter 예상 세그먼트 수
    bloom_filter_error_rate: float = 0.01   # 거짓 양성 허용 비율
    bloom_sync_interval_ms: float = 500.0   # Bloom Filter 동기화 주기 (ms)
    rdma_reuse_discount: float = 0.8        # RDMA 전송 비용 < 재계산 비용 × 0.8 시 재사용
    seed: int = 42


class PeerRegistry:
    """피어 노드별 세그먼트 Bloom Filter 인덱스.

    각 피어 노드의 보유 세그먼트 집합을 Bloom Filter로 근사 관리.
    거짓 양성 허용 (불필요한 RDMA 시도 가능) — 거짓 음성 방지 (히트 누락 방지).

    Bloom Filter 갱신: bloom_sync_interval_ms마다 gRPC로 피어에서 비트맵 수신.
    """

    def __init__(self, config: PegaFlowRDMARouterConfig) -> None: ...

    def has_segment(self, node_id: str, segment_id: bytes) -> bool:
        """피어 node_id가 segment_id를 보유할 가능성 반환 (Bloom Filter 조회)."""
        ...

    def update_bloom_from_peer(self, node_id: str, bloom_bitmap: bytes) -> None:
        """피어에서 수신한 Bloom Filter 비트맵으로 로컬 인덱스 갱신."""
        ...

    def register_local_segment(self, segment_id: bytes) -> None:
        """로컬 노드가 새 세그먼트를 획득 시 피어들에게 전파할 비트맵 갱신."""
        ...


class PegaFlowRDMACrossNodeRouter(BaseScheduler):
    """Bloom Filter 기반 피어 세그먼트 인덱스 + RDMA 크로스-노드 세그먼트 라우터.

    스케줄링 결정 단위: 요청 (세그먼트 조회 단위)
    캐시 상태 접근: PeerRegistry.has_segment() O(1) Bloom Filter 조회 후 RDMA 전송 결정

    알고리즘 (route_segment_request):
      1. 로컬 PegaFlow 조회 (local_connector.get(segment_id))
         → 히트: 반환 ("pegaflow_local")
      2. 피어 노드 Bloom Filter 순회 (has_segment(peer_id, segment_id))
         → 양성 피어 발견: RDMA 전송 비용 vs 재계산 비용 비교
           - rdma_latency < recompute_latency × rdma_reuse_discount → RDMA 조회 선택
           - 아니면 다음 피어 시도
      3. 전체 미스 → None 반환 ("miss")
    """

    def __init__(
        self,
        local_connector: PegaFlowKVConnector,
        peer_registry: PeerRegistry,
        config: PegaFlowRDMARouterConfig,
    ) -> None: ...

    def route_segment_request(
        self,
        segment_id: bytes,
        local_node_id: str,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """세그먼트 조회 라우팅.

        Returns:
          (kv_tensor, hit_type): hit_type ∈ {"pegaflow_local", "rdma_remote", "miss"}
        """
        ...

    def estimate_rdma_latency_ms(
        self,
        segment_size_bytes: int,
        target_node_id: str,
    ) -> float:
        """RDMA 전송 지연 추정: segment_size_bytes / rdma_bandwidth_gbps × 1000."""
        ...

    def estimate_recompute_latency_ms(
        self,
        segment_token_count: int,
        gpu_throughput_tokens_per_ms: float = 1.0,
    ) -> float:
        """재계산 비용 추정: segment_token_count / gpu_throughput_tokens_per_ms."""
        ...

    def schedule(self, requests: List) -> List:
        """BaseScheduler 인터페이스 구현 — 요청 리스트를 그대로 반환 (라우팅 사이드이펙트)."""
        return requests
```

---

### 4. PegaFlowIrminsulDistributedSegmentCache (Activity B-1) — `src/cache/pegaflow_irminsul_distributed_cache.py`

```python
from typing import Literal

HitType = Literal["local_hard_hit", "pegaflow_local_hit", "rdma_remote_hit", "miss"]


@dataclass
class IrminsulKVEntry:
    """Irminsul MLA 세그먼트 항목 (05-26 IrminsulMLASegmentCache 호환)."""
    segment_id: bytes
    c_kv_tensor: torch.Tensor   # [n_tokens, n_heads, d_kv] — 위치-자유
    k_r_tensor: torch.Tensor    # [n_tokens, n_heads, d_r]  — 위치-의존 (δ-rotation 필요)
    source_position: int         # 원래 시퀀스 위치 (δ 계산용)


@dataclass
class DistributedSegmentCacheConfig:
    rdma_reuse_discount: float = 0.8         # RDMA 재사용 결정 임계값 (YAML 외부화)
    bloom_sync_interval_ms: float = 500.0    # Bloom Filter 동기화 주기 (YAML 외부화)
    local_max_entries: int = 5000            # 로컬 HBM 최대 세그먼트 항목 수
    avg_chunk_size: int = 256                # CDC 청킹 평균 크기 (Irminsul 기본값)
    seed: int = 42


class PegaFlowIrminsulDistributedSegmentCache(CacheStore):
    """PegaFlow RDMA + Irminsul MLA δ-회전 분산 비연속 세그먼트 캐시 (Activity B-1).

    로컬 Irminsul 캐시(05-26 기구현) + PegaFlow 계층을 통합한 4단계 조회 경로:
      1. 로컬 HBM (IrminsulMLASegmentCache 기구현 활용)
      2. PegaFlow 로컬 호스트/SSD (PegaFlowKVConnector.get)
      3. PegaFlow RDMA 원격 노드 (PegaFlowRDMACrossNodeRouter.route_segment_request)
      4. Miss → 재계산

    δ-회전 적용:
      - c_KV: 위치-자유이므로 원격 수신 즉시 재사용 가능
      - k_r: source_position 기반 δ = target_position - source_position
              k_r_corrected = RoPE(k_r_tensor, δ) — 로컬 GPU에서 처리

    재사용 결정:
      rdma_transfer_latency < recompute_latency × rdma_reuse_discount → 원격 재사용

    4단계 히트율:
      DistributedHitRateMetrics.record(local_hard, pegaflow_local, rdma_remote, miss) 호출
    """

    def __init__(
        self,
        local_irminsul_cache: "IrminsulMLASegmentCache",   # 05-26 기구현, 타입 힌트만
        pegaflow_local: PegaFlowKVConnector,
        pegaflow_rdma_router: PegaFlowRDMACrossNodeRouter,
        config: DistributedSegmentCacheConfig,
    ) -> None: ...

    # ---- CacheStore 추상 메서드 구현 ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """로컬 HBM에 저장 + PegaFlow 로컬에도 비동기 put."""
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        """4단계 조회. 내부적으로 get_distributed를 호출."""
        ...

    def evict(self) -> int:
        """로컬 HBM에서 LRU 퇴거 후 PegaFlow에 비동기 put (오프로딩)."""
        ...

    def hit_rate(self) -> float:
        """전체 히트율 (local + pegaflow_local + rdma_remote) / total."""
        ...

    def memory_bytes(self) -> int:
        """로컬 HBM KV 메모리 크기."""
        ...

    def reset_stats(self) -> None: ...

    # ---- 분산 조회 핵심 API ----

    def get_distributed(
        self,
        segment_id: bytes,
        target_position: int,
    ) -> Tuple[Optional[torch.Tensor], HitType]:
        """4단계 분산 조회 + δ-회전 적용.

        알고리즘:
          # 1. 로컬 HBM (Irminsul 기존 경로)
          entry = local_irminsul_cache.get_entry(segment_id)
          if entry:
              kv = apply_delta_rotation(entry, target_position)
              metrics.record("local_hard_hit")
              return kv, "local_hard_hit"

          # 2. PegaFlow 로컬 호스트/SSD
          raw = pegaflow_local.get(segment_id.hex())
          if raw is not None:
              entry = deserialize_irminsul_entry(raw)
              kv = apply_delta_rotation(entry, target_position)
              metrics.record("pegaflow_local_hit")
              return kv, "pegaflow_local_hit"

          # 3. PegaFlow RDMA 원격 노드 (비용 비교 후 결정)
          rdma_result, hit_type_rdma = rdma_router.route_segment_request(
              segment_id, local_node_id
          )
          if rdma_result is not None and hit_type_rdma == "rdma_remote":
              entry = deserialize_irminsul_entry(rdma_result)
              kv = apply_delta_rotation(entry, target_position)
              metrics.record("rdma_remote_hit")
              return kv, "rdma_remote_hit"

          # 4. Miss
          metrics.record("miss")
          return None, "miss"
        """
        ...

    def apply_delta_rotation(
        self,
        entry: IrminsulKVEntry,
        target_position: int,
    ) -> torch.Tensor:
        """Irminsul δ-회전 위치 수정.

        알고리즘:
          delta = target_position - entry.source_position
          k_r_corrected = rope_rotate(entry.k_r_tensor, delta)
          # c_KV는 위치-자유이므로 delta 무관
          return concat(entry.c_kv_tensor, k_r_corrected, dim=-1)

        RoPE 회전: 기존 src/cache/irminsul_mla_segment_cache.py의
                   delta_rotate_k_r() 함수 import하여 재사용
        """
        ...

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        source_position: int,
        layer_idx: int = 0,
    ) -> bytes:
        """Irminsul 스타일 CDC 청킹 + 세그먼트 저장. segment_id(bytes) 반환."""
        ...

    def distributed_hit_rate_breakdown(self) -> dict:
        """4단계 히트율 분리 집계 반환.

        Returns:
          {
            "local_hard_hit_rate": float,
            "pegaflow_local_hit_rate": float,
            "rdma_remote_hit_rate": float,
            "miss_rate": float,
            "distributed_hit_rate": float,  # local + pegaflow_local + rdma_remote
          }
        """
        ...
```

---

### 5. DistributedHitRateMetrics (변경) — `src/metrics/hit_rate.py`에 추가

```python
@dataclass
class DistributedHitRateMetrics:
    """4단계 분산 비연속 히트율 지표.
    기존 WeightedHitRateMetrics, HitRateMetrics와 독립적.
    """
    total_lookups: int = 0
    local_hard_hits: int = 0
    pegaflow_local_hits: int = 0
    rdma_remote_hits: int = 0

    def record(self, hit_type: str) -> None:
        """hit_type: "local_hard_hit" | "pegaflow_local_hit" | "rdma_remote_hit" | "miss"."""
        self.total_lookups += 1
        if hit_type == "local_hard_hit":
            self.local_hard_hits += 1
        elif hit_type == "pegaflow_local_hit":
            self.pegaflow_local_hits += 1
        elif hit_type == "rdma_remote_hit":
            self.rdma_remote_hits += 1

    def distributed_hit_rate(self) -> float:
        """(local + pegaflow_local + rdma_remote) / total."""
        if self.total_lookups == 0:
            return 0.0
        return (
            self.local_hard_hits + self.pegaflow_local_hits + self.rdma_remote_hits
        ) / self.total_lookups

    def noncontiguous_rdma_fraction(self) -> float:
        """RDMA 원격 히트 / 전체 히트. 0이면 분산 재사용 없음."""
        total_hits = self.local_hard_hits + self.pegaflow_local_hits + self.rdma_remote_hits
        if total_hits == 0:
            return 0.0
        return self.rdma_remote_hits / total_hits

    def reset(self) -> None:
        self.total_lookups = 0
        self.local_hard_hits = 0
        self.pegaflow_local_hits = 0
        self.rdma_remote_hits = 0

    def summary(self) -> dict:
        return {
            "local_hard_hit_rate": self.local_hard_hits / max(1, self.total_lookups),
            "pegaflow_local_hit_rate": self.pegaflow_local_hits / max(1, self.total_lookups),
            "rdma_remote_hit_rate": self.rdma_remote_hits / max(1, self.total_lookups),
            "miss_rate": 1.0 - self.distributed_hit_rate(),
            "distributed_hit_rate": self.distributed_hit_rate(),
            "noncontiguous_rdma_fraction": self.noncontiguous_rdma_fraction(),
            "total_lookups": self.total_lookups,
        }
```

---

### 6. HexAGeTIrminsulPegaFlowPipeline (Cross-1) — `src/engine/hexagent_irminsul_pegaflow_pipeline.py`

```python
@dataclass
class CrossABPipelineConfig:
    prefetch_confidence_threshold: float = 0.7   # 선제 확보 신뢰도 임계값
    prefetch_hbm_budget_ratio: float = 0.10       # HBM의 10%를 선제 확보 예약
    async_prefetch: bool = True                   # 비동기 선제 확보 (현재 태스크 블로킹 방지)
    seed: int = 42


class HexAGeTIrminsulPegaFlowPipeline:
    """HexAGenT DAG 스케줄링 + PegaFlow RDMA + Irminsul δ-회전 A+B 통합 파이프라인.

    5단계 처리 흐름:
      Step 1 (DAG 구성 + SLO 위험 우선순위 배정, A-1):
        에이전틱 세션 수신 → WorkflowDAG 초기화
        → SLO 위험 가중 정렬 → build_batch (KV 용량 제약 포함)

      Step 2 (다음 단계 세그먼트 선제 예측 + 확보, A-1/B-1):
        현재 태스크 prefill 완료 → 후속 ready 태스크의 CDC 세그먼트 ID 예측
        confidence > prefetch_confidence_threshold 이면 PegaFlow 비동기 선제 확보 트리거
        prefetch_hbm_budget_ratio HBM 예약 공간 내에서 관리

      Step 3 (세그먼트 분산 검색, B-1):
        태스크 실행 시 PegaFlowIrminsulDistributedSegmentCache.get_distributed() 호출
        로컬 HBM → PegaFlow 로컬 → PegaFlow RDMA 원격 순서

      Step 4 (δ-회전 위치 수정, B-1/Irminsul):
        원격 수신 세그먼트의 k_r에 δ-회전 적용. c_KV 즉시 재사용.

      Step 5 (DAG 갱신 + 다음 사이클):
        태스크 완료 → reveal_dag_edges() → 우선순위 재계산 → Step 1 반복

    비동기 파이프라이닝:
      Step 2 선제 확보(asyncio.create_task)와 Step 3 분산 검색이 중첩 가능.
      Step 2에서 시작된 확보가 Step 3 시점에 완료되면 HBM에서 즉시 히트.
    """

    def __init__(
        self,
        dag_scheduler: HexAGeTWorkflowScheduler,
        distributed_cache: PegaFlowIrminsulDistributedSegmentCache,
        rdma_router: PegaFlowRDMACrossNodeRouter,
        config: CrossABPipelineConfig,
    ) -> None: ...

    async def process_workflow_session(
        self,
        workflow_id: str,
        initial_tasks: List[TaskNode],
    ) -> None:
        """워크플로우 세션 처리 (비동기 5단계 루프)."""
        ...

    def _predict_next_segment_ids(
        self,
        dag: WorkflowDAG,
        current_task_id: str,
    ) -> List[Tuple[bytes, float]]:
        """후속 태스크 CDC 세그먼트 ID + 신뢰도 예측.

        알고리즘:
          next_tasks = dag.get_successors(current_task_id)
          branch_factor = len(next_tasks)
          confidence = 1.0 / branch_factor if branch_factor > 0 else 0.0
          for task in next_tasks if confidence > 0:
              predicted_input = concat(
                  shared_system_prompt,
                  current_task_output_estimate,
                  task_type_template[task.task_type]
              )
              segment_ids = cdc_chunk(predicted_input)  # Irminsul CDC 재사용
              yield (segment_id, confidence)
        """
        ...

    async def _async_prefetch_segment(
        self,
        segment_id: bytes,
        confidence: float,
    ) -> None:
        """비동기 선제 확보 — 현재 태스크 실행 블로킹 없음.

        PegaFlow 로컬 → PegaFlow RDMA 원격 순서로 비동기 확보.
        HBM 예약 공간(prefetch_hbm_budget_ratio × HBM_total) 내에서 관리.
        공간 부족 시 가장 낮은 confidence 선제 확보 항목 제거.
        """
        ...
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-28.yaml
experiment:
  date: "2026-05-28"
  activity: "A+B"
  cache_type: "pegaflow_irminsul_distributed"    # Activity B-1
  compression_method: "none"                      # Activity C 미포함
  scheduler_type: "hexagent_workflow"             # Activity A-1
  seed: 42

# Activity A-1: HexAGenT Workflow Scheduler
hexagent_scheduler:
  schedule_cycle_ms: 50.0            # 배치 결정 주기
  risk_weight: 2.0                   # SLO 위험 가중치
  alpha_gpu_affinity: 0.5            # GPU 어피니티 가중치
  heartbeat_interval_ms: 100.0       # gRPC heartbeat 주기
  kv_size_per_token_bytes: 512       # 토큰당 KV 바이트 추정값
  n_layers: 32
  ema_decay: 0.9                     # 실행 시간 EMA 감쇠
  rdma_bandwidth_table_path: "configs/gpu_rdma_bandwidth_table.yaml"

# Activity A-2: PegaFlow KV Connector
pegaflow_connector:
  socket_path: "/tmp/pegaflow.sock"
  async_put: true
  timeout_ms: 100
  use_mock: true                     # 테스트 환경: MockPegaFlowConnector 사용

# Activity A-2: PegaFlow RDMA Router
pegaflow_rdma_router:
  peer_nodes_config_path: "configs/pegaflow_peer_nodes.yaml"
  bloom_filter_capacity: 100000
  bloom_filter_error_rate: 0.01
  bloom_sync_interval_ms: 500.0      # Bloom Filter 동기화 주기
  rdma_reuse_discount: 0.8           # RDMA 재사용 결정 임계값

# Activity B-1: 분산 비연속 세그먼트 캐시
distributed_segment_cache:
  rdma_reuse_discount: 0.8
  bloom_sync_interval_ms: 500.0
  local_max_entries: 5000
  avg_chunk_size: 256                # CDC 청킹 평균 크기 (Irminsul 기본값)

# Cross-1 A+B Pipeline
cross_ab_pipeline:
  prefetch_confidence_threshold: 0.7
  prefetch_hbm_budget_ratio: 0.10    # HBM 10% 선제 확보 예약
  async_prefetch: true

# 측정 지표 저장 경로
metrics:
  output_dir: "results/2026-05-28"
  metrics_file: "results/2026-05-28/metrics.json"
```

```yaml
# configs/gpu_rdma_bandwidth_table.yaml
# A100/H100/H200 간 RDMA 대역폭 사전 측정값 (GB/s)
# 실제 클러스터 측정값으로 교체 필요 (현재는 이론값 기반 기본값)
rdma_bandwidth_gbps:
  A100_to_A100: 200.0    # InfiniBand HDR (200Gbps)
  A100_to_H100: 200.0
  A100_to_H200: 200.0
  H100_to_H100: 400.0    # InfiniBand NDR (400Gbps)
  H100_to_H200: 400.0
  H200_to_H200: 800.0    # NVLink 900GB/s 기준 inter-node 추정
  default: 100.0          # 미등록 GPU 쌍 fallback
```

```yaml
# configs/pegaflow_peer_nodes.yaml
# 피어 노드 등록 목록 (테스트 환경에서는 빈 리스트 사용)
peer_nodes: []
# 실제 멀티 노드 환경 예시:
# peer_nodes:
#   - node_id: "node-1"
#     rdma_address: "192.168.1.101"
#     port: 9000
#   - node_id: "node-2"
#     rdma_address: "192.168.1.102"
#     port: 9000
```

---

## results/2026-05-28/metrics.json 저장 지표

```json
{
  "experiment_date": "2026-05-28",
  "activity": "A+B",

  "scheduling": {
    "tokens_per_sec_baseline": null,
    "tokens_per_sec_hexagent": null,
    "throughput_improvement_pct": null,
    "ttft_p50_baseline_ms": null,
    "ttft_p50_hexagent_ms": null,
    "ttft_p50_delta_pct": null,
    "dag_schedule_overhead_ms_mean": null,
    "dag_schedule_overhead_ms_p50": null,
    "dag_schedule_overhead_ms_p99": null,
    "slo_achievement_rate_95pct": null,
    "slo_achievement_rate_99pct": null,
    "kv_eviction_rate_baseline": null,
    "kv_eviction_rate_hexagent": null
  },

  "hit_rate": {
    "local_hard_hit_rate": null,
    "pegaflow_local_hit_rate": null,
    "rdma_remote_hit_rate": null,
    "miss_rate": null,
    "distributed_hit_rate": null,
    "noncontiguous_rdma_fraction": null,
    "noncontiguous_hit_rate_vs_total_30pct_target": null
  },

  "pegaflow_connector": {
    "gil_contention_rate_pct": null,
    "async_put_latency_p50_ms": null,
    "async_put_latency_p99_ms": null,
    "local_throughput_improvement_pct": null
  },

  "rdma_routing": {
    "rdma_transfer_latency_p50_ms": null,
    "rdma_transfer_latency_p99_ms": null,
    "recompute_latency_p50_ms": null,
    "rdma_reuse_decision_rate_pct": null,
    "bloom_filter_false_positive_rate": null
  },

  "cross_ab_pipeline": {
    "prefetch_hit_rate": null,
    "prefetch_confidence_threshold": 0.7,
    "prefetch_hbm_budget_used_ratio": null,
    "combined_throughput_improvement_pct": null,
    "single_activity_a_improvement_pct": null,
    "single_activity_b_improvement_pct": null
  },

  "kv_memory": {
    "baseline_kv_bytes": null,
    "distributed_cache_kv_bytes": null,
    "memory_footprint_delta_pct": null
  }
}
```

---

## 테스트 요구사항

### 필수 단위 테스트

- [ ] `tests/unit/test_hexagent_workflow_scheduler.py`
  - `test_workflow_dag_init_from_tasks()`: TaskNode + dependency_ids로 WorkflowDAG 생성
  - `test_standalone_completion_horizon_basic()`: horizon > current_time 검증
  - `test_slo_risk_score_zero_when_slack()`: horizon < deadline → risk = 0.0
  - `test_slo_risk_score_positive_when_overdue()`: horizon > deadline → risk > 0.0
  - `test_build_batch_kv_capacity_constraint()`: kv_available_bytes 초과 태스크 제외
  - `test_build_batch_priority_ordering()`: 높은 SLO risk 태스크가 먼저 선택
  - `test_reveal_dag_edges_updates_ready_status()`: 의존성 해소 후 pending → ready 전환
  - `test_gpu_affinity_score_prefers_low_transfer_cost()`: 전송 비용 낮은 GPU 선호
  - `test_ema_update_execution_time()`: EMA 갱신이 ema_decay 계수로 수렴
  - `test_schedule_interface_returns_list()`: BaseScheduler.schedule() 반환 타입 검증
  - `test_schedule_deterministic_with_seed()`: 동일 seed + 동일 입력 → 동일 배치

- [ ] `tests/unit/test_pegaflow_kv_connector.py`
  - `test_mock_connector_cache_store_interface()`: CacheStore 추상 메서드 전부 동작
  - `test_mock_put_get_round_trip()`: put → get 동일 텐서 반환
  - `test_mock_get_returns_none_on_miss()`: 저장 안 된 key → None
  - `test_mock_hit_rate_tracking()`: put/get 후 hit_rate() 계산
  - `test_mock_evict_frees_memory()`: evict() 후 memory_bytes() 감소
  - `test_mock_delete_removes_key()`: delete 후 get → None
  - `test_mock_reset_stats_clears_counters()`: reset_stats() 후 hit_rate() = 0.0
  - `test_connector_factory_uses_mock_when_configured()`: use_mock=True → MockPegaFlowConnector

- [ ] `tests/unit/test_pegaflow_rdma_router.py`
  - `test_peer_registry_has_segment_false_for_unknown()`: 등록 안 된 세그먼트 → False
  - `test_peer_registry_has_segment_true_after_register()`: 등록 후 → True
  - `test_route_returns_pegaflow_local_on_local_hit()`: 로컬 히트 → "pegaflow_local"
  - `test_route_returns_miss_when_no_peers()`: 피어 없음 + 로컬 미스 → "miss"
  - `test_rdma_reuse_decision_below_discount()`: RDMA 비용 < 재계산 × discount → RDMA 선택
  - `test_rdma_reuse_decision_above_discount()`: RDMA 비용 >= 재계산 × discount → 미스
  - `test_estimate_rdma_latency_formula()`: segment_size / bandwidth × 1000 수치 검증
  - `test_estimate_recompute_latency_formula()`: token_count / throughput 수치 검증
  - `test_schedule_returns_input_unchanged()`: schedule() 입력 그대로 반환

- [ ] `tests/unit/test_pegaflow_irminsul_distributed_cache.py`
  - `test_cache_store_interface_all_methods()`: CacheStore 추상 메서드 전부 동작
  - `test_get_distributed_local_hard_hit()`: 로컬 HBM 히트 → "local_hard_hit"
  - `test_get_distributed_pegaflow_local_hit()`: PegaFlow 로컬 히트 → "pegaflow_local_hit"
  - `test_get_distributed_miss_all_layers()`: 전 계층 미스 → "miss"
  - `test_delta_rotation_c_kv_unchanged()`: c_KV는 δ와 무관하게 동일
  - `test_delta_rotation_k_r_changes_with_delta()`: δ≠0이면 k_r 변환됨
  - `test_hit_rate_breakdown_sums_to_one()`: 4단계 비율 합 = 1.0
  - `test_distributed_hit_rate_above_local_only()`: 로컬 + 원격 히트율 > 로컬 히트율
  - `test_evict_offloads_to_pegaflow()`: evict() 후 PegaFlow에 비동기 put 호출됨

### 필수 통합 테스트

- [ ] `tests/integration/test_hexagent_irminsul_pegaflow_pipeline_e2e.py`
  - `test_cross_ab_pipeline_full_workflow_flow()`: 5단계 파이프라인 완전 흐름 (Mock 환경)
  - `test_cross_ab_dag_scheduling_improves_throughput()`: DAG 스케줄러 적용 후 처리량 향상
  - `test_cross_ab_distributed_hit_rate_above_local()`: 분산 히트율 > 로컬 히트율
  - `test_cross_ab_prefetch_reduces_miss_rate()`: 선제 확보로 미스율 감소 확인
  - `test_cross_ab_ttft_overhead_within_5pct()`: TTFT p50 증가 +5% 이내
  - `test_cross_ab_cachestore_interface_compat()`: PegaFlowIrminsulDistributedSegmentCache를
    InferenceRunner가 CacheStore로 사용 가능
  - `test_cross_ab_base_scheduler_interface_compat()`: HexAGeTWorkflowScheduler.schedule()이
    BaseScheduler 인터페이스를 통해 정상 동작

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 100% 통과**: 위 명시된 모든 단위 테스트 케이스 통과
2. **통합 테스트 100% 통과**: `test_hexagent_irminsul_pegaflow_pipeline_e2e.py` 전체 통과
3. **기존 테스트 회귀 없음**: 이전 사이클 구현의 모든 단위·통합 테스트 계속 통과
4. **evaluation_criteria.md §2 (Activity A) 필수 항목 모두 Pass**:
   - Scheduling Overhead TTFT p50 증가 +5% 이내 (필수)
   - 캐시 히트율 향상 스케줄링 미적용 대비 +10%p 이상 (높음)
   - 요청 처리 공정성 최대 대기 시간 2× 미초과 (높음)
5. **evaluation_criteria.md §3 (Activity B) 기준 충족**:
   - 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상 (높음)
   - 비연속 세그먼트 히트율 전체 히트의 30% 이상 (높음)
   - KV Memory Footprint 베이스라인 대비 +20% 이내 (높음)
6. **evaluation_criteria.md §5 (크로스 조합) 기준**:
   - 복합 Throughput 향상 단일 Activity 대비 추가 +5% 이상 (높음)
7. **CacheStore 인터페이스 준수**: PegaFlowKVConnector, PegaFlowIrminsulDistributedSegmentCache 추상 메서드 전부 구현
8. **BaseScheduler 인터페이스 준수**: HexAGeTWorkflowScheduler.schedule(), PegaFlowRDMACrossNodeRouter.schedule() 구현
9. **설정 파일 존재**: `configs/experiments/2026-05-28.yaml`, `configs/gpu_rdma_bandwidth_table.yaml`, `configs/pegaflow_peer_nodes.yaml`
10. **metrics.json 생성**: `results/2026-05-28/metrics.json`에 모든 지표 기록
11. **시드 고정 재현성**: seed=42로 동일 결과 재현 가능
12. **MockPegaFlowConnector 제공**: PegaFlow 프로세스 없는 환경에서 단위 테스트 전부 통과

---

## 구현 우선순위 순서

1. A-1: `TaskNode`, `WorkflowDAG` 자료구조 + `HexAGeTWorkflowScheduler` (DAG 초기화, horizon 추정, SLO-risk 우선순위, build_batch)
2. A-2: `MockPegaFlowConnector` + `PegaFlowKVConnector` (use_mock=True 기본값으로 즉시 동작)
3. A-2: `PeerRegistry` + `PegaFlowRDMACrossNodeRouter` (Bloom Filter 인덱스, RDMA 라우팅)
4. B-1: `IrminsulKVEntry`, `DistributedSegmentCacheConfig` + `PegaFlowIrminsulDistributedSegmentCache` (4단계 조회, δ-회전)
5. 지표: `DistributedHitRateMetrics` (src/metrics/hit_rate.py 추가)
6. Cross-1: `CrossABPipelineConfig` + `HexAGeTIrminsulPegaFlowPipeline` (5단계 비동기 파이프라인)
7. 설정 파일: `configs/experiments/2026-05-28.yaml`, `configs/gpu_rdma_bandwidth_table.yaml`, `configs/pegaflow_peer_nodes.yaml`
8. 단위 테스트 전부 (A-1 → A-2 → B-1 → Cross-1 순서)
9. 통합 테스트

---

## 보존 파일 (수정 금지)

이전 사이클 구현 파일은 수정하지 않는다:

- `src/cache/irminsul_mla_segment_cache.py` (05-26 B-1) — 로컬 Irminsul 캐시, B-1에서 import만
- `src/cache/cdc_content_hash_interface.py` (05-26 B-2) — CDC 주소 체계, 재사용만
- `src/cache/arch_aware_noncontiguous_router.py` (05-26)
- `src/cache/mla_two_axis_compression_codec.py` (05-26 C-1)
- `src/scheduler/objectcache_s3_tier_router.py` (05-26 A-1)
- `src/engine/irminsul_objectcache_pipeline.py` (05-26 Cross-1)
- `src/cache/indexmem_eviction_codec.py` (05-27 C-1)
- `src/cache/indexmem_latent_memory_module.py` (05-27 C-1)
- `src/cache/indexmem_learnable_indexer.py` (05-27 C-1)
- `src/cache/indexmem_soft_hit_segment_cache.py` (05-27 B-1)
- `src/engine/indexmem_bc_pipeline.py` (05-27 Cross-1)
- 기타 모든 이전 사이클 파일 — 기존 단위·통합 테스트 회귀 없이 통과해야 한다.

**주의**: `src/metrics/hit_rate.py`에는 `DistributedHitRateMetrics` 클래스만 추가한다.
기존 `WeightedHitRateMetrics`, `HitRateMetrics` 클래스와 메서드는 수정하지 않는다.
