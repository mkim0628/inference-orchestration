<!-- 변경 이유 (이전 Spec.md: 2026-05-26 대비):
이전 사이클(2026-05-26)은 A+B(+C 시너지) 조합이었다:
  - B-1 IrminsulMLANativeDeltaRotationArchAwareNonContiguousRouter (MLA δ-회전 비연속 재사용)
  - A-1 ObjectCacheS3TierBreakEvenRoutingPolicy (S3 4번째 계층 라우터)
  - B-2 CDCContentHashUnifiedSegmentIDInterface (통합 주소 체계)
  - Cross-1 IrminsulObjectCacheCDCLayerwiseRDMAMLAPipeline (A+B 통합)
  - C-1 MLATwoAxisCompressionCodec (선택적 C 시너지)

이번 사이클(2026-05-27)은 B+C 조합으로 전환된다.
핵심 전환:
  - Activity C 최우선: IndexMem(arXiv 2605.25475)의 "퇴거 = 잠재 압축 + 조건부 복원"
    패러다임을 Learnable Indexer (225-param MLP) + Latent Memory Module (경량 트랜스포머 인코더)로
    구현한다. 과거 27개 사이클의 모든 C 기법에서 퇴거 후 잠재 기억으로 어텐션 기여를 보상하는
    기법이 전무했다.
  - Activity B 2순위: IndexMem 잠재 기억 소프트 히트 경로를 비연속 세그먼트 캐시에 통합해
    비연속 재사용의 이진(binary) 모델을 연속체로 확장한다.
  - Cross-1 B+C: UnifiedLatentPool로 세그먼트-레벨(B)과 토큰-레벨(C) 잠재 상태를 통합 관리.
  - Cross-2 (Low-effort): VeriCacheSpeculativeCodec의 set_draft_codec()에 IndexMemEvictionCodec을
    플러그인으로 연결한다 (기구현 VeriCache 재활용, 신규 코드 최소).

주요 변경:
1. [신규] src/cache/indexmem_learnable_indexer.py (C-1 Learnable Indexer MLP)
2. [신규] src/cache/indexmem_latent_memory_module.py (C-1 Latent Memory Module)
3. [신규] src/cache/indexmem_eviction_codec.py (C-1 통합 코덱, CompressionCodec + DraftCodec 구현)
4. [신규] src/cache/indexmem_soft_hit_segment_cache.py (B-1 소프트 히트 비연속 캐시)
5. [신규] src/engine/indexmem_bc_pipeline.py (Cross-1 B+C 통합 파이프라인)
6. [변경] src/metrics/hit_rate.py — weighted_hit_rate 지표 추가
7. [신규] configs/experiments/2026-05-27.yaml
8. [신규] configs/indexmem_indexer_weights.yaml (Learnable Indexer 가중치 저장)
9. [신규] tests/unit/test_indexmem_learnable_indexer.py
10. [신규] tests/unit/test_indexmem_latent_memory_module.py
11. [신규] tests/unit/test_indexmem_eviction_codec.py
12. [신규] tests/unit/test_indexmem_soft_hit_segment_cache.py
13. [신규] tests/unit/test_compression_accuracy.py — IndexMem 케이스 추가
14. [신규] tests/integration/test_indexmem_bc_pipeline_e2e.py (Cross-1)
15. [보존] 모든 이전 사이클 구현 파일 수정 금지 (irminsul, objectcache, vericache, kv_packet 등).
-->

# Spec — 2026-05-27: IndexMem Learnable Indexer + Latent Memory Eviction Codec (B+C)

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-27.md`

**최우선 구현 타겟**: C-1 `IndexMemLearnableIndexerLatentMemoryEvictionCodec`
**2순위 구현 타겟**: Cross-2 `IndexMemVeriCacheLatentDraftVerifyPipeline` (플러그인 연결)
**3순위 구현 타겟**: Cross-1 `IndexMemLatentSegmentBCIntegrationPipeline` (B+C 통합)
**4순위 구현 타겟**: B-1 `IndexMemSoftHitSegmentCache`

**해결하려는 문제**:

- **Activity C (IndexMem "퇴거 = 잠재 압축 + 조건부 복원" 패러다임)**:
  기존 모든 퇴거 기법(H2O, SnapKV, PyramidKV, DapQ 등)은 "어떤 토큰을 유지할 것인가(선택)"에만
  집중하며, 퇴거된 토큰의 정보는 영구히 폐기한다. IndexMem(arXiv 2605.25475)은 (a) 학습
  가능한 중요도 예측기(Learnable Indexer, 225-param MLP)로 입력-적응적 보존 결정을 내리고,
  (b) Latent Memory Module(경량 트랜스포머 인코더)로 퇴거 토큰을 온라인-갱신 잠재 상태로
  압축 보존해 디코딩 중 잔차 리드아웃(residual readout)으로 어텐션 기여를 보상한다.
  RULER 스위트에서 SnapKV/PyramidKV 대비 최대 25포인트 개선.

- **Activity B (소프트 히트 비연속 재사용)**:
  기존 비연속 세그먼트 재사용은 물리적 KV가 캐시에 존재할 때만 히트를 발생시키는 이진 모델이다.
  IndexMem의 잠재 기억 모듈을 비연속 세그먼트 캐시에 통합해, 물리적으로 퇴거된 세그먼트라도
  잠재 상태가 존재하면 "소프트 히트"로 잔차 리드아웃을 제공한다. 비연속 재사용을 이진 → 연속체로 확장.

- **Cross-2 (VeriCache + IndexMem 드래프트 수락률 향상)**:
  VeriCacheSpeculativeCodec(05-25 기구현)의 set_draft_codec() 플러그인 인터페이스에
  IndexMemEvictionCodec을 연결해 드래프트 품질을 향상시킨다. 신규 코드 최소.

---

## 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     IndexMem B+C Integration Pipeline                    │
│                                                                          │
│  입력 토큰 시퀀스                                                         │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │          IndexMemSoftHitSegmentCache (Activity B-1)             │    │
│  │                                                                 │    │
│  │  세그먼트 해시 조회                                               │    │
│  │    ├─ Hard Hit  → 물리적 KV 텐서 반환                            │    │
│  │    ├─ Soft Hit  → SegmentLatentPool.readout(query) → 잔차 반환  │    │
│  │    └─ Miss      → 재계산 필요                                    │    │
│  │                                                                 │    │
│  └───────────────────────────┬─────────────────────────────────────┘    │
│                              │  Hard Hit 경로 (물리적 KV)                │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │          IndexMemEvictionCodec (Activity C-1)                   │    │
│  │                                                                 │    │
│  │  IndexMemLearnableIndexer                                       │    │
│  │    입력: [k_norm, v_norm, cumul_attn, pos_decay, query_sim]     │    │
│  │    출력: retention_prob [0,1] per token                         │    │
│  │       │                                                         │    │
│  │       ▼  budget_ratio로 분리                                    │    │
│  │  ┌─────────┐   ┌──────────────┐                                │    │
│  │  │ to_keep │   │  to_evict    │                                │    │
│  │  │  (물리  │   │              │                                │    │
│  │  │  KV)   │   │              │                                │    │
│  │  └────┬────┘   └──────┬───────┘                               │    │
│  │       │               ▼                                        │    │
│  │       │     IndexMemLatentMemoryModule                         │    │
│  │       │       encode_evicted() → latent_state                  │    │
│  │       │       online-update: α × new + (1-α) × prev           │    │
│  │       │               │                                        │    │
│  │       │               ▼                                        │    │
│  │       │     디코딩 시: residual_readout(query, latent_state)   │    │
│  │       │       attn_output += β × readout                      │    │
│  │       │                                                        │    │
│  └───────┴────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │     UnifiedLatentPool (Cross-1 공유 인프라)                     │    │
│  │                                                                 │    │
│  │  segment_id → segment_latent  (B-1 세그먼트 레벨)               │    │
│  │  (segment_id, token_range) → token_latent  (C-1 토큰 레벨)     │    │
│  │                                                                 │    │
│  │  SharedLatentEncoder (B-1 + C-1 파라미터 공유)                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Cross-2: VeriCacheSpeculativeCodec.set_draft_codec(IndexMemEviction)   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling  (이번 사이클 미포함)
- [x] Activity B: Non-Contiguous KV Cache Reuse — IndexMem Soft Hit (B-1)
- [x] Activity C: KV Cache Compression — IndexMem Learnable Indexer + Latent Memory (C-1)

---

## 목표

- [ ] 목표 1: KV Cache Memory Reduction −40% 이상 (budget_ratio=0.5 기준) (evaluation_criteria.md §4)
- [ ] 목표 2: Compression Accuracy Delta ±1% 이내 (WikiText-2 perplexity + RULER 스위트) (§4 필수)
- [ ] 목표 3: Compression Accuracy Delta 실측 목표 ±0.3~0.8% (RULER-4K/16K 기준, IndexMem 원논문 근거) (§4)
- [ ] 목표 4: 비연속 가중 히트율(weighted_hit_rate) 베이스라인 대비 +15~25%p (소프트 히트 포함) (§3)
- [ ] 목표 5: Non-Contiguous Soft Hit Rate (n_soft_hits / total) 측정 및 보고 (신규 지표)
- [ ] 목표 6: Cross-2 IndexMem 드래프트 수락률 vs 기존 Int8/TokenEviction 코덱 비교 (§5)
- [ ] 목표 7: 복합 처리량 향상 (Cross-1) 베이스라인 대비 +20% 이상 (§5 복합 Throughput 향상)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/indexmem_learnable_indexer.py` | C-1 | 225-param MLP 기반 입력-적응적 토큰 중요도 예측기; zero-shot DapQ fallback 포함 |
| `src/cache/indexmem_latent_memory_module.py` | C-1 | 경량 트랜스포머 인코더 기반 잠재 기억 모듈; 온라인-갱신 잠재 상태 관리 + 잔차 리드아웃 |
| `src/cache/indexmem_eviction_codec.py` | C-1 | CompressionCodec + DraftCodec 구현; Learnable Indexer + Latent Memory 통합 코덱 |
| `src/cache/indexmem_soft_hit_segment_cache.py` | B-1 | CacheStore 구현; 소프트 히트 경로 + HitResult 반환 타입 + SegmentLatentPool |
| `src/engine/indexmem_bc_pipeline.py` | Cross-1 | B+C 통합 파이프라인; UnifiedLatentPool + SharedLatentEncoder 공유 |
| `configs/experiments/2026-05-27.yaml` | 공통 | 이번 사이클 실험 설정 |
| `configs/indexmem_indexer_weights.yaml` | C-1 | Learnable Indexer 학습 완료 가중치 저장 위치 |
| `tests/unit/test_indexmem_learnable_indexer.py` | C-1 | Learnable Indexer MLP 단위 테스트 |
| `tests/unit/test_indexmem_latent_memory_module.py` | C-1 | Latent Memory Module 단위 테스트 |
| `tests/unit/test_indexmem_eviction_codec.py` | C-1 | IndexMemEvictionCodec 단위 테스트 |
| `tests/unit/test_indexmem_soft_hit_segment_cache.py` | B-1 | 소프트 히트 경로 + CacheStore 인터페이스 단위 테스트 |
| `tests/integration/test_indexmem_bc_pipeline_e2e.py` | Cross-1 | B+C 통합 엔드-투-엔드 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/metrics/hit_rate.py` | `WeightedHitRateMetrics` 클래스 추가 (weighted_hit_rate, soft_hit_rate 지표). 기존 `HitRateMetrics` 수정 금지 |
| `tests/unit/test_compression_accuracy.py` | IndexMemEvictionCodec accuracy 검증 케이스 추가 (기존 케이스 보존) |

---

## 알고리즘 상세

### 1. IndexMemLearnableIndexer (Activity C-1) — `src/cache/indexmem_learnable_indexer.py`

```python
@dataclass
class LearnableIndexerConfig:
    input_dim: int = 5                # [k_norm, v_norm, cumul_attn_score, position_decay, query_sim]
    hidden_dim: int = 32
    output_dim: int = 1               # retention_prob
    gamma_position_decay: float = 0.01
    ema_alpha_cumul_attn: float = 0.1  # EMA 계수로 cumul_attn_score 갱신
    zero_shot_mode: bool = False       # True: DapQ fallback (Learnable Indexer 비활성)
    weights_path: Optional[str] = None # 학습 완료 가중치 경로 (YAML 외부화)
    seed: int = 42


class IndexMemLearnableIndexer:
    """IndexMem Learnable Indexer — 입력-적응적 KV 토큰 중요도 예측기.

    아키텍처:
      inputs = [k_norm, v_norm, cumul_attn_score, position_decay, query_sim]
      hidden = ReLU(Linear(5 → 32)(inputs))
      retention_prob = Sigmoid(Linear(32 → 1)(hidden))

    파라미터 수: 5×32 + 32 + 32×1 + 1 = 225개 (극도로 경량)

    zero_shot_mode=True 시 DapQ 방식으로 대체:
      retention_prob = Sigmoid(query_sim × cumul_attn_score)
      — 학습 데이터 없이 즉시 동작하는 fallback.

    학습:
      LoRA rank=4, 어텐션 레이어만, 에폭 3~5.
      data/processed/ 내 대표 샘플 + cumul_attn_score 기반 자기지도.
      학습 완료 가중치: configs/indexmem_indexer_weights.yaml.
    """

    def __init__(self, config: LearnableIndexerConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._w1: torch.Tensor  # [32, 5]
        self._b1: torch.Tensor  # [32]
        self._w2: torch.Tensor  # [1, 32]
        self._b2: torch.Tensor  # [1]
        self._cumul_attn: Dict[str, torch.Tensor] = {}  # token_key → EMA 누적 어텐션
        self._init_weights()

    def _init_weights(self) -> None:
        """가중치 초기화. weights_path가 설정되어 있으면 파일에서 로드."""
        ...

    def predict(
        self,
        k: torch.Tensor,           # [n_tokens, d_head] Key 텐서
        v: torch.Tensor,           # [n_tokens, d_head] Value 텐서
        query: torch.Tensor,       # [d_head] 현재 쿼리 벡터 (또는 [n_q, d_head])
        token_positions: torch.Tensor,   # [n_tokens] 절대 위치 인덱스
        current_position: int,           # 현재 디코딩 위치
        segment_key: str = "",           # cumul_attn_score EMA 추적용 키
    ) -> torch.Tensor:
        """토큰별 retention_prob 예측.

        알고리즘:
          1. k_norm = k.float().norm(dim=-1) / sqrt(d_head)  → [n_tokens]
          2. v_norm = v.float().norm(dim=-1) / sqrt(d_head)  → [n_tokens]
          3. cumul_attn = self._get_or_init_cumul_attn(segment_key, n_tokens)
          4. pos_decay = exp(-γ × (current_position - token_positions))  → [n_tokens]
          5. query_vec = query.float().mean(dim=0) if query.dim()==2 else query.float()
             query_sim = F.cosine_similarity(k.float(), query_vec.unsqueeze(0), dim=-1) → [n_tokens]
          6. features = stack([k_norm, v_norm, cumul_attn, pos_decay, query_sim], dim=-1) → [n_tokens, 5]

          if zero_shot_mode:
            retention_prob = Sigmoid(query_sim * cumul_attn)
          else:
            hidden = ReLU(features @ w1.T + b1)   # [n_tokens, 32]
            retention_prob = Sigmoid(hidden @ w2.T + b2).squeeze(-1)  # [n_tokens]

        Returns:
          retention_prob: [n_tokens], dtype=float32, range [0, 1]
        """
        ...

    def update_cumul_attn(
        self,
        segment_key: str,
        attn_scores: torch.Tensor,  # [n_tokens] 현재 스텝 어텐션 가중치
    ) -> None:
        """누적 어텐션 EMA 갱신.
        cumul_attn = α × attn_scores + (1 - α) × cumul_attn
        """
        ...

    def select_tokens_by_budget(
        self,
        retention_prob: torch.Tensor,  # [n_tokens]
        budget_ratio: float,            # 0.0~1.0, 보존할 토큰 비율
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """retention_prob 기반으로 budget_ratio에 따라 to_keep/to_evict 인덱스 분리.

        알고리즘:
          n_keep = max(1, int(n_tokens * budget_ratio))
          kept_idx = retention_prob.topk(n_keep).indices.sort().values
          evict_idx = complement(kept_idx)

        Returns:
          (kept_indices [n_keep], evict_indices [n_evict])
        """
        ...

    def save_weights(self, path: str) -> None:
        """학습 완료 가중치를 YAML 파일로 저장."""
        ...

    def load_weights(self, path: str) -> None:
        """YAML 파일에서 가중치 로드."""
        ...
```

---

### 2. IndexMemLatentMemoryModule (Activity C-1) — `src/cache/indexmem_latent_memory_module.py`

```python
@dataclass
class LatentMemoryConfig:
    kv_dim: int = 128               # head_dim × n_heads (레이어별 전체 KV 차원)
    latent_dim: int = 64            # 잠재 상태 차원 (기본값; kv_dim // 4 권장)
    n_layers: int = 32              # 모델 레이어 수
    encoder_hidden_dim: int = 128   # 경량 트랜스포머 인코더 hidden dim
    encoder_n_heads: int = 4
    encoder_ffn_dim: int = 256
    encoder_n_layers: int = 2       # 트랜스포머 인코더 레이어 수
    alpha_ema: float = 0.3          # 온라인 갱신 EMA 계수 (YAML 외부화)
    beta_readout: float = 0.1       # 잔차 리드아웃 강도 (YAML 외부화)
    readout_threshold: float = 0.0  # readout 활성화 임계값 (0.0 = 항상 적용)
    seed: int = 42


class IndexMemLatentMemoryModule:
    """IndexMem Latent Memory Module — 퇴거 토큰 잠재 상태 압축 보존 + 잔차 리드아웃.

    아키텍처:
      latent_encoder: 2-layer lightweight transformer encoder
        hidden_dim=128, n_heads=4, ffn_dim=256
        입력: evicted_kv [n_evicted, kv_dim] → 출력: latent [n_layers, latent_dim]

    잠재 상태 메모리:
      n_layers × latent_dim × FP16 = 32 × 64 × 2 = 4KB/요청 (극도로 소형)

    온라인 갱신:
      new_latent = alpha × encode(evicted_kv) + (1 - alpha) × current_latent

    잔차 리드아웃:
      attn_output += beta × readout(query_states, latent_state)
    """

    def __init__(self, config: LatentMemoryConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        # latent_encoder: 경량 트랜스포머 (파라미터 ~200K, GPU 메모리 < 1MB)
        self._latent_states: Dict[str, torch.Tensor] = {}  # request_key → [n_layers, latent_dim]
        self._readout_proj: torch.Tensor   # [kv_dim, latent_dim]  projection W_v
        self._readout_key_proj: torch.Tensor  # [kv_dim, latent_dim]  projection W_k
        self._init_encoder()

    def _init_encoder(self) -> None:
        """트랜스포머 인코더 + readout projection 초기화."""
        ...

    def encode_evicted(
        self,
        evicted_kv: torch.Tensor,      # [n_evicted, kv_dim]
        request_key: str,               # 잠재 상태 추적용 키
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """퇴거 토큰들의 KV를 잠재 상태로 압축 및 온라인 갱신.

        알고리즘:
          1. encoded = latent_encoder(evicted_kv)  → [latent_dim]
          2. prev_latent = self._latent_states.get(request_key, zeros)
             → [n_layers, latent_dim]에서 layer_idx 행 참조
          3. new_latent = alpha × encoded + (1 - alpha) × prev_latent[layer_idx]
          4. self._latent_states[request_key][layer_idx] = new_latent

        Returns:
          new_latent: [latent_dim]
        """
        ...

    def residual_readout(
        self,
        query_states: torch.Tensor,    # [n_heads, seq_len, head_dim] 또는 [n_q, d_head]
        request_key: str,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """잠재 상태 기반 잔차 리드아웃 계산.

        알고리즘:
          latent = self._latent_states.get(request_key)  → [n_layers, latent_dim]
          if latent is None: return zeros_like(query_states)
          latent_l = latent[layer_idx]  → [latent_dim]

          readout_v = latent_l @ readout_proj   → [kv_dim]
          readout_k = latent_l @ readout_key_proj → [kv_dim]

          # 쿼리를 [n_q, d_head]로 정규화
          q_flat = query_states.reshape(-1, d_head)  → [n_q, d_head]
          attn_weight = softmax(q_flat @ readout_k.unsqueeze(-1) / sqrt(d_head))  → [n_q, 1]
          readout = attn_weight * readout_v.unsqueeze(0)  → [n_q, kv_dim]
          return beta_readout × readout.reshape_as(query_states)

        오버헤드: latent_dim=64, n_q≤512 기준 < 0.3ms/스텝
        """
        ...

    def get_latent_state(
        self, request_key: str, layer_idx: int = 0
    ) -> Optional[torch.Tensor]:
        """현재 잠재 상태 반환. 없으면 None."""
        ...

    def clear(self, request_key: str) -> None:
        """요청 완료 후 잠재 상태 삭제."""
        ...

    def latent_memory_bytes(self) -> int:
        """전체 잠재 상태 메모리 크기 (바이트)."""
        ...
```

---

### 3. IndexMemEvictionCodec (Activity C-1) — `src/cache/indexmem_eviction_codec.py`

```python
@dataclass
class IndexMemEvictionConfig:
    budget_ratio: float = 0.5          # 보존할 KV 비율 (0.0~1.0)
    base_eviction_policy: str = "snapkv"  # "h2o" | "snapkv" | "dapq" | "learned"
    alpha_ema: float = 0.3             # Latent Memory 온라인 갱신 EMA (YAML 외부화)
    beta_readout: float = 0.1          # 잔차 리드아웃 강도 (YAML 외부화)
    latent_dim: int = 64               # 잠재 상태 차원
    n_layers: int = 32                 # 모델 레이어 수
    kv_dim: int = 128                  # head_dim × n_heads
    zero_shot_mode: bool = False       # Learnable Indexer → DapQ fallback
    max_accuracy_delta: float = 0.01   # accuracy delta 임계값 (±1%)
    fallback_budget_ratio: float = 0.6 # accuracy delta 초과 시 fallback budget_ratio
    fallback_beta: float = 0.05        # accuracy delta 초과 시 fallback beta_readout
    seed: int = 42


class IndexMemEvictionCodec(CompressionCodec):
    """IndexMem 학습 가능한 중요도 예측기 + 잠재 기억 모듈 퇴거 코덱.

    Activity C: "퇴거 = 잠재 압축 + 조건부 복원" 패러다임.

    구성:
      - IndexMemLearnableIndexer: 입력-적응적 토큰 중요도 예측 (225-param MLP)
      - IndexMemLatentMemoryModule: 퇴거 후 잠재 기억 보존 + 잔차 리드아웃

    플러그인 통합:
      - base_eviction_policy 파라미터로 기존 휴리스틱 교체 가능
        ("h2o": H2O 정책, "snapkv": SnapKV, "dapq": DapQ, "learned": Learnable Indexer)
      - VeriCacheSpeculativeCodec.set_draft_codec(self)로 Cross-2 즉시 연동

    DraftCodec 인터페이스도 구현하여 VeriCache 호환성 보장.

    Fallback 메커니즘:
      accuracy delta > max_accuracy_delta(1%) 감지 시:
        budget_ratio → fallback_budget_ratio(0.6)
        beta_readout → fallback_beta(0.05)
    """

    def __init__(self, config: IndexMemEvictionConfig) -> None:
        ...
        self.indexer = IndexMemLearnableIndexer(
            LearnableIndexerConfig(
                zero_shot_mode=config.zero_shot_mode,
                seed=config.seed,
            )
        )
        self.latent_memory = IndexMemLatentMemoryModule(
            LatentMemoryConfig(
                latent_dim=config.latent_dim,
                n_layers=config.n_layers,
                kv_dim=config.kv_dim,
                alpha_ema=config.alpha_ema,
                beta_readout=config.beta_readout,
                seed=config.seed,
            )
        )

    # ---- CompressionCodec interface (src/cache/compression.py 호환) ----

    def encode(
        self,
        kv: torch.Tensor,      # [n_tokens, kv_dim] 또는 [n_tokens, d_head]
        layer_idx: int,
        tensor_id: int = 0,
        query: Optional[torch.Tensor] = None,
        token_positions: Optional[torch.Tensor] = None,
        current_position: int = 0,
        request_key: str = "",
    ) -> torch.Tensor:
        """KV 압축: Learnable Indexer로 중요도 예측 → budget_ratio 기반 to_keep/to_evict 분리
        → to_evict를 Latent Memory로 인코딩 → to_keep만 반환.

        알고리즘:
          1. query가 None이면 kv.mean(dim=0) 사용 (fallback)
          2. retention_prob = indexer.predict(k, v, query, positions, current_position)
          3. kept_idx, evict_idx = indexer.select_tokens_by_budget(retention_prob, budget_ratio)
          4. latent_memory.encode_evicted(kv[evict_idx], request_key, layer_idx)
          5. return kv[kept_idx]  # 압축 KV (budget_ratio × n_tokens 크기)
        """
        ...

    def decode(
        self,
        compressed: torch.Tensor,  # [n_keep, kv_dim]  encode() 반환값
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """decode()는 압축된 KV 그대로 반환 (복원은 residual_readout을 통해).
        VeriCache verify 단계에서 전체 KV를 별도 유지하므로 여기서 완전 복원 불필요.
        """
        return compressed

    def compression_ratio(self, layer_idx: int) -> float:
        """이론적 압축율: 1 / budget_ratio."""
        return 1.0 / self.config.budget_ratio

    # ---- DraftCodec interface (VeriCacheSpeculativeCodec 호환) ----

    def compress(self, kv: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """DraftCodec.compress 인터페이스 구현 (Cross-2 VeriCache 연동).

        Returns (compressed_kv, request_key).
        request_key는 kv tensor ID 기반으로 자동 생성.
        """
        ...

    def decompress(self, compressed_kv: Tuple[torch.Tensor, str]) -> torch.Tensor:
        """DraftCodec.decompress 인터페이스 구현.

        잠재 기억 잔차 리드아웃을 포함한 근사 KV 반환.
        """
        ...

    @property
    def compression_ratio_float(self) -> float:
        """DraftCodec.compression_ratio 호환."""
        return 1.0 / self.config.budget_ratio

    # ---- Readout API ----

    def get_readout(
        self,
        query_states: torch.Tensor,   # [n_q, d_head]
        layer_idx: int = 0,
        request_key: str = "",
    ) -> torch.Tensor:
        """잠재 기억 잔차 리드아웃 반환. attn_output += get_readout(...)로 사용."""
        return self.latent_memory.residual_readout(query_states, request_key, layer_idx)

    # ---- Accuracy fallback ----

    def auto_adjust_on_accuracy_delta(
        self,
        accuracy_delta: float,
    ) -> bool:
        """accuracy delta > max_accuracy_delta 감지 시 budget_ratio + beta_readout 조정.

        Returns True if adjustment was made.
        """
        if abs(accuracy_delta) > self.config.max_accuracy_delta:
            self.config.budget_ratio = self.config.fallback_budget_ratio
            self.latent_memory.config.beta_readout = self.config.fallback_beta
            return True
        return False

    # ---- Statistics ----

    def compression_stats(self) -> dict:
        """JSON-serializable 압축 통계."""
        ...
```

---

### 4. IndexMemSoftHitSegmentCache (Activity B-1) — `src/cache/indexmem_soft_hit_segment_cache.py`

```python
from typing import Literal


@dataclass
class HitResult:
    """캐시 조회 결과. 이진 hit/miss를 연속체로 확장."""
    type: Literal["hard", "soft", "miss"]
    kv_tensor: Optional[torch.Tensor]     # hard hit 시 물리적 KV, 나머지 None
    latent_state: Optional[torch.Tensor]  # soft hit 시 잠재 상태 벡터, 나머지 None
    segment_key: str = ""


@dataclass
class SoftHitSegmentConfig:
    chunk_size: int = 128                  # 세그먼트 청크 크기 (고정)
    max_physical_entries: int = 1000       # 물리적 KV 캐시 최대 항목 수
    latent_pool_max_segments: int = 10000  # 잠재 풀 최대 세그먼트 수 (YAML 외부화)
    beta_soft: float = 0.1                 # 소프트 히트 잔차 기여 강도 (YAML 외부화)
    beta_weight: float = 0.5               # weighted_hit_rate에서 소프트 히트 가중치
    latent_dim: int = 64                   # 잠재 상태 차원
    kv_dim: int = 128
    n_layers: int = 32
    seed: int = 42


class IndexMemSoftHitSegmentCache(CacheStore):
    """IndexMem 소프트 히트 비연속 세그먼트 캐시 (Activity B-1).

    비연속 재사용 모델을 이진(hard hit / miss)에서 연속체로 확장:
      - Hard Hit: 물리적 KV가 캐시에 존재 → kv_tensor 반환
      - Soft Hit: 물리적 KV가 퇴거되었지만 잠재 상태 존재 → latent_state 반환
      - Miss: 물리적 KV도 잠재 상태도 없음 → 재계산 필요

    HitResult 반환 타입으로 소프트 히트를 명시적으로 분리.
    CacheStore 인터페이스 완전 구현.

    가중 히트율:
      weighted_hit_rate = (n_hard + beta_weight × n_soft) / total
    """

    def __init__(self, config: SoftHitSegmentConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._physical_store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._segment_latent_pool: Dict[str, torch.Tensor] = {}   # segment_key → latent_state
        self._latent_encoder: IndexMemLatentMemoryModule = IndexMemLatentMemoryModule(
            LatentMemoryConfig(
                latent_dim=config.latent_dim,
                kv_dim=config.kv_dim,
                n_layers=config.n_layers,
                seed=config.seed,
            )
        )
        self._n_hard_hits: int = 0
        self._n_soft_hits: int = 0
        self._n_misses: int = 0
        self._n_noncontiguous_hard_hits: int = 0
        self._n_noncontiguous_soft_hits: int = 0

    # ---- CacheStore 추상 메서드 구현 ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """물리적 KV 저장. 용량 초과 시 LRU 퇴거 + 잠재 상태 인코딩."""
        ...

    def get(self, key: str) -> Optional[torch.Tensor]:
        """물리적 KV 조회. 소프트 히트는 get_hit_result()를 사용할 것."""
        ...

    def evict(self) -> int:
        """LRU 퇴거. 퇴거 전 잠재 상태 인코딩 후 SegmentLatentPool 저장."""
        ...

    def hit_rate(self) -> float:
        """이진 히트율 (하드 히트만). 하위 호환 유지."""
        total = self._n_hard_hits + self._n_soft_hits + self._n_misses
        return self._n_hard_hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """물리적 KV 메모리 크기."""
        ...

    def reset_stats(self) -> None:
        self._n_hard_hits = 0
        self._n_soft_hits = 0
        self._n_misses = 0
        self._n_noncontiguous_hard_hits = 0
        self._n_noncontiguous_soft_hits = 0

    # ---- 소프트 히트 확장 API ----

    def get_hit_result(
        self,
        segment_key: str,
    ) -> HitResult:
        """소프트 히트 경로 포함 HitResult 반환.

        알고리즘:
          if segment_key in _physical_store:
            return HitResult(type="hard", kv_tensor=_physical_store[key])
          elif segment_key in _segment_latent_pool:
            return HitResult(type="soft", latent_state=_segment_latent_pool[key])
          else:
            return HitResult(type="miss")
        """
        ...

    def soft_hit_residual(
        self,
        query_states: torch.Tensor,   # [n_heads, seq_len, head_dim] 또는 [n_q, d_head]
        latent_state: torch.Tensor,   # [latent_dim]
    ) -> torch.Tensor:
        """소프트 히트 경로: 잠재 상태 기반 잔차 리드아웃.

        알고리즘:
          readout = cross_attn(query_states, latent_state)
          return beta_soft × readout

        beta_soft 자동 조정 (EMA):
          하드 히트 비율 높으면 beta_soft 감소 (소프트 기여 줄임)
          소프트 히트만 있으면 beta_soft 증가 (최대 config.beta_soft × 2)
        """
        ...

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> str:
        """SegmentedHashCache 호환 API. segment_key 반환."""
        ...

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, HitResult]], List[int]]:
        """소프트 히트 경로 포함 세그먼트 조회.

        Returns:
          hits: [(chunk_idx, HitResult), ...] — hard + soft hit 모두 포함
          miss_chunk_indices: [int, ...] — 완전 미스 청크 인덱스
        """
        ...

    def noncontiguous_hit_rate(self) -> float:
        """비연속 하드 히트율."""
        ...

    def soft_hit_rate(self) -> float:
        """소프트 히트율: n_soft_hits / total."""
        total = self._n_hard_hits + self._n_soft_hits + self._n_misses
        return self._n_soft_hits / total if total > 0 else 0.0

    def weighted_hit_rate(self) -> float:
        """가중 히트율: (n_hard + beta_weight × n_soft) / total."""
        total = self._n_hard_hits + self._n_soft_hits + self._n_misses
        if total == 0:
            return 0.0
        return (
            self._n_hard_hits + self.config.beta_weight * self._n_soft_hits
        ) / total
```

---

### 5. WeightedHitRateMetrics (변경) — `src/metrics/hit_rate.py`에 추가

```python
@dataclass
class WeightedHitRateMetrics:
    """소프트 히트 경로 포함 가중 히트율 지표. 기존 HitRateMetrics와 별개."""
    total_requests: int = 0
    total_chunks: int = 0
    hard_hit_chunks: int = 0
    soft_hit_chunks: int = 0
    noncontiguous_hard_hits: int = 0
    noncontiguous_soft_hits: int = 0
    beta_weight: float = 0.5         # 소프트 히트 가중치 (config에서 주입)

    def record(
        self,
        n_hard_hits: int,
        n_soft_hits: int,
        n_misses: int,
        noncontiguous_hard: int = 0,
        noncontiguous_soft: int = 0,
    ) -> None: ...

    def hard_hit_rate(self) -> float: ...
    def soft_hit_rate(self) -> float: ...

    def weighted_hit_rate(self) -> float:
        """(n_hard + beta_weight × n_soft) / total"""
        ...

    def noncontiguous_weighted_fraction(self) -> float:
        """비연속 (하드 + 소프트 가중) / 전체 (하드 + 소프트 가중)"""
        ...

    def reset(self) -> None: ...

    def summary(self) -> dict:
        return {
            "hard_hit_rate": self.hard_hit_rate(),
            "soft_hit_rate": self.soft_hit_rate(),
            "weighted_hit_rate": self.weighted_hit_rate(),
            "noncontiguous_weighted_fraction": self.noncontiguous_weighted_fraction(),
            "hard_hit_chunks": self.hard_hit_chunks,
            "soft_hit_chunks": self.soft_hit_chunks,
            "miss_chunks": self.total_chunks - self.hard_hit_chunks - self.soft_hit_chunks,
            "soft_hit_beta_weight": self.beta_weight,
        }
```

---

### 6. IndexMemBCIntegrationPipeline (Cross-1) — `src/engine/indexmem_bc_pipeline.py`

```python
@dataclass
class BCPipelineConfig:
    segment_latent_pool_mb: float = 128.0   # 세그먼트 레벨 잠재 풀 메모리 상한 (YAML 외부화)
    token_latent_pool_mb: float = 64.0      # 토큰 레벨 잠재 풀 메모리 상한 (YAML 외부화)
    unified_weighted_hit_weights: Tuple[float, float, float] = (1.0, 0.7, 0.4)
    # (hard_weight, segment_soft_weight, token_soft_weight)
    budget_ratio: float = 0.5
    beta_readout: float = 0.1
    seed: int = 42


class UnifiedLatentPool:
    """세그먼트-레벨(B-1)과 토큰-레벨(C-1) 잠재 상태 통합 관리.

    segment_id → segment_latent (세그먼트 전체 KV 집약 표현)
    (segment_id, evicted_token_range) → token_latent (개별 퇴거 토큰 표현)

    메모리 상한: segment_latent_pool_mb + token_latent_pool_mb (YAML 외부화)
    퇴거 정책: LRU (풀 용량 초과 시)
    """

    def __init__(self, config: BCPipelineConfig) -> None: ...

    def put_segment_latent(
        self, segment_id: str, latent: torch.Tensor
    ) -> None: ...

    def put_token_latent(
        self,
        segment_id: str,
        token_range: Tuple[int, int],
        latent: torch.Tensor,
    ) -> None: ...

    def lookup_segment(self, segment_id: str) -> Optional[torch.Tensor]: ...
    def lookup_token(self, segment_id: str, token_range: Tuple[int, int]) -> Optional[torch.Tensor]: ...

    def memory_bytes(self) -> int: ...


class SharedLatentEncoder:
    """B-1과 C-1 파라미터를 공유하는 통합 잠재 인코더.

    B-1의 IndexMemLatentMemoryModule과 C-1의 동일 아키텍처 인코더가
    파라미터를 공유해 학습 효율 2배 향상.
    """
    def __init__(self, config: LatentMemoryConfig) -> None: ...
    def encode(self, kv: torch.Tensor) -> torch.Tensor: ...  # [n_tokens, kv_dim] → [latent_dim]


class IndexMemBCIntegrationPipeline:
    """IndexMem B+C 통합 파이프라인 (Cross-1).

    UnifiedLatentPool로 세그먼트-레벨 소프트 히트(B-1)와
    토큰-레벨 잔차 리드아웃(C-1)을 2레이어 잠재 보존으로 통합.

    처리 흐름:
      Step 1 (B-1 조회): UnifiedLatentPool.lookup()
        → 하드 히트(물리적 KV) / 소프트 히트(세그먼트 잠재) / 미스 3단계
      Step 2 (하드 히트): IndexMemEvictionCodec.encode()로 토큰 중요도 계산
        → 낮은 중요도 토큰 → Latent Memory 인코딩 → UnifiedLatentPool 저장
      Step 3 (소프트 히트): 세그먼트 잠재 기반 잔차 + 토큰 잠재 추가 잔차
      Step 4 (미스): 재계산 → UnifiedLatentPool에 잠재 상태 등록

    통합 가중 히트율:
      unified_weighted_hit_rate = (n_hard + 0.7×n_seg_soft + 0.4×n_token_soft) / total
    """

    def __init__(
        self,
        soft_hit_cache: IndexMemSoftHitSegmentCache,
        eviction_codec: IndexMemEvictionCodec,
        unified_pool: UnifiedLatentPool,
        shared_encoder: SharedLatentEncoder,
        config: BCPipelineConfig,
    ) -> None: ...

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, HitResult]], List[int]]:
        """InferenceRunner 호환 API. B+C 통합 조회."""
        ...

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """InferenceRunner 호환 API."""
        ...

    def unified_weighted_hit_rate(self) -> float:
        """(n_hard + 0.7×n_seg_soft + 0.4×n_token_soft) / total"""
        ...
```

---

### 7. Cross-2: VeriCache + IndexMem 플러그인 연결 (신규 코드 최소)

```python
# 사용 예시 — 별도 파일 없이 실험 스크립트에서 연결
from src.cache.vericache_speculative_codec import VeriCacheSpeculativeCodec, VeriCacheConfig
from src.cache.indexmem_eviction_codec import IndexMemEvictionCodec, IndexMemEvictionConfig

vericache = VeriCacheSpeculativeCodec(VeriCacheConfig(d_head=128, seed=42))
indexmem_codec = IndexMemEvictionCodec(
    IndexMemEvictionConfig(budget_ratio=0.5, zero_shot_mode=True)
)
vericache.set_draft_codec(indexmem_codec)  # 플러그인 연결

# 비교 실험:
# 1. vericache.set_draft_codec(Int8DraftCodec())     → 기존 INT8 드래프트 수락률
# 2. vericache.set_draft_codec(TokenEvictionDraftCodec(keep_ratio=0.5))  → 기존 퇴거 드래프트
# 3. vericache.set_draft_codec(indexmem_codec)       → IndexMem 잠재 기억 드래프트
```

`IndexMemEvictionCodec`은 `DraftCodec` 인터페이스(`compress()`, `decompress()`,
`compression_ratio` property)를 구현하므로 추가 코드 없이 `set_draft_codec()`에 바로 전달 가능.

---

## Activity C — Accuracy Preservation 검증 계획 (MANDATORY)

### perplexity 측정

- **데이터셋**: WikiText-2 (test split, 2,048 토큰 시퀀스 단위)
- **모델 패밀리**: Qwen2-7B / Mistral-7B-v0.3 / Llama-3.1-8B (3개 모델)
  - 테스트 환경 (GPU/모델 없는 경우): d_head=128, n_heads=8, n_layers=32, n_tokens=512 합성 모델
- **허용 오차**: perplexity 변화 ±1% 이내 (필수 기준)
- **목표 오차**: ±0.3~0.8% (IndexMem 원논문 근거)
- **비교 기준**:
  - Baseline: 압축 없는 전체 KV
  - Learnable Indexer 단독 (latent_memory.beta_readout=0, alpha_ema=0)
  - Latent Memory 단독 (zero_shot_mode=True, indexer.retention_prob=budget_ratio for all tokens)
  - Combined (Learnable Indexer + Latent Memory, 표준 설정)

### 태스크 정확도 측정 (벤치마크)

- **벤치마크 1**: RULER-4K — 바늘 깊이 15 / 25 / 50 / 75 / 95% 각각 정확도 측정
- **벤치마크 2**: RULER-16K — 동일 바늘 깊이 5개 정확도 측정
- **벤치마크 3**: LongBench 8개 서브태스크
  (NarrativeQA, Qasper, MultiFieldQA-EN, MultiFieldQA-ZH, HotpotQA, 2WikiMultihopQA, GovReport, QMSum)
- **허용 오차**: ±1% 이내 (필수 기준; evaluation_criteria.md §4)
- **비교 기준**: SnapKV (budget_ratio=0.5 동일 조건) 대비 정확도 delta

### 파라미터 스윕 계획

1. **budget_ratio 스윕**: [0.3, 0.4, 0.5, 0.6, 0.7]
   - 각 값에서 (memory_reduction, perplexity_delta, RULER-4K 정확도) 3-way 측정
   - accuracy-memory 트레이드오프 곡선 작성

2. **beta_readout 스윕**: [0.05, 0.10, 0.15, 0.20]
   - 각 값에서 (perplexity_delta, RULER-4K 정확도) 2-way 측정
   - beta > 0.15 시 accuracy delta > 1% 위험 → YAML 외부화 경고

3. **3-way ablation** (필수):
   - Learnable Indexer 단독 vs. Latent Memory 단독 vs. Combined
   - 각 모듈의 accuracy 기여 분리 검증

### Fallback 메커니즘

```python
# 자동 fallback (IndexMemEvictionCodec.auto_adjust_on_accuracy_delta() 내부)
if abs(measured_accuracy_delta) > 0.01:  # > 1%
    budget_ratio → 0.6   (더 많은 KV 보존)
    beta_readout → 0.05  (잔차 기여 축소)
```

### 검증 테스트 파일

`tests/unit/test_compression_accuracy.py` — 기존 파일에 추가:

```python
def test_indexmem_learnable_indexer_only_accuracy():
    """Learnable Indexer 단독: 동일 budget_ratio에서 SnapKV 대비 accuracy 동등 이상."""
    ...

def test_indexmem_latent_memory_only_accuracy():
    """Latent Memory 단독: beta_readout=0.1에서 perplexity delta < 1%."""
    ...

def test_indexmem_combined_accuracy_within_tolerance():
    """Combined: WikiText-2 perplexity delta ±1% 이내, budget_ratio=0.5."""
    ...

def test_indexmem_budget_ratio_sweep():
    """budget_ratio [0.3..0.7] 전 범위에서 accuracy delta 측정. 0.4 이상에서 ±1% 이내."""
    ...

def test_indexmem_beta_sweep():
    """beta_readout [0.05..0.20] 전 범위에서 perplexity delta 측정."""
    ...

def test_indexmem_fallback_adjusts_on_high_delta():
    """accuracy delta > 1% 시 budget_ratio + beta 자동 조정 동작 검증."""
    ...

def test_indexmem_ruler_needle_depth_accuracy():
    """RULER-4K 바늘 깊이 15/50/95%에서 accuracy delta 측정 (합성 롱컨텍스트)."""
    ...

def test_indexmem_vericache_draft_acceptance_rate():
    """Cross-2: IndexMem 드래프트 수락률 > Int8DraftCodec 수락률."""
    ...
```

---

## Learnable Indexer 학습 절차

1. **학습 데이터**: `data/processed/` 내 대표 샘플 (없으면 합성 데이터 자동 생성)
   - 입력 피처: [k_norm, v_norm, cumul_attn_score, position_decay, query_sim]
   - 라벨: cumul_attn_score 기반 자기지도 (상위 budget_ratio 토큰 = 1, 나머지 = 0)

2. **학습 방법**: LoRA rank=4, 어텐션 레이어만 업데이트, 에폭 3~5
   - Learnable Indexer 자체는 225파라미터이므로 LoRA 없이 직접 AdamW로 학습 가능
   - 모델 가중치 동결 + indexer MLP만 최적화

3. **가중치 저장**: `configs/indexmem_indexer_weights.yaml`
   ```yaml
   # configs/indexmem_indexer_weights.yaml
   # Learnable Indexer 학습 완료 가중치 (2026-05-27 사이클)
   # 형식: YAML-직렬화된 PyTorch 텐서 (base64 또는 float 리스트)
   w1: [...]   # shape: [32, 5]
   b1: [...]   # shape: [32]
   w2: [...]   # shape: [1, 32]
   b2: [...]   # shape: [1]
   trained_epochs: 5
   training_date: "2026-05-27"
   ```

4. **zero-shot fallback**: `zero_shot_mode=True` 시 Learnable Indexer를 DapQ 방식으로 대체
   ```python
   retention_prob = Sigmoid(query_sim * cumul_attn_score)
   ```
   학습 데이터 없이 즉시 동작. 기본 설정에서 활성화.

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-27.yaml
experiment:
  date: "2026-05-27"
  activity: "B+C"
  cache_type: "indexmem_soft_hit_segment"    # Activity B-1
  compression_method: "indexmem_eviction"    # Activity C-1
  scheduler_type: "default"                  # Activity A 미포함
  seed: 42

# Activity C-1 IndexMem Eviction 설정
indexmem_eviction:
  budget_ratio: 0.5                    # 기본 KV 보존 비율
  base_eviction_policy: "snapkv"       # "h2o" | "snapkv" | "dapq" | "learned"
  alpha_ema: 0.3                       # Latent Memory 온라인 갱신 EMA
  beta_readout: 0.1                    # 잔차 리드아웃 강도
  latent_dim: 64                       # 잠재 상태 차원
  n_layers: 32
  kv_dim: 128                          # head_dim × n_heads
  zero_shot_mode: true                 # DapQ fallback 기본값 (학습 없이 동작)
  max_accuracy_delta: 0.01             # accuracy fallback 임계값
  fallback_budget_ratio: 0.6
  fallback_beta: 0.05

  # Accuracy 검증 스윕 설정
  budget_ratio_sweep: [0.3, 0.4, 0.5, 0.6, 0.7]
  beta_sweep: [0.05, 0.10, 0.15, 0.20]

# Activity B-1 Soft Hit Segment Cache 설정
indexmem_soft_hit:
  chunk_size: 128
  max_physical_entries: 1000
  latent_pool_max_segments: 10000      # 잠재 풀 최대 세그먼트 수
  beta_soft: 0.1                       # 소프트 히트 잔차 기여 강도
  beta_weight: 0.5                     # weighted_hit_rate 소프트 히트 가중치
  latent_dim: 64
  kv_dim: 128
  n_layers: 32

# Cross-1 B+C 통합 파이프라인 설정
bc_pipeline:
  segment_latent_pool_mb: 128.0
  token_latent_pool_mb: 64.0
  unified_hit_weights: [1.0, 0.7, 0.4]  # [hard, seg_soft, token_soft]

# Learnable Indexer 학습 설정
learnable_indexer:
  weights_path: "configs/indexmem_indexer_weights.yaml"
  zero_shot_mode: true       # 기본: DapQ fallback (학습 필요 없음)
  lora_rank: 4               # 학습 시 LoRA rank
  epochs: 5
  lr: 1.0e-3

# 측정 지표 저장 경로
metrics:
  output_dir: "results/2026-05-27"
  metrics_file: "results/2026-05-27/metrics.json"
```

---

## results/2026-05-27/metrics.json 저장 지표

```json
{
  "experiment_date": "2026-05-27",
  "activity": "B+C",

  "compression_accuracy": {
    "wikitext2_perplexity_baseline": null,
    "wikitext2_perplexity_compressed": null,
    "wikitext2_perplexity_delta_pct": null,
    "ruler_4k_depth15_accuracy_delta": null,
    "ruler_4k_depth25_accuracy_delta": null,
    "ruler_4k_depth50_accuracy_delta": null,
    "ruler_4k_depth75_accuracy_delta": null,
    "ruler_4k_depth95_accuracy_delta": null,
    "ruler_16k_depth15_accuracy_delta": null,
    "ruler_16k_depth95_accuracy_delta": null,
    "longbench_8task_mean_accuracy_delta": null,
    "accuracy_within_1pct_tolerance": null
  },

  "kv_memory": {
    "baseline_kv_bytes": null,
    "compressed_kv_bytes": null,
    "latent_state_bytes": null,
    "memory_reduction_pct": null,
    "effective_context_length_ratio": null
  },

  "hit_rate": {
    "hard_hit_rate": null,
    "soft_hit_rate": null,
    "weighted_hit_rate": null,
    "noncontiguous_fraction": null,
    "noncontiguous_weighted_fraction": null,
    "soft_hit_beta_weight": 0.5
  },

  "latent_memory": {
    "latent_readout_score_mean": null,
    "latent_readout_score_p50": null,
    "latent_readout_score_p95": null,
    "latent_memory_bytes_per_request": null,
    "online_update_alpha": 0.3,
    "beta_readout_effective": null
  },

  "throughput": {
    "tokens_per_sec_baseline": null,
    "tokens_per_sec_compressed": null,
    "throughput_improvement_pct": null,
    "ttft_p50_baseline_ms": null,
    "ttft_p50_compressed_ms": null,
    "ttft_p50_delta_pct": null
  },

  "ablation": {
    "indexer_only_accuracy_delta": null,
    "latent_only_accuracy_delta": null,
    "combined_accuracy_delta": null
  },

  "budget_ratio_sweep": {
    "0.3": {"memory_reduction_pct": null, "perplexity_delta_pct": null},
    "0.4": {"memory_reduction_pct": null, "perplexity_delta_pct": null},
    "0.5": {"memory_reduction_pct": null, "perplexity_delta_pct": null},
    "0.6": {"memory_reduction_pct": null, "perplexity_delta_pct": null},
    "0.7": {"memory_reduction_pct": null, "perplexity_delta_pct": null}
  },

  "beta_sweep": {
    "0.05": {"perplexity_delta_pct": null},
    "0.10": {"perplexity_delta_pct": null},
    "0.15": {"perplexity_delta_pct": null},
    "0.20": {"perplexity_delta_pct": null}
  },

  "cross2_vericache": {
    "int8_draft_acceptance_rate": null,
    "token_eviction_draft_acceptance_rate": null,
    "indexmem_draft_acceptance_rate": null,
    "indexmem_vs_int8_acceptance_delta": null
  }
}
```

---

## 테스트 요구사항

### 필수 단위 테스트

- [ ] `tests/unit/test_indexmem_learnable_indexer.py`
  - `test_predict_output_shape()`: predict() 반환이 [n_tokens] float32 [0,1] 범위
  - `test_predict_zero_shot_mode()`: zero_shot_mode=True에서 DapQ fallback 동작
  - `test_select_tokens_by_budget()`: budget_ratio=0.5에서 n_keep = n_tokens//2
  - `test_update_cumul_attn_ema()`: EMA 갱신이 alpha_ema 계수로 정확하게 동작
  - `test_select_tokens_deterministic()`: 동일 입력 + seed → 동일 kept_idx
  - `test_save_load_weights()`: save_weights → load_weights 왕복 일치

- [ ] `tests/unit/test_indexmem_latent_memory_module.py`
  - `test_encode_evicted_output_shape()`: encode_evicted 반환이 [latent_dim]
  - `test_online_update_ema_alpha()`: alpha=0.3 EMA 갱신 정확성
  - `test_residual_readout_output_shape()`: readout 반환이 query_states와 동일 shape
  - `test_readout_zero_before_encode()`: encode 전 readout → zeros
  - `test_latent_state_persists_across_calls()`: 동일 request_key로 누적 갱신 동작
  - `test_clear_removes_latent_state()`: clear() 후 readout → zeros
  - `test_latent_memory_bytes()`: 잠재 상태 메모리 크기 = n_layers × latent_dim × 2

- [ ] `tests/unit/test_indexmem_eviction_codec.py`
  - `test_encode_output_smaller_than_input()`: encode 후 n_tokens × budget_ratio 크기
  - `test_encode_preserves_important_tokens()`: 높은 retention_prob 토큰이 to_keep에 포함
  - `test_get_readout_after_encode()`: encode 후 get_readout() → non-zero tensor
  - `test_auto_adjust_fallback_on_high_delta()`: accuracy_delta=0.02 → budget_ratio 증가
  - `test_draft_codec_interface_compress()`: compress() 반환 타입이 (Tensor, str)
  - `test_draft_codec_interface_decompress()`: decompress(compress(kv)) shape 유효
  - `test_compression_ratio_property()`: compression_ratio_float == 1/budget_ratio
  - `test_base_eviction_policy_h2o()`: base_eviction_policy="h2o" 설정 시 정상 동작
  - `test_compression_stats_json_serializable()`: compression_stats() dict JSON 직렬화

- [ ] `tests/unit/test_indexmem_soft_hit_segment_cache.py`
  - `test_cache_store_interface_all_methods()`: CacheStore 추상 메서드 전부 동작
  - `test_hard_hit_returns_kv_tensor()`: put → get_hit_result → type="hard"
  - `test_soft_hit_after_eviction()`: put → evict → get_hit_result → type="soft"
  - `test_miss_no_physical_no_latent()`: 미저장 key → type="miss"
  - `test_soft_hit_residual_output_shape()`: soft_hit_residual 반환 shape = query shape
  - `test_weighted_hit_rate_formula()`: (n_hard + beta_weight×n_soft)/total 수치 검증
  - `test_soft_hit_rate()`: n_soft / total 수치 검증
  - `test_latent_pool_lru_eviction()`: latent_pool_max_segments 초과 시 LRU 퇴거
  - `test_put_segment_returns_key()`: put_segment() segment_key 반환
  - `test_get_segments_returns_hit_results()`: get_segments() 반환이 (hits, misses) 형식

### 필수 통합 테스트

- [ ] `tests/integration/test_indexmem_bc_pipeline_e2e.py`
  - `test_bc_pipeline_hard_hit_flow()`: B+C 통합 파이프라인 하드 히트 전체 흐름
  - `test_bc_pipeline_soft_hit_flow()`: 세그먼트 퇴거 후 소프트 히트 잔차 리드아웃 제공
  - `test_bc_pipeline_unified_weighted_hit_rate()`: 통합 가중 히트율이 이진 히트율 이상
  - `test_bc_pipeline_memory_reduction_above_40pct()`: budget_ratio=0.5에서 −40% 이상
  - `test_bc_pipeline_inference_runner_compat()`: InferenceRunner가 IndexMemBCIntegrationPipeline을
    `cache`로 받아 get_segments/put_segment API를 통해 정상 동작
  - `test_cross2_vericache_indexmem_plugin()`: VeriCacheSpeculativeCodec.set_draft_codec(IndexMem)
    연결 후 draft_and_verify() 정상 동작

### Activity C 검증 테스트 (기존 파일 추가)

- [ ] `tests/unit/test_compression_accuracy.py` — 추가 케이스 (기존 케이스 보존):
  - `test_indexmem_learnable_indexer_only_accuracy()`
  - `test_indexmem_latent_memory_only_accuracy()`
  - `test_indexmem_combined_accuracy_within_tolerance()`
  - `test_indexmem_budget_ratio_sweep()`
  - `test_indexmem_beta_sweep()`
  - `test_indexmem_fallback_adjusts_on_high_delta()`
  - `test_indexmem_ruler_needle_depth_accuracy()`
  - `test_indexmem_vericache_draft_acceptance_rate()`

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 100% 통과**: 위 명시된 모든 단위 테스트 케이스 통과
2. **통합 테스트 100% 통과**: `test_indexmem_bc_pipeline_e2e.py` 전체 통과
3. **기존 테스트 회귀 없음**: 이전 사이클 구현의 모든 단위·통합 테스트 계속 통과
4. **evaluation_criteria.md §4 (Activity C) 필수 항목 모두 Pass**:
   - WikiText-2 perplexity delta ±1% 이내
   - RULER-4K/16K 정확도 delta ±1% 이내
   - LongBench 8개 서브태스크 accuracy delta ±1% 이내
   - KV Memory Reduction −30% 이상 (목표 −40%)
5. **evaluation_criteria.md §3 (Activity B) 기준 충족**:
   - 가중 비연속 히트율(weighted_hit_rate) 베이스라인 대비 +15%p 이상
   - Soft Hit Rate 측정 및 보고
6. **evaluation_criteria.md §5 (크로스 조합) 기준**:
   - Accuracy 보존 복합 적용 후에도 ±1% 이내 (C 포함 필수)
7. **CacheStore 인터페이스 준수**: IndexMemSoftHitSegmentCache 추상 메서드 전부 구현
8. **CompressionCodec 호환**: IndexMemEvictionCodec이 encode()/decode()/compression_ratio() 구현
9. **DraftCodec 호환**: IndexMemEvictionCodec이 compress()/decompress()/compression_ratio_float 구현
10. **설정 파일 존재**: `configs/experiments/2026-05-27.yaml` + `configs/indexmem_indexer_weights.yaml`
11. **metrics.json 생성**: `results/2026-05-27/metrics.json`에 모든 지표 기록
12. **시드 고정 재현성**: seed=42로 동일 결과 재현 가능

---

## 구현 우선순위 순서

1. C-1: `IndexMemLearnableIndexer` (zero_shot_mode=True 기본값으로 학습 없이 즉시 동작)
2. C-1: `IndexMemLatentMemoryModule` (encode_evicted + residual_readout)
3. C-1: `IndexMemEvictionCodec` (두 모듈 통합 + DraftCodec 인터페이스)
4. Cross-2: VeriCacheSpeculativeCodec.set_draft_codec(IndexMemEvictionCodec) 연결 (1줄)
5. B-1: `HitResult` dataclass + `WeightedHitRateMetrics` (metrics/hit_rate.py 추가)
6. B-1: `IndexMemSoftHitSegmentCache` (소프트 히트 경로 + CacheStore 구현)
7. Cross-1: `UnifiedLatentPool` + `SharedLatentEncoder` + `IndexMemBCIntegrationPipeline`
8. 설정 파일: `configs/experiments/2026-05-27.yaml`, `configs/indexmem_indexer_weights.yaml`
9. 단위 테스트 전부 (C-1 → B-1 → Cross-1 순서)
10. 통합 테스트 + accuracy 검증 테스트

---

## 보존 파일 (수정 금지)

이전 사이클 구현 파일은 수정하지 않는다:

- `src/cache/irminsul_mla_segment_cache.py` (05-26 B-1)
- `src/cache/arch_aware_noncontiguous_router.py` (05-26 B-1)
- `src/cache/cdc_content_hash_interface.py` (05-26 B-2)
- `src/cache/mla_two_axis_compression_codec.py` (05-26 C-1)
- `src/scheduler/objectcache_s3_tier_router.py` (05-26 A-1)
- `src/engine/irminsul_objectcache_pipeline.py` (05-26 Cross-1)
- `src/cache/vericache_speculative_codec.py` (05-25 C-1)
- `src/cache/kv_packet.py`, `src/cache/kv_packet_adapter.py` (05-25 B-1)
- `src/cache/segmented.py` (베이스라인)
- 기타 모든 이전 사이클 파일 — 기존 단위·통합 테스트 회귀 없이 통과해야 한다.

**주의**: `src/metrics/hit_rate.py`에는 `WeightedHitRateMetrics` 클래스만 추가한다.
기존 `HitRateMetrics` 클래스와 메서드는 수정하지 않는다.
