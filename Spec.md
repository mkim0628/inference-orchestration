<!-- 변경 이유 (이전 Spec.md: 2026-05-05 대비):
이전 사이클(2026-05-05)은 B+C (DiffAwareSegmentStore + NQKVCodec + FireQCodec + CompressedDiffStore) 조합이었다.
이번 사이클은 B+C (QueryCentricRecomputeCache + InfoFlowChunkReorderCache + TriAttentionCodec + QueryCentricTriAttentionCache)
조합으로 완전히 전환한다. 알고리즘 패러다임이 다음과 같이 바뀐다:

주요 변경:
1. [Activity B 교체] DiffAwareSegmentStore(마스터+블록-희소 차분 저장) →
   QueryCentricRecomputeCache(사용자 쿼리 관련성 기반 이중 단계 재계산 예산 배분, ProphetKV 기반).
   "에이전트 간 공통 KV 마스터 저장"에서 "사용자 쿼리 외부 관련성으로 재계산 예산 배분"으로
   알고리즘 패러다임이 전환된다. 이전 사이클의 DHD/SemanticSegmentCache가 세그먼트 내부 편차로
   재계산 여부를 결정한 것과 달리, QueryCentricRecomputeCache는 쿼리 관련성을 기준으로
   예산 배분 주체 자체가 바뀐다.

2. [Activity B 보조 추가] InfoFlowChunkReorderCache(정보 흐름 기반 비연속 세그먼트 최적 배치 순서 결정,
   InfoFlowKV 기반)가 신규 추가된다. 이전 사이클의 모든 B 기법이 세그먼트 선택·저장·퇴거에
   집중한 반면, 이 기법은 선택된 세그먼트의 "배치 순서"를 최적화한다.

3. [Activity C 완전 교체] NQKVCodec(정규분포 분위수 양자화) + FireQCodec(RoPE-인식 2단계 평활화) →
   TriAttentionCodec(pre-RoPE 삼각함수 시리즈 기반 위치-안정적 KV 중요도 추정, TriAttention 기반).
   post-RoPE 양자화에서 pre-RoPE 기하학적 집중 패턴 기반 프루닝으로 이론적 기반이 전환된다.
   실증 근거: AIME25 32K CoT에서 10.7× KV 감소 + 전체 어텐션 동등 정확도.

4. [Cross-1 교체] CompressedDiffStore(DiffAwareSegmentStore + NQKVCodec) →
   QueryCentricTriAttentionCache(QueryCentricRecomputeCache + TriAttentionCodec).
   쿼리 관련 세그먼트는 선택적 재계산(원본 품질 보존),
   비관련 세그먼트는 TriAttentionCodec으로 고압축하는 이중 경로가 신규 설계된다.

5. [보존 파일] 이전 사이클 파일
   (diff_aware_store.py, nq_kv_codec.py, fireq_codec.py, compressed_diff_store.py,
   dag_topology_scheduler.py, workload_ttl_cache.py, redundancy_eviction.py,
   turbo_quant.py, dhd_segment_cache.py, speculative_fetcher.py, sign_vq_segment.py,
   leverage_compressor.py, compression.py, segmented.py, contiguous.py,
   tri_state_compressor.py, compressed_segment.py, segment_adapter.py)은
   이번 사이클에서 수정하지 않는다. 기존 모든 단위·통합 테스트가 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-06

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-06.md`
**최우선 구현 타겟**: Cross-1 (B+C) — QueryCentricRecomputeCache(B-1) + TriAttentionCodec(C-1)
+ DualFilterSegmentSelector 통합, 보조로 InfoFlowChunkReorderCache(B-2)

**해결하려는 문제**:
- 기존 비연속 세그먼트 재사용 기법(해시, DHD, TTL, 차분 저장)은 "사용자 쿼리와의 관련성"을
  재계산 예산 배분 기준으로 활용하지 않아, 쿼리 비관련 고현저성 토큰이 제한된 예산을 낭비하는
  crowding-out 효과가 발생한다. QueryCentricRecomputeCache(ProphetKV 기반)가 이를 해소한다.
- post-RoPE 공간의 쿼리 벡터는 위치마다 다르게 회전하여 KV 중요도 추정이 불안정하다.
  pre-RoPE 공간에서는 Q/K 벡터가 비제로 중심 주변에 집중되므로, 삼각함수 시리즈로
  위치-안정적 중요도 추정이 가능하다(TriAttention 기반). 실증: 10.7× KV 감소 + 동등 정확도.
- 비연속 세그먼트를 "어떤 순서로 배치할 것인가"가 정확도에 영향을 주지만, 이전 사이클의
  모든 B 기법은 배치 순서를 최적화하지 않았다. InfoFlowChunkReorderCache가 이를 다룬다.

---

## 이번 사이클 Activity

- [ ] Activity A: KV Cache-aware Scheduling (이번 사이클 미포함 — B+C 집중)
- [x] Activity B: Non-Contiguous KV Cache Reuse — QueryCentricRecomputeCache + InfoFlowChunkReorderCache
- [x] Activity C: KV Cache Compression — TriAttentionCodec (pre-RoPE 삼각함수 중요도)

---

## 목표

- [ ] 목표 1 (§1 Throughput): tokens/sec 베이스라인 대비 +20% 이상 — TriAttentionCodec 10.7× KV 감소로 배치 슬롯 증가 (evaluation_criteria.md §1)
- [ ] 목표 2 (§4 KV Memory Reduction): 베이스라인 대비 −30% 이상 — TriAttentionCodec 10× 압축 기여 (evaluation_criteria.md §4)
- [ ] 목표 3 (§3 Non-Contiguous Hit Rate): 전체 히트 중 비연속 히트 비율 ≥ 30% — QueryCentricRecomputeCache 이중 단계 달성 (evaluation_criteria.md §3)
- [ ] 목표 4 (§4 Accuracy 필수): perplexity 변화 ±1% 이내 — TriAttentionCodec pre-RoPE 안정성으로 달성 (evaluation_criteria.md §4)
- [ ] 목표 5 (§4 Accuracy 필수): downstream 태스크 정확도 변화 ±1% 이내 (evaluation_criteria.md §4)
- [ ] 목표 6 (§5 Cross): 복합 처리량 향상 단일 Activity 대비 추가 +5% 이상 (evaluation_criteria.md §5)
- [ ] 목표 7 (§5 Cross): 복합 메모리 감소 단일 Activity 대비 추가 −10% 이상 (evaluation_criteria.md §5)
- [ ] 목표 8 (§4 Compression Overhead): Encode/Decode 추가 지연 TTFT +10% 이내 (evaluation_criteria.md §4)

---

## 아키텍처 개요

```
요청 도착 (query_tokens + context_segments)
    │
    ├─ Stage 1: QueryCentricRecomputeCache.selective_recompute()
    │       │
    │       ├─ [전역 현저성 필터] attention norm 상위 50% 세그먼트 선택
    │       └─ [쿼리 관련성 재순위] 코사인 유사도로 상위 20% 재계산 대상 결정
    │
    ├─ Stage 2: InfoFlowChunkReorderCache.reorder_chunks()
    │       │
    │       └─ 어텐션-정규 신호 기반 최적 배치 순서 결정 (O(N log N))
    │
    ├─ Stage 3: TriAttentionCodec.compress()
    │       │
    │       ├─ pre-RoPE 중심 벡터(mu_k) 대비 거리 계산
    │       ├─ 삼각함수 시리즈 중요도 추정
    │       └─ 128 토큰 윈도우 프루닝 (compression_ratio=0.1)
    │
    └─ QueryCentricTriAttentionCache (Cross-1 통합)
            │
            ├─ 쿼리 관련 세그먼트 → 원본 KV 보존 (재계산 품질)
            └─ 비관련 세그먼트 → TriAttentionCodec 고압축 저장
```

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/query_centric_recompute.py` | B | ProphetKV 기반 쿼리 관련성 이중 단계 재계산 예산 배분 |
| `src/cache/info_flow_reorder.py` | B | 정보 흐름 기반 비연속 세그먼트 배치 순서 최적화 |
| `src/cache/tri_attention_codec.py` | C | pre-RoPE 삼각함수 시리즈 KV 중요도 추정 + 윈도우 프루닝 |
| `src/cache/qc_tri_store.py` | B+C | QueryCentricRecomputeCache + TriAttentionCodec 통합 |
| `src/cache/dual_filter_selector.py` | B+C | 쿼리 관련성 + pre-RoPE 중요도 이중 필터링 |
| `tests/unit/test_query_centric_recompute.py` | B | 쿼리 관련성 스코어 + 재계산 예산 제한 단위 테스트 |
| `tests/unit/test_info_flow_reorder.py` | B | 재순서화 정확도 + RoPE 재계산 단위 테스트 |
| `tests/unit/test_tri_attention_accuracy.py` | C | pre-RoPE 압축 정확도 + perplexity ±1% 단위 테스트 |
| `tests/unit/test_qc_tri_store.py` | B+C | Cross-1 통합 단위 테스트 |
| `tests/integration/test_cross_bc_dual_filter.py` | B+C | 이중 필터 통합 E2E 테스트 |
| `experiments/run_triattention_accuracy.py` | C | TriAttentionCodec perplexity + LongBench 정확도 측정 스크립트 |
| `configs/experiments/2026-05-06.yaml` | 공통 | 실험 설정 파일 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/__init__.py` | 신규 클래스 export 추가 |

### 수정 금지 파일 (이전 사이클 보존)

`dag_topology_scheduler.py`, `diff_aware_store.py`, `nq_kv_codec.py`, `fireq_codec.py`,
`compressed_diff_store.py`, `workload_ttl_cache.py`, `redundancy_eviction.py`,
`turbo_quant.py`, `dhd_segment_cache.py`, `speculative_fetcher.py`, `sign_vq_segment.py`,
`leverage_compressor.py`, `compression.py`, `segmented.py`, `contiguous.py`,
`tri_state_compressor.py`, `compressed_segment.py`, `segment_adapter.py`

---

## 알고리즘 상세

### 1. QueryCentricRecomputeCache (Activity B)

`CacheStore` 인터페이스를 상속하며, 비연속 세그먼트 캐시 위에 쿼리 관련성 기반 이중 단계 재계산
파이프라인 레이어를 추가한다.

```python
# src/cache/query_centric_recompute.py 의사코드

class QueryCentricRecomputeCache(CacheStore):
    """
    ProphetKV 원리 기반 쿼리 관련성 이중 단계 재계산 예산 배분.

    이전 사이클 SemanticSegmentCache(DHD)와의 구조적 차별점:
    - DHD: 세그먼트 내부 편차(레이어별 어텐션 스코어 변화)로 재계산 여부 결정.
    - QCRC: 외부 쿼리와의 관련성(코사인 유사도)으로 재계산 예산 배분 주체를 전환.
      예산 배분이 세그먼트 내부에서 쿼리 외부로 이동하는 것이 핵심 차별점.
    """

    def __init__(
        self,
        capacity_bytes: int,
        recompute_budget_ratio: float = 0.20,  # 전체 토큰 중 최대 재계산 비율
        stage1_top_k_ratio: float = 0.50,      # Stage1: attention norm 상위 50%
    ) -> None: ...

    def put(self, key: str, value: torch.Tensor) -> None:
        # key: 세그먼트 해시 (문자열)
        # value: KV 텐서 [layers, heads, seq_len, head_dim]
        # 세그먼트 평균 K 벡터를 임베딩으로 저장 (추가 모델 불필요)
        segment_embedding = value[:, :, :, :].mean(dim=(0, 1, 2))  # [head_dim]
        self._store[key] = {"kv": value, "embedding": segment_embedding, "attn_norm": ...}

    def get(self, key: str) -> Optional[torch.Tensor]:
        # 정확 해시 히트 → 직접 반환
        entry = self._store.get(key)
        if entry is None:
            self._miss_count += 1
            return None
        self._hit_count += 1
        return entry["kv"]

    def get(self, query_tokens: List[int], segment_tokens: List[int]) -> Optional[torch.Tensor]:
        # 비연속 세그먼트 조회: query_tokens를 사용해 관련성 스코어 계산
        key = self._hash(segment_tokens)
        return self._store.get(key, {}).get("kv")

    def selective_recompute(
        self,
        query: torch.Tensor,          # [head_dim] — 쿼리 토큰 평균 K 벡터
        cached_segments: List[str],   # 후보 세그먼트 해시 목록
        budget: float = 0.20,         # 재계산 예산 (전체 토큰 비율)
    ) -> List[str]:
        """
        이중 단계 재계산 예산 배분.
        반환: 실제 재계산 대상 세그먼트 해시 목록.
        """
        # --- Stage 1: 전역 현저성 필터 (attention norm 상위 50%) ---
        attn_norms = {
            seg_key: self._store[seg_key]["attn_norm"]
            for seg_key in cached_segments
            if seg_key in self._store
        }
        # attention norm 기준 상위 50% 선택
        sorted_by_norm = sorted(attn_norms, key=lambda k: attn_norms[k], reverse=True)
        stage1_candidates = sorted_by_norm[: max(1, int(len(sorted_by_norm) * self.stage1_top_k_ratio))]

        # --- Stage 2: 쿼리 관련성 재순위 (코사인 유사도) ---
        relevance_scores = {}
        for seg_key in stage1_candidates:
            seg_emb = self._store[seg_key]["embedding"]  # [head_dim]
            # 코사인 유사도: query와 세그먼트 평균 K 벡터
            score = torch.nn.functional.cosine_similarity(
                query.unsqueeze(0), seg_emb.unsqueeze(0)
            ).item()
            relevance_scores[seg_key] = score

        sorted_by_relevance = sorted(
            relevance_scores, key=lambda k: relevance_scores[k], reverse=True
        )

        # 예산 제한: 전체 토큰의 budget 비율 내로 제한
        total_tokens = sum(
            self._store[k]["kv"].shape[2] for k in sorted_by_relevance
            if k in self._store
        )
        recompute_token_budget = int(total_tokens * budget)
        selected, accumulated = [], 0
        for seg_key in sorted_by_relevance:
            seg_len = self._store[seg_key]["kv"].shape[2]
            if accumulated + seg_len > recompute_token_budget:
                break
            selected.append(seg_key)
            accumulated += seg_len

        return selected  # 재계산 대상 세그먼트

    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...
```

**쿼리 임베딩 생성 방법**: 쿼리 토큰의 평균 K 벡터 (`kv_query[:, :, :, :].mean(dim=(0,1,2))`)를 사용한다. 추가 인코더 모델이 불필요하며, 기존 인코딩 과정에서 생성된 K 벡터를 재활용한다.

---

### 2. InfoFlowChunkReorderCache (Activity B)

```python
# src/cache/info_flow_reorder.py 의사코드

class InfoFlowChunkReorderCache(CacheStore):
    """
    정보 흐름 기반 비연속 세그먼트 배치 순서 최적화.
    어텐션-정규(attention_score × norm(K_i)) 신호로 O(N log N) 정렬.
    재순서화 후 RoPE 재계산 비용을 측정하고 보고해야 한다.
    """

    def __init__(self, capacity_bytes: int) -> None: ...

    def put(self, key: str, value: torch.Tensor) -> None:
        # value: [layers, heads, seq_len, head_dim]
        # 정보 흐름 신호 사전 계산 및 저장
        infoflow_score = self._compute_infoflow_score(value)
        self._store[key] = {"kv": value, "infoflow_score": infoflow_score}

    def get(self, key: str) -> Optional[torch.Tensor]:
        entry = self._store.get(key)
        if entry is None:
            self._miss_count += 1
            return None
        self._hit_count += 1
        return entry["kv"]

    def _compute_infoflow_score(self, kv: torch.Tensor) -> float:
        """
        정보 흐름 신호: attention_score × norm(K_i)
        kv: [layers, heads, seq_len, head_dim]
        반환: 세그먼트 전체 정보 흐름 스칼라 스코어
        """
        # K 벡터 norm (레이어·헤드 평균)
        k_norm = kv.norm(dim=-1).mean(dim=(0, 1))  # [seq_len]
        # 단순 어텐션 스코어 근사: softmax(K @ K^T / sqrt(d)) 대각합
        d = kv.shape[-1]
        k_flat = kv.mean(dim=(0, 1))  # [seq_len, head_dim]
        attn_approx = torch.softmax(
            (k_flat @ k_flat.T) / (d ** 0.5), dim=-1
        ).diagonal()  # [seq_len]
        # 정보 흐름 신호: 어텐션 스코어 × K norm
        infoflow = (attn_approx * k_norm).sum().item()
        return infoflow

    def reorder_chunks(
        self,
        chunks: List[str],           # 세그먼트 해시 목록 (재순서화 대상)
        attention_scores: Optional[Dict[str, float]] = None,  # 외부 어텐션 스코어 (선택)
    ) -> List[str]:
        """
        정보 흐름 스코어 기준 내림차순 정렬 → 정보 전파에 유리한 세그먼트를 앞에 배치.
        O(N log N), N ≤ 50인 일반 RAG 시나리오에서 오버헤드 무시 가능.
        """
        scored = []
        for chunk_key in chunks:
            if chunk_key not in self._store:
                continue
            score = self._store[chunk_key]["infoflow_score"]
            if attention_scores and chunk_key in attention_scores:
                # 외부 어텐션 스코어 제공 시 가중합 (0.5:0.5)
                score = 0.5 * score + 0.5 * attention_scores[chunk_key]
            scored.append((chunk_key, score))

        # 정보 흐름 높은 순 정렬 (앞쪽 배치 = 어텐션에 더 큰 영향)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored]

    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...
```

**RoPE 재계산**: 재순서화 후 새로운 위치 인덱스를 할당할 때 RoPE 재계산이 필요하다. 구현체는 재순서화 시간과 RoPE 재계산 시간을 분리 측정하여 `results/<exp>/metrics.json`에 기록해야 한다.

---

### 3. TriAttentionCodec (Activity C)

```python
# src/cache/tri_attention_codec.py 의사코드

class TriAttentionCodec:
    """
    pre-RoPE 삼각함수 시리즈 기반 위치-안정적 KV 중요도 추정 + 128 토큰 윈도우 프루닝.

    핵심 원리:
    - post-RoPE 공간에서 쿼리는 위치마다 다르게 회전 → 대표성 제한 → 불안정한 중요도 추정.
    - pre-RoPE 공간에서 Q/K 벡터는 비제로 중심(mu_q, mu_k) 주변에 집중됨.
    - 이 집중이 선호 거리를 삼각함수 시리즈로 표현 가능 → 위치-안정적 중요도 추정.

    실증 근거: TriAttention(arXiv 2604.04921) AIME25 32K CoT에서 10.7× KV 감소 + 동등 정확도.
    CacheStore를 상속하지 않음 (순수 코덱 클래스). CacheStore 구현체 내부에서 호출됨.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        compression_ratio: float = 0.10,  # 보존 비율 (0.1 = 10× 압축)
        series_terms: int = 8,            # 삼각함수 시리즈 항 수 (m=1..8)
        prune_window: int = 128,          # 프루닝 트리거 윈도우 크기 (토큰 수)
    ) -> None:
        self.mu_k: Optional[torch.Tensor] = None   # [layers, heads, head_dim] — 캘리브레이션 결과
        self.a_m: Optional[torch.Tensor] = None    # [layers, heads, series_terms] — 계수
        ...

    def calibrate(
        self,
        calibration_kvs: List[torch.Tensor],  # 캘리브레이션 배치 (≥10 요청의 pre-RoPE KV)
        save_path: Optional[str] = None,       # 레이어별 캘리브레이션 파일 저장 경로
    ) -> None:
        """
        캘리브레이션 배치에서 mu_k(비제로 중심)와 a_m(시리즈 계수)를 추정.
        1회만 실행 후 파일로 저장. 저장 경로 미지정 시 메모리에만 보관.
        """
        # mu_k 추정: 레이어·헤드별 K 벡터 평균
        all_keys = torch.cat([kv for kv in calibration_kvs], dim=2)  # [layers, heads, total_seq, dim]
        self.mu_k = all_keys.mean(dim=2)  # [layers, heads, head_dim]

        # a_m 추정: 각 k_i와 mu_k 간 거리를 삼각함수로 분해 후 최소자승 피팅
        distances = (all_keys - self.mu_k.unsqueeze(2)).norm(dim=-1)  # [layers, heads, total_seq]
        # Fourier 피팅: a_m = argmin ||importance_approx - attn_norm||
        # (단순화: 어텐션 정규값을 삼각함수 시리즈로 회귀)
        self.a_m = self._fit_fourier_coefficients(distances, all_keys)

        if save_path:
            torch.save({"mu_k": self.mu_k, "a_m": self.a_m}, save_path)

    def load_calibration(self, load_path: str) -> None:
        """저장된 캘리브레이션 파일 로드."""
        ckpt = torch.load(load_path)
        self.mu_k = ckpt["mu_k"]
        self.a_m = ckpt["a_m"]

    def estimate_importance(self, keys_pre_rope: torch.Tensor) -> torch.Tensor:
        """
        pre-RoPE K 벡터에서 삼각함수 시리즈 중요도 추정.

        수식:
          importance_i = || sum_{m=1}^{M} a_m * (sin(m * d_i) + cos(m * d_i)) ||

        여기서 d_i = || k_i - mu_k || (pre-RoPE 중심까지의 거리).

        Args:
            keys_pre_rope: [layers, heads, seq_len, head_dim]
        Returns:
            importance: [layers, heads, seq_len] — 토큰별 중요도 스칼라
        """
        assert self.mu_k is not None, "calibrate() 또는 load_calibration() 먼저 호출 필요"
        # 중심까지의 거리 계산
        diff = keys_pre_rope - self.mu_k.unsqueeze(2)  # [layers, heads, seq_len, head_dim]
        d = diff.norm(dim=-1)                           # [layers, heads, seq_len]

        # 삼각함수 시리즈 합산
        importance = torch.zeros_like(d)
        for m in range(1, self.series_terms + 1):
            m_d = m * d  # [layers, heads, seq_len]
            importance += self.a_m[:, :, m - 1].unsqueeze(2) * (torch.sin(m_d) + torch.cos(m_d))

        return importance.abs()  # 절댓값으로 단조성 보장

    def compress(
        self,
        kv_tensor: torch.Tensor,        # [layers, heads, seq_len, head_dim]
        keys_pre_rope: torch.Tensor,    # [layers, heads, seq_len, head_dim] — RoPE 적용 전 K
        compression_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """
        중요도 하위 토큰 제거. 128 토큰 윈도우마다 트리거.
        compression_ratio: 보존 비율 (0.1 = 상위 10% 토큰만 보존).
        """
        seq_len = kv_tensor.shape[2]
        importance = self.estimate_importance(keys_pre_rope)  # [layers, heads, seq_len]

        # 레이어·헤드 평균 중요도로 토큰 단위 스칼라 생성
        token_importance = importance.mean(dim=(0, 1))  # [seq_len]

        # 윈도우별 프루닝 (128 토큰 단위로 분할 처리)
        kept_indices = []
        for window_start in range(0, seq_len, self.prune_window):
            window_end = min(window_start + self.prune_window, seq_len)
            window_imp = token_importance[window_start:window_end]
            n_keep = max(1, int(len(window_imp) * compression_ratio))
            top_local = window_imp.topk(n_keep).indices + window_start
            kept_indices.append(top_local)

        kept_indices = torch.cat(kept_indices).sort().values  # 원래 순서 유지
        compressed_kv = kv_tensor[:, :, kept_indices, :]  # [layers, heads, n_keep, head_dim]

        return {
            "kv": compressed_kv,
            "kept_indices": kept_indices,
            "original_seq_len": seq_len,
            "compression_ratio": compression_ratio,
        }

    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """
        압축 해제: 보존된 인덱스 위치에 KV 복원, 나머지는 0 패딩.
        (정보 손실이 있으므로 근사 복원임)
        """
        kv_compressed = compressed["kv"]
        kept_indices = compressed["kept_indices"]
        original_len = compressed["original_seq_len"]
        layers, heads, _, dim = kv_compressed.shape
        reconstructed = torch.zeros(layers, heads, original_len, dim, dtype=kv_compressed.dtype)
        reconstructed[:, :, kept_indices, :] = kv_compressed
        return reconstructed

    def _fit_fourier_coefficients(
        self,
        distances: torch.Tensor,
        all_keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        캘리브레이션 데이터에서 삼각함수 시리즈 계수 최소자승 피팅.
        반환: [layers, heads, series_terms]
        """
        # 어텐션 정규값 계산 (피팅 타겟)
        d = all_keys.shape[-1]
        target_norms = all_keys.norm(dim=-1)  # [layers, heads, total_seq]

        layers, heads, total_seq = distances.shape
        a_m = torch.zeros(layers, heads, self.series_terms)

        for l in range(layers):
            for h in range(heads):
                dist = distances[l, h]      # [total_seq]
                y = target_norms[l, h]      # [total_seq]
                # 설계 행렬 구성: [sin(m*d) + cos(m*d)] for m=1..M
                X = torch.stack([
                    torch.sin(m * dist) + torch.cos(m * dist)
                    for m in range(1, self.series_terms + 1)
                ], dim=1)  # [total_seq, series_terms]
                # 최소자승 해
                coeff = torch.linalg.lstsq(X, y).solution  # [series_terms]
                a_m[l, h] = coeff

        return a_m
```

---

### 4. QueryCentricTriAttentionCache (Activity B+C, Cross-1)

```python
# src/cache/qc_tri_store.py 의사코드

class QueryCentricTriAttentionCache(CacheStore):
    """
    Cross-1: QueryCentricRecomputeCache + TriAttentionCodec 통합.

    설계 원칙:
    - 쿼리 관련성 높은 세그먼트: 원본 KV 보존 (재계산 품질 유지)
    - 쿼리 관련성 낮은 세그먼트: TriAttentionCodec으로 고압축 저장 (10× 압축)
    - 재계산 시에는 원본(비압축) 세그먼트 사용, 압축 마스터는 장기 보존용
    """

    def __init__(
        self,
        capacity_bytes: int,
        codec: TriAttentionCodec,
        recompute_budget_ratio: float = 0.20,
        relevance_threshold: float = 0.60,   # 코사인 유사도 임계값: 이상이면 고관련
        compression_ratio: float = 0.10,
    ) -> None:
        self._qcrc = QueryCentricRecomputeCache(capacity_bytes, recompute_budget_ratio)
        self._codec = codec
        self._compressed_store: Dict[str, Dict] = {}   # 저관련 세그먼트 압축 저장
        self._raw_store: Dict[str, torch.Tensor] = {}  # 고관련 세그먼트 원본 저장
        ...

    def put(self, key: str, value: torch.Tensor) -> None:
        # 초기 저장 시: 원본 KV를 QCRC에 저장 (쿼리 없으므로 관련성 미결정)
        self._qcrc.put(key, value)

    def put_with_query(
        self,
        key: str,
        value: torch.Tensor,             # [layers, heads, seq_len, head_dim]
        keys_pre_rope: torch.Tensor,     # pre-RoPE K (압축용)
        query_embedding: torch.Tensor,   # [head_dim] — 쿼리 K 평균 벡터
    ) -> None:
        """쿼리 관련성 기반 압축 여부 결정 후 저장."""
        seg_emb = value.mean(dim=(0, 1, 2))  # [head_dim]
        relevance = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), seg_emb.unsqueeze(0)
        ).item()

        if relevance >= self.relevance_threshold:
            # 고관련: 원본 보존
            self._raw_store[key] = value
            self._qcrc.put(key, value)
        else:
            # 저관련: TriAttentionCodec으로 압축 저장
            compressed = self._codec.compress(value, keys_pre_rope, self.compression_ratio)
            self._compressed_store[key] = compressed

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._raw_store:
            self._hit_count += 1
            return self._raw_store[key]
        if key in self._compressed_store:
            self._hit_count += 1
            return self._codec.decompress(self._compressed_store[key])
        result = self._qcrc.get(key)
        if result is None:
            self._miss_count += 1
        else:
            self._hit_count += 1
        return result

    def selective_recompute(
        self,
        query: torch.Tensor,
        cached_segments: List[str],
        budget: float = 0.20,
    ) -> List[str]:
        # 재계산은 반드시 원본 KV 기반 (압축본 미사용)
        raw_segments = [k for k in cached_segments if k in self._raw_store]
        return self._qcrc.selective_recompute(query, raw_segments, budget)

    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...
```

---

### 5. DualFilterSegmentSelector (Activity B+C)

```python
# src/cache/dual_filter_selector.py 의사코드

class DualFilterSegmentSelector:
    """
    B-1 쿼리 관련성 + C-1 pre-RoPE 중요도 이중 필터링 파이프라인.
    QueryCentricTriAttentionCache 외부에서 독립적으로 세그먼트 후보를 필터링할 때 사용.

    파이프라인:
      1단계: 쿼리 관련성 필터 → 상위 40% 세그먼트 선택 (Stage1 확대 버전)
      2단계: TriAttentionCodec 중요도 필터 → 선택된 세그먼트 내 상위 20% 토큰 보존
    """

    def __init__(
        self,
        qcrc: QueryCentricRecomputeCache,
        codec: TriAttentionCodec,
        stage1_filter_ratio: float = 0.40,   # 1단계 쿼리 관련성 필터 비율
        stage2_token_budget: float = 0.20,   # 2단계 토큰 보존 비율
    ) -> None: ...

    def select(
        self,
        query_embedding: torch.Tensor,     # [head_dim]
        candidate_segments: List[str],     # 후보 세그먼트 해시 목록
        segment_store: Dict[str, Dict],    # {key: {"kv": tensor, "embedding": tensor}}
        keys_pre_rope: Dict[str, torch.Tensor],  # {key: pre-RoPE K 텐서}
    ) -> Dict[str, torch.Tensor]:
        """
        반환: {seg_key: filtered_kv_tensor} — 이중 필터 통과 세그먼트와 필터된 KV
        """
        # --- 1단계: 쿼리 관련성 필터 (상위 40%) ---
        scores = {}
        for key in candidate_segments:
            if key not in segment_store:
                continue
            emb = segment_store[key]["embedding"]
            scores[key] = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), emb.unsqueeze(0)
            ).item()
        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        n_stage1 = max(1, int(len(sorted_keys) * self.stage1_filter_ratio))
        stage1_passed = sorted_keys[:n_stage1]

        # --- 2단계: TriAttentionCodec 중요도 필터 (토큰 상위 20%) ---
        result = {}
        for key in stage1_passed:
            kv = segment_store[key]["kv"]
            k_pre = keys_pre_rope.get(key)
            if k_pre is None:
                result[key] = kv
                continue
            compressed = self.codec.compress(kv, k_pre, self.stage2_token_budget)
            result[key] = self.codec.decompress(compressed)

        return result
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(TriAttentionCodec)를 포함하므로 다음 정확도 보존 검증이 필수다.

### perplexity 측정

| 항목 | 상세 |
|------|------|
| 모델 | GPT-2 (small, 117M) — 추가 라이선스 없이 재현 가능 |
| 데이터셋 | WikiText-2 (test split, 표준 벤치마크) |
| 측정 방법 | stride=512, max_length=1024로 sliding window perplexity |
| 허용 오차 | ±1% 이내 (예: 베이스라인 30.0 → 압축 후 29.7~30.3 허용) |
| 압축률 범위 | compression_ratio = 0.1, 0.2, 0.3, 0.5 각각 측정 |
| 비교 대상 | (a) 전체 KV 유지 (베이스라인), (b) TriAttentionCodec 0.1, (c) TriAttentionCodec 0.3 |

### 태스크 정확도 측정

| 항목 | 상세 |
|------|------|
| 벤치마크 | LongBench (3개 서브태스크: HotpotQA, QMSum, MultiFieldQA-en) |
| 허용 오차 | ±1% 이내 |
| 측정 스크립트 | `experiments/run_triattention_accuracy.py` |
| 캘리브레이션 | 학습 데이터에서 무작위 10개 요청으로 mu_k, a_m 추정 (단 1회) |

### 검증 테스트

**파일**: `tests/unit/test_tri_attention_accuracy.py`

```python
# tests/unit/test_tri_attention_accuracy.py 의사코드

def test_perplexity_within_tolerance():
    """
    TriAttentionCodec 적용 후 perplexity 변화가 ±1% 이내임을 검증.
    실제 GPT-2 모델 없이도 통과할 수 있도록 mock KV로 단위 테스트.
    실제 perplexity는 experiments/run_triattention_accuracy.py에서 측정.
    """
    codec = TriAttentionCodec(n_layers=2, n_heads=2, head_dim=64)
    # 캘리브레이션: 합성 KV 텐서 10개
    synthetic_kvs = [torch.randn(2, 2, 128, 64) for _ in range(10)]
    codec.calibrate(synthetic_kvs)

    kv = torch.randn(2, 2, 512, 64)
    keys_pre_rope = torch.randn(2, 2, 512, 64)

    compressed = codec.compress(kv, keys_pre_rope, compression_ratio=0.1)
    decompressed = codec.decompress(compressed)

    # 보존 토큰 위치에서 복원 오차 검증
    kept_idx = compressed["kept_indices"]
    original_kept = kv[:, :, kept_idx, :]
    restored_kept = decompressed[:, :, kept_idx, :]
    assert torch.allclose(original_kept, restored_kept, atol=1e-5), \
        "보존 토큰 위치 복원 오차 허용 범위 초과"

def test_compression_ratio_respected():
    """지정된 compression_ratio 비율로 토큰이 선택되는지 검증."""
    codec = TriAttentionCodec(n_layers=2, n_heads=2, head_dim=64)
    codec.calibrate([torch.randn(2, 2, 64, 64) for _ in range(10)])

    kv = torch.randn(2, 2, 256, 64)
    keys_pre_rope = torch.randn(2, 2, 256, 64)

    for ratio in [0.1, 0.2, 0.5]:
        compressed = codec.compress(kv, keys_pre_rope, compression_ratio=ratio)
        n_kept = compressed["kv"].shape[2]
        expected_max = int(256 * ratio) + 1  # 각 윈도우당 최소 1개 보장으로 약간 초과 가능
        assert n_kept <= expected_max * 2, f"ratio={ratio}: 보존 토큰 수 {n_kept} 초과"

def test_importance_scores_position_stable():
    """
    동일 K 벡터에 다른 RoPE 회전을 적용해도 pre-RoPE 기반 중요도가 안정적임을 검증.
    post-RoPE 중요도와 비교해 분산이 작아야 함.
    """
    codec = TriAttentionCodec(n_layers=1, n_heads=1, head_dim=32, series_terms=4)
    codec.calibrate([torch.randn(1, 1, 32, 32) for _ in range(10)])

    keys_pre_rope = torch.randn(1, 1, 64, 32)
    # 위치 0과 위치 32에서 동일 pre-RoPE K 벡터
    keys_pre_rope[0, 0, 32:, :] = keys_pre_rope[0, 0, :32, :]

    imp = codec.estimate_importance(keys_pre_rope)  # [1, 1, 64]
    # 동일 pre-RoPE K에 대한 중요도 차이가 작아야 함
    diff = (imp[0, 0, :32] - imp[0, 0, 32:]).abs().mean()
    assert diff < 0.1, f"pre-RoPE 중요도가 위치에 따라 불안정 (diff={diff:.4f})"
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-06.yaml
experiment:
  date: "2026-05-06"
  activity: "B+C"
  description: "QueryCentricRecomputeCache + TriAttentionCodec (B+C Cross-1)"

cache:
  type: "qc_tri"                    # QueryCentricTriAttentionCache
  capacity_bytes: 4294967296        # 4 GiB
  recompute_budget_ratio: 0.20      # 전체 토큰 중 최대 재계산 비율
  stage1_top_k_ratio: 0.50          # Stage1: attention norm 상위 50%
  relevance_threshold: 0.60         # 고관련 세그먼트 임계값

compression:
  method: "tri_attention"           # TriAttentionCodec
  compression_ratio: 0.10           # 기본 10× 압축 (보존 비율 0.1)
  series_terms: 8                   # 삼각함수 시리즈 항 수
  prune_window: 128                 # 윈도우 프루닝 크기 (토큰)
  calibration_min_requests: 10      # 최소 캘리브레이션 요청 수
  calibration_save_path: "results/2026-05-06/tri_attention_calibration.pt"

scheduler:
  type: "default"                   # Activity A 미포함

reorder:
  enabled: true                     # InfoFlowChunkReorderCache 활성화
  max_chunks: 50                    # 최대 재순서화 청크 수

dual_filter:
  enabled: true                     # DualFilterSegmentSelector 활성화
  stage1_filter_ratio: 0.40         # 1단계 쿼리 관련성 상위 40%
  stage2_token_budget: 0.20         # 2단계 토큰 보존 상위 20%

seed: 42
results_dir: "results/2026-05-06"
```

---

## 테스트 요구사항

- [x] `tests/unit/test_query_centric_recompute.py`
  - `test_put_get_basic`: put/get 기본 동작 검증
  - `test_stage1_attn_norm_filter`: Stage1 attention norm 상위 50% 필터 검증
  - `test_stage2_cosine_relevance`: Stage2 코사인 유사도 기반 관련성 순위 검증
  - `test_recompute_budget_limit`: 20% 예산 제한 — 예산 초과 세그먼트 미선택 검증
  - `test_budget_ratio_parameter`: recompute_budget_ratio 파라미터 범위 [0.10, 0.30] 검증
  - `test_hit_rate_tracking`: hit_rate() 메서드 정확성 검증
  - `test_cachestore_interface`: CacheStore 추상 메서드 전부 구현 검증

- [x] `tests/unit/test_info_flow_reorder.py`
  - `test_infoflow_score_computation`: 정보 흐름 스코어 계산 정확성
  - `test_reorder_chunks_descending`: 재순서화 후 스코어 내림차순 보장
  - `test_reorder_with_external_scores`: 외부 어텐션 스코어 가중합 검증
  - `test_rope_overhead_measurement`: 재순서화 + RoPE 재계산 시간 측정 (오버헤드 보고)
  - `test_cachestore_interface`: CacheStore 추상 메서드 전부 구현 검증

- [x] `tests/unit/test_tri_attention_accuracy.py`
  - `test_perplexity_within_tolerance`: 압축 후 보존 토큰 복원 오차 검증
  - `test_compression_ratio_respected`: 지정 비율 준수 검증
  - `test_importance_scores_position_stable`: pre-RoPE 위치 안정성 검증
  - `test_calibrate_and_save_load`: 캘리브레이션 저장·로드 왕복 검증
  - `test_decompress_roundtrip`: compress → decompress 왕복 검증 (보존 위치)
  - `test_window_pruning_boundary`: 128 토큰 윈도우 경계 처리 검증

- [x] `tests/unit/test_qc_tri_store.py`
  - `test_high_relevance_raw_storage`: 고관련 세그먼트 원본 저장 검증
  - `test_low_relevance_compressed_storage`: 저관련 세그먼트 압축 저장 검증
  - `test_selective_recompute_uses_raw`: 재계산 시 원본 KV 사용 검증
  - `test_cachestore_interface`: CacheStore 추상 메서드 전부 구현 검증

- [x] `tests/integration/test_cross_bc_dual_filter.py`
  - `test_dual_filter_pipeline_e2e`: DualFilterSegmentSelector 전체 파이프라인 E2E
  - `test_dual_filter_vs_single_filter`: 이중 필터 vs 단일 필터 정확도 비교
  - `test_qc_tri_cache_full_pipeline`: QueryCentricTriAttentionCache 전체 파이프라인

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 전부 통과** (100%, evaluation_criteria.md §0)
2. **통합 테스트 전부 통과** (100%, evaluation_criteria.md §0)
3. **CacheStore 인터페이스 준수** — QueryCentricRecomputeCache, InfoFlowChunkReorderCache, QueryCentricTriAttentionCache 각각 모든 추상 메서드 구현 (evaluation_criteria.md §0)
4. **Activity C Accuracy 보존 필수** — `test_tri_attention_accuracy.py` 통과 + `experiments/run_triattention_accuracy.py` 실행 시 perplexity 변화 ±1% 이내 (evaluation_criteria.md §4 필수 항목)
5. **Activity C 태스크 정확도 필수** — downstream 정확도 변화 ±1% 이내 (evaluation_criteria.md §4 필수 항목)
6. **비연속 히트율 목표** — 전체 히트 중 비연속 히트 비율 ≥30% (evaluation_criteria.md §3)
7. **처리량 목표** — tokens/sec 베이스라인 대비 +20% 이상 (evaluation_criteria.md §1)
8. **메모리 감소 목표** — 베이스라인 대비 −30% 이상 (evaluation_criteria.md §4)
9. **타입 힌트** — 모든 공개 함수·메서드에 존재 (evaluation_criteria.md §0)
10. **설정 YAML 존재** — `configs/experiments/2026-05-06.yaml` 생성됨 (evaluation_criteria.md §0)
11. **기존 파일 회귀 없음** — 수정 금지 파일 목록의 모든 기존 테스트 통과
12. **결과 기록** — `results/2026-05-06/metrics.json` 에 목표 지표 수치 기록

SPEC_SAVED
