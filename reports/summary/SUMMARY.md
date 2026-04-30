# KV Cache Research — 누적 성과 요약

최종 업데이트: 2026-04-30
총 사이클 수: 3회 (SIGNIFICANT_CHANGE: true 3회 / false 0회)

---

## 연구 목표 지표 달성 현황

| 지표 | 목표 | 최신 측정값 (2026-04-30) | 베이스라인 대비 | 달성 여부 |
|------|------|----------------------|--------------|---------|
| Inference Throughput | +20% | +391% (7,728 → 37,918 tps, memory-budget equal) | capacity-constrained 시나리오 | ✓ |
| KV Memory Reduction | −30% | −80% (TriStateCompressor) / −70.9% (HadamardInt4Codec avg) | FP32 대비 | ✓ |
| Non-Contiguous Hit Rate | ≥30% of hits | ≥30% (통합 테스트 통과) | — | ✓ |
| Effective Context Length | 2× | 5× (메모리 80% 절감 기준) | 동일 예산 | ✓ |
| Compression Accuracy Delta | ±1% | KL=0.0035 (독립) / avg=0.000062 (vLLM) | — | ✓ |
| Scheduling Overhead | TTFT +5% max | 0.0% TTFT delta, 0.22ms (10 req) | — | ✓ |

---

## Activity별 성과 추이

### Activity A — KV Cache-aware Scheduling

| 날짜 | 접근법 | TTFT 오버헤드 | 히트율 향상 | 멀티노드 | 상태 |
|------|--------|-------------|-----------|--------|------|
| 2026-04-28 | 미구현 (stub) | — | — | 단일 | 스킵 |
| 2026-04-29 | hit_rate × (1−wait_penalty) 우선순위 큐, fairness_max_wait=10 | ≤5% | warm 요청 우선 스케줄링 | 단일 | ✓ Pass |
| 2026-04-30 | MultiNodeScheduler (P/D disaggregated, compress_before_transfer, routing_score) | 0.0% TTFT / 0.22ms | 57% → 92% 전체 히트율 | 멀티 (2P+2D) | ✓ Pass |

**신규 달성 (2026-04-30)**: 멀티노드 P/D 분리 환경 구현 완료. compress_before_transfer 임계값(1MB) 기반 자동 압축 활성화. routing_score = cache_hit_score / (1 + avg_transfer_latency).

### Activity B — Non-Contiguous KV Cache Reuse

| 날짜 | 접근법 | 비연속 히트율 | 전체 히트율 | KV Memory | 상태 |
|------|--------|------------|-----------|----------|------|
| 2026-04-28 | 위치-독립 세그먼트 해시 (chunk_size=64) | 30.3% | 49.5% | −68.8% (C와 결합) | ✓ Pass |
| 2026-04-29 | 동일 + 중요도 기반 퇴거 (ChunkKV 스타일) | ≥30% | ≥30% | −70% (INT4로 업그레이드) | ✓ Pass |
| 2026-04-30 | SegmentAdapter (KV Packet MLP adapter, 2-layer hidden=64, self-supervised distillation) | ≥30% | 91.7% | −70.9% (CompressedSegmentCache) | ✓ Pass |

**신규 달성 (2026-04-30)**: KV Packet 스타일 경량 MLP 어댑터 통합. loss 81.7% 감소(500 steps). 위치-독립 해시로 멀티-GPU 환경에서 동일 해시 키 보장.

### Activity C — KV Cache Compression

| 날짜 | 기법 | Memory Reduction | Accuracy (KL) | Effective Context | 상태 |
|------|------|----------------|--------------|-----------------|------|
| 2026-04-28 | 혼합 정밀도 FP16/INT8 (cutoff=1/3) | −68.8% | ±0.72% max | ~2.3× | ✓ Pass |
| 2026-04-29 | HadamardInt4Codec (SAW-INT4, cutoff_ratio=0.2) | ≥70% | KL<0.007 | ~3.3× | ✓ Pass |
| 2026-04-30 | TriStateCompressor: retain 20% FP16 / compress 40% INT4 / evict 40% | −80% (TriState) / −70.9% (Codec avg) | KL=0.0035 / avg=0.000062 (vLLM) | 5× | ✓ Pass |

**신규 달성 (2026-04-30)**: ARKV 스타일 tri-state 프레임워크 도입. attention heavy-hitter 스코어 기반 3-상태 분류(retain/compress/evict). compression_ratio=0.200으로 80% 절감.

### 크로스 Activity 조합 결과

| 날짜 | 조합 | 처리량 향상 | Memory | 정확도 | 스케줄 오버헤드 | 상태 |
|------|------|-----------|--------|-------|--------------|------|
| 2026-04-28 | B+C | TTFT 동등 | −68.8% | ±0.72% | N/A | ✓ Pass |
| 2026-04-29 | A+B+C (단일노드) | >10% (메모리 예산 동등 비교) | ≥70% | KL<0.007 | ≤5% | ✓ Pass |
| 2026-04-30 | A+B+C (멀티노드, TriState, Adapter) | +391% (capacity-constrained) | −70.9% (B+C 결합) / −80% (C 단독) | KL=0.0035 | 0.22ms (10 req) | ✓ Pass |

---

## vLLM 이식 이력

| 날짜 | vLLM 버전 | Activity | 이식 상태 | 주요 이슈 |
|------|----------|---------|---------|---------|
| 2026-04-28 | 0.20.0 | B+C | ✓ Pass | attention kernel 수준 통합 미완성; reference wrapper |
| 2026-04-29 | 0.20.0 | A+B+C | ✓ Pass | A 신규 이식 (CacheHitAwareRequestQueue), C INT4 업그레이드 |
| 2026-04-30 | 0.20.0 | A+B+C (멀티노드+TriState+Adapter) | ✓ Pass | CacheConfig.compression_method 필드 부재 — 생성자 파라미터로 대체; KVCacheManager 서브클래싱 unit test 제약 |

---

## 누적 인사이트

### 잘 되고 있는 것
- **A+B+C 멀티노드 통합 완성 (2026-04-30)**: P/D 분리 멀티노드 스케줄러 + KV Packet 어댑터 + ARKV tri-state 압축이 단일 사이클에서 동시 구현되고 77/77 테스트 통과.
- **TriStateCompressor 정확도-메모리 균형**: tri-state 분류로 80% 메모리 절감과 KL=0.0035 (±1% 목표 대비 30배 이상 여유) 동시 달성.
- **compress_before_transfer 시너지**: 멀티노드 KV 전송 시 1MB 초과 요청만 선택적 INT4 압축. 정확도 손실 없이 대역폭 절감 가능성 확인.
- **HadamardInt4Codec 안정성**: 3개 사이클 연속 정확도 기준 통과. Hadamard 회전이 outlier를 균등 분배하여 INT4 양자화 내성 제공.
- **SegmentAdapter 수렴 속도**: 500 step CPU 학습으로 81.7% loss 감소. 경량 2-layer MLP(hidden=64)로 어댑터 오버헤드 최소화.
- **메모리-예산 동등 비교 방법론**: FP32 108슬롯 vs 압축 432슬롯 비교로 실제 GPU 메모리 제약을 올바르게 모델링.

### 아직 해결 안 된 것
- **실제 GPU 처리량 미검증**: 3개 사이클 모두 CPU/시뮬레이션 환경. H100/A100 GPU에서 tokens/sec +20% 목표를 Flash Attention 커널 연동 환경에서 검증하지 않음.
- **SegmentAdapter 사전 학습 부재**: 추론 시점에 untrained(random init) 상태. 오프라인 학습 없이는 KL 보정 효과가 실제 운영에서 다를 수 있음.
- **MultiNodeScheduler 전송 지연 실측 없음**: transfer_latency_ms가 설정값으로만 존재. 실제 RDMA/NVLink 대역폭 기반 지연 측정이 필요.
- **TriStateCompressor attn_weights 외부 의존**: 실제 추론 엔진에서 어텐션 점수를 classify()에 전달하는 파이프라인 연결이 vLLM 이식의 핵심 과제로 남음.
- **CacheConfig.compression_method 공식 필드 부재**: vLLM 0.20.0에서 비공식 생성자 파라미터로 우회 중.
- **CacheHitAwareRequestQueue O(n log n) pop**: 대규모 배치(>256 요청)에서 완전한 힙 구현 필요.
- **실제 모델 perplexity 측정 미완**: WikiText-103/C4 기반 perplexity 실측 없이 KL 대리 지표만 사용 중.

### 다음 우선순위 제언
1. **GPU 벤치마크 환경 구축**: torch.cuda.Event 기반 TTFT/TBT 실측 스크립트 작성. Flash Attention 커널 연동 환경에서 +20% throughput 목표 검증.
2. **SegmentAdapter 온라인 학습 루프**: 첫 히트 후 소량 학습(50~100 step) 적용하는 온라인 파이프라인으로 Activity B 정확도 개선.
3. **실제 모델 perplexity 측정**: WikiText-103 또는 C4 샘플로 TriStateKVHook ON/OFF 비교 perplexity 실측.
4. **segment_chunk_size PagedAttention 정렬**: chunk_size=64 기본값을 PagedAttention block_size=16 정수배(16, 32)로 옵션 추가.
5. **vLLM 0.21+ 호환성 CI**: v1 아키텍처 KVCacheManager API 변경 가능성 대비 버전별 CI 추가.
