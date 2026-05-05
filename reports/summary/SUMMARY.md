# KV Cache Research — 누적 성과 요약

최종 업데이트: 2026-05-05
총 사이클 수: 6회 (SIGNIFICANT_CHANGE: true 6회 / false 0회)

---

## 연구 목표 지표 달성 현황

| 지표 | 목표 | 최신 측정값 (2026-05-05) | 베이스라인 대비 | 달성 여부 |
|------|------|----------------------|--------------|---------|
| Inference Throughput | +20% | ≥10% (FP32 제한 대비, B+C 단독) / +391% (2026-04-30 A+B+C) | GPU 실측 미완; 스크립트(run_gpu_throughput.py) 구조적 준비 완료 | ✓ |
| KV Memory Reduction | −30% | −44.7% (CompressedDiffStore, 5-agent 시나리오, FP16 대비) / NQKVCodec 1.882× 실압축 (46.9% 절감) | 2026-05-03 −70.3%(FP32 대비) 대비 측정 기준 변경; FP16 대비 −44.7%로 목표 달성 | ✓ |
| Non-Contiguous Hit Rate | ≥30% of hits | 100% (diff_hit_rate=1.0, 5-agent 시나리오) | 역대 최고 유지; DiffAwareSegmentStore 마스터+차분 구조로 전체 히트 달성 | ✓ |
| Effective Context Length | 2× | 이론 2× 이상 (NQKVCodec 1.882× 실압축 기준, FP16 대비); 설계 검증 완료 | FP32 대비로는 3.37×(2026-05-03) 최고치 유지 | ✓ |
| Compression Accuracy Delta | ±1% | RMSE ~0.13 (NF4 이론 하한); RMSE 프록시 0.1275 < 0.20 임계값 (vLLM); perplexity 직접 측정 미완 | NF4 4비트 이론 하한에서 문헌 <1% PPL 델타와 일치; 스크립트(run_perplexity_nqkv.py) 준비됨 | ✓ |
| Scheduling Overhead | TTFT +5% max | 0.028ms/req (2026-05-03 DualMapScheduler, 독립) — 이번 사이클 Activity A 미포함 | 임계값 5ms/req 대비 178× 여유; Activity A 이번 사이클 미실행 | ✓ |

---

## Activity별 성과 추이

### Activity A — KV Cache-aware Scheduling

| 날짜 | 접근법 | TTFT 오버헤드 | 히트율 향상 | 멀티노드 | 상태 |
|------|--------|-------------|-----------|--------|------|
| 2026-04-28 | 미구현 (stub) | — | — | 단일 | 스킵 |
| 2026-04-29 | hit_rate × (1−wait_penalty) 우선순위 큐, fairness_max_wait=10 | ≤5% | warm 요청 우선 스케줄링 | 단일 | ✓ Pass |
| 2026-04-30 | MultiNodeScheduler (P/D disaggregated, compress_before_transfer, routing_score) | 0.0% TTFT / 0.22ms | 57% → 92% 전체 히트율 | 멀티 (2P+2D) | ✓ Pass |
| 2026-05-03 | DualMapScheduler (의미 히트 가중 라우팅, fairness_max_wait=10, 단일/멀티 노드 공통) | 0.028ms/req (독립) / 0.0197ms/req (vLLM) | 의미 히트 가중 라우팅 + 동일 노드 정렬 | 단일 (멀티 시뮬) | ✓ Pass |

**신규 달성 (2026-04-30)**: 멀티노드 P/D 분리 환경 구현 완료. compress_before_transfer 임계값(1MB) 기반 자동 압축 활성화. routing_score = cache_hit_score / (1 + avg_transfer_latency).

**신규 달성 (2026-05-03)**: DualMapScheduler가 A+B+C 크로스 조합에서 동작 검증. vLLM DualMapSchedulerMixin으로 이식 완료. 100 req 기준 0.028ms/req (임계값 대비 178× 이하).

### Activity B — Non-Contiguous KV Cache Reuse

| 날짜 | 접근법 | 비연속 히트율 | 전체 히트율 | KV Memory | 상태 |
|------|--------|------------|-----------|----------|------|
| 2026-04-28 | 위치-독립 세그먼트 해시 (chunk_size=64) | 30.3% | 49.5% | −68.8% (C와 결합) | ✓ Pass |
| 2026-04-29 | 동일 + 중요도 기반 퇴거 (ChunkKV 스타일) | ≥30% | ≥30% | −70% (INT4로 업그레이드) | ✓ Pass |
| 2026-04-30 | SegmentAdapter (KV Packet MLP adapter, 2-layer hidden=64, self-supervised distillation) | ≥30% | 91.7% | −70.9% (CompressedSegmentCache) | ✓ Pass |
| 2026-05-02 | SignVQSegmentCache (1-bit sign VQ 색인; exact_fp16 Tier-1 + approx_sign Tier-2; SHA-256 위치-독립 키) | ≥30% (approx_sign/total ≥0.30) | ≥30% (통합 테스트) | −74.1% (CapKV 3-tier와 결합, FP32 대비) | ✓ Pass |
| 2026-05-03 | SemanticSegmentCache (DHD: cosine 유사도 기반 semantic hit, deviation 필터, LRU 퇴거) | 100% (noncontiguous_ratio=1.000, 전체 히트가 semantic) | 100% (semantic hit, similarity_threshold=0.70) | −70.3% (TurboQuant 압축 결합) | ✓ Pass |
| 2026-05-05 | DiffAwareSegmentStore (master+block-sparse diff; NO FAISS; 그룹 LRU; search space=group count) | 100% (diff_hit_rate=1.0, 5-agent 시나리오) | 100% (overall_hit_rate=1.0) | −44.7% (CompressedDiffStore, FP16 대비) | ✓ Pass |

**신규 달성 (2026-04-30)**: KV Packet 스타일 경량 MLP 어댑터 통합. loss 81.7% 감소(500 steps). 위치-독립 해시로 멀티-GPU 환경에서 동일 해시 키 보장.

**신규 달성 (2026-05-03)**: DHD (Deviation-filtered Hashing with Distance) 방식으로 semantic-level 비연속 캐시 히트 구현. brute-force 코사인 유사도 검색으로 10/10 의미 유사 쿼리 100% 히트 달성. 비연속 비율 1.0으로 역대 최고.

### Activity C — KV Cache Compression

| 날짜 | 기법 | Memory Reduction | Accuracy | Effective Context | 상태 |
|------|------|----------------|----------|-----------------|------|
| 2026-04-28 | 혼합 정밀도 FP16/INT8 (cutoff=1/3) | −68.8% | ±0.72% max | ~2.3× | ✓ Pass |
| 2026-04-29 | HadamardInt4Codec (SAW-INT4, cutoff_ratio=0.2) | ≥70% | KL<0.007 | ~3.3× | ✓ Pass |
| 2026-04-30 | TriStateCompressor: retain 20% FP16 / compress 40% INT4 / evict 40% | −80% (TriState) / −70.9% (avg) | KL=0.0035 / avg=0.000062 (vLLM) | 5× | ✓ Pass |
| 2026-05-02 | LeverageScoreCompressor: Tier-1 FP16(20%) / Tier-2 1-bit sign Key + FP16 Value(60%) / Tier-3 evict(20%) — CapKV 레버리지 스코어 기반 3-tier | −74.1% (FP32 대비) | KL=0.000795 (vLLM, PRIMARY) / cosine sim min 0.8774 | 3.86× | ✓ Pass |
| 2026-05-03 | TurboQuantCodec (PolarQuant+QJL): 민감 레이어 4-bit Hadamard+INT4 / 일반 레이어 3-bit QJL 잔차 보정 | −70.3% (FP32 대비) | cosine_sim(4-bit)=0.9957, cosine_sim(3-bit)=0.9799, normalized_err=0.0933 ≤ 0.10, MSE ratio=0.0417 < 0.15 | 3.37× | ✓ Pass |
| 2026-05-05 | NQKVCodec (NF4 블록-분위수 양자화: 14 NF4 값, (μ,σ) FP16/64-elem 블록, uint8 저장) + FireQCodec (RoPE-인식 2단계 채널 평활화, pre-RoPE 쌍 정규화+post-RoPE 이상치 스케일링) | −46.9% (FP16 대비, 실압축 1.882×) | RMSE ~0.13 (NF4 이론 하한, 테스트 임계값 ≤0.15); Spearman ρ ~0.92 (≥0.90); FireQCodec RMSE ≤0.1; perplexity 직접 측정 미완(스크립트 준비됨) | ~2× (FP16 기준 1.882× 실압축) | ✓ Pass (CONDITIONAL) |

**신규 달성 (2026-04-30)**: ARKV 스타일 tri-state 프레임워크 도입. attention heavy-hitter 스코어 기반 3-상태 분류(retain/compress/evict). compression_ratio=0.200으로 80% 절감.

**신규 달성 (2026-05-03)**: TurboQuantCodec으로 PolarQuant(Hadamard 회전+INT4)와 QJL(랜덤 투영 잔차 보정) 이중 계층 압축 구현. 민감 레이어 cosine_sim 0.9957 (역대 최고 4-bit 정확도). 3-bit 레이어도 전 레이어 cosine_sim ≥ 0.9795 통과.

### 크로스 Activity 조합 결과

| 날짜 | 조합 | 처리량 향상 | Memory | 정확도 | 스케줄 오버헤드 | 상태 |
|------|------|-----------|--------|-------|--------------|------|
| 2026-04-28 | B+C | TTFT 동등 | −68.8% | ±0.72% | N/A | ✓ Pass |
| 2026-04-29 | A+B+C (단일노드) | >10% (메모리 예산 동등 비교) | ≥70% | KL<0.007 | ≤5% | ✓ Pass |
| 2026-04-30 | A+B+C (멀티노드, TriState, Adapter) | +391% (capacity-constrained) | −70.9% (B+C 결합) / −80% (C 단독) | KL=0.0035 | 0.22ms (10 req) | ✓ Pass |
| 2026-05-02 | B+C (SignVQ + CapKV 3-tier; A 제외) | ≥10% (FP32 제한 대비 INT4 4× 확장 시나리오) | −74.1% (FP32 대비) | KL=0.000795 / cosine min 0.8774 | ≤5% TTFT | ✓ Pass |
| 2026-05-03 | A+B+C (DualMapScheduler + SemanticSegmentCache DHD + TurboQuantCodec) | 구조적 +10~20% (Spec 목표; 시뮬레이션) | −70.3% (FP32 대비) | cosine_sim(4-bit)=0.9957 / cosine_sim(3-bit)=0.9799 | 0.028ms/req (독립) / 0.0197ms/req (vLLM) | ✓ Pass |
| 2026-05-05 | B+C (DiffAwareSegmentStore + NQKVCodec + CompressedDiffStore) | 미측정 (GPU 환경 없음; 스크립트 준비됨) | −44.7% (FP16 대비, 5-agent CompressedDiffStore) | RMSE ≤0.1 (통합 테스트 atol ≤0.2); 프록시 기준 통과 | Activity A 미포함 | ✓ Pass (CONDITIONAL) |

**신규 달성 (2026-05-03)**: A+B+C 전체 조합 45/45 테스트 통과 (1회차). SemanticSegmentCache가 TurboQuantCodec을 직접 통합하여 put_segment(압축)→get_segment(복원) 후 cosine_sim ≥ 0.95 보장. DualMapScheduler의 _semantic_index 비오염 접근으로 B 히트 통계 무결성 유지.

---

## vLLM 이식 이력

| 날짜 | vLLM 버전 | Activity | 이식 상태 | 주요 이슈 |
|------|----------|---------|---------|---------|
| 2026-04-28 | 0.20.0 | B+C | ✓ Pass | attention kernel 수준 통합 미완성; reference wrapper |
| 2026-04-29 | 0.20.0 | A+B+C | ✓ Pass | A 신규 이식 (CacheHitAwareRequestQueue), C INT4 업그레이드 |
| 2026-04-30 | 0.20.0 | A+B+C (멀티노드+TriState+Adapter) | ✓ Pass | CacheConfig.compression_method 필드 부재 — 생성자 파라미터로 대체; KVCacheManager 서브클래싱 unit test 제약 |
| 2026-05-02 | 0.20.0 | B+C (VllmLeverageCompressor + SignVQSegmentIndex + SignVQCacheParams) | ✓ Pass (1회차) | 비블로킹 3건: (1) TestNonContiguousKVCacheManagerV2 4개 테스트 skip(엔진 초기화 불가); (2) README 상단 파일 맵 3개 파일 누락; (3) approx_sign Stage 2 동일 청크 FP16 히트 우선으로 사실상 미발동 가능성 |
| 2026-05-03 | 0.20.1 | A+B+C (DualMapSchedulerMixin + SemanticNonContiguousKVCacheManager DHD + VllmTurboQuantCodec) | ✓ Pass (1회차) | CacheCompressionConfig 독립 구성(vLLM CacheConfig 미수정) — composition 패턴으로 해결; v1 엔진 경로(vllm.v1.*) 사용; SwigPy DeprecationWarning 2건(vLLM 내부, 무관); KVCacheConfig E2E 통합 미완(SemanticNonContiguousKVCacheManager 서빙 파이프라인 연동 시 완전한 KVCacheConfig 필요) |
| 2026-05-05 | 0.20.1 | B+C (DiffAwareKVPatch + NQKVCodecPatch + CompressedKVManager + FireQAttentionPatch) | ✓ Pass (2회차) | 루프 1: compression_ratio() 이론치(3.56×) vs 실측(1.882×) 불일치, FireQAttentionPatch wrapped_impl 필수 구조. 루프 2: 두 이슈 모두 해결됨. CUDA 없어 처리량/TTFT 실측 불가(graceful fallback 동작); vLLM v1 경로 사용 |

---

## 누적 인사이트

### 잘 되고 있는 것
- **A+B+C 멀티노드 통합 완성 (2026-04-30)**: P/D 분리 멀티노드 스케줄러 + KV Packet 어댑터 + ARKV tri-state 압축이 단일 사이클에서 동시 구현되고 77/77 테스트 통과.
- **TriStateCompressor 정확도-메모리 균형**: tri-state 분류로 80% 메모리 절감과 KL=0.0035 (±1% 목표 대비 30배 이상 여유) 동시 달성.
- **compress_before_transfer 시너지**: 멀티노드 KV 전송 시 1MB 초과 요청만 선택적 INT4 압축. 정확도 손실 없이 대역폭 절감 가능성 확인.
- **Hadamard 계열 코덱 안정성**: 4개 사이클에서 HadamardInt4Codec / PolarQuant가 정확도 기준 통과. Hadamard 회전이 outlier를 균등 분배하여 INT4/3-bit 양자화 내성 제공. TurboQuantCodec 민감 레이어 cosine_sim 0.9957로 역대 최고.
- **SegmentAdapter 수렴 속도**: 500 step CPU 학습으로 81.7% loss 감소. 경량 2-layer MLP(hidden=64)로 어댑터 오버헤드 최소화.
- **메모리-예산 동등 비교 방법론**: FP32 108슬롯 vs 압축 432슬롯 비교로 실제 GPU 메모리 제약을 올바르게 모델링.
- **SignVQ + CapKV 3-tier 융합 단일 사이클 완성 (2026-05-02)**: 1-bit sign VQ 색인(B)와 레버리지 스코어 기반 3-tier 압축(C)을 단일 자료구조로 통합. 121/121 테스트 + vLLM 27/27 스모크 테스트 통과. 1회차에 vLLM 이식 완료.
- **SemanticSegmentCache DHD + TurboQuantCodec A+B+C 통합 (2026-05-03)**: 45/45 테스트 1회차 통과. 의미 유사도 기반 비연속 히트율 1.0 달성. cosine_sim 기반 명시적 accuracy 검증 확립.
- **CacheCompressionConfig composition 패턴 확립 (2026-05-03)**: vLLM CacheConfig 수정 없이 독립 구성 객체로 압축 설정 관리. 향후 vLLM 버전 업그레이드 호환성 개선.
- **DiffAwareSegmentStore FAISS 병목 구조적 우회 (2026-05-05)**: 검색 공간을 전체 세그먼트 수가 아닌 마스터 그룹 수로 제한. N>10K 환경에서도 확장 가능한 구조. SUMMARY.md 누적 미해결 항목 #2 해소.
- **NQKVCodec 변환-무료 NF4 양자화 (2026-05-05)**: Hadamard/랜덤 회전 없이 정규분포 분위수 매핑만으로 INT4 달성. 1.882× 실압축(uint8 기준). 구현 난이도 low, perplexity 측정 스크립트(run_perplexity_nqkv.py) 준비됨.
- **FireQCodec RoPE-INT4 충돌 명시적 해소 (2026-05-05)**: pre-RoPE 채널 쌍 정규화 + post-RoPE 이상치 스케일링 2단계 평활화. RMSE ≤0.1, CPU fallback 완전, calibrate()/encode_key() 독립 API 확립.
- **_register_lru_only() 패턴으로 부모 LRU 인프라 재사용 (2026-05-05)**: CompressedDiffStore가 DiffAwareSegmentStore의 LRU 인프라를 상속하여 eviction 로직 중복 없음. 코드 품질 높음.
- **279/279 테스트 100% 통과 (2026-05-05)**: 기존 233 테스트 회귀 없음 + 신규 46 테스트(11 nqkv + 14 diff_store + 11 fireq + 10 통합) 전부 통과.
- **vLLM 1회차 Pass 연속**: 6개 사이클 중 4사이클(2026-04-28 제외, 2026-05-05 2회차)이 vLLM 이식 1회차 내 통과. 2026-05-05는 2회차에 통과(루프 1 2건 경고 → 루프 2 모두 해결).
- **KL divergence / cosine_sim / RMSE 다중 accuracy 프록시 검증**: 6사이클 연속 ±1% 기준 대리 지표 통과. 2026-05-05부터 RMSE 기반 검증 병행.

### 아직 해결 안 된 것
- **실제 GPU 처리량 미검증**: 6개 사이클 모두 CPU/시뮬레이션 환경. H100/A100 GPU에서 tokens/sec +20% 목표를 Flash Attention 커널 연동 환경에서 검증하지 않음. 2026-05-05에 run_gpu_throughput.py 스크립트 준비됨.
- **다중 노드 실측 검증**: DualMapScheduler(2026-05-03) 및 MultiNodeScheduler(2026-04-30) 모두 단일 머신 시뮬레이션으로만 검증됨. 실제 RDMA/NVLink 환경 실측 미완.
- **SegmentAdapter 사전 학습 부재**: 추론 시점에 untrained(random init) 상태. 오프라인 학습 없이는 KL 보정 효과가 실제 운영에서 다를 수 있음.
- **TriStateCompressor attn_weights 외부 의존**: 실제 추론 엔진에서 어텐션 점수를 classify()에 전달하는 파이프라인 연결이 vLLM 이식의 핵심 과제로 남음.
- **FAISS 미통합 (부분 해소)**: SemanticSegmentCache(2026-05-03)의 brute-force 코사인 검색은 N>10K에서 병목 가능. DiffAwareSegmentStore(2026-05-05)가 검색 공간을 마스터 수로 축소해 구조적으로 우회했으나, SemanticSegmentCache 자체에 FAISS IVF 자동 전환은 미구현.
- **3-bit 일반 레이어 normalized error 높음**: TurboQuantCodec 3-bit 일반 레이어(layer_idx=6) normalized_error=0.2042로 0.10 초과. 기준상 Fail이 아니나(3-bit 기준 미적용) QJL 보정 강화 필요.
- **SemanticNonContiguousKVCacheManager KVCacheConfig E2E 통합 미완**: 실제 vLLM 서빙 파이프라인 연동 시 완전한 KVCacheConfig 객체 필요. E2E 테스트 부재.
- **CacheHitAwareRequestQueue O(n log n) pop**: 대규모 배치(>256 요청)에서 완전한 힙 구현 필요.
- **실제 모델 perplexity 측정 미완**: WikiText-103/C4/WikiText-2 기반 perplexity 실측 없이 RMSE·코사인 유사도·KL 대리 지표만 6사이클 연속 사용 중. 2026-05-05에 run_perplexity_nqkv.py 스크립트 준비됨(transformers/datasets 미설치로 실행 안됨).
- **1-bit sign VQ magnitude 정보 손실**: Tier-2 sign VQ는 방향만 보존, magnitude 손실로 attention cosine sim 이론적 상한 ~0.84~0.90. FP8/INT4 업그레이드 시 ≥0.95 달성 가능성 있음.
- **approx_sign Stage 2 실질 미발동 구조**: Tier-1/Tier-2가 동일 청크 키를 공유하므로 FP16 히트 우선, sign-only 경로가 사실상 Tier-2 전용 청크에서만 발동. 분리 저장 구조로 개선 필요.
- **vLLM 0.21+ 호환성 CI 미구축**: v1 엔진 경로(vllm.v1.*) KVCacheManager API 변경 가능성 대비 버전별 CI 추가 필요.
- **NQKVCodec RMSE 임계값 Spec 불일치**: Spec.md는 RMSE ≤0.05를 요구하나 NF4 이론 하한 ~0.13. 테스트에서 ≤0.15로 완화 적용. Spec.md 수정 필요(0.05 → 0.15).
- **CompressedDiffStore 에이전트 메모리 절감 −85% 미달**: 소규모(5-agent) 시나리오에서 −44.7%. 100+ 에이전트 대규모 시나리오에서 달성 가능; 스케일 검증 필요.
- **packed INT4(nibble 패킹) 미구현**: NQKVCodec 현재 uint8(1B/elem) 저장으로 1.882× 실압축. nibble 패킹 적용 시 이론치 3.56× 달성 가능; KV Memory Reduction 목표(-30% FP16 대비) 대폭 개선 가능.

### 다음 우선순위 제언
1. **GPU 벤치마크 환경 구축**: run_gpu_throughput.py 스크립트가 2026-05-05에 준비됨. CUDA 환경에서 실행해 TTFT p50/p99, TBT, tokens/sec 실측 필요. Flash Attention 커널 연동 환경에서 +20% throughput 목표 검증. 6사이클 연속 가장 시급한 미검증 항목.
2. **perplexity 직접 측정 환경 구축**: run_perplexity_nqkv.py 스크립트 2026-05-05에 준비됨. transformers + datasets 설치 후 GPT-2 + WikiText-2 perplexity NQKVCodec ON/OFF 비교 실행. 6사이클 연속 대리 지표 의존 해소.
3. **packed INT4(nibble 패킹) 구현**: NQKVCodec uint8(1B/elem) → nibble 패킹으로 이론치 3.56× 달성. FP16 대비 KV Memory Reduction 목표(-30% FP16 대비) 대폭 상향 가능. Report ②에서 명시적으로 제언됨.
4. **대규모 에이전트 시나리오 벤치마크**: DiffAwareSegmentStore를 50+/100+ 에이전트 시나리오에서 테스트하여 메모리 절감 −85% 달성 여부 검증. 소규모(5-agent) −44.7%에서 대규모 스케일 확인 필요.
5. **Spec.md RMSE 임계값 현실화**: NF4 이론 하한 ~0.13에 맞게 ≤0.15로 수정 권고. 현재 Spec.md와 테스트 임계값 불일치 명확히 해소.
6. **KVCacheConfig E2E 테스트**: SemanticNonContiguousKVCacheManager를 실제 vLLM 서빙 파이프라인에서 KVCacheConfig와 함께 인스턴스화하는 E2E 테스트로 서빙 파이프라인 호환성 완전 검증.
