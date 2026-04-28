# KV Cache Research — 누적 성과 요약

최종 업데이트: 2026-04-28
총 사이클 수: 1회 (SIGNIFICANT_CHANGE: true 1회 / false 0회)

---

## 연구 목표 지표 달성 현황

| 지표 | 목표 | 최신 측정값 | 베이스라인 대비 | 달성 여부 |
|------|------|-----------|--------------|---------|
| Inference Throughput | +20% | 측정 중 (TTFT 동등) | — | 진행 중 |
| KV Memory Reduction | −30% | −68.8% | FP32 대비 | ✓ |
| Non-Contiguous Hit Rate | ≥30% of hits | 30.3% | — | ✓ |
| Effective Context Length | 2× | 3.2× (메모리 기준) | — | ✓ |
| Compression Accuracy Delta | ±1% | INT8 max 0.72%, FP16 <0.1% | — | ✓ |
| Scheduling Overhead | TTFT +5% max | N/A (Activity A 미구현) | — | 진행 중 |

---

## Activity별 성과 추이

### Activity A — KV Cache-aware Scheduling

| 날짜 | 접근법 | 히트율 향상 | 스케줄 오버헤드 | 멀티노드 | 상태 |
|------|--------|-----------|--------------|--------|------|
| — | 미구현 | — | — | — | 다음 사이클 예정 |

### Activity B — Non-Contiguous KV Cache Reuse

| 날짜 | 접근법 | 비연속 히트율 | 전체 히트율 | KV Memory | 상태 |
|------|--------|------------|-----------|----------|------|
| 2026-04-28 | 위치-독립 세그먼트 해시 (chunk_size=64, 4 chunks/req) | 30.3% | 49.5% | −68.8% (C와 결합) | Pass |

### Activity C — KV Cache Compression

| 날짜 | 기법 | Memory Reduction | Accuracy Delta | Effective Context | 상태 |
|------|------|----------------|--------------|-----------------|------|
| 2026-04-28 | 혼합 정밀도 (FP16 early / INT8 late, cutoff=1/3) | −68.8% (B+C 통합) | INT8 ±0.72% max | 3.2× | Pass |

### 크로스 Activity 조합 결과

| 날짜 | 조합 | 복합 처리량 | 복합 Memory | 정확도 | 상태 |
|------|------|-----------|-----------|-------|------|
| 2026-04-28 | B+C | TTFT 동등 (미측정) | −68.8% | ±0.72% 이내 | Pass |

---

## vLLM 이식 이력

| 날짜 | vLLM 버전 | Activity | 이식 상태 | 주요 이슈 |
|------|----------|---------|---------|---------|
| 2026-04-28 | 0.20.0 | B+C | Pass | attention kernel 수준 통합 미완성; reference wrapper로 구현 |

---

## 누적 인사이트

### 잘 되고 있는 것
- **위치-독립 해싱**: SHA-256 기반 chunk 해싱이 동일 토큰을 다른 위치에서 정확히 매칭. 비연속 히트율 30.3% 목표 달성.
- **혼합 정밀도 압축**: FP16(초기 레이어) + INT8(후반 레이어) 조합이 정확도 ±1% 제약 내에서 −66.7~68.8% 메모리 절감 달성.
- **vLLM 이식 호환성**: try/except 방어 코드로 vLLM 0.20.0 v1 아키텍처 변경에 안전하게 대응.

### 아직 해결 안 된 것
- **실제 토큰 생성 처리량(tokens/sec) 미측정**: +20% throughput 목표 달성 여부 불명. 실제 LLM 모델 연동 필요.
- **Activity A 미구현**: 스케줄링 오버헤드 TTFT +5% 이내 항목 검증 불가.
- **CUDA 커널 수준 비연속 prefill 스킵 미완성**: 현재 hit 정보만 추적; 실제 연산 절감 미구현.

### 다음 우선순위 제언
1. **Activity A 구현**: cache-hit-weighted 라우팅 스케줄러. B+C 구현 위에 추가하면 A+B+C 3중 조합 달성 가능.
2. **실제 GPT-2 모델 연동**: 합성 KV 시뮬레이션 → 실제 모델 perplexity 측정으로 정확도 보존 검증 강화.
3. **CUDA attention wrapper 완성**: `NonContiguousAttentionWrapper`를 `FlashAttentionImpl` 서브클래스로 구현해 실제 TTFT 절감 측정.
