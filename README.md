# Orchestrating Non-Contiguous KV Cache Reuse with Accuracy-Preserving KV Cache Compression

> **Autonomous research harness** that continuously discovers, proposes, implements, and validates novel KV cache optimization techniques — and ports the best ones to production-grade [vLLM](https://github.com/vllm-project/vllm).

---

## Research Goal

Efficiently reuse and compress KV Caches to simultaneously increase **inference throughput** and **memory efficiency**, enabling long-context processing and long-term memory support within limited GPU memory budgets.

| Goal Metric | Target |
|-------------|--------|
| Inference Throughput | +20% vs baseline |
| KV Cache Memory Reduction | −30% vs baseline |
| Non-Contiguous Cache Hit Rate | ≥30% of hits from non-contiguous segments |
| Effective Context Length | 2× at same memory budget |
| Compression Accuracy Delta | ±1% perplexity / task accuracy |
| Scheduling Overhead | TTFT p50 +5% max |

## Three Research Activities

| Activity | Focus | Key Question |
|----------|-------|-------------|
| **A** | KV Cache-aware Scheduling / Orchestration | How to batch and order requests to maximize cache reuse? |
| **B** | Non-Contiguous KV Cache Reuse | Can we reuse KV segments at arbitrary positions, not just shared prefixes? |
| **C** | KV Cache Compression | How small can we make the KV cache while preserving accuracy? |

Each daily cycle, the pipeline selects the highest-priority activity or combination (A+B, B+C, A+B+C) and implements it end-to-end.

---

## Daily Pipeline

Every day, an autonomous multi-agent pipeline runs end-to-end:

1. Scans the latest papers and open-source releases for new ideas
2. Proposes novel algorithms based on what it finds
3. Implements and validates them against strict evaluation criteria
4. **Ports the validated algorithm into vLLM** and benchmarks it there

---

## Daily Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  매일 KST 06:00                                                       │
│                                                                       │
│  1. 트렌드 수집  ─── trend-sensor      → reports/trends/              │
│  2. 아이디어    ─── idea-generator    → reports/ideas/                │
│                    │                                                  │
│                    └─ SIGNIFICANT_CHANGE: false → STOP               │
│                                                                       │
│  3. 스펙 작성   ─── planner           → Spec.md                       │
│  4. 구현        ─── implementer   ◄──────────────────────┐            │
│  5. 평가        ─── evaluator     ──── feedback (×3 max) ┘            │
│                    │                                                  │
│                    └→ Report ① (통과 여부 무관하게 저장 후 계속)        │
│                                                                       │
│  6. vLLM 이식   ─── vllm-porter   → vllm_integration/                │
│  7. vLLM 평가   ─── vllm-evaluator ─── feedback (×3 max) ┐           │
│                    │               ◄─── vllm-porter       ┘           │
│                    └→ Report ②                                        │
└──────────────────────────────────────────────────────────────────────┘
```

| 단계 | 에이전트 | 역할 |
|------|---------|------|
| 1. 트렌드 수집 | `trend-sensor` | arXiv·GitHub·블로그에서 KV 캐시 관련 동향 수집 |
| 2. 아이디어 생성 | `idea-generator` | 트렌드 + 과거 아이디어 종합 → 새 아이디어 제안 |
| 3. 스펙 작성 | `planner` | 아이디어 → 구체적인 `Spec.md` 작성 |
| 4. 구현 | `implementer` | `Spec.md` 기반 Python 구현 |
| 5. 평가 | `evaluator` | `evaluation_criteria.md` 기준 평가·피드백 루프 |
| 6. vLLM 이식 | `vllm-porter` | 알고리즘을 최신 vLLM 코드베이스에 이식 |
| 7. vLLM 평가 | `vllm-evaluator` | vLLM 환경에서 성능·정확성 검증 |

---

## Repository Structure

```
.
├── CLAUDE.md                    # 하네스 전체 규칙 및 가이드
├── Spec.md                      # 현재 구현 스펙 (planner 생성)
├── evaluation_criteria.md       # 평가 기준 (수동 관리)
│
├── .claude/
│   ├── agents/                  # 에이전트 정의
│   │   ├── trend-sensor.md
│   │   ├── idea-generator.md
│   │   ├── planner.md
│   │   ├── implementer.md
│   │   ├── evaluator.md
│   │   ├── vllm-porter.md       # vLLM 이식 에이전트
│   │   └── vllm-evaluator.md    # vLLM 환경 평가 에이전트
│   └── commands/
│       ├── run-pipeline.md      # /run-pipeline
│       ├── run-trend.md         # /run-trend
│       └── run-idea.md          # /run-idea
│
├── src/
│   ├── cache/                   # KV 캐시 구현체
│   │   ├── base.py              # CacheStore 추상 인터페이스
│   │   ├── contiguous.py        # 베이스라인 (연속 캐시)
│   │   ├── segmented.py         # 세그먼트 해시 캐시
│   │   └── radix.py             # Radix 트리 캐시
│   ├── engine/                  # 추론 엔진 래퍼
│   ├── metrics/                 # 측정 지표 (히트율·지연·메모리)
│   └── utils/
│
├── vllm_integration/            # vLLM 이식 코드 (vllm-porter 생성)
│   ├── block_manager_patch.py
│   ├── attention_backend_patch.py
│   └── install.sh               # pip install --upgrade vllm + 패치 적용
│
├── reports/
│   ├── trends/                  # 트렌드 리포트
│   ├── ideas/                   # 아이디어 리포트
│   ├── evaluations/             # 알고리즘 검증 리포트 (Report ①)
│   └── vllm-evaluations/        # vLLM 환경 리포트 (Report ②)
│
├── configs/                     # 실험 설정 YAML
├── data/                        # 프롬프트 데이터셋
├── tests/
│   ├── unit/
│   └── integration/
└── results/                     # 실험 결과 (git-ignored)
```

---

## Output Reports

| 리포트 | 경로 | 생성 에이전트 | 내용 |
|--------|------|------------|------|
| ① 알고리즘 검증 | `reports/evaluations/YYYY-MM-DD.md` | `evaluator` | Cache Hit Rate, TTFT, TBT, 메모리, 코드 품질 |
| ② vLLM 검증 | `reports/vllm-evaluations/YYYY-MM-DD.md` | `vllm-evaluator` | vLLM latest 기준 처리량, 지연, 호환성, 회귀 |

---

## Evaluation Criteria

[`evaluation_criteria.md`](./evaluation_criteria.md) 파일에 정의된 기준:

- **필수**: 단위·통합 테스트 통과, CacheStore 인터페이스 준수, 재현성
- **성능**: 베이스라인 대비 Cache Hit Rate +5%p 이상
- **지연**: TTFT p50 회귀 +10% 이내
- **vLLM**: 최신 vLLM 빌드 통과, 처리량 회귀 없음

---

## Tech Stack

| 구성 요소 | 버전 |
|---------|------|
| Python | ≥ 3.10 |
| vLLM | latest (`pip install --upgrade vllm`) |
| PyTorch | ≥ 2.2 |
| Transformers | ≥ 4.40 |
| Anthropic SDK | ≥ 0.40 |

```bash
pip install -r requirements.txt
pip install --upgrade vllm
```

---

## Manual Execution

```bash
# 전체 RALPH 루프 실행
/run-pipeline

# 개별 단계
/run-trend      # 트렌드 수집만
/run-idea       # 아이디어 생성만
```

자동 실행: 매일 KST 06:00 (Anthropic Cloud Remote Agent)

---

## License

MIT
