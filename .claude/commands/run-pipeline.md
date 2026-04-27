# /run-pipeline

비연속 KV 캐시 재사용 연구의 전체 RALPH 루프를 실행한다.

## 실행 순서

오늘 날짜를 `YYYY-MM-DD` 형식으로 결정한 뒤 다음 순서로 진행한다.

### 1단계 — Research (트렌드 수집)

`trend-sensor` 에이전트를 호출한다.

- 출력: `reports/trends/YYYY-MM-DD.md`
- "TREND_REPORT_SAVED" 확인 후 다음 단계 진행

### 2단계 — Analyze (아이디어 생성)

`idea-generator` 에이전트를 호출한다.

- 출력: `reports/ideas/YYYY-MM-DD.md`
- **SIGNIFICANT_CHANGE 확인**:
  - `false` → "오늘은 새로운 아이디어 변화가 없습니다. 파이프라인을 중단합니다." 출력 후 종료
  - `true` → 3단계 진행

### 3단계 — Launch (스펙 작성)

`planner` 에이전트를 호출한다.

- 출력: `Spec.md`
- "SPEC_SAVED" 확인 후 다음 단계 진행

### 4~5단계 — Program + Heuristic (구현 + 평가 루프)

최대 3회 다음을 반복한다:

```
루프 N (N = 1, 2, 3):
  1. implementer 에이전트 호출 (N=1이면 Spec.md 기반, N>1이면 피드백 반영)
     → "IMPLEMENTATION_COMPLETE" 확인
  2. evaluator 에이전트 호출
     → N < 3이고 미충족 항목 있으면 → 피드백 획득, N+1 루프 진행
     → N = 3이거나 모든 항목 충족 → "EVAL_REPORT_SAVED" 확인 후 루프 종료
```

### 완료

다음을 출력한다:

```
RALPH 루프 완료 — YYYY-MM-DD
트렌드 리포트: reports/trends/YYYY-MM-DD.md
아이디어 리포트: reports/ideas/YYYY-MM-DD.md
스펙: Spec.md
평가 리포트: reports/evaluations/YYYY-MM-DD.md
총 구현-평가 루프: N회
```
