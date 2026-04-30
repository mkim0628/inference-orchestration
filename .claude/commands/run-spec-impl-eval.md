# /run-spec-impl-eval

스펙 작성(3단계) + 구현·평가 루프(4~5단계)를 단독으로 실행한다.
분할 스케줄 파이프라인의 **새벽 5시 블록**이다.

## 전제 조건 확인

오늘 날짜(`YYYY-MM-DD`)를 결정한 뒤, `reports/ideas/YYYY-MM-DD.md` 파일을 읽는다.

- 파일이 없으면: "오늘 아이디어 리포트가 없습니다. /run-idea를 먼저 실행하세요." 출력 후 종료.
- `SIGNIFICANT_CHANGE: false`이면: "아이디어 변화 없음 — 구현 스킵." 출력 후 종료.
- `SIGNIFICANT_CHANGE: true`이면: 아래 단계 진행.

---

### 3단계 — 스펙 작성

`planner` 에이전트를 호출한다.

- 입력: `reports/ideas/YYYY-MM-DD.md`
- 출력: `Spec.md`
- "SPEC_SAVED" 확인 후 다음 단계 진행

---

### 4~5단계 — 구현 + 평가 루프 (최대 3회)

```
루프 N (N = 1, 2, 3):
  1. implementer 에이전트 호출
     (N=1이면 Spec.md 기반, N>1이면 evaluator 피드백 반영)
     → "IMPLEMENTATION_COMPLETE" 확인
  2. evaluator 에이전트 호출
     → N < 3이고 미충족 항목 있으면 → 피드백 획득, N+1 루프 진행
     → N = 3이거나 모든 항목 충족 → Report ① 저장 후 루프 종료
```

- 출력 (Report ①): `reports/evaluations/YYYY-MM-DD.md`
- **평가 결과와 무관하게 Report ①를 저장하고 완료한다.**

---

### 완료 — 중간 커밋

```bash
git add Spec.md src/ tests/ configs/ reports/evaluations/
git commit -m "Pipeline YYYY-MM-DD stages 3-5: spec + impl + eval"
git push origin main
```

다음을 출력한다:

```
[3-5단계 완료] YYYY-MM-DD
스펙:      Spec.md
Report ①: reports/evaluations/YYYY-MM-DD.md
루프 횟수: N회
→ 새벽 6시에 /run-vllm-summary 가 이어서 실행됩니다.
```
