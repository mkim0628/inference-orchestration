# /run-vllm-summary

vLLM 이식·평가 루프(6~7단계) + 누적 요약(8단계) + 최종 커밋을 단독으로 실행한다.
분할 스케줄 파이프라인의 **새벽 6시 블록**이다.

## 전제 조건 확인

오늘 날짜(`YYYY-MM-DD`)를 결정한 뒤, `reports/evaluations/YYYY-MM-DD.md`(Report ①)를 확인한다.

- 파일이 없으면: "오늘 Report ①가 없습니다. /run-spec-impl-eval을 먼저 실행하세요." 출력 후 종료.
- 파일이 있으면: 아래 단계 진행. (Report ①의 통과/미통과와 무관하게 진행한다.)

---

### 6~7단계 — vLLM 이식 + vLLM 평가 루프 (최대 3회)

```
루프 N (N = 1, 2, 3):
  1. vllm-porter 에이전트 호출
     - 작업 시작 전 항상: pip install --upgrade vllm
     - N=1이면 Report ① 및 src/cache/ 기반 이식
     - N>1이면 vllm-evaluator 피드백 반영
     → "VLLM_PORT_COMPLETE" 확인
  2. vllm-evaluator 에이전트 호출
     → N < 3이고 미충족 항목 있으면 → 피드백 획득, N+1 루프 진행
     → N = 3이거나 모든 항목 충족 → Report ② 저장 후 루프 종료
```

- 출력 (Report ②): `reports/vllm-evaluations/YYYY-MM-DD.md`
- **vLLM 평가 결과와 무관하게 Report ②를 저장하고 8단계로 진행한다.**

---

### 8단계 — 누적 요약 갱신

`summarizer` 에이전트를 호출한다.

- 출력: `reports/summary/SUMMARY.md` (갱신)
- 출력: `reports/summary/YYYY-MM-DD-delta.md` (신규)
- "SUMMARY_UPDATED" 확인 후 최종 커밋 진행

---

### 완료 — 최종 커밋 & Push

```bash
git add reports/ vllm_integration/
git commit -m "Pipeline YYYY-MM-DD stages 6-8: vLLM port + summary"
git push origin main
```

다음을 출력한다:

```
[6-8단계 완료] YYYY-MM-DD
Report ②:   reports/vllm-evaluations/YYYY-MM-DD.md
누적 요약:   reports/summary/SUMMARY.md
사이클 델타: reports/summary/YYYY-MM-DD-delta.md
vLLM 이식 루프: M회
vLLM 버전:  X.Y.Z
=== 오늘 파이프라인 전체 완료 ===
```
