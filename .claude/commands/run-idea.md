# /run-idea

아이디어 생성 단계만 단독으로 실행한다.

최신 트렌드 리포트(`reports/trends/`)가 있어야 한다. 없으면 `/run-trend` 를 먼저 실행하라고 안내한다.

`idea-generator` 에이전트를 호출해 오늘 날짜의 아이디어 리포트를 생성한다.

출력: `reports/ideas/YYYY-MM-DD.md`
SIGNIFICANT_CHANGE 값을 출력해 사용자에게 파이프라인 진행 여부를 알린다.
