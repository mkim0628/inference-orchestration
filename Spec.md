<!-- 변경 이유 (이전 Spec.md: 2026-05-15 대비):
이전 사이클(2026-05-15)은 A+B+C 조합이었다:
  - C-1 LookaheadKVEvictionCodec (드래프트-프리 미래-인식 LoRA KV 퇴거)
  - B-1 RelayUShapeLayerSelectiveSegmentCache (U자형 레이어-선택적 세그먼트 재사용)
  - Cross-1 LookaheadRelaySegmentCache (B+C 이중 필터)
  - A-1 RadixFeatherBatchScheduler (Radix 트리 동질성 배치 중단)

이번 사이클(2026-05-16)은 A+C 조합으로 초점이 전환된다. Activity B는 이번 사이클에서
신규 논문이 없어 직접 구현 대상에서 제외된다.

주요 변경:
1. [Activity C 교체] LookaheadKVEvictionCodec(post-write token eviction) →
   GlobalRetentionGateEvictionCodec(전역 크로스-레이어 경쟁 퇴거).
   기존 기법들(LaProx, LookaheadKV)이 레이어별/요청별 독립 퇴거를 결정하는 반면,
   GlobalRetentionGateEvictionCodec은 공유 최종 점수 투영(shared final score projection)으로
   모든 레이어·헤드 KV가 단일 전역 예산 풀에서 경쟁하는 새로운 설계 원칙을 도입한다.
   이론적 근거: "Make Each Token Count"(2605.09649, Yale+CUHK).

2. [Activity A 교체] RadixFeatherBatchScheduler(배치 동질성 신호) →
   NAtHDDROffloadingScheduler(의미-인식 4-티어 DDR 오프로딩 최소-퇴거 스케줄러).
   기존 A 기법들이 "어떤 요청을 어느 노드로 라우팅"에 집중한 반면,
   NAtHDDROffloadingScheduler는 "메모리 압박 시 영구 퇴거 대신 DDR 오프로딩을 기본 정책"으로
   설정하고 누적 어텐션 점수로 4-티어 메모리 분류를 수행한다.
   이론적 근거: NAtH(2605.09490) "정확도는 퇴거 비율에만 의존".

3. [Cross A+C 신규] NAtHRetentionTierDecider — A-2 NAtHDDROffloadingScheduler +
   C-2 GlobalRetentionGateEvictionCodec의 이중 신호(누적 어텐션 점수 + 전역 리텐션 게이트)로
   4-티어 메모리 분류를 수행하는 최초의 A+C 크로스 모듈.

4. [보존 파일] 기존 모든 파일(lookahead_kv_eviction.py, relay_ulayer_segment.py 등)은
   이번 사이클에서 수정하지 않는다. 기존 단위·통합 테스트가 회귀 없이 통과해야 한다.

5. [인터페이스 유지] src/cache/base.py는 이번 사이클에서 수정하지 않는다.
   CacheStore 6개 추상 메서드를 모든 신규 구현체가 완전 구현한다.

6. [Activity C 필수] GlobalRetentionGateEvictionCodec은 accuracy-preserving 검증 계획 없이
   완성 불가. WikiText-2 perplexity proxy ±1% + LongBench 8개 서브태스크 proxy +
   전역 예산(10%/20%/30%/50%/70%) 별 정확도 곡선 + LaProx·LookaheadKV 3방향 비교.
-->

# Spec — 2026-05-16

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-16.md`

**최우선 구현 타겟**:
- **C-2 (주)**: GlobalRetentionGateEvictionCodec — 전역 리텐션 게이트 크로스-레이어 경쟁 퇴거 코덱
  ("Make Each Token Count", arXiv 2605.09649, Yale+CUHK)
- **A-2 (주)**: NAtHDDROffloadingScheduler — NAtH 의미-인식 DDR 오프로딩 최소-퇴거 스케줄러
  (NAtH, arXiv 2605.09490)
- **Cross-1 (주)**: NAtHRetentionTierDecider — A-2 + C-2 이중 신호 4-티어 메모리 결정기

**해결하려는 문제**:

- **Activity C**: 기존 KV 퇴거 기법(LaProx, LookaheadKV)이 레이어별·요청별 독립 퇴거를 결정해
  "레이어 A에서 중요한 토큰이 레이어 B에서 퇴거되는" 비일관성이 발생한다.
  GlobalRetentionGateEvictionCodec은 경량 리텐션 게이트와 공유 최종 점수 투영으로 모든
  레이어·헤드 KV가 단일 전역 예산 풀에서 경쟁해 최적 퇴거 집합을 결정한다.
  이론적으로 긴 컨텍스트에서 전체 캐시 추론 성능을 초과할 수 있다.

- **Activity A**: 기존 KV 퇴거 기반 메모리 관리가 추론(reasoning) 워크로드에서 3% 퇴거만으로
  정확도가 0~2.5%로 붕괴하는 한계를, NAtH의 이론적 발견("정확도는 영구 퇴거 비율에만 의존하며
  DDR 오프로딩은 제로 근사 오류(zero-approximation-error)") 을 스케줄러 설계 원칙으로 적용한다.
  메모리 압박 시 영구 퇴거 대신 DDR 오프로딩을 기본 정책으로 사용하고 누적 어텐션 점수 EMA로
  4-티어 메모리 분류를 수행해 Activity A의 "캐시 퇴거율 최소화"와
  Activity C의 "정확도 ±1% 이내"를 스케줄러 수준에서 동시에 달성한다.

- **Cross A+C**: NAtHDDROffloadingScheduler(누적 어텐션 점수)와 GlobalRetentionGateEvictionCodec
  (전역 리텐션 게이트)의 이중 신호로 4-티어 분류 정밀도를 높이고, 영구 퇴거 비율을 3% 미만으로
  유지하면서 HBM 점유를 DDR 오프로딩으로 해방한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling  (NAtHDDROffloadingScheduler — 주)
- [ ] Activity B: Non-Contiguous KV Cache Reuse  (이번 사이클 미포함)
- [x] Activity C: KV Cache Compression  (GlobalRetentionGateEvictionCodec — 주)

---

## 목표

- [ ] 목표 1 (evaluation_criteria.md §4 Activity C 필수): perplexity 변화 ±1% 이내
      — attention output relative error < 0.01 (WikiText-2 proxy)
      — 전역 예산 budget_ratio = 0.3 / 0.5 / 0.7 각각 측정
- [ ] 목표 2 (evaluation_criteria.md §4 Activity C 필수): downstream 태스크 정확도 변화 ±1% 이내
      — LongBench 8개 서브태스크 proxy (KL divergence < 0.015, cosine >= 0.99)
      — "전체 캐시 초과 성능" 구간 (컨텍스트 > 8K 토큰) 포함 측정
- [ ] 목표 3 (evaluation_criteria.md §4 Activity C): KV Cache Memory Reduction >= −30%
      — budget_ratio=0.3 시 −70% 목표, budget_ratio=0.5 시 −50% 목표
- [ ] 목표 4 (evaluation_criteria.md §4 Activity C): Effective Context Length 동일 메모리 2× 이상
      — budget_ratio=0.3 (30% 보존) → 컨텍스트 길이 ~3× 증가 가능
- [ ] 목표 5 (evaluation_criteria.md §2 Activity A): 스케줄링 오버헤드 TTFT p50 +5% 이내
      — NAtHDDROffloadingScheduler 누적 어텐션 점수 계산 오버헤드 측정
- [ ] 목표 6 (evaluation_criteria.md §2 Activity A): 캐시 퇴거율 최소화
      — 영구 퇴거(Tier 4) 비율 <= max_eviction_ratio(3%) 검증
      — DDR 오프로딩(Tier 2+3) 비율 측정
- [ ] 목표 7 (evaluation_criteria.md §1 처리량): 베이스라인 대비 tokens/sec +20% 이상
      — DDR 오프로딩으로 HBM 해방 + 전역 경쟁 퇴거 노이즈 제거 복합 효과
- [ ] 목표 8 (evaluation_criteria.md §5 크로스 조합 C 포함): 복합 적용 후 accuracy ±1% 이내
      — NAtHRetentionTierDecider(Cross A+C) 기준 이중 신호 적용 후 측정
- [ ] 목표 9 (evaluation_criteria.md §4 Activity C): LaProx·LookaheadKV 대비 정확도 비교
      — 동일 budget_ratio=0.3 설정에서 3방향 비교 (GlobalRetentionGate vs LaProx vs LookaheadKV)
- [ ] 목표 10 (evaluation_criteria.md §4 Activity C): 퇴거 오버헤드 TTFT +10% 이내
      — 리텐션 게이트 점수 계산 + 공유 투영 추가 지연 측정

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/global_retention_gate_eviction.py` | C | GlobalRetentionGateEvictionCodec: 전역 리텐션 게이트 + 공유 최종 점수 투영 경쟁 퇴거 |
| `src/scheduler/nath_ddr_offloading.py` | A | NAtHDDROffloadingScheduler: 누적 어텐션 점수 EMA 기반 4-티어 DDR 오프로딩 최소-퇴거 스케줄러 |
| `src/scheduler/nath_retention_tier_decider.py` | A+C | NAtHRetentionTierDecider: 이중 신호(누적 어텐션 + 전역 리텐션) 4-티어 결정기 |
| `experiments/train_retention_gate.py` | C | 리텐션 게이트(W_r) + 공유 투영(W_final) 파인튜닝 스크립트 (보정 500~1000 샘플, < 0.5 GPU-hour) |
| `experiments/run_nath_attn_calibration.py` | A | 보정 데이터에서 누적 어텐션 점수 분포 측정 + tier_boundaries 자동 설정 스크립트 |
| `tests/unit/test_global_retention_gate_accuracy.py` | C | Activity C accuracy-preserving 검증 (필수) |
| `tests/unit/test_nath_ddr_offloading.py` | A | DDR 오프로딩 4-티어 분류·퇴거 비율 제약·EMA 추적 단위 테스트 |
| `tests/unit/test_nath_retention_tier_decider.py` | A+C | 이중 신호 결합·티어 경계 동적 조정·A+C 복합 효과 단위 테스트 |
| `tests/integration/test_cross_ac_nath_retention.py` | A+C | E2E 통합 테스트: 다중 요청 DDR 오프로딩 + 전역 퇴거 흐름 |
| `configs/experiments/2026-05-16.yaml` | 공통 | 이번 사이클 실험 설정 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| (없음) | `src/cache/base.py` 및 기존 모든 파일 변경하지 않음 |

---

## 알고리즘 상세

### GlobalRetentionGateEvictionCodec (Activity C)

"Make Each Token Count"(arXiv 2605.09649, Yale+CUHK)의 전역 리텐션 기반 KV 퇴거를 구현.
기구현 LaProxOutputAwareLayerEviction(레이어별 독립), LookaheadKVEvictionCodec(미래-인식 토큰별)과의
차이: 공유 최종 점수 투영으로 모든 레이어·헤드 KV가 단일 전역 예산에서 경쟁하는 새로운 퇴거 원칙.

```python
# src/cache/global_retention_gate_eviction.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class GlobalRetentionGateConfig:
    n_layers: int = 12            # 모델 레이어 수
    n_heads: int = 8              # 어텐션 헤드 수
    d_model: int = 512            # 모델 히든 차원 (= n_heads * d_head)
    budget_ratio: float = 0.3    # 전역 보존 비율 (0.3 = 30%만 보존, 70% 퇴거)
    recent_window: int = 32       # 최근 N 토큰은 항상 보존 (퇴거 예외)
    ensemble_ratio: float = 0.0   # LaProx 앙상블 비율 (0.0 = 순수 전역 리텐션)
    max_entries: int = 1000       # CacheStore용 최대 항목 수
    seed: int = 42


class RetentionGate(nn.Module):
    """경량 리텐션 게이트: 각 (layer, head)별 독립 선형 투영.

    r_{i,l,h} = sigmoid(W_r[l,h] · kv_{i,l,h})
    W_r[l,h] ∈ R^{d_model → 1} — 레이어·헤드별 독립 nn.Linear(d_model, 1)
    """

    def __init__(self, config: GlobalRetentionGateConfig) -> None:
        super().__init__()
        self.config = config
        # 각 (layer, head)별 독립 게이트: nn.ModuleList of nn.ModuleList
        self.gates = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(config.d_model, 1, bias=True)
                for _ in range(config.n_heads)
            ])
            for _ in range(config.n_layers)
        ])
        # 공유 최종 점수 투영: R^{n_layers × n_heads → 1}
        # 모든 레이어·헤드 스코어를 단일 전역 스코어로 결합
        self.final_proj = nn.Linear(config.n_layers * config.n_heads, 1, bias=False)

    def forward(
        self,
        kv_all_layers: torch.Tensor,  # [n_tokens, n_layers, n_heads, d_head_kv]
    ) -> torch.Tensor:
        """전역 리텐션 점수 계산.

        Algorithm:
          1. 각 (l, h)에 대해 r_{i,l,h} = sigmoid(W_r[l,h] · kv_{i,l,h})
          2. [n_tokens, n_layers * n_heads] 텐서로 concat
          3. W_final 선형 결합 → R_i ∈ R^{n_tokens}

        Returns:
            global_scores: Tensor[n_tokens] — 값이 클수록 전역적으로 중요한 토큰
        """
        n_tokens, n_layers, n_heads, d_head = kv_all_layers.shape
        gate_scores = []
        for l in range(n_layers):
            for h in range(n_heads):
                # kv_{i,l,h}: [n_tokens, d_head]
                kv_lh = kv_all_layers[:, l, h, :]  # [n_tokens, d_head]
                # d_model 차원으로 패딩/투영 (d_head != d_model 시)
                if d_head != self.config.d_model:
                    # 간단 선형 확장: repeat-interleave or linear upscale
                    repeat_factor = self.config.d_model // d_head
                    kv_lh = kv_lh.repeat(1, repeat_factor)[
                        :, :self.config.d_model
                    ]
                r_lh = self.gates[l][h](kv_lh).squeeze(-1)  # [n_tokens]
                gate_scores.append(torch.sigmoid(r_lh))
        # [n_tokens, n_layers * n_heads]
        gate_matrix = torch.stack(gate_scores, dim=1)
        # 전역 스코어: [n_tokens]
        global_scores = self.final_proj(gate_matrix).squeeze(-1)
        return global_scores


class GlobalRetentionGateEvictionCodec(CacheStore):
    """Cross-layer competitive eviction via global retention gate.

    Activity C: KV Cache Compression via global retention-based token eviction.
    보존 KV는 FP16 원본 — 양자화 왜곡 없음.
    전역 예산(budget_ratio)만큼 전체 레이어·헤드에서 상위 토큰만 보존한다.
    퇴거된 토큰은 모든 레이어·헤드에서 제거 (전역 일관성).

    CacheStore 인터페이스 완전 준수:
      put / get / evict / hit_rate / memory_bytes / reset_stats
    compression_hook() 오버라이드:
      put() 호출 전 전역 리텐션 게이트 점수로 토큰 퇴거 후 저장.
    """

    def __init__(self, config: GlobalRetentionGateConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._gate = RetentionGate(config)
        self._gate.eval()
        self._store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._total_tokens_original = 0
        self._total_tokens_kept = 0

    # ------------------------------------------------------------------ #
    # CacheStore 추상 메서드                                               #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store KV after global retention gate eviction via compression_hook().

        value shape: [n_tokens, n_layers, n_heads, d_head] (all-layer KV)
                  or [n_tokens, 2, n_heads, d_head] (single-layer K+V)
        """
        compressed = self.compression_hook(key, value)
        if len(self._store) >= self.config.max_entries:
            self.evict()
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = compressed

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return self._store[key]

    def evict(self) -> int:
        if not self._store:
            return 0
        oldest_key, kv = self._store.popitem(last=False)
        return kv.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(kv.nbytes for kv in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_tokens_original = 0
        self._total_tokens_kept = 0

    # ------------------------------------------------------------------ #
    # Activity C 핵심: compression_hook 오버라이드                         #
    # ------------------------------------------------------------------ #

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,  # [n_tokens, n_layers, n_heads, d_head] 또는
                              # [n_tokens, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """전역 리텐션 게이트 퇴거: budget_ratio 비율만 전역 상위 토큰 보존.

        Algorithm:
          1. value 형상에 따라 all-layer 또는 single-layer 경로 분기
          2. RetentionGate.forward() → global_scores [n_tokens]
             (single-layer: K 텐서로 근사 게이트 계산)
          3. recent_window 토큰 항상 보존 (scores += INF)
          4. 상위 ceil(n_tokens * budget_ratio)개 토큰 선택 → keep_mask
          5. value[keep_mask] 반환

        Returns:
            Tensor[kept_tokens, ...] — 퇴거 후 KV (budget_ratio 비율 보존)
        """
        ...

    def get_global_retention_score(
        self,
        token_ids: Optional[List[int]] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """NAtHRetentionTierDecider에서 조회하는 전역 리텐션 점수 인터페이스.

        Args:
            token_ids: 선택적 토큰 ID 목록 (캐시 조회용)
            kv: 직접 KV 텐서 제공 시 (토큰 ID 조회 불필요)

        Returns:
            global_scores: Tensor[n_tokens] — 전역 리텐션 점수
        """
        ...

    # ------------------------------------------------------------------ #
    # 퇴거 메트릭                                                           #
    # ------------------------------------------------------------------ #

    def eviction_rate(self) -> float:
        """실제 퇴거 비율 = (원본 토큰 - 보존 토큰) / 원본 토큰."""
        if self._total_tokens_original == 0:
            return 0.0
        return 1.0 - self._total_tokens_kept / self._total_tokens_original

    def memory_reduction_ratio(self) -> float:
        """베이스라인 FP16 대비 메모리 감소율 (budget_ratio=0.3 → 0.7 감소)."""
        return 1.0 - self.config.budget_ratio

    # ------------------------------------------------------------------ #
    # 파인튜닝 지원                                                         #
    # ------------------------------------------------------------------ #

    def train_retention_gate(
        self,
        calibration_data: List[torch.Tensor],  # List of [n_tokens, n_layers, n_heads, d_head]
        n_epochs: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """보정 데이터로 리텐션 게이트(W_r) + 공유 투영(W_final) 파인튜닝.

        LLM 가중치 동결, 리텐션 게이트 파라미터만 학습.
        손실: 다음-토큰 예측 어텐션 출력과 퇴거 후 어텐션 출력 간 MSE.
        훈련 비용: 500~1000 샘플, 3~5 에포크, < 0.5 GPU-hour.

        Returns: {"final_loss": float, "n_samples": int}
        """
        ...

    def save(self, path: str) -> None:
        torch.save({"gate_state_dict": self._gate.state_dict(), "config": self.config}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self._gate.load_state_dict(data["gate_state_dict"])
        self.config = data["config"]
```

---

### NAtHDDROffloadingScheduler (Activity A)

NAtH(arXiv 2605.09490)의 이론적 발견("정확도는 영구 퇴거 비율에만 의존")을 스케줄러 설계 원칙으로
적용. 기구현 CacheAwareScheduler(히트율 가중 우선순위)와의 차이: 메모리 압박 시 영구 퇴거 대신
DDR 오프로딩을 기본 정책으로 설정하고 누적 어텐션 점수 EMA로 4-티어 분류.

스케줄링 결정 단위: **요청(request) 단위** — 각 요청의 KV 토큰별 4-티어 분류.
캐시 상태 접근 방법: `_attn_score_ema` dict를 통해 토큰별 누적 어텐션 점수를 실시간 추적.

```python
# src/scheduler/nath_ddr_offloading.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import torch

from src.cache.base import CacheStore


@dataclass
class NAtHDDROffloadingConfig:
    # 4-티어 경계: [p1, p2, p3] 누적 어텐션 점수 백분위 기준
    # Tier 1 (HBM): >= p1 백분위, Tier 2 (DDR 프리페치): p1~p2,
    # Tier 3 (DDR INT8 압축): p2~p3, Tier 4 (영구 퇴거): < 1-p3
    tier_boundaries: List[float] = field(
        default_factory=lambda: [0.30, 0.70, 0.97]
    )  # [p1, p2, p3]: 기본 상위30%=HBM, 다음40%=DDR프리페치, 다음27%=DDR압축, 하위3%=퇴거
    max_eviction_ratio: float = 0.03   # 영구 퇴거 비율 상한 (3%)
    ema_alpha: float = 0.95            # 누적 어텐션 점수 EMA 감쇠 계수
    prefetch_chunk_size: int = 64      # DDR→HBM 비동기 프리페치 단위 (토큰 수)
    max_wait_ratio: float = 2.0        # 요청 최대 대기 시간 배수 (공정성)
    seed: int = 42


class NAtHDDROffloadingScheduler:
    """NAtH semantic-aware 4-tier DDR offloading minimal-eviction scheduler.

    Activity A: KV Cache-aware Scheduling
    스케줄링 결정 단위: 요청(request) 단위 — 각 요청의 KV 토큰별 4-티어 분류
    캐시 상태 접근: _attn_score_ema dict (토큰별 누적 어텐션 점수 EMA 추적)

    4-티어 메모리 정책:
      Tier 1 (HBM 즉시 접근): 상위 p1 백분위 — FP16 GPU HBM 유지
      Tier 2 (DDR 프리페치):  p1~p2 — CPU DDR 오프로딩 + 어텐션 전 비동기 프리페치
      Tier 3 (DDR 압축 보존): p2~p3 — CPU DDR INT8 압축 보존
      Tier 4 (영구 퇴거):     < 1-p3 — 영구 제거 (비율 <= max_eviction_ratio)

    Activity A 평가 기준 (evaluation_criteria.md §2):
      - 스케줄링 오버헤드: TTFT p50 증가 +5% 이내
      - 캐시 히트율 향상: 스케줄링 미적용 대비 +10%p
      - 요청 공정성: 최대 대기 시간 2× 초과 금지

    NAtH 이론 검증:
      - 영구 퇴거 비율 3% 이하 → GSM8K 91%+ 정확도 보존
      - DDR 오프로딩 토큰은 어텐션 계산 시 전체 정밀도(FP16) 복원 (제로 근사 오류)
    """

    def __init__(
        self,
        config: NAtHDDROffloadingConfig,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.config = config
        self._cache = cache
        # 토큰별 누적 어텐션 점수 EMA: {token_key: float}
        self._attn_score_ema: Dict[str, float] = {}
        # 티어 분류 결과: {token_key: int(1~4)}
        self._token_tier: Dict[str, int] = {}
        # DDR 오프로딩 버퍼 (CPU 텐서): {token_key: torch.Tensor}
        self._ddr_buffer_fp16: Dict[str, torch.Tensor] = {}  # Tier 2
        self._ddr_buffer_int8: Dict[str, torch.Tensor] = {}  # Tier 3
        # 스케줄링 오버헤드 측정
        self._scheduling_times: List[float] = []
        self._permanent_evictions = 0
        self._total_decisions = 0

    def update_attention_score(
        self,
        token_key: str,
        attn_score: float,
    ) -> None:
        """매 디코딩 스텝에서 토큰별 누적 어텐션 점수 EMA 갱신.

        Algorithm:
          new_score = alpha * old_score + (1 - alpha) * attn_score
          오래된 토큰의 점수가 자연적으로 감소 → 시간적 중요도 반영.
        """
        alpha = self.config.ema_alpha
        old = self._attn_score_ema.get(token_key, 0.0)
        self._attn_score_ema[token_key] = alpha * old + (1.0 - alpha) * attn_score

    def classify_tokens(
        self,
        token_keys: List[str],
    ) -> Dict[str, int]:
        """누적 어텐션 점수로 4-티어 분류.

        Algorithm:
          1. 각 토큰의 EMA 점수 수집
          2. 백분위 경계(p1, p2, p3) 계산
          3. 점수에 따라 Tier 1/2/3/4 할당
          4. max_eviction_ratio 초과 시 Tier 3→4 경계 자동 상향

        Returns:
            {token_key: tier(1~4)}
        """
        ...

    def offload_to_ddr(
        self,
        token_key: str,
        kv_tensor: torch.Tensor,  # GPU HBM 텐서
        tier: int,               # 2 또는 3
    ) -> None:
        """KV 텐서를 DDR로 비동기 오프로딩.

        Tier 2: FP16 그대로 CPU DDR로 전송 (torch.Tensor.cpu() 비동기)
        Tier 3: INT8 양자화 후 CPU DDR로 전송 (메모리 2× 추가 절감)
        제로 근사 오류 보장: Tier 2 복원 시 수치적으로 FP16 원본과 동일.
        """
        ...

    def prefetch_from_ddr(
        self,
        token_keys: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Tier 2 DDR 버퍼에서 GPU HBM으로 비동기 프리페치.

        어텐션 계산 전 호출하여 Tier 2 토큰이 GPU에서 사용 가능하도록 준비.
        FP16 전체 정밀도로 복원 (제로 근사 오류).

        Returns:
            {token_key: kv_tensor(GPU)} — 프리페치 완료된 토큰 KV
        """
        ...

    def schedule_request(
        self,
        request: Dict,  # {"id": str, "token_ids": List[int], "arrival_time": float}
    ) -> Dict:
        """단일 요청의 KV 토큰 4-티어 분류 수행 후 스케줄링 메타데이터 반환.

        Returns:
            request에 "tier_assignment", "ddr_offload_keys", "evict_keys" 키 추가
        """
        t0 = time.monotonic()
        ...
        overhead_ms = (time.monotonic() - t0) * 1000
        self._scheduling_times.append(overhead_ms)
        return request

    def scheduling_overhead_ms_p50(self) -> float:
        if not self._scheduling_times:
            return 0.0
        sorted_times = sorted(self._scheduling_times)
        return sorted_times[len(sorted_times) // 2]

    def permanent_eviction_ratio(self) -> float:
        """영구 퇴거(Tier 4) 비율 = 영구 퇴거 건수 / 전체 분류 결정 수."""
        if self._total_decisions == 0:
            return 0.0
        return self._permanent_evictions / self._total_decisions

    def cache_hit_rate(self) -> float:
        """DDR 오프로딩 포함 전체 캐시 히트율 (Tier 1+2+3 비율)."""
        if self._total_decisions == 0:
            return 0.0
        return 1.0 - self.permanent_eviction_ratio()
```

---

### NAtHRetentionTierDecider (Cross A+C)

A-2 NAtHDDROffloadingScheduler의 누적 어텐션 점수와 C-2 GlobalRetentionGateEvictionCodec의
전역 리텐션 점수를 이중 신호로 결합해 4-티어 분류 정밀도를 높이는 플러그인 모듈.

```python
# src/scheduler/nath_retention_tier_decider.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

from src.scheduler.nath_ddr_offloading import NAtHDDROffloadingScheduler, NAtHDDROffloadingConfig
from src.cache.global_retention_gate_eviction import GlobalRetentionGateEvictionCodec


@dataclass
class NAtHRetentionTierDeciderConfig:
    # 이중 신호 결합 가중치: final_score = alpha * norm(attn) + (1-alpha) * retention
    alpha: float = 0.5             # 누적 어텐션 점수 가중치
    max_eviction_ratio: float = 0.03  # 영구 퇴거 비율 상한 (3%)
    tier_boundaries: List[float] = None  # None이면 NAtHDDROffloadingConfig 기본값 사용
    seed: int = 42


class NAtHRetentionTierDecider:
    """Dual-signal 4-tier decider: NAtH cumulative attention + global retention gate.

    Cross Activity A+C: A-2 NAtHDDROffloadingScheduler + C-2 GlobalRetentionGateEvictionCodec
    이중 신호 결합:
      final_score_i = α × norm(cumulative_attn_score_i) + (1-α) × global_retention_score_i

    티어 경계 동적 조정:
      - 영구 퇴거 비율 > max_eviction_ratio 감지 시 Tier 3→4 경계 자동 상향
      - HBM 압박 감지 시 Tier 1→2 경계 자동 하향 (DDR 오프로딩 비율 자동 증가)

    A+C 복합 효과 측정 (목표 9):
      단독 A-2 / 단독 C-2 / 결합 Cross-1 의 처리량·메모리·정확도 3방향 비교.
    """

    def __init__(
        self,
        config: NAtHRetentionTierDeciderConfig,
        nath_scheduler: NAtHDDROffloadingScheduler,
        retention_codec: GlobalRetentionGateEvictionCodec,
    ) -> None:
        self.config = config
        self._nath = nath_scheduler
        self._retention = retention_codec
        self._tier_stats: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    def decide_tier(
        self,
        token_keys: List[str],
        kv_tensor: Optional[torch.Tensor] = None,  # [n_tokens, n_layers, n_heads, d_head]
    ) -> Dict[str, int]:
        """이중 신호로 토큰별 최종 4-티어 결정.

        Algorithm:
          1. NAtHDDROffloadingScheduler._attn_score_ema에서 누적 어텐션 점수 조회
          2. GlobalRetentionGateEvictionCodec.get_global_retention_score() 조회
          3. 두 점수를 L2 정규화 후 alpha 가중 결합 → final_scores [n_tokens]
          4. 분위수 기반 4-티어 할당
          5. 영구 퇴거 비율 초과 시 경계 자동 조정

        Returns:
            {token_key: tier(1~4)}
        """
        ...

    def adjust_tier_boundaries(
        self,
        current_eviction_ratio: float,
        hbm_pressure: float = 0.0,  # 0.0~1.0, HBM 사용률
    ) -> None:
        """영구 퇴거 비율 초과 또는 HBM 압박 시 티어 경계 자동 조정.

        - current_eviction_ratio > max_eviction_ratio:
            Tier 3→4 경계를 높여 Tier 4 비율 감소
        - hbm_pressure > 0.8:
            Tier 1→2 경계를 낮춰 DDR 오프로딩 비율 증가
        """
        ...

    def tier_distribution(self) -> Dict[int, float]:
        """현재까지의 티어별 토큰 비율 통계."""
        total = sum(self._tier_stats.values())
        if total == 0:
            return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        return {k: v / total for k, v in self._tier_stats.items()}
```

---

### 파인튜닝 스크립트 (Activity C)

```python
# experiments/train_retention_gate.py

"""
GlobalRetentionGateEvictionCodec의 리텐션 게이트(W_r) + 공유 투영(W_final) 파인튜닝.
LLM 가중치 동결, 게이트 파라미터만 학습.
훈련 비용: 500~1000 샘플, 3~5 에포크, < 0.5 GPU-hour.

목표 손실: MSE(attention_output_after_eviction, attention_output_before_eviction)
-- 퇴거 후 어텐션 출력이 원본과 최대한 일치하도록 게이트 학습.

Usage:
  python experiments/train_retention_gate.py \\
    --n_samples 500 --n_epochs 5 --lr 1e-3 \\
    --budget_ratio 0.3 --n_layers 12 --n_heads 8 --d_model 512 \\
    --output configs/retention_gate_weights.pt --seed 42
"""

def train(
    n_samples: int = 500,
    n_epochs: int = 5,
    lr: float = 1e-3,
    n_layers: int = 12,
    n_heads: int = 8,
    d_model: int = 512,
    budget_ratio: float = 0.3,
    output_path: str = "configs/retention_gate_weights.pt",
    seed: int = 42,
) -> Dict[str, float]:
    """Returns {"final_loss": float, "n_samples": int, "training_time_sec": float}"""
    ...
```

---

### NAtH 보정 스크립트 (Activity A)

```python
# experiments/run_nath_attn_calibration.py

"""
NAtHDDROffloadingScheduler용 누적 어텐션 점수 분포 보정 스크립트.
보정 데이터 100~200 시퀀스에서 각 토큰의 누적 어텐션 점수를 측정하고,
tier_boundaries [p1, p2, p3]를 자동 설정한다.

출력: configs/nath_tier_boundaries.yaml

Usage:
  python experiments/run_nath_attn_calibration.py \\
    --n_sequences 100 --n_layers 12 --n_heads 8 \\
    --max_eviction_ratio 0.03 \\
    --output configs/nath_tier_boundaries.yaml --seed 42
"""

def run_calibration(
    n_sequences: int = 100,
    n_layers: int = 12,
    n_heads: int = 8,
    max_seq_len: int = 512,
    max_eviction_ratio: float = 0.03,
    output_path: str = "configs/nath_tier_boundaries.yaml",
    seed: int = 42,
) -> Dict[str, float]:
    """Returns {"p1": float, "p2": float, "p3": float}"""
    ...
```

---

## Activity C — Accuracy Preservation 검증 계획

Activity C(GlobalRetentionGateEvictionCodec)를 포함하므로 반드시 작성한다.

### perplexity 측정

- **데이터셋**: WikiText-2 proxy (실 데이터셋 없을 경우 synthetic 토큰 시퀀스로 대체)
- **측정 방법**: `src/metrics/perplexity.py`의 `attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)` < 0.01 (1%)
  - `k_kept`, `v_kept`: 전역 리텐션 게이트 퇴거 후 budget_ratio 비율로 남은 토큰의 K, V
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)
- **전역 예산별 측정**: budget_ratio = 0.3 / 0.5 / 0.7 각각 독립 측정

### 태스크 정확도 측정

- **벤치마크**: LongBench 8개 서브태스크 proxy (KL divergence, cosine similarity)
- **측정 방법**: `attention_kl_divergence(q, k_orig, k_kept)` < 0.015,
  `cosine_similarity_output(q, k_orig, v_orig, k_kept, v_kept)` >= 0.99
- **허용 오차**: ±1% 이내 (evaluation_criteria.md §4 필수)
- **특별 측정**: 긴 컨텍스트(n_tokens > 512) 구간에서 "전체 캐시 초과 성능" 여부 확인
  (cosine이 budget_ratio=1.0 full cache 대비 >= 1.0 인 경우 노이즈 제거 효과로 간주)

### 전역 예산 — 정확도 곡선

| 전역 예산 (budget_ratio) | 목표 메모리 감소 | attention error 한계 | cosine 최소값 |
|------------------------|----------------|---------------------|--------------|
| 0.7 (30% 퇴거)  | −30% | < 0.003 (0.3%) | >= 0.998 |
| 0.5 (50% 퇴거)  | −50% | < 0.007 (0.7%) | >= 0.993 |
| 0.3 (70% 퇴거)  | −70% | < 0.01  (1.0%) | >= 0.99  |
| 0.2 (80% 퇴거)  | −80% | < 0.02  (2.0%) | >= 0.98 (경고, non-mandatory) |
| 0.1 (90% 퇴거)  | −90% | < 0.05  (5.0%) | >= 0.95 (경고, non-mandatory) |

### LaProx·LookaheadKV와의 동일 설정 비교 (3방향 비교, 필수)

- budget_ratio=0.3 (70% 퇴거) 설정에서 동일 테스트 데이터로:
  1. GlobalRetentionGateEvictionCodec (이번 사이클 C-2)
  2. LaProxOutputAwareLayerEviction (기구현, 레이어별 독립)
  3. LookaheadKVEvictionCodec (기구현, 미래-인식 토큰별)
- 비교 지표: attention error, KL divergence, cosine similarity, memory_reduction_ratio
- GlobalRetentionGateEvictionCodec cosine >= LookaheadKV cosine - 0.005 조건을 테스트로 명시
- `test_global_vs_laprox_vs_lookahead_comparison` 테스트 케이스에 포함

### 검증 테스트 파일

`tests/unit/test_global_retention_gate_accuracy.py`

**테스트 케이스 목록**:

```python
# tests/unit/test_global_retention_gate_accuracy.py

"""Activity C — GlobalRetentionGateEvictionCodec accuracy-preserving verification.

Mandatory per evaluation_criteria.md §4:
  - perplexity change ±1% (proxied by attention output relative error < 0.01)
  - downstream task accuracy ±1% (KL < 0.015, cosine >= 0.99)
  - budget_ratio = 0.3 / 0.5 / 0.7 각각 측정
  - LaProx·LookaheadKV 동일 설정 3방향 비교
  - 긴 컨텍스트(n_tokens=512) "전체 캐시 초과 성능" 구간 측정
All tests use synthetic data (no real model API calls).
Seed 42 고정으로 재현성 보장.
"""

import pytest
import torch
from src.cache.global_retention_gate_eviction import (
    GlobalRetentionGateEvictionCodec,
    GlobalRetentionGateConfig,
)
from src.metrics.perplexity import (
    attention_output_relative_error,
    attention_kl_divergence,
    cosine_similarity_output,
)

SEED = 42
N_LAYERS = 4
N_HEADS = 4
D_HEAD = 64
D_MODEL = N_HEADS * D_HEAD  # 256
N_TOKENS = 64
N_TOKENS_LONG = 512  # 긴 컨텍스트 테스트용

# ── Fixtures ──────────────────────────────────────────────────────────── #

# test_budget_70pct_attention_error: budget_ratio=0.7 (30% 퇴거), error < 0.003 (MANDATORY ±1%)
# test_budget_50pct_attention_error: budget_ratio=0.5 (50% 퇴거), error < 0.007 (MANDATORY ±1%)
# test_budget_30pct_attention_error: budget_ratio=0.3 (70% 퇴거), error < 0.01  (MANDATORY ±1%)
# test_budget_20pct_attention_error: budget_ratio=0.2 (80% 퇴거), error < 0.02  (경고 only)
# test_kl_divergence_budget_30pct: KL < 0.015 at budget_ratio=0.3 (LongBench proxy, MANDATORY)
# test_cosine_similarity_budget_30pct: cosine >= 0.99 at budget_ratio=0.3 (MANDATORY)
# test_recent_window_preserved: recent_window=32 토큰 항상 보존
# test_eviction_rate_matches_budget: 실제 퇴거율 = 1 - budget_ratio (±5%p)
# test_memory_reduction_30pct: memory_reduction_ratio() >= 0.30
# test_cachestore_interface: put/get/evict/hit_rate/memory_bytes/reset_stats 동작
# test_global_vs_laprox_vs_lookahead_comparison: 3방향 비교 (budget_ratio=0.3, 동일 설정)
#   - GlobalRetentionGate cosine >= LookaheadKV cosine - 0.005
# test_long_context_noise_reduction: n_tokens=512, budget_ratio=0.3
#   - "전체 캐시 초과 성능" 측정: 선택적 퇴거로 노이즈 제거 효과 존재 여부 확인
# test_multilayer_consistency: N_LAYERS 각 레이어에서 전역 일관성 (error < 1%)
# test_get_global_retention_score_interface: NAtHRetentionTierDecider 연동 인터페이스 동작
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-16.yaml
experiment:
  date: "2026-05-16"
  activity: "A+C"
  description: >
    C-2 GlobalRetentionGateEvictionCodec (전역 크로스-레이어 경쟁 퇴거 코덱) +
    A-2 NAtHDDROffloadingScheduler (의미-인식 4-티어 DDR 오프로딩 최소-퇴거 스케줄러) +
    Cross-1 NAtHRetentionTierDecider (이중 신호 A+C 4-티어 결정기)
  cache_type: global_retention_gate
  compression_method: eviction
  scheduler_type: nath_ddr_offloading

global_retention_gate_eviction:
  n_layers: 12
  n_heads: 8
  d_model: 512           # n_heads * d_head (8 * 64)
  budget_ratio: 0.3      # 기본 전역 보존 비율 (30%만 보존, 70% 퇴거)
                         # sweep: 0.1 / 0.2 / 0.3 / 0.5 / 0.7
  recent_window: 32      # 최근 N 토큰 항상 보존
  ensemble_ratio: 0.0    # LaProx 앙상블 비율 (0.0=순수 전역 리텐션)
  max_entries: 1000
  seed: 42

nath_ddr_offloading:
  tier_boundaries: [0.30, 0.70, 0.97]  # [p1, p2, p3] 백분위 경계
  max_eviction_ratio: 0.03              # 영구 퇴거 비율 상한 (3%)
  ema_alpha: 0.95                       # 누적 어텐션 점수 EMA 감쇠 계수
  prefetch_chunk_size: 64              # DDR→HBM 비동기 프리페치 단위
  max_wait_ratio: 2.0                   # 요청 최대 대기 시간 배수
  seed: 42

nath_retention_tier_decider:
  alpha: 0.5            # 이중 신호 결합 가중치 (0.5=균등)
  max_eviction_ratio: 0.03
  seed: 42

benchmark:
  accuracy:
    method: "attention_output_proxy"
    proxy_tolerance: 0.01           # 1% attention output error limit
    kl_tolerance: 0.015
    cosine_min: 0.99
    perplexity_dataset: "wikitext-2"
    perplexity_tolerance_pct: 1.0
    task_accuracy_tolerance_pct: 1.0
  budget_sweep:
    ratios: [0.1, 0.2, 0.3, 0.5, 0.7]  # 전역 예산 비율 sweep
  nath_eviction:
    max_eviction_ratio: 0.03       # 영구 퇴거 비율 상한 검증
    eviction_sweep_ratios: [0.01, 0.03, 0.05, 0.10]  # GSM8K 정확도 곡선용
  memory_reduction:
    target_ratio: 0.30             # 최소: −30%
    target_ratio_goal: 0.70        # 목표: −70% (budget_ratio=0.3)
  throughput:
    target_improvement_pct: 20    # +20% 이상
  effective_context:
    target_multiplier: 2.0        # 2× 이상
  scheduling:
    ttft_overhead_limit_pct: 5.0  # TTFT p50 +5% 이내
  comparison:
    methods: ["global_retention_gate", "laprox", "lookahead_kv"]
    budget_ratio: 0.3
    cosine_tolerance: 0.005       # GlobalRetentionGate >= LookaheadKV - 0.005
  long_context:
    n_tokens: 512                 # 긴 컨텍스트 테스트 토큰 수
    check_noise_reduction: true   # "전체 캐시 초과 성능" 측정

retention_gate_training:
  n_samples: 500
  n_epochs: 5
  lr: 1.0e-3
  budget_ratio: 0.3
  output: "configs/retention_gate_weights.pt"

nath_calibration:
  n_sequences: 100
  max_eviction_ratio: 0.03
  output: "configs/nath_tier_boundaries.yaml"

seed: 42
results_dir: "results/2026-05-16"
```

---

## 테스트 요구사항

- [x] `tests/unit/test_global_retention_gate_accuracy.py` — Activity C 필수 accuracy 검증 (14개 테스트, 위 목록 참조)
- [ ] `tests/unit/test_nath_ddr_offloading.py` — Activity A DDR 오프로딩·4-티어 분류·퇴거 비율 제약·EMA 추적 단위 테스트
- [ ] `tests/unit/test_nath_retention_tier_decider.py` — Cross A+C 이중 신호 결합·티어 경계 동적 조정·3방향 비교 단위 테스트
- [ ] `tests/integration/test_cross_ac_nath_retention.py` — E2E 통합: 다중 요청 DDR 오프로딩 + 전역 퇴거 흐름

### 단위 테스트 최소 요구 사항 (test_nath_ddr_offloading.py)

```
- test_ema_update_convergence: EMA 갱신이 올바른 지수 감쇠로 수렴하는지 확인
- test_tier_classification_4tiers: 4-티어 분류 결과가 Tier 1~4 범위 내인지 확인
- test_max_eviction_ratio_enforced: Tier 4 비율이 max_eviction_ratio(3%) 미만인지 확인
- test_ddr_offload_tier2_fp16: Tier 2 DDR 오프로딩이 FP16 원본 정밀도 유지하는지 확인
- test_ddr_offload_tier3_int8: Tier 3 DDR 오프로딩이 INT8 압축 후 저장되는지 확인
- test_prefetch_zero_approximation_error: Tier 2 프리페치 후 어텐션 출력이 원본과 동일한지 확인
- test_scheduling_overhead_under_5pct: scheduling_overhead_ms_p50 < 5ms
- test_fairness_max_wait: 최대 대기 요청이 max_wait_ratio 초과하지 않음
- test_permanent_eviction_ratio_metric: permanent_eviction_ratio() 정확성
```

### 단위 테스트 최소 요구 사항 (test_nath_retention_tier_decider.py)

```
- test_dual_signal_combination: alpha 가중치로 이중 신호가 올바르게 결합되는지 확인
- test_tier_boundary_auto_adjust_eviction: 퇴거 비율 초과 시 경계 자동 상향 확인
- test_tier_boundary_auto_adjust_hbm: HBM 압박 시 경계 자동 하향 확인
- test_eviction_ratio_below_3pct: 이중 신호 결합 후에도 영구 퇴거 3% 이하 유지
- test_triple_comparison_throughput: 단독 A-2 / 단독 C-2 / 결합 Cross-1 처리량 비교
- test_triple_comparison_memory: 단독 A-2 / 단독 C-2 / 결합 Cross-1 메모리 비교
- test_triple_comparison_accuracy: 단독 A-2 / 단독 C-2 / 결합 Cross-1 정확도 비교
- test_get_global_retention_score_called: 리텐션 코덱 get_global_retention_score 연동 확인
```

---

## 완료 기준 (Definition of Done)

- [ ] 단위 테스트 전부 통과 (신규 4개 파일 + 기존 회귀 없음)
- [ ] `evaluation_criteria.md` §4 Activity C 필수 항목 충족:
      - perplexity 변화 ±1% 이내 (attention error < 0.01, budget_ratio 0.3/0.5/0.7 각각)
      - downstream 태스크 정확도 ±1% 이내 (KL < 0.015, cosine >= 0.99)
      - LaProx·LookaheadKV와 동일 설정 3방향 비교 포함
      - "전체 캐시 초과 성능" 구간(n_tokens=512) 측정 포함
- [ ] `evaluation_criteria.md` §2 Activity A 항목 충족:
      - 스케줄링 오버헤드 TTFT p50 +5% 이내
      - 영구 퇴거(Tier 4) 비율 <= 3% (NAtH 이론 검증)
      - 캐시 히트율 향상 +10%p (DDR 오프로딩 포함 전체 비휘발성 KV 보존율)
- [ ] `evaluation_criteria.md` §5 크로스 조합 C 포함: 복합 적용 후 accuracy ±1% 이내
      - NAtHRetentionTierDecider 이중 신호 결합 후 측정
      - 단독 A-2 / 단독 C-2 / 결합 Cross-1 3방향 비교 수치 확인
- [ ] `evaluation_criteria.md` §0 공통 필수:
      - CacheStore 인터페이스 모든 추상 메서드 구현 (GlobalRetentionGateEvictionCodec)
      - 시드 42 고정 재현성
      - `configs/experiments/2026-05-16.yaml` 존재
      - 모든 공개 함수·메서드 타입 힌트
- [ ] 목표 지표 수치 `results/2026-05-16/metrics.json`에 JSON 기록:
      ```json
      {
        "inference_throughput_improvement_pct": ...,
        "kv_memory_reduction_ratio_budget_70pct": ...,
        "kv_memory_reduction_ratio_budget_50pct": ...,
        "kv_memory_reduction_ratio_budget_30pct": ...,
        "compression_accuracy_delta_budget_70pct": ...,
        "compression_accuracy_delta_budget_50pct": ...,
        "compression_accuracy_delta_budget_30pct": ...,
        "effective_context_length_multiplier": ...,
        "eviction_overhead_ttft_pct": ...,
        "scheduling_overhead_ttft_p50_pct": ...,
        "permanent_eviction_ratio": ...,
        "ddr_offloading_ratio": ...,
        "global_retention_cosine_budget30": ...,
        "laprox_cosine_budget30": ...,
        "lookahead_kv_cosine_budget30": ...,
        "long_context_noise_reduction_cosine": ...,
        "cross_ac_throughput_vs_solo_a2": ...,
        "cross_ac_throughput_vs_solo_c2": ...
      }
      ```
- [ ] `src/cache/base.py` CacheStore 인터페이스 깨지지 않음 (수정 없음)
- [ ] 기존 모든 단위·통합 테스트 회귀 없이 통과
- [ ] `configs/retention_gate_weights.pt` 생성 (`experiments/train_retention_gate.py` 실행 후)
- [ ] `configs/nath_tier_boundaries.yaml` 생성 (`experiments/run_nath_attn_calibration.py` 실행 후)
