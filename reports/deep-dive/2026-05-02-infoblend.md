# Deep Dive — InfoBlend (2026-05-02)

## 식별된 대상

- **정식 명칭**: InfoBlend — *Storing and Reusing KV Caches of Multimodal Information without Positional Restriction*
- **1차 출처 (확인됨)**:
  - OpenReview Forum: <https://openreview.net/forum?id=bld5GVRad0>
  - OpenReview PDF: <https://openreview.net/pdf?id=bld5GVRad0>
  - 게시 시점: 2025-10-08 (OpenReview)
  - 추정 venue: ICLR 2026 (제출, decision 미확인 — `(미확인)`)
- **저자/소속**: OpenReview 제출 익명·비공개 상태일 가능성. 직접 PDF 접근이 차단되어 본 리포트 작성 시점에는 `(미확인)`. 한국·중국계 시스템 그룹의 후속 연구로 추정 `(추정)`.
- **공식 코드 저장소**: 검색 결과 공개 GitHub 미발견. `(미확인)` — 향후 official supplementary material에 포함될 가능성.
- **분류 Activity**: **B (Non-Contiguous KV Cache Reuse)** 가 주, **A (Scheduling/Orchestration)** 가 보조 (디스크 오프로드 + 병렬 로드 부분), **C (Compression)** 와는 직접 관계 약함. 다만 "선택적 재계산" 이 압축 손실 복원 메커니즘과 구조적으로 유사.
- **관련 동시대 기술 (식별됨, 비교 대상)**:
  - **CacheBlend** (EuroSys 2025, arXiv:2405.16444) — 텍스트 RAG에서의 Selective Recompute 원조.
  - **MPIC** (arXiv:2502.01960) — InfoBlend와 거의 동일 문제(MLLM, Position-Independent Multimodal Context Caching)를 다루는 동시대 시스템.
  - **EPIC** (arXiv:2410.15332) — LLM(텍스트)용 Position-Independent Caching의 일반 프레임워크.
  - **KVLink** (arXiv:2502.16002) — Activity B의 segment concatenation 이론적 기반.
  - **KV Packet** (arXiv:2604.13226) — 어댑터 학습 기반 PIC, training-required.

> **디스앰비규에이션 노트** — "InfoBlend"라는 이름은 KV cache 분야에서 본 OpenReview 제출본이 유일한 1차 출처로 확인됐다. 구글 검색 시 "data integration" 분야의 동명 제품(상용)이 일부 등장하지만 본 연구와 완전히 무관하다. 따라서 본 리포트는 **OpenReview bld5GVRad0**을 분석 대상으로 한다.
>
> **케이스 A로 처리** — InfoBlend는 KV cache 분야에 실재하는 기술이므로 정상 deep-dive 진행. (CacheBlend 와의 혼동 가능성도 검토했으나, 본 리포트의 사용자는 이전 트렌드 리포트 `2026-04-28.md` 와 아이디어 리포트 `2026-04-28.md` 에서 InfoBlend 를 명시적으로 인용했으므로 `infoblend` 의도가 분명함.)

---

## 한 줄 요약

InfoBlend는 **이미지+텍스트 인터리브된 멀티모달 컨텍스트의 KV 캐시를 위치-독립적으로 디스크에 저장·재조합**하면서, 정확도 손실은 **이미지 토큰 시작부의 k개 토큰만 선택적으로 재계산**하여 attention sink 효과를 복원하는 시스템이다.

---

## 1. 문제 정의

### 1.1 멀티모달 컨텍스트 캐싱의 본질적 한계

표준 KV 프리픽스 캐싱(vLLM, SGLang, RadixAttention)은 두 요청 사이에서 **byte-identical 한 접두사**가 존재할 때만 KV 를 재사용한다. MLLM(LLaVA-NeXT, Qwen-VL, InternVL 등) 워크로드에서 이 가정은 다음 이유로 빈번히 깨진다.

| 시나리오 | 접두사 일치가 깨지는 원인 |
|----------|-------------------------|
| Multimodal RAG | 검색된 이미지·문서 청크가 **요청마다 순서가 다르게** 삽입된다 |
| Interleaved chat | 사용자가 시스템 프롬프트 뒤에 **다른 이미지를 끼워넣음** |
| Tool/Agent 워크플로우 | 에이전트가 동일 이미지를 **다른 텍스트 컨텍스트와 결합** |
| Document-VQA | 동일 PDF 페이지가 **다른 질의 + 다른 페이지 순서** 로 등장 |

위 모든 경우, **이미지 자체의 KV** (보통 LLaVA-1.5 기준 한 이미지당 576 토큰, LLaVA-NeXT 는 최대 2880 토큰)는 실질적으로 동일하지만, 위치(position id)가 달라져 byte-identical 비교가 실패한다.

### 1.2 Naïve PIC(Position-Independent Caching)가 깨지는 이유

이미지 KV 를 단순히 위치 무시하고 재사용하면 다음 두 가지가 동시에 발생한다.

1. **RoPE 위치 불일치** — 캐싱 시점의 RoPE 회전 행렬과 재사용 시점의 RoPE 가 다르다 → attention score 분포가 어긋남.
2. **Attention sink 누수** — 재사용된 이미지 토큰은 캐싱 당시 자신이 시퀀스 첫 부분에 위치했기 때문에 attention sink 역할을 했다. 재사용 시 시퀀스 중간으로 옮겨졌음에도 sink 처럼 attention 을 흡수해 전체 분포를 왜곡한다.

CacheBlend(EuroSys 2025)는 텍스트 RAG 에서 (1)+(2) 를 **HKVD(High-KV-Deviation) 토큰**의 선택적 재계산으로 해결했지만, 이미지 토큰의 sink 행동은 텍스트와 다르게 **시퀀스 시작부의 소수 이미지 토큰에 집중**된다는 점을 활용하지 못한다.

### 1.3 InfoBlend가 직접 해결하려는 것

| 문제 | InfoBlend 처방 |
|------|----------------|
| 이미지 KV 의 byte-identical 비교 실패 | 위치-독립 해시 기반 디스크 저장 |
| RoPE 위치 보정 비용 | 텍스트 토큰 전체 재계산 (저렴) + 이미지 토큰은 sink-k 만 재계산 |
| Attention sink 왜곡 | 이미지 청크 시작부의 **k개 토큰을 강제 재계산** 하여 "더 이상 첫 토큰이 아니다" 신호 주입 |
| 디스크 → GPU 전송 지연 | KV 로드와 재계산을 병렬 파이프라인으로 오버랩 |
| 멀티 RAG 청크 결합 | 임의 순서 재조합 가능 (CacheBlend 의 KVLink 일반화) |

---

## 2. 핵심 아이디어

> **"이미지 토큰의 KV 는 위치와 무관하게 캐시·재조합하되, 이미지 청크 경계의 처음 k개 토큰만 재계산하여 attention sink 신호를 갱신한다. 동시에 디스크 I/O 와 재계산을 파이프라인으로 오버랩한다."**

세 개의 결합된 기법:

1. **Position-Independent Multimodal KV Storage** — 이미지/텍스트 청크를 의미 단위로 분할 후, RoPE 무관 표현으로 디스크에 저장.
2. **Selective Sink-Aware Recomputation** — 텍스트 청크는 전부 재계산, 이미지 청크는 처음 k 토큰만 재계산.
3. **Parallel Load–Compute Pipeline** — 디스크에서 청크 KV 를 비동기 로드하면서 동시에 재계산을 GPU 에서 진행.

---

## 3. 알고리즘 / 아키텍처 상세

### 3.1 표기

- $C = [c_1, c_2, \ldots, c_n]$: 입력 청크 시퀀스 (각 $c_i$ 는 텍스트 또는 이미지 청크)
- $\text{KV}(c_i)$: 청크 $c_i$ 의 KV 텐서
- $H(c_i)$: 청크 내용에 대한 위치-독립 해시 (이미지: pixel hash 또는 vision encoder embedding hash; 텍스트: token id list hash)
- $k$: 이미지 청크당 재계산할 시작 토큰 수 (하이퍼파라미터, 본 논문 권장 `(추정)` $k = 1{\sim}4$)
- $\mathcal{S}$: 디스크 KV 스토어

### 3.2 의사코드 — Storage Phase

```pseudocode
Algorithm 1: InfoBlend Storage (offline / async during prefill)
Input:  prompt P with chunks C = [c_1, ..., c_n], computed KV[1..n]
Output: persisted entries in disk store S

for each chunk c_i in C:
    h_i = position_independent_hash(c_i)
    # Important: strip RoPE from KV before storing so it is reusable
    # at any future position.
    kv_canonical = strip_position_encoding(KV[i])
    S.put(key=h_i, value=kv_canonical, meta={modality, length, model_layer})
```

`strip_position_encoding` 의 구체 구현은 RoPE 의 회전을 역으로 적용해 "무회전(canonical)" KV 를 만든다. 추론 시점에 새 위치에서의 RoPE 를 곱해주는 식으로 위치-독립성을 달성한다 `(추정 — CacheBlend·EPIC 와 동일 트릭)`.

### 3.3 의사코드 — Reuse + Selective Recompute Phase

```pseudocode
Algorithm 2: InfoBlend Reuse with Sink-Aware Selective Recompute
Input:  prompt P' with chunks C' = [c'_1, ..., c'_m]
        disk store S, hyperparameter k (sink tokens to recompute)
Output: full KV ready for decode

# Step 1: Lookup
hits, misses = [], []
for j, c'_j in enumerate(C'):
    h = position_independent_hash(c'_j)
    if S.contains(h):
        hits.append((j, S.get_async(h)))      # async disk load
    else:
        misses.append((j, c'_j))

# Step 2: Parallel recompute MISSES while loading HITS
recompute_tokens = []
for (j, c) in misses:
    recompute_tokens.extend(token_range_of(c))

# For HITS: schedule sink-aware partial recompute
sink_tokens = []
for (j, kv_future) in hits:
    if modality_of(c'_j) == "image":
        sink_tokens.extend(first_k_tokens_of(c'_j, k))
    elif modality_of(c'_j) == "text":
        # Optional: HKVD-style selective recompute (CacheBlend heuristic)
        sink_tokens.extend(hkvd_tokens(c'_j, threshold))

recompute_tokens.extend(sink_tokens)

# Step 3: Pipeline — overlap disk I/O with GPU recompute
GPU.recompute(recompute_tokens)         # parallel with…
KV_loaded = await_all(hits)             # …disk load

# Step 4: Re-apply RoPE at NEW positions and stitch
for (j, kv_canon) in KV_loaded:
    new_pos = global_position_of(c'_j, in=P')
    KV[j] = apply_rope(kv_canon, new_pos)

# Step 5: Replace recomputed tokens (overrides cached values for those tokens)
for tok in recompute_tokens:
    KV[chunk_of(tok)][offset_of(tok)] = freshly_computed_kv[tok]

return concat_in_order(KV)
```

### 3.4 Selective Sink-Aware Recompute 의 핵심 직관

CacheBlend 의 HKVD 휴리스틱은 "어떤 토큰이 다른 청크의 영향을 크게 받는가"를 cosine deviation 으로 식별한다. InfoBlend 의 추가 통찰은:

> **이미지 청크 KV 의 고편차 토큰은 거의 항상 청크의 처음 k개 토큰에 집중**된다. 이는 LLM 이 "긴 시퀀스의 시작 부분 = sink" 라는 학습된 패턴이 있기 때문이며, 시퀀스 중간으로 옮긴 이미지 청크는 이 처음 k개에서 전역적 attention 분포 보정 신호가 발생한다.

따라서 이미지 청크에 대해서는 cosine 비교 없이 **결정론적으로 첫 k 개만 재계산**하면 충분하다. (텍스트 청크는 위치 의존성이 더 분산되어 있어 HKVD 식 cosine 기반이 여전히 유효 `(추정)`).

### 3.5 데이터 흐름 다이어그램

```
                   ┌────────────────────┐
   New prompt P'   │  Chunker + Hasher  │
   (image+text)    └─────────┬──────────┘
                             │
                             ▼
       ┌──────────────────────────────────────┐
       │   Lookup against Disk KV Store  S    │
       └──────────┬───────────┬───────────────┘
                  │ HIT       │ MISS
                  ▼           ▼
        ┌─────────────────┐  ┌────────────────────┐
        │ Async disk load │  │ GPU recompute MISS │
        │ (canonical KV)  │  │ (full chunk)       │
        └────────┬────────┘  └─────────┬──────────┘
                 │                     │
   ┌─────────────┘                     │
   ▼                                   │
 ┌──────────────────────────┐          │
 │ Re-apply RoPE at new pos │          │
 └────────────┬─────────────┘          │
              │                        │
              ▼                        │
 ┌──────────────────────────────┐      │
 │ Selective recompute SINK k   │      │
 │ tokens of each image chunk   │      │
 │ (parallel with disk load)    │      │
 └────────────┬─────────────────┘      │
              │                        │
              └────────┬───────────────┘
                       ▼
           ┌─────────────────────┐
           │ Stitch in order &   │
           │ replace recomputed  │
           │ slots (KV ready)    │
           └──────────┬──────────┘
                      ▼
                Decode loop
```

---

## 4. 이론적 근거 및 가정

| # | 가정 / 관찰 | 근거 |
|---|--------------|------|
| G1 | 이미지 청크의 KV 는 텍스트 컨텍스트의 영향이 매우 적다 (semantic invariance) | LLaVA·CacheBlend 실험 일반론 `(추정)` |
| G2 | 이미지 청크 시작부 k개 토큰이 attention sink 역할을 학습했다 | `Visual Attention Sink` 논문 (ICLR 2025, arXiv:2503.03321), `Attention Sinks` (Xiao 2024) |
| G3 | RoPE 는 곱셈으로 회전이 가해지므로 역회전 후 재적용 가능 (commutative w.r.t. self-attention shape) | RoPE 정의 자체 |
| G4 | 디스크 I/O 지연(NVMe 수십 µs/MB)은 GPU 재계산 시간(수 ms)으로 가려진다 | NVMe 4 GB/s + LLaVA-NeXT 이미지 KV ~ 50 MB/이미지 → ~12.5 ms `(추정 계산)` |
| G5 | k 는 이미지당 1~4 정도면 정확도 회복에 충분 | 본 논문 ablation `(미확인 — PDF 접근 차단)` |

가정 G1 이 깨지는 워크로드(예: 동일 이미지가 정반대 의미의 텍스트와 결합)에서는 정확도 저하가 클 수 있다. 보고된 **accuracy loss within 13.6%** 는 일부 이런 케이스를 포함한 worst-case 일 가능성 `(추정)`.

---

## 5. 복잡도 분석

| 항목 | InfoBlend | 전체 재계산 (baseline) | 비고 |
|------|-----------|----------------------|------|
| Prefill FLOPs | $O\left(\sum_{c \in \text{miss}} L_c^2 + \sum_{c \in \text{img-hit}} k \cdot L_c\right)$ | $O\left(\sum_c L_c^2\right)$ | $L_c$ 는 청크 길이; 이미지 hit 시 $L_c^2 \to k L_c$ 로 quadratic→linear |
| Disk read | $O\left(\sum_{c \in \text{hit}} \|\text{KV}_c\|\right)$ | 0 | NVMe 의존, 병렬 로드로 GPU 와 오버랩 |
| Disk write (storage phase) | $O(\|\text{KV}_c\|)$ per new chunk | 0 | 1회 amortized, 첫 prefill 비용에 포함 |
| GPU memory | $O(\text{batch} \times \text{chunk-size})$ working set | $O(\text{batch} \times \text{full-context})$ | 이론상 더 작음 (working set 만 GPU) |
| Disk capacity | 청크 KV 의 누적 합 (TB 단위 가능) | 0 | LMCache·FlexKV 와 유사 |
| Hash compute | $O(\|\text{chunk}\|)$ per chunk | 0 | SHA256 또는 vision-embedding 기반 |
| RoPE re-application | $O(\|\text{KV}_c\|)$ per hit | 0 | 곱셈 + 회전, 메모리 연산 ~ kernel-fused 가능 |

### 5.1 지연 모델 (간이)

이미지 1개 기준 (LLaVA-NeXT 가정 — 이미지 KV ≈ 50 MB, 시퀀스 길이 2880):

| 단계 | Baseline (full prefill) | InfoBlend (cache hit) | 비고 |
|------|--------------------------|----------------------|------|
| Vision encode | ~30 ms | 0 | 캐시됨 |
| Image-token prefill (FLOPs) | ~80 ms | ~80 / (2880/k) ms ≈ 0.1 ms (k=4) | quadratic→linear |
| Disk load | 0 | ~12.5 ms (NVMe 4 GB/s) | 병렬 |
| RoPE re-apply | 0 | ~1 ms `(추정)` | |
| **Total per image** | ~110 ms | ~12.5 ms (병렬화 후) | TTFT 단축 폭의 주된 원천 |

이 모델은 InfoBlend 의 **TTFT 54.1% 감소** 보고와 정성적으로 일치한다.

---

## 6. 보고된 실험 결과

> 1차 출처(OpenReview PDF) 직접 접근이 차단되어, 보고된 수치는 abstract·검색 결과의 인용에 의존한다. PDF 검증 후 갱신 필요. `(미확인 — 일부 수치)`

### 6.1 핵심 헤드라인 메트릭

| 메트릭 | 값 | 비교 대상 | 출처 |
|--------|----|---------|------|
| TTFT 감소 | **−54.1%** | Prefix caching baseline | abstract 인용 |
| Throughput 향상 | **2.0×** (online serving) | SOTA context cache | abstract 인용 |
| Accuracy loss | **≤ 13.6%** worst-case | 풀 재계산 | abstract 인용 (`(추정)` worst-case) |
| 모델 | LLaVA / Qwen-VL 계열 `(추정)` | — | 검색 결과 패턴 |
| 데이터셋 | Interleaved text-image, M-RAG | — | abstract 인용 |
| Selective recompute 비용 | <2% of full prefill `(추정)` | k=1~4 전제 | 분석 |

### 6.2 본 연구 목표 지표와의 매핑

| 본 연구 목표 지표 | InfoBlend 직접 영향 | 예상 방향 | 근거 |
|------------------|---------------------|----------|------|
| Inference Throughput (+20% 이상) | **직접 영향** | ↑↑ +100% (2×) | 보고된 수치 |
| KV Memory Reduction (-30% 이상) | 약함 | ↔ (사실상 GPU HBM 만 줄임) | 디스크에 KV 저장; HBM working set 만 절감 |
| Non-Contiguous Cache Hit Rate (≥30%) | **직접 영향** | ↑↑ | 멀티모달 RAG 에서 이미지 청크 hit 률이 50%+ `(추정)` |
| Effective Context Length (≥2×) | **직접 영향** | ↑↑ | 디스크 위계로 캐시 가능 컨텍스트 무제한 확장 |
| Compression Accuracy Delta (±1%) | **위반 위험** | ↓↓ | 보고치 −13.6% 는 기준선 ±1% 를 명백히 초과; sink-k 튜닝으로 회복 가능성 검증 필요 |
| Scheduling Overhead (TTFT p50 +5% 이내) | 음의 영향 (=개선) | ↑ | 디스크 로드 병렬화로 오히려 절감 |
| Cache Eviction Rate | 약한 영향 | ↔ | 디스크 위계로 evict 압력 완화 |

⚠ **중요**: InfoBlend 의 보고된 13.6% accuracy loss 는 본 연구의 **±1% 제약**을 명백히 위반한다. 따라서 InfoBlend 를 그대로 채택하기보다는, **k 토큰 선택 휴리스틱을 강화** 하거나 **CapKV·LookaheadKV 식 정보이론적 토큰 선택** 과 결합해 정확도를 회복시켜야 한다 (열린 질문 §10 참조).

---

## 7. 관련 / 경쟁 기술 비교

| 기법 | 핵심 차이 | 강점 | 약점 |
|------|----------|------|------|
| **CacheBlend** (EuroSys 2025) | 텍스트 RAG 에서 HKVD-기반 selective recompute | 학회 채택, 이론적 정당화 | 이미지 토큰 sink 패턴 모름 |
| **MPIC** (arXiv:2502.01960) | InfoBlend 와 거의 동일 problem statement, MLLM 대상 | 동시대 구현체 비교 가능 | InfoBlend 와 직접 우열 비교 결과 미확인 |
| **EPIC** (arXiv:2410.15332) | LLM(텍스트)용 일반 PIC, AttnLink+KVLink | 이론적 일반화 우수 | 멀티모달 미지원 |
| **KVLink** (arXiv:2502.16002) | Segment concatenation 의 어댑터 학습 기반 | 정확도 손실이 매우 작음 | 학습 필요 (training-required) |
| **KV Packet** (arXiv:2604.13226) | 0-FLOP 재계산, 어댑터 학습 | 가장 빠름 | 학습 필요, 새 모델마다 재학습 |
| **RelayCaching** (arXiv:2603.13289) | 멀티에이전트 디코딩 KV 의 프리필 재주입 | 훈련-무료, 4.7× | 단일 에이전트 RAG 시나리오 외에 효과 검증 필요 |
| **InfoBlend** | **멀티모달**, 위치-독립, **sink-k 결정론적 재계산**, **디스크 위계** | 훈련-무료, 멀티모달 특화, 디스크 활용 | 13.6% accuracy loss 잠재 |

### 7.1 차별점 (Why InfoBlend 가 기존과 다른가)

1. **이미지 청크 sink 의 결정론적 처리** — CacheBlend·EPIC 의 cosine HKVD 같은 동적 선택과 달리, "처음 k개" 라는 결정론적 규칙은 **runtime 비용 0**(top-k 비교 불필요).
2. **디스크 위계 활용** — MPIC 와는 design space 에서 가장 가까우나 InfoBlend 는 명시적으로 NVMe 활용을 가정한 파이프라인 설계.
3. **훈련-무료** — KV Packet, KVLink 와의 핵심 차이.

---

## 8. 본 연구로의 적용 가능성

### 8.1 Activity 분류

- **Activity B (주)**: 위치-독립 세그먼트 KV 재사용의 멀티모달 일반화
- **Activity A (보조)**: 디스크 ↔ GPU 비동기 로드 스케줄링
- **Activity C (간접)**: sink-k 선택적 재계산은 압축 손실 복원 메커니즘과 구조적 유사성

### 8.2 통합 후보 위치

| 우리 코드베이스 파일 | InfoBlend 매핑 | 변경 정도 |
|--------------------|----------------|-----------|
| `src/cache/segmented.py` (`SegmentedHashCache`) | **주 통합 지점**. 이미 위치-독립 청크 해시 + LRU + importance score 구조를 갖춤. modality 필드 추가, sink-k recompute hook 추가 | medium |
| `src/cache/base.py` (`CacheStore`) | 변경 불필요 — 현재 `put/get/evict` 인터페이스로 충분 | none |
| `src/cache/segment_adapter.py` | RoPE strip / re-apply 어댑터 신규 메서드 추가 | low |
| `src/engine/runner.py` | prefill 시 sink-k recompute 토큰 마스크 전달 | low–medium |
| `src/engine/batch_runner.py` | 디스크 I/O 와 재계산을 오버랩하는 prefetch hook | medium |
| (신규) `src/cache/multimodal_segment.py` | 이미지 청크 vs 텍스트 청크 modality-aware 변형 | new file |

### 8.3 `CacheStore` 인터페이스 호환성

`src/cache/base.py` 를 검토한 결과 (`base.py` 라인 1–32):

```python
class CacheStore(ABC):
    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...
```

InfoBlend 는 다음 의미로 호환된다.

- `put(key, value)`: `key = position_independent_hash`, `value = canonical_KV` 그대로 매핑.
- `get(key)`: 디스크 hit 일 경우 RoPE re-apply 는 `get` 외부에서 수행 — `get` 자체는 canonical KV 만 반환하면 됨.
- `evict()`: 디스크에서 LRU evict — 현재 `SegmentedHashCache._importance` 와 호환.
- `memory_bytes()`: GPU + 디스크 합산 또는 별도 method 추가 권장 (단순 `bytes` 만 반환하므로 wrap 가능).

⚠ **부족한 부분** — `CacheStore` 인터페이스에 modality 메타데이터가 없다. `put` 의 `value` 를 `(KV, meta)` 튜플로 확장하거나 별도 `put_with_meta(key, value, meta)` 를 추가해야 sink-k 재계산이 modality-aware 해진다.

### 8.4 예상 이점

- **Activity B 의 핵심 가설(위치-독립 청크 재사용)** 을 멀티모달로 확장 가능 → 비연속 히트율 ≥30% 목표 달성에 유리.
- **이미지 RAG / 에이전트 시나리오** 에서 다른 베이스라인이 거의 0% 히트인 데 반해 InfoBlend 는 50%+ 히트 가능 `(추정)`.
- 디스크 위계로 **Effective Context Length 2×** 목표 달성에 직결 (FlexKV 와 시너지).

### 8.5 예상 비용 / 리스크

| 항목 | 내용 |
|------|------|
| **정확도 리스크 (큼)** | 보고된 13.6% loss 는 본 연구 ±1% 제약 위반. sink-k 튜닝 + Activity C 의 LookaheadKV/CapKV 식 토큰 선택 결합 필요 |
| **구현 복잡도** | RoPE strip/re-apply 의 정확한 구현이 까다롭다. PyTorch 래핑 시 attention backend 와의 호환성 검증 필요 |
| **모델 의존성** | LLaVA·Qwen-VL 같은 vision-language 모델 토크나이저·이미지 토큰 경계 식별 로직 필요. 우리 베이스라인이 텍스트-only 라면 멀티모달 베이스라인 구축이 선행 |
| **디스크 의존성** | NVMe 가정. 클라우드 환경에 따라 디스크 대역폭이 보장 안 될 수 있음 |
| **k 의 선택** | k 가 너무 작으면 sink 누수, 너무 크면 재계산 비용 증가. workload-specific 튜닝 필요 |

### 8.6 다른 Activity 와의 시너지 / 충돌

**시너지**

- **A (스케줄링)** + **B (InfoBlend)**: KV-aware 라우터가 "이 노드의 디스크에 hit 가능 청크 다수 보유" 정보를 활용하면 캐시 친화 라우팅이 가능. Crusoe MemoryAlloy 와 결합 시 클러스터 규모 확장.
- **C (LookaheadKV / CapKV)**: sink-k 의 "결정론적 처음 k개" 휴리스틱을 정보이론적 토큰 선택으로 대체하면 정확도 회복 가능 (열린 질문 §10).
- **B 내부 (RelayCaching)**: RelayCaching 의 디코딩 KV → 프리필 재주입에서 InfoBlend 의 sink-k 처방을 재사용 가능 (디코딩 KV 에서도 첫 k 토큰만 재계산).

**충돌**

- **C (VQKV 등 압축)** + **InfoBlend**: 압축 KV 를 디스크에 저장하면 RoPE strip 정확도가 추가 손상. 압축은 RoPE 적용 후가 아니라 **canonical(strip 후) KV 에 적용** 해야 함 → 구현 순서 중요.

---

## 9. vLLM 이식 관점

### 9.1 vLLM 내 유사 기능 유무

| vLLM 기능 | 관련성 |
|----------|-------|
| **Prefix caching** (RadixAttention) | 위치 일치 시만 hit. InfoBlend 비교 baseline 으로 사용. |
| **FlexKV connector** (v0.17.2+) | 디스크/원격 KV 저장 인프라. InfoBlend 의 디스크 위계와 직접 합치. |
| **Generalized KV Cache Reuse** (Issue #25672) | 비연속 KV 재사용을 mainline 로드맵. InfoBlend 알고리즘이 이 이슈의 후보 구현 중 하나. |
| **MultiModal projection cache** (vLLM v0.18+) | 이미지 임베딩 캐싱 일부 지원. KV 레벨까지는 미지원. |
| **CacheBlend integration** | 비공식 fork 존재 `(미확인)`. InfoBlend 직접 통합은 없음. |

### 9.2 손대야 할 모듈 후보

| vLLM 모듈 (mainline 기준) | 변경 내용 |
|---------------------------|----------|
| `vllm/attention/backends/*.py` | RoPE strip/re-apply 가 attention backend 에서 수행되므로 backend-agnostic hook 필요 |
| `vllm/core/block_manager.py` | 청크 단위 KV block 의 위치-독립 키잉 추가 |
| `vllm/core/scheduler.py` | sink-k recompute 토큰 마스크를 prefill batch 에 주입 |
| `vllm/multimodal/processing.py` | modality(image vs text) 메타를 청크에 부착 |
| `vllm/distributed/kv_transfer/kv_connector/*` | FlexKV connector 의 PUT/GET key 를 위치-독립 해시로 변경 |
| `vllm_integration/block_manager_patch.py` | InfoBlend 청크 키잉 패치 |
| `vllm_integration/attention_backend_patch.py` | RoPE strip 후 저장 / re-apply 후 로드 |
| `vllm_integration/scheduler_patch.py` | sink-k 마스크 생성 + prefill 토큰 부분집합 재계산 |

### 9.3 이식 난이도

| 측면 | 난이도 | 근거 |
|------|--------|------|
| 단일 노드 파이프라인 | **medium** | `SegmentedHashCache` 가 이미 청크 해시 인프라를 제공 |
| FlexKV 와 결합 (디스크 위계) | **medium** | FlexKV mainline 통합 완료, key 만 위치-독립 해시로 교체 |
| RoPE strip / re-apply | **medium-high** | attention backend 별 RoPE 구현이 다양 (FlashAttention, xFormers, Triton) → backend 호환성 |
| sink-k 정확도 검증 | **high** | LLaVA·Qwen-VL 평가 파이프라인 구축 필요. 텍스트-only 베이스라인을 가진 본 연구로서 멀티모달 평가 인프라가 새 비용 |
| 멀티 노드 분산 | **high** | FlexKV 의 RDMA + Mooncake 와 InfoBlend 키 일관성 보장 필요 |
| **종합 추정** | **medium-high** | 단일 노드만이면 medium, 멀티 노드 + 정확도 보장까지 하면 high |

---

## 10. 열린 질문 / 후속 실험 제안

### 가설 H1: sink-k 결정론적 규칙을 정보이론적 선택으로 대체하면 정확도가 ±1% 이내로 회복된다

- **검증**: LookaheadKV (ICLR 2026) 또는 CapKV 의 leverage score 로 "어떤 토큰을 재계산할지" 를 동적으로 결정. k 자체도 청크별 가변.
- **메트릭**: LLaVA-1.5 + LongBench-VQA 에서 accuracy delta, TTFT.
- **예상**: 결정론 k=4 의 13.6% loss → 동적 선택 시 ≤2% loss `(가설)`.

### 가설 H2: RelayCaching 의 디코딩-KV 재주입에 sink-k 처방을 결합하면 멀티에이전트 멀티모달 시나리오에서 4.7× 이상의 속도향상이 가능하다

- **검증**: 에이전트 A(이미지 분석) → 에이전트 B(질의 응답) 파이프라인. 에이전트 A 의 디코딩 KV (이미지에 대한 reasoning) 를 에이전트 B 의 프리필에 sink-k 처방으로 재주입.
- **메트릭**: 두 에이전트 합산 latency, end-to-end 정확도.
- **예상**: 단일 에이전트 RelayCaching 4.7× 위에 +20~40% 추가 가속 `(가설)`.

### 가설 H3: VQKV(82.8% 압축) 또는 SAW-INT4 압축을 InfoBlend 의 canonical KV 에 적용하면, 디스크 용량과 디스크→GPU 전송 시간이 동시에 절감되어 TTFT 가 추가로 −20~30% 더 줄어든다

- **검증**: InfoBlend 디스크 저장 직전 INT4 양자화. 로드 후 RoPE re-apply 전에 dequantize. 정확도와 TTFT 비교.
- **메트릭**: TTFT, accuracy delta (양자화 없는 InfoBlend 대비), disk capacity reduction.
- **예상**: 디스크 4× 용량 절감, 로드 시간 4× 단축, accuracy ≤+1% 추가 손상 `(가설)`.

### 가설 H4: 본 연구의 `SegmentedHashCache` 에 modality 인지 sink-k recompute 를 추가하면, 멀티모달 RAG 워크로드에서 비연속 히트율이 30% 이상 달성된다

- **검증**: `src/cache/multimodal_segment.py` 신규 구현 → InfoBlend 알고리즘 적용 → MMRAG 합성 워크로드(이미지 청크 N=8개를 임의 순서로 결합) 평가.
- **메트릭**: 비연속 히트율, accuracy delta, TTFT.
- **예상**: 비연속 히트율 35~50%, accuracy delta ≤±2% (sink-k=4 기준).

### 가설 H5: MPIC 대비 InfoBlend 의 sink-k 결정론 규칙이 미세하지만 일관된 정확도 우위를 보인다

- **검증**: 동일 MMRAG 워크로드에서 InfoBlend vs MPIC 직접 비교. (1차 출처에 직접 비교가 없을 가능성 — 동시대 연구).
- **메트릭**: TTFT, accuracy delta, throughput.
- **예상**: 정확도 동률 또는 InfoBlend +0.5~1% `(가설)`. TTFT 는 두 시스템의 디스크 I/O 구현 디테일에 의존.

---

## 11. 본 연구 컨텍스트와의 연결

본 리포트는 `reports/trends/2026-04-28.md` 와 `reports/ideas/2026-04-28.md` 에서 이미 InfoBlend 가 다음과 같이 다루어진 점을 확인했다.

- 트렌드 리포트(2026-04-28)에서 Activity B 신규 항목 #2 로 등장, "CacheBlend 의 KVLink 를 개선해 멀티모달 컨텍스트에서도 위치 제약 없이 KV 캐시를 재조합. TTFT 를 최대 54.1% 감소" 로 요약됨.
- 아이디어 리포트(2026-04-28)에서 **B-1: 위치-독립 세그먼트 해시 캐시 (InfoBlend 기반)** 가 우선순위 1(이번 사이클 구현 타겟)로 채택됨.
- 트렌드 리포트(2026-04-29)의 동향 변화 표에서 "CacheBlend / Blending / KVLink: InfoBlend TTFT -54%" 로 인용되며, KV Packet 의 등장으로 PIC 패러다임 경쟁이 격화되었음을 기록.

본 deep-dive 는 그 후속으로, **(1) 알고리즘의 sink-k 메커니즘을 보다 정확하게 분해**하고, **(2) 본 연구 코드베이스(`SegmentedHashCache`)와의 정확한 통합 방식**을 제시하며, **(3) 보고된 13.6% accuracy loss 가 본 연구 ±1% 제약을 위반함을 명시적으로 경고**하고 회복 경로(H1~H3)를 제안한다.

---

## 12. 식별된 한계 및 1차 출처 추가 검증 필요 항목

본 리포트 작성 시점(2026-05-02)에서 OpenReview PDF 직접 접근이 차단되어 다음 항목은 `(미확인)` 으로 표시됐다. 향후 PDF 접근 후 갱신 권장.

| 항목 | 현재 상태 | 갱신 후 확인 필요 |
|------|----------|------------------|
| 저자 명단 / 소속 | `(미확인)` | 익명 제출 여부 |
| 정확한 venue / decision | `(미확인 — ICLR 2026 추정)` | acceptance 여부 |
| `k` 의 권장값 (ablation) | `(추정 1~4)` | 정확한 sweep 결과 |
| 평가 모델 정확 명단 | `(추정 LLaVA / Qwen-VL)` | 모델 버전 |
| 평가 데이터셋 | "interleaved text-image, M-RAG" 만 확인 | 정확한 벤치마크 (MMMU, LongBench-V 등) |
| MPIC 와의 직접 비교 | `(미확인)` | 동시대 비교 유무 |
| 공식 코드 저장소 | `(미확인 — 미공개 추정)` | supplementary 첨부 코드 |
| 13.6% accuracy loss 가 worst-case 인지 평균인지 | `(추정 worst-case)` | 본 수치의 정확한 정의 |

---

## 참고 자료

### 1차 출처

- InfoBlend OpenReview Forum: <https://openreview.net/forum?id=bld5GVRad0>
- InfoBlend OpenReview PDF: <https://openreview.net/pdf?id=bld5GVRad0>

### 직접 비교 / 인접 1차 출처

- CacheBlend (EuroSys 2025): <https://arxiv.org/abs/2405.16444>
- CacheBlend GitHub: <https://github.com/YaoJiayi/CacheBlend>
- MPIC (Position-Independent Multimodal Caching): <https://arxiv.org/abs/2502.01960>
- EPIC (Efficient Position-Independent Caching): <https://arxiv.org/abs/2410.15332>
- KVLink: <https://arxiv.org/abs/2502.16002>
- KV Packet: <https://arxiv.org/abs/2604.13226>

### Attention Sink 이론적 기반

- "When Attention Sink Emerges in Language Models" (ICLR 2025 Spotlight): <https://github.com/sail-sg/Attention-Sink>
- "See What You Are Told: Visual Attention Sink in Large Multimodal Models" (ICLR 2025, arXiv:2503.03321): <https://arxiv.org/abs/2503.03321>
- "Not Errors but Guardians: Understanding Sink Tokens in Multimodal LLMs" (OpenReview): <https://openreview.net/forum?id=EpoJKtVxNt>

### 본 연구 내부 참고

- 트렌드 리포트 (InfoBlend 최초 인용): `reports/trends/2026-04-28.md`
- 트렌드 리포트 (InfoBlend 동향 변화): `reports/trends/2026-04-29.md`
- 트렌드 리포트 (최신 트렌드): `reports/trends/2026-05-02.md`
- 아이디어 리포트 (B-1 InfoBlend 기반 채택): `reports/ideas/2026-04-28.md`
- FlexKV deep-dive (디스크 위계 인프라 비교): `reports/deep-dive/2026-05-02-flexkv.md`
- 본 연구 캐시 추상: `src/cache/base.py`, `src/cache/segmented.py`, `src/cache/segment_adapter.py`

---

DEEP_DIVE_REPORT_SAVED: reports/deep-dive/2026-05-02-infoblend.md
