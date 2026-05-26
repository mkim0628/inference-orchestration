<!-- 변경 이유 (이전 Spec.md: 2026-05-25 대비):
이전 사이클(2026-05-25)은 B+C 조합이었다:
  - C-1 VeriCacheSpeculativeKVDraftVerifyLosslessCodec (투기적 드래프트-검증 무손실 코덱)
  - B-1 KVPacketSoftTokenAdapterZeroFLOPsNonContiguousCache (소프트-토큰 어댑터 비연속 재사용)
  - Cross-1 VeriCacheKVPacketZeroFLOPsLosslessNonContiguousPipeline (B+C 통합)

이번 사이클(2026-05-26)은 A+B 조합(C 시너지 포함)으로 전환된다.
핵심 전환:
  - Activity B 최우선: KV Packet/어댑터 기반 "소프트-토큰 근사" 방식에서
    Irminsul의 "MLA 아키텍처 구조적 c_KV/k_r 분리 + 수학적 δ-회전 위치 수정"으로 전환.
    훈련 없이 수학적으로 위치-독립 비연속 재사용을 보장하며,
    GQA/MHA 모델은 기존 AdapShot RoPE 재인코딩 경로로 자동 fallback한다.
  - Activity A 2순위: S3 오브젝트 스토리지를 4번째 KV 계층으로 추가하는
    ObjectCacheS3TierRouter를 구현한다. 브레이크이븐 히트율 기반 동적 티어 전환으로
    TTFT +5% 이내 제약을 S3 계층까지 유지한다.
  - Activity B 3순위: CDC 콘텐츠 해시 통합 세그먼트 ID 인터페이스(B-2)로
    Irminsul과 ObjectCache를 동일 주소 체계로 연결한다.
  - Cross-1 A+B: IrminsulObjectCachePipeline으로 엔드-투-엔드 통합 파이프라인 완성.
  - Activity C 시너지: MLATwoAxisCompressionCodec (위치 축 × 깊이 축) — C는
    accuracy-preserving 검증 계획 포함으로 이번 사이클에서 선택적으로 구현한다.

주요 변경:
1. [신규] src/cache/irminsul_mla_segment_cache.py (B-1, 최우선)
2. [신규] src/cache/arch_aware_noncontiguous_router.py (B-1 라우터)
3. [신규] src/cache/cdc_content_hash_interface.py (B-2)
4. [신규] src/scheduler/objectcache_s3_tier_router.py (A-1)
5. [신규] src/engine/irminsul_objectcache_pipeline.py (Cross-1)
6. [신규] src/cache/mla_two_axis_compression_codec.py (C-1 시너지, 선택적)
7. [신규] configs/experiments/2026-05-26.yaml
8. [신규] configs/arch_registry.yaml
9. [신규] configs/objectcache_breakeven_table.yaml
10. [신규] configs/objectcache_breakeven_calibration.py
11. [신규] tests/unit/test_irminsul_mla_segment_cache.py (B-1)
12. [신규] tests/unit/test_arch_aware_noncontiguous_router.py (B-1 라우터)
13. [신규] tests/unit/test_cdc_content_hash_interface.py (B-2)
14. [신규] tests/unit/test_objectcache_s3_tier_router.py (A-1)
15. [신규] tests/unit/test_compression_accuracy.py에 C-1 MLATwoAxis 케이스 추가
16. [신규] tests/integration/test_irminsul_objectcache_pipeline_e2e.py (Cross-1)
17. [보존] 이전 사이클 구현 파일 전부 수정 금지.
    특히 vericache_speculative_codec.py, kv_packet.py, kv_packet_adapter.py,
    speculative_packet_pipeline.py 등 기존 단위·통합 테스트 회귀 없이 통과해야 한다.
-->

# Spec — 2026-05-26: Irminsul MLA δ-Rotation Non-Contiguous Reuse + ObjectCache S3 4-Tier Routing

## 배경

**기반 아이디어 리포트**: `reports/ideas/2026-05-26.md`

**최우선 구현 타겟**: B-1 `IrminsulMLANativeDeltaRotationArchAwareNonContiguousRouter`
**2순위 구현 타겟**: A-1 `ObjectCacheS3TierBreakEvenRoutingPolicy`
**3순위 구현 타겟**: B-2 `CDCContentHashUnifiedSegmentIDInterface`
**통합 타겟**: Cross-1 `IrminsulObjectCacheCDCLayerwiseRDMAMLAPipeline` (A+B)
**선택적 C 시너지**: C-1 `MLATwoAxisCompressionCodec` (위치 축 × 깊이 축)

**해결하려는 문제**:

- **Activity B (Irminsul MLA-네이티브 δ-회전 아키텍처-인식 비연속 재사용)**:
  기존 비연속 재사용 기법(AdapShot RoPE 재인코딩, KV Packet 어댑터 등)은 GQA/MHA
  아키텍처를 전제하거나 훈련이 필요하다. MLA 아키텍처(DeepSeek-V2/V3/R1,
  Kimi-K2/Moonlight)는 c_KV(위치-자유 압축 표현)와 k_r(64-dim, δ-회전으로 수정
  가능) 분리 구조를 가지므로, c_KV를 위치와 무관하게 재사용하고 k_r에만 δ-회전을
  적용해 훈련 없이 수학적으로 보장되는 위치-독립 비연속 재사용이 가능하다.
  CDC 청킹 + 콘텐츠-해시 키잉으로 절대 위치에 무관한 세그먼트 인덱싱을 구현한다.

- **Activity A (ObjectCache S3 티어 브레이크이븐 기반 동적 KV 라우팅)**:
  기존 3-티어(HBM/DRAM/SSD) 구성은 물리적 용량 한계로 긴 컨텍스트와 에이전틱
  세션을 지원하기 어렵다. S3 호환 오브젝트 스토리지를 4번째 계층으로 추가하되,
  브레이크이븐 히트율 계산으로 S3 전달 비용이 재계산 비용보다 낮을 때만 활성화해
  TTFT +5% 이내 제약을 유지하면서 유효 컨텍스트 길이를 2× 이상 확장한다.

- **Activity B-2 (CDC 통합 주소 인터페이스)**:
  Irminsul(B-1)의 CDC 콘텐츠 해시와 ObjectCache(A-1)의 S3 오브젝트 키가 동일한
  SHA256 기반 주소 체계를 공유하도록 통합 인터페이스를 구현해, 계층 간 투명한
  세그먼트 접근을 가능하게 한다.

---

## 이번 사이클 Activity

- [x] Activity A: KV Cache-aware Scheduling / S3 4-Tier Routing (A-1)
- [x] Activity B: Non-Contiguous KV Cache Reuse — MLA δ-Rotation (B-1, B-2)
- [x] Activity C: KV Cache Compression — MLA 2-Axis Codec (C-1, 선택적, accuracy 검증 포함)

---

## 목표

- [ ] 목표 1: 비연속 캐시 히트율 전체 히트의 30% 이상 달성 (evaluation_criteria.md §3)
- [ ] 목표 2: 추가 토큰 회수율 MLA 모델에서 +77% 이상 (에이전틱 세션 반복 청크 기준)
- [ ] 목표 3: TTFT p50 증가 +5% 이내 (A-1 ObjectCacheS3TierRouter, §2)
- [ ] 목표 4: 유효 컨텍스트 길이 베이스라인 대비 2× 이상 (S3 4번째 계층, §4)
- [ ] 목표 5 (C-1 포함 시): KV 캐시 메모리 감소 −30% 이상, accuracy delta ±1% 이내 (§4)

---

## 구현 범위

### 새로 만들 파일

| 파일 | Activity | 역할 |
|------|----------|------|
| `src/cache/irminsul_mla_segment_cache.py` | B-1 | MLA c_KV/k_r 분리 저장 + δ-회전 위치 수정, CDC 청킹, CacheStore 구현 |
| `src/cache/arch_aware_noncontiguous_router.py` | B-1 | MLA vs GQA/MHA 아키텍처 감지 분기 라우터 |
| `src/cache/cdc_content_hash_interface.py` | B-2 | CDC 콘텐츠 해시 통합 세그먼트 ID 인터페이스, 계층 투명 조회 |
| `src/scheduler/objectcache_s3_tier_router.py` | A-1 | S3 4번째 계층 브레이크이븐 기반 동적 티어 라우터, BaseScheduler 상속 |
| `src/engine/irminsul_objectcache_pipeline.py` | Cross-1 | A+B 통합 파이프라인 (CDC→계층 조회→S3 RDMA→δ-회전→어텐션) |
| `src/cache/mla_two_axis_compression_codec.py` | C-1 | MLA c_KV 위치 축 × 깊이 축 2축 압축 코덱 (선택적) |
| `configs/arch_registry.yaml` | B-1 | 지원 모델 아키텍처 레지스트리 |
| `configs/objectcache_breakeven_table.yaml` | A-1 | 컨텍스트 길이별 브레이크이븐 히트율 룩업 테이블 |
| `configs/objectcache_breakeven_calibration.py` | A-1 | T_s3, T_recompute 측정 및 브레이크이븐 테이블 생성 스크립트 |
| `configs/experiments/2026-05-26.yaml` | 공통 | 이번 사이클 실험 설정 |
| `tests/unit/test_irminsul_mla_segment_cache.py` | B-1 | δ-회전 정확성 + CDC 청킹 + CacheStore 인터페이스 단위 테스트 |
| `tests/unit/test_arch_aware_noncontiguous_router.py` | B-1 | 아키텍처 감지 분기 단위 테스트 |
| `tests/unit/test_cdc_content_hash_interface.py` | B-2 | 통합 주소 조회 단위 테스트 |
| `tests/unit/test_objectcache_s3_tier_router.py` | A-1 | 브레이크이븐 계산 + EMA 티어 전환 단위 테스트 |
| `tests/unit/test_compression_accuracy.py` | C-1 | MLATwoAxisCompressionCodec accuracy 검증 (기존 파일에 케이스 추가) |
| `tests/integration/test_irminsul_objectcache_pipeline_e2e.py` | Cross-1 | A+B 통합 엔드-투-엔드 테스트 |

### 변경할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/cache/segmented.py` | 변경 없음 — 기존 SegmentedHashCache 보존. IrminsulMLASegmentCache가 독립 클래스로 CacheStore 직접 구현 |
| `tests/unit/test_compression_accuracy.py` | C-1 MLATwoAxisCompressionCodec 테스트 케이스 추가 (기존 케이스 보존) |

---

## 알고리즘 상세

### 1. CDC 청킹 (Content-Defined Chunking) — B-1, B-2 공통

```python
def cdc_chunk(
    token_ids: List[int],
    avg_chunk_size: int = 256,
    min_chunk_size: int = 64,
    max_chunk_size: int = 1024,
    window_size: int = 32,
    modulus: int = 0,  # 0 → 자동 계산: avg_chunk_size - 1
) -> List[List[int]]:
    """Rabin 핑거프린트 기반 CDC 청킹.

    알고리즘:
      1. modulus = avg_chunk_size - 1 (기본값)
      2. 슬라이딩 윈도우(window_size 토큰)로 Rabin 핑거프린트 계산
      3. fingerprint & modulus == 0 인 위치에서 청크 경계 설정
      4. min_chunk_size / max_chunk_size 제약 적용
      5. 마지막 청크: 남은 토큰 모두 포함

    Returns:
      List of chunks, each chunk is a list of token IDs.
    """
    if modulus == 0:
        modulus = avg_chunk_size - 1

    chunks: List[List[int]] = []
    start = 0
    n = len(token_ids)
    fp = 0
    BASE = 31
    MOD = 2**32

    while start < n:
        end = start
        current_len = 0
        while end < n:
            # 슬라이딩 윈도우 Rabin 핑거프린트 업데이트
            fp = (fp * BASE + token_ids[end]) % MOD
            current_len += 1
            end += 1
            # 최소 길이 이상이고 핑거프린트 경계 조건 만족 시 분할
            if current_len >= min_chunk_size and (fp & modulus) == 0:
                break
            # 최대 길이 도달 시 강제 분할
            if current_len >= max_chunk_size:
                break
        chunks.append(token_ids[start:end])
        start = end
        fp = 0

    return chunks


def cdc_segment_key(chunk_tokens: List[int]) -> str:
    """청크 콘텐츠의 SHA256 해시를 세그먼트 키로 반환 (위치-독립)."""
    import hashlib, struct
    raw = struct.pack(f"{len(chunk_tokens)}I", *chunk_tokens)
    return hashlib.sha256(raw).hexdigest()
```

---

### 2. IrminsulMLASegmentCache (Activity B-1) — `src/cache/irminsul_mla_segment_cache.py`

```python
@dataclass
class IrminsulKVEntry:
    segment_key: str          # SHA256(token_content) — 위치-독립 키
    c_kv: torch.Tensor        # [n_tokens, d_c]  위치-자유 압축 표현
    k_r: torch.Tensor         # [n_tokens, 64]   64-dim RoPE 성분
    source_position: int      # 원래 저장 위치 오프셋 (δ 계산용)
    n_tokens: int
    layer_idx: int

@dataclass
class IrminsulMLAConfig:
    avg_chunk_size: int = 256     # YAML 외부화
    min_chunk_size: int = 64
    max_chunk_size: int = 1024
    rope_base: float = 10000.0
    k_r_dim: int = 64             # MLA k_r 차원 (DeepSeek 고정값)
    max_entries: int = 2000
    seed: int = 42


class IrminsulMLASegmentCache(CacheStore):
    """MLA-native position-independent non-contiguous KV cache (Activity B-1).

    CDC 청킹 + SHA256 콘텐츠 해시 키잉으로 세그먼트를 인덱싱한다.
    재사용 시 k_r에만 δ-회전을 적용하고 c_kv는 그대로 재사용한다.

    CacheStore 인터페이스 완전 구현.
    """

    def __init__(self, config: IrminsulMLAConfig) -> None: ...

    # CacheStore 추상 메서드
    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...

    # MLA 전용 확장 메서드
    def put_mla_segment(
        self,
        chunk_tokens: List[int],
        c_kv: torch.Tensor,       # [n_tokens, d_c]
        k_r: torch.Tensor,        # [n_tokens, k_r_dim]
        source_position: int,
        layer_idx: int = 0,
    ) -> str:
        """CDC 청크의 MLA KV를 저장하고 segment_key를 반환."""
        ...

    def get_mla_segment_with_delta_rotation(
        self,
        segment_key: str,
        target_position: int,
        layer_idx: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """segment_key에 해당하는 (c_kv, k_r_corrected) 반환.

        k_r_corrected = apply_delta_rotation(k_r, delta=target_position - source_position)
        c_kv는 위치-자유이므로 수정 없이 반환.
        미스 시 None 반환.
        """
        ...

    def get_segments_mla(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], List[List[int]]]:
        """CDC 청킹 후 전체 청크를 조회.

        Returns:
          hits: [(chunk_local_idx, c_kv, k_r_corrected), ...]
          miss_chunks: [[token_ids...], ...]  재계산 필요한 청크의 토큰 목록
        """
        ...

    def noncontiguous_hit_rate(self) -> float: ...
```

---

### 3. δ-회전 위치 수정 — B-1 핵심 알고리즘

```python
def apply_delta_rotation(
    k_r: torch.Tensor,          # [n_tokens, k_r_dim]  k_r_dim = 64
    delta: int,                  # target_position - source_position
    rope_base: float = 10000.0,
    k_r_dim: int = 64,
) -> torch.Tensor:
    """MLA k_r에 δ-회전을 적용해 위치 수정.

    수학적 근거 (Irminsul arXiv 2605.05696):
      - RoPE 공식: k_r_pos[i] = k_r[i] * cos(pos * θ_i) - k_r_perp[i] * sin(pos * θ_i)
      - k_r_perp: k_r의 각 pair에서 [-sin, cos] 성분 (직교 회전)
      - δ-회전: source → target은 δ = target - source 만큼의 추가 회전
        k_r_corrected[i] = k_r[i] * cos(δ * θ_i) - k_r_perp[i] * sin(δ * θ_i)

    알고리즘:
      1. θ_i = rope_base^(-2i/k_r_dim), i = 0..k_r_dim//2 - 1
      2. delta_angles = delta * θ_i  shape [k_r_dim//2]
      3. cos_a = cos(delta_angles), sin_a = sin(delta_angles)
      4. k_r를 [..., k_r_dim//2, 2] 형태로 reshape
         (각 쌍 [k_r[2i], k_r[2i+1]])
      5. 2D 회전 적용:
         new_k_r[..., 0] = k_r[..., 0] * cos_a - k_r[..., 1] * sin_a
         new_k_r[..., 1] = k_r[..., 0] * sin_a + k_r[..., 1] * cos_a
      6. 결과를 [..., k_r_dim]으로 reshape

    오버헤드: 64-dim × n_tokens 행렬 연산 → < 0.1ms/세그먼트 (CPU)
    """
    ...


def assert_delta_rotation_correctness(
    k_r_dim: int = 64,
    rope_base: float = 10000.0,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """δ-회전 정확성 단위 테스트용 검증 함수.

    검증 방법:
      1. 임의의 source_position, target_position 설정
      2. RoPE를 source_position에서 직접 적용한 k_r_at_source 계산
      3. apply_delta_rotation(k_r_at_source, delta=target-source)로 수정
      4. RoPE를 target_position에서 직접 적용한 k_r_at_target 계산
      5. k_r_corrected ≈ k_r_at_target (rtol, atol 이내) 검증
    assert torch.allclose(k_r_corrected, k_r_at_target, rtol=rtol, atol=atol)
    """
    ...
```

---

### 4. ArchitectureAwareNonContiguousRouter (Activity B-1) — `src/cache/arch_aware_noncontiguous_router.py`

```python
@dataclass
class ModelConfig:
    """모델 아키텍처 설정 파라미터."""
    model_name: str
    kv_lora_rank: Optional[int] = None     # MLA 전용: c_KV 압축 랭크
    qk_rope_head_dim: Optional[int] = None # MLA 전용: k_r 차원 (=64)
    num_kv_heads: int = 8
    d_head: int = 64
    rope_base: float = 10000.0


def detect_attention_arch(model_config: ModelConfig) -> str:
    """모델 설정에서 어텐션 아키텍처 유형 감지.

    Returns: "MLA" | "GQA" | "MHA"

    MLA 감지 조건:
      model_config.kv_lora_rank is not None AND
      model_config.qk_rope_head_dim is not None AND
      model_config.qk_rope_head_dim == 64
    GQA 감지 조건: num_kv_heads < num_q_heads (단순화: num_kv_heads <= 4)
    MHA: 그 외
    arch_registry.yaml의 model_name 매핑이 우선 적용됨.
    """
    ...


class ArchitectureAwareNonContiguousRouter(CacheStore):
    """MLA vs GQA/MHA 아키텍처에 따라 최적 비연속 재사용 경로를 선택하는 라우터.

    MLA 경로: IrminsulMLASegmentCache (δ-회전)
    GQA/MHA 경로: RoPEReencodingNonContiguousCache (AdapShot RoPE 재인코딩)

    CacheStore 인터페이스 완전 구현.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        mla_cache: IrminsulMLASegmentCache,
        gqa_mha_cache: CacheStore,   # RoPEReencodingNonContiguousCache 또는 SegmentedHashCache
    ) -> None: ...

    # CacheStore 추상 메서드 — 내부적으로 아키텍처에 따라 위임
    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...

    @property
    def arch(self) -> str:
        """현재 모델의 감지된 아키텍처 유형."""
        ...

    def get_segments_routed(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """아키텍처 감지 후 적절한 비연속 세그먼트 조회 경로를 선택.

        MLA: IrminsulMLASegmentCache.get_segments_mla()
        GQA/MHA: RoPEReencodingNonContiguousCache.get_segments_with_rope()
        반환 형식: (hits, miss_chunk_indices) — SegmentedHashCache 호환
        """
        ...
```

---

### 5. CDCContentHashSegmentIDInterface (Activity B-2) — `src/cache/cdc_content_hash_interface.py`

```python
@dataclass
class SegmentMetadata:
    segment_id: str                  # SHA256(CDC_chunk_token_bytes)
    source_position: int
    n_tokens: int
    model_arch: str                  # "MLA" | "GQA" | "MHA"
    storage_tier: str                # "HBM" | "DRAM" | "SSD" | "S3"
    layer_idx: int = 0


class CDCContentHashSegmentIDInterface:
    """Irminsul(B-1) + ObjectCache(A-1) 통합 세그먼트 주소 체계.

    통일 키: SegmentID = SHA256(CDC_chunk_token_bytes)
    계층 조회 순서: HBM → DRAM → SSD → S3 → 재계산

    S3 접근은 boto3 / minio 클라이언트로 추상화한다.
    S3 엔드포인트는 configs/objectcache_breakeven_table.yaml에서 로드한다.
    S3 접근 불가 환경에서는 S3 계층을 건너뛰고 재계산으로 fallback한다.
    """

    def __init__(
        self,
        hbm_cache: CacheStore,
        dram_cache: Optional[CacheStore] = None,
        ssd_cache: Optional[CacheStore] = None,
        s3_client: Optional[object] = None,   # boto3.client('s3') 또는 None
        s3_bucket: str = "kvcache",
        model_name: str = "default",
    ) -> None: ...

    def lookup(
        self,
        segment_id: str,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """계층 순서대로 세그먼트 조회.

        Returns:
          (kv_tensor, tier_name): 히트한 계층 이름과 KV 텐서
          (None, "miss"): 전 계층 미스
        """
        ...

    def store(
        self,
        segment_id: str,
        kv: torch.Tensor,
        tier: str = "HBM",
        layer_idx: int = 0,
    ) -> None:
        """지정 계층에 KV 저장."""
        ...

    def s3_object_key(self, segment_id: str, layer_idx: int) -> str:
        """S3 오브젝트 키 생성: f"{model_name}/{segment_id}_{layer_idx}.kvcache" """
        ...

    @staticmethod
    def make_segment_id(chunk_tokens: List[int]) -> str:
        """CDC 청크 콘텐츠의 SHA256 해시를 segment_id로 반환."""
        return cdc_segment_key(chunk_tokens)  # B-1의 함수 재사용
```

---

### 6. ObjectCacheS3TierRouter (Activity A-1) — `src/scheduler/objectcache_s3_tier_router.py`

```python
@dataclass
class S3TierConfig:
    """ObjectCache S3 티어 라우터 설정."""
    context_lengths: List[int]         # [4096, 8192, 16384, 32768, 65536]
    breakeven_table: Dict[int, float]  # {context_length: hit_rate_breakeven}
    hysteresis_band: float = 0.05
    ema_gamma: float = 0.9
    max_s3_requests_per_batch: int = 4
    s3_enabled_by_default: bool = False  # 테스트 환경에서 False
    rdma_bandwidth_gbps: float = 100.0   # RoCE 100 Gbps


class ObjectCacheS3TierRouter(BaseScheduler):
    """S3 오브젝트 스토리지 4번째 KV 계층 브레이크이븐 기반 동적 라우터.

    브레이크이븐 히트율 수식:
      hit_rate_breakeven = T_recompute / (T_recompute + T_s3)

    EMA 히트율 갱신:
      hit_rate_ema = γ × current_hit_rate + (1 - γ) × hit_rate_ema

    S3 티어 활성화 조건:
      hit_rate_ema ≥ breakeven(context_length) + hysteresis_band

    S3 티어 비활성화 조건:
      hit_rate_ema < breakeven(context_length) - hysteresis_band

    스케줄링 결정 단위: 요청(request) 단위.
    캐시 상태 접근: CDCContentHashSegmentIDInterface.lookup() 결과를
      배치 처리 전에 확인해 S3 라우팅 여부 결정.
    """

    def __init__(
        self,
        config: S3TierConfig,
        segment_interface: CDCContentHashSegmentIDInterface,
    ) -> None: ...

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """요청 목록을 받아 S3 라우팅 여부를 결정하고 재정렬.

        1. 각 요청에 대해 context_length 기반 breakeven_table 조회
        2. EMA 히트율과 브레이크이븐 비교로 s3_tier_active 플래그 설정
        3. 배치 내 S3 요청 수 상한(max_s3_requests_per_batch) 적용
        4. S3 라우팅 요청에 InferenceRequest.metadata["s3_tier"] = True 설정
        5. 재정렬: S3 미라우팅 요청 먼저 처리 (캐시 워밍 효과)
        """
        ...

    def update_hit_rate_ema(self, current_hit_rate: float) -> None:
        """EMA 히트율 갱신. O(1)."""
        ...

    def compute_breakeven_hit_rate(
        self,
        t_recompute_ms: float,
        t_s3_ms: float,
    ) -> float:
        """브레이크이븐 히트율 계산.
        hit_rate_breakeven = t_recompute / (t_recompute + t_s3)
        """
        ...

    def get_breakeven_for_context(self, context_length: int) -> float:
        """컨텍스트 길이에 따른 브레이크이븐 히트율 조회 (룩업 테이블)."""
        ...

    @property
    def s3_tier_active(self) -> bool:
        """현재 S3 티어 활성화 여부."""
        ...
```

---

### 7. IrminsulObjectCachePipeline (Cross-1, A+B) — `src/engine/irminsul_objectcache_pipeline.py`

```python
class IrminsulObjectCachePipeline:
    """Irminsul(B-1) + ObjectCache(A-1) A+B 통합 파이프라인.

    처리 흐름:
      Step 1 (B-1): CDC 청킹 + segment_id = SHA256(chunk_tokens)
      Step 2 (B-2): CDCContentHashSegmentIDInterface.lookup(segment_id)
                    → HBM/DRAM/SSD 히트 시 즉시 반환
      Step 3 (A-1): S3 전체 미스 시 ObjectCacheS3TierRouter가 브레이크이븐 확인
                    → 활성화 시 S3 모의 조회 (실제 환경: layerwise RDMA)
      Step 4 (B-1): MLA 모델: apply_delta_rotation(k_r, delta) → 위치 수정
                    GQA/MHA: AdapShot RoPE 재인코딩 fallback
      Step 5: 수정된 (c_kv, k_r_corrected) 반환 → 어텐션 입력

    InferenceRunner와 호환: get_segments() API 구현으로
    runner.py의 `if hasattr(cache, "get_segments")` 분기에서 동작.
    """

    def __init__(
        self,
        arch_router: ArchitectureAwareNonContiguousRouter,
        s3_router: ObjectCacheS3TierRouter,
        segment_interface: CDCContentHashSegmentIDInterface,
        config: IrminsulObjectCachePipelineConfig,
    ) -> None: ...

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """InferenceRunner 호환 API.
        SegmentedHashCache.get_segments()와 동일한 반환 형식.
        """
        ...

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """InferenceRunner 호환 API."""
        ...
```

---

### 8. MLATwoAxisCompressionCodec (Activity C-1, 선택적) — `src/cache/mla_two_axis_compression_codec.py`

```python
@dataclass
class MLATwoAxisConfig:
    depth_sharing_threshold: float = 0.90  # 레이어 간 c_KV 유사도 임계값
    depth_sharing_k: int = 2               # k=2이면 50% 레이어 공유
    position_dedup_enabled: bool = True    # 위치 축 중복 제거
    fallback_threshold: float = 0.95       # accuracy delta > 1% 시 상향값


class MLATwoAxisCompressionCodec(CacheStore):
    """MLA c_KV 위치 축(Irminsul 재사용 지분) × 깊이 축(레이어 공유) 2축 압축 코덱.

    위치 축 압축:
      - 에이전틱 세션에서 동일 segment_id의 c_KV는 포인터만 유지 (중복 제거)
      - c_KV 위치-자유성으로 정확도 손실 없음 (수학적 보장)

    깊이 축 압축:
      - cos_sim(c_kv[l], c_kv[l-1]) >= depth_sharing_threshold 이면
        레이어 l의 c_KV를 레이어 l-1로 공유 (포인터)
      - 임계값 이하 레이어는 독립 유지

    2축 결합 절감율:
      total_reduction = 1 - (1 - pos_reduction) × (1 - depth_reduction)

    accuracy-preserving 설계:
      - 위치 축: c_KV 위치-자유성으로 수학적 무손실
      - 깊이 축: 유사도 임계값 ≥ 0.90에서만 공유
      - Fallback: accuracy delta > 1% 시 depth_sharing_threshold → fallback_threshold

    CacheStore 인터페이스 완전 구현.
    """

    def __init__(
        self,
        base_cache: IrminsulMLASegmentCache,
        config: MLATwoAxisConfig,
    ) -> None: ...

    # CacheStore 추상 메서드
    def put(self, key: str, value: torch.Tensor) -> None: ...
    def get(self, key: str) -> Optional[torch.Tensor]: ...
    def evict(self) -> int: ...
    def hit_rate(self) -> float: ...
    def memory_bytes(self) -> int: ...
    def reset_stats(self) -> None: ...

    def compress_layer_kv(
        self,
        c_kv_by_layer: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], float]:
        """레이어별 c_KV 딕셔너리를 깊이 축 공유로 압축.

        Returns:
          compressed: 중복 제거된 레이어 → c_KV 매핑 (포인터 기반)
          depth_reduction: 절감율 (0.0~1.0)
        """
        ...

    def depth_sharing_reduction_rate(
        self,
        c_kv_by_layer: Dict[int, torch.Tensor],
    ) -> float:
        """현재 레이어 KV 세트에서의 깊이 축 절감율 계산."""
        ...
```

---

## Activity C — Accuracy Preservation 검증 계획

(Activity C-1 `MLATwoAxisCompressionCodec` 포함)

### perplexity 측정

- **데이터셋**: WikiText-2 (test split, 2,048 토큰 시퀀스 단위)
- **모델**: DeepSeek-V2-Lite 또는 동등 MLA 모델 (테스트 환경에서는 d_c=512, k_r_dim=64 합성 모델)
- **허용 오차**: perplexity 변화 ±1% 이내
- **비교 기준**:
  - Baseline: 압축 없는 전체 c_KV
  - 위치 축 단독 (position_dedup_enabled=True, depth_sharing_threshold=1.0)
  - 깊이 축 단독 (position_dedup_enabled=False, depth_sharing_threshold=0.90)
  - 2축 결합 (position_dedup_enabled=True, depth_sharing_threshold=0.90)

### 태스크 정확도 측정

- **벤치마크**: LongBench 8개 서브태스크 (NarrativeQA, Qasper, MultiFieldQA-EN/ZH, HotpotQA, 2WikiMultihopQA, GovReport, QMSum 등)
- **허용 오차**: ±1% 이내
- **에이전틱 세션 반복 토큰 비율 측정**: 위치 축 절감율 실측 (목표: 에이전틱 세션에서 최대 48%)

### depth_sharing_threshold 스윕

- threshold ∈ {0.85, 0.90, 0.95, 1.00}
- 각 threshold에서 (memory_reduction, perplexity_delta, task_accuracy_delta) 3-way 측정
- accuracy delta > 1% 감지 시 threshold를 fallback_threshold(0.95)로 자동 상향

### Fallback 메커니즘

```python
def auto_adjust_threshold(
    self,
    accuracy_delta: float,
    max_allowed_delta: float = 0.01,
) -> bool:
    """accuracy delta > 1% 시 depth_sharing_threshold를 fallback_threshold로 상향.
    Returns True if threshold was adjusted.
    """
    if abs(accuracy_delta) > max_allowed_delta:
        self.config.depth_sharing_threshold = self.config.fallback_threshold
        return True
    return False
```

### 검증 테스트 파일

`tests/unit/test_compression_accuracy.py` — 기존 파일에 다음 케이스 추가:

```python
def test_mla_two_axis_position_axis_lossless():
    """위치 축 중복 제거가 정확도 손실 없음을 검증 (c_KV 위치-자유성 수학적 보장)."""
    ...  # 동일 segment_id의 c_KV 두 번 저장 시 포인터만 추가, 텐서 동일

def test_mla_two_axis_depth_axis_cosine_threshold():
    """깊이 축: cos_sim >= 0.90 레이어만 공유, 이하 독립 유지."""
    ...

def test_mla_two_axis_combined_reduction():
    """2축 결합 절감율이 위치 축 × 깊이 축 곱연산과 일치."""
    ...

def test_mla_two_axis_fallback_threshold():
    """accuracy delta > 1% 시 threshold 자동 상향 동작 검증."""
    ...
```

---

## 설정 파라미터

```yaml
# configs/experiments/2026-05-26.yaml
experiment:
  date: "2026-05-26"
  activity: "A+B+C_synergy"
  cache_type: "irminsul_mla_arch_aware"
  compression_method: "mla_two_axis"   # Activity C 시너지
  scheduler_type: "objectcache_s3_tier"

# Activity B-1 Irminsul MLA 설정
irminsul_mla:
  avg_chunk_size: 256
  min_chunk_size: 64
  max_chunk_size: 1024
  rope_base: 10000.0
  k_r_dim: 64
  max_entries: 2000
  seed: 42

# Activity A-1 ObjectCache S3 브레이크이븐 설정
objectcache_s3_tier:
  context_lengths: [4096, 8192, 16384, 32768, 65536]
  hysteresis_band: 0.05
  ema_gamma: 0.9
  max_s3_requests_per_batch: 4
  s3_enabled_by_default: false  # 테스트 환경 기본값
  rdma_bandwidth_gbps: 100.0

# Activity C-1 MLA 2축 압축 설정 (선택적)
mla_two_axis:
  depth_sharing_threshold: 0.90
  depth_sharing_k: 2
  position_dedup_enabled: true
  fallback_threshold: 0.95
  max_allowed_accuracy_delta: 0.01
```

```yaml
# configs/arch_registry.yaml
architectures:
  - model_pattern: "deepseek-v2*"
    arch: "MLA"
    kv_lora_rank: 512
    qk_rope_head_dim: 64
  - model_pattern: "deepseek-v3*"
    arch: "MLA"
    kv_lora_rank: 512
    qk_rope_head_dim: 64
  - model_pattern: "kimi*"
    arch: "MLA"
    kv_lora_rank: 512
    qk_rope_head_dim: 64
  - model_pattern: "llama*"
    arch: "GQA"
  - model_pattern: "gpt*"
    arch: "MHA"
  - model_pattern: "default"
    arch: "MHA"
```

```yaml
# configs/objectcache_breakeven_table.yaml
# T_recompute / (T_recompute + T_s3) per context length
# 값은 objectcache_breakeven_calibration.py로 측정해 채운다.
# 테스트 환경 기본값 (ObjectCache 원논문 5.6% TTFT 기준 역산):
breakeven_table:
  4096:  0.15
  8192:  0.18
  16384: 0.22
  32768: 0.28
  65536: 0.35

s3_endpoint: "http://localhost:9000"   # MinIO 테스트 엔드포인트
s3_bucket: "kvcache"
rdma_target: "gpu-node-0"
```

---

## 테스트 요구사항

### 필수 단위 테스트

- [ ] `tests/unit/test_irminsul_mla_segment_cache.py`
  - `test_cdc_chunking_avg_size()`: 평균 청크 크기가 avg_chunk_size ± 50% 이내
  - `test_cdc_segment_key_position_independence()`: 동일 토큰 다른 위치에서 동일 segment_id
  - `test_delta_rotation_correctness()`: `assert_delta_rotation_correctness()` 호출, 수학적 일치 검증
  - `test_mla_put_get_segment()`: put_mla_segment → get_mla_segment_with_delta_rotation 왕복
  - `test_mla_cache_store_interface()`: CacheStore 추상 메서드 전부 동작 확인
  - `test_mla_noncontiguous_hit_rate()`: 비연속 히트 카운팅 정확성
  - `test_mla_lru_eviction()`: max_entries 초과 시 LRU 퇴거 동작

- [ ] `tests/unit/test_arch_aware_noncontiguous_router.py`
  - `test_detect_mla_arch()`: kv_lora_rank + qk_rope_head_dim=64 → "MLA" 감지
  - `test_detect_gqa_arch()`: num_kv_heads=4 → "GQA" 감지
  - `test_detect_mha_arch()`: 기본 설정 → "MHA" 감지
  - `test_arch_registry_override()`: arch_registry.yaml 모델명 매핑 우선 적용
  - `test_mla_route_uses_irminsul()`: MLA 모델에서 IrminsulMLASegmentCache 경로 사용
  - `test_gqa_mha_route_fallback()`: GQA/MHA 모델에서 기존 경로 fallback
  - `test_cache_store_interface()`: CacheStore 추상 메서드 전부 동작

- [ ] `tests/unit/test_cdc_content_hash_interface.py`
  - `test_segment_id_equals_sha256()`: make_segment_id가 SHA256(token_bytes)와 일치
  - `test_hbm_hit_returns_first()`: HBM 히트 시 즉시 반환, S3 미조회
  - `test_tier_waterfall_order()`: HBM miss → DRAM miss → SSD miss → S3 조회 순서
  - `test_s3_unavailable_fallback()`: S3 없을 때 "miss" 반환 (예외 없이)
  - `test_s3_object_key_format()`: model_name/segment_id_layer.kvcache 형식 검증
  - `test_segment_id_dedup()`: 동일 segment_id 두 번 저장 시 중복 없음

- [ ] `tests/unit/test_objectcache_s3_tier_router.py`
  - `test_breakeven_formula()`: compute_breakeven_hit_rate(T_r, T_s3) 수식 검증
  - `test_ema_update()`: update_hit_rate_ema γ=0.9 EMA 계산 정확성
  - `test_s3_activation_above_threshold()`: hit_rate >= breakeven + hysteresis → s3_tier_active=True
  - `test_s3_deactivation_below_threshold()`: hit_rate < breakeven - hysteresis → s3_tier_active=False
  - `test_hysteresis_prevents_oscillation()`: breakeven 경계 근처에서 토글 없음 (히스테리시스 밴드)
  - `test_max_s3_requests_per_batch()`: 배치 내 S3 요청 수 max_s3_requests_per_batch 상한 적용
  - `test_schedule_returns_list()`: BaseScheduler.schedule() 반환 타입 List[InferenceRequest]
  - `test_s3_disabled_default_no_crash()`: s3_enabled_by_default=False 시 예외 없이 동작

### 필수 통합 테스트

- [ ] `tests/integration/test_irminsul_objectcache_pipeline_e2e.py`
  - `test_pipeline_mla_full_flow()`: CDC → B-2 계층 조회 → A-1 S3 라우팅 결정 → B-1 δ-회전 → 반환
  - `test_pipeline_noncontiguous_hit_rate_above_30pct()`: 에이전틱 세션 시뮬레이션에서 비연속 히트율 ≥ 30%
  - `test_pipeline_s3_breakeven_respected()`: EMA 히트율 < 브레이크이븐 시 S3 미활성화 확인
  - `test_pipeline_gqa_fallback_works()`: GQA 모델 설정에서 기존 RoPEReencodingCache 경로 사용
  - `test_pipeline_inference_runner_compat()`: InferenceRunner가 IrminsulObjectCachePipeline을
    `cache`로 받아 get_segments/put_segment API를 통해 정상 동작

### Activity C 검증 테스트 (기존 파일 추가)

- [ ] `tests/unit/test_compression_accuracy.py` — 다음 케이스 추가:
  - `test_mla_two_axis_position_axis_lossless()`
  - `test_mla_two_axis_depth_axis_cosine_threshold()`
  - `test_mla_two_axis_combined_reduction()`
  - `test_mla_two_axis_fallback_threshold()`
  - `test_mla_two_axis_cache_store_interface()`

---

## 완료 기준 (Definition of Done)

1. **단위 테스트 100% 통과**: 위 명시된 모든 단위 테스트 케이스 통과
2. **통합 테스트 100% 통과**: `test_irminsul_objectcache_pipeline_e2e.py` 전체 통과
3. **기존 테스트 회귀 없음**: 이전 사이클 구현 파일의 모든 단위·통합 테스트 계속 통과
4. **evaluation_criteria.md §3 (Activity B)** 기준 충족:
   - 비연속 세그먼트 히트율 전체 히트의 30% 이상 (에이전틱 세션 시뮬레이션 기준)
   - 전체 Cache Hit Rate 베이스라인 대비 +5%p 이상
5. **evaluation_criteria.md §2 (Activity A)** 기준 충족:
   - 스케줄링 오버헤드 TTFT p50 +5% 이내
   - EMA 히트율 < 브레이크이븐 시 S3 미활성화 (불필요한 원격 지연 방지)
6. **CacheStore 인터페이스 준수**: IrminsulMLASegmentCache, ArchitectureAwareNonContiguousRouter,
   MLATwoAxisCompressionCodec 모두 추상 메서드 전부 구현
7. **(C-1 포함 시) evaluation_criteria.md §4 Accuracy 보존 필수 항목 충족**:
   - perplexity 변화 ±1% 이내
   - depth_sharing_threshold ≥ 0.90 기준에서 WikiText-2 perplexity delta 검증
8. **설정 파일 존재**: `configs/experiments/2026-05-26.yaml` 생성됨
9. **시드 고정 재현성**: seed=42로 동일 결과 재현 가능

---

## 구현 우선순위 순서

1. B-1: `IrminsulMLASegmentCache` + `apply_delta_rotation` + `cdc_chunk`
2. B-1: `ArchitectureAwareNonContiguousRouter` + `detect_attention_arch`
3. A-1: `ObjectCacheS3TierRouter` + 브레이크이븐 계산
4. B-2: `CDCContentHashSegmentIDInterface`
5. Cross-1: `IrminsulObjectCachePipeline`
6. C-1: `MLATwoAxisCompressionCodec` (선택적)
7. 설정 파일: `configs/experiments/2026-05-26.yaml`, `configs/arch_registry.yaml`,
              `configs/objectcache_breakeven_table.yaml`
8. 단위 테스트 전부
9. 통합 테스트

---

## 보존 파일 (수정 금지)

이전 사이클 구현 파일은 수정하지 않는다:
- `src/cache/vericache_speculative_codec.py`
- `src/cache/kv_packet.py`
- `src/cache/kv_packet_adapter.py`
- `src/engine/speculative_packet_pipeline.py`
- `src/cache/segmented.py`
- `src/cache/rope_reencoding_cache.py`
- `src/scheduler/dualpath_nic_load_balancer.py`
- 기타 모든 이전 사이클 파일
