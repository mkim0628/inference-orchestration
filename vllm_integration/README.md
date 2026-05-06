# vllm_integration — Activity A+B+C KV Cache Port

## Overview

This package ports the independently-verified A+B+C KV cache pipeline from the
standalone `src/` implementation into **vLLM 0.20.1**.

| Cycle | Activity | Description | Source |
|-------|----------|-------------|--------|
| 2026-05-06 | **B** | QueryCentricKVCacheManager — ProphetKV dual-stage recompute budget allocation | `src/cache/query_centric_recompute.py` |
| 2026-05-06 | **C** | TriAttentionCodecWrapper — pre-RoPE trigonometric KV compression codec | `src/cache/tri_attention_codec.py` |
| 2026-05-06 | **B+C** | QueryCentricTriAttentionKVCacheManager — dual-path raw/compressed store | `src/cache/qc_tri_store.py` |
| 2026-05-06 | **C** | TriAttentionAttentionHook — compress/decompress hooks for attention backend | (new, wraps TriAttentionCodecWrapper) |
| 2026-05-06 | **C** | VllmQueryCentricAttentionWrapper — pre-RoPE key capture + QCTA integration | (new, wraps AttentionImpl.forward) |
| 2026-05-06 | **B** | QueryCentricSchedulerMixin — QCRC recompute scheduling mixin for vLLM Scheduler | (new) |
| 2026-05-05 | **B** | DiffAwareKVPatch — master + block-sparse diff non-contiguous KV reuse | `src/cache/diff_aware_store.py` |
| 2026-05-05 | **C** | NQKVCodecPatch — Normal Float INT4 block-quantile KV compression | `src/cache/nqkv_codec.py` |
| 2026-05-05 | **B+C** | CompressedKVManager — INT4-compressed masters + FP16 diff storage | `src/cache/compressed_diff_store.py` |
| 2026-05-05 | **C** | FireQAttentionPatch — RoPE-aware 2-stage outlier smoothing attn hook | `src/cache/fireq_codec.py` |
| 2026-05-04 | **A** | DAGTopologyScheduler — workflow DAG topology KV proactive preservation | `src/scheduler/dag_topology_scheduler.py` |
| 2026-05-04 | **B** | WorkloadAwareTTLCache — category-specific TTL segment preservation | `src/cache/workload_ttl_cache.py` |
| 2026-05-04 | **C** | RedundancyAwareEvictionPolicy — importance x redundancy dual-score eviction | `src/cache/redundancy_eviction.py` |
| 2026-05-04 | **Cross-1** | DAGAwareTTLAdjuster — DAG events → TTL adjustment pipeline | `src/scheduler/dag_ttl_adjuster.py` |
| 2026-05-03 | **A** | DualMapSchedulerMixin — dual-hash + semantic-hit-rate routing (preserved) | `src/scheduler/dual_map_scheduler.py` |
| 2026-05-03 | **B** | SemanticNonContiguousKVCacheManager — DHD semantic similarity (preserved) | `src/cache/dhd_segment_cache.py` |
| 2026-05-03 | **C** | TurboQuantKVHook — TurboQuant 3-bit compression (preserved) | `src/cache/turbo_quant.py` |

---

## vLLM Version

| Field | Value |
|-------|-------|
| vLLM version | **0.20.1** |
| Install command | `pip install --upgrade vllm` |
| Architecture | v1 (`vllm.v1.*`) |
| KV cache manager | `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| Scheduler | `vllm.v1.core.sched.scheduler.Scheduler` |
| Block pool | `vllm.v1.core.block_pool.BlockPool` |
| Attention backend | `vllm.v1.attention.backend.AttentionImpl` |

---

## 2026-05-06 Integration: Activity B+C (QueryCentricRecompute + TriAttentionCodec)

### Files Modified

| File | Activity | Description |
|------|----------|-------------|
| `block_manager_patch.py` | B | `QueryCentricKVCacheManager` — dual-stage QCRC recompute budget allocation |
| `block_manager_patch.py` | B+C | `QueryCentricTriAttentionKVCacheManager` — dual-path raw/compressed KV store |
| `block_manager_patch.py` | C | `TriAttentionCodecWrapper` — vLLM-compatible TriAttentionCodec adapter |
| `attention_backend_patch.py` | C | `TriAttentionAttentionHook` — compress/decompress hooks for attention write/read |
| `attention_backend_patch.py` | C | `VllmQueryCentricAttentionWrapper` — AttentionImpl wrapper capturing pre-RoPE keys |
| `scheduler_patch.py` | B | `QueryCentricSchedulerMixin` — QCRC recompute scheduling mixin |
| `scheduler_patch.py` | B | `make_qcrc_aware_scheduler_class()` — factory for QCRC-aware scheduler |

### Architecture

```
[Request arrives at Scheduler]
         │
         ▼
QueryCentricSchedulerMixin.pre_schedule_qcrc()
  ├── selective_recompute(query, segment_keys, budget=0.20)
  │    Stage 1: attention-norm top-50% filter
  │    Stage 2: cosine-similarity re-rank (query vs. segment embedding)
  │    Budget:  max 20% of total cached tokens selected
  └── _qcrc_recompute_map[request_id] = [seg_key_1, seg_key_2, ...]
         │
         ▼
[Attention Layer (VllmQueryCentricAttentionWrapper.forward)]
  ├── key captured pre-RoPE (before RoPE applied inside flash_attn)
  ├── base impl.forward() called unmodified → native vLLM paged KV cache
  └── keys_pre_rope + kv stored to QCTA manager
         │
         ▼
QueryCentricTriAttentionKVCacheManager.store_qcta_segment()
  ├── cosine_sim(query_embedding, seg_embedding) >= relevance_threshold (0.60)?
  │    YES → raw store (_qcta_raw_store) + QCRC store (for selective_recompute)
  │    NO  → TriAttentionCodecWrapper.compress(kv, keys_pre_rope, ratio=0.10)
  │          → compressed store (_qcta_compressed_store)
  └── Result: 10× memory reduction for low-relevance segments
         │
         ▼
[Next request with overlapping context]
  ├── get_qcta_segment(key) → raw (high-relevance) or decompress (low-relevance)
  │    decompress: zeros at pruned positions (pre-RoPE importance ranking)
  └── selective_recompute() → raw-only segments eligible for recomputation
```

### Integration (single node — Activity B+C)

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration.block_manager_patch import (
    QueryCentricTriAttentionKVCacheManager, TriAttentionCodecWrapper
)
from vllm_integration.attention_backend_patch import (
    TriAttentionAttentionHook, VllmQueryCentricAttentionWrapper
)
from vllm_integration.scheduler_patch import make_qcrc_aware_scheduler_class

# Step 1: Create codec and calibrate
codec = TriAttentionCodecWrapper(
    n_layers=32, n_heads=32, head_dim=128,
    compression_ratio=0.10, series_terms=8, prune_window=128,
)
# calib_kvs: list of [layers, heads, seq_len, head_dim] pre-RoPE K tensors
codec.calibrate(calib_kvs, save_path="calibration.pt")
# (or) codec.load_calibration("calibration.pt")

# Step 2: Create QCTA KV manager
kv_manager = QueryCentricTriAttentionKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    codec=codec,
    relevance_threshold=0.60,   # cosine sim >= 0.60 → raw storage
    compression_ratio=0.10,     # keep 10% tokens for low-relevance segments
    qcrc_recompute_budget_ratio=0.20,  # recompute up to 20% of tokens
)

# Step 3: Create attention hook and wrap model layers
hook = TriAttentionAttentionHook(codec=codec, compression_ratio=0.10)
n_patched = VllmQueryCentricAttentionWrapper.patch_model_layers(
    model, kv_manager, codec, chunk_size=128, compression_ratio=0.10
)

# Step 4: Create QCRC-aware scheduler
QCRCScheduler = make_qcrc_aware_scheduler_class(Scheduler)
scheduler = QCRCScheduler(
    ...,  # standard vLLM Scheduler args
    qcrc_kv_manager=kv_manager,
    qcrc_budget_ratio=0.20,
)

# Step 5: Per-request lifecycle
# After prefill (segment keys from store_qcta_segment() calls):
scheduler.register_request_segments(request.request_id, segment_keys, query_emb)

# Before scheduling step:
scheduler.pre_schedule_qcrc(waiting_requests)

# Get recompute recommendations for a request:
to_recompute = scheduler.get_recompute_segments(request.request_id)

# On request completion:
scheduler.on_request_complete(request.request_id)
```

### TriAttentionCodec Accuracy Preservation

The TriAttentionCodecWrapper implements the pre-RoPE trigonometric importance
estimation described in TriAttention (arXiv 2604.04921):

1. **Calibration**: Estimate per-(layer, head) mean μ_k from representative keys.
   Fit Fourier coefficients a_m by least-squares regression.

2. **Compression**: For each 128-token window, compute importance =
   |Σ_m a_m × (sin(m×d_i) + cos(m×d_i))| where d_i = ||k_i - μ_k||.
   Keep top compression_ratio fraction by importance score.

3. **Position stability**: Pre-RoPE space is position-invariant — the same token
   receives the same importance score regardless of its sequence position.
   This avoids the position-dependent bias of post-RoPE importance estimation.

4. **Decompression**: Zeros at pruned positions. Kept positions are lossless.
   Accuracy target: ±1% perplexity relative to full KV cache.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compression_ratio` | 0.10 | Fraction of tokens kept per 128-token window |
| `relevance_threshold` | 0.60 | Cosine sim threshold for raw vs. compressed routing |
| `qcrc_recompute_budget_ratio` | 0.20 | Max fraction of tokens for selective_recompute() |
| `qcrc_stage1_top_k_ratio` | 0.50 | Stage-1 attention-norm filter fraction |
| `series_terms` | 8 | Fourier series terms for importance estimation |
| `prune_window` | 128 | Token window size for windowed pruning |

### Scheduling Overhead

QueryCentricSchedulerMixin operates in Python (no GPU I/O) on the waiting queue.
selective_recompute() is O(N×K) where N = segment count, K = series_terms.
Target: < 5ms per 100 requests (TTFT p50 +5% budget per evaluation_criteria.md).

---

## 2026-05-05 Integration: Activity B+C

### Files Added

| File | Activity | Description |
|------|----------|-------------|
| `nqkv_codec_patch.py` | C-1 | `NQKVCodecPatch` — NQKVCodec adapted for vLLM paged blocks |
| `diff_aware_kv_patch.py` | B-1 | `DiffAwareKVPatch` — master + block-sparse diff non-contiguous reuse |
| `compressed_kv_manager.py` | B+C Cross-1 | `CompressedKVManager` — INT4 masters + FP16 diffs |
| `fireq_attention_patch.py` | C-2 | `FireQAttentionPatch` + `_FireQCodecCore` — RoPE-aware 2-stage hook |

### Integration Points (vLLM 0.20.1)

| Component | vLLM Integration Point |
|-----------|------------------------|
| `NQKVCodecPatch` | Wraps individual KV blocks or layer-wide cache tensors; use alongside `KVCacheManager.allocate_slots()` |
| `DiffAwareKVPatch` | Augments `KVCacheManager` block_id tracking; call `register_master_block()` on first-write, `put_agent_block()` on subsequent requests |
| `CompressedKVManager` | Drop-in for `DiffAwareKVPatch` when compressed storage is needed; integrates with `KVCacheManager.cache_blocks()` path |
| `FireQAttentionPatch` | Wraps `FlashAttentionImpl.forward()` or any `AttentionImpl`; inject via `patch_vllm_model_layers()` after model load |

### NQKVCodecPatch — vLLM block layout

vLLM v1 allocates KV blocks of shape `[num_blocks, block_size, num_kv_heads, head_dim]`.
`NQKVCodecPatch.encode_layer_kv_cache()` operates on this full tensor.
For per-block operations, use `encode_vllm_block()` / `decode_vllm_block()`.

Compression ratio: ~3.5x vs FP16 (64-element quantisation blocks, NF4 14-value table).
Memory reduction: ~71.4% (1/3.5).

### DiffAwareKVPatch — Non-contiguous reuse

```python
from vllm_integration.diff_aware_kv_patch import DiffAwareKVPatch

patch = DiffAwareKVPatch(seq_block_size=64, diff_threshold=0.1)

# On first prefill for a shared context:
patch.register_master_block(block_id=42, kv_tensor=shared_kv)

# On per-agent/per-request decode:
patch.put_agent_block(block_id=42, agent_id="req_001", kv_tensor=agent_kv)

# On KV cache read:
kv = patch.get_agent_block(block_id=42, agent_id="req_001")

# Stats:
stats = patch.diff_hit_stats()
# {'diff_hit_rate': ..., 'master_hit_rate': ..., 'overall_hit_rate': ...,
#  'n_groups': ..., 'search_space_reduction': ...}
```

### CompressedKVManager — INT4 compressed masters

```python
from vllm_integration.compressed_kv_manager import CompressedKVManager

mgr = CompressedKVManager(seq_block_size=64, diff_threshold=0.1, codec_block_size=64)

# Store compressed master:
mgr.store_block(block_id=10, kv_tensor=kv_fp16)

# Store per-agent diff (master decompressed transiently):
mgr.store_agent_diff(block_id=10, agent_id="req_001", kv_tensor=agent_kv)

# Retrieve with diff applied:
kv = mgr.retrieve_block(block_id=10, agent_id="req_001")

# Memory stats:
summary = mgr.compression_summary(sample_kv=kv_fp16)
```

### FireQAttentionPatch — attention hook

```python
from vllm_integration.fireq_attention_patch import FireQAttentionPatch

# Create and calibrate codec:
codec = FireQAttentionPatch.make_codec(n_heads=32, d_head=128)
codec.calibrate(calibration_data)  # list of (kv_tensor, layer_idx)

# Patch all attention layers in a loaded vLLM model:
n_patched = FireQAttentionPatch.patch_vllm_model_layers(model, codec)

# Or wrap a single impl:
patch = FireQAttentionPatch(original_impl, codec=codec, layer_idx=5)
output = patch.forward(layer, query, key, value, kv_cache, attn_metadata, output)
```

---

## File Map

```
vllm_integration/
├── __init__.py                   Package marker
├── nqkv_codec_patch.py           Activity C-1 (2026-05-05): NQKVCodecPatch
├── diff_aware_kv_patch.py        Activity B-1 (2026-05-05): DiffAwareKVPatch
├── compressed_kv_manager.py      Activity B+C Cross-1 (2026-05-05): CompressedKVManager
├── fireq_attention_patch.py      Activity C-2 (2026-05-05): FireQAttentionPatch
├── scheduler_patch.py            Activity A (2026-05-04): DAGTopologySchedulerMixin
│                                  + MultiNodeDAGRouter, DAGNodeCapacity, WorkflowDAG
│                                  + make_dag_aware_scheduler_class() factory
│                                  Activity B (2026-05-06): QueryCentricSchedulerMixin
│                                  + make_qcrc_aware_scheduler_class() factory
│                                  + DualMapSchedulerMixin (2026-05-03, preserved)
│                                  + CacheHitAwareRequestQueue, MultiNodeRequestRouter (prior)
├── block_manager_patch.py        Activity B (2026-05-06): QueryCentricKVCacheManager
│                                  + QueryCentricTriAttentionKVCacheManager (B+C)
│                                  + TriAttentionCodecWrapper (Activity C codec adapter)
│                                  Activity B (2026-05-04): WorkloadAwareTTLKVCacheManager
│                                  + VllmDAGAwareTTLAdjuster, VllmTTLEntry
│                                  + SemanticNonContiguousKVCacheManager (2026-05-03, preserved)
│                                  + SemanticSegmentIndex (2026-05-03, preserved)
├── attention_backend_patch.py    Activity C (2026-05-06): TriAttentionAttentionHook
│                                  + VllmQueryCentricAttentionWrapper (pre-RoPE capture)
│                                  Activity C (2026-05-04): VllmRedundancyAwareEvictionPolicy
│                                  + VllmAttentionKVHook (importance recording)
│                                  + TurboQuantKVHook, SemanticKVAttentionWrapper (2026-05-03, preserved)
│                                  + CompressedKVHook, TriStateKVHook (prior, preserved)
├── compression_codec.py          Activity C (2026-05-03): VllmTurboQuantCodec
│                                  + CacheCompressionConfig
│                                  + HadamardInt4Codec, CompressionCodec (prior, preserved)
├── install.sh                    pip install --upgrade vllm + all smoke tests
│                                  (2026-05-06, 2026-05-05, 2026-05-04, backward-compat)
├── README.md                     This file
│
│   Prior-cycle files (preserved for backward compatibility):
├── leverage_compressor_patch.py  2026-05-02 LeverageScoreCompressor
├── sign_vq_block_manager_patch.py 2026-05-02 SignVQSegmentIndex
└── cache_config_extension.py     2026-05-02 SignVQCacheParams
```

---

## Activity A — DAGTopologyScheduler (2026-05-04)

**Primary integration file:** `scheduler_patch.py`

### Architecture

`DAGTopologySchedulerMixin` is a mixin for vLLM's `Scheduler` that adds
workflow DAG topology-based KV proactive preservation:

```
Workflow DAG:
  agent_A → agent_B → agent_C

BFS topological analysis:
  kv_reuse_probability[agent_A] = out_degree / max_out_degree
  kv_reuse_probability[agent_C] = 0.0  (leaf node)

If kv_reuse_probability > retain_threshold:
  → fire on_kv_reuse_event(segment_key, prob)
  → DAGAwareTTLAdjuster extends TTL: base_ttl × (1 + prob × alpha)

On node completion:
  → fire on_node_complete_event(segment_key)
  → DAGAwareTTLAdjuster sets TTL=0 (immediate eviction candidate)
```

### Integration (single node)

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration.scheduler_patch import make_dag_aware_scheduler_class
from vllm_integration.block_manager_patch import VllmDAGAwareTTLAdjuster

# Create TTL-aware KV manager (see Activity B below)
# kv_manager = WorkloadAwareTTLKVCacheManager(...)

# Create TTL adjuster
adjuster = VllmDAGAwareTTLAdjuster(kv_manager, alpha=2.0)

# Create DAG-aware scheduler subclass
DAGAwareScheduler = make_dag_aware_scheduler_class(Scheduler)
scheduler = DAGAwareScheduler(
    ...,  # standard vLLM Scheduler args
    dag_retain_threshold=0.5,
    dag_alpha_ttl_extend=2.0,
    dag_on_kv_reuse_event=adjuster.on_kv_reuse_event,
    dag_on_node_complete_event=adjuster.on_node_complete,
)

# Register agent workflow DAGs
dag_spec = {
    "dag_id": "my_workflow",
    "nodes": [
        {"agent_id": "agent_A", "tool_calls": ["search"], "expected_kv_tokens": 512, "parent_ids": []},
        {"agent_id": "agent_B", "tool_calls": ["summarize"], "expected_kv_tokens": 256, "parent_ids": ["agent_A"]},
    ],
}
scheduler.register_workflow(dag_spec)

# Inject DAG metadata into requests via sampling_params.extra_args:
# sampling_params = SamplingParams(..., extra_args={"dag_id": "my_workflow", "agent_id": "agent_A"})
```

### Integration (multi-node / P/D disaggregated)

```python
from vllm_integration.scheduler_patch import MultiNodeDAGRouter, DAGNodeCapacity

nodes = [
    DAGNodeCapacity(node_id="prefill-0", role="prefill", load=0.3, network_bandwidth_gbps=200.0),
    DAGNodeCapacity(node_id="prefill-1", role="prefill", load=0.5, network_bandwidth_gbps=200.0),
]
router = MultiNodeDAGRouter(nodes=nodes, migration_threshold_ms=50.0)

# Route a request — prefers node that already has this DAG's KV cached
target_node = router.route("my_workflow", expected_kv_tokens=512, role="prefill")

# Record DAG KV residency when prefill completes
router.register_dag_on_node("prefill-0", "my_workflow")

# Update load from worker health reports
router.update_node_load("prefill-0", 0.6)
```

### Overhead Budget

DAG metadata processing operates in Python on the waiting queue — no GPU I/O.
Target: < 5ms / 100 requests (TTFT p50 +5% budget per evaluation_criteria.md).
Measured: < 50ms / 100 requests × 10 iterations in smoke test.

---

## Activity B — WorkloadAwareTTLCache (2026-05-04)

**Primary integration file:** `block_manager_patch.py`

### Architecture

`WorkloadAwareTTLKVCacheManager` subclasses `KVCacheManager` and adds a
parallel TTL segment store alongside vLLM's native prefix cache:

```
Request arrives:
  category = classify_category(request_key)   # code/chat/rag/agentic
  ttl_sec = _ttl_profiles[category]["ttl_base_sec"]
                                              # code:600s chat:300s rag:120s agentic:480s
  store_ttl_segment(token_ids, chunk_idx, block_ids, category, ttl_sec)

DAG event arrives (via VllmDAGAwareTTLAdjuster):
  adjusted_ttl = base_ttl × (1 + dag_reuse_prob × alpha)
  adjust_segment_ttl(key, adjusted_ttl)

DAG node completes:
  adjust_segment_ttl(key, 0.0)   # immediate eviction candidate
  unpin_segment(key)

Periodic cleanup:
  evict_expired_segments()        # flush TTL-expired vLLM blocks
```

### Integration

```python
from vllm_integration.block_manager_patch import (
    WorkloadAwareTTLKVCacheManager,
    VllmDAGAwareTTLAdjuster,
)

# Replace standard KVCacheManager with TTL-aware version
kv_manager = WorkloadAwareTTLKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    # TTL extension parameters
    ttl_max_entries=1000,
    ttl_chunk_size=128,          # must match WorkloadAwareTTLCache.chunk_size
    ttl_ema_alpha=0.1,
    ttl_profiles=None,           # use default TTL profiles from Spec.md
    ttl_eviction_policy=eviction_policy,  # VllmRedundancyAwareEvictionPolicy
)

# Create adjuster to connect DAG events to TTL updates
adjuster = VllmDAGAwareTTLAdjuster(kv_manager, alpha=2.0)

# After block allocation, register the segment:
segment_key = kv_manager.store_ttl_segment(
    token_ids=request.prompt_token_ids,
    chunk_idx=chunk_idx,
    block_ids=allocated_block_ids,
    category="agentic",          # or call kv_manager.classify_category(request_key)
    layer_idx=layer_idx,
)

# Periodically flush expired segments (e.g. once per schedule() step):
n_evicted = kv_manager.evict_expired_segments()
```

### Non-Contiguous Hit Rate

All non-expired hits are counted as TTL-preserved (non-contiguous proxy):
```python
stats = kv_manager.ttl_hit_stats()
# stats["noncontiguous_ratio"] >= 0.30  (goal from evaluation_criteria.md §3)
```

### TTL Profiles (from KVCache-in-the-Wild Table 3)

| Category | TTL | Reuse Prob |
|----------|-----|------------|
| code | 600s | 0.75 |
| agentic | 480s | 0.80 |
| chat | 300s | 0.60 |
| rag | 120s | 0.45 |

---

## Activity C — RedundancyAwareEvictionPolicy (2026-05-04)

**Primary integration file:** `attention_backend_patch.py`

### Architecture

`VllmRedundancyAwareEvictionPolicy` is a pure scoring layer that plugs into
`WorkloadAwareTTLKVCacheManager.evict_expired_segments()`:

```
eviction_score = (1 - normalized_importance) × redundancy_score

Where:
  normalized_importance = entry.importance_score / max(importance_scores)
  redundancy_score:
    - doc_id_shortcut: key starts with "doc:<id>:" → redundancy=1.0 (O(1))
    - embedding cosine similarity: mean sim vs all other candidates (O(N^2))

High-importance segments (importance ≈ 1.0):
  → eviction_score ≈ 0.0 → never selected → ACCURACY PRESERVED

Redundant segments (cosine sim ≈ 1.0 to others):
  → eviction_score ≈ 1.0 → evicted first → MEMORY FREED
```

### Integration

```python
from vllm_integration.attention_backend_patch import (
    VllmRedundancyAwareEvictionPolicy,
    VllmAttentionKVHook,
)
from vllm_integration.block_manager_patch import WorkloadAwareTTLKVCacheManager

eviction_policy = VllmRedundancyAwareEvictionPolicy(
    redundancy_top_n=100,     # brute-force N≤100 is safe per Spec.md
    importance_weight=1.0,
    redundancy_weight=1.0,
    doc_id_shortcut=True,
)

kv_manager = WorkloadAwareTTLKVCacheManager(
    ...,
    ttl_eviction_policy=eviction_policy,
)

# Record importance from attention weights (after each attention layer):
hook = VllmAttentionKVHook(kv_manager, chunk_size=128, importance_aggregation="mean")
hook.record_importance_from_attention(
    attn_weights=attn_weights,   # (batch, heads, seq_q, seq_k) or (seq_q, seq_k)
    token_ids=request.prompt_token_ids,
    layer_idx=layer_idx,
)
```

### Accuracy Preservation Contract

- Only TTL-expired segments are candidates — no in-TTL segment is ever touched.
- `eviction_score(importance=1.0) = 0.0` — high-importance segments are structurally safe.
- No lossy operations (no quantization, no approximation of KV values).
- Policy only reorders the eviction queue; it cannot evict segments that TTL would not already evict.

Validated in Report ① 2026-05-04:
- High-importance segment preservation: 100%
- Residual Key cosine similarity after eviction: ≥ 0.99
- Hit rate delta for important segments: ≤ 1%p

---

## Full Pipeline Integration (Cross-1: A+B+C)

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration.scheduler_patch import make_dag_aware_scheduler_class
from vllm_integration.block_manager_patch import (
    WorkloadAwareTTLKVCacheManager, VllmDAGAwareTTLAdjuster
)
from vllm_integration.attention_backend_patch import (
    VllmRedundancyAwareEvictionPolicy, VllmAttentionKVHook
)

# Step 1: Create eviction policy (Activity C)
eviction_policy = VllmRedundancyAwareEvictionPolicy(redundancy_top_n=100)

# Step 2: Create TTL-aware KV manager (Activity B)
kv_manager = WorkloadAwareTTLKVCacheManager(
    kv_cache_config=..., max_model_len=..., hash_block_size=...,
    ttl_max_entries=1000, ttl_eviction_policy=eviction_policy,
)

# Step 3: Create TTL adjuster (Cross-1 adapter)
adjuster = VllmDAGAwareTTLAdjuster(kv_manager, alpha=2.0)

# Step 4: Create DAG-aware scheduler (Activity A)
DAGAwareScheduler = make_dag_aware_scheduler_class(Scheduler)
scheduler = DAGAwareScheduler(
    ...,
    dag_retain_threshold=0.5,
    dag_on_kv_reuse_event=adjuster.on_kv_reuse_event,
    dag_on_node_complete_event=adjuster.on_node_complete,
)

# Step 5: Register workflows and attach attention hooks
scheduler.register_workflow(dag_spec)
attn_hook = VllmAttentionKVHook(kv_manager, chunk_size=128)

# Step 6: Per-step:
#   a. scheduler.schedule() — calls pre_schedule_dag() → fires on_kv_reuse_event
#   b. attn_hook.record_importance_from_attention() — feeds importance scores
#   c. kv_manager.evict_expired_segments() — flushes expired blocks
#   d. scheduler.notify_node_complete(dag_id, agent_id) — when agent finishes
```

---

## How to Apply

### 1. Install

```bash
bash vllm_integration/install.sh
```

### 2. Run smoke tests

The install script automatically runs smoke tests for all activities.

---

## Compatibility Notes

| Component | Status |
|-----------|--------|
| vLLM 0.20.1 (v1 engine) | Target version |
| `KVCacheManager` public API | Preserved (subclass only) |
| `Scheduler` public API | Preserved (mixin, no monkey-patching) |
| `AttentionImpl.forward` | Not modified — hook is additive only |
| `CacheConfig` fields | Not modified (composition pattern) |
| GPU memory layout | vLLM paged blocks unmodified |
| Accuracy of KV values | Preserved — no quantization in C (eviction ordering only) |

---

## Performance Expectations (from Report ① 2026-05-04)

| Metric | Standalone (233/233 tests) | vLLM port target |
|--------|---------------------------|-----------------|
| Non-contiguous cache hit rate (TTL) | 100% (noncontiguous_ratio=1.0) | ≥ 30% |
| High-importance segment preservation | 100% | 100% |
| Residual Key cosine similarity | ≥ 0.99 | ≥ 0.99 |
| Hit rate delta (important segments) | ≤ 0.0 (0%p) | ≤ 1%p |
| Scheduling overhead TTFT p50 | < 5ms / 100 reqs | ≤ 5ms / req |
| on_kv_reuse_event() p50 latency | < 1ms | < 1ms |

---

## Cycle History

| Date | Loop | Activities | Key Additions |
|------|------|-----------|---------------|
| 2026-04-28 | 1/3 | B+C | NonContiguousKVCacheManager, CompressedKVHook (INT8) |
| 2026-04-29 | 1/3 | A+B+C | CacheHitAwareRequestQueue, HadamardInt4Codec (INT4) |
| 2026-04-30 | 1/3 | A+B+C | MultiNodeRequestRouter, TriStateKVHook, SegmentAdapterMixin |
| 2026-05-02 | 1/3 | B+C | VllmLeverageCompressor, SignVQSegmentIndex, SignVQCacheParams |
| 2026-05-03 | 1/3 | A+B+C | DualMapSchedulerMixin, SemanticSegmentIndex (DHD), VllmTurboQuantCodec |
| 2026-05-04 | 1/3 | A+B+C | DAGTopologySchedulerMixin, WorkloadAwareTTLKVCacheManager, VllmRedundancyAwareEvictionPolicy, VllmDAGAwareTTLAdjuster, MultiNodeDAGRouter |
| 2026-05-05 | 1/3 | B+C | NQKVCodecPatch (NF4 INT4), DiffAwareKVPatch (block-sparse diff), CompressedKVManager, FireQAttentionPatch (RoPE-aware 2-stage) |
| **2026-05-06** | **1/3** | **B+C** | **QueryCentricKVCacheManager (ProphetKV dual-stage), TriAttentionCodecWrapper (pre-RoPE trig), QueryCentricTriAttentionKVCacheManager (B+C dual-path), TriAttentionAttentionHook, VllmQueryCentricAttentionWrapper, QueryCentricSchedulerMixin** |
