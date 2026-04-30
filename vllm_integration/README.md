# vllm_integration — Activity A+B+C KV Cache Port

## Overview

This package ports the independently-verified A+B+C KV cache pipeline from the
standalone `src/` implementation into **vLLM 0.20.0**.

| Activity | Description | Source |
|----------|-------------|--------|
| **A** | Cache-hit-rate-aware request scheduling | `src/scheduler/cache_aware_scheduler.py` |
| **B** | Position-independent segmented hash cache | `src/cache/segmented.py` |
| **C** | Hadamard INT4 + mixed-precision KV compression | `src/cache/compression.py` |
| **A+B+C** | Full pipeline: scheduler + NC reuse + compression | `src/cache/compressed_segment.py` |

---

## vLLM Version

| Field | Value |
|-------|-------|
| vLLM version | **0.20.0** |
| Install command | `pip install --upgrade vllm` |
| Architecture | v1 (vllm.v1.*) |
| KV cache manager | `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| Request queue | `vllm.v1.core.sched.request_queue.RequestQueue` |
| Attention backend | `vllm.v1.attention.backend.AttentionImpl` |

---

## File Map

```
vllm_integration/
├── __init__.py                  Package marker
├── compression_codec.py         Activity C: HadamardInt4Codec (new) + CompressionCodec (prior)
├── block_manager_patch.py       Activity B: Position-independent segment index
│                                + NonContiguousKVCacheManager subclass
├── attention_backend_patch.py   Activity B+C: CompressedKVHook + wrapper
├── scheduler_patch.py           Activity A: CacheHitAwareRequestQueue
├── install.sh                   Install + smoke-test script
└── README.md                    This file
```

---

## vLLM Integration Points

### Activity A — Cache-Hit-Aware Scheduling

**Primary integration file:** `scheduler_patch.py`

vLLM 0.20.0 v1 scheduler uses a pluggable `RequestQueue` (defined in
`vllm.v1.core.sched.request_queue`). The default policy is FCFS or priority.

`CacheHitAwareRequestQueue` subclasses `RequestQueue` and maintains a
priority heap that orders requests by predicted KV segment hit rate:

```
priority = hit_rate × (1 − min(wait_steps / fairness_max_wait, 1.0))
```

Hit rate prediction peeks at `CompressedSegmentIndex._store` keys without
calling `get()`, so cache statistics are not polluted.  A wait-step penalty
prevents cold requests from being indefinitely starved.

**How to enable:**
```python
from vllm_integration.scheduler_patch import create_cache_hit_aware_queue

# In Scheduler.__init__, replace default request queue:
self.waiting = create_cache_hit_aware_queue(
    segment_index=kv_manager._segment_index,
    chunk_size=kv_manager._segment_chunk_size,
    fairness_max_wait=10,
)
```

### Activity B — Non-Contiguous KV Cache Reuse

**Primary integration file:** `block_manager_patch.py`

`NonContiguousKVCacheManager` subclasses `KVCacheManager` and adds:

1. `SegmentHashMixin.get_segment_key(token_ids, chunk_idx, layer_idx)` — a
   position-independent SHA-256 hash of a token chunk (content-only, no
   absolute position), enabling reuse when tokens appear at different offsets.

2. `CompressedSegmentIndex` — LRU dict mapping segment keys to compressed KV
   tensors (backed by `HadamardInt4Codec` by default).

3. `get_computed_blocks` override — queries the segment index after the
   standard prefix cache lookup for non-contiguous hits.

4. `store_segment` / `lookup_segment` — attention backend API.

**Default codec changed to `HadamardInt4Codec`** in this cycle (pass
`use_hadamard_int4=False` for the prior INT8 codec).

### Activity C — KV Cache Compression

**Primary integration file:** `compression_codec.py`, `attention_backend_patch.py`

**`HadamardInt4Codec`** (new in cycle 2026-04-29, recommended):
- Early layers (cutoff_ratio=0.2): FP16 — 50% savings
- Late layers: Hadamard rotation + INT4-range quantized, stored as int8 — 75% savings
- Average: ~70% memory reduction vs FP32
- Accuracy: attention KL divergence < 0.05, cosine similarity ≥ 0.95

**`CompressionCodec`** (prior cycle, reference):
- Early layers: FP16; later layers: symmetric INT8 — ~67% average savings

---

## How to Apply Patches

### 1. Install

```bash
bash vllm_integration/install.sh
```

### 2. Substitute KV Cache Manager (Activity B+C)

```python
from vllm_integration.compression_codec import HadamardInt4Codec
from vllm_integration.block_manager_patch import NonContiguousKVCacheManager

kv_manager = NonContiguousKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    use_hadamard_int4=True,   # HadamardInt4Codec (recommended)
    segment_chunk_size=64,
    segment_max_entries=2000,
)
```

### 3. Enable Cache-Hit-Aware Scheduling (Activity A)

```python
from vllm_integration.scheduler_patch import create_cache_hit_aware_queue

# During Scheduler construction:
self.waiting = create_cache_hit_aware_queue(
    segment_index=kv_manager._segment_index,
    chunk_size=64,
    fairness_max_wait=10,
)
```

### 4. Wire Compression Hooks into Attention (Activity B+C)

```python
from vllm_integration.attention_backend_patch import (
    CompressedKVHook, NonContiguousAttentionWrapper,
)

hook = CompressedKVHook(kv_manager._segment_index.codec)
wrapped = NonContiguousAttentionWrapper(
    impl=original_attn_impl, hook=hook, kv_manager=kv_manager, chunk_size=64,
)

# Prefill write path:
wrapped.store_kv_chunks(token_ids, k, v, layer_idx)

# Decode / re-prefill read path:
hit_chunks, miss_chunks = wrapped.load_cached_chunks(token_ids, layer_idx)
```

---

## Compatibility Notes

| Component | Status |
|-----------|--------|
| vLLM 0.20.0 (v1 engine) | Tested |
| `KVCacheManager` public API | Preserved (subclass, no monkey-patching) |
| `RequestQueue` interface | Fully implemented (`CacheHitAwareRequestQueue`) |
| `AttentionImpl.forward` | Delegated (wrapper is pass-through) |
| `SchedulerConfig` fields | Not modified |
| GPU memory layout | Follows vLLM block_size |

---

## Performance Expectations (from Report ①)

| Metric | Standalone (55/55 tests) | vLLM port target |
|--------|--------------------------|-----------------|
| Throughput improvement | > 10% (memory-budget test) | ≥ 10% |
| Non-contiguous cache hit rate | ≥ 30% | ≥ 30% |
| KV cache memory reduction | ≥ 70% vs FP32 | ≥ 30% (goal) |
| Compression accuracy (KL) | < 0.05 all layers | ≤ 0.05 |
| Scheduling overhead (TTFT) | ≤ 5% | ≤ 5% |

---

## Cycle

- Date: 2026-04-29
- Loop: 1 / 3
- Activity: A+B+C
- Report ①: `reports/evaluations/2026-04-29.md`
