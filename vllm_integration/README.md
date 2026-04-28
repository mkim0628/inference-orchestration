# vllm_integration — Activity B+C KV Cache Port

## Overview

This package ports the independently-verified B+C KV cache algorithm
(30.3 % non-contiguous hit rate, −68.8 % memory) from the standalone
`src/cache/` implementation into **vLLM 0.20.0**.

| Activity | Description | Source |
|----------|-------------|--------|
| **B** | Position-independent segmented hash cache | `src/cache/segmented.py` |
| **C** | Mixed-precision KV quantization (FP16/INT8) | `src/cache/compression.py` |
| **B+C** | Compressed non-contiguous segment cache | `src/cache/compressed_segment.py` |

---

## vLLM Version

| Field | Value |
|-------|-------|
| vLLM version | **0.20.0** |
| Install command | `pip install --upgrade vllm` |
| Architecture | v1 (vllm.v1.*) |
| KV cache manager | `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| Attention backend | `vllm.v1.attention.backend.AttentionImpl` |

---

## File Map

```
vllm_integration/
├── __init__.py                  Package marker
├── compression_codec.py         Activity C: FP16/INT8 mixed-precision codec
├── block_manager_patch.py       Activity B: Position-independent segment index
│                                + NonContiguousKVCacheManager subclass
├── attention_backend_patch.py   Activity B+C: CompressedKVHook + wrapper
├── scheduler_patch.py           Activity A: stub (not in scope this cycle)
├── install.sh                   Install script
└── README.md                    This file
```

---

## vLLM Integration Points

### Activity B — Non-Contiguous KV Cache Reuse

**Primary integration file:** `block_manager_patch.py`

vLLM 0.20.0 uses a v1 KV cache architecture. The relevant classes are:

- `vllm.v1.core.kv_cache_manager.KVCacheManager` — manages block allocation
  and prefix cache lookup via `get_computed_blocks`.
- `vllm.v1.core.kv_cache_utils.hash_block_tokens` — hashes token blocks for
  prefix caching.

Our extension `NonContiguousKVCacheManager` subclasses `KVCacheManager` and
adds:

1. `SegmentHashMixin.get_segment_key(token_ids, chunk_idx, layer_idx)` — a
   position-independent SHA-256 hash of a fixed-size token chunk. The key
   depends only on token *content*, not absolute position, enabling reuse
   when the same tokens appear at different offsets.

2. `CompressedSegmentIndex` — an LRU dictionary mapping segment keys to
   compressed KV tensors. Backed by `CompressionCodec` (Activity C).

3. `NonContiguousKVCacheManager.get_computed_blocks` — overrides the parent to
   additionally query the segment index after the standard prefix cache lookup.

4. `store_segment` / `lookup_segment` — API for the attention backend to
   populate and query the index.

### Activity C — KV Cache Compression

**Primary integration file:** `compression_codec.py`, `attention_backend_patch.py`

The `CompressionCodec` implements mixed-precision storage:

- **Early layers** (first ~1/3): FP16 — critical attention patterns, 50 %
  savings vs FP32.
- **Later layers** (remaining ~2/3): symmetric per-tensor INT8 — 75 % savings
  vs FP32, ~68.8 % overall memory reduction (measured in Report ①).

`CompressedKVHook` wraps the codec with `encode_kv` / `decode_kv` methods
designed to be called at the write and read points in the attention pipeline:

- **Write**: call `encode_kv` immediately after K/V projection, before storing.
- **Read**: call `decode_kv` immediately before passing K/V to the attention
  kernel — kernels always receive native float32 tensors.

### vLLM v1 Attention Backend Integration

vLLM 0.20.0 attention backends live under `vllm.v1.attention.backends.*`.
The main abstract class is `vllm.v1.attention.backend.AttentionImpl` with the
key method `forward`.

`NonContiguousAttentionWrapper` wraps any `AttentionImpl` instance and
provides helpers `store_kv_chunks` (write path) and `load_cached_chunks` (read
path). Full production integration requires subclassing the specific backend
used by the model (e.g. `FlashAttentionImpl`).

---

## How to Apply Patches

### 1. Install

```bash
bash vllm_integration/install.sh
```

Or manually:

```bash
pip install --upgrade vllm
python -c "import vllm; print(vllm.__version__)"
```

### 2. Substitute the KV Cache Manager

```python
from vllm_integration.compression_codec import CompressionCodec
from vllm_integration.block_manager_patch import NonContiguousKVCacheManager

codec = CompressionCodec(num_layers=model.config.num_hidden_layers)

# Pass additional kwargs to NonContiguousKVCacheManager;
# all standard KVCacheManager kwargs are forwarded.
kv_manager = NonContiguousKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    codec=codec,
    segment_chunk_size=64,
    segment_max_entries=2000,
)
```

### 3. Wire Compression Hooks into the Attention Layer

```python
from vllm_integration.attention_backend_patch import (
    CompressedKVHook,
    NonContiguousAttentionWrapper,
)

hook = CompressedKVHook(codec)
wrapped_attn = NonContiguousAttentionWrapper(
    impl=original_attn_impl,
    hook=hook,
    kv_manager=kv_manager,
    chunk_size=64,
)

# In the forward pass (prefill):
wrapped_attn.store_kv_chunks(token_ids, k, v, layer_idx)

# In the forward pass (decode / re-prefill):
hit_chunks, miss_chunks = wrapped_attn.load_cached_chunks(token_ids, layer_idx)
```

---

## Compatibility Notes

| Component | Status |
|-----------|--------|
| vLLM 0.20.0 (v1 engine) | Tested |
| `KVCacheManager` public API | Preserved (subclass, no monkey-patching) |
| `AttentionImpl.forward` | Delegated (wrapper is pass-through) |
| `SchedulerConfig` fields | Not modified (Activity A stub only) |
| GPU memory layout | Follows vLLM block_size; no cross-block segments |

---

## Performance Expectations (from Report ①)

| Metric | Standalone result | vLLM port target |
|--------|-------------------|-----------------|
| Non-contiguous cache hit rate | 30.3 % | ≥ 30 % |
| KV cache memory reduction | −68.8 % | ≥ −30 % (goal) |
| Compression accuracy delta | ±0.3 % | ≤ ±1 % |

---

## Cycle

- Date: 2026-04-28
- Loop: 1 / 3
- Activity: B+C
- Report ①: `reports/evaluations/2026-04-28.md`
