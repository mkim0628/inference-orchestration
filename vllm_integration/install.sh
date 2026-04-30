#!/bin/bash
# install.sh — Install the latest vLLM and verify the A+B+C integration package.
#
# Usage:
#   bash vllm_integration/install.sh
#
# This script:
#   1. Upgrades vLLM to the latest available version (no version pinning).
#   2. Prints the installed version for record-keeping.
#   3. Runs smoke tests for all three activities (A, B, C).

set -euo pipefail

echo "=== Installing latest vLLM ==="
pip install --upgrade vllm

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo "vLLM version: ${VLLM_VERSION}"

echo ""
echo "=== Smoke-testing vllm_integration imports (A+B+C) ==="
python - <<'PYEOF'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch

# --- Activity C: HadamardInt4Codec ---
from vllm_integration.compression_codec import HadamardInt4Codec, CompressionCodec

codec_int4 = HadamardInt4Codec(num_layers=32, cutoff_ratio=0.2)
codec_int8 = CompressionCodec(num_layers=32)

kv = torch.randn(8, 64)
for layer_idx in [0, 10, 20, 31]:
    enc = codec_int4.encode(kv, layer_idx, tensor_id=0)
    dec = codec_int4.decode(enc, layer_idx, tensor_id=0)
    assert dec.shape == kv.shape, f"HadamardInt4Codec shape mismatch at layer {layer_idx}"
print("Activity C (HadamardInt4Codec): OK")

# --- Activity B: NonContiguousKVCacheManager + CompressedSegmentIndex ---
from vllm_integration.block_manager_patch import (
    SegmentHashMixin,
    CompressedSegmentIndex,
    NonContiguousKVCacheManager,
)

index = CompressedSegmentIndex(codec=codec_int4, max_entries=100)
key = SegmentHashMixin.get_segment_key([1, 2, 3, 4, 5, 6, 7, 8], chunk_idx=0, layer_idx=0, chunk_size=8)
assert len(key) == 64, "Expected 64-char hex key"
kv8 = torch.randn(8, 64)
index.put(key, kv8, layer_idx=5, tensor_id=0)
retrieved = index.get(key, tensor_id=0)
assert retrieved is not None and retrieved.shape == kv8.shape
print("Activity B (SegmentHashMixin + CompressedSegmentIndex): OK")

# --- Activity A: CacheHitAwareRequestQueue ---
from vllm_integration.scheduler_patch import (
    CacheHitAwareRequestQueue,
    create_cache_hit_aware_queue,
)

queue = create_cache_hit_aware_queue(segment_index=index, chunk_size=8)
assert len(queue) == 0
print("Activity A (CacheHitAwareRequestQueue): OK")

# --- Attention backend hook ---
from vllm_integration.attention_backend_patch import (
    CompressedKVHook,
    NonContiguousAttentionWrapper,
)
hook = CompressedKVHook(codec_int4)
kv_3d = torch.randn(8, 4, 64)
enc = hook.encode_kv(kv_3d, layer_idx=5, is_key=True)
dec = hook.decode_kv(enc, layer_idx=5, is_key=True)
assert dec.shape == kv_3d.shape
print("Activity B+C attention hook: OK")

print(f"\nAll A+B+C smoke tests passed.  vLLM version: {__import__('vllm').__version__}")
PYEOF

echo ""
echo "=== Installation complete ==="
