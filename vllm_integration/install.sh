#!/bin/bash
# install.sh — Install the latest vLLM and verify the integration package.
#
# Usage:
#   bash vllm_integration/install.sh
#
# This script:
#   1. Upgrades vLLM to the latest available version (no version pinning).
#   2. Prints the installed version for record-keeping.
#   3. Runs a quick smoke-test import to confirm the integration package loads.

set -euo pipefail

echo "=== Installing latest vLLM ==="
pip install --upgrade vllm

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo "vLLM version: ${VLLM_VERSION}"

echo ""
echo "=== Smoke-testing vllm_integration imports ==="
python - <<'PYEOF'
import sys, pathlib
# Ensure the repo root is on the path when running from within the repo.
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from vllm_integration.compression_codec import CompressionCodec
from vllm_integration.block_manager_patch import (
    SegmentHashMixin,
    CompressedSegmentIndex,
    NonContiguousKVCacheManager,
)
from vllm_integration.attention_backend_patch import (
    CompressedKVHook,
    NonContiguousAttentionWrapper,
)

# Quick functional check
codec = CompressionCodec(num_layers=32)
hook = CompressedKVHook(codec)

import torch
kv = torch.randn(8, 4, 64)  # [seq, heads, head_dim]
enc = hook.encode_kv(kv, layer_idx=0, is_key=True)
dec = hook.decode_kv(enc, layer_idx=0, is_key=True)
assert dec.shape == kv.shape, "decode shape mismatch"

key = SegmentHashMixin.get_segment_key([1, 2, 3, 4], chunk_idx=0, layer_idx=0)
assert len(key) == 64, "expected 64-char hex key"

print("All integration imports and smoke tests passed.")
print(f"vLLM version: {__import__('vllm').__version__}")
PYEOF

echo ""
echo "=== Installation complete ==="
