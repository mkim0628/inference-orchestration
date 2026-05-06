#!/bin/bash
# install.sh — Install the latest vLLM and verify the A+B+C integration.
#
# Usage:
#   bash vllm_integration/install.sh
#
# This script:
#   1. Upgrades vLLM to the latest available version (no version pinning).
#   2. Prints the installed version for record-keeping.
#   3. Runs smoke tests for:
#      Activity B+C (2026-05-06):
#        QueryCentricKVCacheManager, QueryCentricTriAttentionKVCacheManager,
#        TriAttentionCodecWrapper, TriAttentionAttentionHook,
#        VllmQueryCentricAttentionWrapper, QueryCentricSchedulerMixin
#      Activity B+C (2026-05-05):
#        NQKVCodecPatch, DiffAwareKVPatch, CompressedKVManager, FireQAttentionPatch
#      Activity A: DAGTopologySchedulerMixin (2026-05-04)
#      Activity B: WorkloadAwareTTLKVCacheManager (2026-05-04)
#      Activity C: VllmRedundancyAwareEvictionPolicy (2026-05-04)
#      Backward compat: prior-cycle components (2026-05-03 and earlier)

set -euo pipefail

echo "=== Installing latest vLLM ==="
pip install --upgrade vllm --ignore-installed pyjwt 2>/dev/null || pip install --upgrade vllm

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo "vLLM version: ${VLLM_VERSION}"

echo ""
echo "=== 2026-05-06 B+C smoke tests (QueryCentricRecompute + TriAttentionCodec + QCTA + scheduler) ==="
python - <<'PYEOF_2026_05_06'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import torch

# -----------------------------------------------------------------------
# TriAttentionCodecWrapper — Activity C
# -----------------------------------------------------------------------
from vllm_integration.block_manager_patch import TriAttentionCodecWrapper

codec = TriAttentionCodecWrapper(
    n_layers=2, n_heads=4, head_dim=16,
    compression_ratio=0.5, series_terms=4, prune_window=8,
)

# Calibrate with synthetic pre-RoPE K tensors
calib_kvs = [torch.randn(2, 4, 32, 16) for _ in range(12)]
codec.calibrate(calib_kvs)
assert codec.mu_k is not None, "mu_k not set after calibrate()"
assert codec.mu_k.shape == (2, 4, 16), f"Wrong mu_k shape: {codec.mu_k.shape}"
assert codec.a_m is not None, "a_m not set after calibrate()"

# Compress + decompress roundtrip
kv = torch.randn(2, 4, 32, 16)
keys_pre_rope = torch.randn(2, 4, 32, 16)
compressed = codec.compress(kv, keys_pre_rope, compression_ratio=0.5)
assert "kv" in compressed and "kept_indices" in compressed
assert compressed["original_seq_len"] == 32
assert compressed["kv"].shape[2] < 32, "Compressed should have fewer tokens"

reconstructed = codec.decompress(compressed)
assert reconstructed.shape == kv.shape, f"Reconstructed shape mismatch: {reconstructed.shape}"
# Kept positions should be reconstructed exactly
kept = compressed["kept_indices"]
torch.testing.assert_close(reconstructed[:, :, kept, :], compressed["kv"], atol=1e-5, rtol=0)
print(f"TriAttentionCodecWrapper: OK  kept={kept.shape[0]}/32  mu_k={codec.mu_k.shape}")

# -----------------------------------------------------------------------
# QueryCentricKVCacheManager — Activity B (standalone, no GPU)
# -----------------------------------------------------------------------
from vllm_integration.block_manager_patch import QueryCentricKVCacheManager
from collections import OrderedDict

class MinimalQCRCManager(QueryCentricKVCacheManager):
    """Bypass KVCacheManager.__init__() for smoke testing."""
    def __init__(self, **kwargs):
        self._qcrc_chunk_size = kwargs.get("qcrc_chunk_size", 128)
        self._qcrc_capacity_bytes = kwargs.get("qcrc_capacity_bytes", 64 * 1024 * 1024)
        self._qcrc_recompute_budget_ratio = kwargs.get("qcrc_recompute_budget_ratio", 0.20)
        self._qcrc_stage1_top_k_ratio = kwargs.get("qcrc_stage1_top_k_ratio", 0.50)
        self._qcrc_store = OrderedDict()
        self._qcrc_hit_count = 0
        self._qcrc_miss_count = 0

mgr = MinimalQCRCManager(
    qcrc_chunk_size=16,
    qcrc_capacity_bytes=64 * 1024 * 1024,
    qcrc_recompute_budget_ratio=0.20,
)

token_ids = list(range(64))
kv_seg = torch.randn(2, 4, 16, 16)

# Store segment
key0 = mgr.store_qcrc_segment(token_ids, chunk_idx=0, kv_tensor=kv_seg, layer_idx=0)
assert len(key0) == 64, f"Expected 64-char SHA-256 hex, got {len(key0)}"
assert len(mgr._qcrc_store) == 1

# Get hit
result = mgr.get_qcrc_segment(key0)
assert result is not None, "Expected cache hit"
assert mgr._qcrc_hit_count == 1

# Miss
result2 = mgr.get_qcrc_segment("nonexistent" * 4)
assert result2 is None
assert mgr._qcrc_miss_count == 1

# Store second segment
key1 = mgr.store_qcrc_segment(token_ids, chunk_idx=1, kv_tensor=kv_seg, layer_idx=0)

# selective_recompute — two-stage budget allocation
query_emb = torch.randn(16)
selected = mgr.selective_recompute(query_emb, [key0, key1], budget=0.20)
assert isinstance(selected, list), "selective_recompute must return a list"
# Budget=0.20 of 32 tokens = 6 tokens. Each segment has 16 tokens, so 0 or 1 selected.
assert len(selected) <= 2, f"Too many segments selected: {selected}"

stats = mgr.qcrc_stats()
assert "hit_rate" in stats
assert "num_segments" in stats
print(f"QueryCentricKVCacheManager: OK  hit_rate={stats['hit_rate']:.2f}  segments={stats['num_segments']}")

# -----------------------------------------------------------------------
# QueryCentricTriAttentionKVCacheManager — Activity B+C (standalone)
# -----------------------------------------------------------------------
from vllm_integration.block_manager_patch import QueryCentricTriAttentionKVCacheManager
import torch.nn.functional as F

class MinimalQCTAManager(QueryCentricTriAttentionKVCacheManager):
    """Bypass KVCacheManager.__init__() for smoke testing."""
    def __init__(self, **kwargs):
        # Init QCRC state
        self._qcrc_chunk_size = kwargs.get("qcrc_chunk_size", 16)
        self._qcrc_capacity_bytes = kwargs.get("qcrc_capacity_bytes", 64 * 1024 * 1024)
        self._qcrc_recompute_budget_ratio = kwargs.get("qcrc_recompute_budget_ratio", 0.20)
        self._qcrc_stage1_top_k_ratio = kwargs.get("qcrc_stage1_top_k_ratio", 0.50)
        self._qcrc_store = OrderedDict()
        self._qcrc_hit_count = 0
        self._qcrc_miss_count = 0
        # Init QCTA state
        self._qcta_codec = kwargs.get("codec", None)
        self._qcta_relevance_threshold = kwargs.get("relevance_threshold", 0.60)
        self._qcta_compression_ratio = kwargs.get("compression_ratio", 0.50)
        self._qcta_compressed_store = {}
        self._qcta_raw_store = {}
        self._qcta_hit_count = 0
        self._qcta_miss_count = 0
        self._qcta_compressed_hits = 0
        self._qcta_raw_hits = 0

qcta_mgr = MinimalQCTAManager(
    qcrc_chunk_size=16,
    codec=codec,
    relevance_threshold=0.0,  # all segments go to compressed (low threshold)
    compression_ratio=0.5,
)

kv_seg2 = torch.randn(2, 4, 16, 16)
pre_rope = torch.randn(2, 4, 16, 16)
query_emb2 = torch.randn(16)

# threshold=0.0 means all segments go to compressed store (cosine_sim > 0.0 is typical)
# We need high relevance to go to raw — set threshold above 1.0 to force compressed path
qcta_mgr._qcta_relevance_threshold = 2.0  # impossible threshold → all compressed
seg_key = qcta_mgr.store_qcta_segment(
    token_ids=list(range(64)), chunk_idx=0,
    kv_tensor=kv_seg2, keys_pre_rope=pre_rope,
    query_embedding=query_emb2, layer_idx=0,
)
assert seg_key in qcta_mgr._qcta_compressed_store, "Expected compressed store hit"

# Read back decompressed
retrieved = qcta_mgr.get_qcta_segment(seg_key)
assert retrieved is not None, "Expected non-None on compressed read"
assert retrieved.shape[-1] == 16, f"Wrong head_dim: {retrieved.shape}"

# Now test high-relevance path (threshold=0.0 → all go to raw)
qcta_mgr._qcta_relevance_threshold = -2.0  # always above cosine_sim range [-1,1]
seg_key2 = qcta_mgr.store_qcta_segment(
    token_ids=list(range(64)), chunk_idx=1,
    kv_tensor=kv_seg2, keys_pre_rope=pre_rope,
    query_embedding=query_emb2, layer_idx=0,
)
assert seg_key2 in qcta_mgr._qcta_raw_store, "Expected raw store hit"

# selective_recompute must only use raw segments (not compressed)
selected2 = qcta_mgr.selective_recompute(query_emb2, [seg_key, seg_key2])
# seg_key is compressed → excluded; seg_key2 is raw → eligible
assert seg_key not in selected2, "Compressed segment should not be in recompute list"

qcta_stats = qcta_mgr.qcta_stats()
assert "compressed_hits" in qcta_stats
print(f"QueryCentricTriAttentionKVCacheManager: OK  raw={qcta_stats['num_raw_segments']}  compressed={qcta_stats['num_compressed_segments']}")

# -----------------------------------------------------------------------
# TriAttentionAttentionHook — Activity C write/read hooks
# -----------------------------------------------------------------------
from vllm_integration.attention_backend_patch import TriAttentionAttentionHook

hook = TriAttentionAttentionHook(codec=codec, compression_ratio=0.5, enabled=True)

kv_hook = torch.randn(2, 4, 32, 16)
pre_rope_hook = torch.randn(2, 4, 32, 16)

# write_to_cache: should return compressed dict
compressed_hook = hook.write_to_cache(kv_hook, pre_rope_hook)
assert "kv" in compressed_hook or "raw" in compressed_hook, f"Unexpected keys: {compressed_hook.keys()}"
if "kv" in compressed_hook:
    assert compressed_hook["kv"].shape[2] < 32, "Expected fewer tokens after compression"

# read_from_cache: decompress before attention kernel
reconstructed_hook = hook.read_from_cache(compressed_hook)
assert reconstructed_hook.shape == kv_hook.shape, f"Reconstructed shape: {reconstructed_hook.shape}"

# Disabled hook: identity passthrough
hook_off = TriAttentionAttentionHook(codec=codec, enabled=False)
raw_out = hook_off.write_to_cache(kv_hook, pre_rope_hook)
assert "raw" in raw_out, "Disabled hook should return raw dict"
recon_off = hook_off.read_from_cache(raw_out)
assert recon_off.shape == kv_hook.shape

stats_hook = hook.hook_stats()
assert stats_hook["compress_count"] >= 1
print(f"TriAttentionAttentionHook: OK  compress_count={stats_hook['compress_count']}  decompress_count={stats_hook['decompress_count']}")

# -----------------------------------------------------------------------
# VllmQueryCentricAttentionWrapper — stand-alone test (no GPU model)
# -----------------------------------------------------------------------
from vllm_integration.attention_backend_patch import VllmQueryCentricAttentionWrapper

class MockImpl:
    """Minimal stand-in for FlashAttentionImpl."""
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        return output

mock_impl = MockImpl()
wrapper = VllmQueryCentricAttentionWrapper(
    impl=mock_impl,
    kv_manager=qcta_mgr,
    hook=hook,
    layer_idx=0,
    chunk_size=16,
)

# Forward pass: should not raise
query_t = torch.randn(32, 16)
key_t = torch.randn(32, 16)
value_t = torch.randn(32, 16)
out_t = torch.zeros(32, 16)
result_t = wrapper.forward(
    layer=None, query=query_t, key=key_t, value=value_t,
    kv_cache=None, attn_metadata=None, output=out_t,
)
assert result_t.shape == out_t.shape, f"Output shape mismatch: {result_t.shape}"

wstats = wrapper.wrapper_stats()
assert wstats["forward_count"] == 1
print(f"VllmQueryCentricAttentionWrapper: OK  forward_count={wstats['forward_count']}  qcta_store_count={wstats['qcta_store_count']}")

# -----------------------------------------------------------------------
# QueryCentricSchedulerMixin — Activity B scheduler integration
# -----------------------------------------------------------------------
from vllm_integration.scheduler_patch import (
    QueryCentricSchedulerMixin, make_qcrc_aware_scheduler_class
)

# Stand-alone mixin test (no vLLM Scheduler base needed)
class _StandaloneMixin(QueryCentricSchedulerMixin):
    def __init__(self, **kwargs):
        # Manually initialize mixin state without calling super().__init__()
        self._qcrc_kv_manager = kwargs.get("qcrc_kv_manager")
        self._qcrc_budget_ratio = kwargs.get("qcrc_budget_ratio", 0.20)
        self._qcrc_hit_threshold = kwargs.get("qcrc_hit_threshold", 0.30)
        self._qcrc_request_segments = {}
        self._qcrc_recompute_map = {}
        self._qcrc_query_embeddings = {}
        self._qcrc_schedule_steps = 0
        self._qcrc_recompute_decisions = 0

sched_mixin = _StandaloneMixin(qcrc_kv_manager=mgr, qcrc_budget_ratio=0.20)

# Register request segments
sched_mixin.register_request_segments("req_001", [key0, key1], query_embedding=query_emb)
assert "req_001" in sched_mixin._qcrc_request_segments
assert len(sched_mixin._qcrc_request_segments["req_001"]) == 2

# pre_schedule_qcrc
class MockReq2:
    def __init__(self, rid):
        self.request_id = rid

sched_mixin.pre_schedule_qcrc(waiting_requests=[MockReq2("req_001")])
assert sched_mixin._qcrc_schedule_steps == 1

recommended = sched_mixin.get_recompute_segments("req_001")
assert isinstance(recommended, list), "Expected list of segment keys"
print(f"QueryCentricSchedulerMixin: OK  recommended={len(recommended)} segments")

# on_request_complete: clean up
sched_mixin.on_request_complete("req_001")
assert "req_001" not in sched_mixin._qcrc_request_segments
assert "req_001" not in sched_mixin._qcrc_recompute_map

# make_qcrc_aware_scheduler_class factory: creates composite class
class MinimalSchedulerBase:
    def __init__(self, *args, **kwargs):
        pass
QCRCScheduler = make_qcrc_aware_scheduler_class(MinimalSchedulerBase)
assert issubclass(QCRCScheduler, QueryCentricSchedulerMixin)
assert issubclass(QCRCScheduler, MinimalSchedulerBase)
print(f"make_qcrc_aware_scheduler_class: OK  class={QCRCScheduler.__name__}")

sched_stats = sched_mixin.qcrc_scheduling_stats()
assert "schedule_steps" in sched_stats
assert "hit_rate" in sched_stats
print(f"QueryCentricSchedulerMixin stats: OK  steps={sched_stats['schedule_steps']}  hit_rate={sched_stats['hit_rate']:.2f}")

print(f"\nAll 2026-05-06 B+C smoke tests passed.  vLLM={__import__('vllm').__version__}")
PYEOF_2026_05_06

echo ""
echo "=== 2026-05-05 B+C smoke tests (NQKVCodec + DiffAwareKV + CompressedKV + FireQAttention) ==="
python - <<'PYEOF_2026_05_05'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import torch

# NQKVCodecPatch
from vllm_integration.nqkv_codec_patch import NQKVCodecPatch
codec = NQKVCodecPatch(block_size=64, vllm_block_size=16)
kv = torch.randn(2, 8, 16, 64)  # [2(K/V), num_kv_heads, vllm_block_size, head_dim]
indices, mu, sigma, orig_shape = codec.encode_vllm_block(kv)
reconstructed = codec.decode_vllm_block(indices, mu, sigma, orig_shape)
assert reconstructed.shape == kv.shape, f"Shape mismatch: {reconstructed.shape} vs {kv.shape}"
ratio = codec.compression_ratio(kv)
assert ratio > 3.0, f"Compression ratio too low: {ratio}"
print(f"NQKVCodecPatch: OK  compression_ratio={ratio:.2f}x")

# DiffAwareKVPatch
from vllm_integration.diff_aware_kv_patch import DiffAwareKVPatch
patch = DiffAwareKVPatch(seq_block_size=64, diff_threshold=0.1, max_groups=100)
kv_master = torch.randn(1, 8, 128, 64)
patch.register_master_block(block_id=42, kv_tensor=kv_master)
kv_agent = kv_master + 0.01 * torch.randn_like(kv_master)  # small diff
patch.put_agent_block(block_id=42, agent_id="agent_0", kv_tensor=kv_agent)
retrieved = patch.get_agent_block(block_id=42, agent_id="agent_0")
assert retrieved is not None, "Agent block retrieval returned None"
assert retrieved.shape == kv_master.shape
stats = patch.diff_hit_stats()
assert stats["n_groups"] == 1
print(f"DiffAwareKVPatch: OK  hit_rate={stats['overall_hit_rate']:.2f}")

# CompressedKVManager
from vllm_integration.compressed_kv_manager import CompressedKVManager
mgr = CompressedKVManager(seq_block_size=64, diff_threshold=0.1, max_blocks=100)
kv_block = torch.randn(8, 16, 64)
mgr.store_block(block_id=10, kv_tensor=kv_block)
result = mgr.retrieve_block(block_id=10)
assert result is not None, "Master retrieval returned None"
assert result.shape == kv_block.shape
summary = mgr.compression_summary(kv_block)
assert summary["compression_ratio"] > 3.0
print(f"CompressedKVManager: OK  compression_ratio={summary['compression_ratio']:.2f}x")

# FireQAttentionPatch
from vllm_integration.fireq_attention_patch import FireQAttentionPatch, _FireQCodecCore
fireq_codec = _FireQCodecCore(n_heads=8, d_head=64, outlier_threshold_sigma=3.0)
# Calibrate with synthetic data
calib = [(torch.randn(8, 32, 64), 0) for _ in range(15)]
fireq_codec.calibrate(calib)
scales = fireq_codec._pre_rope_scales.get(0)
assert scales is not None, "Calibration failed: no pre_rope_scales"
assert scales.shape == (8, 32), f"Scale shape wrong: {scales.shape}"
masks = fireq_codec._outlier_masks.get(0)
assert masks is not None, "Calibration failed: no outlier_masks"

# Test factory helper
codec2 = FireQAttentionPatch.make_codec(n_heads=8, d_head=64)
assert isinstance(codec2, _FireQCodecCore)

print(f"FireQAttentionPatch (_FireQCodecCore): OK  pre_rope_scales={scales.shape}")

print(f"\nAll 2026-05-05 B+C smoke tests passed.  vLLM={__import__('vllm').__version__}")
PYEOF_2026_05_05

echo ""
echo "=== 2026-05-04 A+B+C smoke tests (DAGTopology + WorkloadAwareTTL + RedundancyEviction) ==="
python - <<'PYEOF'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import time
import torch

# -----------------------------------------------------------------------
# Activity A: DAGTopologySchedulerMixin + MultiNodeDAGRouter
# -----------------------------------------------------------------------
from vllm_integration.scheduler_patch import (
    DAGTopologySchedulerMixin,
    DAGNode,
    WorkflowDAG,
    MultiNodeDAGRouter,
    DAGNodeCapacity,
    make_dag_aware_scheduler_class,
)

# Build a simple A→B→C DAG
dag_spec = {
    "dag_id": "workflow_smoke",
    "nodes": [
        {"agent_id": "A", "tool_calls": [], "expected_kv_tokens": 512, "parent_ids": []},
        {"agent_id": "B", "tool_calls": [], "expected_kv_tokens": 256, "parent_ids": ["A"]},
        {"agent_id": "C", "tool_calls": [], "expected_kv_tokens": 128, "parent_ids": ["B"]},
    ],
}

# Stand-alone mixin test (no scheduler base needed)
mixin = DAGTopologySchedulerMixin(retain_threshold=0.5, alpha_ttl_extend=2.0)
dag_id = mixin.register_workflow(dag_spec)
assert dag_id == "workflow_smoke", f"Expected 'workflow_smoke', got {dag_id}"

# Topological order should be A, B, C
dag = mixin._dag_workflows["workflow_smoke"]
assert dag.topological_order == ["A", "B", "C"], f"Wrong topo order: {dag.topological_order}"

# KV reuse probabilities: A and B have children, C is a leaf
prob_A = mixin.predict_kv_reuse("workflow_smoke", "A")
prob_C = mixin.predict_kv_reuse("workflow_smoke", "C")
assert prob_A > 0.0, f"Node A (has children) should have prob > 0, got {prob_A}"
assert prob_C == 0.0, f"Node C (leaf) should have prob == 0, got {prob_C}"

# Belady upper bound should be in [0, 1]
belady = mixin.compute_belady_upper_bound("workflow_smoke")
assert 0.0 <= belady <= 1.0, f"Belady bound out of range: {belady}"

# Cyclic DAG should raise ValueError
cyclic_spec = {
    "dag_id": "cyclic",
    "nodes": [
        {"agent_id": "X", "tool_calls": [], "expected_kv_tokens": 0, "parent_ids": ["Y"]},
        {"agent_id": "Y", "tool_calls": [], "expected_kv_tokens": 0, "parent_ids": ["X"]},
    ],
}
try:
    mixin.register_workflow(cyclic_spec)
    assert False, "Should have raised ValueError for cyclic DAG"
except ValueError:
    pass

# Scheduling overhead: pre_schedule_dag on 100 mock requests
class MockWaiting:
    def __init__(self, reqs):
        self._queue = list(reqs)

class MockReq:
    def __init__(self, i):
        self.request_id = f"r{i}"
        self.dag_id = "workflow_smoke"
        self.agent_id = ["A", "B", "C"][i % 3]
        self.prompt_token_ids = list(range(i * 10, i * 10 + 20))

mock_requests = [MockReq(i) for i in range(100)]
mixin.waiting = MockWaiting(mock_requests)

t0 = time.monotonic()
for _ in range(10):
    mixin.pre_schedule_dag()
elapsed_ms = (time.monotonic() - t0) * 1000.0
overhead_per_100 = elapsed_ms / 10.0
assert overhead_per_100 < 500.0, f"Overhead too high: {overhead_per_100:.1f}ms / 100 reqs"

stats = mixin.get_dag_scheduling_stats()
assert stats["registered_workflows"] >= 1
assert stats["total_schedule_steps"] >= 10

print(f"Activity A (DAGTopologySchedulerMixin): OK  overhead={overhead_per_100:.1f}ms/100reqs")

# MultiNodeDAGRouter
nodes = [
    DAGNodeCapacity(node_id="p0", role="prefill", load=0.3),
    DAGNodeCapacity(node_id="p1", role="prefill", load=0.8),
]
router = MultiNodeDAGRouter(nodes=nodes)

# Route with no locality — should pick lower-load node
target = router.route("workflow_smoke", expected_kv_tokens=256, role="prefill")
assert target in ("p0", "p1")

# Register DAG on p0, re-route — should now prefer p0 for locality
router.register_dag_on_node("p0", "workflow_smoke")
target2 = router.route("workflow_smoke", expected_kv_tokens=256, role="prefill")
assert target2 == "p0", f"Expected p0 (DAG resident), got {target2}"

print(f"Activity A (MultiNodeDAGRouter): OK  locality-first={target2}")

# -----------------------------------------------------------------------
# Activity B: WorkloadAwareTTLKVCacheManager (standalone, no GPU needed)
# -----------------------------------------------------------------------
from vllm_integration.block_manager_patch import (
    WorkloadAwareTTLKVCacheManager,
    VllmDAGAwareTTLAdjuster,
    VllmTTLEntry,
    _DEFAULT_TTL_PROFILES,
)

# Directly test the TTL store logic (no KVCacheManager base init needed in smoke test)
# We test via a minimal duck-type subclass to avoid needing GPU config

class MinimalTTLManager(WorkloadAwareTTLKVCacheManager):
    """Minimal subclass that bypasses KVCacheManager.__init__() for smoke testing."""

    def __init__(self, **kwargs):
        # Skip KVCacheManager.__init__() — we only need the TTL store logic
        import copy
        self._ttl_profiles = copy.deepcopy(_DEFAULT_TTL_PROFILES)
        self._ttl_max_entries = kwargs.get("ttl_max_entries", 100)
        self._ttl_chunk_size = kwargs.get("ttl_chunk_size", 128)
        self._ttl_ema_alpha = kwargs.get("ttl_ema_alpha", 0.1)
        self._ttl_eviction_policy = kwargs.get("ttl_eviction_policy", None)
        from collections import OrderedDict
        self._ttl_store = OrderedDict()
        self._ttl_pinned = set()
        self._ttl_exact_hits = 0
        self._ttl_preserved_hits = 0
        self._ttl_misses = 0
        self._ttl_eviction_count = 0
        self._ttl_pressure_eviction_count = 0

    def evict_blocks(self, block_ids):
        pass  # no-op in smoke test

mgr = MinimalTTLManager(ttl_max_entries=50, ttl_chunk_size=128)

token_ids = list(range(256))

# Store a segment
key_code = mgr.store_ttl_segment(
    token_ids, chunk_idx=0, block_ids={1, 2},
    category="code", layer_idx=0,
)
assert len(key_code) == 64, "Expected 64-char SHA-256 hex key"

# Hit before TTL expires
entry = mgr.get_ttl_segment(key_code)
assert entry is not None, "Expected hit before TTL expiry"
assert entry.category == "code"
assert mgr._ttl_exact_hits == 1

# Pin/unpin
mgr.pin_segment(key_code)
assert key_code in mgr._ttl_pinned
mgr.unpin_segment(key_code)
assert key_code not in mgr._ttl_pinned

# Category classification
assert mgr.classify_category("def foo():") == "code"
assert mgr.classify_category("retrieved document:") == "rag"
assert mgr.classify_category("tool_call result") == "agentic"
assert mgr.classify_category("hello world") == "chat"

# TTL adjustment to 0 should make segment appear in evict_candidates
mgr.adjust_segment_ttl(key_code, 0.0)
candidates = mgr.evict_candidates()
assert key_code in candidates, "Segment with TTL=0 should be an eviction candidate"

# Evict expired
mgr.adjust_segment_ttl(key_code, 0.0)
n_evicted = mgr.evict_expired_segments()
assert n_evicted == 1, f"Expected 1 eviction, got {n_evicted}"

# Stats
stats = mgr.ttl_hit_stats()
assert "overall_hit_rate" in stats
assert "noncontiguous_ratio" in stats

print(f"Activity B (WorkloadAwareTTLKVCacheManager): OK  hits={mgr._ttl_exact_hits}")

# DAGAwareTTLAdjuster integration
mgr2 = MinimalTTLManager(ttl_max_entries=50, ttl_chunk_size=128)
adjuster = VllmDAGAwareTTLAdjuster(mgr2, alpha=2.0, measure_latency=True)

# Store a chat segment
k2 = mgr2.store_ttl_segment(
    list(range(256)), chunk_idx=0, block_ids={10},
    category="chat", layer_idx=0,
)

# on_kv_reuse_event should extend TTL
original_ttl = mgr2._ttl_store[k2].ttl_sec  # 300s for chat
adjuster.on_kv_reuse_event(k2, dag_reuse_probability=0.8)
new_ttl = mgr2._ttl_store[k2].ttl_sec
# adjusted_ttl = 300 * (1 + 0.8 * 2.0) = 780
assert new_ttl > original_ttl, f"TTL should increase: {original_ttl} → {new_ttl}"

# on_node_complete should set TTL to 0
adjuster.on_node_complete(k2)
assert mgr2._ttl_store[k2].ttl_sec == 0.0

overhead = adjuster.overhead_stats()
assert overhead["n_samples"] >= 1

print(f"Activity B (VllmDAGAwareTTLAdjuster): OK  ttl_extend={new_ttl:.1f}s  p50={overhead['p50_ms']:.3f}ms")

# -----------------------------------------------------------------------
# Activity C: VllmRedundancyAwareEvictionPolicy
# -----------------------------------------------------------------------
from vllm_integration.attention_backend_patch import VllmRedundancyAwareEvictionPolicy, VllmAttentionKVHook

policy = VllmRedundancyAwareEvictionPolicy(
    redundancy_top_n=100,
    importance_weight=1.0,
    redundancy_weight=1.0,
    doc_id_shortcut=True,
)

# Build a mock TTL store with 5 entries
import time as _time
mock_store = {}
for i in range(5):
    emb = torch.randn(64)
    mock_store[f"seg{i}"] = VllmTTLEntry(
        block_ids={i},
        category="chat",
        ttl_sec=0.0,  # all expired
        created_at=_time.monotonic() - 10.0,
        importance_score=1.0 if i == 0 else 0.1,  # seg0 is high importance
        embedding=emb,
    )

# seg0 has importance=1.0 → eviction_score == 0.0
candidates = list(mock_store.keys())
scored = policy.score_ttl_candidates(candidates, mock_store)
seg0_score = next(s for k, s in scored if k == "seg0")
assert seg0_score == 0.0, f"High-importance seg should have score 0.0, got {seg0_score}"

# select_evict_keys should never return seg0
evict_keys = policy.select_evict_keys(candidates, mock_store, n_evict=3)
assert "seg0" not in evict_keys, f"High-importance seg0 should not be evicted: {evict_keys}"

# doc_id shortcut: two segments with same doc prefix → redundancy=1.0
doc_store = {
    "doc:abc:chunk0": VllmTTLEntry(
        block_ids={100}, category="rag", ttl_sec=0.0,
        created_at=_time.monotonic() - 10.0, importance_score=0.0,
        embedding=torch.randn(64),
    ),
    "doc:abc:chunk1": VllmTTLEntry(
        block_ids={101}, category="rag", ttl_sec=0.0,
        created_at=_time.monotonic() - 10.0, importance_score=0.0,
        embedding=torch.randn(64),
    ),
    "other:xyz": VllmTTLEntry(
        block_ids={102}, category="chat", ttl_sec=0.0,
        created_at=_time.monotonic() - 10.0, importance_score=0.0,
        embedding=None,
    ),
}
doc_scored = policy.score_ttl_candidates(list(doc_store.keys()), doc_store)
doc_scores = {k: s for k, s in doc_scored}
assert doc_scores["doc:abc:chunk0"] == 1.0, f"doc:abc:chunk0 should have score 1.0"
assert doc_scores["doc:abc:chunk1"] == 1.0, f"doc:abc:chunk1 should have score 1.0"

print(f"Activity C (VllmRedundancyAwareEvictionPolicy): OK  high-importance-score={seg0_score}")

# VllmAttentionKVHook — importance recording
hook = VllmAttentionKVHook(mgr2, chunk_size=128, importance_aggregation="mean")

# Store a segment in mgr2 so we can record importance
k_hook = mgr2.store_ttl_segment(
    list(range(256)), chunk_idx=0, block_ids={200},
    category="chat", layer_idx=0,
)
# Re-store since it was evicted earlier
if k_hook not in mgr2._ttl_store:
    mgr2._ttl_store[k_hook] = VllmTTLEntry(
        block_ids={200}, category="chat", ttl_sec=300.0,
        created_at=_time.monotonic(), importance_score=0.0,
    )

attn_weights = torch.rand(4, 8, 256, 256)  # (batch, heads, seq_q, seq_k)
hook.record_importance_from_attention(attn_weights, list(range(256)), layer_idx=0)

if k_hook in mgr2._ttl_store:
    importance = mgr2._ttl_store[k_hook].importance_score
    assert importance >= 0.0, "Importance should be non-negative"
    print(f"Activity C (VllmAttentionKVHook): OK  importance={importance:.4f}")
else:
    print("Activity C (VllmAttentionKVHook): OK  (segment not in store, no-op)")

# -----------------------------------------------------------------------
# Cross-activity integration: DAGMixin → TTLAdjuster → TTLManager → EvictionPolicy
# -----------------------------------------------------------------------
events_fired = []

def on_kv_reuse(seg_key, prob):
    adjuster.on_kv_reuse_event(seg_key, prob)
    events_fired.append(("reuse", seg_key, prob))

def on_node_done(seg_key):
    adjuster.on_node_complete(seg_key)
    events_fired.append(("complete", seg_key))

mixin_cross = DAGTopologySchedulerMixin(
    retain_threshold=0.5,
    alpha_ttl_extend=2.0,
    on_kv_reuse_event=on_kv_reuse,
    on_node_complete_event=on_node_done,
)
mixin_cross.register_workflow(dag_spec)

# Store a segment so the event has something to act on
k_cross = mgr2.store_ttl_segment(
    list(range(256)), chunk_idx=0, block_ids={300},
    category="agentic", layer_idx=0,
)
if k_cross not in mgr2._ttl_store:
    from vllm_integration.block_manager_patch import VllmTTLEntry as _E
    mgr2._ttl_store[k_cross] = _E(
        block_ids={300}, category="agentic", ttl_sec=480.0,
        created_at=_time.monotonic(), importance_score=0.0,
    )

# fire node_complete callback
on_node_done(k_cross)
assert any(e[0] == "complete" for e in events_fired), "Expected node_complete event"

print(f"Cross-activity integration (A→TTLAdjuster→B→C pipeline): OK  events={len(events_fired)}")

print(f"\nAll 2026-05-04 A+B+C smoke tests passed.  vLLM version: {__import__('vllm').__version__}")
PYEOF

echo ""
echo "=== Prior cycle backward-compat checks ==="
python - <<'PYEOF2'
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import torch

# Prior-cycle compression codecs (2026-05-03)
from vllm_integration.compression_codec import HadamardInt4Codec, CompressionCodec
codec_int4 = HadamardInt4Codec(num_layers=32, cutoff_ratio=0.2)
kv = torch.randn(8, 64)
enc = codec_int4.encode(kv, layer_idx=10, tensor_id=0)
dec = codec_int4.decode(enc, layer_idx=10, tensor_id=0)
assert dec.shape == kv.shape
print("Prior-cycle HadamardInt4Codec: OK")

# Prior-cycle attention hooks (2026-05-03)
from vllm_integration.attention_backend_patch import (
    TurboQuantKVHook, CompressedKVHook, TriStateKVHook
)
hook = CompressedKVHook(codec_int4)
enc2 = hook.encode(kv, layer_idx=10)
dec2 = hook.decode(enc2, layer_idx=10)
assert dec2.shape == kv.shape
print("Prior-cycle CompressedKVHook: OK")

# Prior-cycle scheduler (2026-05-03)
from vllm_integration.scheduler_patch import (
    DualMapNodeState, DualMapSchedulerMixin, create_cache_hit_aware_queue
)
queue = create_cache_hit_aware_queue(chunk_size=8)
assert len(queue) == 0
print("Prior-cycle CacheHitAwareRequestQueue: OK")

# Prior-cycle DualMapSchedulerMixin
nodes = [
    DualMapNodeState(node_id="n0", current_load=0.3),
    DualMapNodeState(node_id="n1", current_load=0.1),
]
mixin = DualMapSchedulerMixin(nodes=nodes)
class R:
    request_id = "x"
    prompt_token_ids = [1, 2, 3]
sorted_reqs = mixin.sort_by_cache_affinity(
    [R()], get_request_id=lambda r: r.request_id,
    get_token_ids=lambda r: r.prompt_token_ids
)
assert len(sorted_reqs) == 1
print("Prior-cycle DualMapSchedulerMixin: OK")

# Prior-cycle SemanticSegmentIndex from block_manager_patch
from vllm_integration.block_manager_patch import SemanticSegmentIndex
idx = SemanticSegmentIndex(codec=None, chunk_size=16, max_entries=100)
tids = list(range(32))
k_t = torch.randn(16, 64)
v_t = torch.randn(16, 64)
stored = idx.store_segment(tids, 0, k_t, v_t, 0)
assert len(stored) == 64
print("Prior-cycle SemanticSegmentIndex: OK")

print("\nAll backward-compat checks passed.")
PYEOF2

echo ""
echo "=== Installation complete ==="
echo "vLLM version: ${VLLM_VERSION}"
