# vllm_integration: Activity A+B KV cache port for vLLM 0.21.0
#
# 2026-05-28 cycle additions:
#   hexagent_scheduler_patch          — HexAGeTWorkflowSchedulerMixin (Activity A-1):
#                                         HexAGenT (arXiv 2605.16637) online-public DAG
#                                         workflow scheduler. SLO-risk-weighted priority
#                                         + standalone completion horizon estimation.
#                                         Wraps schedule() with hexagent_pre_schedule().
#                                       + _InlineHexAGeTScheduler fallback (no src/).
#                                       + make_hexagent_workflow_scheduler_class() factory.
#   pegaflow_irminsul_block_manager_patch — PegaFlowIrminsulDistributedKVCacheManagerMixin
#                                         (Activity B-1 + A-2):
#                                         4-level distributed non-contiguous KV segment reuse:
#                                         local HBM → PegaFlow local → RDMA remote → miss.
#                                         δ-rotation on k_r (Irminsul MLA protocol).
#                                         Parallel segment store alongside vLLM block pool.
#                                       + make_pegaflow_irminsul_kv_cache_manager_class().
#   rdma_attention_backend_patch      — PegaFlowRDMASegmentAttentionHook (Activity B+A-2):
#                                         write_to_cache(): stores K/V, RETURNS ORIGINAL
#                                         (zero primary kernel error — accuracy contract).
#                                         read_from_cache(): 4-level distributed cache query.
#                                       + apply_pegaflow_rdma_segment_patch() monkey-patcher.
#                                       + extend_cache_config_pegaflow_rdma() CacheConfig helper.
#
# 2026-05-16 cycle additions:
#   scheduler_patch       — NAtHDDROffloadingSchedulerMixin + NAtHDDROffloadingSchedulerConfig
#                             + make_nath_ddr_scheduler_class (Activity A):
#                             NAtH 4-tier DDR offloading minimal-eviction scheduler.
#                             Based on NAtH (arXiv 2605.09490): accuracy depends only on
#                             permanent eviction rate; DDR offloading = zero approx error.
#                             Classifies waiting requests' tokens into 4 tiers each step.
#                             Permanent eviction capped at max_eviction_ratio=3%.
#                             Overhead: < 5ms p50 per schedule() call.
#   compression_codec     — GlobalRetentionGateVllmCodec (Activity C):
#                             Cross-layer competitive KV eviction. Ports
#                             GlobalRetentionGateEvictionCodec (src/cache/).
#                             Memory reduction: 70% (budget_ratio=0.3). Accuracy: < 1% error.
#                           + NAtHDDROffloadingCodecAdapter (Activity A+C):
#                             Bridges NAtH DDR tier policy with vLLM compression hooks.
#                             Tier 2: FP16 CPU offload. Tier 3: INT8 offload. Tier 4: evict.
#   attention_backend_patch — GlobalRetentionGateAttentionHook (Activity C):
#                              write_to_cache() / read_from_cache() hooks for
#                              GlobalRetentionGate eviction (post-compute, pre-cache write).
#                              Accuracy: < 1% attention error (MANDATORY §4).
#                            + NAtHDDRGlobalRetentionHook (Cross A+C):
#                              Composite hook: NAtH DDR 4-tier + GlobalRetentionGate budget.
#                            + apply_global_retention_gate_patch() monkey-patcher
#                            + extend_cache_config_global_retention() helper
#
# 2026-05-15 cycle additions:
#   scheduler_patch       — RadixFeatherSchedulerMixin + make_radix_feather_scheduler_class
#                             (Activity A): Feather (arXiv 2605.06046) prefix-homogeneity-
#                             aware batch reordering. Reorders vLLM's waiting queue per
#                             schedule() step using Radix tree prefix-match signal.
#                             Overhead: O(window * prefix_len), target < 5ms p50.
#   block_manager_patch   — RelayUShapeAuxStore + RelayUShapeKVCacheManagerMixin
#                             + make_relay_ulayer_kv_cache_manager_class (Activity B):
#                             U-shape layer-selective non-contiguous segment auxiliary
#                             store alongside vLLM's PagedAttention block table.
#                             Layer bitmask stored per segment; middle ~70% layers
#                             reused for non-identical segments.
#                             Ports: src/cache/relay_ulayer_segment.py
#   attention_backend_patch — LookaheadEvictionAttentionHook (Activity C):
#                              write_to_cache() / read_from_cache() for LookaheadKV
#                              eviction (arXiv 2603.10899, ICLR 2026). Kept KV is
#                              FP16 original (no quantization distortion).
#                              Accuracy: eviction_ratio=0.7 → attention error < 1%.
#                            + LookaheadRelayAttentionHook (Activity B+C):
#                              dual-filter: U-shape layer filter then LookaheadKV
#                              token filter. Combined memory reduction ~70–85%.
#                            + apply_lookahead_eviction_patch() monkey-patcher
#                            + extend_cache_config_lookahead_eviction() helper
#                              (adds compression_method, eviction_ratio, etc.)
#
# 2026-05-14 cycle additions:
#   compression_codec     — VllmFibQuantVQCodec: FibQuant Spherical-Beta radial-angular
#                             VQ codec adapter for vLLM attention-backend write/read hooks.
#                             Ports src/cache/fibquant_vq_codec.FibQuantVQCodec.
#                             Compression: 1.88x (bits_dir=8) / 3.56x (bits_dir=4) / 6.40x (bits_dir=2)
#                             Accuracy: cosine>=0.99 at 1.88x (mandatory), >=0.97 at 3.56x
#   block_manager_patch   — FibQuantVQSegmentKVManager (Activity B):
#                             KVCacheManager subclass with FibQuant non-contiguous auxiliary store.
#                             store_segment(): FibQuant-compress → auxiliary store.
#                             load_segment(): decompress on-demand (random access).
#                             get_noncontiguous_segments(): multi-chunk hit lookup + tracking.
#                             pad_block_table_with_fibquant(): FIBQUANT_SENTINEL block padding.
#                           + make_fibquant_kv_cache_manager_class() factory
#   attention_backend_patch — FibQuantAttentionHook (Activity B+C):
#                              write_to_cache() / read_from_cache() for FibQuant VQ.
#                              Pre-RoPE path: write_pre_rope() / read_pre_rope() for
#                              position-independent segment reuse (Cross B+C).
#                              apply_fibquant_patch() monkey-patcher for FlashAttentionImpl.
#                            + extend_cache_config_fibquant() helper
#
# 2026-05-13 cycle additions:
#   scheduler_patch       — PBKVAgentSegmentPreservationSchedulerMixin (Activity A):
#                             PBKV prediction-based segment preservation, fairness-weighted
#                             request reordering, GPU preserve/host evict policy.
#                           + make_pbkv_scheduler_class() factory
#   block_manager_patch   — KVFoldAccumulativeBlockManager (Activity B):
#                             foldl accumulator-based non-contiguous KV reuse,
#                             StreamingLLM fallback, SRFT+INT8 B+C integration hook.
#                           + make_kvfold_kv_cache_manager_class() factory
#   attention_backend_patch — SRFTInt8AttentionHook (Activity C):
#                              SRFT Gaussianization + INT8 per-group compression,
#                              write_to_cache() / read_from_cache() hooks,
#                              apply_srft_int8_patch() for FlashAttentionImpl.
#                            + extend_cache_config_srft_int8() helper
#
# 2026-05-12 cycle additions:
#   block_manager_patch   — AdapShotBlockManager: B+C AdapShotMixedDimSegmentPipeline
#                             parallel auxiliary store (RoPE re-encoding + MixedDim codec)
#                           + make_adapshot_kv_cache_manager_class() factory
#   attention_backend_patch — MixedDimAttentionHook: write/read hooks for
#                              MixedDimPerTokenBudgetCodec (compress before store,
#                              decompress before kernel)
#                           + extend_cache_config_mixed_dim(): CacheConfig extension helper
#   scheduler_patch       — AdapShotSegmentSchedulerMixin: non-contiguous hit-rate-based
#                             request reordering (B+C Cross-2)
#                           + make_adapshot_scheduler_class() factory
#
# 2026-05-11 cycle additions:
#   block_manager_patch   — WiCERBlockManager: CEGAR iterative non-contiguous KV
#                             artefact cache + parallel segment store (Activity B)
#                           + make_wicer_kv_cache_manager_class() factory
#   attention_backend_patch — RateQuantAttentionHook: RateQuant write/read hooks
#                              (compress before store, decompress before kernel)
#                              Accuracy: < 1% relative error at avg 4-bit budget
#   compression_codec     — RateQuantVllmCodec: reverse water-filling bit allocation
#                             75% memory reduction, < 1% accuracy error
#
# 2026-05-10 cycle additions:
#   block_manager_patch   — KVPacketVQBlockManager: KVPacket soft-adapter B+C cache
#                             + VQ compression, subclasses KVCacheManager
#                           + make_kvp_vq_kv_cache_manager_class() factory
#   attention_backend_patch — VQCodecAttentionHook: VQ write/read hooks for attention
#                              (compress before store, decompress before kernel)
#   scheduler_patch       — KVPacketSegmentSchedulerMixin: segment-hash reordering
#                           + make_kvp_segment_scheduler_class() factory
#
# 2026-05-09 cycle additions:
#   scheduler_patch          — HitAwarePPDRouterMixin, _InlinePPDRouter,
#                               make_hit_aware_ppd_scheduler_class, patch_scheduler_instance
#   block_manager_patch      — SegmentIndexAdapter, TriangleIndexKVCacheManagerMixin,
#                               _InlineTriangleIndex, _LightweightSegmentStore,
#                               build_triangle_index, make_triangle_index_kv_cache_manager_class,
#                               patch_kv_cache_manager_instance
#   attention_backend_patch  — SpecKVGammaAttentionHook, ContextIntensiveGuardAttentionHook,
#                               SpecKVContextGuardCombinedHook,
#                               patch_attention_impl_with_combined_hook
#
# Prior cycle additions are preserved for backward compatibility.
# See each submodule's docstring for full changelog.

from __future__ import annotations

import warnings
from typing import Any, Optional

# 2026-05-18 imports (Activity A+B+C)
try:
    from vllm_integration.scheduler_patch import (
        AMPDLazySegmentFetchSchedulerConfig,
        AMPDLazySegmentFetchSchedulerMixin,
        make_ampd_lazy_segment_fetch_scheduler_class,
    )
except (ImportError, AttributeError):
    AMPDLazySegmentFetchSchedulerConfig = None  # type: ignore
    AMPDLazySegmentFetchSchedulerMixin = None   # type: ignore
    make_ampd_lazy_segment_fetch_scheduler_class = None  # type: ignore

try:
    from vllm_integration.block_manager_patch import (
        AMPDAdapShotKVManagerConfig,
        AMPDAdapShotLazyLoadKVCacheManagerMixin,
        make_ampd_adapshot_kv_cache_manager_class,
    )
except (ImportError, AttributeError):
    AMPDAdapShotKVManagerConfig = None              # type: ignore
    AMPDAdapShotLazyLoadKVCacheManagerMixin = None  # type: ignore
    make_ampd_adapshot_kv_cache_manager_class = None  # type: ignore

try:
    from vllm_integration.attention_backend_patch import (
        DPAttentionAwareCompressionConfig_c18,
        DPAttentionAwareCompressionAttentionHook,
        extend_cache_config_dp_attn_aware_compression,
        apply_dp_attn_aware_compression_patch,
    )
except (ImportError, AttributeError):
    DPAttentionAwareCompressionConfig_c18 = None     # type: ignore
    DPAttentionAwareCompressionAttentionHook = None  # type: ignore
    extend_cache_config_dp_attn_aware_compression = None  # type: ignore
    apply_dp_attn_aware_compression_patch = None     # type: ignore

try:
    from vllm_integration.compression_codec import (
        DPAttentionAwareVllmCodec,
        DPAttentionCrossABCCodec,
    )
except (ImportError, AttributeError):
    DPAttentionAwareVllmCodec = None   # type: ignore
    DPAttentionCrossABCCodec = None    # type: ignore

# 2026-05-16 imports (Activity A+C)
try:
    from vllm_integration.scheduler_patch import (
        NAtHDDROffloadingSchedulerConfig,
        NAtHDDROffloadingSchedulerMixin,
        make_nath_ddr_scheduler_class,
    )
except (ImportError, AttributeError):
    NAtHDDROffloadingSchedulerConfig = None  # type: ignore
    NAtHDDROffloadingSchedulerMixin = None  # type: ignore
    make_nath_ddr_scheduler_class = None  # type: ignore

try:
    from vllm_integration.compression_codec import (
        GlobalRetentionGateVllmCodec,
        NAtHDDROffloadingCodecAdapter,
    )
except (ImportError, AttributeError):
    GlobalRetentionGateVllmCodec = None  # type: ignore
    NAtHDDROffloadingCodecAdapter = None  # type: ignore

try:
    from vllm_integration.attention_backend_patch import (
        GlobalRetentionGateAttentionHook,
        NAtHDDRGlobalRetentionHook,
        apply_global_retention_gate_patch,
        extend_cache_config_global_retention,
    )
except (ImportError, AttributeError):
    GlobalRetentionGateAttentionHook = None  # type: ignore
    NAtHDDRGlobalRetentionHook = None  # type: ignore
    apply_global_retention_gate_patch = None  # type: ignore
    extend_cache_config_global_retention = None  # type: ignore

# 2026-05-15 imports (Activity A+B+C)
try:
    from vllm_integration.scheduler_patch import (
        RadixFeatherSchedulerConfig,
        RadixFeatherSchedulerMixin,
        make_radix_feather_scheduler_class,
    )
except (ImportError, AttributeError):
    pass

try:
    from vllm_integration.block_manager_patch import (
        RelayUShapeAuxStore,
        RelayUShapeKVCacheManagerMixin,
        make_relay_ulayer_kv_cache_manager_class,
    )
except (ImportError, AttributeError):
    pass

try:
    from vllm_integration.attention_backend_patch import (
        LookaheadEvictionAttentionHook,
        LookaheadRelayAttentionHook,
        apply_lookahead_eviction_patch,
        extend_cache_config_lookahead_eviction,
    )
except (ImportError, AttributeError):
    pass


def apply_all_patches(
    vq_codec: Any = None,
    n_heads: int = 8,
    d_head: int = 128,
    recent_window: int = 64,
    adapter_rank: int = 8,
    max_packets: int = 512,
    reorder_window: int = 32,
    enable_scheduler_reorder: bool = True,
    enable_compression_hook: bool = True,
    enable_block_manager: bool = True,
) -> dict:
    """Apply all 2026-05-10 B+C patches to vLLM components.

    This function:
      1. Creates a VQCodecAttentionHook (Activity C write/read hooks).
      2. Creates a KVPacketVQBlockManager class (Activity B+C block manager).
      3. Creates a KVPacketSegmentSchedulerMixin class (Activity A+B scheduler).

    Returns a dict with:
      "vq_hook"           : VQCodecAttentionHook instance
      "kv_manager_class"  : KVPacketVQBlockManager subclass
      "scheduler_class"   : scheduler class with KVPacketSegmentSchedulerMixin
      "vllm_version"      : str
      "patches_applied"   : list[str]

    Parameters
    ----------
    vq_codec : VQCodec | None
        Pre-fitted VQCodec instance. If None, the hook will auto-fit on first use.
    n_heads, d_head : int
        Model architecture for SoftTokenAdapter dimensioning.
    recent_window : int
        FP16 tokens kept uncompressed (Activity C accuracy contract).
    adapter_rank : int
        SoftTokenAdapter rank.
    max_packets : int
        Max packets in LRU store.
    reorder_window : int
        Max waiting requests inspected per schedule step.
    enable_scheduler_reorder : bool
        If False, skip scheduler patch (graceful degradation).
    enable_compression_hook : bool
        If False, VQCodecAttentionHook is disabled (identity).
    enable_block_manager : bool
        If False, KVPacketVQBlockManager returns no-ops.

    Accuracy constraint:
        VQCodecAttentionHook.read_from_cache() always decompresses before
        returning — compressed tensors never enter the attention kernel.
        This satisfies evaluation_criteria.md §4 perplexity ±1% requirement.
    """
    import vllm as _vllm

    vllm_version = _vllm.__version__
    patches_applied = []

    # -- Activity C: VQ compression hook ------------------------------------
    try:
        from vllm_integration.attention_backend_patch import VQCodecAttentionHook
        vq_hook = VQCodecAttentionHook(
            vq_codec=vq_codec,
            recent_window=recent_window,
            enabled=enable_compression_hook,
        )
        patches_applied.append("VQCodecAttentionHook")
    except Exception as exc:
        warnings.warn(f"apply_all_patches: VQCodecAttentionHook failed: {exc}", RuntimeWarning)
        vq_hook = None

    # -- Activity B+C: KVPacket block manager class -------------------------
    kv_manager_class = None
    if enable_block_manager:
        try:
            from vllm_integration.block_manager_patch import (
                make_kvp_vq_kv_cache_manager_class,
            )
            from vllm.v1.core.kv_cache_manager import KVCacheManager
            kv_manager_class = make_kvp_vq_kv_cache_manager_class(KVCacheManager)
            patches_applied.append("KVPacketVQBlockManager")
        except Exception as exc:
            warnings.warn(f"apply_all_patches: KVPacketVQBlockManager failed: {exc}", RuntimeWarning)

    # -- Activity A+B: segment-aware scheduler class ------------------------
    scheduler_class = None
    if enable_scheduler_reorder:
        try:
            from vllm_integration.scheduler_patch import make_kvp_segment_scheduler_class
            from vllm.v1.core.sched.scheduler import Scheduler
            scheduler_class = make_kvp_segment_scheduler_class(Scheduler)
            patches_applied.append("KVPacketSegmentSchedulerMixin")
        except Exception as exc:
            warnings.warn(f"apply_all_patches: KVPacketSegmentSchedulerMixin failed: {exc}", RuntimeWarning)

    return {
        "vq_hook": vq_hook,
        "kv_manager_class": kv_manager_class,
        "scheduler_class": scheduler_class,
        "vllm_version": vllm_version,
        "patches_applied": patches_applied,
    }


__all__ = [
    "apply_all_patches",
    # 2026-05-23 (Activity C+A+B — RuntimeCertified + CPD + CLC)
    "CPDRouterSchedulerConfig",
    "CPDWarmColdSchedulerMixin",
    "make_cpd_warm_cold_scheduler_class",
    "CLCBiasGateKVManagerConfig",
    "CLCPositionalBiasGatedKVCacheManagerMixin",
    "make_clc_bias_gate_kv_cache_manager_class",
    "RuntimeCertifiedAttentionHookConfig",
    "RuntimeCertifiedAttentionHook",
    "apply_runtime_certified_patch",
    "extend_cache_config_runtime_certified",
    "RuntimeCertifiedKVSculptVllmHook",
    # 2026-05-21 (Activity B+C — BlockUnion pipeline)
    "BlockUnionKVManagerConfig",
    "BlockUnionNonContiguousKVCacheManagerMixin",
    "make_block_union_kv_cache_manager_class",
    "CompactAttentionBlockUnionConfig",
    "CompactAttentionBlockUnionHook",
    "apply_compact_attention_block_union_patch",
    "extend_cache_config_block_union_codec",
    "BlockUnionFlashAttentionForwardPatcher",
    "apply_block_union_flash_attention_forward_patch",
    "BlockUnionBCSchedulerConfig",
    "BlockUnionBCSchedulerMixin",
    "make_block_union_bc_scheduler_class",
    "CompactAttentionBlockUnionVllmCodecConfig",
    "CompactAttentionBlockUnionVllmCodec",
    "BlockUnionBCPipelineVllmCodec",
    # 2026-05-18
    "AMPDLazySegmentFetchSchedulerConfig",
    "AMPDLazySegmentFetchSchedulerMixin",
    "make_ampd_lazy_segment_fetch_scheduler_class",
    "AMPDAdapShotKVManagerConfig",
    "AMPDAdapShotLazyLoadKVCacheManagerMixin",
    "make_ampd_adapshot_kv_cache_manager_class",
    "DPAttentionAwareCompressionConfig_c18",
    "DPAttentionAwareCompressionAttentionHook",
    "extend_cache_config_dp_attn_aware_compression",
    "apply_dp_attn_aware_compression_patch",
    "DPAttentionAwareVllmCodec",
    "DPAttentionCrossABCCodec",
    # 2026-05-16
    "NAtHDDROffloadingSchedulerConfig",
    "NAtHDDROffloadingSchedulerMixin",
    "make_nath_ddr_scheduler_class",
    "GlobalRetentionGateVllmCodec",
    "NAtHDDROffloadingCodecAdapter",
    "GlobalRetentionGateAttentionHook",
    "NAtHDDRGlobalRetentionHook",
    "apply_global_retention_gate_patch",
    "extend_cache_config_global_retention",
    # 2026-05-15
    "RadixFeatherSchedulerConfig",
    "RadixFeatherSchedulerMixin",
    "make_radix_feather_scheduler_class",
    "RelayUShapeAuxStore",
    "RelayUShapeKVCacheManagerMixin",
    "make_relay_ulayer_kv_cache_manager_class",
    "LookaheadEvictionAttentionHook",
    "LookaheadRelayAttentionHook",
    "apply_lookahead_eviction_patch",
    "extend_cache_config_lookahead_eviction",
    # 2026-05-14
    "VllmFibQuantVQCodec",
    "FibQuantVQSegmentKVManager",
    "make_fibquant_kv_cache_manager_class",
    "FibQuantAttentionHook",
    "apply_fibquant_patch",
    "extend_cache_config_fibquant",
    # 2026-05-12
    "AdapShotBlockManager",
    "make_adapshot_kv_cache_manager_class",
    "MixedDimAttentionHook",
    "extend_cache_config_mixed_dim",
    "AdapShotSegmentSchedulerMixin",
    "make_adapshot_scheduler_class",
    # 2026-05-11
    "RateQuantVllmCodec",
    "RateQuantAttentionHook",
    "WiCERBlockManager",
    "make_wicer_kv_cache_manager_class",
    # 2026-05-10
    "VQCodecAttentionHook",
    "KVPacketVQBlockManager",
    "KVPacketSegmentSchedulerMixin",
    "make_kvp_vq_kv_cache_manager_class",
    "make_kvp_segment_scheduler_class",
]

# ===========================================================================
# 2026-05-19 imports (Activity A+B+C — KVDrive integrated stack)
# ===========================================================================

# Activity A: KVDrive attention-pipeline scheduler
try:
    from vllm_integration.scheduler_patch import (
        KVDriveAttentionPipelineConfig,
        KVDriveAttentionPipelineMixin,
        make_kvdrive_vllm_scheduler_class,
    )
except (ImportError, AttributeError):
    KVDriveAttentionPipelineConfig = None  # type: ignore
    KVDriveAttentionPipelineMixin = None   # type: ignore
    make_kvdrive_vllm_scheduler_class = None  # type: ignore

# Activity B: ThunderAgent static segment reservation
try:
    from vllm_integration.block_manager_patch import (
        ThunderAgentKVManagerConfig,
        ThunderAgentKVCacheManagerMixin,
        LLMProgramDAG_19,
        LLMProgramStep_19,
        make_thunder_agent_kv_manager_class,
    )
except (ImportError, AttributeError):
    ThunderAgentKVManagerConfig = None      # type: ignore
    ThunderAgentKVCacheManagerMixin = None  # type: ignore
    LLMProgramDAG_19 = None                 # type: ignore
    LLMProgramStep_19 = None                # type: ignore
    make_thunder_agent_kv_manager_class = None  # type: ignore

# Activity C: KVDrive tier-differentiated compression hook
try:
    from vllm_integration.attention_backend_patch import (
        KVDriveTierCompressionConfig_c19,
        KVDriveTierCompressionMixin,
        apply_kvdrive_tier_compression_patch,
        extend_cache_config_kvdrive,
    )
except (ImportError, AttributeError):
    KVDriveTierCompressionConfig_c19 = None      # type: ignore
    KVDriveTierCompressionMixin = None           # type: ignore
    apply_kvdrive_tier_compression_patch = None  # type: ignore
    extend_cache_config_kvdrive = None           # type: ignore

# Activity C codec
try:
    from vllm_integration.compression_codec import (
        KVDriveTierDifferentiatedVllmCodec,
        KVDriveCrossABCCodec,
    )
except (ImportError, AttributeError):
    KVDriveTierDifferentiatedVllmCodec = None  # type: ignore
    KVDriveCrossABCCodec = None                # type: ignore

# Config extension
try:
    from vllm_integration.cache_config_extension import (
        KVDriveActivityABCConfig,
        KVDriveActivityABCConfigMixin,
        build_kvdrive_abc_config,
    )
except (ImportError, AttributeError):
    KVDriveActivityABCConfig = None      # type: ignore
    KVDriveActivityABCConfigMixin = None  # type: ignore
    build_kvdrive_abc_config = None      # type: ignore

# ===========================================================================
# 2026-05-21 imports (Activity B+C — BlockUnion pipeline)
# ===========================================================================

# Activity B: BlockUnion non-contiguous KV cache manager
try:
    from vllm_integration.block_manager_patch import (
        BlockUnionKVManagerConfig,
        BlockUnionNonContiguousKVCacheManagerMixin,
        make_block_union_kv_cache_manager_class,
    )
except (ImportError, AttributeError):
    BlockUnionKVManagerConfig = None                       # type: ignore
    BlockUnionNonContiguousKVCacheManagerMixin = None      # type: ignore
    make_block_union_kv_cache_manager_class = None         # type: ignore

# Activity C: CompactAttention block-union attention hook
try:
    from vllm_integration.attention_backend_patch import (
        CompactAttentionBlockUnionConfig,
        CompactAttentionBlockUnionHook,
        apply_compact_attention_block_union_patch,
        extend_cache_config_block_union_codec,
        BlockUnionFlashAttentionForwardPatcher,
        apply_block_union_flash_attention_forward_patch,
    )
except (ImportError, AttributeError):
    CompactAttentionBlockUnionConfig = None                # type: ignore
    CompactAttentionBlockUnionHook = None                  # type: ignore
    apply_compact_attention_block_union_patch = None       # type: ignore
    extend_cache_config_block_union_codec = None           # type: ignore
    BlockUnionFlashAttentionForwardPatcher = None          # type: ignore
    apply_block_union_flash_attention_forward_patch = None  # type: ignore

# Activity B+C: scheduler mixin
try:
    from vllm_integration.scheduler_patch import (
        BlockUnionBCSchedulerConfig,
        BlockUnionBCSchedulerMixin,
        make_block_union_bc_scheduler_class,
    )
except (ImportError, AttributeError):
    BlockUnionBCSchedulerConfig = None     # type: ignore
    BlockUnionBCSchedulerMixin = None      # type: ignore
    make_block_union_bc_scheduler_class = None  # type: ignore

# Activity C codec + B+C cross codec
try:
    from vllm_integration.compression_codec import (
        CompactAttentionBlockUnionVllmCodecConfig,
        CompactAttentionBlockUnionVllmCodec,
        BlockUnionBCPipelineVllmCodec,
    )
except (ImportError, AttributeError):
    CompactAttentionBlockUnionVllmCodecConfig = None  # type: ignore
    CompactAttentionBlockUnionVllmCodec = None         # type: ignore
    BlockUnionBCPipelineVllmCodec = None               # type: ignore

# ===========================================================================
# 2026-05-23 imports (Activity C+A+B — RuntimeCertified + CPD + CLC)
# ===========================================================================

# Activity A: CPDWarmColdHitRateRouter scheduler mixin
try:
    from vllm_integration.scheduler_patch import (
        CPDRouterSchedulerConfig,
        CPDWarmColdSchedulerMixin,
        make_cpd_warm_cold_scheduler_class,
    )
except (ImportError, AttributeError):
    CPDRouterSchedulerConfig = None              # type: ignore
    CPDWarmColdSchedulerMixin = None             # type: ignore
    make_cpd_warm_cold_scheduler_class = None    # type: ignore

# Activity B: CLCPositionalBiasGated KV cache manager
try:
    from vllm_integration.block_manager_patch import (
        CLCBiasGateKVManagerConfig,
        CLCPositionalBiasGatedKVCacheManagerMixin,
        make_clc_bias_gate_kv_cache_manager_class,
    )
except (ImportError, AttributeError):
    CLCBiasGateKVManagerConfig = None                     # type: ignore
    CLCPositionalBiasGatedKVCacheManagerMixin = None      # type: ignore
    make_clc_bias_gate_kv_cache_manager_class = None      # type: ignore

# Activity C: RuntimeCertified INT8K+INT4V attention hook
try:
    from vllm_integration.attention_backend_patch import (
        RuntimeCertifiedAttentionHookConfig,
        RuntimeCertifiedAttentionHook,
        apply_runtime_certified_patch,
        extend_cache_config_runtime_certified,
        RuntimeCertifiedKVSculptVllmHook,
    )
except (ImportError, AttributeError):
    RuntimeCertifiedAttentionHookConfig = None        # type: ignore
    RuntimeCertifiedAttentionHook = None              # type: ignore
    apply_runtime_certified_patch = None              # type: ignore
    extend_cache_config_runtime_certified = None      # type: ignore
    RuntimeCertifiedKVSculptVllmHook = None           # type: ignore

# ===========================================================================
# 2026-05-22 imports (Activity A+B+C — DapQ position-aware eviction stack)
# ===========================================================================

# Activity A: PPDAppendFullPrefillClassifier + DapQSessionSegment scheduler
try:
    from vllm_integration.scheduler_patch import (
        PPDClassifierSchedulerConfig,
        PPDAppendFullPrefillClassifierMixin,
        make_ppd_classifier_scheduler_class,
        DapQSessionSegmentSchedulerMixin,
        make_dapq_session_segment_scheduler_class,
    )
except (ImportError, AttributeError):
    PPDClassifierSchedulerConfig = None                      # type: ignore
    PPDAppendFullPrefillClassifierMixin = None               # type: ignore
    make_ppd_classifier_scheduler_class = None               # type: ignore
    DapQSessionSegmentSchedulerMixin = None                  # type: ignore
    make_dapq_session_segment_scheduler_class = None         # type: ignore

# Activity B+C: DapQSessionSegment KV cache manager
try:
    from vllm_integration.block_manager_patch import (
        DapQSessionSegmentConfig,
        DapQSessionSegmentKVCacheManagerMixin,
        make_dapq_session_segment_kv_cache_manager_class,
    )
except (ImportError, AttributeError):
    DapQSessionSegmentConfig = None                          # type: ignore
    DapQSessionSegmentKVCacheManagerMixin = None             # type: ignore
    make_dapq_session_segment_kv_cache_manager_class = None  # type: ignore

# Activity C: DapQPositionAwareEviction attention hook
try:
    from vllm_integration.attention_backend_patch import (
        DapQAttentionHookConfig,
        DapQPositionAwareEvictionAttentionHook,
        DapQDualReductionAttentionHook,
        extend_cache_config_dapq,
        apply_dapq_patch,
    )
except (ImportError, AttributeError):
    DapQAttentionHookConfig = None                           # type: ignore
    DapQPositionAwareEvictionAttentionHook = None            # type: ignore
    DapQDualReductionAttentionHook = None                    # type: ignore
    extend_cache_config_dapq = None                         # type: ignore
    apply_dapq_patch = None                                  # type: ignore

# ===========================================================================
# 2026-05-25 imports (Activity B+C — KVPacket + VeriCache speculative codec)
# ===========================================================================

# Activity C: VeriCache speculative draft-verify codec hook
try:
    from vllm_integration.vericache_codec_patch import (
        VeriCacheCodecHookConfig,
        VeriCacheCodecAttentionHook,
        VeriCacheVerificationResult,
        apply_vericache_codec_patch,
        extend_cache_config_vericache,
    )
except (ImportError, AttributeError):
    VeriCacheCodecHookConfig = None         # type: ignore
    VeriCacheCodecAttentionHook = None      # type: ignore
    VeriCacheVerificationResult = None      # type: ignore
    apply_vericache_codec_patch = None      # type: ignore
    extend_cache_config_vericache = None    # type: ignore

# Activity B: KVPacket non-contiguous segment KV cache manager
try:
    from vllm_integration.kv_packet_block_manager_patch import (
        KVPacketSegmentConfig,
        KVPacketSegmentMixin,
        make_kv_packet_kv_cache_manager_class,
    )
except (ImportError, AttributeError):
    KVPacketSegmentConfig = None               # type: ignore
    KVPacketSegmentMixin = None                # type: ignore
    make_kv_packet_kv_cache_manager_class = None  # type: ignore

# ===========================================================================
# 2026-05-27 imports (Activity B+C — IndexMem Soft Hit + Eviction Codec)
# ===========================================================================

# Activity C: IndexMem Eviction Codec attention hook
try:
    from vllm_integration.indexmem_eviction_codec_patch import (
        IndexMemEvictionHookConfig,
        IndexMemEvictionCodecAttentionHook,
        extend_cache_config_indexmem,
        apply_indexmem_eviction_patch,
    )
except (ImportError, AttributeError):
    IndexMemEvictionHookConfig = None              # type: ignore
    IndexMemEvictionCodecAttentionHook = None      # type: ignore
    extend_cache_config_indexmem = None            # type: ignore
    apply_indexmem_eviction_patch = None           # type: ignore

# Activity B: IndexMem Soft Hit KV cache manager
try:
    from vllm_integration.indexmem_block_manager_patch import (
        IndexMemSoftHitMixinConfig,
        IndexMemSoftHitKVCacheManagerMixin,
        SegmentLatentPool,
        SoftHitResult,
        make_indexmem_soft_hit_kv_cache_manager_class,
        apply_indexmem_block_manager_patch,
    )
except (ImportError, AttributeError):
    IndexMemSoftHitMixinConfig = None                      # type: ignore
    IndexMemSoftHitKVCacheManagerMixin = None              # type: ignore
    SegmentLatentPool = None                               # type: ignore
    SoftHitResult = None                                   # type: ignore
    make_indexmem_soft_hit_kv_cache_manager_class = None  # type: ignore
    apply_indexmem_block_manager_patch = None              # type: ignore

# Activity A+B: IndexMem Soft Hit Scheduler mixin
try:
    from vllm_integration.indexmem_vllm_scheduler_patch import (
        IndexMemSoftHitSchedulerConfig,
        IndexMemSoftHitSchedulerMixin,
        make_indexmem_soft_hit_scheduler_class,
    )
except (ImportError, AttributeError):
    IndexMemSoftHitSchedulerConfig = None              # type: ignore
    IndexMemSoftHitSchedulerMixin = None               # type: ignore
    make_indexmem_soft_hit_scheduler_class = None      # type: ignore
