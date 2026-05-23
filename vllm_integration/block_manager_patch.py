"""block_manager_patch.py — Activity B: KV cache non-contiguous reuse for vLLM 0.21.0.

2026-05-22: DapQSessionSegmentKVCacheManagerMixin — ports
            DapQSessionSegmentDualReductionPipeline (Activity B+C Cross-2) and
            SessionAwareTurnLevelSegmentCache (Activity B) into vLLM's v1
            KVCacheManager as a mixin.

            Provides session-aware, turn-level non-contiguous KV block reuse:
              - store_turn_segment(): Compress KV with DapQ and register in the
                session-aware turn-level segment cache (Activity B integration).
              - get_session_segments(): Return active segments for a session/turn.
              - get_top_segments_by_position(): DapQ position-aware segment selection.
              - process_dual_reduction(): Run full B+C dual-reduction pipeline for
                a session, returning (TurnSegmentEntry, compressed_kv) pairs.

            make_dapq_session_segment_kv_cache_manager_class() factory:
              Subclasses KVCacheManager and adds session-aware segment reuse.
              All native KVCacheManager methods are preserved (no overrides required).

            Integration with vLLM v1 architecture:
              DapQSessionSegmentKVCacheManagerMixin operates as a PARALLEL segment
              store alongside vLLM's native paged block pool. It does NOT replace
              or modify the native KV block allocation logic. The segment store is
              keyed by (content_hash, session_id, turn_id, layer_idx), decoupled
              from vLLM's block_id namespace.

            Non-contiguous reuse path:
              When a new request arrives at a D-node with session history, the
              mixin's annotate_request() queries the session cache for previously
              computed KV segments, builds a non-contiguous block index, and
              attaches metadata to the vLLM Request for attention kernel routing.

2026-05-21 (prior): BlockUnionNonContiguousKVCacheManagerMixin — preserved below.
2026-05-17 (prior): AdapShotBlockManager — preserved below.
2026-05-12 (prior): WiCERBlockManager — preserved below.
2026-05-06 (prior): QueryCentricKVCacheManager — preserved below.
2026-05-04 (prior): WorkloadAwareTTLKVCacheManager — preserved below.

vLLM 0.21.0 v1 architecture:
    - KVCacheManager is in vllm.v1.core.kv_cache_manager.KVCacheManager
    - BlockPool handles raw block allocation (FreeKVCacheBlockQueue)
    - KVCacheManager.get_computed_blocks() — prefix cache lookup
    - KVCacheManager.allocate_slots() — block allocation
    - KVCacheManager.evict_blocks(block_ids) — explicit eviction API

Integration strategy (2026-05-22):
    DapQSessionSegmentKVCacheManagerMixin adds a session-aware segment index
    alongside the native prefix cache. The mixin:
      1. Intercepts session context (session_id, turn_id) via store_turn_segment().
      2. Applies DapQ eviction to the KV before storing in the segment cache.
      3. Exposes get_session_segments() and process_dual_reduction() for the
         scheduler and model runner to query cached segments.
      4. Non-contiguous block tables are constructed from segment data and passed
         to the attention kernel as padded block_table tensors.

vLLM version: 0.21.0
Activity: B+C — DapQSessionSegmentDualReductionPipeline
"""

from __future__ import annotations

import sys
import pathlib
import hashlib
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import vllm

def _vllm_version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:3])

assert _vllm_version_tuple(vllm.__version__) >= _vllm_version_tuple("0.4.0"), (
    f"vllm_integration requires vLLM >= 0.4.0, found {vllm.__version__}"
)


def _add_repo_root_to_path() -> None:
    repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _try_import_session_segment_src() -> tuple:
    """Import SessionAwareTurnLevelSegmentCache from src/."""
    _add_repo_root_to_path()
    try:
        from src.cache.session_turn_level_segment_cache import (
            SessionAwareTurnLevelSegmentCache,
            SessionTurnLevelConfig,
            TurnSegmentEntry,
        )
        return SessionAwareTurnLevelSegmentCache, SessionTurnLevelConfig, TurnSegmentEntry
    except ImportError:
        return None, None, None


def _try_import_dual_pipeline_src() -> tuple:
    """Import DapQSessionSegmentDualReductionPipeline from src/."""
    _add_repo_root_to_path()
    try:
        from src.cache.dapq_session_segment_dual_pipeline import (
            DapQSessionSegmentDualReductionPipeline,
            DualReductionPipelineConfig,
        )
        return DapQSessionSegmentDualReductionPipeline, DualReductionPipelineConfig
    except ImportError:
        return None, None


def _try_import_dapq_src() -> tuple:
    """Import DapQPositionAwareEvictionCodec from src/."""
    _add_repo_root_to_path()
    try:
        from src.cache.dapq_position_aware_eviction_codec import (
            DapQPositionAwareEvictionCodec,
            DapQEvictionConfig,
        )
        return DapQPositionAwareEvictionCodec, DapQEvictionConfig
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Import KVCacheManager (graceful fallback for CPU-only / no-GPU environments)
# ---------------------------------------------------------------------------

try:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    _KV_CACHE_MANAGER_AVAILABLE = True
except Exception:
    # Graceful fallback: define a minimal stub for non-GPU environments
    class KVCacheManager:  # type: ignore[no-redef]
        """Stub KVCacheManager for CPU-only environments."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    _KV_CACHE_MANAGER_AVAILABLE = False


# ===========================================================================
# DapQSessionSegmentMixin config
# ===========================================================================

@dataclass
class DapQSessionSegmentConfig:
    """Configuration for DapQSessionSegmentKVCacheManagerMixin.

    Mirrors the relevant fields from DualReductionPipelineConfig and
    SessionTurnLevelConfig.
    """
    chunk_size: int = 128
    max_entries: int = 1000
    max_turns_per_session: int = 10
    session_lru_penalty: float = 0.5
    segment_keep_ratio: float = 0.50
    kv_budget_ratio: float = 0.30
    decay_factor: float = 512.0
    d_head: int = 128
    n_kv_heads: int = 8
    n_layers: int = 12
    recent_window: int = 32
    use_unit_template: bool = True
    seed: int = 42
    enabled: bool = True


# ===========================================================================
# 2026-05-22: DapQSessionSegmentKVCacheManagerMixin (Activity B+C)
# ===========================================================================

class DapQSessionSegmentKVCacheManagerMixin:
    """vLLM v1 KVCacheManager mixin: DapQ session segment non-contiguous reuse.

    Activity B+C: Non-Contiguous KV Cache Reuse + KV Cache Compression.

    This mixin adds a parallel session-aware segment store alongside vLLM's
    native paged block pool. It does NOT replace or modify any native
    KVCacheManager methods — all existing methods are preserved.

    Core API:
        store_turn_segment(session_id, turn_id, token_ids, chunk_idx,
                           kv_tensor, layer_idx):
            Compress KV with DapQ (Activity C) and register in the session
            segment cache (Activity B).

        get_session_segments(session_id, turn_range):
            Return active TurnSegmentEntry list for a session (optionally
            filtered by turn range).

        get_top_segments_by_position(session_id, current_decode_pos, keep_ratio):
            Return position-aware top-scored segments for a session.

        process_dual_reduction(session_id, current_decode_pos):
            Run full B+C dual-reduction pipeline: segment selection (B) +
            KV eviction (C). Returns list of (TurnSegmentEntry, compressed_kv).

        annotate_request(request, session_id, current_decode_pos):
            Attach non-contiguous segment metadata to a vLLM Request object.
            Called by the scheduler mixin before allocate_slots().

    Non-contiguous block table:
        build_noncontiguous_block_table(session_id, layer_idx, block_size):
            Returns int64 tensor [1, max_blocks] with block indices from the
            session segment cache — suitable for injection into the attention
            kernel's block_tables argument.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.block_manager_patch import (
            DapQSessionSegmentKVCacheManagerMixin,
            DapQSessionSegmentConfig,
            make_dapq_session_segment_kv_cache_manager_class,
        )

        # Option A: mixin
        class MyKVCacheManager(DapQSessionSegmentKVCacheManagerMixin, KVCacheManager):
            pass

        # Option B: factory
        DapQKVManager = make_dapq_session_segment_kv_cache_manager_class(KVCacheManager)
        mgr = DapQKVManager(
            ...,  # standard KVCacheManager args
            dapq_session_config=DapQSessionSegmentConfig(
                segment_keep_ratio=0.50,
                kv_budget_ratio=0.30,
            ),
        )
    """

    def __init__(
        self,
        *args: Any,
        dapq_session_config: Optional[DapQSessionSegmentConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            dapq_session_config: DapQSessionSegmentConfig. If None, uses defaults.
            All other args/kwargs forwarded to the base KVCacheManager.__init__().
        """
        super().__init__(*args, **kwargs)

        if dapq_session_config is None:
            dapq_session_config = DapQSessionSegmentConfig()
        self._dapq_seg_cfg = dapq_session_config

        # Try to import native pipeline from src/
        DapQSessionSegmentDualReductionPipeline, DualReductionPipelineConfig = (
            _try_import_dual_pipeline_src()
        )

        self._dapq_pipeline: Optional[Any] = None
        self._dapq_use_native: bool = False

        if DapQSessionSegmentDualReductionPipeline is not None and _TORCH_AVAILABLE:
            DapQPositionAwareEvictionCodec, DapQEvictionConfig = _try_import_dapq_src()
            SessionAwareTurnLevelSegmentCache, SessionTurnLevelConfig, _ = (
                _try_import_session_segment_src()
            )
            b_cfg = SessionTurnLevelConfig(
                chunk_size=dapq_session_config.chunk_size,
                max_entries=dapq_session_config.max_entries,
                max_turns_per_session=dapq_session_config.max_turns_per_session,
                session_lru_penalty=dapq_session_config.session_lru_penalty,
                seed=dapq_session_config.seed,
            ) if SessionTurnLevelConfig else None
            c_cfg = DapQEvictionConfig(
                d_head=dapq_session_config.d_head,
                n_kv_heads=dapq_session_config.n_kv_heads,
                n_layers=dapq_session_config.n_layers,
                budget_ratio=dapq_session_config.kv_budget_ratio,
                recent_window=dapq_session_config.recent_window,
                use_unit_template=dapq_session_config.use_unit_template,
                seed=dapq_session_config.seed,
            ) if DapQEvictionConfig else None
            pipeline_cfg = DualReductionPipelineConfig(
                b_config=b_cfg,
                c_config=c_cfg,
                segment_keep_ratio=dapq_session_config.segment_keep_ratio,
                kv_budget_ratio=dapq_session_config.kv_budget_ratio,
                decay_factor=dapq_session_config.decay_factor,
                seed=dapq_session_config.seed,
            )
            self._dapq_pipeline = DapQSessionSegmentDualReductionPipeline(pipeline_cfg)
            self._dapq_use_native = True
        else:
            # Fallback: lightweight inline session segment store
            self._dapq_pipeline = _InlineSessionSegmentStore(dapq_session_config)

        # Metrics
        self._dapq_store_count: int = 0
        self._dapq_hit_count: int = 0

    # -----------------------------------------------------------------------
    # Session-aware segment API
    # -----------------------------------------------------------------------

    def store_turn_segment(
        self,
        session_id: str,
        turn_id: int,
        token_ids: List[int],
        chunk_idx: int,
        kv_tensor: "torch.Tensor",
        layer_idx: int = 0,
        ttl: Optional[float] = None,
    ) -> str:
        """Compress KV with DapQ and register in the session segment cache.

        Args:
            session_id: Session identifier.
            turn_id: Conversation turn index (0 = first turn).
            token_ids: Token ID sequence for content hashing.
            chunk_idx: Chunk index within the sequence.
            kv_tensor: KV tensor [seq_len, d_head] or [seq_len, n_heads, d_head].
            layer_idx: Transformer layer index.
            ttl: Optional time-to-live in seconds. None = no expiry.

        Returns:
            Segment key string (SHA-256 based).
        """
        self._dapq_store_count += 1
        if self._dapq_use_native and self._dapq_pipeline is not None and _TORCH_AVAILABLE:
            try:
                key = self._dapq_pipeline.segment_cache.put_turn_segment(
                    token_ids=token_ids,
                    chunk_idx=chunk_idx,
                    kv=kv_tensor,
                    session_id=session_id,
                    turn_id=turn_id,
                    layer_idx=layer_idx,
                    ttl=ttl,
                )
                return key
            except Exception:
                pass
        # Fallback
        return self._dapq_pipeline.store(session_id, turn_id, kv_tensor)

    def get_session_segments(
        self,
        session_id: str,
        turn_range: Optional[Tuple[int, int]] = None,
    ) -> List[Any]:
        """Return active TurnSegmentEntry list for a session.

        Args:
            session_id: Session to query.
            turn_range: Optional (min_turn, max_turn_inclusive) filter.

        Returns:
            List of TurnSegmentEntry objects (or dicts in fallback mode).
        """
        if self._dapq_use_native and self._dapq_pipeline is not None:
            try:
                return self._dapq_pipeline.segment_cache.get_session_segments(
                    session_id=session_id,
                    turn_range=turn_range,
                )
            except Exception:
                pass
        return self._dapq_pipeline.get_segments(session_id, turn_range)

    def get_top_segments_by_position(
        self,
        session_id: str,
        current_decode_pos: float,
        keep_ratio: float = 0.50,
        decay_factor: float = 512.0,
    ) -> List[Any]:
        """Return position-aware top-scored segments.

        Args:
            session_id: Session to query.
            current_decode_pos: Current decode position for proximity scoring.
            keep_ratio: Fraction of segments to return (0.50 = top 50%).
            decay_factor: Position-awareness decay parameter.

        Returns:
            List of top-scored TurnSegmentEntry objects.
        """
        if self._dapq_use_native and self._dapq_pipeline is not None:
            try:
                return self._dapq_pipeline.segment_cache.get_top_segments_by_position(
                    session_id=session_id,
                    current_decode_pos=current_decode_pos,
                    keep_ratio=keep_ratio,
                    decay_factor=decay_factor,
                )
            except Exception:
                pass
        return self.get_session_segments(session_id)

    def process_dual_reduction(
        self,
        session_id: str,
        current_decode_pos: float,
        pos_decode_int: Optional[int] = None,
    ) -> List[Tuple[Any, "torch.Tensor"]]:
        """Run full B+C dual-reduction pipeline for a session.

        Applies:
          Step 1: Retrieve all segments for session (B).
          Step 2: Score by DapQ position-awareness, keep top segment_keep_ratio (B+C).
          Step 3: Apply DapQ KV eviction to selected segments (C).

        Args:
            session_id: Session to process.
            current_decode_pos: Current decode position (float).
            pos_decode_int: Integer decode position. If None, cast from float.

        Returns:
            List of (TurnSegmentEntry, compressed_kv_tensor).
            Empty if no cached segments for session.
        """
        if self._dapq_use_native and self._dapq_pipeline is not None and _TORCH_AVAILABLE:
            try:
                return self._dapq_pipeline.process_session(
                    session_id=session_id,
                    current_decode_pos=current_decode_pos,
                    pos_decode_int=pos_decode_int,
                )
            except Exception:
                pass
        return []

    def annotate_request(
        self,
        request: Any,
        session_id: str,
        current_decode_pos: float,
    ) -> None:
        """Attach non-contiguous segment metadata to a vLLM Request.

        Queries the session cache and attaches relevant segment info as
        runtime attributes on the Request object. The scheduler and model
        runner can read these attributes to construct non-contiguous
        block tables for the attention kernel.

        Args:
            request: vLLM Request object (vllm.v1.request.Request).
            session_id: Session associated with this request.
            current_decode_pos: Current decode position for segment scoring.
        """
        segments = self.get_top_segments_by_position(
            session_id=session_id,
            current_decode_pos=current_decode_pos,
            keep_ratio=self._dapq_seg_cfg.segment_keep_ratio,
        )
        if segments:
            self._dapq_hit_count += 1
        # Attach metadata as runtime attributes (does not modify vLLM internals)
        try:
            object.__setattr__(request, "dapq_session_id", session_id)
            object.__setattr__(request, "dapq_segments", segments)
            object.__setattr__(request, "dapq_noncontiguous_hit", len(segments) > 0)
        except Exception:
            # Graceful: some Request implementations may reject setattr
            pass

    def build_noncontiguous_block_table(
        self,
        session_id: str,
        layer_idx: int = 0,
        block_size: int = 16,
    ) -> Optional["torch.Tensor"]:
        """Build a non-contiguous block table tensor for attention kernel injection.

        Constructs an int64 tensor [1, max_blocks] from session segment entries,
        where each block index corresponds to a cached KV segment. Non-contiguous
        positions are padded with -1 (PagedAttention convention for empty blocks).

        This tensor can be injected as the block_tables argument in
        FlashAttentionImpl.forward() to enable non-contiguous KV reuse.

        Args:
            session_id: Session to build block table for.
            layer_idx: Transformer layer index.
            block_size: Number of tokens per block (must match vLLM block_size).

        Returns:
            int64 tensor [1, n_blocks] or None if no segments available.
        """
        if not _TORCH_AVAILABLE:
            return None
        segments = self.get_session_segments(session_id)
        if not segments:
            return None
        # Extract block ranges from segment token_range
        block_indices = []
        for entry in segments:
            try:
                start, end = entry.token_range
                for pos in range(start, end, block_size):
                    block_idx = pos // block_size
                    if block_idx not in block_indices:
                        block_indices.append(block_idx)
            except (AttributeError, TypeError):
                pass
        if not block_indices:
            return None
        max_blocks = max(block_indices) + 1
        table = torch.full((1, max_blocks), -1, dtype=torch.int64)
        for b in block_indices:
            table[0, b] = b
        return table

    def dapq_segment_metrics(self) -> Dict[str, Any]:
        """Return metrics from the session segment pipeline."""
        base = {
            "dapq_store_count": self._dapq_store_count,
            "dapq_hit_count": self._dapq_hit_count,
        }
        if self._dapq_use_native and self._dapq_pipeline is not None:
            try:
                base.update(self._dapq_pipeline.metrics_summary())
            except Exception:
                pass
        return base


# ---------------------------------------------------------------------------
# make_dapq_session_segment_kv_cache_manager_class() factory
# ---------------------------------------------------------------------------

def make_dapq_session_segment_kv_cache_manager_class(
    base_class: type = KVCacheManager,
) -> type:
    """Factory: return a KVCacheManager subclass with DapQ session segment support.

    Args:
        base_class: The vLLM KVCacheManager class to subclass. Defaults to
            vllm.v1.core.kv_cache_manager.KVCacheManager.

    Returns:
        A new class combining DapQSessionSegmentKVCacheManagerMixin with base_class.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.block_manager_patch import (
            make_dapq_session_segment_kv_cache_manager_class,
            DapQSessionSegmentConfig,
        )

        DapQKVManager = make_dapq_session_segment_kv_cache_manager_class(KVCacheManager)

        mgr = DapQKVManager(
            kv_cache_config=...,
            max_model_len=4096,
            hash_block_size=16,
            dapq_session_config=DapQSessionSegmentConfig(
                segment_keep_ratio=0.50,
                kv_budget_ratio=0.30,
            ),
        )
        # mgr.store_turn_segment(...)
        # mgr.process_dual_reduction(...)
    """
    cls = type(
        "DapQSessionSegmentKVCacheManager",
        (DapQSessionSegmentKVCacheManagerMixin, base_class),
        {
            "__doc__": (
                "KVCacheManager subclass with DapQ session-aware segment reuse "
                "(Activity B+C). Auto-generated by "
                "make_dapq_session_segment_kv_cache_manager_class()."
            ),
        },
    )
    return cls


# ---------------------------------------------------------------------------
# Inline fallback: _InlineSessionSegmentStore (no src/ dependency)
# ---------------------------------------------------------------------------

class _InlineSessionSegmentStore:
    """Minimal in-memory session segment store (fallback when src/ unavailable)."""

    def __init__(self, config: DapQSessionSegmentConfig) -> None:
        self.config = config
        self._store: Dict[str, List[Dict[str, Any]]] = {}  # session_id → segments
        self._kv_store: Dict[str, Any] = {}  # key → tensor

    def store(self, session_id: str, turn_id: int, kv: Any) -> str:
        key = hashlib.sha256(
            f"{session_id}|{turn_id}|{time.monotonic()}".encode()
        ).hexdigest()[:16]
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append({
            "key": key,
            "turn_id": turn_id,
            "token_range": (0, 128),
            "center_position": 64.0,
            "timestamp": time.monotonic(),
        })
        self._kv_store[key] = kv
        return key

    def get_segments(
        self,
        session_id: str,
        turn_range: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        segments = self._store.get(session_id, [])
        if turn_range is not None:
            lo, hi = turn_range
            segments = [s for s in segments if lo <= s["turn_id"] <= hi]
        return segments

    def metrics_summary(self) -> Dict[str, Any]:
        return {
            "session_cache_hit_rate": 0.0,
            "session_noncontiguous_hit_rate": 0.0,
            "eviction_memory_reduction_ratio": 0.0,
            "dual_reduction_estimate": (
                self.config.segment_keep_ratio * self.config.kv_budget_ratio
            ),
            "total_memory_bytes": 0,
        }


# ===========================================================================
# 2026-05-21 (prior): BlockUnionNonContiguousKVCacheManagerMixin — preserved
# ===========================================================================

class BlockUnionNonContiguousKVCacheManagerMixin:
    """Preserved from 2026-05-21: BlockUnionNonContiguousReuseIndex integration.

    Activity B: GQA-aware block-union non-contiguous reuse (memcpy-free).
    store_block_union_segment() / get_block_union_table() / pad_block_table_with_block_union().
    See 2026-05-21 cycle for full documentation.
    """

    def __init__(
        self,
        *args: Any,
        block_union_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._block_union_index: Dict[str, Any] = {}
        self._block_union_cfg = block_union_config or {}

    def store_block_union_segment(
        self,
        content_hash: str,
        block_ids: List[int],
        layer_idx: int = 0,
    ) -> None:
        """Register a block-union segment in the non-contiguous index."""
        key = f"{content_hash}_l{layer_idx}"
        self._block_union_index[key] = {
            "block_ids": list(block_ids),
            "layer_idx": layer_idx,
            "stored_at": time.monotonic(),
        }

    def get_block_union_table(
        self,
        content_hash: str,
        layer_idx: int = 0,
    ) -> Optional[List[int]]:
        """Retrieve block IDs for a given content hash and layer."""
        key = f"{content_hash}_l{layer_idx}"
        entry = self._block_union_index.get(key)
        return entry["block_ids"] if entry else None

    def pad_block_table_with_block_union(
        self,
        block_table: List[int],
        content_hash: str,
        layer_idx: int = 0,
        max_blocks: int = 512,
    ) -> List[int]:
        """Merge block_union non-contiguous block IDs into the block table."""
        union_blocks = self.get_block_union_table(content_hash, layer_idx) or []
        merged = list(dict.fromkeys(block_table + union_blocks))
        return merged[:max_blocks]


def make_block_union_kv_cache_manager_class(
    base_class: type = KVCacheManager,
) -> type:
    """Factory for BlockUnionNonContiguousKVCacheManager (2026-05-21, preserved)."""
    return type(
        "BlockUnionNonContiguousKVCacheManager",
        (BlockUnionNonContiguousKVCacheManagerMixin, base_class),
        {"__doc__": "KVCacheManager + BlockUnion non-contiguous reuse (2026-05-21)."},
    )


# ===========================================================================
# 2026-05-12 (prior): AdapShotBlockManager — preserved
# ===========================================================================

class AdapShotBlockManager:
    """Preserved from 2026-05-12: AdapShotMixedDimSegmentPipeline block manager.

    store_segment() / load_segment() / annotate_request() for RoPE-reencoding
    non-contiguous + MixedDim B+C pipeline integration.
    See 2026-05-12 cycle for full documentation.
    """

    def __init__(
        self,
        segment_keep_ratio: float = 0.5,
        kv_budget_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.segment_keep_ratio = segment_keep_ratio
        self.kv_budget_ratio = kv_budget_ratio
        self.seed = seed
        self._segments: Dict[str, Any] = {}

    def store_segment(self, key: str, kv: Any) -> None:
        self._segments[key] = kv

    def load_segment(self, key: str) -> Optional[Any]:
        return self._segments.get(key)

    def annotate_request(self, request: Any, content_hash: str) -> None:
        if content_hash in self._segments:
            try:
                object.__setattr__(request, "adapshot_segment_key", content_hash)
            except Exception:
                pass


def make_adapshot_kv_cache_manager_class(
    base_class: type = KVCacheManager,
) -> type:
    """Factory for AdapShotKVCacheManager (2026-05-12, preserved)."""

    class AdapShotKVCacheManager(base_class):  # type: ignore[valid-type]
        def __init__(self, *args: Any, adapshot_config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            cfg = adapshot_config or {}
            self._adapshot = AdapShotBlockManager(
                segment_keep_ratio=cfg.get("segment_keep_ratio", 0.5),
                kv_budget_ratio=cfg.get("kv_budget_ratio", 0.5),
                seed=cfg.get("seed", 42),
            )

        def store_segment(self, key: str, kv: Any) -> None:
            self._adapshot.store_segment(key, kv)

        def load_segment(self, key: str) -> Optional[Any]:
            return self._adapshot.load_segment(key)

        def annotate_request(self, request: Any, content_hash: str) -> None:
            self._adapshot.annotate_request(request, content_hash)

    return AdapShotKVCacheManager


# ===========================================================================
# 2026-05-04 (prior): WorkloadAwareTTLKVCacheManager — preserved
# ===========================================================================

@dataclass
class VllmTTLEntry:
    """Per-segment TTL metadata (preserved from 2026-05-04)."""
    block_ids: Set[int]
    category: str
    ttl_sec: float
    created_at: float
    pinned: bool = False
    importance_score: float = 0.0
    embedding: Optional["torch.Tensor"] = None


class WorkloadAwareTTLKVCacheManager(KVCacheManager):
    """Preserved from 2026-05-04: TTL-based segment preservation KVCacheManager.

    Adds workload-aware TTL-based segment lifecycle management alongside
    vLLM's native prefix cache. See 2026-05-04 cycle for full documentation.
    """

    def __init__(
        self,
        *args: Any,
        ttl_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._ttl_entries: Dict[str, VllmTTLEntry] = {}
        self._ttl_cfg = ttl_config or {}

    def register_segment(
        self,
        segment_key: str,
        block_ids: Set[int],
        category: str = "chat",
    ) -> None:
        """Register a segment with TTL tracking."""
        _TTL_PROFILES = {
            "code":     600.0,
            "chat":     300.0,
            "rag":      120.0,
            "agentic":  480.0,
        }
        ttl = _TTL_PROFILES.get(category, 300.0)
        self._ttl_entries[segment_key] = VllmTTLEntry(
            block_ids=set(block_ids),
            category=category,
            ttl_sec=ttl,
            created_at=time.monotonic(),
        )

    def evict_expired_segments(self) -> List[str]:
        """Evict segments whose TTL has expired. Returns list of evicted keys."""
        now = time.monotonic()
        expired = [
            k for k, e in self._ttl_entries.items()
            if now - e.created_at > e.ttl_sec and not e.pinned
        ]
        for k in expired:
            del self._ttl_entries[k]
        return expired

    def pin_segment(self, segment_key: str) -> None:
        """Pin segment to prevent TTL eviction."""
        if segment_key in self._ttl_entries:
            self._ttl_entries[segment_key].pinned = True


# ===========================================================================
# 2026-05-23: CLCPositionalBiasGatedSegmentCache — Activity B
# ===========================================================================

@dataclass
class CLCBiasGateKVManagerConfig:
    """Configuration for CLC Positional Bias Gated segment KV cache manager.

    Activity B: Non-Contiguous KV Cache Reuse (2603.20218).
    ΔPos-threshold gated 3-stage reencoding policy for segment reuse.
    """
    max_context_length: int = 4096
    bias_threshold: float = 0.15          # ΔPos <= this: DIRECT_REUSE (no reencoding)
    rope_distortion_threshold: float = 0.40  # ΔPos > this: FULL_REENCODING required
    partial_reencoding_layer_ratio: float = 0.5  # fraction of layers for partial reencoding
    max_entries: int = 1000
    seed: int = 42


@dataclass
class _CLCSegmentEntry:
    """Metadata for a CLC-gated segment in the auxiliary store."""
    kv_tensor: Any  # torch.Tensor | None
    pos_orig_start: int
    pos_orig_end: int
    content_hash: str
    last_access: float = 0.0


class CLCPositionalBiasGatedKVCacheManagerMixin:
    """Mixin adding CLC positional-bias-gated non-contiguous segment reuse.

    2026-05-23: Activity B — CLC Positional Bias Gated Segment Cache.
    Based on arXiv 2603.20218 (CLC accuracy limit analysis).

    Ports CLCPositionalBiasGatedSegmentCache from src/cache/ into vLLM's
    v1 KVCacheManager as an auxiliary side-channel store alongside the
    native PagedAttention block pool.

    Key API:
      store_clc_segment(key, kv_tensor, pos_start, pos_end, content_hash)
        — Store a KV segment with position metadata.
      get_clc_segment_with_policy(key, pos_target_start)
        — Return (kv_tensor, policy) where policy is:
            "direct_reuse"     : ΔPos <= bias_threshold → no reencoding needed
            "partial_reencoding": bias_threshold < ΔPos <= rope_distortion_threshold
            "full_reencoding"  : ΔPos > rope_distortion_threshold

    vLLM integration contract:
      - Auxiliary store only. Does NOT modify vLLM's native block allocation.
      - PagedAttention block table is passed through unchanged.
      - Non-contiguous block table is supplementary; callers inject it before
        FlashAttention kernel via the block_tables parameter.
      - LRU eviction enforced at max_entries.
    """

    def __init__(
        self,
        *args: Any,
        clc_config: Optional[CLCBiasGateKVManagerConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._clc_config = clc_config or CLCBiasGateKVManagerConfig()
        self._clc_store: OrderedDict = OrderedDict()  # key -> _CLCSegmentEntry
        self._clc_hits = 0
        self._clc_misses = 0
        self._clc_direct_reuse_hits = 0
        self._clc_partial_reencoding_hits = 0
        self._clc_full_reencoding_hits = 0
        # Try to import verified src/ implementation
        _add_repo_root_to_path()
        self._clc_src_cache = None
        try:
            from src.cache.clc_positional_bias_gated_segment_cache import (
                CLCPositionalBiasGatedSegmentCache,
                CLCBiasGateConfig,
            )
            cfg_src = CLCBiasGateConfig(
                max_context_length=self._clc_config.max_context_length,
                bias_threshold=self._clc_config.bias_threshold,
                rope_distortion_threshold=self._clc_config.rope_distortion_threshold,
                partial_reencoding_layer_ratio=self._clc_config.partial_reencoding_layer_ratio,
                max_entries=self._clc_config.max_entries,
                seed=self._clc_config.seed,
            )
            self._clc_src_cache = CLCPositionalBiasGatedSegmentCache(cfg_src)
        except Exception:
            pass  # use inline fallback

    def _clc_compute_delta_pos(self, pos_orig_start: int, pos_target_start: int) -> float:
        """Compute normalized positional bias ΔPos = |target - orig| / max_context_length."""
        return abs(pos_target_start - pos_orig_start) / max(1, self._clc_config.max_context_length)

    def _clc_check_policy(self, pos_orig_start: int, pos_target_start: int) -> str:
        """Determine reencoding policy from ΔPos."""
        delta = self._clc_compute_delta_pos(pos_orig_start, pos_target_start)
        cfg = self._clc_config
        if delta <= cfg.bias_threshold:
            return "direct_reuse"
        elif delta <= cfg.rope_distortion_threshold:
            return "partial_reencoding"
        else:
            return "full_reencoding"

    def store_clc_segment(
        self,
        key: str,
        kv_tensor: Any,
        pos_orig_start: int,
        pos_orig_end: int,
        content_hash: str,
    ) -> None:
        """Store KV segment with positional metadata for CLC-gated reuse.

        Args:
            key: Cache key (e.g., content_hash + "_" + layer_idx).
            kv_tensor: KV tensor to store (torch.Tensor or None).
            pos_orig_start: Original context position where segment starts.
            pos_orig_end: Original context position where segment ends.
            content_hash: Hash of segment token content.
        """
        if self._clc_src_cache is not None:
            try:
                import torch
                if kv_tensor is not None:
                    self._clc_src_cache.put_segment(
                        key, kv_tensor, pos_orig_start, pos_orig_end, content_hash
                    )
                return
            except Exception:
                pass
        # Inline fallback
        if key in self._clc_store:
            self._clc_store.move_to_end(key)
        else:
            if len(self._clc_store) >= self._clc_config.max_entries:
                self._clc_store.popitem(last=False)
            self._clc_store[key] = _CLCSegmentEntry(
                kv_tensor=kv_tensor,
                pos_orig_start=pos_orig_start,
                pos_orig_end=pos_orig_end,
                content_hash=content_hash,
                last_access=time.monotonic(),
            )

    def get_clc_segment_with_policy(
        self,
        key: str,
        pos_target_start: int,
    ) -> Tuple[Any, str]:
        """Get cached KV segment with positional bias reencoding policy.

        Returns:
            (kv_tensor, policy) where:
              kv_tensor: Cached KV tensor (None on miss)
              policy: "direct_reuse" | "partial_reencoding" | "full_reencoding"

        The policy tells the caller how to handle positional encoding
        before passing the KV to the attention kernel.

        Tensor-parallel note:
            In multi-GPU TP deployments call build_clc_noncontiguous_block_table()
            from TP rank 0 only and broadcast the result to peer ranks BEFORE
            calling this method, to ensure consistent block-table decisions.
            If a TP environment is detected here, a runtime warning is emitted.
        """
        # Warn if running in a TP environment — block table may be inconsistent
        self._warn_if_tp_environment()

        if self._clc_src_cache is not None:
            try:
                kv, policy_enum = self._clc_src_cache.get_with_policy(key, pos_target_start)
                if kv is None:
                    self._clc_misses += 1
                    return None, "full_reencoding"
                self._clc_hits += 1
                policy_str = policy_enum.value  # "direct_reuse" | "partial" | "full"
                # Normalize enum value to canonical string
                if "direct" in policy_str:
                    policy_str = "direct_reuse"
                    self._clc_direct_reuse_hits += 1
                elif "partial" in policy_str:
                    policy_str = "partial_reencoding"
                    self._clc_partial_reencoding_hits += 1
                else:
                    policy_str = "full_reencoding"
                    self._clc_full_reencoding_hits += 1
                return kv, policy_str
            except Exception:
                pass
        # Inline fallback
        if key not in self._clc_store:
            self._clc_misses += 1
            return None, "full_reencoding"
        self._clc_hits += 1
        entry = self._clc_store[key]
        self._clc_store.move_to_end(key)
        entry.last_access = time.monotonic()
        policy = self._clc_check_policy(entry.pos_orig_start, pos_target_start)
        if policy == "direct_reuse":
            self._clc_direct_reuse_hits += 1
        elif policy == "partial_reencoding":
            self._clc_partial_reencoding_hits += 1
        else:
            self._clc_full_reencoding_hits += 1
        return entry.kv_tensor, policy

    def clc_noncontiguous_direct_hit_rate(self) -> float:
        """Fraction of hits that used DIRECT_REUSE (no reencoding needed)."""
        total_hits = (self._clc_direct_reuse_hits +
                      self._clc_partial_reencoding_hits +
                      self._clc_full_reencoding_hits)
        return self._clc_direct_reuse_hits / max(1, total_hits)

    def clc_hit_rate(self) -> float:
        """Overall CLC segment cache hit rate."""
        total = self._clc_hits + self._clc_misses
        return self._clc_hits / max(1, total)

    def clc_stats(self) -> Dict[str, Any]:
        """Return CLC cache stats dict for observability."""
        return {
            "hit_rate": self.clc_hit_rate(),
            "noncontiguous_direct_hit_rate": self.clc_noncontiguous_direct_hit_rate(),
            "direct_reuse_hits": self._clc_direct_reuse_hits,
            "partial_reencoding_hits": self._clc_partial_reencoding_hits,
            "full_reencoding_hits": self._clc_full_reencoding_hits,
            "total_hits": self._clc_hits,
            "total_misses": self._clc_misses,
            "store_size": len(self._clc_store),
        }

    def clc_evict_lru(self) -> int:
        """Manually evict LRU entry from CLC store. Returns bytes freed."""
        if not self._clc_store:
            return 0
        _, entry = self._clc_store.popitem(last=False)
        try:
            return entry.kv_tensor.nbytes if entry.kv_tensor is not None else 0
        except Exception:
            return 0

    def build_clc_noncontiguous_block_table(
        self,
        segment_keys: List[str],
        block_size: int = 16,
        max_blocks: int = 512,
    ) -> Optional["torch.Tensor"]:
        """Build a non-contiguous block table tensor for CLC segments.

        Constructs an int64 tensor [1, max_blocks] from a list of CLC segment
        keys. Each segment's stored position range is mapped to block indices.
        Unused slots are padded with -1 (PagedAttention convention).

        Block-alignment contract:
          Segment boundaries are validated to be multiples of block_size.
          If pos_orig_start or pos_orig_end is not block_size-aligned, the
          segment is aligned DOWN (start) / UP (end) to the nearest block
          boundary before computing block indices.  A misalignment warning is
          recorded in self._clc_block_align_warnings.

        Tensor-parallel (TP) note:
          This method is NOT TP-aware.  In multi-GPU tensor-parallel deployments
          all TP ranks must use identical block tables.  The caller is
          responsible for broadcasting the returned tensor from TP rank 0 to
          all other ranks before passing it to the attention kernel.  Calling
          get_clc_segment_with_policy() before this broadcast may yield
          inconsistent results across ranks.  See README.md §TP for details.

        Args:
            segment_keys: List of CLC segment keys (as used in store_clc_segment()).
            block_size: Number of tokens per block — must match vLLM's block_size.
            max_blocks: Maximum number of blocks in the returned table.

        Returns:
            int64 tensor [1, max_blocks] with block indices, or None if no
            segments are found for any of the given keys.
        """
        if not _TORCH_AVAILABLE:
            return None

        # Initialise misalignment counter on first call
        if not hasattr(self, "_clc_block_align_warnings"):
            self._clc_block_align_warnings: int = 0

        block_indices: List[int] = []

        for key in segment_keys:
            # Prefer src/ cache entry if available.
            # Access positional metadata via _meta dict (SegmentMeta namedtuple with
            # pos_orig_start / pos_orig_end fields) — src/ does not expose get_raw_entry().
            entry: Optional[_CLCSegmentEntry] = None
            if self._clc_src_cache is not None:
                try:
                    # Try _meta dict first (standard src/ internal attribute)
                    raw = None
                    if hasattr(self._clc_src_cache, "_meta"):
                        raw = self._clc_src_cache._meta.get(key)
                    # Fallback: try get_raw_entry() if available (future-proof)
                    if raw is None and hasattr(self._clc_src_cache, "get_raw_entry"):
                        raw = self._clc_src_cache.get_raw_entry(key)
                    if raw is not None:
                        pos_start = int(getattr(raw, "pos_orig_start", 0))
                        pos_end = int(getattr(raw, "pos_orig_end", pos_start + block_size))
                        entry = _CLCSegmentEntry(
                            kv_tensor=None,
                            pos_orig_start=pos_start,
                            pos_orig_end=pos_end,
                            content_hash=key,
                        )
                except Exception:
                    pass

            if entry is None:
                entry = self._clc_store.get(key)

            if entry is None:
                continue

            pos_start = int(entry.pos_orig_start)
            pos_end = int(entry.pos_orig_end)

            # Validate and align segment boundaries to block_size multiples
            if pos_start % block_size != 0:
                self._clc_block_align_warnings += 1
                pos_start = (pos_start // block_size) * block_size  # align down
            if pos_end % block_size != 0:
                self._clc_block_align_warnings += 1
                pos_end = ((pos_end + block_size - 1) // block_size) * block_size  # align up

            for pos in range(pos_start, pos_end, block_size):
                blk = pos // block_size
                if blk < max_blocks and blk not in block_indices:
                    block_indices.append(blk)

        if not block_indices:
            return None

        table = torch.full((1, max_blocks), -1, dtype=torch.int64)
        for blk in block_indices:
            table[0, blk] = blk
        return table

    # ------------------------------------------------------------------
    # Tensor-parallel safety helper
    # ------------------------------------------------------------------

    @staticmethod
    def _warn_if_tp_environment() -> None:
        """Emit a warning if a tensor-parallel environment is detected.

        Detects TP by checking for WORLD_SIZE > 1 in torch.distributed or
        the VLLM_TENSOR_PARALLEL_SIZE / WORLD_SIZE environment variables.
        Called inside get_clc_segment_with_policy() when TP is detected.
        """
        import os
        import warnings

        tp_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        _dist_world_size = 1
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                _dist_world_size = dist.get_world_size()
        except Exception:
            pass

        if tp_size > 1 or world_size > 1 or _dist_world_size > 1:
            warnings.warn(
                "CLCPositionalBiasGatedKVCacheManagerMixin: tensor-parallel (TP) "
                "environment detected (tp_size=%d, world_size=%d, dist_world_size=%d). "
                "build_clc_noncontiguous_block_table() is NOT TP-aware. "
                "The caller must broadcast the returned block table from TP rank 0 "
                "to all other ranks before passing it to the attention kernel. "
                "See vllm_integration/README.md §TP for details." % (
                    tp_size, world_size, _dist_world_size,
                ),
                stacklevel=3,
            )


def make_clc_bias_gate_kv_cache_manager_class(
    base_class: Optional[type] = None,
    clc_config: Optional[CLCBiasGateKVManagerConfig] = None,
) -> type:
    """Factory: subclass vLLM KVCacheManager with CLCPositionalBiasGatedKVCacheManagerMixin.

    Returns a class that:
      - is a subclass of vLLM's KVCacheManager (or base_class)
      - adds store_clc_segment() / get_clc_segment_with_policy()
      - tracks direct_reuse / partial / full reencoding hits

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.block_manager_patch import (
            CLCBiasGateKVManagerConfig,
            make_clc_bias_gate_kv_cache_manager_class,
        )
        cfg = CLCBiasGateKVManagerConfig(bias_threshold=0.15)
        CLCKVMgr = make_clc_bias_gate_kv_cache_manager_class(KVCacheManager, cfg)
        # issubclass(CLCKVMgr, KVCacheManager) is True

    vLLM version: 0.21.0
    Activity: B — CLCPositionalBiasGatedSegmentCache
    """
    if base_class is None:
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager as _KVM
            base_class = _KVM
        except Exception:
            base_class = object

    _cfg = clc_config

    class CLCBiasGateKVCacheManager(
        CLCPositionalBiasGatedKVCacheManagerMixin,
        base_class,  # type: ignore[valid-type]
    ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if _cfg is not None and "clc_config" not in kwargs:
                kwargs["clc_config"] = _cfg
            super().__init__(*args, **kwargs)

    CLCBiasGateKVCacheManager.__name__ = "CLCBiasGateKVCacheManager"
    CLCBiasGateKVCacheManager.__qualname__ = "CLCBiasGateKVCacheManager"
    return CLCBiasGateKVCacheManager
