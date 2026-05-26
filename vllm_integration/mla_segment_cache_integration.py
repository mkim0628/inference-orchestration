"""Activity B-1: Irminsul MLA Segment Cache — vLLM v1 Integration Adapter.

Adapts IrminsulMLASegmentCache (src/cache/irminsul_mla_segment_cache.py) to
vLLM 0.21.0's v1 architecture (KVCacheManager / block_pool / prefix caching).

Integration approach: monkey-patch mixin on top of vLLM's KVCacheManager.
Does NOT modify vLLM source; all hooks are installed at runtime via
`install_irminsul_mla_hooks(manager)`.

vLLM version: 0.21.0
Activity: B-1 (Non-Contiguous MLA KV Cache Reuse with δ-rotation)
Source: src/cache/irminsul_mla_segment_cache.py
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Inline CDC + δ-rotation (no src/ import to keep vllm_integration standalone)
# ---------------------------------------------------------------------------


def _cdc_chunk(
    token_ids: List[int],
    avg_chunk_size: int = 256,
    min_chunk_size: int = 64,
    max_chunk_size: int = 1024,
) -> List[List[int]]:
    """Rabin fingerprint-based CDC chunking (position-independent boundaries)."""
    modulus = avg_chunk_size - 1
    chunks: List[List[int]] = []
    start = 0
    n = len(token_ids)
    BASE = 31
    MOD = 2**32

    while start < n:
        end = start
        current_len = 0
        fp = 0
        while end < n:
            fp = (fp * BASE + token_ids[end]) % MOD
            current_len += 1
            end += 1
            if current_len >= min_chunk_size and (fp & modulus) == 0:
                break
            if current_len >= max_chunk_size:
                break
        chunks.append(token_ids[start:end])
        start = end

    return chunks


def _cdc_segment_key(chunk_tokens: List[int]) -> str:
    """Position-independent SHA256 hash of chunk token content."""
    raw = struct.pack(f"{len(chunk_tokens)}I", *chunk_tokens)
    return hashlib.sha256(raw).hexdigest()


def _apply_delta_rotation(
    k_r: torch.Tensor,
    delta: int,
    rope_base: float = 10000.0,
    k_r_dim: int = 64,
) -> torch.Tensor:
    """Apply δ-rotation to MLA k_r to correct for position shift.

    Mathematical basis (Irminsul arXiv 2605.05696):
      RoPE rotates (k[2i], k[2i+1]) by pos * θ_i.
      A segment stored at source_pos reused at target_pos requires an
      additional rotation by (target_pos - source_pos) * θ_i for k_r only.
      c_kv is position-free and requires no correction.
    """
    if delta == 0:
        return k_r

    n_tokens, dim = k_r.shape
    half_dim = dim // 2

    i_vals = torch.arange(half_dim, dtype=torch.float32, device=k_r.device)
    theta = torch.pow(
        torch.tensor(rope_base, dtype=torch.float32, device=k_r.device),
        -2.0 * i_vals / k_r_dim,
    )
    delta_angles = delta * theta
    cos_a = torch.cos(delta_angles)
    sin_a = torch.sin(delta_angles)

    k_r_f = k_r.float().reshape(n_tokens, half_dim, 2)
    k0 = k_r_f[..., 0]
    k1 = k_r_f[..., 1]
    new_k0 = k0 * cos_a - k1 * sin_a
    new_k1 = k0 * sin_a + k1 * cos_a

    result = torch.stack([new_k0, new_k1], dim=-1)
    return result.reshape(n_tokens, dim).to(k_r.dtype)


# ---------------------------------------------------------------------------
# KV entry dataclass
# ---------------------------------------------------------------------------


@dataclass
class _IrminsulKVEntry:
    segment_key: str
    c_kv: torch.Tensor        # [n_tokens, d_c] position-free
    k_r: torch.Tensor         # [n_tokens, k_r_dim]
    source_position: int
    n_tokens: int
    layer_idx: int


# ---------------------------------------------------------------------------
# IrminsulMLASegmentMixin — attach to vLLM KVCacheManager
# ---------------------------------------------------------------------------


class IrminsulMLASegmentMixin:
    """Mixin that adds Irminsul MLA non-contiguous segment reuse to
    vLLM's KVCacheManager.

    vLLM's prefix caching (via block_pool hash matching) handles contiguous
    prefix reuse.  This mixin adds a *secondary* CDC-based segment store that
    also matches non-contiguous segments at arbitrary positions.

    Usage (attach at engine startup):
        install_irminsul_mla_hooks(kv_cache_manager_instance)

    The mixin stores:
        _irminsul_store: OrderedDict[(segment_key, layer_idx) → _IrminsulKVEntry]
        _irminsul_hits / _irminsul_misses / _irminsul_noncontiguous_hits
    """

    # Configuration (set by install_irminsul_mla_hooks)
    _irminsul_avg_chunk_size: int = 256
    _irminsul_min_chunk_size: int = 64
    _irminsul_max_chunk_size: int = 1024
    _irminsul_rope_base: float = 10000.0
    _irminsul_k_r_dim: int = 64
    _irminsul_max_entries: int = 2000

    def _irminsul_init(self) -> None:
        """Initialize segment store state. Called by install hook."""
        self._irminsul_store: OrderedDict[Tuple[str, int], _IrminsulKVEntry] = (
            OrderedDict()
        )
        self._irminsul_hits = 0
        self._irminsul_misses = 0
        self._irminsul_noncontiguous_hits = 0

    # -- Write API ---------------------------------------------------------

    def store_mla_segment(
        self,
        chunk_tokens: List[int],
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        source_position: int,
        layer_idx: int = 0,
    ) -> str:
        """Store MLA KV for a CDC chunk; returns content-hash segment_key.

        Maps to: IrminsulMLASegmentCache.put_mla_segment()
        """
        segment_key = _cdc_segment_key(chunk_tokens)
        store_key = (segment_key, layer_idx)

        if store_key in self._irminsul_store:
            self._irminsul_store.move_to_end(store_key)
            return segment_key

        # LRU eviction
        while len(self._irminsul_store) >= self._irminsul_max_entries:
            self._irminsul_store.popitem(last=False)

        entry = _IrminsulKVEntry(
            segment_key=segment_key,
            c_kv=c_kv.detach().clone(),
            k_r=k_r.detach().clone(),
            source_position=source_position,
            n_tokens=len(chunk_tokens),
            layer_idx=layer_idx,
        )
        self._irminsul_store[store_key] = entry
        return segment_key

    # -- Read API ----------------------------------------------------------

    def find_mla_segment(
        self,
        segment_key: str,
        target_position: int,
        layer_idx: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (c_kv, k_r_corrected) with δ-rotation applied.

        Maps to: IrminsulMLASegmentCache.get_mla_segment_with_delta_rotation()
        Returns None on miss.
        """
        store_key = (segment_key, layer_idx)
        if store_key not in self._irminsul_store:
            self._irminsul_misses += 1
            return None

        self._irminsul_store.move_to_end(store_key)
        self._irminsul_hits += 1

        entry = self._irminsul_store[store_key]
        delta = target_position - entry.source_position
        k_r_corrected = _apply_delta_rotation(
            entry.k_r, delta, self._irminsul_rope_base, self._irminsul_k_r_dim
        )
        return entry.c_kv, k_r_corrected

    def find_noncontiguous_mla_hits(
        self,
        token_ids: List[int],
        target_offset: int = 0,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], List[List[int]]]:
        """CDC chunk token_ids and look up each chunk in the segment store.

        This is the core non-contiguous reuse entry point called by the
        vLLM model runner after vLLM's own prefix cache lookup (which handles
        contiguous prefixes).  We handle arbitrary non-contiguous hits.

        Returns:
            hits: [(chunk_local_idx, c_kv, k_r_corrected), ...]
            miss_chunks: [[token_ids...], ...]  — chunks requiring full compute

        Maps to: IrminsulMLASegmentCache.get_segments_mla()
        """
        chunks = _cdc_chunk(
            token_ids,
            avg_chunk_size=self._irminsul_avg_chunk_size,
            min_chunk_size=self._irminsul_min_chunk_size,
            max_chunk_size=self._irminsul_max_chunk_size,
        )

        hits: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        miss_chunks: List[List[int]] = []
        position = target_offset

        for chunk_idx, chunk in enumerate(chunks):
            segment_key = _cdc_segment_key(chunk)
            result = self.find_mla_segment(segment_key, position, layer_idx)
            if result is not None:
                c_kv, k_r_corrected = result
                hits.append((chunk_idx, c_kv, k_r_corrected))
                if miss_chunks:
                    self._irminsul_noncontiguous_hits += 1
            else:
                miss_chunks.append(chunk)
            position += len(chunk)

        return hits, miss_chunks

    # -- Stats API ---------------------------------------------------------

    def irminsul_noncontiguous_hit_rate(self) -> float:
        """Fraction of MLA segment hits that are non-contiguous."""
        if self._irminsul_hits == 0:
            return 0.0
        return self._irminsul_noncontiguous_hits / self._irminsul_hits

    def irminsul_hit_rate(self) -> float:
        total = self._irminsul_hits + self._irminsul_misses
        return self._irminsul_hits / total if total > 0 else 0.0

    def irminsul_memory_bytes(self) -> int:
        return sum(
            e.c_kv.nbytes + e.k_r.nbytes for e in self._irminsul_store.values()
        )

    def irminsul_reset_stats(self) -> None:
        self._irminsul_hits = 0
        self._irminsul_misses = 0
        self._irminsul_noncontiguous_hits = 0


# ---------------------------------------------------------------------------
# install_irminsul_mla_hooks — runtime attachment to KVCacheManager
# ---------------------------------------------------------------------------


def install_irminsul_mla_hooks(
    manager: object,
    avg_chunk_size: int = 256,
    min_chunk_size: int = 64,
    max_chunk_size: int = 1024,
    rope_base: float = 10000.0,
    k_r_dim: int = 64,
    max_entries: int = 2000,
) -> None:
    """Attach IrminsulMLASegmentMixin methods to a live KVCacheManager instance.

    This is a non-destructive monkey-patch: no vLLM source is modified.
    After this call, `manager` gains:
        store_mla_segment(chunk_tokens, c_kv, k_r, source_position, layer_idx)
        find_mla_segment(segment_key, target_position, layer_idx)
        find_noncontiguous_mla_hits(token_ids, target_offset, layer_idx)
        irminsul_noncontiguous_hit_rate()
        irminsul_hit_rate()
        irminsul_memory_bytes()
        irminsul_reset_stats()

    The vLLM-side prefix caching (block_pool hash matching) is unaffected.
    """
    mixin = IrminsulMLASegmentMixin

    # Bind configuration
    manager._irminsul_avg_chunk_size = avg_chunk_size  # type: ignore[attr-defined]
    manager._irminsul_min_chunk_size = min_chunk_size  # type: ignore[attr-defined]
    manager._irminsul_max_chunk_size = max_chunk_size  # type: ignore[attr-defined]
    manager._irminsul_rope_base = rope_base  # type: ignore[attr-defined]
    manager._irminsul_k_r_dim = k_r_dim  # type: ignore[attr-defined]
    manager._irminsul_max_entries = max_entries  # type: ignore[attr-defined]

    # Bind methods
    import types
    for method_name in [
        "_irminsul_init",
        "store_mla_segment",
        "find_mla_segment",
        "find_noncontiguous_mla_hits",
        "irminsul_noncontiguous_hit_rate",
        "irminsul_hit_rate",
        "irminsul_memory_bytes",
        "irminsul_reset_stats",
    ]:
        setattr(
            manager,
            method_name,
            types.MethodType(getattr(mixin, method_name), manager),
        )

    # Initialize state
    manager._irminsul_init()  # type: ignore[attr-defined]
