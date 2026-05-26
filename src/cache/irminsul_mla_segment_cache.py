"""Activity B-1: Irminsul MLA-native position-independent non-contiguous KV cache.

MLA c_KV/k_r separation + δ-rotation position correction + CDC chunking.
Reference: Irminsul arXiv 2605.05696
"""

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore


# ------------------------------------------------------------------ #
# CDC Chunking (Content-Defined Chunking via Rabin fingerprint)       #
# ------------------------------------------------------------------ #


def cdc_chunk(
    token_ids: List[int],
    avg_chunk_size: int = 256,
    min_chunk_size: int = 64,
    max_chunk_size: int = 1024,
    window_size: int = 32,
    modulus: int = 0,
) -> List[List[int]]:
    """Rabin fingerprint-based CDC chunking.

    Splits token_ids into variable-length chunks whose boundaries are
    content-defined so that identical token sequences yield identical chunks
    regardless of their absolute position in the stream.
    """
    if modulus == 0:
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


def cdc_segment_key(chunk_tokens: List[int]) -> str:
    """Position-independent SHA256 hash of chunk token content."""
    raw = struct.pack(f"{len(chunk_tokens)}I", *chunk_tokens)
    return hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------ #
# δ-rotation: apply positional correction to MLA k_r component       #
# ------------------------------------------------------------------ #


def apply_delta_rotation(
    k_r: torch.Tensor,
    delta: int,
    rope_base: float = 10000.0,
    k_r_dim: int = 64,
) -> torch.Tensor:
    """Apply δ-rotation to MLA k_r to correct for position shift.

    Mathematical basis (Irminsul arXiv 2605.05696):
      RoPE rotates each pair (k[2i], k[2i+1]) by angle pos * θ_i.
      A segment cached at source_pos reused at target_pos needs an
      additional rotation by (target_pos - source_pos) * θ_i.

    Args:
        k_r: [n_tokens, k_r_dim] — k_r component at source position
        delta: target_position - source_position
        rope_base: RoPE base frequency
        k_r_dim: dimension of k_r (64 for MLA/DeepSeek)

    Returns:
        k_r_corrected: [n_tokens, k_r_dim] with position corrected
    """
    n_tokens, dim = k_r.shape
    half_dim = dim // 2

    # θ_i = rope_base^(-2i/k_r_dim), i = 0..half_dim-1
    i_vals = torch.arange(half_dim, dtype=torch.float32, device=k_r.device)
    theta = torch.pow(
        torch.tensor(rope_base, dtype=torch.float32, device=k_r.device),
        -2.0 * i_vals / k_r_dim,
    )

    # Angles for this delta
    delta_angles = delta * theta  # [half_dim]
    cos_a = torch.cos(delta_angles)  # [half_dim]
    sin_a = torch.sin(delta_angles)  # [half_dim]

    # Reshape k_r to [n_tokens, half_dim, 2]
    k_r_f = k_r.float().reshape(n_tokens, half_dim, 2)

    k0 = k_r_f[..., 0]  # [n_tokens, half_dim]
    k1 = k_r_f[..., 1]  # [n_tokens, half_dim]

    # 2D rotation in each frequency pair
    new_k0 = k0 * cos_a - k1 * sin_a
    new_k1 = k0 * sin_a + k1 * cos_a

    result = torch.stack([new_k0, new_k1], dim=-1)  # [n_tokens, half_dim, 2]
    return result.reshape(n_tokens, dim).to(k_r.dtype)


def assert_delta_rotation_correctness(
    k_r_dim: int = 64,
    rope_base: float = 10000.0,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Verify δ-rotation mathematical correctness.

    Constructs a raw k_r vector, applies RoPE at source_pos to get k_r_src,
    then applies δ-rotation and checks it matches k_r applied at target_pos.
    """
    torch.manual_seed(42)
    n_tokens = 8
    source_pos = 100
    target_pos = 250
    delta = target_pos - source_pos

    half_dim = k_r_dim // 2
    i_vals = torch.arange(half_dim, dtype=torch.float32)
    theta = torch.pow(torch.tensor(rope_base, dtype=torch.float32), -2.0 * i_vals / k_r_dim)

    # Raw (pre-RoPE) k_r
    k_raw = torch.randn(n_tokens, k_r_dim)

    def apply_rope_at(k: torch.Tensor, pos: int) -> torch.Tensor:
        angles = pos * theta
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        k_r = k.reshape(n_tokens, half_dim, 2)
        k0, k1 = k_r[..., 0], k_r[..., 1]
        new_k0 = k0 * cos_a - k1 * sin_a
        new_k1 = k0 * sin_a + k1 * cos_a
        return torch.stack([new_k0, new_k1], dim=-1).reshape(n_tokens, k_r_dim)

    k_at_source = apply_rope_at(k_raw, source_pos)
    k_at_target = apply_rope_at(k_raw, target_pos)
    k_corrected = apply_delta_rotation(k_at_source, delta, rope_base, k_r_dim)

    assert torch.allclose(k_corrected, k_at_target, rtol=rtol, atol=atol), (
        f"δ-rotation correctness failed: max diff = {(k_corrected - k_at_target).abs().max():.6f}"
    )


# ------------------------------------------------------------------ #
# Data classes                                                        #
# ------------------------------------------------------------------ #


@dataclass
class IrminsulKVEntry:
    segment_key: str          # SHA256(token_content) — position-independent key
    c_kv: torch.Tensor        # [n_tokens, d_c] position-free compressed representation
    k_r: torch.Tensor         # [n_tokens, k_r_dim] RoPE component
    source_position: int      # original storage position offset (for δ calculation)
    n_tokens: int
    layer_idx: int


@dataclass
class IrminsulMLAConfig:
    avg_chunk_size: int = 256
    min_chunk_size: int = 64
    max_chunk_size: int = 1024
    rope_base: float = 10000.0
    k_r_dim: int = 64
    max_entries: int = 2000
    seed: int = 42


# ------------------------------------------------------------------ #
# IrminsulMLASegmentCache                                             #
# ------------------------------------------------------------------ #


class IrminsulMLASegmentCache(CacheStore):
    """MLA-native position-independent non-contiguous KV cache (Activity B-1).

    CDC chunking + SHA256 content-hash keying for position-independent indexing.
    On reuse, δ-rotation is applied only to k_r; c_kv is reused as-is.

    Fully implements the CacheStore interface.
    """

    def __init__(self, config: IrminsulMLAConfig) -> None:
        self.config = config
        # Keyed by (segment_key, layer_idx) → IrminsulKVEntry, ordered for LRU
        self._store: OrderedDict[Tuple[str, int], IrminsulKVEntry] = OrderedDict()
        # Generic tensor store for base CacheStore put/get API
        self._tensor_store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        torch.manual_seed(config.seed)

    # ---------------------------------------------------------------- #
    # CacheStore abstract methods                                       #
    # ---------------------------------------------------------------- #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store tensor under generic key (base CacheStore API)."""
        if key in self._tensor_store:
            self._tensor_store.move_to_end(key)
        else:
            if len(self._tensor_store) >= self.config.max_entries:
                self._tensor_store.popitem(last=False)
            self._tensor_store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve tensor by generic key (base CacheStore API)."""
        if key in self._tensor_store:
            self._tensor_store.move_to_end(key)
            self._hits += 1
            return self._tensor_store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        """Evict the least-recently-used MLA segment; returns bytes freed."""
        if self._store:
            _, entry = self._store.popitem(last=False)
            return entry.c_kv.nbytes + entry.k_r.nbytes
        if self._tensor_store:
            _, tensor = self._tensor_store.popitem(last=False)
            return tensor.nbytes
        return 0

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        mla_bytes = sum(e.c_kv.nbytes + e.k_r.nbytes for e in self._store.values())
        tensor_bytes = sum(v.nbytes for v in self._tensor_store.values())
        return mla_bytes + tensor_bytes

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # ---------------------------------------------------------------- #
    # MLA-specific API                                                  #
    # ---------------------------------------------------------------- #

    def put_mla_segment(
        self,
        chunk_tokens: List[int],
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        source_position: int,
        layer_idx: int = 0,
    ) -> str:
        """Store MLA KV for a CDC chunk; returns the content-hash segment_key."""
        segment_key = cdc_segment_key(chunk_tokens)
        store_key = (segment_key, layer_idx)

        if store_key in self._store:
            self._store.move_to_end(store_key)
            return segment_key

        # Evict if at capacity
        while len(self._store) >= self.config.max_entries:
            self._store.popitem(last=False)

        entry = IrminsulKVEntry(
            segment_key=segment_key,
            c_kv=c_kv.detach().clone(),
            k_r=k_r.detach().clone(),
            source_position=source_position,
            n_tokens=len(chunk_tokens),
            layer_idx=layer_idx,
        )
        self._store[store_key] = entry
        return segment_key

    def get_mla_segment_with_delta_rotation(
        self,
        segment_key: str,
        target_position: int,
        layer_idx: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (c_kv, k_r_corrected) after applying δ-rotation.

        c_kv is position-free and returned unchanged.
        k_r_corrected = apply_delta_rotation(k_r, target_position - source_position).
        Returns None on cache miss.
        """
        store_key = (segment_key, layer_idx)
        if store_key not in self._store:
            self._misses += 1
            return None

        self._store.move_to_end(store_key)
        self._hits += 1

        entry = self._store[store_key]
        delta = target_position - entry.source_position

        if delta == 0:
            k_r_corrected = entry.k_r
        else:
            k_r_corrected = apply_delta_rotation(
                entry.k_r,
                delta,
                self.config.rope_base,
                self.config.k_r_dim,
            )

        return entry.c_kv, k_r_corrected

    def get_segments_mla(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], List[List[int]]]:
        """CDC chunk all token_ids, then look up each chunk.

        Returns:
            hits: [(chunk_local_idx, c_kv, k_r_corrected), ...]
            miss_chunks: [[token_ids...], ...] — chunks requiring recomputation
        """
        chunks = cdc_chunk(
            token_ids,
            avg_chunk_size=self.config.avg_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
        )

        hits: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        miss_chunks: List[List[int]] = []
        position = target_offset

        for chunk_idx, chunk in enumerate(chunks):
            segment_key = cdc_segment_key(chunk)
            result = self.get_mla_segment_with_delta_rotation(
                segment_key, position, layer_idx
            )
            if result is not None:
                c_kv, k_r_corrected = result
                hits.append((chunk_idx, c_kv, k_r_corrected))
                # Track non-contiguous hits: hits where any earlier chunk is a miss
                if miss_chunks:
                    self._noncontiguous_hits += 1
            else:
                miss_chunks.append(chunk)
            position += len(chunk)

        return hits, miss_chunks

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of total hits that are non-contiguous."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits

    # InferenceRunner-compatible segment API
    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """InferenceRunner-compatible API (returns chunk indices not token lists)."""
        chunks = cdc_chunk(
            token_ids,
            avg_chunk_size=self.config.avg_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
        )

        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []
        position = 0

        for chunk_idx, chunk in enumerate(chunks):
            segment_key = cdc_segment_key(chunk)
            result = self.get_mla_segment_with_delta_rotation(
                segment_key, position, layer_idx
            )
            if result is not None:
                c_kv, k_r_corrected = result
                # Concatenate c_kv and k_r for the runner's generic tensor interface
                combined = torch.cat([c_kv, k_r_corrected], dim=-1)
                hits.append((chunk_idx, combined))
                if misses:
                    self._noncontiguous_hits += 1
            else:
                misses.append(chunk_idx)
            position += len(chunk)

        return hits, misses

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """InferenceRunner-compatible segment storage.

        Splits kv into c_kv / k_r components based on k_r_dim config.
        """
        chunks = cdc_chunk(
            token_ids,
            avg_chunk_size=self.config.avg_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
        )

        if chunk_idx >= len(chunks):
            return

        chunk = chunks[chunk_idx]
        n_tokens = kv.shape[0]
        d_c = kv.shape[-1] - self.config.k_r_dim

        if d_c <= 0:
            # Cannot split — store k_r only with zero c_kv
            c_kv = torch.zeros(n_tokens, 1, dtype=kv.dtype, device=kv.device)
            k_r = kv
        else:
            c_kv = kv[..., :d_c]
            k_r = kv[..., d_c:]

        # Compute position offset as sum of preceding chunk lengths
        position = sum(len(chunks[i]) for i in range(chunk_idx))
        self.put_mla_segment(chunk, c_kv, k_r, position, layer_idx)
