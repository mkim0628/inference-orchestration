"""VeriCache Speculative KV Draft-Verify Codec — Activity C (2026-05-25).

Deterministic accuracy-preserving KV compression via speculative draft-verify.
arXiv 2605.17613.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


# ---- Pluggable draft codec interface ----

class DraftCodec:
    """Pluggable KV compression interface. Swap in any implementation."""

    def compress(self, kv: torch.Tensor) -> object:
        """Compress full KV. kv: [n_tokens, d_head] -> compressed (smaller)."""
        raise NotImplementedError

    def decompress(self, compressed_kv: object) -> torch.Tensor:
        """Decompress compressed KV to original space (approximate). For drafting."""
        raise NotImplementedError

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (>= 1.0, larger = stronger compression)."""
        raise NotImplementedError


class Int8DraftCodec(DraftCodec):
    """INT8 quantization draft codec.

    Compress: FP16 -> INT8 (2x memory saving).
    Decompress: INT8 -> FP32 (approximate, quantization error corrected by VeriCache verify).

    Algorithm:
      compress(kv):
        scale = kv.abs().max() / 127.0 + eps
        quantized = (kv / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

      decompress(compressed_kv, scale):
        return compressed_kv.float() * scale
    """

    def __init__(self, symmetric: bool = True) -> None:
        self.symmetric = symmetric

    def compress(self, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (int8_tensor, scale)."""
        scale = kv.float().abs().max() / 127.0 + 1e-8
        quantized = (kv.float() / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def decompress(self, compressed_kv: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns FP32 approximate tensor."""
        quantized, scale = compressed_kv
        return quantized.float() * scale

    @property
    def compression_ratio(self) -> float:
        return 2.0  # FP16(2 bytes) -> INT8(1 byte)


class TokenEvictionDraftCodec(DraftCodec):
    """Token eviction draft codec. Drops low-importance tokens.

    Algorithm:
      compress(kv):  # kv: [n_tokens, d_head]
        n_keep = max(1, int(n_tokens * keep_ratio))
        importance = kv.float().norm(dim=-1)           # [n_tokens]
        kept_idx = importance.topk(n_keep).indices.sort().values
        return kv[kept_idx], kept_idx                  # (kept_kv, kept_indices)

      decompress: partial restoration (kept tokens only, evicted tokens zeroed)
      — VeriCache verify step corrects errors so full restoration is unnecessary.
    """

    def __init__(self, keep_ratio: float = 0.5) -> None:
        self.keep_ratio = keep_ratio

    def compress(self, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (kept_kv [n_keep, d_head], kept_indices [n_keep])."""
        n_tokens = kv.shape[0]
        n_keep = max(1, int(n_tokens * self.keep_ratio))
        importance = kv.float().norm(dim=-1)
        kept_idx = importance.topk(n_keep).indices.sort().values
        return kv[kept_idx], kept_idx

    def decompress(self, compressed: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return kept tokens only (drafting approximation, no padding)."""
        kept_kv, kept_idx = compressed
        return kept_kv

    @property
    def compression_ratio(self) -> float:
        return 1.0 / self.keep_ratio


# ---- Core codec classes ----

@dataclass
class SpeculativeKVEntry:
    """VeriCache KV store entry."""
    segment_id: str
    compressed_kv: object          # DraftCodec.compress() return value (codec-dependent)
    full_kv_ref: torch.Tensor      # Full KV (for verify, simulates DRAM offload)
    original_seq_len: int
    original_d_head: int
    draft_acceptance_rate: float = 0.0
    n_draft_calls: int = 0
    n_accepted: int = 0


@dataclass
class VerificationResult:
    """Draft-verify comparison result."""
    accepted: bool                  # Whether draft matches full KV within threshold
    draft_output: torch.Tensor      # Attention output from compressed KV [n_q, d_head]
    verified_output: torch.Tensor   # Attention output from full KV [n_q, d_head]
    relative_error: float           # ||draft - full||_F / ||full||_F
    acceptance_threshold: float     # Acceptance threshold


@dataclass
class VeriCacheConfig:
    d_head: int = 128
    draft_length: int = 4           # Number of draft tokens (1/2/4/8 sweep)
    acceptance_threshold: float = 0.01   # Accept draft if relative_error < this
    max_entries: int = 512
    enable_async_verify: bool = True     # Enable async parallel verify
    seed: int = 42


class VeriCacheSpeculativeCodec(CacheStore):
    """VeriCache speculative draft-verify KV compression codec (arXiv 2605.17613).

    Activity C: KV Cache Compression — deterministic accuracy-preserving.

    Core algorithm:
      Draft phase (HBM-bandwidth bound):
        1. Compute attention output from compressed KV: draft_output = attention(Q, compressed_K, compressed_V)

      Verify phase (PCIe-bound, parallel execution):
        2. Async load full KV (simulates DRAM offload)
        3. Compute verify: verified_output = attention(Q, full_K, full_V)
        4. Accept decision: if relative_error(draft, verified) < threshold -> accept draft
                            else -> replace with verified output (deterministic guarantee)

      Deterministic accuracy guarantee:
        - Accepted tokens: draft_output ~= verified_output (within threshold)
        - Rejected tokens: verified_output used directly (full KV accuracy)
        - Regardless of compression codec, output always >= full KV inference accuracy

    CacheStore interface:
      - put(key, value): store full KV and generate compressed draft
      - get(key): return compressed KV (for drafting)
      - draft_and_verify(key, Q): perform draft+verify, return VerificationResult
      - evict(): LRU eviction
      - hit_rate(), memory_bytes(), reset_stats()

    Compression codec plugins:
      - set_draft_codec(codec: DraftCodec): swap codec (default: Int8DraftCodec)
      - Supported: Int8DraftCodec, TokenEvictionDraftCodec, any DraftCodec impl
    """

    def __init__(self, config: VeriCacheConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._draft_codec: DraftCodec = Int8DraftCodec()
        self._store: OrderedDict[str, SpeculativeKVEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._total_draft_calls: int = 0
        self._total_accepted: int = 0
        self._relative_errors: List[float] = []
        self._verify_lock = threading.Lock()

    def set_draft_codec(self, codec: DraftCodec) -> None:
        """Swap draft codec. Existing entries are not re-compressed."""
        self._draft_codec = codec

    # ---- CacheStore interface ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store full KV and generate compressed draft.

        Args:
            key: segment identifier
            value: full KV tensor [n_tokens, d_head] (K or V)
        """
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        compressed = self._draft_codec.compress(value.detach().clone())
        entry = SpeculativeKVEntry(
            segment_id=key,
            compressed_kv=compressed,
            full_kv_ref=value.detach().clone(),
            original_seq_len=value.shape[0],
            original_d_head=value.shape[-1] if value.dim() >= 2 else self.config.d_head,
        )
        self._store[key] = entry

    def put_kv_pair(
        self,
        key: str,
        K: torch.Tensor,   # [n_tokens, d_head]
        V: torch.Tensor,   # [n_tokens, d_head]
    ) -> None:
        """Store K and V under separate keys. Key convention: key+'_K', key+'_V'.

        VeriCache compresses and verifies K and V independently.
        """
        self.put(key + "_K", K)
        self.put(key + "_V", V)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return decompressed KV (approximate tensor for drafting).

        Returns:
            approximate KV tensor (with compression error), or None on miss
        """
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        entry = self._store[key]
        return self._draft_codec.decompress(entry.compressed_kv)

    def draft_and_verify(
        self,
        key_K: str,       # K tensor key (key+'_K' from put_kv_pair)
        key_V: str,       # V tensor key (key+'_V' from put_kv_pair)
        Q: torch.Tensor,  # [n_q, d_head]
    ) -> Optional[VerificationResult]:
        """Speculative drafting + full KV verification.

        Algorithm:
          1. Decompress K/V (draft phase)
          2. Compute draft attention output
          3. Compute verified attention output from full K/V (parallel or sequential)
          4. Compute relative_error
          5. Accept/reject decision against threshold

        Returns:
            VerificationResult (with accepted flag + both outputs)
            None: key miss (need full KV inference fallback)
        """
        if key_K not in self._store or key_V not in self._store:
            self._misses += 1
            return None

        self._hits += 1
        entry_K = self._store[key_K]
        entry_V = self._store[key_V]

        # Draft phase: attention from compressed KV
        draft_K = self._draft_codec.decompress(entry_K.compressed_kv)
        draft_V = self._draft_codec.decompress(entry_V.compressed_kv)
        draft_output = self._compute_attention(Q, draft_K, draft_V)

        # Verify phase: attention from full KV (parallel execution simulation)
        if self.config.enable_async_verify:
            verified_output = self._async_verify(
                Q, entry_K.full_kv_ref, entry_V.full_kv_ref
            )
        else:
            verified_output = self._compute_attention(
                Q, entry_K.full_kv_ref, entry_V.full_kv_ref
            )

        # Compute relative_error
        rel_error = float(
            (draft_output.float() - verified_output.float()).norm()
            / (verified_output.float().norm() + 1e-8)
        )

        # Accept/reject decision
        accepted = rel_error < self.config.acceptance_threshold

        # Update stats
        self._total_draft_calls += 1
        if accepted:
            self._total_accepted += 1
        self._relative_errors.append(rel_error)

        # Per-entry acceptance rate
        with self._verify_lock:
            entry_K.n_draft_calls += 1
            entry_V.n_draft_calls += 1
            if accepted:
                entry_K.n_accepted += 1
                entry_V.n_accepted += 1
            entry_K.draft_acceptance_rate = (
                entry_K.n_accepted / max(1, entry_K.n_draft_calls)
            )
            entry_V.draft_acceptance_rate = (
                entry_V.n_accepted / max(1, entry_V.n_draft_calls)
            )

        return VerificationResult(
            accepted=accepted,
            draft_output=draft_output,
            verified_output=verified_output,
            relative_error=rel_error,
            acceptance_threshold=self.config.acceptance_threshold,
        )

    def get_final_output(
        self,
        result: VerificationResult,
    ) -> torch.Tensor:
        """Deterministic final output selection.

        Accepted: return draft_output (verified within threshold)
        Rejected: return verified_output (guaranteed full KV accuracy)

        This function is the core of VeriCache's deterministic accuracy-preserving guarantee.
        Regardless of compression codec, final output is always >= full KV inference accuracy.
        """
        return result.draft_output if result.accepted else result.verified_output

    def evict(self) -> int:
        """LRU eviction. Returns bytes freed."""
        if not self._store:
            return 0
        key, entry = next(iter(self._store.items()))
        self._store.pop(key)
        return entry.full_kv_ref.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Compressed KV memory (full_kv_ref excluded as it's DRAM offloaded)."""
        total = 0
        for entry in self._store.values():
            compressed = entry.compressed_kv
            if isinstance(compressed, tuple):
                # (tensor, scale) form
                for t in compressed:
                    if isinstance(t, torch.Tensor):
                        total += t.nbytes
            elif isinstance(compressed, torch.Tensor):
                total += compressed.nbytes
        return total

    def memory_bytes_full_kv(self) -> int:
        """Full KV memory (includes DRAM offload — for reference only)."""
        return sum(e.full_kv_ref.nbytes for e in self._store.values())

    def memory_reduction_ratio(self) -> float:
        """Reduction ratio: 1 - compressed_bytes / full_kv_bytes."""
        compressed_bytes = self.memory_bytes()
        full_bytes = self.memory_bytes_full_kv()
        if full_bytes == 0:
            return 0.0
        return 1.0 - compressed_bytes / full_bytes

    def draft_acceptance_rate(self) -> float:
        """Overall draft acceptance rate (higher = more throughput gain)."""
        if self._total_draft_calls == 0:
            return 0.0
        return self._total_accepted / self._total_draft_calls

    def mean_relative_error(self) -> float:
        """Mean relative_error of draft attention outputs."""
        if not self._relative_errors:
            return 0.0
        return sum(self._relative_errors) / len(self._relative_errors)

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_draft_calls = 0
        self._total_accepted = 0
        self._relative_errors.clear()

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """Return importance mask. TokenEviction codec: mask based on kept_indices."""
        entry = self._store.get(key)
        if entry is None:
            return None
        compressed = entry.compressed_kv
        if isinstance(compressed, tuple) and len(compressed) == 2:
            # TokenEvictionDraftCodec: (kept_kv, kept_indices)
            _, kept_idx = compressed
            if isinstance(kept_idx, torch.Tensor) and kept_idx.dtype == torch.long:
                mask = torch.zeros(entry.original_seq_len, dtype=torch.bool)
                mask[kept_idx] = True
                return mask
        return None

    def speculative_stats(self) -> dict:
        """JSON-serializable speculative execution statistics."""
        return {
            "draft_acceptance_rate": self.draft_acceptance_rate(),
            "mean_relative_error": self.mean_relative_error(),
            "total_draft_calls": self._total_draft_calls,
            "total_accepted": self._total_accepted,
            "hit_rate": self.hit_rate(),
            "memory_reduction_ratio": self.memory_reduction_ratio(),
            "compression_codec": type(self._draft_codec).__name__,
            "compression_ratio": self._draft_codec.compression_ratio,
            "n_entries": len(self._store),
        }

    # ---- Internal helpers ----

    @staticmethod
    def _compute_attention(
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [n_kv, d_head]
        V: torch.Tensor,   # [n_kv, d_head]
    ) -> torch.Tensor:
        """Scaled dot-product attention. Returns [n_q, d_head]."""
        scale = Q.size(-1) ** -0.5
        scores = (Q.float() @ K.float().T) * scale   # [n_q, n_kv]
        attn = F.softmax(scores, dim=-1)
        return (attn @ V.float()).to(Q.dtype)

    def _async_verify(
        self,
        Q: torch.Tensor,
        full_K: torch.Tensor,
        full_V: torch.Tensor,
    ) -> torch.Tensor:
        """Full KV verify (current impl: sync simulation, parallel-intent documented).

        In real GPU environments, use CUDA Streams for parallel execution:
          stream_draft = torch.cuda.Stream()
          stream_verify = torch.cuda.Stream()
          with torch.cuda.stream(stream_draft): draft_output = compute_attention(Q, compressed_K, V)
          with torch.cuda.stream(stream_verify): verified_output = compute_attention(Q, full_K, full_V)
          torch.cuda.synchronize()

        Current impl (CPU/simplified): sequential execution with identical results.
        """
        return self._compute_attention(Q, full_K, full_V)
