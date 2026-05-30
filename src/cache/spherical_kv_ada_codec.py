"""Activity C-1: Spherical KV Angle-Domain Attention (ADA) Codec.

Key decomposition: K[t] = r[t] × direction(θ[t])
Attention score without dense Key reconstruction:
  score[t] = r[t] × (Q · direction(θ[t]))

Reference: arXiv:2605.18856
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from src.cache.base import CacheStore


@dataclass
class ADAConfig:
    angle_quantize_bits: int = 8
    radius_dtype: torch.dtype = torch.float16
    max_entries: int = 1000
    seed: int = 42


@dataclass
class ADAKVEntry:
    K_angle: torch.Tensor
    K_radius: torch.Tensor
    V: torch.Tensor
    n_tokens: int
    original_seq_len: int


class SphericalKVADACodec(CacheStore):
    """Angle-Domain Attention: Key spherical parameterisation compression codec (C-1).

    K[t] = r[t] × direction(θ[t])
    score[t] = r[t] × (Q · direction(θ[t]))  — no dense Key reconstruction needed.

    angle_quantize_bits=8 uses INT8; 4 would use INT4 (simulated via float scaling).
    """

    def __init__(self, config: ADAConfig) -> None:
        self.config = config
        self._store: OrderedDict[str, ADAKVEntry] = OrderedDict()
        self._plain_store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        torch.manual_seed(config.seed)

    # ------------------------------------------------------------------ #
    # ADA core operations                                                  #
    # ------------------------------------------------------------------ #

    def decompose(self, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """K [n_tokens, d_head] → (K_radius [n_tokens], K_direction [n_tokens, d_head]).

        K_radius[t] = ||K[t]||_2
        K_direction[t] = K[t] / (K_radius[t] + 1e-8)   (unit vector)
        """
        K_f = K.float()
        K_radius = K_f.norm(dim=-1)  # [n_tokens]
        K_direction = K_f / (K_radius.unsqueeze(-1) + 1e-8)  # [n_tokens, d_head]
        return K_radius, K_direction

    def quantize_angle(self, K_direction: torch.Tensor) -> torch.Tensor:
        """Unit vector → INT8 angle codes.

        Linearly maps [-1, 1] → [-127, 127].
        Returns: [n_tokens, d_head] int8
        """
        # Clamp to valid unit-vector range before quantization
        clamped = K_direction.float().clamp(-1.0, 1.0)
        quantized = (clamped * 127.0).round().clamp(-127, 127).to(torch.int8)
        return quantized

    def dequantize_angle(self, K_angle_int8: torch.Tensor) -> torch.Tensor:
        """INT8 angle codes → FP32 approximate unit vector.

        Returns: [n_tokens, d_head] float32
        """
        direction = K_angle_int8.float() / 127.0
        # Re-normalise to restore unit-vector property after quantisation error
        norms = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return direction / norms

    def reconstruct_score(
        self,
        Q: torch.Tensor,
        K_angle_int8: torch.Tensor,
        K_radius: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention logits without dense Key reconstruction.

        score[q, t] = K_radius[t] × (Q[q] · dequantize_angle(K_angle[t]))

        Returns: [n_q, n_kv] float32
        """
        direction = self.dequantize_angle(K_angle_int8)  # [n_kv, d_head]
        # Q: [n_q, d_head]; direction: [n_kv, d_head]
        dot = Q.float() @ direction.T  # [n_q, n_kv]
        # Scale by radius
        score = dot * K_radius.float().unsqueeze(0)  # [n_q, n_kv]
        return score

    # ------------------------------------------------------------------ #
    # ADA storage API                                                      #
    # ------------------------------------------------------------------ #

    def put_ada(self, key: str, K: torch.Tensor, V: torch.Tensor) -> ADAKVEntry:
        """ADA decomposition then compressed storage.

        Steps:
          1. decompose(K) → K_radius, K_direction
          2. quantize_angle(K_direction) → K_angle INT8
          3. Store ADAKVEntry
        """
        K_radius, K_direction = self.decompose(K)
        K_angle = self.quantize_angle(K_direction)

        entry = ADAKVEntry(
            K_angle=K_angle,
            K_radius=K_radius.to(self.config.radius_dtype),
            V=V.detach().clone(),
            n_tokens=K.shape[0],
            original_seq_len=K.shape[0],
        )

        if len(self._store) >= self.config.max_entries:
            self.evict()
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = entry
        return entry

    # ------------------------------------------------------------------ #
    # CacheStore abstract methods                                          #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Plain tensor storage for compatibility."""
        if key in self._plain_store:
            self._plain_store.move_to_end(key)
        else:
            total = len(self._store) + len(self._plain_store)
            if total >= self.config.max_entries:
                self.evict()
            self._plain_store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return V tensor from ADA entry; fall back to plain store."""
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key].V
        if key in self._plain_store:
            self._plain_store.move_to_end(key)
            self._hits += 1
            return self._plain_store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        """LRU eviction from ADA store first, then plain store."""
        if self._store:
            _, entry = self._store.popitem(last=False)
            return entry.K_angle.nbytes + entry.K_radius.nbytes + entry.V.nbytes
        if self._plain_store:
            _, tensor = self._plain_store.popitem(last=False)
            return tensor.nbytes
        return 0

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        ada_bytes = sum(
            e.K_angle.nbytes + e.K_radius.nbytes + e.V.nbytes
            for e in self._store.values()
        )
        plain_bytes = sum(v.nbytes for v in self._plain_store.values())
        return ada_bytes + plain_bytes

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    def get_entry(self, key: str) -> Optional[ADAKVEntry]:
        """Return full ADAKVEntry for inspection."""
        return self._store.get(key)

    def memory_reduction_ratio(self) -> float:
        """Mean compression ratio across stored ADA entries.

        Original: n_tokens × d_head × 2 bytes (FP16 K)
        Compressed: K_angle (INT8: 1 byte/elem) + K_radius (FP16: 2 bytes/token)
        """
        if not self._store:
            return 0.0
        ratios = []
        for entry in self._store.values():
            n = entry.n_tokens
            d = entry.K_angle.shape[1] if entry.K_angle.dim() > 1 else 1
            orig_k_bytes = n * d * 2  # FP16 Key
            comp_k_bytes = entry.K_angle.nbytes + entry.K_radius.nbytes
            reduction = 1.0 - comp_k_bytes / max(1, orig_k_bytes)
            ratios.append(reduction)
        return float(sum(ratios) / len(ratios))
