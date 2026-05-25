"""KV Packet Cache — Activity B (2026-05-25).

Zero-FLOPs non-contiguous KV reuse via soft-token adapter self-supervised distillation.
Optimized for VeriCache integration (KVPacketSoftTokenAdapterZeroFLOPsNonContiguousCache).
arXiv 2604.13226.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class KVPacketConfig:
    n_heads: int = 8
    d_head: int = 128
    n_adapter_tokens: int = 4       # Soft-token adapter tokens (KV Packet paper default)
    adapter_lr: float = 1e-4        # AdamW learning rate
    adapter_steps: int = 100        # Adapter training steps
    distillation_loss_threshold: float = 0.1   # Low-quality packet eviction threshold
    max_packets: int = 512
    seed: int = 42


@dataclass
class KVPacket:
    """VeriCache-integrated KV Packet.

    kv_data: [n_tokens, 2, n_heads, d_head] FP16 immutable KV block
    adapter_K: [n_adapter_tokens, n_heads, d_head] soft-token K adapter
    adapter_V: [n_adapter_tokens, n_heads, d_head] soft-token V adapter
    distillation_loss: adapter training loss (eviction priority)
    segment_id: segment hash key
    """
    segment_id: str
    kv_data: torch.Tensor              # [n_tokens, 2, n_heads, d_head]
    adapter_K: torch.Tensor            # [n_adapter_tokens, n_heads, d_head]
    adapter_V: torch.Tensor            # [n_adapter_tokens, n_heads, d_head]
    distillation_loss: float = 1.0
    n_reuses: int = 0


class KVPacketCache(CacheStore):
    """KV Packet (arXiv 2604.13226) zero-FLOPs non-contiguous reuse cache.

    VeriCache-integration-optimized version (independent from kv_packet_adapter.py).

    Key differences vs. kv_packet_adapter.py:
      - get_for_vericache(): returns K and V separately for VeriCacheSpeculativeCodec.put_kv_pair()
      - assemble_multi(): inserts adapter tokens at each segment boundary to absorb discontinuities
      - n_adapter_tokens=4 (KV Packet paper default, vs. kv_packet_adapter.py rank=8)
      - Eviction policy: LRU + distillation_loss > threshold priority eviction

    put(key, value): store kv_data [n_tokens, 2, n_heads, d_head], initialize adapter
    get(key): return adapter-applied KV [n_adapter_tokens+n_tokens, 2, n_heads, d_head]
    get_for_vericache(key): return (K, V) tuple for VeriCache integration
    train_adapter(key, context_sample): self-supervised distillation (recompute FLOPs = 0)
    assemble_multi(keys): non-contiguous multi-packet assembly
    """

    def __init__(self, config: KVPacketConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: OrderedDict[str, KVPacket] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0
        self._access_order: List[str] = []

    # ---- CacheStore interface ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store kv_data + initialize soft-token adapters.

        Args:
            key: segment identifier
            value: kv_data [n_tokens, 2, n_heads, d_head] FP16
        """
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.config.max_packets:
            self.evict()
        c = self.config
        # Initialize soft-token adapters (n_adapter_tokens × n_heads × d_head)
        adapter_K = torch.randn(c.n_adapter_tokens, c.n_heads, c.d_head) * 0.02
        adapter_V = torch.randn(c.n_adapter_tokens, c.n_heads, c.d_head) * 0.02
        packet = KVPacket(
            segment_id=key,
            kv_data=value.detach().clone().to(torch.float16),
            adapter_K=adapter_K,
            adapter_V=adapter_V,
        )
        self._store[key] = packet

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return adapter-applied KV [n_adapter_tokens+n_tokens, 2, n_heads, d_head].

        Recompute FLOPs = 0: adapter weights are trained once, then only looked up.
        """
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        packet = self._store[key]
        packet.n_reuses += 1
        self._track_noncontiguous(key)
        return self._apply_adapter(packet)

    def get_for_vericache(
        self,
        key: str,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return (K, V) tuple for VeriCache integration.

        Returns:
            (K [n_adapter_tokens+n_tokens, n_heads, d_head],
             V [n_adapter_tokens+n_tokens, n_heads, d_head]) or None on miss
        """
        adapted = self.get(key)
        if adapted is None:
            return None
        K = adapted[:, 0, :, :]   # [n_adapter_tokens+n_tokens, n_heads, d_head]
        V = adapted[:, 1, :, :]
        return K, V

    def evict(self) -> int:
        """LRU + distillation_loss priority eviction.

        Algorithm:
          1. Evict LRU entry among those with distillation_loss > threshold
          2. If none, fall back to LRU eviction (first item in OrderedDict)
        """
        if not self._store:
            return 0
        threshold = self.config.distillation_loss_threshold
        low_quality = [
            k for k, p in self._store.items()
            if p.distillation_loss > threshold
        ]
        evict_key = low_quality[0] if low_quality else next(iter(self._store))
        packet = self._store.pop(evict_key)
        return packet.kv_data.nbytes + packet.adapter_K.nbytes + packet.adapter_V.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        total = 0
        for p in self._store.values():
            total += p.kv_data.nbytes + p.adapter_K.nbytes + p.adapter_V.nbytes
        return total

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._access_order.clear()

    # ---- KV Packet-specific API ----

    def store_packet(self, key: str, K: torch.Tensor, V: torch.Tensor) -> None:
        """Store KV block with adapter initialization.

        Convenience method that assembles kv_data from separate K and V tensors.

        Args:
            key: segment identifier
            K: key tensor [n_tokens, n_heads, d_head]
            V: value tensor [n_tokens, n_heads, d_head]
        """
        # Stack K and V into [n_tokens, 2, n_heads, d_head]
        kv_data = torch.stack([K, V], dim=1)
        self.put(key, kv_data)

    def retrieve_packet(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve adapter-applied KV (zero-FLOPs reuse).

        Alias for get() with explicit zero-FLOPs documentation.
        """
        return self.get(key)

    def train_adapter(
        self,
        key: str,
        context_kv: torch.Tensor,     # Reference KV [m_ref, 2, n_heads, d_head]
        n_steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> float:
        """Self-supervised distillation to train soft-token adapters.

        Objective (KV Packet paper method):
          L_adapter = ||attn_output(kv_packet + adapter, q) - attn_output(kv_ref, q)||_F
          q: 16 randomly sampled reference queries from context_kv

        No recomputation: only adapter parameters are updated. kv_data is immutable.

        Returns:
            Final training loss (distillation_loss)
        """
        if key not in self._store:
            return float("inf")
        packet = self._store[key]
        steps = n_steps or self.config.adapter_steps
        learning_rate = lr or self.config.adapter_lr

        adapter_K = nn.Parameter(packet.adapter_K.float().clone())
        adapter_V = nn.Parameter(packet.adapter_V.float().clone())
        optimizer = torch.optim.AdamW([adapter_K, adapter_V], lr=learning_rate)

        kv_data_f = packet.kv_data.float()
        ctx_f = context_kv.float()

        final_loss = float("inf")
        for _ in range(steps):
            # Sample 16 random reference queries
            n_ref = min(16, ctx_f.shape[0])
            ref_idx = torch.randperm(ctx_f.shape[0])[:n_ref]
            q_ref = ctx_f[ref_idx, 0, 0, :]   # [n_ref, d_head] using K channel, first head

            # Adapter-applied KV: [n_adapter_tokens + n_tokens, n_heads, d_head]
            full_K = torch.cat([adapter_K, kv_data_f[:, 0, :, :]], dim=0)
            full_V = torch.cat([adapter_V, kv_data_f[:, 1, :, :]], dim=0)

            # Reference KV (no adapter)
            ref_K = ctx_f[:, 0, :, :]  # [m_ref_full, n_heads, d_head]
            ref_V = ctx_f[:, 1, :, :]

            # Attention output comparison (first head only for efficiency)
            pred_out = self._attn_single_head(q_ref, full_K[:, 0, :], full_V[:, 0, :])
            target_out = self._attn_single_head(q_ref, ref_K[:, 0, :], ref_V[:, 0, :])

            loss = F.mse_loss(pred_out, target_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        packet.adapter_K = adapter_K.detach().to(packet.adapter_K.dtype)
        packet.adapter_V = adapter_V.detach().to(packet.adapter_V.dtype)
        packet.distillation_loss = final_loss
        return final_loss

    def assemble_multi(
        self,
        keys: List[str],
    ) -> Optional[torch.Tensor]:
        """Non-contiguous multi-packet assembly (zero-FLOPs recomputation).

        Inserts adapter tokens at each packet boundary to absorb context discontinuities.

        Returns:
            Assembled KV [sum(n_adapter_tokens + n_tokens_i), 2, n_heads, d_head]
            None if any key is a miss
        """
        parts: List[torch.Tensor] = []
        for k in keys:
            if k not in self._store:
                return None
            packet = self._store[k]
            parts.append(self._apply_adapter(packet))
        return torch.cat(parts, dim=0) if parts else None

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous."""
        if self._hits == 0:
            return 0.0
        return self._noncontiguous_hits / self._hits

    # ---- Internal helpers ----

    def _apply_adapter(self, packet: KVPacket) -> torch.Tensor:
        """Insert soft-token adapter before KV data. FLOPs = tensor lookup + concat.

        Returns [n_adapter_tokens + n_tokens, 2, n_heads, d_head].
        """
        # adapter_K/V: [n_adapter_tokens, n_heads, d_head] -> [n_adapter_tokens, 2, n_heads, d_head]
        adapter_kv = torch.stack([packet.adapter_K, packet.adapter_V], dim=1)
        return torch.cat([adapter_kv.to(packet.kv_data.dtype), packet.kv_data], dim=0)

    def _track_noncontiguous(self, key: str) -> None:
        """Track non-contiguous hits based on store insertion order."""
        if self._access_order:
            prev = self._access_order[-1]
            keys_list = list(self._store.keys())
            if key in keys_list and prev in keys_list:
                if abs(keys_list.index(key) - keys_list.index(prev)) > 1:
                    self._noncontiguous_hits += 1
            else:
                self._noncontiguous_hits += 1
        self._access_order.append(key)

    @staticmethod
    def _attn_single_head(
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [n_kv, d_head]
        V: torch.Tensor,   # [n_kv, d_head]
    ) -> torch.Tensor:
        """Single-head scaled dot-product attention."""
        scale = Q.size(-1) ** -0.5
        attn = F.softmax(Q @ K.T * scale, dim=-1)
        return attn @ V
