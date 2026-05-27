"""IndexMem Eviction Codec — Learnable Indexer + Latent Memory (Activity C-1).

Integrates IndexMemLearnableIndexer and IndexMemLatentMemoryModule into a
unified CompressionCodec + DraftCodec compatible codec.
arXiv 2605.25475.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from src.cache.compression import CompressionCodec
from src.cache.vericache_speculative_codec import DraftCodec
from src.cache.indexmem_learnable_indexer import (
    IndexMemLearnableIndexer,
    LearnableIndexerConfig,
)
from src.cache.indexmem_latent_memory_module import (
    IndexMemLatentMemoryModule,
    LatentMemoryConfig,
)


@dataclass
class IndexMemEvictionConfig:
    budget_ratio: float = 0.5
    base_eviction_policy: str = "snapkv"   # "h2o" | "snapkv" | "dapq" | "learned"
    alpha_ema: float = 0.3
    beta_readout: float = 0.1
    latent_dim: int = 64
    n_layers: int = 32
    kv_dim: int = 128
    zero_shot_mode: bool = False
    max_accuracy_delta: float = 0.01
    fallback_budget_ratio: float = 0.6
    fallback_beta: float = 0.05
    seed: int = 42


class IndexMemEvictionCodec(CompressionCodec, DraftCodec):
    """IndexMem Learnable Indexer + Latent Memory eviction codec.

    Activity C: "eviction = latent compression + conditional restoration" paradigm.

    Dual interface:
      - CompressionCodec: encode()/decode()/compression_ratio()
      - DraftCodec: compress()/decompress()/compression_ratio_float
    """

    def __init__(self, config: IndexMemEvictionConfig) -> None:
        # CompressionCodec expects num_layers; pass 1 as dummy (we override methods)
        super().__init__(num_layers=config.n_layers)
        self.config = config

        self.indexer = IndexMemLearnableIndexer(
            LearnableIndexerConfig(
                zero_shot_mode=config.zero_shot_mode,
                seed=config.seed,
            )
        )
        self.latent_memory = IndexMemLatentMemoryModule(
            LatentMemoryConfig(
                latent_dim=config.latent_dim,
                n_layers=config.n_layers,
                kv_dim=config.kv_dim,
                alpha_ema=config.alpha_ema,
                beta_readout=config.beta_readout,
                seed=config.seed,
            )
        )

        self._encode_count: int = 0
        self._total_tokens_in: int = 0
        self._total_tokens_kept: int = 0
        self._auto_key_counter: int = 0

    # ---- CompressionCodec interface ----

    def encode(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
        query: Optional[torch.Tensor] = None,
        token_positions: Optional[torch.Tensor] = None,
        current_position: int = 0,
        request_key: str = "",
    ) -> torch.Tensor:
        """Compress KV: Learnable Indexer selects tokens, evicted tokens → latent memory."""
        n_tokens = kv.shape[0]
        d_head = kv.shape[-1]

        # Split kv into separate key and value (assume kv is concatenated or just use as key)
        # For simplicity, treat kv as key state; duplicate for value if needed
        k = kv
        v = kv

        if query is None:
            query = kv.mean(dim=0)

        if token_positions is None:
            token_positions = torch.arange(n_tokens, dtype=torch.float32)

        seg_key = request_key or f"auto_{tensor_id}_{layer_idx}"

        retention_prob = self.indexer.predict(
            k, v, query, token_positions, current_position, seg_key
        )
        kept_idx, evict_idx = self.indexer.select_tokens_by_budget(
            retention_prob, self.config.budget_ratio
        )

        if evict_idx.numel() > 0:
            self.latent_memory.encode_evicted(kv[evict_idx], seg_key, layer_idx)

        self._encode_count += 1
        self._total_tokens_in += n_tokens
        self._total_tokens_kept += kept_idx.numel()

        return kv[kept_idx]

    def decode(
        self,
        compressed: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Return compressed KV as-is; full restoration is via residual readout."""
        return compressed

    def compression_ratio(self, layer_idx: int) -> float:
        """Theoretical compression: 1 / budget_ratio."""
        return 1.0 / self.config.budget_ratio

    # ---- DraftCodec interface ----

    def compress(self, kv: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """DraftCodec compress interface (Cross-2 VeriCache integration).

        Returns (compressed_kv, request_key).
        """
        self._auto_key_counter += 1
        request_key = f"draft_{id(kv)}_{self._auto_key_counter}"
        compressed = self.encode(kv, layer_idx=0, request_key=request_key)
        return compressed, request_key

    def decompress(self, compressed_kv: object) -> torch.Tensor:
        """DraftCodec decompress: return approximate KV with latent readout if applicable."""
        if isinstance(compressed_kv, tuple) and len(compressed_kv) == 2:
            compressed, _ = compressed_kv
            return self.decode(compressed, layer_idx=0)
        return self.decode(compressed_kv, layer_idx=0)

    @property
    def compression_ratio_float(self) -> float:
        """DraftCodec compression_ratio property."""
        return 1.0 / self.config.budget_ratio

    # Override DraftCodec.compression_ratio property to avoid conflict
    # (DraftCodec uses property, CompressionCodec uses method with args)

    # ---- Readout API ----

    def get_readout(
        self,
        query_states: torch.Tensor,
        layer_idx: int = 0,
        request_key: str = "",
    ) -> torch.Tensor:
        """Latent memory residual readout. Usage: attn_output += get_readout(...)"""
        return self.latent_memory.residual_readout(query_states, request_key, layer_idx)

    # ---- Accuracy fallback ----

    def auto_adjust_on_accuracy_delta(self, accuracy_delta: float) -> bool:
        """Raise budget_ratio and reduce beta if accuracy delta exceeds threshold.

        Returns True if adjustment was made.
        """
        if abs(accuracy_delta) > self.config.max_accuracy_delta:
            self.config.budget_ratio = self.config.fallback_budget_ratio
            self.latent_memory.config.beta_readout = self.config.fallback_beta
            return True
        return False

    # ---- Statistics ----

    def compression_stats(self) -> dict:
        """JSON-serializable compression statistics."""
        avg_kept = (
            self._total_tokens_kept / self._total_tokens_in
            if self._total_tokens_in > 0
            else 0.0
        )
        return {
            "encode_count": self._encode_count,
            "total_tokens_in": self._total_tokens_in,
            "total_tokens_kept": self._total_tokens_kept,
            "effective_budget_ratio": avg_kept,
            "configured_budget_ratio": self.config.budget_ratio,
            "compression_ratio": self.compression_ratio(0),
            "latent_memory_bytes": self.latent_memory.latent_memory_bytes(),
            "base_eviction_policy": self.config.base_eviction_policy,
            "zero_shot_mode": self.config.zero_shot_mode,
        }
