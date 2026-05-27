"""IndexMem Latent Memory Module — compressed latent state for evicted KV tokens (Activity C-1).

2-layer lightweight transformer encoder with online EMA updates and residual readout.
arXiv 2605.25475.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class LatentMemoryConfig:
    kv_dim: int = 128               # head_dim × n_heads
    latent_dim: int = 64            # latent state dimension
    n_layers: int = 32              # model layer count
    encoder_hidden_dim: int = 128   # lightweight transformer encoder hidden dim
    encoder_n_heads: int = 4
    encoder_ffn_dim: int = 256
    encoder_n_layers: int = 2       # transformer encoder layer count
    alpha_ema: float = 0.3          # online update EMA coefficient
    beta_readout: float = 0.1       # residual readout strength
    readout_threshold: float = 0.0  # readout activation threshold (0.0 = always active)
    seed: int = 42


class _TransformerEncoderLayer:
    """Single lightweight transformer encoder layer (manual implementation, no nn.Module)."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Self-attention projections
        self.W_q = torch.randn(hidden_dim, hidden_dim) * math.sqrt(2.0 / (hidden_dim * 2))
        self.W_k = torch.randn(hidden_dim, hidden_dim) * math.sqrt(2.0 / (hidden_dim * 2))
        self.W_v = torch.randn(hidden_dim, hidden_dim) * math.sqrt(2.0 / (hidden_dim * 2))
        self.W_o = torch.randn(hidden_dim, hidden_dim) * math.sqrt(2.0 / (hidden_dim * 2))

        # FFN
        self.W_ff1 = torch.randn(ffn_dim, hidden_dim) * math.sqrt(2.0 / (hidden_dim + ffn_dim))
        self.b_ff1 = torch.zeros(ffn_dim)
        self.W_ff2 = torch.randn(hidden_dim, ffn_dim) * math.sqrt(2.0 / (hidden_dim + ffn_dim))
        self.b_ff2 = torch.zeros(hidden_dim)

        # LayerNorm parameters (scale=1, bias=0)
        self.ln1_w = torch.ones(hidden_dim)
        self.ln1_b = torch.zeros(hidden_dim)
        self.ln2_w = torch.ones(hidden_dim)
        self.ln2_b = torch.zeros(hidden_dim)

    def _layer_norm(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (var + 1e-5).sqrt() * w + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, hidden_dim] -> [seq_len, hidden_dim]"""
        seq_len = x.shape[0]
        x_f = x.float()

        # Self-attention
        Q = x_f @ self.W_q.T  # [seq_len, hidden_dim]
        K = x_f @ self.W_k.T
        V = x_f @ self.W_v.T
        scale = math.sqrt(self.head_dim)
        scores = (Q @ K.T) / scale  # [seq_len, seq_len]
        attn = F.softmax(scores, dim=-1)
        attn_out = (attn @ V) @ self.W_o.T  # [seq_len, hidden_dim]
        x_f = self._layer_norm(x_f + attn_out, self.ln1_w, self.ln1_b)

        # FFN
        ffn = F.relu(x_f @ self.W_ff1.T + self.b_ff1)
        ffn = ffn @ self.W_ff2.T + self.b_ff2
        x_f = self._layer_norm(x_f + ffn, self.ln2_w, self.ln2_b)

        return x_f


class IndexMemLatentMemoryModule:
    """IndexMem Latent Memory Module — compressed latent state + residual readout.

    Architecture:
      latent_encoder: 2-layer lightweight transformer encoder
        hidden_dim=128, n_heads=4, ffn_dim=256
        input: evicted_kv [n_evicted, kv_dim] -> latent [n_layers, latent_dim]

    Latent state memory:
      n_layers × latent_dim × FP16 = 32 × 64 × 2 = 4KB/request

    Online update:
      new_latent = alpha × encode(evicted_kv) + (1 - alpha) × current_latent

    Residual readout:
      attn_output += beta × readout(query_states, latent_state)
    """

    def __init__(self, config: LatentMemoryConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._latent_states: Dict[str, torch.Tensor] = {}

        # Input projection: kv_dim -> encoder_hidden_dim
        self._input_proj = torch.randn(
            config.encoder_hidden_dim, config.kv_dim
        ) * math.sqrt(2.0 / (config.kv_dim + config.encoder_hidden_dim))

        # Transformer encoder layers
        self._encoder_layers = [
            _TransformerEncoderLayer(
                hidden_dim=config.encoder_hidden_dim,
                n_heads=config.encoder_n_heads,
                ffn_dim=config.encoder_ffn_dim,
                seed=config.seed + i,
            )
            for i in range(config.encoder_n_layers)
        ]

        # Output projection: encoder_hidden_dim -> latent_dim
        self._output_proj = torch.randn(
            config.latent_dim, config.encoder_hidden_dim
        ) * math.sqrt(2.0 / (config.encoder_hidden_dim + config.latent_dim))

        # Readout projections: latent_dim -> kv_dim
        self._readout_proj = torch.randn(
            config.kv_dim, config.latent_dim
        ) * math.sqrt(2.0 / (config.latent_dim + config.kv_dim))
        self._readout_key_proj = torch.randn(
            config.kv_dim, config.latent_dim
        ) * math.sqrt(2.0 / (config.latent_dim + config.kv_dim))

    def _encode_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Encode KV block to latent vector. kv: [n_tokens, kv_dim] -> [latent_dim]"""
        if kv.shape[0] == 0:
            return torch.zeros(self.config.latent_dim)

        x = kv.float() @ self._input_proj.T  # [n_tokens, encoder_hidden_dim]
        for layer in self._encoder_layers:
            x = layer.forward(x)
        # Pool to single vector
        pooled = x.mean(dim=0)  # [encoder_hidden_dim]
        latent = pooled @ self._output_proj.T  # [latent_dim]
        return latent

    def encode_evicted(
        self,
        evicted_kv: torch.Tensor,
        request_key: str,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Compress evicted tokens' KV to latent state with online EMA update.

        Returns:
            new_latent: [latent_dim]
        """
        encoded = self._encode_kv(evicted_kv)  # [latent_dim]

        # Initialize latent state for this request if needed
        if request_key not in self._latent_states:
            self._latent_states[request_key] = torch.zeros(
                self.config.n_layers, self.config.latent_dim
            )

        prev_latent = self._latent_states[request_key][layer_idx]
        alpha = self.config.alpha_ema
        new_latent = alpha * encoded + (1 - alpha) * prev_latent
        self._latent_states[request_key][layer_idx] = new_latent.detach()

        return new_latent

    def residual_readout(
        self,
        query_states: torch.Tensor,
        request_key: str,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Compute latent-state-based residual readout.

        attn_output += beta × readout(query_states, latent_state)

        Returns tensor with same shape as query_states.
        """
        if request_key not in self._latent_states:
            return torch.zeros_like(query_states)

        latent = self._latent_states[request_key]
        if layer_idx >= latent.shape[0]:
            return torch.zeros_like(query_states)

        latent_l = latent[layer_idx].float()  # [latent_dim]

        readout_v = latent_l @ self._readout_proj.T    # [kv_dim]
        readout_k = latent_l @ self._readout_key_proj.T  # [kv_dim]

        # Flatten query to [n_q, d_head]
        orig_shape = query_states.shape
        q_flat = query_states.float().reshape(-1, orig_shape[-1])  # [n_q, d_head]
        n_q = q_flat.shape[0]
        d_head = q_flat.shape[-1]

        # Readout via cross-attention against latent key/value
        # readout_k shape: [kv_dim]; treat as single key vector [1, kv_dim]
        # We dot each query row with readout_k
        attn_weight = torch.softmax(
            q_flat @ readout_k.unsqueeze(-1) / math.sqrt(d_head), dim=0
        )  # [n_q, 1]
        readout = attn_weight * readout_v.unsqueeze(0)  # [n_q, kv_dim]

        result = self.config.beta_readout * readout.reshape(orig_shape)
        return result.to(query_states.dtype)

    def get_latent_state(
        self, request_key: str, layer_idx: int = 0
    ) -> Optional[torch.Tensor]:
        """Return current latent state, or None if not present."""
        if request_key not in self._latent_states:
            return None
        latent = self._latent_states[request_key]
        if layer_idx >= latent.shape[0]:
            return None
        return latent[layer_idx]

    def clear(self, request_key: str) -> None:
        """Delete latent state after request completion."""
        self._latent_states.pop(request_key, None)

    def latent_memory_bytes(self) -> int:
        """Total latent state memory in bytes (FP32 tensor storage)."""
        total = 0
        for latent in self._latent_states.values():
            total += latent.nbytes
        return total
