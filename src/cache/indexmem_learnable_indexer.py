"""IndexMem Learnable Indexer — input-adaptive KV token importance predictor (Activity C-1).

225-parameter MLP with zero-shot DapQ fallback. arXiv 2605.25475.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml


@dataclass
class LearnableIndexerConfig:
    input_dim: int = 5               # [k_norm, v_norm, cumul_attn_score, position_decay, query_sim]
    hidden_dim: int = 32
    output_dim: int = 1              # retention_prob
    gamma_position_decay: float = 0.01
    ema_alpha_cumul_attn: float = 0.1
    zero_shot_mode: bool = True      # DapQ fallback by default
    weights_path: Optional[str] = None
    seed: int = 42


class IndexMemLearnableIndexer:
    """IndexMem Learnable Indexer — input-adaptive KV token importance predictor.

    Architecture:
      inputs = [k_norm, v_norm, cumul_attn_score, position_decay, query_sim]
      hidden = ReLU(Linear(5 → 32)(inputs))
      retention_prob = Sigmoid(Linear(32 → 1)(hidden))

    Parameter count: 5×32 + 32 + 32×1 + 1 = 225 (extremely lightweight)

    zero_shot_mode=True uses DapQ-style post-RoPE query similarity instead of
    learned weights: retention_prob = Sigmoid(query_sim × cumul_attn_score)
    """

    def __init__(self, config: LearnableIndexerConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._w1: torch.Tensor  # [32, 5]
        self._b1: torch.Tensor  # [32]
        self._w2: torch.Tensor  # [1, 32]
        self._b2: torch.Tensor  # [1]
        self._cumul_attn: Dict[str, torch.Tensor] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights; load from file if weights_path is set."""
        torch.manual_seed(self.config.seed)
        # Xavier initialization scaled for 5→32→1 MLP
        scale1 = math.sqrt(2.0 / (self.config.input_dim + self.config.hidden_dim))
        scale2 = math.sqrt(2.0 / (self.config.hidden_dim + self.config.output_dim))
        self._w1 = torch.randn(self.config.hidden_dim, self.config.input_dim) * scale1
        self._b1 = torch.zeros(self.config.hidden_dim)
        self._w2 = torch.randn(self.config.output_dim, self.config.hidden_dim) * scale2
        self._b2 = torch.zeros(self.config.output_dim)

        if self.config.weights_path is not None:
            try:
                self.load_weights(self.config.weights_path)
            except (FileNotFoundError, KeyError):
                pass  # Use initialized weights if file missing or malformed

    def _get_or_init_cumul_attn(self, segment_key: str, n_tokens: int) -> torch.Tensor:
        if segment_key not in self._cumul_attn:
            self._cumul_attn[segment_key] = torch.full((n_tokens,), 0.5)
        else:
            existing = self._cumul_attn[segment_key]
            if existing.shape[0] != n_tokens:
                # Resize: truncate or pad with 0.5
                new = torch.full((n_tokens,), 0.5)
                copy_len = min(existing.shape[0], n_tokens)
                new[:copy_len] = existing[:copy_len]
                self._cumul_attn[segment_key] = new
        return self._cumul_attn[segment_key]

    def predict(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        query: torch.Tensor,
        token_positions: torch.Tensor,
        current_position: int,
        segment_key: str = "",
    ) -> torch.Tensor:
        """Predict per-token retention probability.

        Returns:
            retention_prob: [n_tokens], float32, range [0, 1]
        """
        n_tokens = k.shape[0]
        d_head = k.shape[-1]

        k_f = k.float()
        v_f = v.float()

        # Feature 1: key norm
        k_norm = k_f.norm(dim=-1) / math.sqrt(d_head)  # [n_tokens]

        # Feature 2: value norm
        v_norm = v_f.norm(dim=-1) / math.sqrt(d_head)  # [n_tokens]

        # Feature 3: cumulative attention score (EMA-tracked)
        cumul_attn = self._get_or_init_cumul_attn(segment_key, n_tokens).clone()

        # Feature 4: position decay
        pos_diff = (current_position - token_positions.float()).clamp(min=0)
        pos_decay = torch.exp(-self.config.gamma_position_decay * pos_diff)  # [n_tokens]

        # Feature 5: query similarity
        if query.dim() == 2:
            query_vec = query.float().mean(dim=0)  # [d_head]
        else:
            query_vec = query.float()
        query_sim = F.cosine_similarity(
            k_f, query_vec.unsqueeze(0).expand(n_tokens, -1), dim=-1
        )  # [n_tokens]

        if self.config.zero_shot_mode:
            # DapQ-style fallback: no learned weights needed
            retention_prob = torch.sigmoid(query_sim * cumul_attn)
        else:
            features = torch.stack(
                [k_norm, v_norm, cumul_attn, pos_decay, query_sim], dim=-1
            )  # [n_tokens, 5]
            hidden = F.relu(features @ self._w1.T + self._b1)  # [n_tokens, 32]
            retention_prob = torch.sigmoid(
                hidden @ self._w2.T + self._b2
            ).squeeze(-1)  # [n_tokens]

        return retention_prob.float()

    def update_cumul_attn(
        self,
        segment_key: str,
        attn_scores: torch.Tensor,
    ) -> None:
        """Update cumulative attention EMA.

        cumul_attn = alpha × attn_scores + (1 - alpha) × cumul_attn
        """
        n_tokens = attn_scores.shape[0]
        current = self._get_or_init_cumul_attn(segment_key, n_tokens)
        alpha = self.config.ema_alpha_cumul_attn
        self._cumul_attn[segment_key] = (
            alpha * attn_scores.float() + (1 - alpha) * current
        )

    def select_tokens_by_budget(
        self,
        retention_prob: torch.Tensor,
        budget_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select kept/evicted token indices based on budget_ratio.

        Returns:
            (kept_indices [n_keep], evict_indices [n_evict])
        """
        n_tokens = retention_prob.shape[0]
        n_keep = max(1, int(n_tokens * budget_ratio))
        kept_idx = retention_prob.topk(n_keep).indices.sort().values
        all_idx = torch.arange(n_tokens, dtype=torch.long)
        mask = torch.ones(n_tokens, dtype=torch.bool)
        mask[kept_idx] = False
        evict_idx = all_idx[mask]
        return kept_idx, evict_idx

    def save_weights(self, path: str) -> None:
        """Save trained weights to YAML file."""
        data = {
            "w1": self._w1.tolist(),
            "b1": self._b1.tolist(),
            "w2": self._w2.tolist(),
            "b2": self._b2.tolist(),
            "trained_epochs": 0,
            "training_date": "2026-05-27",
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

    def load_weights(self, path: str) -> None:
        """Load weights from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self._w1 = torch.tensor(data["w1"], dtype=torch.float32)
        self._b1 = torch.tensor(data["b1"], dtype=torch.float32)
        self._w2 = torch.tensor(data["w2"], dtype=torch.float32)
        self._b2 = torch.tensor(data["b2"], dtype=torch.float32)
