"""KV Packet-style MLP adapter for non-contiguous KV segment correction (Activity B).

A lightweight 2-layer MLP with residual connection that learns to map
cached (potentially stale / position-shifted) KV tensors towards the
distribution of fresh target KVs.  The residual design ensures that a
zero-initialised (or randomly initialised) adapter never corrupts the
cached value by more than the MLP correction term.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentAdapter(nn.Module):
    """2-layer MLP adapter with residual connection for KV segment correction.

    forward(x) = x + mlp(x)

    The residual guarantees that the adapter is identity-like at initialisation,
    making training stable and ensuring graceful degradation when untrained.
    """

    def __init__(
        self,
        kv_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.kv_dim = kv_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(kv_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, kv_dim),
        )

    def forward(self, cached_kv: torch.Tensor) -> torch.Tensor:
        """Apply residual-corrected MLP to cached KV tensor.

        Args:
            cached_kv: (..., kv_dim) — arbitrary leading dimensions are preserved.

        Returns:
            Corrected KV tensor with the same shape as input.
        """
        return cached_kv + self.mlp(cached_kv)

    def train_step(
        self,
        cached_kv: torch.Tensor,
        target_kv: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single gradient step minimising MSE between adapter output and target.

        Returns:
            Scalar loss value for this step.
        """
        self.train()
        optimizer.zero_grad()
        output = self.forward(cached_kv)
        loss = F.mse_loss(output, target_kv)
        loss.backward()
        optimizer.step()
        return loss.item()

    def fit(
        self,
        cached_kvs: List[torch.Tensor],
        target_kvs: List[torch.Tensor],
        n_steps: int = 500,
        lr: float = 1e-3,
    ) -> List[float]:
        """Train adapter on paired (cached, target) KV tensors.

        Args:
            cached_kvs: list of cached KV tensors (input to adapter)
            target_kvs: list of corresponding fresh KV tensors (supervision)
            n_steps:    number of gradient steps
            lr:         Adam learning rate

        Returns:
            Loss history (one value per step).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_pairs = len(cached_kvs)
        loss_history: List[float] = []

        for step in range(n_steps):
            pair_idx = step % n_pairs
            loss_val = self.train_step(cached_kvs[pair_idx], target_kvs[pair_idx], optimizer)
            loss_history.append(loss_val)

        self.eval()
        return loss_history

    def save(self, path: str) -> None:
        """Persist adapter weights to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Restore adapter weights from disk."""
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)
        self.eval()
