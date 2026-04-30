"""Unit tests for SegmentAdapter (Activity B).

Tests verify:
  - Output shape is preserved (residual connection)
  - Residual connection keeps adapter near-identity at initialisation
  - Training reduces loss
  - Adapter corrects perturbed KV (KL improvement after fitting)
"""

import torch
import torch.nn.functional as F
import pytest

from src.cache.segment_adapter import SegmentAdapter


KV_DIM = 64
HIDDEN_DIM = 32


@pytest.fixture
def adapter() -> SegmentAdapter:
    torch.manual_seed(42)
    return SegmentAdapter(kv_dim=KV_DIM, hidden_dim=HIDDEN_DIM)


def test_forward_shape_preserved(adapter: SegmentAdapter) -> None:
    """forward(x).shape must equal x.shape for arbitrary leading dims."""
    for shape in [(10, KV_DIM), (4, 16, KV_DIM), (1, KV_DIM)]:
        x = torch.randn(*shape)
        out = adapter(x)
        assert out.shape == x.shape, (
            f"Shape mismatch: input {x.shape}, output {out.shape}"
        )


def test_residual_connection(adapter: SegmentAdapter) -> None:
    """With near-zero MLP output, forward(x) should be close to x.

    Zero-init the last linear layer so mlp(x) ≈ 0, meaning output ≈ x.
    """
    torch.manual_seed(0)
    # Zero out the final linear layer weights+bias so mlp output is 0
    with torch.no_grad():
        for module in adapter.mlp.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == KV_DIM:
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()

    x = torch.randn(8, KV_DIM)
    out = adapter(x)
    # residual: out = x + 0 = x
    assert torch.allclose(out, x, atol=1e-5), (
        "With zero-weight MLP, adapter output should equal input (residual)"
    )


def test_training_reduces_loss(adapter: SegmentAdapter) -> None:
    """fit() should reduce MSE loss: loss[-1] < loss[0]."""
    torch.manual_seed(7)
    # Perturbed cached KV → target is the clean version
    target = torch.randn(16, KV_DIM)
    cached = target + 0.5 * torch.randn_like(target)

    loss_history = adapter.fit(
        cached_kvs=[cached],
        target_kvs=[target],
        n_steps=100,
        lr=1e-3,
    )

    assert len(loss_history) == 100, "Should return one loss per step"
    assert loss_history[-1] < loss_history[0], (
        f"Loss did not decrease: initial={loss_history[0]:.6f}, "
        f"final={loss_history[-1]:.6f}"
    )


def test_noncontiguous_correction(adapter: SegmentAdapter) -> None:
    """After fitting, adapter(perturbed) should have lower KL vs target than perturbed alone.

    Simulates the non-contiguous KV correction use case:
    - kv = clean target
    - perturbed = kv + noise (what cache returns for non-contiguous hit)
    - fit(perturbed → kv)
    - KL(adapter(perturbed), kv) < KL(perturbed, kv)
    """
    torch.manual_seed(11)
    kv = torch.randn(20, KV_DIM).abs() + 0.1  # positive for softmax stability
    noise = 0.3 * torch.randn_like(kv)
    perturbed = kv + noise

    # Train adapter to correct the perturbation
    adapter.fit(
        cached_kvs=[perturbed],
        target_kvs=[kv],
        n_steps=300,
        lr=1e-3,
    )

    adapter.eval()
    with torch.no_grad():
        corrected = adapter(perturbed)

    # KL divergence (perplexity proxy) — compare flattened distributions
    def kl(a: torch.Tensor, b: torch.Tensor) -> float:
        p = F.softmax(b.flatten(), dim=0)
        q = F.softmax(a.flatten(), dim=0)
        return F.kl_div(q.log(), p, reduction="sum").item()

    kl_before = kl(perturbed, kv)
    kl_after = kl(corrected, kv)

    assert kl_after < kl_before, (
        f"Adapter did not improve KL: before={kl_before:.6f}, after={kl_after:.6f}"
    )
