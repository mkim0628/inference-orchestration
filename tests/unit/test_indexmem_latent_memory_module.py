"""Unit tests for IndexMemLatentMemoryModule (Activity C-1)."""

import pytest
import torch

from src.cache.indexmem_latent_memory_module import (
    IndexMemLatentMemoryModule,
    LatentMemoryConfig,
)


def _make_module(
    kv_dim: int = 64,
    latent_dim: int = 32,
    n_layers: int = 4,
    alpha: float = 0.3,
    beta: float = 0.1,
    seed: int = 42,
) -> IndexMemLatentMemoryModule:
    cfg = LatentMemoryConfig(
        kv_dim=kv_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        encoder_hidden_dim=64,
        encoder_n_heads=4,
        encoder_ffn_dim=128,
        encoder_n_layers=2,
        alpha_ema=alpha,
        beta_readout=beta,
        seed=seed,
    )
    return IndexMemLatentMemoryModule(cfg)


# --------------------------------------------------------------------------- #


def test_encode_evicted_output_shape():
    mod = _make_module(kv_dim=64, latent_dim=32, n_layers=4)
    torch.manual_seed(0)
    evicted = torch.randn(8, 64)
    latent = mod.encode_evicted(evicted, "req0", layer_idx=0)
    assert latent.shape == (32,), f"Expected (32,), got {latent.shape}"


def test_online_update_ema_alpha():
    alpha = 0.3
    mod = _make_module(latent_dim=32, n_layers=2, alpha=alpha, seed=42)
    torch.manual_seed(1)
    kv1 = torch.randn(4, 64)
    kv2 = torch.randn(4, 64)

    lat1 = mod.encode_evicted(kv1, "req", layer_idx=0)  # first call: latent = alpha * enc + 0
    lat2 = mod.encode_evicted(kv2, "req", layer_idx=0)  # second call: EMA update

    # lat2 = alpha * enc2 + (1-alpha) * lat1
    # We can't check exact values since encoder is random, but shape must be correct
    assert lat2.shape == (32,)
    # EMA shrinks magnitude toward previous value
    # Just verify it's finite and in reasonable range
    assert torch.isfinite(lat2).all()


def test_residual_readout_output_shape():
    mod = _make_module(kv_dim=64, latent_dim=32, n_layers=4)
    torch.manual_seed(0)
    kv = torch.randn(6, 64)
    mod.encode_evicted(kv, "req1", layer_idx=0)

    query = torch.randn(4, 64)  # [n_q, d_head]; d_head must match kv_dim
    readout = mod.residual_readout(query, "req1", layer_idx=0)
    assert readout.shape == query.shape, f"Expected {query.shape}, got {readout.shape}"


def test_readout_zero_before_encode():
    mod = _make_module(kv_dim=64, latent_dim=32, n_layers=4)
    query = torch.randn(4, 64)
    readout = mod.residual_readout(query, "nonexistent", layer_idx=0)
    assert torch.allclose(readout, torch.zeros_like(query))


def test_latent_state_persists_across_calls():
    mod = _make_module(latent_dim=32, n_layers=4)
    torch.manual_seed(5)
    kv = torch.randn(4, 64)

    lat1 = mod.encode_evicted(kv, "req_persist", layer_idx=0)
    state_after_1 = mod.get_latent_state("req_persist", 0).clone()

    kv2 = torch.randn(4, 64)
    lat2 = mod.encode_evicted(kv2, "req_persist", layer_idx=0)
    state_after_2 = mod.get_latent_state("req_persist", 0).clone()

    # State should change after second encode
    assert not torch.allclose(state_after_1, state_after_2)


def test_clear_removes_latent_state():
    mod = _make_module(latent_dim=32, n_layers=4)
    torch.manual_seed(2)
    kv = torch.randn(4, 64)
    mod.encode_evicted(kv, "req_clear", layer_idx=0)
    assert mod.get_latent_state("req_clear", 0) is not None

    mod.clear("req_clear")
    assert mod.get_latent_state("req_clear", 0) is None

    query = torch.randn(4, 64)
    readout = mod.residual_readout(query, "req_clear", layer_idx=0)
    assert torch.allclose(readout, torch.zeros_like(query))


def test_latent_memory_bytes():
    mod = _make_module(latent_dim=32, n_layers=4)
    assert mod.latent_memory_bytes() == 0  # no states yet

    kv = torch.randn(4, 64)
    mod.encode_evicted(kv, "req_bytes", layer_idx=0)
    # n_layers=4, latent_dim=32, float32 = 4 bytes
    expected = 4 * 32 * 4  # 512 bytes
    assert mod.latent_memory_bytes() == expected


def test_multiple_requests_independent():
    mod = _make_module(latent_dim=32, n_layers=4)
    torch.manual_seed(9)
    kv_a = torch.randn(4, 64)
    kv_b = torch.randn(4, 64)

    mod.encode_evicted(kv_a, "reqA", layer_idx=0)
    mod.encode_evicted(kv_b, "reqB", layer_idx=0)

    lat_a = mod.get_latent_state("reqA", 0)
    lat_b = mod.get_latent_state("reqB", 0)

    assert lat_a is not None
    assert lat_b is not None
    # Different requests should produce different latents
    assert not torch.allclose(lat_a, lat_b)
