"""Unit tests for WynerZivAdaptiveWindowEvictionCache (Activity C-1).

Covers: CacheStore interface, W* formula, EMA convergence, warmup, fallback,
layer-wise independence, eviction enforcement, and KV pressure relaxation.
"""

import math
import pytest
import torch

from src.cache.wyner_ziv_adaptive_window_eviction import (
    WynerZivAdaptiveWindowEvictionCache,
    WynerZivConfig,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_cache(
    n_layers: int = 4,
    min_window: int = 64,
    max_window: int = 4096,
    warmup: int = 100,
    budget: float = 0.01,
    ema_gamma: float = 0.1,
    fitting_interval: int = 1000,
    window_config_path: str = "configs/wyner_ziv_window_config.yaml",
) -> WynerZivAdaptiveWindowEvictionCache:
    cfg = WynerZivConfig(
        accuracy_budget=budget,
        min_window_size=min_window,
        max_window_size=max_window,
        warmup_batches=warmup,
        ema_gamma=ema_gamma,
        n_layers=n_layers,
        fitting_interval=fitting_interval,
        window_config_path=window_config_path,
    )
    return WynerZivAdaptiveWindowEvictionCache(cfg)


def rand_tensor(size: int = 16) -> torch.Tensor:
    return torch.randn(size)


# ------------------------------------------------------------------ #
# Test 1: CacheStore interface — all abstract methods callable         #
# ------------------------------------------------------------------ #

def test_cache_store_interface_all_methods():
    cache = make_cache()
    t = rand_tensor()
    # put / get
    cache.put("k1", t, layer_idx=0)
    result = cache.get("k1")
    assert result is not None

    # evict
    freed = cache.evict()
    assert isinstance(freed, int) and freed >= 0

    # hit_rate
    cache.put("k2", t, layer_idx=0)
    cache.get("k2")
    assert 0.0 <= cache.hit_rate() <= 1.0

    # memory_bytes
    cache.put("k3", t, layer_idx=0)
    assert cache.memory_bytes() >= 0

    # reset_stats
    cache.reset_stats()
    assert cache.hit_rate() == 0.0


# ------------------------------------------------------------------ #
# Test 2: put → get round trip, single layer                           #
# ------------------------------------------------------------------ #

def test_put_get_round_trip_single_layer():
    cache = make_cache()
    t = rand_tensor(32)
    cache.put("key_a", t, layer_idx=0)
    result = cache.get("key_a")
    assert result is not None
    assert torch.allclose(result, t)


# ------------------------------------------------------------------ #
# Test 3: W* = (C/ε)^(1/α) numerical verification                     #
# ------------------------------------------------------------------ #

def test_compute_optimal_window_formula():
    """α=1.5, C=1.0, ε=0.01 → W* = ceil((1/0.01)^(1/1.5)) = ceil(100^(2/3)) = ceil(21.54) = 22
    (but clamped to min_window_size if below).
    Uses ceiling to preserve S_l(W*) ≤ ε theoretical guarantee.
    """
    import math as _math
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=10,
        max_window_size=4096,
        n_layers=1,
        warmup_batches=0,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    # Set known α and C
    cache._alpha_ema[0] = 1.5
    cache._c_ema[0] = 1.0
    w = cache.compute_optimal_window(layer_idx=0, epsilon=0.01)
    expected = _math.ceil((1.0 / 0.01) ** (1.0 / 1.5))
    assert w == max(10, min(expected, 4096)), f"got {w}, expected {expected}"


def test_w_star_computation_basic():
    """Verify formula: W* = ceil((C/ε)^(1/α)) for several (α, C, ε) combos."""
    import math as _math
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=1,
        max_window_size=100000,
        n_layers=1,
        warmup_batches=0,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    for alpha, c, eps in [(2.0, 1.0, 0.01), (1.0, 2.0, 0.05), (3.0, 0.5, 0.001)]:
        cache._alpha_ema[0] = alpha
        cache._c_ema[0] = c
        w = cache.compute_optimal_window(0, epsilon=eps)
        expected = _math.ceil((c / eps) ** (1.0 / alpha))
        assert w == expected, f"α={alpha},C={c},ε={eps}: got {w}, expected {expected}"


# ------------------------------------------------------------------ #
# Test 4: W* clamped to min_window_size                                #
# ------------------------------------------------------------------ #

def test_w_star_clamped_to_min():
    """When formula gives W* < min_window_size, result is min_window_size."""
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=200,
        max_window_size=4096,
        n_layers=1,
        warmup_batches=0,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    cache._alpha_ema[0] = 5.0  # steep decay → very small W*
    cache._c_ema[0] = 0.001
    w = cache.compute_optimal_window(0, epsilon=0.01)
    assert w == 200


# ------------------------------------------------------------------ #
# Test 5: W* clamped to max_window_size                                #
# ------------------------------------------------------------------ #

def test_w_star_clamped_to_max():
    """When formula gives W* > max_window_size, result is max_window_size."""
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=64,
        max_window_size=512,
        n_layers=1,
        warmup_batches=0,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    cache._alpha_ema[0] = 0.1  # very shallow → huge W*
    cache._c_ema[0] = 1000.0
    w = cache.compute_optimal_window(0, epsilon=0.01)
    assert w == 512


# ------------------------------------------------------------------ #
# Test 6: polynomial sensitivity fitting — EMA convergence              #
# ------------------------------------------------------------------ #

def test_polynomial_sensitivity_fitting():
    """After many identical (delta, d) measurements, EMA should converge near true α."""
    cache = make_cache(n_layers=1, ema_gamma=0.1, warmup=0)
    true_alpha = 2.0
    true_c = 1.0
    d = 100
    delta = true_c * (d ** -true_alpha)  # S(d) = C × d^{-α}

    for _ in range(200):
        cache.update_sensitivity_model(delta, d, layer_idx=0)

    # After convergence, α_ema should be close to true_alpha
    estimated_alpha = cache._alpha_ema[0]
    # Allow generous tolerance since estimation from single point has rounding
    assert abs(estimated_alpha - true_alpha) < 0.5, \
        f"estimated α={estimated_alpha}, true={true_alpha}"


# ------------------------------------------------------------------ #
# Test 7: EMA warmup — max_window_size used during warmup               #
# ------------------------------------------------------------------ #

def test_ema_warmup_uses_max_window():
    """During warmup (batch_count < warmup_batches), W* is max_window_size."""
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=64,
        max_window_size=4096,
        warmup_batches=100,
        n_layers=2,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    # Before any puts, window sizes should be max_window_size
    for l in range(2):
        assert cache._window_size[l] == 4096
    assert not cache._warmup_done


# ------------------------------------------------------------------ #
# Test 8: trigger_fallback_if_needed doubles window                    #
# ------------------------------------------------------------------ #

def test_trigger_fallback_doubles_window_on_budget_exceeded():
    """observed_delta > ε → trigger_fallback doubles all window sizes."""
    cache = make_cache(n_layers=2, max_window=1000, warmup=0, budget=0.01)
    cache._warmup_done = True
    cache._window_size[0] = 200
    cache._window_size[1] = 300

    triggered = cache.trigger_fallback_if_needed(0.02)  # > ε=0.01
    assert triggered
    assert cache._window_size[0] == 400
    assert cache._window_size[1] == 600


def test_fallback_not_triggered_within_budget():
    """observed_delta ≤ ε → no fallback, window sizes unchanged."""
    cache = make_cache(n_layers=2, warmup=0, budget=0.01)
    cache._warmup_done = True
    cache._window_size[0] = 200
    cache._window_size[1] = 300

    triggered = cache.trigger_fallback_if_needed(0.005)  # ≤ ε
    assert not triggered
    assert cache._window_size[0] == 200
    assert cache._window_size[1] == 300


# ------------------------------------------------------------------ #
# Test 9: insert and lookup                                             #
# ------------------------------------------------------------------ #

def test_insert_and_lookup():
    """put followed by get returns the same tensor."""
    cache = make_cache(n_layers=1)
    t = torch.ones(10)
    cache.put("x", t, layer_idx=0)
    out = cache.get("x")
    assert out is not None
    assert torch.allclose(out, t)


# ------------------------------------------------------------------ #
# Test 10: eviction respects window size                                #
# ------------------------------------------------------------------ #

def test_eviction_respects_window():
    """When more tokens than window_size are stored, oldest are evicted."""
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=3,
        max_window_size=3,  # very small window
        warmup_batches=0,
        n_layers=1,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    cache._warmup_done = True
    cache._window_size[0] = 3

    for i in range(5):
        cache.put(f"key_{i}", torch.tensor([float(i)]), layer_idx=0)

    # Only 3 entries should remain (the 3 most recent)
    assert len(cache._kv_store[0]) == 3
    # First two keys should be evicted
    assert "key_0" not in cache._kv_store[0]
    assert "key_1" not in cache._kv_store[0]
    assert "key_4" in cache._kv_store[0]


# ------------------------------------------------------------------ #
# Test 11: stats tracking                                               #
# ------------------------------------------------------------------ #

def test_stats_tracking():
    """get_stats returns expected keys and plausible values."""
    cache = make_cache(n_layers=2, warmup=0)
    cache._warmup_done = True
    t = rand_tensor()
    cache.put("k1", t, layer_idx=0)
    cache.get("k1")  # hit
    cache.get("missing")  # miss

    stats = cache.get_stats()
    assert "window_sizes" in stats
    assert "alpha_ema" in stats
    assert "c_ema" in stats
    assert stats["hit_rate"] > 0.0
    assert stats["batch_count"] >= 1
    assert isinstance(stats["warmup_done"], bool)


# ------------------------------------------------------------------ #
# Test 12: perplexity budget not violated (simulation)                  #
# ------------------------------------------------------------------ #

def test_perplexity_budget_not_violated():
    """S_l(W*) ≤ ε × tolerance for the computed W* with known α and C.

    The int() truncation can lower W* by up to 1 from the theoretical optimum,
    so a 10% tolerance accounts for this discretization effect.
    """
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=1,
        max_window_size=100000,
        n_layers=1,
        warmup_batches=0,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    eps = 0.01
    tolerance = 1.10  # 10% tolerance for integer floor truncation
    for alpha, c in [(1.5, 1.0), (2.0, 0.5), (1.0, 2.0)]:
        cache._alpha_ema[0] = alpha
        cache._c_ema[0] = c
        w_star = cache.compute_optimal_window(0, epsilon=eps)
        # Simulate S_l(W*) = C × W*^{-α}
        simulated_s = c * (w_star ** (-alpha))
        assert simulated_s <= eps * tolerance, \
            f"S_l(W*)={simulated_s:.4f} > ε×tol={eps*tolerance:.4f} for α={alpha},C={c},W*={w_star}"


# ------------------------------------------------------------------ #
# Test 13: layer-wise sensitivity independence                          #
# ------------------------------------------------------------------ #

def test_layer_wise_sensitivity():
    """Each layer maintains independent α_ema and can have different W*.

    With min_window_size=1 and distinct α values, W* ordering reflects α ordering.
    Lower α (shallower decay) → larger W* for same ε.
    """
    cache = make_cache(n_layers=3, warmup=0, min_window=1)
    cache._warmup_done = True

    # Set different α for each layer (lower α → larger W*)
    cache._alpha_ema[0] = 1.0
    cache._alpha_ema[1] = 2.0
    cache._alpha_ema[2] = 3.0
    cache._c_ema[0] = 1.0
    cache._c_ema[1] = 1.0
    cache._c_ema[2] = 1.0

    w0 = cache.compute_optimal_window(0, epsilon=0.01)
    w1 = cache.compute_optimal_window(1, epsilon=0.01)
    w2 = cache.compute_optimal_window(2, epsilon=0.01)

    # Lower α → shallower decay → larger W*
    assert w0 > w1 > w2, f"Expected w0 > w1 > w2 but got {w0}, {w1}, {w2}"


# ------------------------------------------------------------------ #
# Test 14: recompute_all_windows updates state                          #
# ------------------------------------------------------------------ #

def test_recompute_all_windows_updates_state():
    """recompute_all_windows() updates _window_size for all layers."""
    cache = make_cache(n_layers=2, warmup=0)
    cache._warmup_done = True

    # Manually set alpha/c
    cache._alpha_ema[0] = 1.5
    cache._c_ema[0] = 1.0
    cache._alpha_ema[1] = 2.0
    cache._c_ema[1] = 1.0

    windows = cache.recompute_all_windows(epsilon=0.01)
    assert 0 in windows
    assert 1 in windows
    assert cache._window_size[0] == windows[0]
    assert cache._window_size[1] == windows[1]


# ------------------------------------------------------------------ #
# Test 15: warmup flag set after warmup_batches                         #
# ------------------------------------------------------------------ #

def test_warmup_flag_set_after_100_batches():
    """_warmup_done becomes True after warmup_batches puts."""
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=64,
        max_window_size=4096,
        warmup_batches=10,  # small for test speed
        n_layers=1,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    assert not cache._warmup_done

    t = rand_tensor()
    for i in range(10):
        cache.put(f"k{i}", t, layer_idx=0)

    assert cache._warmup_done


# ------------------------------------------------------------------ #
# Test 16: KV pressure relaxes epsilon (test via recompute_all_windows) #
# ------------------------------------------------------------------ #

def test_kv_pressure_relaxes_epsilon():
    """Under KV pressure, ε is relaxed → W* shrinks vs non-pressure case."""
    cfg = WynerZivConfig(
        accuracy_budget=0.01,
        min_window_size=1,
        max_window_size=100000,
        n_layers=1,
        warmup_batches=0,
        kv_pressure_threshold=0.5,
        epsilon_relaxation_factor=2.0,
        window_config_path="configs/wyner_ziv_window_config.yaml",
    )
    cache = WynerZivAdaptiveWindowEvictionCache(cfg)
    cache._warmup_done = True
    cache._alpha_ema[0] = 1.5
    cache._c_ema[0] = 1.0

    w_normal = cache.recompute_all_windows(epsilon=0.01, kv_pressure=0.3)[0]
    w_pressure = cache.recompute_all_windows(epsilon=0.01, kv_pressure=0.9)[0]
    # Under pressure, ε relaxed to 0.02 → W* = (C/0.02)^(1/α) < W* at ε=0.01
    assert w_pressure <= w_normal, f"pressure W*={w_pressure} should ≤ normal W*={w_normal}"
