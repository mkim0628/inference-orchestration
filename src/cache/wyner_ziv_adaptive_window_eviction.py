"""WynerZivAdaptiveWindowEviction — Activity C-1.

Wyner-Ziv polynomial sensitivity bound based adaptive sliding window KV eviction.
Online power-law fitting: S_l(d) = C_l × d^{-α_l} per layer via EMA.
Theoretical lower bound: W*_l(ε) = (C_l / ε)^(1/α_l).
Sliding window: OrderedDict per layer, evict oldest when len > window_size[l].
Fallback: if observed perplexity_delta > ε, double W*.
"""

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.cache.base import CacheStore

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class WynerZivConfig:
    accuracy_budget: float = 0.01           # ε: perplexity delta upper bound (1%)
    min_window_size: int = 64               # W* lower clamp
    max_window_size: int = 4096             # W* upper clamp
    fitting_interval: int = 1000            # W* recompute period (batches)
    sensitivity_sample_ratio: float = 0.05  # fraction of batches to measure sensitivity
    ema_gamma: float = 0.1                  # EMA coefficient for α_l / C_l (slow update)
    warmup_batches: int = 100               # batches before W* is used (use max_window_size)
    kv_pressure_threshold: float = 0.8      # KV memory pressure threshold
    epsilon_relaxation_factor: float = 2.0  # ε multiplier under KV pressure
    n_layers: int = 32                      # number of model layers
    truncation_distances: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    window_config_path: str = "configs/wyner_ziv_window_config.yaml"
    seed: int = 42


class WynerZivAdaptiveWindowEvictionCache(CacheStore):
    """Wyner-Ziv polynomial sensitivity based theory-guided adaptive sliding window KV eviction.

    Online fitting: per-layer S_l(d) = C_l × d^{-α_l} power law EMA update.
    Theoretical lower bound: W*_l(ε) = (C_l / ε)^(1/α_l)  → global W* = max_l W*_l
    Sliding window: token KVs in OrderedDict (time-ordered) per layer.
                    When len > window_size[l], evict oldest token.
    Fallback: if observed perplexity_delta > ε → W* ×2.
    """

    def __init__(self, config: WynerZivConfig) -> None:
        self._config = config
        # Per-layer EMA state
        self._alpha_ema: Dict[int, float] = {}
        self._c_ema: Dict[int, float] = {}
        self._window_size: Dict[int, int] = {}
        # Per-layer KV store: layer_idx → OrderedDict{key → tensor}
        self._kv_store: Dict[int, OrderedDict] = {}
        # Statistics
        self._hits: int = 0
        self._misses: int = 0
        self._batch_count: int = 0
        self._warmup_done: bool = False
        self._fallback_count: int = 0
        self._init_from_config()

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def _init_from_config(self) -> None:
        """Load per-layer (C_l_init, alpha_l_init, window_size) from YAML config.

        During warmup, max_window_size is used regardless of config file value
        to avoid premature eviction before α/C estimates stabilize.
        """
        c_default = 1.0
        alpha_default = 1.5
        # During warmup we always use max_window_size
        window_default = self._config.max_window_size

        layer_overrides: Dict = {}
        config_path = self._config.window_config_path
        if _YAML_AVAILABLE and os.path.exists(config_path):
            with open(config_path, "r") as fh:
                cfg_yaml = yaml.safe_load(fh)
            if cfg_yaml:
                defaults = cfg_yaml.get("layer_defaults", {})
                c_default = defaults.get("C_l_init", c_default)
                alpha_default = defaults.get("alpha_l_init", alpha_default)
                # window_size from yaml is post-warmup default; still use max during warmup
                layer_overrides = cfg_yaml.get("layer_overrides") or {}

        for l in range(self._config.n_layers):
            override = layer_overrides.get(l, {}) or {}
            self._alpha_ema[l] = override.get("alpha_l_init", alpha_default)
            self._c_ema[l] = override.get("C_l_init", c_default)
            # Use max_window_size during warmup (Spec.md EMA warmup requirement)
            self._window_size[l] = window_default
            self._kv_store[l] = OrderedDict()

    # ------------------------------------------------------------------ #
    # CacheStore abstract method implementations                           #
    # ------------------------------------------------------------------ #

    def put(
        self,
        key: str,
        value: torch.Tensor,
        layer_idx: int = 0,
        perplexity_delta: Optional[float] = None,
    ) -> None:
        """Store KV in layer l's sliding window; evict oldest if over budget."""
        if layer_idx not in self._kv_store:
            self._kv_store[layer_idx] = OrderedDict()
            self._window_size[layer_idx] = self._config.max_window_size
            self._alpha_ema[layer_idx] = 1.5
            self._c_ema[layer_idx] = 1.0

        window = self._kv_store[layer_idx]
        if key in window:
            window.move_to_end(key)
        window[key] = value.detach().clone()

        if perplexity_delta is not None:
            self.update_sensitivity_model(perplexity_delta, len(window), layer_idx)

        self._batch_count += 1
        # Check warmup completion once threshold is crossed
        if not self._warmup_done and self._batch_count >= self._config.warmup_batches:
            self._warmup_done = True
            self.recompute_all_windows()

        # Recompute W* periodically after warmup
        if (
            self._warmup_done
            and self._batch_count % self._config.fitting_interval == 0
        ):
            self.recompute_all_windows()

        # Evict excess tokens beyond current window
        w_max = self._window_size[layer_idx]
        while len(window) > w_max:
            self._evict_oldest(layer_idx)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Search all layer windows for key; return first hit."""
        for layer_store in self._kv_store.values():
            if key in layer_store:
                layer_store.move_to_end(key)
                self._hits += 1
                return layer_store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        """Evict oldest token from the largest layer window; return bytes freed."""
        if not self._kv_store:
            return 0
        largest_layer = max(self._kv_store, key=lambda l: len(self._kv_store[l]))
        return self._evict_oldest(largest_layer)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(
            v.nbytes
            for layer_store in self._kv_store.values()
            for v in layer_store.values()
        )

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # Wyner-Ziv core API                                                   #
    # ------------------------------------------------------------------ #

    def update_sensitivity_model(
        self,
        perplexity_delta: float,
        truncation_length: int,
        layer_idx: int = 0,
    ) -> None:
        """EMA update of power-law parameters (α_l, C_l) for layer l.

        Power law: S_l(d) = C_l × d^{-α_l}
        Single measurement estimate from (perplexity_delta, truncation_length):
          α_measured = -log(S) / log(d)
          C_measured  = S × d^α_measured
        EMA with γ=0.1 suppresses variance (slow update per Spec.md).
        """
        if truncation_length <= 1 or perplexity_delta <= 0:
            return

        safe_delta = max(perplexity_delta, 1e-9)
        log_d = math.log(max(truncation_length, 1))
        alpha_measured = max(0.1, -math.log(safe_delta) / log_d)
        c_measured = safe_delta * (truncation_length ** alpha_measured)

        gamma = self._config.ema_gamma
        prev_alpha = self._alpha_ema.get(layer_idx, 1.5)
        prev_c = self._c_ema.get(layer_idx, 1.0)
        self._alpha_ema[layer_idx] = gamma * alpha_measured + (1 - gamma) * prev_alpha
        self._c_ema[layer_idx] = gamma * c_measured + (1 - gamma) * prev_c

    def compute_optimal_window(
        self,
        layer_idx: int,
        epsilon: Optional[float] = None,
    ) -> int:
        """Compute W*_l(ε) = (C_l / ε)^(1/α_l), clamped to [min_window_size, max_window_size].

        Global W* = max_l W*_l ensures the most sensitive layer drives the bound.
        Layer-wise application: window_size[l] = W*_l for independent sizing.
        """
        eps = epsilon if epsilon is not None else self._config.accuracy_budget
        alpha = self._alpha_ema.get(layer_idx, 1.5)
        c = self._c_ema.get(layer_idx, 1.0)

        if alpha <= 0 or eps <= 0:
            return self._config.max_window_size

        try:
            # Use ceiling to preserve the S_l(W*) ≤ ε guarantee:
            # int() (floor) can produce W* s.t. S(W*) > ε for small values.
            w_star = math.ceil((c / eps) ** (1.0 / alpha))
        except (OverflowError, ZeroDivisionError):
            return self._config.max_window_size

        return max(
            self._config.min_window_size,
            min(w_star, self._config.max_window_size),
        )

    def recompute_all_windows(
        self,
        epsilon: Optional[float] = None,
        kv_pressure: Optional[float] = None,
    ) -> Dict[int, int]:
        """Recompute W*_l for all layers and update self._window_size.

        Under KV pressure (kv_pressure > kv_pressure_threshold),
        apply ε_relaxed = ε × epsilon_relaxation_factor to allow smaller windows.
        Called automatically every fitting_interval batches after warmup.
        """
        eps = epsilon if epsilon is not None else self._config.accuracy_budget
        if (
            kv_pressure is not None
            and kv_pressure > self._config.kv_pressure_threshold
        ):
            eps = eps * self._config.epsilon_relaxation_factor

        result: Dict[int, int] = {}
        for l in list(self._alpha_ema.keys()):
            w_star = self.compute_optimal_window(l, epsilon=eps)
            self._window_size[l] = w_star
            result[l] = w_star
        return result

    def trigger_fallback_if_needed(
        self,
        observed_perplexity_delta: float,
    ) -> bool:
        """If observed delta > ε, double all window sizes (accuracy fallback).

        Returns True if fallback was triggered, False otherwise.
        Window is clamped to max_window_size to prevent unbounded growth.
        """
        if observed_perplexity_delta > self._config.accuracy_budget:
            for l in list(self._window_size.keys()):
                self._window_size[l] = min(
                    self._window_size[l] * 2,
                    self._config.max_window_size,
                )
            self._fallback_count += 1
            return True
        return False

    def get_stats(self) -> dict:
        """Return current state summary dict."""
        return {
            "window_sizes": dict(self._window_size),
            "alpha_ema": dict(self._alpha_ema),
            "c_ema": dict(self._c_ema),
            "hit_rate": self.hit_rate(),
            "memory_bytes": self.memory_bytes(),
            "batch_count": self._batch_count,
            "warmup_done": self._warmup_done,
            "fallback_count": self._fallback_count,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _evict_oldest(self, layer_idx: int) -> int:
        """Evict the oldest (first inserted) entry from the given layer's window."""
        window = self._kv_store.get(layer_idx)
        if not window:
            return 0
        oldest_key = next(iter(window))
        bytes_freed = window[oldest_key].nbytes
        del window[oldest_key]
        return bytes_freed
