from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MemoryMetrics:
    baseline_bytes: int = 0
    current_bytes: int = 0

    def reduction_ratio(self) -> float:
        """Fractional memory reduction vs baseline (positive = less memory)."""
        if self.baseline_bytes == 0:
            return 0.0
        return 1.0 - self.current_bytes / self.baseline_bytes

    def reduction_percent(self) -> float:
        return self.reduction_ratio() * 100.0

    def summary(self) -> dict:
        return {
            "baseline_bytes": self.baseline_bytes,
            "current_bytes": self.current_bytes,
            "reduction_percent": self.reduction_percent(),
        }


@dataclass
class WynerZivWindowMetrics:
    """Per-layer W*_l time series and α_l EMA tracking. Independent of MemoryMetrics."""
    _window_history: Dict[int, List[int]] = field(default_factory=dict)
    _alpha_history: Dict[int, List[float]] = field(default_factory=dict)
    _batch_indices: List[int] = field(default_factory=list)

    def record(
        self,
        batch_idx: int,
        window_sizes: Dict[int, int],
        alpha_ema: Dict[int, float],
    ) -> None:
        """Record window_sizes and alpha_ema for a batch.

        window_sizes: {layer_idx → W*_l}
        alpha_ema:    {layer_idx → α_l}
        """
        self._batch_indices.append(batch_idx)
        for layer_idx, w in window_sizes.items():
            self._window_history.setdefault(layer_idx, []).append(w)
        for layer_idx, a in alpha_ema.items():
            self._alpha_history.setdefault(layer_idx, []).append(a)

    def latest_window_sizes(self) -> Dict[int, int]:
        return {l: hist[-1] for l, hist in self._window_history.items() if hist}

    def latest_alpha_ema(self) -> Dict[int, float]:
        return {l: hist[-1] for l, hist in self._alpha_history.items() if hist}

    def memory_reduction_vs_fixed(self, fixed_window: int = 512) -> float:
        """(fixed_window × n_layers − Σ W*_l) / (fixed_window × n_layers). Positive = saved."""
        total_w_star = sum(self.latest_window_sizes().values())
        n_layers = len(self._window_history)
        if n_layers == 0:
            return 0.0
        total_fixed = fixed_window * n_layers
        return (total_fixed - total_w_star) / total_fixed if total_fixed > 0 else 0.0

    def summary(self) -> dict:
        return {
            "latest_window_sizes": self.latest_window_sizes(),
            "latest_alpha_ema": self.latest_alpha_ema(),
            "memory_reduction_vs_fixed_512": self.memory_reduction_vs_fixed(512),
            "total_batches_recorded": len(self._batch_indices),
        }
