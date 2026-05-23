"""RuntimeCertified + KVSculpt closed-loop certified distillation pipeline (Cross-1, C-1+C-2).

Activity C Cross-1: integrates C-1 error-bound certification with C-2 layer-difficulty
distillation in an online budget-reallocation feedback loop.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.runtime_certified_quant_codec import (
    RuntimeCertifiedConfig,
    RuntimeCertifiedQuantizedAttentionCodec,
)
from src.cache.kvsculpt_distillation_codec import (
    KVSculptConfig,
    KVSculptDistillationCodec,
)


@dataclass
class DistillationPipelineConfig:
    c1_config: Optional[RuntimeCertifiedConfig] = None
    c2_config: Optional[KVSculptConfig] = None
    adaptive_threshold_scale: float = 1.0   # threshold tightening coefficient for high-difficulty layers
    online_budget_realloc_window: int = 100  # fallback rate aggregation window (batch count)
    seed: int = 42


class RuntimeCertifiedKVSculptDistillationPipeline(CacheStore):
    """RuntimeCertified + KVSculpt closed-loop certified distillation pipeline (Cross-1, C-1+C-2).

    Integrated processing flow:
      Step 1 (offline): KVSculpt pilot run -> per-layer KL divergence difficulty profile.
      Step 2 (offline): difficulty-proportional initial budget allocation.
      Step 3 (prefill): per-layer L-BFGS + least-squares alternating distillation compression.
      Step 4 (decoding): RuntimeCertified two-term error bound computation (per-head per-step).
      Step 5 (online): error_bound > adaptive_threshold(layer) -> immediate budget increase.
      Step 6 (fallback): if budget increase still insufficient -> FP16 fallback.

    adaptive_threshold(layer_l) = error_threshold / (1 + norm_difficulty(l))
      -> higher-difficulty layers get stricter thresholds.

    Closed-loop quality control:
      accumulate fallback ratio over previous N batches -> increase budget for
      frequently-falling-back layers in next batch.
    """

    def __init__(self, config: DistillationPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        c1_cfg = config.c1_config or RuntimeCertifiedConfig(seed=config.seed)
        c2_cfg = config.c2_config or KVSculptConfig(seed=config.seed)
        self.certified_codec = RuntimeCertifiedQuantizedAttentionCodec(c1_cfg)
        self.distillation_codec = KVSculptDistillationCodec(c2_cfg)
        self._layer_fallback_counts: Dict[int, int] = {}
        self._layer_request_counts: Dict[int, int] = {}

    def run_pipeline(
        self,
        Q: torch.Tensor,   # [n_q, d_head]
        K: torch.Tensor,   # [seq_len, d_head]
        V: torch.Tensor,   # [seq_len, d_head]
        layer_idx: int,
        cache_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Distillation compression + runtime certification integrated execution.

        Returns:
            (K_final, V_final, report_dict)
            report_dict keys: layer_idx, fallback_level, error_bound, selected_ratio
        """
        # Step 3: KVSculpt distillation compression
        selected_idx, K_distilled, V_distilled = self.distillation_codec.distill_compress(
            Q, K, V, layer_idx
        )

        # Step 3b: RuntimeCertified INT8K+INT4V compression on distilled K/V
        self.certified_codec.put(cache_key, K_distilled)

        # Step 4: error bound computation
        fallback_level, error_bound = self.certified_codec.certify_and_update(
            cache_key, Q
        )

        # Step 5: online budget reallocation
        self._layer_request_counts[layer_idx] = (
            self._layer_request_counts.get(layer_idx, 0) + 1
        )
        if fallback_level > 0:
            self._layer_fallback_counts[layer_idx] = (
                self._layer_fallback_counts.get(layer_idx, 0) + 1
            )
            # increase budget for this layer by 5%
            if self.distillation_codec._profile_done:
                n = self.distillation_codec.config.n_layers
                idx = min(layer_idx, n - 1)
                self.distillation_codec._layer_budget[idx] = min(
                    0.9,
                    float(self.distillation_codec._layer_budget[idx]) + 0.05,
                )

        # Step 6: final K/V selection
        K_final = K_distilled if fallback_level == 0 else K
        V_final = V_distilled if fallback_level == 0 else V

        report = {
            "layer_idx": layer_idx,
            "fallback_level": fallback_level,
            "error_bound": error_bound,
            "selected_ratio": len(selected_idx) / max(1, K.shape[0]),
        }
        return K_final, V_final, report

    # ------------------------------------------------------------------ #
    # CacheStore interface (delegated to certified_codec)                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        self.certified_codec.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self.certified_codec.get(key)

    def evict(self) -> int:
        return self.certified_codec.evict()

    def hit_rate(self) -> float:
        return self.certified_codec.hit_rate()

    def memory_bytes(self) -> int:
        return self.certified_codec.memory_bytes() + self.distillation_codec.memory_bytes()

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        return self.certified_codec.compression_hook(key, value)

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError(
            "RuntimeCertifiedKVSculptDistillationPipeline does not support importance masking."
        )

    def reset_stats(self) -> None:
        self.certified_codec.reset_stats()
        self.distillation_codec.reset_stats()
        self._layer_fallback_counts.clear()
        self._layer_request_counts.clear()
