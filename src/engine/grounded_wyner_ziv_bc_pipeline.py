"""GroundedWynerZivBCPipeline — Cross-1 B+C Integration.

Coordinates GroundedSafetyGatedSegmentCache (Activity B-1) and
WynerZivAdaptiveWindowEvictionCache (Activity C-1) under a shared ε accuracy
budget with sample-batch perplexity monitoring and auto-fallback.

Processing flow:
  Step 1 (B-1): 4-gate safety check on cached segment
    → safe_reuse_hit: return KV from safety cache
    → partial_reuse_hit: return c_KV (content component only)
    → miss / gate_rejected / stale_evicted: compute KV then go to Step 2

  Step 2 (C-1): store newly computed KV in WynerZiv sliding window
    → window exceeding W*(ε) triggers oldest-token eviction

  Step 3 (ε monitoring): on perplexity_sample_ratio fraction of requests,
    call trigger_fallback_if_needed() and record WynerZivWindowMetrics.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from src.cache.grounded_safety_gated_cache import (
    GroundedSafetyGatedSegmentCache,
    HitOutcome,
)
from src.cache.wyner_ziv_adaptive_window_eviction import (
    WynerZivAdaptiveWindowEvictionCache,
)
from src.metrics.hit_rate import SafetyGatedHitRateMetrics
from src.metrics.memory import WynerZivWindowMetrics


@dataclass
class BCPipelineConfig:
    shared_accuracy_budget: float = 0.01   # shared ε budget for both B-1 and C-1
    perplexity_sample_ratio: float = 0.01  # fraction of requests that trigger ε monitoring
    auto_fallback: bool = True             # automatically double W* when ε is exceeded
    seed: int = 42


class GroundedWynerZivBCPipeline:
    """B+C integrated pipeline combining 4-gate safety reuse and Wyner-Ziv window eviction.

    Both components share the same ε accuracy budget (shared_accuracy_budget).
    Perplexity monitoring on a sample of requests auto-triggers W* doubling fallback.
    """

    def __init__(
        self,
        safety_cache: GroundedSafetyGatedSegmentCache,
        window_eviction: WynerZivAdaptiveWindowEvictionCache,
        safety_metrics: SafetyGatedHitRateMetrics,
        window_metrics: WynerZivWindowMetrics,
        config: BCPipelineConfig,
    ) -> None:
        self._safety_cache = safety_cache
        self._window_eviction = window_eviction
        self._safety_metrics = safety_metrics
        self._window_metrics = window_metrics
        self._config = config
        self._request_count: int = 0
        self._rng = random.Random(config.seed)

    def process_request(
        self,
        query_tokens: List[int],
        current_context_tokens: List[int],
        kv_computed: Optional[torch.Tensor] = None,
        query_embedding: Optional[torch.Tensor] = None,
        current_source_version_hash: Optional[str] = None,
        layer_idx: int = 0,
        perplexity_delta: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], HitOutcome]:
        """Process a single request through the B+C pipeline.

        Returns (kv_tensor_or_None, hit_outcome).
        On cache miss / gate failure, caller is expected to supply kv_computed
        for storage in the WynerZiv window.
        """
        self._request_count += 1

        # Step 1 (B-1): 4-gate safety lookup
        kv, outcome = self._safety_cache.lookup(
            query_tokens=query_tokens,
            current_context_tokens=current_context_tokens,
            query_embedding=query_embedding,
            current_source_version_hash=current_source_version_hash,
            layer_idx=layer_idx,
        )
        self._safety_metrics.record(outcome)

        # Step 2 (C-1): if miss/rejected/stale, store new KV in window eviction cache
        if outcome in ("miss", "gate_rejected", "stale_evicted"):
            if kv_computed is not None:
                key = f"req_{self._request_count}_l{layer_idx}"
                self._window_eviction.put(
                    key=key,
                    value=kv_computed,
                    layer_idx=layer_idx,
                    perplexity_delta=perplexity_delta,
                )
                # Also insert into safety cache for future gate validation
                self._safety_cache.insert(
                    token_ids=query_tokens,
                    kv_data=kv_computed,
                    source_context_token_ids=current_context_tokens,
                    layer_idx=layer_idx,
                )
                kv = kv_computed

        # Step 3 (ε budget monitoring): sample a fraction of requests
        is_sample = self._rng.random() < self._config.perplexity_sample_ratio
        if is_sample and perplexity_delta is not None:
            if self._config.auto_fallback:
                self._window_eviction.trigger_fallback_if_needed(perplexity_delta)
            # Record window metrics
            stats = self._window_eviction.get_stats()
            self._window_metrics.record(
                batch_idx=self._request_count,
                window_sizes=stats["window_sizes"],
                alpha_ema=stats["alpha_ema"],
            )

        return kv, outcome

    def get_combined_stats(self) -> dict:
        """Return merged B+C pipeline statistics."""
        return {
            "safety_gate": self._safety_metrics.summary(),
            "window_eviction": self._window_eviction.get_stats(),
            "window_metrics": self._window_metrics.summary(),
            "shared_accuracy_budget": self._config.shared_accuracy_budget,
            "total_requests": self._request_count,
        }
