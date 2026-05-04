"""DAGAwareTTLAdjuster — adapter connecting DAGTopologyScheduler to WorkloadAwareTTLCache (Cross-1).

Receives KV reuse probability events from DAGTopologyScheduler and applies
adjusted TTL values to WorkloadAwareTTLCache via its adjust_ttl() API.

On node completion, sets TTL to 0 to allow immediate eviction.
Measures event-to-TTL-update latency to verify scheduling overhead ≤ 5%.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np


class DAGAwareTTLAdjuster:
    """Adapter: DAG reuse events → WorkloadAwareTTLCache TTL adjustments (Cross-1).

    Minimises coupling between DAGTopologyScheduler and WorkloadAwareTTLCache
    by acting as an intermediate event handler.
    """

    def __init__(
        self,
        cache: Any,  # WorkloadAwareTTLCache — avoid circular import
        alpha: float = 2.0,
        measure_latency: bool = True,
    ) -> None:
        self.cache = cache
        self.alpha = alpha
        self.measure_latency = measure_latency
        self._latency_samples: List[float] = []  # event-to-update latency in ms

    def on_kv_reuse_event(
        self,
        segment_key: str,
        dag_reuse_probability: float,
    ) -> None:
        """Receive a KV reuse probability event and extend the segment TTL.

        adjusted_ttl = base_ttl × (1 + dag_reuse_probability × alpha)
        """
        t0 = time.monotonic()

        store = getattr(self.cache, "_store", None)
        if store is None or segment_key not in store:
            if self.measure_latency:
                self._latency_samples.append((time.monotonic() - t0) * 1000.0)
            return

        entry = store[segment_key]
        profiles = getattr(self.cache, "_ttl_profiles", {})
        category = entry.category
        base_ttl: float = profiles.get(category, {}).get("ttl_base_sec", 300.0)

        adjusted_ttl = base_ttl * (1.0 + dag_reuse_probability * self.alpha)
        self.cache.adjust_ttl(segment_key, adjusted_ttl)

        if self.measure_latency:
            self._latency_samples.append((time.monotonic() - t0) * 1000.0)

    def on_node_complete(self, segment_key: str) -> None:
        """Signal that a downstream node has completed — allow immediate eviction.

        Sets TTL to 0 and unpins the segment so the next evict() pass can free it.
        """
        self.cache.adjust_ttl(segment_key, new_ttl_sec=0.0)
        self.cache.unpin(segment_key)

    def overhead_stats(self) -> dict:
        """Return latency statistics for event-to-TTL-update operations.

        Returns:
            dict with keys: p50_ms, p99_ms, mean_ms, n_samples
        """
        n = len(self._latency_samples)
        if n == 0:
            return {"p50_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "n_samples": 0}

        arr = np.array(self._latency_samples, dtype=float)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "n_samples": n,
        }
