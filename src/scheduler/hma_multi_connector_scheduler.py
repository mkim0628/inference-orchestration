"""Activity A: HMAMultiConnectorCompressionPluginScheduler.

HMA multi-connector plugin meta-scheduler that selects and chains compression
connectors at runtime based on request profile (context length, RL mode,
memory pressure). Compatible with vLLM v0.21.0 HMA multi-connector API concept.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


class HMAConnectorInterface(ABC):
    """vLLM v0.21.0 HMA multi-connector interface abstract base.

    Each compression codec is wrapped as an independent HMA connector.
    compress() follows CacheStore.compression_hook() semantics.
    """

    @abstractmethod
    def compress(
        self,
        kv: torch.Tensor,
        request_profile: Dict,
    ) -> torch.Tensor:
        """Compress KV tensor and return. Shape may change (eviction) or stay same (quantization)."""

    @abstractmethod
    def decompress(
        self,
        compressed_kv: torch.Tensor,
        request_profile: Dict,
    ) -> torch.Tensor:
        """Restore compressed KV. For quantization codecs performs dequantization."""

    @property
    @abstractmethod
    def connector_name(self) -> str:
        """Connector identifier (matches registry key)."""


class HMAConnectorAdapter(HMAConnectorInterface):
    """Wraps existing codecs (CacheStore/CompressionCodec) as HMAConnectorInterface.

    Auto-detects compression_hook() or encode/decode interface and delegates.
    """

    def __init__(self, name: str, codec: object) -> None:
        self._name = name
        self._codec = codec

    @property
    def connector_name(self) -> str:
        return self._name

    def compress(self, kv: torch.Tensor, request_profile: Dict) -> torch.Tensor:
        # CacheStore.compression_hook() takes priority
        if hasattr(self._codec, "compression_hook"):
            return self._codec.compression_hook("__hma__", kv)
        # encode(kv, layer_idx=0) fallback
        if hasattr(self._codec, "encode"):
            return self._codec.encode(kv, layer_idx=0)
        return kv

    def decompress(self, compressed_kv: torch.Tensor, request_profile: Dict) -> torch.Tensor:
        if hasattr(self._codec, "decode"):
            return self._codec.decode(compressed_kv, layer_idx=0)
        return compressed_kv


@dataclass
class HMAMultiConnectorConfig:
    """Configuration for HMAMultiConnectorCompressionPluginScheduler.

    All parameters are YAML-externalizable via configs/experiments/*.yaml.
    """
    long_ctx_threshold: int = 4096
    memory_pressure_threshold: float = 0.8
    default_connector: str = "global_retention"
    pipeline_mode: bool = False       # True: chain primary connector + global_retention
    max_wait_ratio: float = 2.0       # fairness: max wait time multiplier
    seed: int = 42

    # connector_dispatch_policy (YAML connector_dispatch_policy section)
    rl_mode_connector: str = "rl_adaptive"
    long_ctx_connector: str = "global_retention"
    high_pressure_connector: str = "ratequant"


class HMAMultiConnectorCompressionPluginScheduler:
    """HMA multi-connector compression plugin meta-scheduler.

    Activity A: KV Cache-aware Scheduling
    Scheduling unit: per request — evaluates request profile and selects connector.
    Cache access: _connector_registry Dict (O(1) lookup).

    Connector registry:
      - "rl_adaptive"      : RLAdaptivePrecisionQuantizer (C-1, RL workloads)
      - "global_retention" : GlobalRetentionGateEvictionCodec (long context)
      - "ratequant"        : RateQuantReverseWaterfillingCodec (short high-throughput)
      - "lookahead"        : LookaheadKVEvictionCodec (future-aware eviction)

    vLLM v0.21.0 HMA multi-connector integration:
      register_connector() → register in HMA OffloadingConnector registry
      select_connector()   → O(1) selection based on request characteristics
      pipeline_mode=True   → chain primary connector + global_retention sequentially

    Evaluation criteria (evaluation_criteria.md §2):
      - Scheduling overhead: TTFT p50 +5% within limit (connector selection < 0.1ms/req)
      - Cache hit rate improvement: +10%p
      - Request fairness: max wait time within max_wait_ratio
    """

    def __init__(
        self,
        config: HMAMultiConnectorConfig,
        cache: Optional[CacheStore] = None,
    ) -> None:
        self.config = config
        self._cache = cache
        self._connector_registry: Dict[str, HMAConnectorInterface] = {}
        self._scheduling_times: List[float] = []
        self._request_connector_map: Dict[str, str] = {}
        self._connector_selection_counts: Dict[str, int] = {}
        self._arrival_times: Dict[str, float] = {}

    def register_connector(
        self,
        name: str,
        connector: HMAConnectorInterface,
    ) -> None:
        """Register an HMA connector in the registry.

        Algorithm:
          1. Store connector in _connector_registry[name]
          2. Initialize _connector_selection_counts[name] = 0
        """
        self._connector_registry[name] = connector
        self._connector_selection_counts[name] = 0

    def select_connector(
        self,
        request: InferenceRequest,
        request_meta: Optional[Dict] = None,
    ) -> str:
        """Select optimal connector based on request profile (O(1) dictionary lookup).

        Algorithm (connector_dispatch_policy):
          1. is_rl_mode=True or num_completions>1 → "rl_adaptive"
          2. context_length > long_ctx_threshold → "global_retention"
          3. context_length <= long_ctx_threshold
             and memory_pressure > memory_pressure_threshold → "ratequant"
          4. else → config.default_connector
          5. if selected not in registry → fallback to config.default_connector

        Args:
            request: InferenceRequest
            request_meta: additional metadata {"is_rl_mode": bool, "num_completions": int,
                          "memory_pressure": float}

        Returns:
            connector_name: str
        """
        t0 = time.monotonic()
        meta = request_meta or {}
        context_length = len(request.token_ids)
        is_rl = meta.get("is_rl_mode", False)
        num_completions = meta.get("num_completions", 1)
        memory_pressure = meta.get("memory_pressure", 0.0)

        cfg = self.config
        if (is_rl or num_completions > 1) and cfg.rl_mode_connector in self._connector_registry:
            selected = cfg.rl_mode_connector
        elif context_length > cfg.long_ctx_threshold and cfg.long_ctx_connector in self._connector_registry:
            selected = cfg.long_ctx_connector
        elif (context_length <= cfg.long_ctx_threshold
              and memory_pressure > cfg.memory_pressure_threshold
              and cfg.high_pressure_connector in self._connector_registry):
            selected = cfg.high_pressure_connector
        else:
            selected = cfg.default_connector

        # Fallback if selected not in registry
        if selected not in self._connector_registry:
            available = list(self._connector_registry.keys())
            selected = available[0] if available else cfg.default_connector

        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)
        return selected

    def apply_connector(
        self,
        request: InferenceRequest,
        kv: torch.Tensor,
        request_meta: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, str]:
        """Compress KV with selected connector (or chain in pipeline_mode).

        pipeline_mode=True:
          1. select_connector() to pick primary connector
          2. Compress with primary connector
          3. If "global_retention" in registry and primary != global_retention, chain it

        Returns:
            (compressed_kv, selected_connector_name)
        """
        meta = request_meta or {}
        connector_name = self.select_connector(request, meta)
        connector = self._connector_registry.get(connector_name)

        request_profile = {"context_length": len(request.token_ids), **meta}
        compressed = connector.compress(kv, request_profile) if connector else kv

        if self.config.pipeline_mode and connector_name != "global_retention":
            global_conn = self._connector_registry.get("global_retention")
            if global_conn is not None:
                compressed = global_conn.compress(compressed, request_profile)

        self._request_connector_map[request.request_id] = connector_name
        self._connector_selection_counts[connector_name] = (
            self._connector_selection_counts.get(connector_name, 0) + 1
        )
        return compressed, connector_name

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """Sort request list by predicted cache hit rate and return.

        Maintains same schedule() interface as CacheAwareScheduler.
        Records arrival times for fairness tracking.

        Returns:
            List[InferenceRequest] — ordered requests (FIFO baseline, extensible)
        """
        t0 = time.monotonic()
        for req in requests:
            self._arrival_times.setdefault(req.request_id, t0)
        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)
        return requests

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def scheduling_overhead_ms_p50(self) -> float:
        """Median scheduling overhead in ms."""
        if not self._scheduling_times:
            return 0.0
        sorted_t = sorted(self._scheduling_times)
        return sorted_t[len(sorted_t) // 2]

    def connector_selection_stats(self) -> Dict[str, int]:
        """Per-connector selection count statistics."""
        return dict(self._connector_selection_counts)

    def reset_stats(self) -> None:
        """Reset all counters and tracking state."""
        self._scheduling_times.clear()
        self._request_connector_map.clear()
        self._connector_selection_counts.clear()
        self._arrival_times.clear()
