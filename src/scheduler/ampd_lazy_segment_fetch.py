"""Activity A: AMPDLazySegmentFetchScheduler — pull-on-demand lazy segment fetch.

Implements AMPD (arXiv 2602.14516) "KV lazy-read" principle applied to
non-contiguous segment paths: KV data transfer is withheld until the reuse
set is confirmed via Louver search; only metadata is forwarded first.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Literal, Optional

import torch

from src.cache.segmented import SegmentedHashCache
from src.engine.runner import InferenceRequest


SegmentTier = Literal["HBM", "DDR", "REMOTE"]


@dataclass
class SegmentMeta:
    segment_id: str           # segment content hash (same format as chunk_key)
    source_node_id: str       # node identifier holding this segment ("local" or IP)
    tier: SegmentTier         # HBM / DDR / REMOTE
    approx_size_bytes: int    # estimated KV tensor size in bytes
    position_range: tuple     # (start_token_idx, end_token_idx)


@dataclass
class KVSegment:
    segment_id: str
    kv_tensor: torch.Tensor   # actual KV data
    source_tier: str          # "HBM" | "DDR" | "REMOTE"
    load_latency_ms: float    # time spent loading


@dataclass
class AMPDLazySchedulerConfig:
    # Default latencies for single-node; tune remote_fetch_latency_ms for multi-node
    hbm_fetch_latency_ms: float = 0.01
    ddr_fetch_latency_ms: float = 0.5
    remote_fetch_latency_ms: float = 5.0
    metadata_overhead_max_ms: float = 0.1   # upper bound for metadata delivery overhead
    max_concurrent_fetches: int = 8
    seed: int = 42


class SegmentMetadataRegistry:
    """Lightweight segment_id → SegmentMeta registry.

    Tracks candidate segments registered on request arrival (no KV data),
    cancels ones not confirmed, and exposes unnecessary_transfer_ratio.

    Single-node: source_node_id = "local", no gRPC stream.
    Multi-node: register_remote() registers segments from external nodes.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, SegmentMeta] = {}
        self._pre_resolved_count: int = 0   # candidates registered
        self._cancelled_count: int = 0      # cancelled before confirmation

    def register(self, meta: SegmentMeta) -> None:
        """Register segment metadata (no KV data)."""
        self._registry[meta.segment_id] = meta
        self._pre_resolved_count += 1

    def get(self, segment_id: str) -> Optional[SegmentMeta]:
        """Return SegmentMeta or None on miss."""
        return self._registry.get(segment_id)

    def cancel(self, segment_id: str) -> None:
        """Mark segment as cancelled (will not be pulled)."""
        if segment_id in self._registry:
            self._cancelled_count += 1

    def unnecessary_transfer_ratio(self) -> float:
        """Fraction of candidates cancelled before KV transfer."""
        if self._pre_resolved_count == 0:
            return 0.0
        return self._cancelled_count / self._pre_resolved_count

    def reset_stats(self) -> None:
        self._pre_resolved_count = 0
        self._cancelled_count = 0


class AMPDLazySegmentFetchScheduler:
    """AMPD pull-on-demand non-contiguous segment lazy-fetch scheduler.

    Activity A: KV Cache-aware Scheduling.
    Scheduling unit: per-request.
    Cache state access: SegmentMetadataRegistry only (no KV access).

    Processing flow:
      1. pre_resolve_segments(): register candidate segment metadata on arrival.
         Zero KV data transmitted.
      2. confirm_segments(): after Louver search, fix reuse set S_reuse;
         cancel remaining candidates.
      3. fetch_segments_lazy(): async pull of confirmed segments only.

    Evaluation criteria (evaluation_criteria.md §2):
      - Scheduling overhead TTFT p50 +5% or less
      - Metadata delivery overhead < 0.1 ms/request
      - unnecessary_transfer_ratio recorded in results/<exp>/metrics.json
    """

    def __init__(
        self,
        config: AMPDLazySchedulerConfig,
        registry: Optional[SegmentMetadataRegistry] = None,
        cache: Optional[SegmentedHashCache] = None,
    ) -> None:
        self.config = config
        self.registry = registry or SegmentMetadataRegistry()
        self._cache = cache
        self._scheduling_times: List[float] = []

    def pre_resolve_segments(
        self,
        request: InferenceRequest,
        candidate_segment_ids: List[str],
        metas: List[SegmentMeta],
    ) -> None:
        """Stage 0: register candidate segment metadata on request arrival.

        Algorithm:
          - Call registry.register() for each (segment_id, SegmentMeta) pair.
          - No KV data is transmitted.
          - Overhead measured and must be < metadata_overhead_max_ms/request.
        """
        t0 = time.monotonic()
        for meta in metas:
            self.registry.register(meta)
        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)

    def confirm_segments(
        self,
        candidate_ids: List[str],
        confirmed_ids: List[str],
    ) -> None:
        """Stage 1 completion: fix confirmed set, cancel the rest.

        Algorithm:
          - confirmed_set = set(confirmed_ids)
          - Candidates not in confirmed_set → registry.cancel()
          - Blocks unnecessary KV transfer at this point.
        """
        confirmed_set = set(confirmed_ids)
        for seg_id in candidate_ids:
            if seg_id not in confirmed_set:
                self.registry.cancel(seg_id)

    async def fetch_segments_lazy(
        self,
        confirmed_segment_ids: List[str],
    ) -> AsyncIterator[KVSegment]:
        """Stage 2: async pull of confirmed segments only.

        Algorithm:
          1. For each confirmed_id, look up SegmentMeta via registry.get().
          2. Simulate tier-based fetch latency:
             - HBM: hbm_fetch_latency_ms
             - DDR: ddr_fetch_latency_ms (asyncio.sleep)
             - REMOTE: remote_fetch_latency_ms (asyncio.sleep)
          3. Yield KVSegment on fetch completion.
          4. If cache is present, attempt cache.get(segment_id) for actual KV.

        Yields:
            KVSegment — individual segment KV load completion events
        """
        for seg_id in confirmed_segment_ids:
            meta = self.registry.get(seg_id)
            latency = self.config.hbm_fetch_latency_ms
            if meta is not None:
                if meta.tier == "DDR":
                    latency = self.config.ddr_fetch_latency_ms
                elif meta.tier == "REMOTE":
                    latency = self.config.remote_fetch_latency_ms

            await asyncio.sleep(latency / 1000.0)

            kv_tensor: Optional[torch.Tensor] = None
            if self._cache is not None:
                kv_tensor = self._cache.get(seg_id)
            if kv_tensor is None:
                kv_tensor = torch.zeros(1, 64)  # fallback synthetic tensor

            yield KVSegment(
                segment_id=seg_id,
                kv_tensor=kv_tensor,
                source_tier=meta.tier if meta else "HBM",
                load_latency_ms=latency,
            )

    def schedule(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceRequest]:
        """InferenceRunner.run_batch()-compatible schedule() interface.

        Algorithm:
          - Maintains FIFO order (extensible to hit-rate-predictive sorting).
          - Measures metadata delivery overhead per call.

        Returns:
            List[InferenceRequest] — ordered requests (FIFO in this implementation)
        """
        t0 = time.monotonic()
        result = list(requests)
        overhead_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(overhead_ms)
        return result

    def scheduling_overhead_ms_p50(self) -> float:
        """Median scheduling overhead in ms."""
        if not self._scheduling_times:
            return 0.0
        sorted_t = sorted(self._scheduling_times)
        return sorted_t[len(sorted_t) // 2]

    def unnecessary_transfer_ratio(self) -> float:
        """Unnecessary transfer ratio (record in results/<exp>/metrics.json)."""
        return self.registry.unnecessary_transfer_ratio()

    def reset_stats(self) -> None:
        self._scheduling_times.clear()
        self.registry.reset_stats()
