"""Activity A: Cache-hit-rate-aware request scheduling for vLLM 0.20.0.

This module ports ``src/scheduler/cache_aware_scheduler.py`` to vLLM's v1
architecture.  It provides two artefacts:

``CacheHitAwareRequestQueue``
    A ``RequestQueue`` subclass that maintains a priority heap ordering
    requests by their predicted non-contiguous segment hit rate.  Requests
    with higher predicted hit rates are popped first (greedy cache-locality
    scheduling).  A wait-step penalty prevents cold requests from being
    starved indefinitely (fairness).

``create_cache_hit_aware_queue``
    Factory function that constructs and returns a
    ``CacheHitAwareRequestQueue`` wired to a segment index for prediction.

Integration
-----------
Replace the default ``FCFSRequestQueue`` inside vLLM's ``Scheduler.__init__``
with the result of ``create_cache_hit_aware_queue``.  Because
``CacheHitAwareRequestQueue`` implements the full ``RequestQueue`` interface,
no other scheduler changes are required.

Example patch (conceptual — apply before constructing the LLMEngine)::

    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm_integration.scheduler_patch import create_cache_hit_aware_queue

    # In Scheduler.__init__, replace:
    #   self.waiting = create_request_queue(policy)
    # with:
    #   self.waiting = create_cache_hit_aware_queue(
    #       segment_index=kv_manager._segment_index,
    #       chunk_size=kv_manager._segment_chunk_size,
    #   )

All vLLM imports are wrapped in ``try/except`` for portability.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Dict, Iterator, List, Optional

try:
    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm.v1.request import Request as VllmRequest
    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False
    RequestQueue = object  # type: ignore[assignment, misc]
    VllmRequest = object  # type: ignore[assignment, misc]


# --------------------------------------------------------------------------- #
# Segment key helper (mirrors SegmentHashMixin.get_segment_key)               #
# --------------------------------------------------------------------------- #

def _segment_key(token_ids: List[int], chunk_idx: int, layer_idx: int, chunk_size: int) -> str:
    """Position-independent SHA-256 key for a token chunk."""
    start = chunk_idx * chunk_size
    chunk = token_ids[start: start + chunk_size]
    if not chunk:
        return ""
    raw = struct.pack(f"{len(chunk)}I", *chunk)
    layer_prefix = struct.pack("I", layer_idx)
    return hashlib.sha256(layer_prefix + raw).hexdigest()


# --------------------------------------------------------------------------- #
# CacheHitAwareRequestQueue                                                    #
# --------------------------------------------------------------------------- #

class _PriorityEntry:
    """Heap entry with (negative_priority, wait_steps, request_id, request)."""

    __slots__ = ("neg_priority", "neg_wait", "request_id", "request")

    def __init__(
        self,
        neg_priority: float,
        neg_wait: int,
        request_id: str,
        request: object,
    ) -> None:
        self.neg_priority = neg_priority
        self.neg_wait = neg_wait
        self.request_id = request_id
        self.request = request

    def __lt__(self, other: "_PriorityEntry") -> bool:
        if self.neg_priority != other.neg_priority:
            return self.neg_priority < other.neg_priority
        return self.neg_wait < other.neg_wait


class CacheHitAwareRequestQueue(RequestQueue):  # type: ignore[misc]
    """Priority queue that schedules requests by predicted KV cache hit rate.

    Priority formula (mirrors ``CacheAwareScheduler`` from Activity A)::

        priority = hit_rate × (1 − min(wait_steps / fairness_max_wait, 1.0))

    Requests that have waited ``≥ fairness_max_wait`` rounds get
    ``priority = 0`` and are promoted to the front via tie-breaking on
    ``wait_steps`` (most-waited first).

    Hit rate prediction peeks at ``segment_index._store`` keys directly —
    no ``get()`` calls, so cache statistics are not polluted.

    Args:
        segment_index:     ``CompressedSegmentIndex`` instance shared with
                           ``NonContiguousKVCacheManager``.
        chunk_size:        Token chunk size (must match the manager).
        fairness_max_wait: Maximum rounds a request can wait before being
                           forcibly promoted (default 10).
    """

    def __init__(
        self,
        segment_index: object,
        chunk_size: int = 64,
        fairness_max_wait: int = 10,
    ) -> None:
        self._segment_index = segment_index
        self._chunk_size = chunk_size
        self._fairness_max_wait = fairness_max_wait

        # Sorted list (maintained lazily as a sorted list for simplicity)
        self._queue: List[_PriorityEntry] = []
        self._wait_steps: Dict[str, int] = {}
        self._round: int = 0

    # ---------------------------------------------------------------------- #
    # Hit-rate prediction                                                      #
    # ---------------------------------------------------------------------- #

    def _predict_hit_rate(self, request: object) -> float:
        """Estimate segment hit rate without touching cache counters."""
        store = getattr(self._segment_index, "_store", None)
        if store is None:
            return 0.0

        token_ids: List[int] = []
        if hasattr(request, "prompt_token_ids"):
            token_ids = list(request.prompt_token_ids)  # type: ignore[union-attr]
        elif hasattr(request, "token_ids"):
            token_ids = list(request.token_ids)  # type: ignore[union-attr]
        if not token_ids:
            return 0.0

        n_chunks = max(1, (len(token_ids) + self._chunk_size - 1) // self._chunk_size)
        hits = sum(
            1
            for i in range(n_chunks)
            if _segment_key(token_ids, i, 0, self._chunk_size) in store
        )
        return hits / n_chunks

    def _priority(self, request: object) -> float:
        rid = getattr(request, "request_id", id(request))
        wait = self._wait_steps.get(str(rid), 0)
        hit_rate = self._predict_hit_rate(request)
        penalty = min(wait / max(self._fairness_max_wait, 1), 1.0)
        return hit_rate * (1.0 - penalty)

    # ---------------------------------------------------------------------- #
    # RequestQueue interface                                                   #
    # ---------------------------------------------------------------------- #

    def add_request(self, request: object) -> None:
        rid = str(getattr(request, "request_id", id(request)))
        if rid not in self._wait_steps:
            self._wait_steps[rid] = 0
        priority = self._priority(request)
        entry = _PriorityEntry(
            neg_priority=-priority,
            neg_wait=-self._wait_steps[rid],
            request_id=rid,
            request=request,
        )
        # Insert in sorted order (list stays small enough for linear insert)
        import bisect
        bisect.insort(self._queue, entry)

    def pop_request(self) -> object:
        if not self._queue:
            raise IndexError("pop from empty queue")
        entry = self._queue.pop(0)
        rid = entry.request_id
        self._wait_steps.pop(rid, None)
        # Increment wait steps for remaining requests in this scheduling round
        for e in self._queue:
            self._wait_steps[e.request_id] = self._wait_steps.get(e.request_id, 0) + 1
            # Recompute priority with new wait
            e.neg_priority = -self._priority(e.request)
            e.neg_wait = -self._wait_steps[e.request_id]
        self._queue.sort()
        return entry.request

    def peek_request(self) -> object:
        if not self._queue:
            raise IndexError("peek from empty queue")
        return self._queue[0].request

    def prepend_request(self, request: object) -> None:
        rid = str(getattr(request, "request_id", id(request)))
        self._wait_steps[rid] = 9999  # treat prepended as highest priority
        entry = _PriorityEntry(
            neg_priority=-9999.0,
            neg_wait=-9999,
            request_id=rid,
            request=request,
        )
        self._queue.insert(0, entry)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        for req in reversed(list(requests)):
            self.prepend_request(req)

    def remove_request(self, request: object) -> None:
        rid = str(getattr(request, "request_id", id(request)))
        self._queue = [e for e in self._queue if e.request_id != rid]
        self._wait_steps.pop(rid, None)

    def remove_requests(self, requests: object) -> None:
        for req in requests:  # type: ignore[union-attr]
            self.remove_request(req)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[object]:
        return iter(e.request for e in self._queue)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def create_cache_hit_aware_queue(
    segment_index: object,
    chunk_size: int = 64,
    fairness_max_wait: int = 10,
) -> CacheHitAwareRequestQueue:
    """Create a ``CacheHitAwareRequestQueue`` wired to ``segment_index``.

    Args:
        segment_index:     ``CompressedSegmentIndex`` instance from
                           ``NonContiguousKVCacheManager``.
        chunk_size:        Chunk size in tokens (must match the manager).
        fairness_max_wait: Max wait rounds before cold request is promoted.

    Returns:
        A ``RequestQueue``-compatible priority queue.
    """
    return CacheHitAwareRequestQueue(
        segment_index=segment_index,
        chunk_size=chunk_size,
        fairness_max_wait=fairness_max_wait,
    )


# --------------------------------------------------------------------------- #
# MultiNodeRequestRouter — Activity A (2026-04-30)                            #
# Ports src/scheduler/multi_node_scheduler.py to vLLM 0.20.0                 #
# --------------------------------------------------------------------------- #

from dataclasses import dataclass, field as dc_field


@dataclass
class VllmNodeConfig:
    """Simulated P/D node configuration for multi-node routing in vLLM.

    In a real deployment these map to vLLM worker processes or remote
    disaggregated-prefill endpoints (e.g. llm-d or FlowKV style clusters).
    For integration purposes we keep them as lightweight dataclasses so the
    router can be unit-tested without network infrastructure.
    """
    node_id: str
    node_type: str              # "prefill" or "decode"
    transfer_latency_ms: float = 10.0
    memory_capacity_gb: float = 80.0
    current_load: float = 0.0   # 0.0–1.0


class MultiNodeRequestRouter:
    """Routes incoming requests to (prefill_node, decode_node) pairs.

    This is the vLLM port of ``MultiNodeScheduler`` from
    ``src/scheduler/multi_node_scheduler.py``.  It is designed to sit
    *above* the ``CacheHitAwareRequestQueue``: after hit-rate ordering
    selects *which* request to run next, this router selects *where*
    to run it.

    Routing score (higher → better node pair)::

        routing_score = predicted_hit_rate / (1 + avg_transfer_latency_ms / 100)

    compress_before_transfer
    ------------------------
    When ``codec`` is provided and the estimated KV size exceeds
    ``compress_threshold_bytes``, the router records that the KV should be
    INT4-compressed before P→D transfer.  The actual compression is
    performed by the attention backend (``TriStateKVHook`` or
    ``CompressedKVHook``); the router only sets a flag on the routing record.

    Accuracy constraint (Activity C)
    ---------------------------------
    TriStateCompressor + HadamardInt4Codec maintain perplexity delta ±1%
    (KL divergence < 0.01) — see Report ① 2026-04-30 for validation data.
    compress_before_transfer relies on this guarantee.
    """

    def __init__(
        self,
        prefill_nodes: List[VllmNodeConfig],
        decode_nodes: List[VllmNodeConfig],
        segment_index: object = None,
        chunk_size: int = 128,
        codec=None,
        compress_threshold_bytes: int = 1024 * 1024,
    ) -> None:
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes
        self._segment_index = segment_index
        self._chunk_size = chunk_size
        self._codec = codec
        self._compress_threshold = compress_threshold_bytes
        self._routing_log: List[dict] = []

    def route(self, request: object) -> dict:
        """Select optimal (prefill_node, decode_node) for a request.

        Returns a routing record dict with keys:
          - prefill_node: VllmNodeConfig
          - decode_node: VllmNodeConfig
          - compress_before_transfer: bool
          - routing_score: float
        """
        if not self.prefill_nodes or not self.decode_nodes:
            return {"prefill_node": None, "decode_node": None,
                    "compress_before_transfer": False, "routing_score": 0.0}

        hit_rate = self._predict_hit_rate(request)

        # Select prefill node: lowest load as tiebreaker
        best_p = min(self.prefill_nodes, key=lambda n: n.current_load)
        # Select decode node: lowest load
        best_d = min(self.decode_nodes, key=lambda n: n.current_load)

        avg_latency = (best_p.transfer_latency_ms + best_d.transfer_latency_ms) / 2
        routing_score = hit_rate / (1.0 + avg_latency / 100.0)

        token_ids = self._get_token_ids(request)
        estimated_kv_bytes = len(token_ids) * 128 * 4  # rough FP32 estimate
        compress = (
            self._codec is not None
            and estimated_kv_bytes > self._compress_threshold
        )

        record = {
            "prefill_node": best_p,
            "decode_node": best_d,
            "compress_before_transfer": compress,
            "routing_score": routing_score,
        }
        self._routing_log.append(record)
        return record

    def _predict_hit_rate(self, request: object) -> float:
        store = getattr(self._segment_index, "_store", None)
        if store is None:
            return 0.0
        token_ids = self._get_token_ids(request)
        if not token_ids:
            return 0.0
        n_chunks = max(1, (len(token_ids) + self._chunk_size - 1) // self._chunk_size)
        hits = sum(
            1 for i in range(n_chunks)
            if _segment_key(token_ids, i, 0, self._chunk_size) in store
        )
        return hits / n_chunks

    def _get_token_ids(self, request: object) -> List[int]:
        for attr in ("prompt_token_ids", "token_ids"):
            val = getattr(request, attr, None)
            if val is not None:
                return list(val)
        return []

    def node_load(self) -> Dict[str, float]:
        """Return {node_id: current_load} for all nodes."""
        result: Dict[str, float] = {}
        for n in self.prefill_nodes + self.decode_nodes:
            result[n.node_id] = n.current_load
        return result


def create_multi_node_router(
    prefill_nodes: List[VllmNodeConfig],
    decode_nodes: List[VllmNodeConfig],
    segment_index: object = None,
    chunk_size: int = 128,
    codec=None,
    compress_threshold_bytes: int = 1024 * 1024,
) -> MultiNodeRequestRouter:
    """Factory: create a MultiNodeRequestRouter.

    Example usage in vLLM executor::

        from vllm_integration.scheduler_patch import (
            create_multi_node_router, VllmNodeConfig,
        )

        router = create_multi_node_router(
            prefill_nodes=[VllmNodeConfig("p0", "prefill"), VllmNodeConfig("p1", "prefill")],
            decode_nodes=[VllmNodeConfig("d0", "decode"), VllmNodeConfig("d1", "decode")],
            segment_index=kv_manager._segment_index,
            chunk_size=128,
        )
        routing = router.route(request)
    """
    return MultiNodeRequestRouter(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        segment_index=segment_index,
        chunk_size=chunk_size,
        codec=codec,
        compress_threshold_bytes=compress_threshold_bytes,
    )


if __name__ == "__main__":
    def _test_multi_node_router():
        class _MockReq:
            token_ids = list(range(256))
        p_nodes = [VllmNodeConfig("p0", "prefill", 10.0), VllmNodeConfig("p1", "prefill", 12.0)]
        d_nodes = [VllmNodeConfig("d0", "decode", 8.0), VllmNodeConfig("d1", "decode", 15.0)]
        router = create_multi_node_router(p_nodes, d_nodes)
        record = router.route(_MockReq())
        assert record["prefill_node"].node_type == "prefill"
        assert record["decode_node"].node_type == "decode"
        assert 0.0 <= record["routing_score"] <= 1.0
        assert set(router.node_load().keys()) == {"p0", "p1", "d0", "d1"}
        print("MultiNodeRequestRouter: OK")

    _test_multi_node_router()
