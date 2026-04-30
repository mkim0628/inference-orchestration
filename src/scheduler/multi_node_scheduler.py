"""Multi-node P/D-disaggregated KV cache scheduler (Activity A).

Extends CacheAwareScheduler with prefill-node / decode-node routing.
Each request is annotated with the best (prefill_node, decode_node) pair
chosen to maximise cache reuse while minimising transfer latency.

Routing score:
    score = predicted_hit_rate(req) / (1 + transfer_latency_ms / 1000)

When prefill_nodes=[] or decode_nodes=[], the scheduler falls back to the
parent CacheAwareScheduler behaviour (single-node mode).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

from src.scheduler.cache_aware_scheduler import CacheAwareScheduler
from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class NodeConfig:
    """Configuration for a single inference node (prefill or decode)."""

    node_id: str
    node_type: str              # "prefill" or "decode"
    transfer_latency_ms: float = 10.0
    memory_capacity_gb: float = 80.0
    current_load: float = 0.0   # 0.0–1.0


class MultiNodeScheduler(CacheAwareScheduler):
    """Cache-aware scheduler with P/D-disaggregated node routing.

    Inherits hit-rate-weighted priority ordering from CacheAwareScheduler and
    adds per-request routing annotations and optional KV compression before
    cross-node transfer.
    """

    def __init__(
        self,
        cache: CacheStore,
        prefill_nodes: List[NodeConfig],
        decode_nodes: List[NodeConfig],
        codec=None,
        compress_threshold_bytes: int = 1024 * 1024,
        fairness_max_wait: int = 10,
        chunk_size: int = 128,
    ) -> None:
        super().__init__(cache, fairness_max_wait, chunk_size)
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes
        self.codec = codec
        self.compress_threshold_bytes = compress_threshold_bytes
        self._transfer_log: List[dict] = []

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Return requests ordered by parent priority; annotate each with routing.

        If no nodes are configured the result is identical to the parent scheduler.
        Routing annotation is stored on each request as `_prefill_node` and
        `_decode_node` attributes (for downstream inspection / simulation).
        """
        ordered = super().schedule(requests)

        if self.prefill_nodes and self.decode_nodes:
            for req in ordered:
                p_node, d_node = self.route(req)
                req._prefill_node = p_node  # type: ignore[attr-defined]
                req._decode_node = d_node   # type: ignore[attr-defined]

        return ordered

    def route(self, request: InferenceRequest) -> Tuple[NodeConfig, NodeConfig]:
        """Select the best (prefill_node, decode_node) pair for a request.

        Prefill node selection: maximise routing_score =
            hit_rate(req) / (1 + transfer_latency_ms / 1000)

        Decode node selection: minimise current_load (least busy node).

        Falls back to first available node when lists have only one element.
        """
        predicted_hr = self._predict_hit_rate(request)

        best_prefill = max(
            self.prefill_nodes,
            key=lambda n: predicted_hr / (1.0 + n.transfer_latency_ms / 1000.0),
        )

        best_decode = min(self.decode_nodes, key=lambda n: n.current_load)

        return best_prefill, best_decode

    def simulate_transfer(
        self,
        kv: torch.Tensor,
        prefill_node: NodeConfig,
        decode_node: NodeConfig,
    ) -> Tuple[torch.Tensor, float]:
        """Simulate KV transfer from prefill node to decode node.

        If the tensor exceeds compress_threshold_bytes and a codec is available,
        compress before transfer (latency reduced by 4× due to bandwidth savings).

        Returns:
            (kv_out, latency_ms) — tensor after optional compress/decompress,
            and simulated transfer latency in milliseconds.
        """
        latency_ms = prefill_node.transfer_latency_ms * (1.0 + decode_node.current_load)

        if kv.nbytes > self.compress_threshold_bytes and self.codec is not None:
            compressed = self.codec.encode(kv, layer_idx=5, tensor_id=0)
            kv_out = self.codec.decode(compressed, layer_idx=5, tensor_id=0).to(kv.dtype)
            latency_ms *= 0.25  # bandwidth savings from compression
            self._transfer_log.append({
                "compressed": True,
                "original_bytes": kv.nbytes,
                "compressed_bytes": compressed.nbytes,
                "latency_ms": latency_ms,
            })
        else:
            kv_out = kv
            self._transfer_log.append({
                "compressed": False,
                "original_bytes": kv.nbytes,
                "latency_ms": latency_ms,
            })

        return kv_out, latency_ms

    def node_load(self) -> Dict[str, float]:
        """Return current load for every known node (prefill + decode)."""
        result: Dict[str, float] = {}
        for node in self.prefill_nodes + self.decode_nodes:
            result[node.node_id] = node.current_load
        return result
