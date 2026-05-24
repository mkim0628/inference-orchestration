"""DualPath storage NIC idle-awareness KV load balancer (arXiv 2602.21548).

Activity A: KV Cache-aware Scheduling — multi-node P/D disaggregation.
Routes KV loads over a secondary decode-node relay path when the primary
storage NIC towards the prefill node is saturated.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import torch

from src.scheduler.base import BaseScheduler


@dataclass
class NodeNICStatus:
    node_id: str
    node_type: str           # "prefill" | "decode"
    nic_utilization: float   # 0.0–1.0
    active_dual_path: int    # current number of dual-path KV loads active
    last_updated: float      # time.monotonic()


@dataclass
class RoutingDecision:
    request_id: str
    path: str                         # "single" | "dual"
    relay_decode_node_id: Optional[str]
    prefill_nic_utilization: float
    decision_latency_ms: float


@dataclass
class DualPathNICConfig:
    nic_saturation_threshold: float = 0.80
    idle_nic_threshold: float = 0.30
    max_dual_path_per_node: int = 4
    nic_monitor_interval_ms: float = 200.0
    stale_threshold_ms: float = 1000.0
    seed: int = 42


class DualPathNICLoadBalancer(BaseScheduler):
    """DualPath storage NIC utilization-aware KV load balancer (arXiv 2602.21548).

    Activity A: KV Cache-aware Scheduling — multi-node P/D disaggregation.

    Scheduling unit: per KV load request.
    Cache state access: NodeNICStatus dict O(1) lookup.

    Multi-node environment:
      - Storage NIC (storage network) and compute NIC (RDMA) physically separated.
      - gRPC heartbeat every 200 ms collects each node's NIC utilization.
      - Single path: storage -> prefill node (default).
      - Dual path: storage -> decode node (idle NIC) -> prefill node (RDMA).
      - Dual path avoids interference with model execution RDMA channels via
        network separation.

    Routing decision algorithm:
      if prefill_nic_util < nic_saturation_threshold (0.80):
          -> single path (default)
      elif len(idle_decode_nodes) > 0:
          -> dual path, min_load_first + round-robin relay node selection
      else:
          -> single path (saturated but no alternative)

    Overhead: NIC load lookup O(1) + idle decode scan O(N_decode) < 0.1ms/request.
    """

    def __init__(self, config: DualPathNICConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._node_status: Dict[str, NodeNICStatus] = {}
        self._single_path_count: int = 0
        self._dual_path_count: int = 0
        self._decision_latencies_ms: List[float] = []
        self._rr_index: int = 0

    def update_nic_status(
        self,
        node_id: str,
        node_type: str,
        nic_utilization: float,
        active_dual_path: int = 0,
    ) -> None:
        """Update node storage NIC load state (called on gRPC heartbeat receipt)."""
        self._node_status[node_id] = NodeNICStatus(
            node_id=node_id,
            node_type=node_type,
            nic_utilization=float(nic_utilization),
            active_dual_path=active_dual_path,
            last_updated=time.monotonic(),
        )

    def _get_prefill_nic_utilization(self) -> float:
        """Maximum storage NIC utilization across current prefill nodes."""
        now = time.monotonic()
        utils = [
            s.nic_utilization
            for s in self._node_status.values()
            if s.node_type == "prefill"
            and (now - s.last_updated) * 1000 < self.config.stale_threshold_ms
        ]
        return max(utils) if utils else 0.0

    def _get_idle_decode_nodes(self) -> List[NodeNICStatus]:
        """Decode nodes with idle NIC and available dual-path slots, sorted min-load first."""
        now = time.monotonic()
        idle = [
            s for s in self._node_status.values()
            if s.node_type == "decode"
            and s.nic_utilization < self.config.idle_nic_threshold
            and s.active_dual_path < self.config.max_dual_path_per_node
            and (now - s.last_updated) * 1000 < self.config.stale_threshold_ms
        ]
        return sorted(idle, key=lambda n: n.nic_utilization)

    def decide_routing(self, request: Any) -> RoutingDecision:
        """Determine KV load routing path for a single request.

        Returns:
            RoutingDecision with path="single" or path="dual"
        """
        t_start = time.monotonic()
        request_id = getattr(request, "request_id", str(id(request)))
        prefill_nic_util = self._get_prefill_nic_utilization()

        def _make_single() -> RoutingDecision:
            self._single_path_count += 1
            lat = (time.monotonic() - t_start) * 1000
            self._decision_latencies_ms.append(lat)
            return RoutingDecision(
                request_id=request_id,
                path="single",
                relay_decode_node_id=None,
                prefill_nic_utilization=prefill_nic_util,
                decision_latency_ms=lat,
            )

        if prefill_nic_util < self.config.nic_saturation_threshold:
            return _make_single()

        idle_decode = self._get_idle_decode_nodes()
        if not idle_decode:
            return _make_single()

        # Dual path: round-robin across min-load-first sorted idle nodes
        relay_node = idle_decode[self._rr_index % len(idle_decode)]
        self._rr_index += 1
        if relay_node.node_id in self._node_status:
            self._node_status[relay_node.node_id].active_dual_path += 1
        self._dual_path_count += 1

        lat = (time.monotonic() - t_start) * 1000
        self._decision_latencies_ms.append(lat)
        return RoutingDecision(
            request_id=request_id,
            path="dual",
            relay_decode_node_id=relay_node.node_id,
            prefill_nic_utilization=prefill_nic_util,
            decision_latency_ms=lat,
        )

    def complete_dual_path(self, decode_node_id: str) -> None:
        """Decrement active dual-path counter on decode node when transfer completes."""
        if decode_node_id in self._node_status:
            s = self._node_status[decode_node_id]
            s.active_dual_path = max(0, s.active_dual_path - 1)

    def schedule(self, requests: List[Any]) -> List[Any]:
        """BaseScheduler interface: attach routing_decision to each request and return."""
        result = []
        for req in requests:
            decision = self.decide_routing(req)
            try:
                req.routing_decision = decision
            except AttributeError:
                pass
            result.append(req)
        return result

    def dual_path_ratio(self) -> float:
        total = self._single_path_count + self._dual_path_count
        return self._dual_path_count / max(1, total)

    def p99_decision_latency_ms(self) -> float:
        if not self._decision_latencies_ms:
            return 0.0
        s = sorted(self._decision_latencies_ms)
        idx = min(int(len(s) * 0.99), len(s) - 1)
        return s[idx]

    def scheduling_stats(self) -> dict:
        return {
            "single_path_count": self._single_path_count,
            "dual_path_count": self._dual_path_count,
            "dual_path_ratio": self.dual_path_ratio(),
            "decision_latency_p99_ms": self.p99_decision_latency_ms(),
            "decision_latency_mean_ms": (
                sum(self._decision_latencies_ms) / max(1, len(self._decision_latencies_ms))
            ),
        }

    def reset_stats(self) -> None:
        self._single_path_count = 0
        self._dual_path_count = 0
        self._decision_latencies_ms.clear()
        self._rr_index = 0
