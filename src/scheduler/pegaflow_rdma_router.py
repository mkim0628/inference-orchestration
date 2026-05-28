"""Activity A-2: PegaFlow Bloom Filter RDMA cross-node segment router.

Maintains per-peer Bloom Filter indexes of cached segments and routes
segment requests to local PegaFlow or remote nodes via RDMA.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from src.scheduler.base import BaseScheduler
from src.cache.pegaflow_kv_connector import PegaFlowKVConnector, MockPegaFlowConnector


@dataclass
class PeerNodeEntry:
    node_id: str
    rdma_address: str
    port: int


@dataclass
class PegaFlowRDMARouterConfig:
    peer_nodes_config_path: str = "configs/pegaflow_peer_nodes.yaml"
    bloom_filter_capacity: int = 100_000
    bloom_filter_error_rate: float = 0.01
    bloom_sync_interval_ms: float = 500.0
    rdma_reuse_discount: float = 0.8
    seed: int = 42


class _BloomFilter:
    """Minimal Bloom Filter using k independent hash functions over a bit array."""

    def __init__(self, capacity: int, error_rate: float) -> None:
        # Optimal number of bits: m = -(n * ln(p)) / (ln(2)^2)
        m = -int(capacity * math.log(error_rate) / (math.log(2) ** 2))
        self._size = max(m, 1)
        # Optimal number of hash functions: k = (m/n) * ln(2)
        self._k = max(1, int((self._size / capacity) * math.log(2)))
        self._bits = bytearray(math.ceil(self._size / 8))

    def _hash_positions(self, item: bytes) -> List[int]:
        positions: List[int] = []
        for i in range(self._k):
            digest = hashlib.sha256(item + i.to_bytes(4, "little")).digest()
            pos = int.from_bytes(digest[:4], "big") % self._size
            positions.append(pos)
        return positions

    def add(self, item: bytes) -> None:
        for pos in self._hash_positions(item):
            self._bits[pos // 8] |= 1 << (pos % 8)

    def __contains__(self, item: bytes) -> bool:
        return all(
            self._bits[pos // 8] & (1 << (pos % 8))
            for pos in self._hash_positions(item)
        )

    def to_bytes(self) -> bytes:
        return bytes(self._bits)

    def update_from_bytes(self, bitmap: bytes) -> None:
        # OR-merge incoming bitmap — conservative (never remove entries)
        for i, b in enumerate(bitmap[: len(self._bits)]):
            self._bits[i] |= b


class PeerRegistry:
    """Per-peer Bloom Filter index for segment presence queries.

    False positives are tolerable (may cause unnecessary RDMA attempts).
    False negatives are avoided (no missed hits).
    """

    def __init__(self, config: PegaFlowRDMARouterConfig) -> None:
        self.config = config
        self._peers: Dict[str, PeerNodeEntry] = {}
        self._filters: Dict[str, _BloomFilter] = {}
        self._local_filter = _BloomFilter(
            config.bloom_filter_capacity, config.bloom_filter_error_rate
        )
        self._load_peers()

    def _load_peers(self) -> None:
        try:
            with open(self.config.peer_nodes_config_path) as f:
                data = yaml.safe_load(f)
            for entry in data.get("peer_nodes", []) or []:
                node = PeerNodeEntry(
                    node_id=entry["node_id"],
                    rdma_address=entry["rdma_address"],
                    port=entry["port"],
                )
                self._peers[node.node_id] = node
                self._filters[node.node_id] = _BloomFilter(
                    self.config.bloom_filter_capacity,
                    self.config.bloom_filter_error_rate,
                )
        except (FileNotFoundError, Exception):
            pass  # No peer config → single-node mode

    def peer_ids(self) -> List[str]:
        return list(self._peers.keys())

    def get_peer(self, node_id: str) -> Optional[PeerNodeEntry]:
        return self._peers.get(node_id)

    def has_segment(self, node_id: str, segment_id: bytes) -> bool:
        """Return True if node_id likely has segment_id (Bloom Filter query)."""
        bf = self._filters.get(node_id)
        if bf is None:
            return False
        return segment_id in bf

    def update_bloom_from_peer(self, node_id: str, bloom_bitmap: bytes) -> None:
        """Merge peer Bloom Filter bitmap into local index."""
        if node_id not in self._filters:
            self._filters[node_id] = _BloomFilter(
                self.config.bloom_filter_capacity, self.config.bloom_filter_error_rate
            )
        self._filters[node_id].update_from_bytes(bloom_bitmap)

    def register_local_segment(self, segment_id: bytes) -> None:
        """Add local segment to the local Bloom Filter for future broadcast."""
        self._local_filter.add(segment_id)

    def local_bloom_bitmap(self) -> bytes:
        """Return local Bloom Filter bitmap for peer broadcast."""
        return self._local_filter.to_bytes()


class PegaFlowRDMACrossNodeRouter(BaseScheduler):
    """Bloom Filter-based peer segment index + RDMA cross-node segment router.

    Routing order:
      1. Local PegaFlow (GIL-free Unix socket)
      2. Peer nodes via Bloom Filter index → RDMA transfer if cost-effective
      3. Miss

    Scheduling unit: per-request (segment lookup granularity)
    Cache state access: O(1) Bloom Filter query then RDMA cost comparison
    """

    def __init__(
        self,
        local_connector: "CacheStore",
        peer_registry: PeerRegistry,
        config: PegaFlowRDMARouterConfig,
        rdma_bandwidth_gbps: float = 200.0,
    ) -> None:
        self.local_connector = local_connector
        self.peer_registry = peer_registry
        self.config = config
        # Default bandwidth used when no per-peer table available
        self._default_rdma_bandwidth_gbps = rdma_bandwidth_gbps
        # Simulated remote KV store for mock RDMA in test environments
        self._mock_remote_stores: Dict[str, Dict[bytes, torch.Tensor]] = {}

    def _get_peer_bandwidth_gbps(self, target_node_id: str) -> float:
        """Return RDMA bandwidth in GB/s for the given peer node."""
        return self._default_rdma_bandwidth_gbps

    def estimate_rdma_latency_ms(
        self,
        segment_size_bytes: int,
        target_node_id: str,
    ) -> float:
        """Estimate RDMA transfer latency: size / bandwidth × 1000 (ms)."""
        bw_gbps = self._get_peer_bandwidth_gbps(target_node_id)
        bw_bytes_per_ms = bw_gbps * 1e9 / 1000.0
        if bw_bytes_per_ms <= 0:
            return float("inf")
        return segment_size_bytes / bw_bytes_per_ms

    def estimate_recompute_latency_ms(
        self,
        segment_token_count: int,
        gpu_throughput_tokens_per_ms: float = 1.0,
    ) -> float:
        """Estimate recomputation latency: tokens / throughput (ms)."""
        if gpu_throughput_tokens_per_ms <= 0:
            return float("inf")
        return segment_token_count / gpu_throughput_tokens_per_ms

    def _mock_rdma_fetch(
        self, node_id: str, segment_id: bytes
    ) -> Optional[torch.Tensor]:
        """Simulate RDMA fetch from a peer node (used in mock/test mode)."""
        store = self._mock_remote_stores.get(node_id, {})
        return store.get(segment_id)

    def register_mock_remote_segment(
        self, node_id: str, segment_id: bytes, tensor: torch.Tensor
    ) -> None:
        """Register a segment in mock remote store and update peer Bloom Filter.

        Also registers the peer in PeerRegistry._peers so peer_ids() returns it.
        """
        if node_id not in self._mock_remote_stores:
            self._mock_remote_stores[node_id] = {}
        self._mock_remote_stores[node_id][segment_id] = tensor
        # Ensure Bloom Filter reflects the registered segment
        if node_id not in self.peer_registry._filters:
            self.peer_registry._filters[node_id] = _BloomFilter(
                self.peer_registry.config.bloom_filter_capacity,
                self.peer_registry.config.bloom_filter_error_rate,
            )
        self.peer_registry._filters[node_id].add(segment_id)
        # Register peer so peer_ids() returns it
        if node_id not in self.peer_registry._peers:
            self.peer_registry._peers[node_id] = PeerNodeEntry(
                node_id=node_id, rdma_address="127.0.0.1", port=9000
            )

    def route_segment_request(
        self,
        segment_id: bytes,
        local_node_id: str,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Route a segment lookup to local or remote storage.

        Returns:
          (tensor, hit_type): hit_type ∈ {"pegaflow_local", "rdma_remote", "miss"}
        """
        # Step 1: Local PegaFlow lookup
        local_result = self.local_connector.get(segment_id.hex())
        if local_result is not None:
            return local_result, "pegaflow_local"

        # Step 2: Peer Bloom Filter scan → RDMA cost comparison
        for peer_id in self.peer_registry.peer_ids():
            if peer_id == local_node_id:
                continue
            if not self.peer_registry.has_segment(peer_id, segment_id):
                continue

            # Estimate transfer vs recompute cost
            # Use a small sentinel size estimate for cost comparison when size unknown
            sentinel_size = 1024 * 512  # 512 KB default segment estimate
            rdma_ms = self.estimate_rdma_latency_ms(sentinel_size, peer_id)
            recompute_ms = self.estimate_recompute_latency_ms(256)  # avg chunk tokens

            if rdma_ms < recompute_ms * self.config.rdma_reuse_discount:
                tensor = self._mock_rdma_fetch(peer_id, segment_id)
                if tensor is not None:
                    return tensor, "rdma_remote"

        # Step 3: Miss
        return None, "miss"

    def route_segment_request_with_size(
        self,
        segment_id: bytes,
        local_node_id: str,
        segment_size_bytes: int,
        segment_token_count: int,
        gpu_throughput_tokens_per_ms: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Route with explicit size and token count for accurate cost comparison."""
        local_result = self.local_connector.get(segment_id.hex())
        if local_result is not None:
            return local_result, "pegaflow_local"

        for peer_id in self.peer_registry.peer_ids():
            if peer_id == local_node_id:
                continue
            if not self.peer_registry.has_segment(peer_id, segment_id):
                continue

            rdma_ms = self.estimate_rdma_latency_ms(segment_size_bytes, peer_id)
            recompute_ms = self.estimate_recompute_latency_ms(
                segment_token_count, gpu_throughput_tokens_per_ms
            )

            if rdma_ms < recompute_ms * self.config.rdma_reuse_discount:
                tensor = self._mock_rdma_fetch(peer_id, segment_id)
                if tensor is not None:
                    return tensor, "rdma_remote"

        return None, "miss"

    def schedule(self, requests: List) -> List:
        """BaseScheduler interface — returns requests unchanged (routing is a side-effect)."""
        return requests
