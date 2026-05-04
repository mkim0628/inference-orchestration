"""DAGTopologyScheduler — workflow DAG topology-based KV proactive preservation (Activity A).

Registers agent workflow DAGs, performs BFS topological analysis to compute
per-node KV reuse probabilities, and pins high-probability segments in
WorkloadAwareTTLCache to prevent premature eviction.

Unknown-DAG requests are delegated to CacheAwareScheduler (fallback).
"""

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.scheduler.cache_aware_scheduler import CacheAwareScheduler
from src.engine.runner import InferenceRequest


@dataclass
class DAGNode:
    agent_id: str
    tool_calls: List[str]
    expected_kv_tokens: int
    parent_ids: List[str]
    out_degree: int = 0               # filled after BFS analysis
    kv_reuse_probability: float = 0.0


@dataclass
class WorkflowDAG:
    dag_id: str
    nodes: Dict[str, DAGNode]         # agent_id → DAGNode
    topological_order: List[str]      # BFS topological order
    completed_nodes: Set[str] = field(default_factory=set)
    belady_upper_bound: float = 0.0   # computed via simulation


class DAGTopologyScheduler:
    """Workflow DAG topology-based KV proactive preservation scheduler (Activity A).

    Scheduling unit: batch.
    Cache interaction: WorkloadAwareTTLCache.pin() / unpin() / adjust_ttl() only.
    """

    def __init__(
        self,
        cache: Any,  # WorkloadAwareTTLCache — avoid circular import at type level
        fallback_scheduler: Optional[CacheAwareScheduler] = None,
        retain_threshold: float = 0.5,
        alpha_ttl_extend: float = 2.0,
        kv_reuse_histogram: Optional[Dict] = None,
        # Optional event callback for DAGAwareTTLAdjuster integration
        on_kv_reuse_event: Optional[Callable[[str, float], None]] = None,
        on_node_complete_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cache = cache
        self.fallback_scheduler = fallback_scheduler
        self.retain_threshold = retain_threshold
        self.alpha_ttl_extend = alpha_ttl_extend
        self._kv_reuse_histogram: Dict[str, list] = kv_reuse_histogram or {}
        self._workflows: Dict[str, WorkflowDAG] = {}
        # Maps (dag_id, agent_id) → segment keys pinned by this scheduler
        self._pinned_segments: Dict[Tuple[str, str], List[str]] = {}
        # Optional callbacks to notify DAGAwareTTLAdjuster
        self._on_kv_reuse_event = on_kv_reuse_event
        self._on_node_complete_event = on_node_complete_event

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def register_workflow(self, dag_spec: dict) -> str:
        """Parse DAG JSON spec, run topological analysis, return dag_id.

        Raises ValueError for cyclic DAGs (detected via Kahn's algorithm).
        """
        dag_id: str = dag_spec["dag_id"]
        raw_nodes: list = dag_spec["nodes"]

        nodes: Dict[str, DAGNode] = {}
        for n in raw_nodes:
            node = DAGNode(
                agent_id=n["agent_id"],
                tool_calls=n.get("tool_calls", []),
                expected_kv_tokens=n.get("expected_kv_tokens", 0),
                parent_ids=n.get("parent_ids", []),
            )
            nodes[node.agent_id] = node

        children = self._build_children_map(nodes)

        # Kahn's algorithm for topological sort
        in_degree: Dict[str, int] = {nid: len(n.parent_ids) for nid, n in nodes.items()}
        queue: deque = deque(nid for nid in nodes if in_degree[nid] == 0)
        topological_order: List[str] = []

        while queue:
            nid = queue.popleft()
            topological_order.append(nid)
            for child_id in children[nid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(topological_order) != len(nodes):
            raise ValueError(
                f"DAG '{dag_id}' contains a cycle — topological sort incomplete."
            )

        # Compute out_degree and kv_reuse_probability
        max_out_degree = max((len(children[nid]) for nid in nodes), default=1)
        max_out_degree = max(max_out_degree, 1)

        # Check if this workflow has historical data (≥10 runs)
        hist = self._kv_reuse_histogram.get(dag_id, [])
        use_histogram = len(hist) >= 10

        for nid in nodes:
            out_deg = len(children[nid])
            nodes[nid].out_degree = out_deg
            if use_histogram:
                # Use histogram mean hit rate for probability
                nodes[nid].kv_reuse_probability = float(sum(hist) / len(hist))
            else:
                nodes[nid].kv_reuse_probability = out_deg / max_out_degree

        dag = WorkflowDAG(
            dag_id=dag_id,
            nodes=nodes,
            topological_order=topological_order,
        )
        dag.belady_upper_bound = self._simulate_belady(dag, children)
        self._workflows[dag_id] = dag
        return dag_id

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Annotate each request with dag_node_id and kv_reuse_probability.

        DAG-registered requests: pin cache segments + annotate.
        Unknown requests: delegate to fallback_scheduler.
        Returns annotated request list in original order.
        """
        dag_requests: List[InferenceRequest] = []
        fallback_requests: List[InferenceRequest] = []

        for req in requests:
            dag_id = getattr(req, "dag_id", None)
            agent_id = getattr(req, "agent_id", None)
            if dag_id is not None and dag_id in self._workflows and agent_id is not None:
                dag_requests.append(req)
            else:
                fallback_requests.append(req)

        # Annotate DAG requests
        for req in dag_requests:
            dag_id = req.dag_id  # type: ignore[attr-defined]
            agent_id = req.agent_id  # type: ignore[attr-defined]
            prob = self.predict_kv_reuse(dag_id, agent_id)
            req.dag_node_id = agent_id  # type: ignore[attr-defined]
            req.kv_reuse_probability = prob  # type: ignore[attr-defined]

            # Pin cache segments for high-probability nodes
            if prob > self.retain_threshold:
                segment_keys = self._get_segment_keys_for_request(req)
                for key in segment_keys:
                    self.cache.pin(key)
                    if self._on_kv_reuse_event is not None:
                        self._on_kv_reuse_event(key, prob)
                self._pinned_segments[(dag_id, agent_id)] = segment_keys

        # Delegate unknown requests to fallback
        if fallback_requests and self.fallback_scheduler is not None:
            fallback_requests = self.fallback_scheduler.schedule(fallback_requests)

        # Reconstruct original order
        annotated = {id(r): r for r in dag_requests}
        result: List[InferenceRequest] = []
        for req in requests:
            if id(req) in annotated:
                result.append(annotated[id(req)])
            else:
                result.append(req)

        # Replace fallback requests in result with scheduled versions
        fallback_iter = iter(fallback_requests)
        fallback_originals = {id(r) for r in requests
                              if getattr(r, "dag_id", None) is None
                              or getattr(r, "dag_id", None) not in self._workflows}
        result2: List[InferenceRequest] = []
        for req in requests:
            req_is_fallback = (id(req) in {id(r) for r in fallback_requests + [req]}
                               and id(req) not in {id(r) for r in dag_requests})
            if req_is_fallback:
                try:
                    result2.append(next(fallback_iter))
                except StopIteration:
                    result2.append(req)
            else:
                result2.append(req)

        return result

    def notify_node_complete(self, dag_id: str, agent_id: str) -> None:
        """Signal that a DAG node has finished processing.

        Unpins associated segments and fires on_node_complete_event callback.
        """
        if dag_id not in self._workflows:
            return

        dag = self._workflows[dag_id]
        dag.completed_nodes.add(agent_id)

        # Update histogram entry
        if dag_id not in self._kv_reuse_histogram:
            self._kv_reuse_histogram[dag_id] = []

        segment_keys = self._pinned_segments.pop((dag_id, agent_id), [])
        for key in segment_keys:
            self.cache.unpin(key)
            if self._on_node_complete_event is not None:
                self._on_node_complete_event(key)

    def predict_kv_reuse(self, dag_id: str, agent_id: str) -> float:
        """Return the KV reuse probability for a specific DAG node."""
        if dag_id not in self._workflows:
            return 0.0
        dag = self._workflows[dag_id]
        if agent_id not in dag.nodes:
            return 0.0
        return dag.nodes[agent_id].kv_reuse_probability

    def compute_belady_upper_bound(self, dag_id: str) -> float:
        """Return the Bélády upper-bound hit rate for this DAG.

        Assumes full future access knowledge (omniscient eviction).
        """
        if dag_id not in self._workflows:
            return 0.0
        return self._workflows[dag_id].belady_upper_bound

    def save_reuse_histogram(self, output_path: str) -> None:
        """Serialize kv_reuse_probability histogram to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload: Dict[str, Any] = {}
        for dag_id, dag in self._workflows.items():
            payload[dag_id] = {
                nid: {
                    "kv_reuse_probability": node.kv_reuse_probability,
                    "out_degree": node.out_degree,
                    "expected_kv_tokens": node.expected_kv_tokens,
                }
                for nid, node in dag.nodes.items()
            }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_children_map(
        self, nodes: Dict[str, DAGNode]
    ) -> Dict[str, List[str]]:
        """Build parent→children adjacency list from parent_ids fields."""
        children: Dict[str, List[str]] = defaultdict(list)
        for nid in nodes:
            children[nid]  # ensure every node has an entry
        for nid, node in nodes.items():
            for parent_id in node.parent_ids:
                children[parent_id].append(nid)
        return dict(children)

    def _get_segment_keys_for_request(self, req: InferenceRequest) -> List[str]:
        """Collect segment keys currently in cache for a request's token_ids."""
        store = getattr(self.cache, "_store", {})
        if not store:
            return []
        # Return all keys that exist in the cache store (lightweight check)
        # In a real system this would generate chunk keys from token_ids;
        # here we return keys that belong to the request via attribute if set.
        segment_keys = getattr(req, "segment_keys", [])
        if not segment_keys:
            # Fallback: generate chunk keys and intersect with store
            import hashlib
            import struct
            chunk_size = getattr(self.cache, "chunk_size", 128)
            token_ids = req.token_ids
            n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                chunk = token_ids[start: start + chunk_size]
                raw = struct.pack(f"{len(chunk)}I", *chunk)
                layer_prefix = struct.pack("I", 0)
                key = hashlib.sha256(layer_prefix + raw).hexdigest()
                if key in store:
                    segment_keys.append(key)
        return segment_keys

    def _simulate_belady(
        self,
        dag: WorkflowDAG,
        children: Dict[str, List[str]],
    ) -> float:
        """Compute Bélády upper-bound hit rate via oracle simulation.

        Simulates a cache with perfect future knowledge (all accesses known).
        Each node accesses its own KV. The Bélády oracle always evicts the
        node whose next access is furthest in the future.
        """
        order = dag.topological_order
        if not order:
            return 0.0

        # Build future-access map: for each position, which nodes will be accessed next
        # A node is "accessed" when a child processes it (KV reuse)
        access_sequence: List[str] = []
        for nid in order:
            access_sequence.append(nid)
            # Each child of nid will re-access nid's KV
            for _ in children.get(nid, []):
                access_sequence.append(nid)

        if len(access_sequence) <= 1:
            return 0.0

        # Simulate with a cache of size = ceil(len(nodes) / 2)
        cache_size = max(1, len(dag.nodes) // 2)
        cached: Set[str] = set()
        hits = 0
        total = 0

        for pos, nid in enumerate(access_sequence):
            total += 1
            if nid in cached:
                hits += 1
            else:
                if len(cached) >= cache_size:
                    # Bélády: evict the item whose next use is furthest away
                    furthest_key = None
                    furthest_pos = -1
                    for c in cached:
                        next_use = len(access_sequence)  # infinity default
                        for future_pos in range(pos + 1, len(access_sequence)):
                            if access_sequence[future_pos] == c:
                                next_use = future_pos
                                break
                        if next_use > furthest_pos:
                            furthest_pos = next_use
                            furthest_key = c
                    if furthest_key is not None:
                        cached.discard(furthest_key)
                cached.add(nid)

        return hits / total if total > 0 else 0.0
