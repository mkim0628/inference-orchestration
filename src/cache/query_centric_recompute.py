"""
QueryCentricRecomputeCache — Activity B (ProphetKV-based).

Two-stage recompute budget allocation driven by external query relevance,
not intra-segment deviation. This is the key departure from DHD/Semantic
segment caches: the budget arbiter moves from segment-internal signals to
query-centric cosine similarity.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


class QueryCentricRecomputeCache(CacheStore):
    """
    ProphetKV-inspired dual-stage recompute budget allocation.

    Stage 1: global saliency filter — retain top-50% by attention norm.
    Stage 2: query-relevance re-rank — cosine similarity with query embedding.

    Difference from DHD/SemanticSegmentCache:
    - DHD: recompute decision based on intra-segment attention score variance.
    - QCRC: recompute budget allocated by external query-segment cosine similarity.
    """

    def __init__(
        self,
        capacity_bytes: int,
        recompute_budget_ratio: float = 0.20,
        stage1_top_k_ratio: float = 0.50,
    ) -> None:
        if not (0.0 < recompute_budget_ratio <= 1.0):
            raise ValueError("recompute_budget_ratio must be in (0, 1]")
        if not (0.0 < stage1_top_k_ratio <= 1.0):
            raise ValueError("stage1_top_k_ratio must be in (0, 1]")

        self.capacity_bytes = capacity_bytes
        self.recompute_budget_ratio = recompute_budget_ratio
        self.stage1_top_k_ratio = stage1_top_k_ratio

        # {key: {"kv": Tensor, "embedding": Tensor, "attn_norm": float}}
        self._store: OrderedDict[str, Dict] = OrderedDict()
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store a KV segment.

        Args:
            key: Segment hash string.
            value: KV tensor shaped [layers, heads, seq_len, head_dim].
        """
        if key in self._store:
            self._store.move_to_end(key)
            return

        # Evict if over capacity
        while self.memory_bytes() + value.nbytes > self.capacity_bytes and self._store:
            self.evict()

        # Segment embedding: mean K vector over layers, heads, seq positions
        embedding = value.mean(dim=(0, 1, 2))  # [head_dim]

        # Attention norm approximation: mean of K-vector norms across positions
        # (proxy for how "active" / salient this segment's keys are)
        attn_norm = value.norm(dim=-1).mean().item()

        self._store[key] = {
            "kv": value.detach().clone(),
            "embedding": embedding.detach().clone(),
            "attn_norm": attn_norm,
        }

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve KV tensor by exact hash. Returns None on miss.

        Args:
            key: Segment hash string.
        """
        entry = self._store.get(key)
        if entry is None:
            self._miss_count += 1
            return None
        self._store.move_to_end(key)
        self._hit_count += 1
        return entry["kv"]

    def evict(self) -> int:
        """Evict the least-recently-used entry (LRU). Returns bytes freed."""
        if not self._store:
            return 0
        key, entry = next(iter(self._store.items()))
        self._store.pop(key)
        return entry["kv"].nbytes

    def hit_rate(self) -> float:
        """Cumulative cache hit rate."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Current KV memory usage in bytes."""
        return sum(e["kv"].nbytes for e in self._store.values())

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------ #
    # Query-centric dual-stage API                                         #
    # ------------------------------------------------------------------ #

    def selective_recompute(
        self,
        query: torch.Tensor,
        cached_segments: List[str],
        budget: float = 0.20,
    ) -> List[str]:
        """Two-stage recompute budget allocation.

        Stage 1 filters by global attention-norm saliency (top stage1_top_k_ratio).
        Stage 2 re-ranks by cosine similarity with the query and enforces the
        token-count budget.

        Args:
            query: Query representation vector [head_dim].
            cached_segments: Candidate segment hash list.
            budget: Max fraction of total cached tokens to recompute.

        Returns:
            List of segment hashes selected for recomputation (ordered by
            descending relevance, within budget).
        """
        # Keep only segments present in store
        present = [k for k in cached_segments if k in self._store]
        if not present:
            return []

        # --- Stage 1: global saliency filter (attention norm top-50%) ---
        attn_norms = {k: self._store[k]["attn_norm"] for k in present}
        sorted_by_norm = sorted(attn_norms, key=lambda k: attn_norms[k], reverse=True)
        n_stage1 = max(1, int(len(sorted_by_norm) * self.stage1_top_k_ratio))
        stage1_candidates = sorted_by_norm[:n_stage1]

        # --- Stage 2: query-relevance re-rank (cosine similarity) ---
        relevance_scores: Dict[str, float] = {}
        for seg_key in stage1_candidates:
            seg_emb = self._store[seg_key]["embedding"]  # [head_dim]
            score = F.cosine_similarity(
                query.unsqueeze(0).float(), seg_emb.unsqueeze(0).float()
            ).item()
            relevance_scores[seg_key] = score

        sorted_by_relevance = sorted(
            relevance_scores, key=lambda k: relevance_scores[k], reverse=True
        )

        # Enforce token-count budget
        total_tokens = sum(
            self._store[k]["kv"].shape[2]
            for k in sorted_by_relevance
        )
        token_budget = max(1, int(total_tokens * budget))
        selected: List[str] = []
        accumulated = 0
        for seg_key in sorted_by_relevance:
            seg_len = self._store[seg_key]["kv"].shape[2]
            if accumulated + seg_len > token_budget:
                break
            selected.append(seg_key)
            accumulated += seg_len

        return selected

    def get_embedding(self, key: str) -> Optional[torch.Tensor]:
        """Return the stored segment embedding for a given key."""
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry["embedding"]

    def get_attn_norm(self, key: str) -> Optional[float]:
        """Return the stored attention norm for a given key."""
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry["attn_norm"]

    def keys(self) -> List[str]:
        """Return all stored segment keys."""
        return list(self._store.keys())
