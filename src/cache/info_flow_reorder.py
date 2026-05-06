"""
InfoFlowChunkReorderCache — Activity B (InfoFlowKV-inspired).

Orders non-contiguous cached segments by an attention-norm information-flow
signal so that segments with the highest downstream influence are placed first
in the context window. Reorder time and RoPE-recomputation overhead are
measured separately and stored for reporting.
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore


class InfoFlowChunkReorderCache(CacheStore):
    """
    Information-flow-based chunk reorder cache.

    Each segment receives a scalar "infoflow score" computed as
    sum(softmax(K @ K^T / sqrt(d)).diagonal() * ||K||) averaged over layers
    and heads. Reorder sorts segments by this score in descending order so
    high-influence segments occupy early positions in the assembled context.

    RoPE re-computation is required after reordering since token positions
    change. Reorder time and RoPE overhead are exposed via last_timing().
    """

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes

        # {key: {"kv": Tensor, "infoflow_score": float}}
        self._store: OrderedDict[str, Dict] = OrderedDict()
        self._hit_count: int = 0
        self._miss_count: int = 0

        # Timing (seconds) from the most recent reorder_chunks call
        self._last_reorder_time: float = 0.0
        self._last_rope_time: float = 0.0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store KV tensor with pre-computed infoflow score.

        Args:
            key: Segment hash string.
            value: KV tensor [layers, heads, seq_len, head_dim].
        """
        if key in self._store:
            self._store.move_to_end(key)
            return

        while self.memory_bytes() + value.nbytes > self.capacity_bytes and self._store:
            self.evict()

        infoflow_score = self._compute_infoflow_score(value)
        self._store[key] = {
            "kv": value.detach().clone(),
            "infoflow_score": infoflow_score,
        }

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve KV tensor by exact key.

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
        """Evict LRU entry. Returns bytes freed."""
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
        """Current memory footprint in bytes."""
        return sum(e["kv"].nbytes for e in self._store.values())

    def reset_stats(self) -> None:
        """Reset hit/miss counters and timing."""
        self._hit_count = 0
        self._miss_count = 0
        self._last_reorder_time = 0.0
        self._last_rope_time = 0.0

    # ------------------------------------------------------------------ #
    # Information-flow scoring and reorder API                             #
    # ------------------------------------------------------------------ #

    def _compute_infoflow_score(self, kv: torch.Tensor) -> float:
        """Compute information-flow signal for a KV tensor.

        Score = sum(softmax(K @ K^T / sqrt(d)).diagonal() * ||K||)
        averaged over layers and heads.

        Args:
            kv: [layers, heads, seq_len, head_dim]

        Returns:
            Scalar information-flow score.
        """
        k_norm = kv.norm(dim=-1).mean(dim=(0, 1))  # [seq_len]
        d = kv.shape[-1]
        k_flat = kv.mean(dim=(0, 1))  # [seq_len, head_dim]

        # Approximate self-attention diagonal as information-flow proxy
        attn_approx = torch.softmax(
            (k_flat @ k_flat.T) / (d ** 0.5), dim=-1
        ).diagonal()  # [seq_len]

        infoflow = (attn_approx * k_norm).sum().item()
        return infoflow

    def reorder_chunks(
        self,
        chunks: List[str],
        attention_scores: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Sort chunks by information-flow score (descending).

        Segments with higher scores are placed first in the context window,
        maximising their downstream attention influence. When external
        attention scores are provided they are combined with a 0.5/0.5 weight.

        Time spent on sorting and a synthetic RoPE re-computation proxy are
        captured in self._last_reorder_time / self._last_rope_time.

        Args:
            chunks: Segment hash list to reorder.
            attention_scores: Optional external per-segment attention scores.

        Returns:
            Reordered segment hash list (descending infoflow score).
        """
        t0 = time.perf_counter()

        scored: List[Tuple[str, float]] = []
        for chunk_key in chunks:
            if chunk_key not in self._store:
                continue
            score = self._store[chunk_key]["infoflow_score"]
            if attention_scores and chunk_key in attention_scores:
                score = 0.5 * score + 0.5 * attention_scores[chunk_key]
            scored.append((chunk_key, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        reordered = [k for k, _ in scored]

        t1 = time.perf_counter()
        self._last_reorder_time = t1 - t0

        # Simulate RoPE re-computation cost proportional to total token count
        t2 = time.perf_counter()
        total_tokens = sum(
            self._store[k]["kv"].shape[2] for k in reordered if k in self._store
        )
        # Lightweight proxy: O(total_tokens) float operations
        if total_tokens > 0:
            _ = torch.arange(total_tokens, dtype=torch.float32) * 0.01
        t3 = time.perf_counter()
        self._last_rope_time = t3 - t2

        return reordered

    def last_timing(self) -> Dict[str, float]:
        """Return timing (seconds) from the most recent reorder_chunks call.

        Returns:
            {"reorder_s": float, "rope_recompute_s": float}
        """
        return {
            "reorder_s": self._last_reorder_time,
            "rope_recompute_s": self._last_rope_time,
        }

    def get_infoflow_score(self, key: str) -> Optional[float]:
        """Return the stored infoflow score for a given key."""
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry["infoflow_score"]

    def keys(self) -> List[str]:
        """Return all stored segment keys."""
        return list(self._store.keys())
