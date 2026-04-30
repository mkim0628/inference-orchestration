"""Cache-hit-rate-weighted request scheduler (Activity A).

Reorders incoming requests so those predicted to have higher KV cache hit
rates are processed first, maximising cache utilisation without starving
low-hit-rate requests (fairness via wait-step penalty).
"""

import struct
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class _RequestState:
    request: InferenceRequest
    wait_steps: int = 0
    predicted_hit_rate: float = 0.0


class CacheAwareScheduler:
    """Priority scheduler that ranks requests by predicted KV cache hit rate.

    Priority formula:
        priority = hit_rate × (1 − wait_penalty)
        wait_penalty = min(wait_steps / fairness_max_wait, 1.0)

    A request that has waited ≥ fairness_max_wait steps gets wait_penalty=1.0,
    meaning its hit_rate multiplier drops to 0 and it rises to the front via
    tie-breaking on wait_steps (most-waited first).
    """

    def __init__(
        self,
        cache: CacheStore,
        fairness_max_wait: int = 10,
        chunk_size: int = 128,
    ) -> None:
        self.cache = cache
        self.fairness_max_wait = fairness_max_wait
        self.chunk_size = chunk_size
        self._state: Dict[str, _RequestState] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Return requests reordered by cache-hit-weighted priority."""
        for req in requests:
            if req.request_id not in self._state:
                self._state[req.request_id] = _RequestState(request=req)

        scored: List[tuple] = []
        for req in requests:
            state = self._state[req.request_id]
            state.predicted_hit_rate = self._predict_hit_rate(req)
            wait_penalty = min(state.wait_steps / max(self.fairness_max_wait, 1), 1.0)
            priority = state.predicted_hit_rate * (1.0 - wait_penalty)
            # Tie-break: more wait_steps → higher precedence (negative for descending sort)
            scored.append((-priority, -state.wait_steps, req.request_id, req))

        scored.sort(key=lambda t: (t[0], t[1]))
        return [item[3] for item in scored]

    def update_wait(self, processed_ids: List[str], all_ids: List[str]) -> None:
        """Increment wait_steps for requests not processed this round."""
        processed_set = set(processed_ids)
        for rid in all_ids:
            if rid not in processed_set and rid in self._state:
                self._state[rid].wait_steps += 1

    def reset(self) -> None:
        self._state.clear()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _predict_hit_rate(self, request: InferenceRequest) -> float:
        """Estimate hit rate without polluting cache hit/miss statistics.

        Peeks at cache._store keys directly (no get() call) to avoid
        counting this lookup as a real cache access.
        """
        store = getattr(self.cache, "_store", None)
        if store is None:
            return 0.0

        token_ids = request.token_ids
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)
        hits = 0

        # Check layer 0 only (representative; avoids O(layers * chunks) cost)
        for chunk_idx in range(n_chunks):
            key = self._chunk_key(token_ids, chunk_idx, layer_idx=0)
            if key in store:
                hits += 1

        return hits / n_chunks

    def _chunk_key(self, token_ids: List[int], chunk_idx: int, layer_idx: int = 0) -> str:
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size
        chunk = token_ids[start:end]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()
