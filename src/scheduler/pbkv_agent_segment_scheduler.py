"""PBKVAgentSegmentPreservationScheduler — PBKV-style predictive agentic KV scheduler (Activity A).

Inspired by PBKV (2605.06472): a lightweight MLP predicts the reuse probability of each
non-contiguous KV segment for the next N agent steps, driving a two-tier GPU/host
preservation policy with Lipschitz-continuity robustness via a preemption margin.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class PBKVConfig:
    segment_emb_dim: int = 256
    history_steps: int = 10
    prediction_horizon: int = 5
    gpu_preserve_threshold: float = 0.6
    host_evict_threshold: float = 0.3
    preemption_margin: float = 0.3
    fairness_max_wait: int = 10
    chunk_size: int = 128
    seed: int = 42


class _SegmentMLP(nn.Module):
    """Lightweight segment reuse probability predictor.

    Input:  segment embedding (d=segment_emb_dim) + recent call history (history_steps)
    Output: reuse probability scalar in [0, 1]
    """

    def __init__(self, segment_emb_dim: int, history_steps: int) -> None:
        super().__init__()
        input_dim = segment_emb_dim + history_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [batch, input_dim] → Output: [batch, 1]"""
        return self.net(x)


class PBKVAgentSegmentPreservationScheduler:
    """PBKV prediction-based agentic segment preservation scheduler (Activity A).

    Scheduling decisions operate at two levels:
      - Request level: reorder batch by predicted reuse probability × fairness weight
      - Segment level: GPU preserve vs host evict per chunk_key

    Cache state is accessed via cache._store key lookup (no get() calls — stats unpolluted).

    Difference from CacheAwareScheduler:
      CacheAwareScheduler: static priority based on current cache hit rate
      PBKVAgentSegmentPreservationScheduler: dynamic preservation policy from
        future N-step MLP prediction with Lipschitz robustness margin
    """

    def __init__(
        self,
        cache: CacheStore,
        config: Optional[PBKVConfig] = None,
    ) -> None:
        self.cache = cache
        self.config = config or PBKVConfig()
        torch.manual_seed(self.config.seed)
        self._predictor = _SegmentMLP(
            self.config.segment_emb_dim,
            self.config.history_steps,
        )
        # agent_id → list of recently accessed chunk_keys (capped at history_steps)
        self._agent_history: Dict[str, List[str]] = {}
        # chunk_key → {'gpu': bool, 'prob': float}
        self._preservation_map: Dict[str, Dict] = {}
        # request_id → wait steps accumulated
        self._wait_steps: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Scheduling API                                                       #
    # ------------------------------------------------------------------ #

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Reorder requests by predicted reuse probability × fairness weight.

        Priority formula:
          priority = predicted_reuse_prob × (1 − wait_penalty)
          wait_penalty = min(wait_steps / fairness_max_wait, 1.0)
        """
        scored: List[Tuple] = []
        for req in requests:
            prob = self._predict_segment_reuse(req)
            wait = self._wait_steps.get(req.request_id, 0)
            wait_penalty = min(wait / max(self.config.fairness_max_wait, 1), 1.0)
            priority = prob * (1.0 - wait_penalty)
            scored.append((-priority, -wait, req.request_id, req))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [item[3] for item in scored]

    def update_preservation_policy(
        self,
        processed_request_ids: List[str],
        all_request_ids: List[str],
    ) -> Tuple[Set[str], Set[str]]:
        """Update GPU preserve / host evict decisions for all cached segments.

        Returns:
          preserve_keys: Set[str] — chunk_keys to keep on GPU
          evict_keys: Set[str] — chunk_keys to evict to host memory

        Uses Lipschitz robustness margin: effective_threshold =
          gpu_preserve_threshold − preemption_margin to avoid premature eviction.
        """
        store = getattr(self.cache, "_store", None)
        if store is None:
            return set(), set()

        all_keys = list(store.keys())
        preserve_keys: Set[str] = set()
        evict_keys: Set[str] = set()

        effective_threshold = (
            self.config.gpu_preserve_threshold - self.config.preemption_margin
        )

        for key in all_keys:
            emb = self._get_segment_embedding(key)
            history_vec = self._get_history_vector()
            input_vec = torch.cat([emb, history_vec]).unsqueeze(0)
            with torch.no_grad():
                prob = self._predictor(input_vec).item()

            if prob >= effective_threshold:
                preserve_keys.add(key)
            elif prob < self.config.host_evict_threshold:
                evict_keys.add(key)

        self._preservation_map = {
            k: {"gpu": k in preserve_keys, "prob": 0.0} for k in all_keys
        }
        return preserve_keys, evict_keys

    def update_agent_history(
        self,
        agent_id: str,
        accessed_chunk_keys: List[str],
    ) -> None:
        """Update agent call history, keeping only the last history_steps entries."""
        history = self._agent_history.get(agent_id, [])
        history.extend(accessed_chunk_keys)
        self._agent_history[agent_id] = history[-self.config.history_steps:]

    def update_wait(
        self,
        processed_ids: List[str],
        all_ids: List[str],
    ) -> None:
        """Increment wait_steps for unprocessed requests this round."""
        processed_set = set(processed_ids)
        for rid in all_ids:
            if rid not in processed_set:
                self._wait_steps[rid] = self._wait_steps.get(rid, 0) + 1

    # ------------------------------------------------------------------ #
    # Internal utilities                                                   #
    # ------------------------------------------------------------------ #

    def _predict_segment_reuse(self, request: InferenceRequest) -> float:
        """Predict mean reuse probability across all chunks of a request."""
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(request.token_ids) + chunk_size - 1) // chunk_size)
        probs: List[float] = []
        for chunk_idx in range(n_chunks):
            key = self._chunk_key(request.token_ids, chunk_idx)
            emb = self._get_segment_embedding(key)
            hist = self._get_history_vector(request.request_id)
            inp = torch.cat([emb, hist]).unsqueeze(0)
            with torch.no_grad():
                prob = self._predictor(inp).item()
            probs.append(prob)
        return sum(probs) / len(probs) if probs else 0.0

    def _get_segment_embedding(self, chunk_key: str) -> torch.Tensor:
        """Produce a deterministic d=segment_emb_dim embedding from a chunk key hash."""
        # Hash the key string to a fixed-length byte sequence, then map to float vector
        h = hashlib.sha256(chunk_key.encode()).digest()  # 32 bytes
        # Repeat the hash bytes to fill segment_emb_dim floats
        dim = self.config.segment_emb_dim
        raw_bytes = (h * ((dim * 4 // 32) + 2))[: dim * 4]
        emb = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32).clone()[:dim]
        # Normalise to zero mean, unit variance to avoid exploding activations
        emb = (emb - emb.mean()) / (emb.std().clamp(min=1e-8))
        return emb

    def _get_history_vector(self, agent_or_request_id: str = "") -> torch.Tensor:
        """Build a history_steps-length vector from agent call history."""
        history = self._agent_history.get(agent_or_request_id, [])
        steps = self.config.history_steps
        vec = torch.zeros(steps)
        for i, key in enumerate(history[-steps:]):
            # Encode each historical key as a single float via hash
            h = hashlib.sha256(key.encode()).digest()
            val = struct.unpack("f", h[:4])[0]
            # Clamp to a safe range to prevent NaN in MLP
            vec[i] = max(-10.0, min(10.0, val))
        return vec

    def _chunk_key(self, token_ids: List[int], chunk_idx: int) -> str:
        """Generate a chunk key using the same method as SegmentedHashCache."""
        start = chunk_idx * self.config.chunk_size
        end = start + self.config.chunk_size
        chunk = token_ids[start:end]
        if not chunk:
            chunk = [0]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", 0)  # layer 0 as representative
        return hashlib.sha256(layer_prefix + raw).hexdigest()
