"""PPDAppendFullPrefillClassifier — Activity A: KV Cache-aware Scheduling.

PPD (arXiv 2603.13358) based append/full-prefill classifier with SLO-aware routing.
Classifies requests using O(1) session context hash comparison.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PrefillTypeDecision:
    request_id: str
    session_id: str
    turn: int
    prefill_type: str            # "append" | "full"
    new_token_ratio: float       # fraction of new tokens in current request
    routed_to: str               # "D_node" | "P_node"
    classifier_overhead_us: float  # classification overhead in microseconds


@dataclass
class SessionContextEntry:
    last_context_hash: str
    last_total_tokens: int
    turn_count: int
    last_accessed: float


@dataclass
class PPDClassifierConfig:
    append_threshold: float = 0.15          # new_token_ratio <= this → append-prefill
    slo_headroom_threshold_ms: float = 30.0  # SLO headroom below this → force P-node offload
    session_ttl_seconds: float = 3600.0      # session expiry TTL
    seed: int = 42


class PPDAppendFullPrefillClassifier:
    """PPD (arXiv 2603.13358) based append/full-prefill classifier + SLO-aware routing.

    Activity A: KV Cache-aware Scheduling.
    Scheduling granularity: per-request.
    Cache state access: PrefillTypeRegistry (session_id → last context hash) O(1) dict lookup.

    Difference from PPDAppendPrefillRouter (ppd_append_prefill_router.py):
      PPDAppendPrefillRouter: hit_probability estimation via TriangleInequalitySegmentIndex (O(log N)).
      PPDAppendFullPrefillClassifier: direct session context hash comparison (O(1)).
                                       Explicit new_token_ratio calculation for classification rationale.
                                       SLO-aware D→P forced switch logic.

    Classification logic:
      append-prefill conditions:
        1. turn_count >= 2 (after first turn)
        2. new_token_ratio = (total_tokens - last_total_tokens) / total_tokens
           new_token_ratio <= append_threshold (default 0.15: <= 15% new tokens)
        → Reuse prior-turn KV at D-node; process new tokens locally.

      full-prefill conditions:
        1. turn_count == 1 (first turn, KV cache cold start)
        2. new_token_ratio > append_threshold
        3. SLO headroom < slo_headroom_threshold_ms (offload to P-node under D-node pressure)
        → Full prefill at P-node.

    Evaluation targets (evaluation_criteria.md §2):
      - Scheduling overhead TTFT p50 +5% within limit (MANDATORY)
      - Cache hit rate improvement: +10%p vs no scheduling (high priority)
    """

    def __init__(self, config: PPDClassifierConfig) -> None:
        self.config = config
        # PrefillTypeRegistry: session_id → SessionContextEntry
        self._registry: Dict[str, SessionContextEntry] = {}

    @staticmethod
    def _context_hash(token_ids: List[int]) -> str:
        """Deterministic hash of token ID sequence for context identity comparison."""
        if not token_ids:
            return ""
        # Use first 512 tokens to bound hash cost while capturing context identity
        raw = b"".join(t.to_bytes(4, "little") for t in token_ids[:512])
        return hashlib.sha256(raw).hexdigest()[:16]

    def classify(
        self,
        request_id: str,
        session_id: str,
        token_ids: List[int],
        remaining_slo_ms: Optional[float] = None,
    ) -> PrefillTypeDecision:
        """Classify request as append-prefill or full-prefill and route accordingly.

        Algorithm (O(1) hash lookup):
          t_start = time.monotonic()
          entry = _registry.get(session_id)  # O(1)
          total_tokens = len(token_ids)
          ctx_hash = _context_hash(token_ids)

          if entry is None or entry.turn_count == 0:
            prefill_type = "full"  # cold start
          else:
            new_tokens = total_tokens - entry.last_total_tokens
            new_token_ratio = max(0.0, new_tokens / max(1, total_tokens))
            if new_token_ratio <= append_threshold:
              if remaining_slo_ms is not None and remaining_slo_ms < slo_headroom_threshold_ms:
                prefill_type = "full"  # SLO pressure → offload to P-node
              else:
                prefill_type = "append"
            else:
              prefill_type = "full"

          Update registry and return PrefillTypeDecision.

        Args:
            request_id: unique request identifier.
            session_id: session identifier for multi-turn context tracking.
            token_ids: current request token ID sequence.
            remaining_slo_ms: remaining SLO headroom in ms; None means no SLO constraint.

        Returns:
            PrefillTypeDecision with routing decision and overhead measurement.
        """
        t_start = time.monotonic()
        total_tokens = len(token_ids)
        ctx_hash = self._context_hash(token_ids)
        entry = self._registry.get(session_id)

        if entry is None or entry.turn_count == 0:
            prefill_type = "full"
            new_token_ratio = 1.0
        else:
            new_tokens = max(0, total_tokens - entry.last_total_tokens)
            new_token_ratio = new_tokens / max(1, total_tokens)
            if new_token_ratio <= self.config.append_threshold:
                if (remaining_slo_ms is not None
                        and remaining_slo_ms < self.config.slo_headroom_threshold_ms):
                    # SLO pressure: force full-prefill on P-node
                    prefill_type = "full"
                else:
                    prefill_type = "append"
            else:
                prefill_type = "full"

        turn = (entry.turn_count if entry else 0) + 1
        self._registry[session_id] = SessionContextEntry(
            last_context_hash=ctx_hash,
            last_total_tokens=total_tokens,
            turn_count=turn,
            last_accessed=time.monotonic(),
        )
        overhead_us = (time.monotonic() - t_start) * 1e6
        routed_to = "D_node" if prefill_type == "append" else "P_node"
        return PrefillTypeDecision(
            request_id=request_id,
            session_id=session_id,
            turn=turn,
            prefill_type=prefill_type,
            new_token_ratio=new_token_ratio,
            routed_to=routed_to,
            classifier_overhead_us=overhead_us,
        )

    def expire_sessions(self) -> int:
        """Remove TTL-expired sessions. Returns count of expired sessions."""
        now = time.monotonic()
        expired = [
            sid for sid, e in self._registry.items()
            if now - e.last_accessed > self.config.session_ttl_seconds
        ]
        for sid in expired:
            del self._registry[sid]
        return len(expired)

    def reset_session(self, session_id: str) -> None:
        """Remove session from registry."""
        self._registry.pop(session_id, None)

    def registry_size(self) -> int:
        """Return number of active sessions in registry."""
        return len(self._registry)
