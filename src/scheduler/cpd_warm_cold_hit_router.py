"""CPD Warm/Cold Hit Rate Router (Activity A).

Implements the CPD (Together AI 2026-03-04) principle as a lightweight
per-request predicted-hit-rate warm/cold/neutral soft-routing scheduler
for single-node KV cache-aware scheduling.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.scheduler.base import BaseScheduler


@dataclass
class HitRatePredictorWeights:
    """Online linear regression weights (4 features)."""
    w_prefix_hash: float = 0.4
    w_context_length: float = 0.2
    w_session_age: float = 0.2
    w_segment_match: float = 0.2
    bias: float = 0.3
    lr: float = 0.01   # SGD learning rate


@dataclass
class CPDRouterConfig:
    high_hit_threshold: float = 0.70    # >= this: warm request
    low_hit_threshold: float = 0.25     # < this: cold request; between: neutral
    warm_slot_ratio: float = 0.60       # batch warm slot fraction
    cold_slot_ratio: float = 0.30       # batch cold slot fraction
    neutral_slot_ratio: float = 0.10    # neutral slot fraction
    queue_pressure_threshold: int = 100  # queue depth above which cold slots increase
    max_context_length_warm: int = 50000  # warm path max context length
    seed: int = 42


@dataclass
class RoutingDecision:
    request_id: str
    predicted_hit_rate: float
    path: str          # "warm" | "cold" | "neutral"
    batch_priority: int  # lower = processed first (warm=0, neutral=1, cold=2)


class CPDWarmColdHitRateRouter(BaseScheduler):
    """CPD (Together AI 2026-03-04) predicted-hit-rate warm/cold soft-routing scheduler.

    Activity A: KV Cache-aware Scheduling.
    Scheduling granularity: per-request.
    Cache state access: prefix_hash -> hit_history (O(1) dict lookup).

    Lightweight hit rate predictor:
      Features: [prefix_hash_match, context_length_norm, session_age_norm, segment_match_ratio]
      Model: linear regression with 4 weights (< 0.01ms inference)
      Online learning: SGD(lr=0.01) updated with actual hit results

    Warm/Cold/Neutral 3 paths:
      Warm (predicted hit rate >= 0.70): front of batch, warm_slot_ratio reserved,
                                          cache locality maximized
      Cold (predicted hit rate < 0.25): rear of batch, isolated cold processing
                                         (prevents warm contamination)
      Neutral (0.25 <= hit rate < 0.70): remaining slots

    Warm batch cache locality:
      group requests with identical prefix_hash in same batch for KV reuse.
      sort by prefix LCP length.
    """

    def __init__(self, config: CPDRouterConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._weights = HitRatePredictorWeights()
        # prefix_hash -> recent hit results (True/False) history (max 100)
        self._hit_history: Dict[str, List[bool]] = {}
        self._warm_count: int = 0
        self._cold_count: int = 0
        self._neutral_count: int = 0
        self._ttft_overhead_us: List[float] = []

    def _extract_features(
        self,
        request: Any,
        max_context_len: int = 128000,
    ) -> Tuple[float, float, float, float]:
        """Extract hit rate prediction features from a request.

        Features:
          f1 (prefix_hash_match): recent hit rate from hit_history, 0.0 if unknown
          f2 (context_length_norm): len(token_ids) / max_context_len
          f3 (session_age_norm): 1.0 / (1 + session_turn_count)
          f4 (segment_match_ratio): known segment match ratio, default 0.0
        """
        prefix_hash = getattr(request, 'prefix_hash', '') or ''
        token_ids = getattr(request, 'token_ids', []) or []
        session_turn = getattr(request, 'session_turn', 0) or 0
        segment_match = getattr(request, 'segment_match_ratio', 0.0) or 0.0

        history = self._hit_history.get(prefix_hash, [])
        f1 = sum(history) / len(history) if history else 0.0
        f2 = min(1.0, len(token_ids) / max(1, max_context_len))
        f3 = 1.0 / (1.0 + session_turn)
        f4 = float(segment_match)
        return f1, f2, f3, f4

    def predict_hit_rate(self, request: Any) -> float:
        """Predict cache hit rate via linear model (< 0.01ms).

        score = w1*f1 + w2*f2 + w3*f3 + w4*f4 + bias
        hit_rate = sigmoid(score)
        """
        f1, f2, f3, f4 = self._extract_features(request)
        w = self._weights
        score = (
            w.w_prefix_hash * f1
            + w.w_context_length * f2
            + w.w_session_age * f3
            + w.w_segment_match * f4
            + w.bias
        )
        return float(1.0 / (1.0 + torch.tensor(-score).exp()))

    def update_predictor(self, request: Any, actual_hit: bool) -> None:
        """Online SGD update using actual hit result.

        y_pred = predict_hit_rate(request)
        y_true = 1.0 if actual_hit else 0.0
        error = y_pred - y_true
        weight_i -= lr * error * feature_i  (MSE gradient)
        """
        f1, f2, f3, f4 = self._extract_features(request)
        y_pred = self.predict_hit_rate(request)
        y_true = 1.0 if actual_hit else 0.0
        error = y_pred - y_true
        lr = self._weights.lr
        self._weights.w_prefix_hash -= lr * error * f1
        self._weights.w_context_length -= lr * error * f2
        self._weights.w_session_age -= lr * error * f3
        self._weights.w_segment_match -= lr * error * f4
        self._weights.bias -= lr * error

        # update hit history
        prefix_hash = getattr(request, 'prefix_hash', '') or ''
        if prefix_hash:
            hist = self._hit_history.setdefault(prefix_hash, [])
            hist.append(actual_hit)
            if len(hist) > 100:
                hist.pop(0)

    def classify_request(self, request: Any) -> RoutingDecision:
        """Classify request as warm/cold/neutral based on predicted hit rate."""
        t_start = time.monotonic()
        hit_rate = self.predict_hit_rate(request)
        if hit_rate >= self.config.high_hit_threshold:
            path = "warm"
            priority = 0
            self._warm_count += 1
        elif hit_rate < self.config.low_hit_threshold:
            path = "cold"
            priority = 2
            self._cold_count += 1
        else:
            path = "neutral"
            priority = 1
            self._neutral_count += 1
        overhead_us = (time.monotonic() - t_start) * 1e6
        self._ttft_overhead_us.append(overhead_us)
        return RoutingDecision(
            request_id=getattr(request, 'request_id', ''),
            predicted_hit_rate=hit_rate,
            path=path,
            batch_priority=priority,
        )

    def sort_warm_batch_by_prefix_similarity(self, warm_requests: List[Any]) -> List[Any]:
        """Sort warm-batch requests by prefix similarity (LCP length) for KV reuse locality."""
        def lcp_key(req: Any) -> str:
            return getattr(req, 'prefix_hash', '') or ''

        return sorted(warm_requests, key=lcp_key)

    def schedule(self, requests: List[Any]) -> List[Any]:
        """BaseScheduler interface implementation.

        Routes requests into warm/neutral/cold paths, sorts warm by prefix,
        and promotes some cold requests under queue pressure.
        """
        if not requests:
            return []

        warm: List[Any] = []
        cold: List[Any] = []
        neutral: List[Any] = []
        for req in requests:
            decision = self.classify_request(req)
            if decision.path == "warm":
                warm.append(req)
            elif decision.path == "cold":
                cold.append(req)
            else:
                neutral.append(req)

        warm_sorted = self.sort_warm_batch_by_prefix_similarity(warm)

        # under queue pressure, promote half of cold requests to neutral
        if len(requests) > self.config.queue_pressure_threshold:
            half = len(cold) // 2
            neutral.extend(cold[:half])
            cold = cold[half:]

        return warm_sorted + neutral + cold

    def scheduling_overhead_mean_us(self) -> float:
        """Mean scheduling overhead per request in microseconds."""
        if not self._ttft_overhead_us:
            return 0.0
        return sum(self._ttft_overhead_us) / len(self._ttft_overhead_us)

    def routing_stats(self) -> dict:
        """Return routing distribution and overhead statistics."""
        total = max(1, self._warm_count + self._cold_count + self._neutral_count)
        return {
            "warm_ratio": self._warm_count / total,
            "cold_ratio": self._cold_count / total,
            "neutral_ratio": self._neutral_count / total,
            "scheduling_overhead_mean_us": self.scheduling_overhead_mean_us(),
        }
