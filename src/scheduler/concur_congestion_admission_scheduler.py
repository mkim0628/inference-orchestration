"""CONCUR-based KV pool congestion admission scheduler.

Activity A: KV Cache-aware Scheduling.
Implements runtime KV pool occupancy monitoring (3-state congestion control)
and an admission gate that prevents middle-phase thrashing without preempting
in-flight agents. Based on the CONCUR paper.
"""

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal, Optional

from src.scheduler.base import BaseScheduler


CongestionLevel = Literal["FREE", "BOUNDARY", "CONGESTED"]


@dataclass
class KVPoolMonitor:
    """KV pool occupancy real-time monitor.

    Cache state access: get_occupancy() O(1).
    """

    capacity_bytes: int
    alpha_low: float = 0.60
    alpha_high: float = 0.85
    _current_bytes: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def update(self, used_bytes: int) -> None:
        with self._lock:
            self._current_bytes = used_bytes

    def get_occupancy(self) -> float:
        """Current KV pool occupancy ratio (0.0–1.0). O(1)."""
        if self.capacity_bytes == 0:
            return 0.0
        return self._current_bytes / self.capacity_bytes

    def congestion_level(self) -> CongestionLevel:
        occ = self.get_occupancy()
        if occ >= self.alpha_high:
            return "CONGESTED"
        elif occ >= self.alpha_low:
            return "BOUNDARY"
        else:
            return "FREE"


@dataclass
class CongestionAdmissionConfig:
    capacity_bytes: int = 1_000_000_000
    alpha_low: float = 0.60
    alpha_high: float = 0.85
    priority_weights: Dict[str, float] = field(default_factory=dict)
    online_adapt_window: int = 100
    enable_multinode: bool = False
    seed: int = 42


class CONCURCongestionBasedAgentAdmissionScheduler(BaseScheduler):
    """CONCUR-based KV pool congestion admission scheduler.

    Activity A: KV Cache-aware Scheduling.
    Scheduling decision unit: agent step.
    Cache state access: KVPoolMonitor.get_occupancy() O(1).

    3-state congestion control:
      FREE (occupancy < alpha_low): standard FIFO + priority resume. Admit all.
      BOUNDARY (alpha_low <= occupancy < alpha_high): admit high-priority only.
      CONGESTED (occupancy >= alpha_high): suspend new admissions.
        In-flight agents' KVs are NOT preemptively evicted (CONCUR core principle).

    Online threshold adaptation:
      Every online_adapt_window steps, compute wait_ratio and adjust alpha_high
      by ±0.02, clamped to [0.70, 0.95]. alpha_low = alpha_high - 0.25.

    Multi-node support:
      enable_multinode=True: aggregate remote node occupancies via
      update_remote_occupancy() to form a global congestion signal.
    """

    def __init__(self, config: CongestionAdmissionConfig) -> None:
        self.config = config
        self.monitor = KVPoolMonitor(
            capacity_bytes=config.capacity_bytes,
            alpha_low=config.alpha_low,
            alpha_high=config.alpha_high,
        )
        self._wait_queue: Deque = deque()
        self._admitted: List = []
        self._scheduling_times: List[float] = []
        self._step_count: int = 0
        self._gate_throughput: List[float] = []
        self._remote_occupancies: Dict[str, float] = {}
        # Tracks admissions in the current adapt window for threshold tuning
        self._window_admitted_count: int = 0

    def admit(self, agent_step: object, priority: float = 1.0) -> bool:
        """Attempt to admit an agent step.

        Algorithm:
          1. Query monitor.congestion_level().
          2. CONGESTED → enqueue, return False.
          3. BOUNDARY → admit only if priority >= median of configured weights.
          4. FREE → admit unconditionally.
          Returns True (admitted) / False (queued).
        """
        level = self.monitor.congestion_level()
        if level == "CONGESTED":
            self._wait_queue.append((agent_step, priority))
            return False
        elif level == "BOUNDARY":
            weights = list(self.config.priority_weights.values()) or [1.0]
            median_priority = statistics.median(weights)
            if priority >= median_priority:
                self._admitted.append(agent_step)
                self._window_admitted_count += 1
                self._step_count += 1
                return True
            else:
                self._wait_queue.append((agent_step, priority))
                return False
        else:  # FREE
            self._admitted.append(agent_step)
            self._window_admitted_count += 1
            self._step_count += 1
            return True

    def release(self, agent_step: object, freed_bytes: int) -> None:
        """Signal agent completion, update pool, and resume queued agents if possible.

        Algorithm:
          1. Update monitor with reduced byte count.
          2. Re-check congestion level.
          3. If FREE: drain wait_queue by priority order.
        """
        new_used = max(0, self.monitor._current_bytes - freed_bytes)
        self.monitor.update(new_used)
        if agent_step in self._admitted:
            self._admitted.remove(agent_step)

        if self.monitor.congestion_level() == "FREE":
            # Re-admit queued steps sorted by priority descending
            queued = sorted(self._wait_queue, key=lambda x: x[1], reverse=True)
            self._wait_queue.clear()
            for step, prio in queued:
                self._admitted.append(step)
                self._window_admitted_count += 1

    def update_kv_pool(self, used_bytes: int) -> None:
        """Update KV pool state (call every step)."""
        self.monitor.update(used_bytes)

    def update_remote_occupancy(self, node_id: str, occupancy: float) -> None:
        """Multi-node: aggregate remote node occupancy."""
        self._remote_occupancies[node_id] = occupancy

    def global_occupancy(self) -> float:
        """Average of local + remote occupancies (multi-node congestion signal)."""
        all_occ = [self.monitor.get_occupancy()] + list(self._remote_occupancies.values())
        return sum(all_occ) / len(all_occ)

    def _adapt_thresholds(self) -> None:
        """Online threshold adaptation.

        Computes wait_ratio over the current window and adjusts alpha_high ±0.02.
        alpha_high is clamped to [0.70, 0.95]; alpha_low = alpha_high - 0.25.
        """
        admitted_in_window = max(1, self._window_admitted_count)
        wait_ratio = len(self._wait_queue) / admitted_in_window

        if wait_ratio > 0.5:
            # High queue pressure: tighten admission gate
            self.monitor.alpha_high = max(0.70, self.monitor.alpha_high - 0.02)
        elif wait_ratio < 0.1:
            # Low queue pressure: relax admission gate
            self.monitor.alpha_high = min(0.95, self.monitor.alpha_high + 0.02)

        self.monitor.alpha_low = self.monitor.alpha_high - 0.25
        self._window_admitted_count = 0

    def schedule(self, requests: List) -> List:
        """BaseScheduler-compatible batch scheduling interface.

        Called once per batch. Uses current congestion level to filter requests.
        Returns: admitted subset of requests.
        """
        t0 = time.monotonic()
        level = self.monitor.congestion_level()

        if level == "CONGESTED":
            result: List = []
        elif level == "BOUNDARY":
            # Admit top half by priority weight
            sorted_reqs = sorted(
                requests,
                key=lambda r: self.config.priority_weights.get(
                    getattr(r, "request_id", ""), 1.0
                ),
                reverse=True,
            )
            result = sorted_reqs[: max(1, len(sorted_reqs) // 2)]
        else:
            result = list(requests)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._scheduling_times.append(elapsed_ms)
        self._step_count += 1

        if self._step_count % self.config.online_adapt_window == 0:
            self._adapt_thresholds()

        return result

    def scheduling_overhead_ms_p50(self) -> float:
        """Median scheduling overhead in ms. Evaluation criterion: < 5ms (MANDATORY)."""
        if not self._scheduling_times:
            return 0.0
        s = sorted(self._scheduling_times)
        return s[len(s) // 2]

    def reset_stats(self) -> None:
        self._scheduling_times.clear()
        self._step_count = 0
        self._gate_throughput.clear()
        self._remote_occupancies.clear()
        self._window_admitted_count = 0
        self._wait_queue.clear()
        self._admitted.clear()
