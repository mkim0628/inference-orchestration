from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass
class WeightedHitRateMetrics:
    """Weighted hit rate metrics including soft hit path. Independent of HitRateMetrics."""
    total_requests: int = 0
    total_chunks: int = 0
    hard_hit_chunks: int = 0
    soft_hit_chunks: int = 0
    noncontiguous_hard_hits: int = 0
    noncontiguous_soft_hits: int = 0
    beta_weight: float = 0.5

    def record(
        self,
        n_hard_hits: int,
        n_soft_hits: int,
        n_misses: int,
        noncontiguous_hard: int = 0,
        noncontiguous_soft: int = 0,
    ) -> None:
        self.total_requests += 1
        self.total_chunks += n_hard_hits + n_soft_hits + n_misses
        self.hard_hit_chunks += n_hard_hits
        self.soft_hit_chunks += n_soft_hits
        self.noncontiguous_hard_hits += noncontiguous_hard
        self.noncontiguous_soft_hits += noncontiguous_soft

    def hard_hit_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.hard_hit_chunks / self.total_chunks

    def soft_hit_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.soft_hit_chunks / self.total_chunks

    def weighted_hit_rate(self) -> float:
        """(n_hard + beta_weight * n_soft) / total"""
        if self.total_chunks == 0:
            return 0.0
        return (
            self.hard_hit_chunks + self.beta_weight * self.soft_hit_chunks
        ) / self.total_chunks

    def noncontiguous_weighted_fraction(self) -> float:
        """Non-contiguous (hard + soft weighted) / total (hard + soft weighted)"""
        weighted_total = self.hard_hit_chunks + self.beta_weight * self.soft_hit_chunks
        if weighted_total == 0:
            return 0.0
        nc_weighted = (
            self.noncontiguous_hard_hits + self.beta_weight * self.noncontiguous_soft_hits
        )
        return nc_weighted / weighted_total

    def reset(self) -> None:
        self.total_requests = 0
        self.total_chunks = 0
        self.hard_hit_chunks = 0
        self.soft_hit_chunks = 0
        self.noncontiguous_hard_hits = 0
        self.noncontiguous_soft_hits = 0

    def summary(self) -> dict:
        return {
            "hard_hit_rate": self.hard_hit_rate(),
            "soft_hit_rate": self.soft_hit_rate(),
            "weighted_hit_rate": self.weighted_hit_rate(),
            "noncontiguous_weighted_fraction": self.noncontiguous_weighted_fraction(),
            "hard_hit_chunks": self.hard_hit_chunks,
            "soft_hit_chunks": self.soft_hit_chunks,
            "miss_chunks": self.total_chunks - self.hard_hit_chunks - self.soft_hit_chunks,
            "soft_hit_beta_weight": self.beta_weight,
        }


@dataclass
class HitRateMetrics:
    total_requests: int = 0
    total_chunks: int = 0
    hit_chunks: int = 0
    noncontiguous_hit_chunks: int = 0

    def record(
        self,
        n_hits: int,
        n_misses: int,
        noncontiguous_hits: int,
    ) -> None:
        self.total_requests += 1
        self.total_chunks += n_hits + n_misses
        self.hit_chunks += n_hits
        self.noncontiguous_hit_chunks += noncontiguous_hits

    def overall_hit_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.hit_chunks / self.total_chunks

    def noncontiguous_fraction(self) -> float:
        """Fraction of hits that are non-contiguous (target ≥ 0.30)."""
        if self.hit_chunks == 0:
            return 0.0
        return self.noncontiguous_hit_chunks / self.hit_chunks

    def reset(self) -> None:
        self.total_requests = 0
        self.total_chunks = 0
        self.hit_chunks = 0
        self.noncontiguous_hit_chunks = 0

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "overall_hit_rate": self.overall_hit_rate(),
            "noncontiguous_fraction": self.noncontiguous_fraction(),
            "hit_chunks": self.hit_chunks,
            "miss_chunks": self.total_chunks - self.hit_chunks,
        }


@dataclass
class DistributedHitRateMetrics:
    """4-level distributed non-contiguous hit rate metrics.

    Tracks local HBM, PegaFlow local, RDMA remote, and miss counts.
    Independent of WeightedHitRateMetrics and HitRateMetrics.
    """

    total_lookups: int = 0
    local_hard_hits: int = 0
    pegaflow_local_hits: int = 0
    rdma_remote_hits: int = 0

    def record(self, hit_type: str) -> None:
        """Record one lookup result.

        hit_type: "local_hard_hit" | "pegaflow_local_hit" | "rdma_remote_hit" | "miss"
        """
        self.total_lookups += 1
        if hit_type == "local_hard_hit":
            self.local_hard_hits += 1
        elif hit_type == "pegaflow_local_hit":
            self.pegaflow_local_hits += 1
        elif hit_type == "rdma_remote_hit":
            self.rdma_remote_hits += 1

    def distributed_hit_rate(self) -> float:
        """(local + pegaflow_local + rdma_remote) / total."""
        if self.total_lookups == 0:
            return 0.0
        return (
            self.local_hard_hits + self.pegaflow_local_hits + self.rdma_remote_hits
        ) / self.total_lookups

    def noncontiguous_rdma_fraction(self) -> float:
        """RDMA remote hits / total hits. 0.0 means no distributed reuse."""
        total_hits = self.local_hard_hits + self.pegaflow_local_hits + self.rdma_remote_hits
        if total_hits == 0:
            return 0.0
        return self.rdma_remote_hits / total_hits

    def reset(self) -> None:
        self.total_lookups = 0
        self.local_hard_hits = 0
        self.pegaflow_local_hits = 0
        self.rdma_remote_hits = 0

    def summary(self) -> dict:
        return {
            "local_hard_hit_rate": self.local_hard_hits / max(1, self.total_lookups),
            "pegaflow_local_hit_rate": self.pegaflow_local_hits / max(1, self.total_lookups),
            "rdma_remote_hit_rate": self.rdma_remote_hits / max(1, self.total_lookups),
            "miss_rate": 1.0 - self.distributed_hit_rate(),
            "distributed_hit_rate": self.distributed_hit_rate(),
            "noncontiguous_rdma_fraction": self.noncontiguous_rdma_fraction(),
            "total_lookups": self.total_lookups,
        }


@dataclass
class SafetyGatedHitRateMetrics:
    """5-level safety gate hit rate metrics. Independent of other metrics classes."""
    total_lookups: int = 0
    safe_reuse_hits: int = 0
    partial_reuse_hits: int = 0
    gate_rejections: int = 0
    stale_evictions: int = 0
    misses: int = 0

    def record(self, outcome: str) -> None:
        """Record a single lookup outcome (HitOutcome literal value)."""
        self.total_lookups += 1
        if outcome == "safe_reuse_hit":
            self.safe_reuse_hits += 1
        elif outcome == "partial_reuse_hit":
            self.partial_reuse_hits += 1
        elif outcome == "gate_rejected":
            self.gate_rejections += 1
        elif outcome == "stale_evicted":
            self.stale_evictions += 1
        else:
            self.misses += 1

    def safety_gate_pass_rate(self) -> float:
        """safe_reuse / (safe_reuse + gate_rejected + stale_evicted). Target: ≥80%."""
        denom = self.safe_reuse_hits + self.gate_rejections + self.stale_evictions
        return self.safe_reuse_hits / denom if denom > 0 else 0.0

    def effective_hit_rate(self) -> float:
        """(safe_reuse + partial_reuse) / total_lookups."""
        if self.total_lookups == 0:
            return 0.0
        return (self.safe_reuse_hits + self.partial_reuse_hits) / self.total_lookups

    def stale_eviction_rate(self) -> float:
        return self.stale_evictions / self.total_lookups if self.total_lookups > 0 else 0.0

    def summary(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "safe_reuse_hit_rate": self.safe_reuse_hits / max(1, self.total_lookups),
            "partial_reuse_hit_rate": self.partial_reuse_hits / max(1, self.total_lookups),
            "gate_rejection_rate": self.gate_rejections / max(1, self.total_lookups),
            "stale_eviction_rate": self.stale_eviction_rate(),
            "miss_rate": self.misses / max(1, self.total_lookups),
            "safety_gate_pass_rate": self.safety_gate_pass_rate(),
            "effective_hit_rate": self.effective_hit_rate(),
        }

    def reset(self) -> None:
        self.total_lookups = 0
        self.safe_reuse_hits = 0
        self.partial_reuse_hits = 0
        self.gate_rejections = 0
        self.stale_evictions = 0
        self.misses = 0
