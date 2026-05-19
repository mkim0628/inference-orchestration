import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch

from src.scheduler.base import BaseScheduler

Tier = Literal["HBM", "DRAM", "SSD"]


@dataclass
class TierInfo:
    tier: Tier
    physical_location: str  # e.g. "hbm:0", "dram:0", "ssd:/tmp/kv_store"
    approx_size: int        # bytes


class KVTierRegistry:
    """O(1) token_id → TierInfo registry."""

    def __init__(self) -> None:
        self._registry: Dict[int, TierInfo] = {}

    def set_tier(self, token_id: int, info: TierInfo) -> None:
        self._registry[token_id] = info

    def get_tier(self, token_id: int) -> Optional[TierInfo]:
        return self._registry.get(token_id)

    def all_token_ids(self) -> List[int]:
        return list(self._registry.keys())

    def clear(self) -> None:
        self._registry.clear()


@dataclass
class KVDriveSchedulerConfig:
    attn_hbm_threshold: float = 0.80       # top 20% cumulative attention → HBM
    attn_dram_threshold: float = 0.30      # bottom 30% → SSD, middle 50% → DRAM
    local_window_size: int = 512           # recent N tokens always in HBM
    tier_update_interval: int = 32         # tier reassignment period (decode steps)
    hbm_latency_ms: float = 0.01
    dram_latency_ms: float = 0.5
    ssd_latency_ms: float = 5.0
    ssd_prefetch_steps_ahead: int = 3
    enable_multinode: bool = False
    seed: int = 42


class KVDriveAttentionAwarePipelineSchedulerMixin:
    """Mixin providing attention-score-based 3-tier KV placement logic.

    Tier assignment:
      1. Recent local_window_size tokens → always HBM.
      2. Outside window: cumul_attn >= attn_hbm_threshold → HBM,
         attn_dram_threshold <= cumul_attn < attn_hbm_threshold → DRAM,
         cumul_attn < attn_dram_threshold → SSD.
      3. Tiers are reassigned every tier_update_interval decode steps.
    """

    def _init_mixin(self, config: KVDriveSchedulerConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.registry = KVTierRegistry()
        self._cumul_attn: Dict[int, float] = {}
        self._step_count: int = 0
        self._scheduling_times: List[float] = []

    def register_token_attention(self, token_id: str, attn_score: float) -> None:
        """Update cumulative attention for a token (EMA alpha=0.95)."""
        tid = int(token_id) if token_id.isdigit() else hash(token_id) % (2 ** 31)
        prev = self._cumul_attn.get(tid, 0.0)
        self._cumul_attn[tid] = 0.95 * prev + 0.05 * attn_score

    def update_tiers(self, force: bool = False) -> None:
        """Reassign tiers for all tracked tokens.

        Called every tier_update_interval steps, or immediately when force=True.
        """
        token_ids = list(self._cumul_attn.keys())
        if token_ids:
            self._assign_tiers_for_ids(token_ids)

    def _assign_tiers_for_ids(self, token_ids: List[int]) -> None:
        n = len(token_ids)
        window_set = set(token_ids[max(0, n - self.config.local_window_size):])

        scores = [self._cumul_attn.get(tid, 0.0) for tid in token_ids]
        max_s = max(scores) if scores and max(scores) > 0 else 1.0
        norm_scores = [s / max_s for s in scores]

        for token_id, norm_s in zip(token_ids, norm_scores):
            if token_id in window_set:
                tier: Tier = "HBM"
            elif norm_s >= self.config.attn_hbm_threshold:
                tier = "HBM"
            elif norm_s >= self.config.attn_dram_threshold:
                tier = "DRAM"
            else:
                tier = "SSD"
            self.registry.set_tier(
                token_id,
                TierInfo(
                    tier=tier,
                    physical_location=f"{tier.lower()}:0",
                    approx_size=128,
                ),
            )

    def get_token_tier(self, token_id: str) -> str:
        """Return the tier string for the given token_id ("HBM"/"DRAM"/"SSD")."""
        tid = int(token_id) if isinstance(token_id, str) and token_id.isdigit() else hash(token_id) % (2 ** 31)
        info = self.registry.get_tier(tid)
        return info.tier if info is not None else "SSD"

    def schedule(self, requests: List[dict]) -> List[dict]:
        """Sort requests by HBM hit potential (descending), preserve fairness via stable sort."""
        t0 = time.monotonic()

        def _hbm_score(req: dict) -> float:
            token_ids = req.get("token_ids", [])
            if not token_ids:
                return 0.0
            hbm_count = sum(
                1 for tid in token_ids
                if (lambda info: info is not None and info.tier == "HBM")(
                    self.registry.get_tier(tid)
                )
            )
            return hbm_count / len(token_ids)

        result = sorted(requests, key=_hbm_score, reverse=True)
        self._scheduling_times.append((time.monotonic() - t0) * 1000.0)
        return result

    def scheduling_overhead_ms(self) -> float:
        """Return the most recent scheduling call duration in ms."""
        return self._scheduling_times[-1] if self._scheduling_times else 0.0

    def scheduling_overhead_ms_p50(self) -> float:
        if not self._scheduling_times:
            return 0.0
        s = sorted(self._scheduling_times)
        return s[len(s) // 2]

    def unnecessary_eviction_rate(self) -> float:
        """Always 0.0 — SSD tier replaces eviction."""
        return 0.0

    def reset_stats(self) -> None:
        self._scheduling_times.clear()
        self._step_count = 0


class KVDriveAttentionAwarePipelineScheduler(KVDriveAttentionAwarePipelineSchedulerMixin, BaseScheduler):
    """KVDrive attention-aware 3-tier pipeline scheduler (Activity A).

    Combines KVDriveAttentionAwarePipelineSchedulerMixin with BaseScheduler.
    Supports asyncio-based prefetch simulation for I/O-compute overlap.
    """

    def __init__(self, config: KVDriveSchedulerConfig) -> None:
        self._init_mixin(config)

    def update_attention_scores(
        self,
        token_ids: List[int],
        attn_weights: torch.Tensor,
    ) -> None:
        """Update cumulative attention scores with EMA (alpha=0.95)."""
        for i, token_id in enumerate(token_ids):
            weight = attn_weights[i].item() if i < len(attn_weights) else 0.0
            prev = self._cumul_attn.get(token_id, 0.0)
            self._cumul_attn[token_id] = 0.95 * prev + 0.05 * weight

    def assign_tiers(self, token_ids: List[int]) -> None:
        """Assign 3-tier placement for token_ids and update KVTierRegistry."""
        self._assign_tiers_for_ids(token_ids)

    def step(
        self,
        token_ids: List[int],
        attn_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Process one decode step: update attention scores, then reassign tiers at interval."""
        if attn_weights is not None:
            self.update_attention_scores(token_ids, attn_weights)
        self._step_count += 1
        if self._step_count % self.config.tier_update_interval == 0:
            self.assign_tiers(token_ids)

    async def _prefetch_async(self, token_id: int) -> None:
        """Simulate async SSD prefetch with asyncio sleep."""
        info = self.registry.get_tier(token_id)
        if info is None:
            return
        latency = {
            "HBM": self.config.hbm_latency_ms,
            "DRAM": self.config.dram_latency_ms,
            "SSD": self.config.ssd_latency_ms,
        }.get(info.tier, self.config.hbm_latency_ms)
        await asyncio.sleep(latency / 1000.0)

    def prefetch(self, token_ids: List[int]) -> None:
        """Trigger async prefetch for a list of token IDs (I/O-compute overlap mock)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                asyncio.gather(*[self._prefetch_async(tid) for tid in token_ids])
            )
        finally:
            loop.close()


def make_kvdrive_scheduler_class() -> type:
    """Factory returning a KVDriveAttentionAwarePipelineScheduler class."""

    class _KVDriveScheduler(KVDriveAttentionAwarePipelineScheduler):
        pass

    return _KVDriveScheduler
