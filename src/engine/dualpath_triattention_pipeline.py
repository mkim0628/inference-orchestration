"""DualPath NIC dual-path routing + TriAttention pre-RoPE compression pipeline (Cross-1).

Activity A+C integrated pipeline: DualPathNICLoadBalancer routes KV loads,
and when the dual path is taken, TriAttentionPreRoPEKVSelectorCodec compresses
the KV cache on the relay decode node before RDMA transfer to prefill.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.cache.triattention_pre_rope_kv_selector_codec import (
    TriAttentionPreRoPEKVSelectorCodec,
    TriAttentionSelectorConfig,
)
from src.scheduler.dualpath_nic_load_balancer import (
    DualPathNICLoadBalancer,
    DualPathNICConfig,
    RoutingDecision,
)


@dataclass
class DualPathTriAttentionPipelineConfig:
    triattention_config: Optional[TriAttentionSelectorConfig] = None
    dualpath_config: Optional[DualPathNICConfig] = None
    seed: int = 42


class DualPathTriAttentionCompressPipeline:
    """DualPath NIC dual-path + TriAttention pre-RoPE compression integrated pipeline (Cross-1).

    Integrated processing flow:
      Step 1 (A-1): DualPathNICLoadBalancer — detect prefill NIC saturation, select idle decode.
      Step 2 (I/O simulation): storage -> decode node NIC loads original KV.
      Step 3 (C-1): TriAttentionPreRoPEKVSelectorCodec.select_kv() on decode node.
                     Retain top kv_budget_ratio important keys (10.7x compression).
      Step 4 (RDMA simulation): transfer compressed KV from decode -> prefill via RDMA.
      Step 5: return compressed KV for standard attention kernel.

    Single path: return K_full, V_full uncompressed.
    Dual path: return TriAttention-compressed KV (kv_bytes = budget_ratio of original).
    """

    def __init__(self, config: DualPathTriAttentionPipelineConfig) -> None:
        torch.manual_seed(config.seed)
        ta_cfg = config.triattention_config or TriAttentionSelectorConfig(seed=config.seed)
        dp_cfg = config.dualpath_config or DualPathNICConfig(seed=config.seed)
        self.triattention = TriAttentionPreRoPEKVSelectorCodec(ta_cfg)
        self.dualpath_lb = DualPathNICLoadBalancer(dp_cfg)
        self._pipeline_stats: List[dict] = []

    def run_pipeline(
        self,
        request: Any,
        Q: torch.Tensor,                          # [T_q, d_head] pre-RoPE
        K_full: torch.Tensor,                     # [N, d_head]
        V_full: torch.Tensor,                     # [N, d_head]
        key_positions: Optional[torch.Tensor] = None,
        pos_q: int = 0,
        is_reasoning_task: bool = False,
        kv_pool_pressure: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Execute dual-path routing + TriAttention compression.

        Returns:
            (K_final, V_final, pipeline_report)
        """
        routing: RoutingDecision = self.dualpath_lb.decide_routing(request)

        if routing.path == "single":
            report = {
                "path": "single",
                "compression_ratio": 1.0,
                "kv_bytes_transferred": K_full.nbytes + V_full.nbytes,
                "routing_latency_ms": routing.decision_latency_ms,
                "conc_q": None,
                "conc_k": None,
                "kv_budget_ratio": None,
            }
            self._pipeline_stats.append(report)
            return K_full, V_full, report

        # Step 3: TriAttention compression on relay decode node
        K_sel, V_sel, kept_idx, conc_q, conc_k, budget = self.triattention.select_kv(
            Q, K_full, V_full, key_positions, pos_q, is_reasoning_task, kv_pool_pressure
        )
        n_keep = K_sel.shape[0]
        N = K_full.shape[0]
        actual_ratio = float(N) / max(1, n_keep)

        # Step 4: RDMA transfer completion accounting
        if routing.relay_decode_node_id:
            self.dualpath_lb.complete_dual_path(routing.relay_decode_node_id)

        report = {
            "path": "dual",
            "relay_decode_node_id": routing.relay_decode_node_id,
            "compression_ratio": actual_ratio,
            "kv_bytes_transferred": K_sel.nbytes + V_sel.nbytes,
            "kv_bytes_original": K_full.nbytes + V_full.nbytes,
            "kv_size_reduction_ratio": 1.0 - (K_sel.nbytes + V_sel.nbytes) / max(
                1, K_full.nbytes + V_full.nbytes
            ),
            "routing_latency_ms": routing.decision_latency_ms,
            "conc_q": conc_q,
            "conc_k": conc_k,
            "kv_budget_ratio": budget,
            "is_reasoning_task": is_reasoning_task,
        }
        self._pipeline_stats.append(report)
        return K_sel, V_sel, report

    def pipeline_summary(self) -> dict:
        if not self._pipeline_stats:
            return {}
        dual = [s for s in self._pipeline_stats if s["path"] == "dual"]
        single = [s for s in self._pipeline_stats if s["path"] == "single"]
        avg_cr = sum(s["compression_ratio"] for s in dual) / len(dual) if dual else 1.0
        avg_sr = (
            sum(s["kv_size_reduction_ratio"] for s in dual) / len(dual) if dual else 0.0
        )
        return {
            "total_runs": len(self._pipeline_stats),
            "dual_path_runs": len(dual),
            "single_path_runs": len(single),
            "dual_path_ratio": len(dual) / max(1, len(self._pipeline_stats)),
            "avg_compression_ratio": avg_cr,
            "avg_kv_size_reduction_ratio": avg_sr,
            **self.dualpath_lb.scheduling_stats(),
            **self.triattention.concentration_stats(),
        }
