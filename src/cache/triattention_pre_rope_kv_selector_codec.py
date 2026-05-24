"""TriAttention pre-RoPE Q/K concentration KV selector codec (arXiv 2604.04921).

Activity C (highest priority): training-free, accuracy-preserving KV selection
using pre-RoPE Q/K concentration metrics and trigonometric distance preference scores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class TriAttentionSelectorConfig:
    d_head: int = 128
    n_kv_heads: int = 8
    rope_base: float = 10000.0
    kv_budget_ratio_reasoning: float = 0.093    # reasoning task (10.7x reduction)
    kv_budget_ratio_default: float = 0.20       # non-reasoning (conservative)
    high_pressure_threshold: float = 0.80       # KV pool occupancy high-pressure threshold
    high_pressure_extra_reduction: float = 0.10 # additional 10% reduction under high pressure
    max_entries: int = 1000
    seed: int = 42


@dataclass
class SelectorKVEntry:
    selected_kv: torch.Tensor      # [n_kept, d_head] FP16
    kept_indices: torch.Tensor     # [n_kept] int64
    original_seq_len: int
    conc_q: float
    conc_k: float
    kv_budget_ratio: float
    is_reasoning_task: bool


class TriAttentionPreRoPEKVSelectorCodec(CacheStore):
    """TriAttention pre-RoPE Q/K concentration trigonometric KV selector codec (arXiv 2604.04921).

    Activity C: training-free, accuracy-preserving KV selection.

    Core algorithm:
      1. Compute pre-RoPE Q centroid: mu_Q = mean(Q, dim=0).
      2. Q concentration: conc_Q = ||mu_Q||_2 / mean(||q_t||_2) in [0, 1].
      3. Trigonometric series distance preference score:
           dist_pref(k_i) = sum_{d=1}^{d_k/2} [|mu_Q[2d-1]| * cos(theta_d * |pos_i - pos_q|)
                                               + |mu_Q[2d]|   * sin(theta_d * |pos_i - pos_q|)]
           theta_d = rope_base^(-2d/d_k)
      4. K norm auxiliary score: norm_score(k_i) = ||k_i|| / max(||k_j||)
      5. Concentration-weighted combination:
           importance(k_i) = conc_Q * dist_pref(k_i) + (1 - conc_Q) * norm_score(k_i)
      6. Select top n_keep = max(1, int(N * kv_budget_ratio)) tokens.

    CacheStore interface:
      - put(key, value): plain storage (no compression, for compatibility)
      - put_compressed(key, Q, K, V, ...): compressed storage (recommended)
      - get(key): return selected KV tensor
      - evict(): FIFO eviction
      - hit_rate(), memory_bytes(), reset_stats()
    """

    def __init__(self, config: TriAttentionSelectorConfig) -> None:
        torch.manual_seed(config.seed)
        self.config = config
        self._store: Dict[str, SelectorKVEntry] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._conc_q_history: List[float] = []
        self._conc_k_history: List[float] = []
        # Pre-compute RoPE frequencies: theta_d = rope_base^(-2d/d_k), d=1..d_k/2
        half_d = config.d_head // 2
        d_indices = torch.arange(1, half_d + 1, dtype=torch.float32)
        self._rope_freqs: torch.Tensor = config.rope_base ** (-2.0 * d_indices / config.d_head)

    @staticmethod
    def compute_concentration(vecs: torch.Tensor) -> float:
        """Concentration metric for a set of Q or K vectors.

        conc = ||mean(vecs)||_2 / mean(||vecs||_2), clamped to [0, 1].

        Args:
            vecs: [T, d_head] FP32

        Returns:
            float in [0, 1]
        """
        vecs_f = vecs.float()
        mu = vecs_f.mean(dim=0)
        mu_norm = float(mu.norm())
        mean_norms = float(vecs_f.norm(dim=-1).mean())
        raw = mu_norm / (mean_norms + 1e-8)
        return float(min(max(raw, 0.0), 1.0))

    def compute_dist_pref_scores(
        self,
        mu_q: torch.Tensor,           # [d_head] FP32
        key_positions: torch.Tensor,  # [N] int64
        pos_q: int,
    ) -> torch.Tensor:
        """Trigonometric series distance preference scores.

        dist_pref(k_i) = sum_{d=1}^{d_k/2}
            [|mu_Q[2d-1]| * cos(theta_d * |pos_i - pos_q|)
           + |mu_Q[2d]|   * sin(theta_d * |pos_i - pos_q|)]

        Returns:
            [N] FP32
        """
        device = mu_q.device
        rope_freqs = self._rope_freqs.to(device=device, dtype=torch.float32)
        delta_pos = (key_positions.float() - float(pos_q)).abs()  # [N]
        phase = delta_pos.unsqueeze(1) * rope_freqs.unsqueeze(0)  # [N, d_k/2]
        half_d = self.config.d_head // 2
        mu_odd  = mu_q.float()[0::2][:half_d].abs()  # |mu_Q[2d-1]|
        mu_even = mu_q.float()[1::2][:half_d].abs()  # |mu_Q[2d]|
        scores = mu_odd.unsqueeze(0) * torch.cos(phase) \
               + mu_even.unsqueeze(0) * torch.sin(phase)  # [N, d_k/2]
        return scores.sum(dim=-1)  # [N]

    def select_kv(
        self,
        Q: torch.Tensor,                          # [T_q, d_head] pre-RoPE
        K: torch.Tensor,                          # [N, d_head] pre-RoPE
        V: torch.Tensor,                          # [N, d_head]
        key_positions: Optional[torch.Tensor] = None,  # [N] int64
        pos_q: int = 0,
        is_reasoning_task: bool = False,
        kv_pool_pressure: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """Select important KV tokens using concentration + trigonometric preference.

        Returns:
            (K_selected, V_selected, kept_indices, conc_q, conc_k, kv_budget_ratio)
        """
        Q_f = Q.float()
        K_f = K.float()
        N = K_f.shape[0]
        device = K_f.device

        # 1. Budget ratio selection
        if is_reasoning_task:
            budget = self.config.kv_budget_ratio_reasoning
        else:
            budget = self.config.kv_budget_ratio_default
        if kv_pool_pressure >= self.config.high_pressure_threshold:
            budget = budget * (1.0 - self.config.high_pressure_extra_reduction)
        n_keep = max(1, int(N * budget))

        # 2. Concentration metrics
        mu_q = Q_f.mean(dim=0)
        conc_q = self.compute_concentration(Q_f)
        conc_k = self.compute_concentration(K_f)

        # 3. Trigonometric distance preference scores
        if key_positions is None:
            key_positions = torch.arange(N, dtype=torch.int64, device=device)
        dist_pref = self.compute_dist_pref_scores(mu_q, key_positions, pos_q)

        # 4. K norm auxiliary scores
        k_norms = K_f.norm(dim=-1)
        norm_score = k_norms / (k_norms.max() + 1e-8)

        # 5. Concentration-weighted combination
        importance = conc_q * dist_pref + (1.0 - conc_q) * norm_score

        # 6. Select top n_keep tokens
        kept_indices = importance.topk(n_keep).indices.sort().values

        self._conc_q_history.append(conc_q)
        self._conc_k_history.append(conc_k)

        return (
            K[kept_indices].to(K.dtype),
            V[kept_indices].to(V.dtype),
            kept_indices,
            conc_q,
            conc_k,
            budget,
        )

    @staticmethod
    def detect_reasoning_task(prompt_text: str) -> bool:
        """Detect reasoning task by presence of <think> tags."""
        return "<think>" in prompt_text or "</think>" in prompt_text

    # ---- CacheStore interface ----

    def put(self, key: str, value: torch.Tensor) -> None:
        """Plain storage (no compression — compatibility when Q is unavailable)."""
        if key in self._store:
            return
        if len(self._store) >= self.config.max_entries:
            self.evict()
        seq_len = value.shape[0]
        entry = SelectorKVEntry(
            selected_kv=value.detach().clone(),
            kept_indices=torch.arange(seq_len, dtype=torch.int64),
            original_seq_len=seq_len,
            conc_q=0.5,
            conc_k=0.5,
            kv_budget_ratio=self.config.kv_budget_ratio_default,
            is_reasoning_task=False,
        )
        self._store[key] = entry

    def put_compressed(
        self,
        key: str,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        key_positions: Optional[torch.Tensor] = None,
        pos_q: int = 0,
        is_reasoning_task: bool = False,
        kv_pool_pressure: float = 0.0,
    ) -> SelectorKVEntry:
        """Select KV tokens and store compressed entry (recommended path)."""
        if key in self._store:
            return self._store[key]
        if len(self._store) >= self.config.max_entries:
            self.evict()
        K_sel, V_sel, kept_idx, conc_q, conc_k, budget = self.select_kv(
            Q, K, V, key_positions, pos_q, is_reasoning_task, kv_pool_pressure
        )
        entry = SelectorKVEntry(
            selected_kv=K_sel.detach().clone(),
            kept_indices=kept_idx,
            original_seq_len=K.shape[0],
            conc_q=conc_q,
            conc_k=conc_k,
            kv_budget_ratio=budget,
            is_reasoning_task=is_reasoning_task,
        )
        self._store[key] = entry
        return entry

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self._store:
            self._misses += 1
            return None
        self._hits += 1
        return self._store[key].selected_kv

    def get_entry(self, key: str) -> Optional[SelectorKVEntry]:
        return self._store.get(key)

    def compression_hook(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Norm-based simple compression fallback (when Q is unavailable)."""
        N = value.shape[0]
        budget = self.config.kv_budget_ratio_default
        n_keep = max(1, int(N * budget))
        norms = value.float().norm(dim=-1)
        kept = norms.topk(n_keep).indices.sort().values
        return value[kept]

    def evict(self) -> int:
        if not self._store:
            return 0
        k = next(iter(self._store))
        entry = self._store.pop(k)
        return entry.selected_kv.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(e.selected_kv.nbytes for e in self._store.values())

    def memory_reduction_ratio(self) -> float:
        """Memory reduction ratio relative to original FP16 equivalent memory."""
        total_selected = 0
        total_original = 0
        for e in self._store.values():
            d = e.selected_kv.shape[-1]
            total_selected += e.selected_kv.nbytes
            total_original += e.original_seq_len * d * 2  # FP16 = 2 bytes
        if total_original == 0:
            return 0.0
        return 1.0 - total_selected / total_original

    def concentration_stats(self) -> dict:
        """Distribution statistics for conc_Q/conc_K (for JSON recording)."""
        if not self._conc_q_history:
            return {"conc_q_mean": 0.0, "conc_k_mean": 0.0}
        return {
            "conc_q_mean": float(sum(self._conc_q_history) / len(self._conc_q_history)),
            "conc_k_mean": float(sum(self._conc_k_history) / len(self._conc_k_history)),
        }

    def get_importance_mask(self, key: str) -> Optional[torch.Tensor]:
        """Return bool mask [original_seq_len] based on kept_indices."""
        entry = self._store.get(key)
        if entry is None:
            return None
        mask = torch.zeros(entry.original_seq_len, dtype=torch.bool)
        mask[entry.kept_indices] = True
        return mask

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._conc_q_history.clear()
        self._conc_k_history.clear()
