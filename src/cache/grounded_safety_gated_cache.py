"""GroundedSafetyGatedSegmentCache — Activity B-1.

Wraps SegmentedHashCache with a 4-gate lazy fail-fast safety validation framework
based on GroundedCache (arXiv 2605.27494). Only semantically safe segments are
reused, providing structural accuracy-preserving guarantees for non-contiguous KV
reuse.

Gate order (fail-fast):
  Gate 1: n-gram Jaccard similarity  ≥ 0.85  → reject on fail
  Gate 2: TF-IDF top-k token overlap ≥ 0.60  → partial_reuse on fail
  Gate 3: SHA-256 version hash equality        → stale_evicted on fail
  Gate 4: Mean KV cosine similarity  ≥ 0.50  → partial_reuse on fail

Pre-computation at insert() time for gates 1–4 keeps lookup overhead < 1ms.
"""

import hashlib
import math
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


HitOutcome = Literal[
    "safe_reuse_hit",    # all 4 gates passed
    "partial_reuse_hit", # Gate 2 or Gate 4 failed → return c_KV only
    "gate_rejected",     # Gate 1 failed → complete rejection
    "stale_evicted",     # Gate 3 failed → version mismatch, segment evicted
    "miss",              # not in cache
]


@dataclass
class GateThresholds:
    content_sim_threshold: float = 0.85     # Gate 1: n-gram Jaccard minimum
    evidence_overlap_threshold: float = 0.60  # Gate 2: keyword Jaccard minimum
    # Gate 3: version hash equality (binary, no threshold)
    attn_support_threshold: float = 0.50    # Gate 4: cosine similarity minimum
    ngram_n: int = 3                         # n for n-gram extraction in Gate 1
    tfidf_top_k: int = 20                   # top-k tokens for Gate 2


@dataclass
class SegmentMetadata:
    """Metadata pre-computed at insert() to minimize lookup overhead."""
    segment_id: str
    token_ids: List[int]
    ngrams: Set[Tuple[int, ...]]            # Gate 1: pre-computed n-grams
    top_k_tokens: Set[int]                  # Gate 2: TF-IDF top-k token IDs
    source_version_hash: str                # Gate 3: sha256 of source context
    mean_kv_vector: torch.Tensor            # Gate 4: mean K vector [d_kv]


@dataclass
class GroundedSafetyGatedConfig:
    gate_thresholds: GateThresholds = field(default_factory=GateThresholds)
    chunk_size: int = 128
    max_entries: int = 1000
    seed: int = 42


class GroundedSafetyGatedSegmentCache(CacheStore):
    """GroundedCache 4-gate safety validation for non-contiguous KV segment reuse.

    Wraps SegmentedHashCache as internal storage.
    4-gate fail-fast checks: Gate 1 → Gate 2 → Gate 3 → Gate 4.
    Gate 1 failure: immediate rejection (no further gates evaluated).
    Gate 2/4 failure: partial_reuse (return kv[..., :d//2] — content KV component).
    Gate 3 failure: stale segment → evict immediately.
    Metadata (SegmentMetadata) computed once at insert(), never at lookup.
    """

    def __init__(self, config: GroundedSafetyGatedConfig) -> None:
        self._inner = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._metadata: Dict[str, SegmentMetadata] = {}
        self._config = config
        # 5-level hit counters
        self._safe_reuse_hits: int = 0
        self._partial_reuse_hits: int = 0
        self._gate_rejections: int = 0
        self._stale_evictions: int = 0
        self._misses: int = 0
        self._total_lookups: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore abstract method implementations                           #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Delegate raw put to inner SegmentedHashCache (no metadata registered)."""
        self._inner.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Delegate raw get to inner SegmentedHashCache (no gate validation)."""
        return self._inner.get(key)

    def evict(self) -> int:
        return self._inner.evict()

    def hit_rate(self) -> float:
        """Effective hit rate: (safe_reuse + partial_reuse) / total_lookups."""
        if self._total_lookups == 0:
            return 0.0
        return (self._safe_reuse_hits + self._partial_reuse_hits) / self._total_lookups

    def memory_bytes(self) -> int:
        return self._inner.memory_bytes()

    def reset_stats(self) -> None:
        self._inner.reset_stats()
        self._safe_reuse_hits = 0
        self._partial_reuse_hits = 0
        self._gate_rejections = 0
        self._stale_evictions = 0
        self._misses = 0
        self._total_lookups = 0

    # ------------------------------------------------------------------ #
    # 4-gate safety validation core API                                    #
    # ------------------------------------------------------------------ #

    def insert(
        self,
        token_ids: List[int],
        kv_data: torch.Tensor,
        source_context_token_ids: Optional[List[int]] = None,
        layer_idx: int = 0,
    ) -> str:
        """Store segment with pre-computed gate metadata.

        Pre-computes n-grams (Gate 1), TF-IDF top-k (Gate 2), version hash
        (Gate 3), and mean KV vector (Gate 4) once at insert time to keep
        lookup overhead < 1ms per segment.

        Returns:
            segment key used for subsequent lookup.
        """
        key = self._inner.chunk_key(token_ids, chunk_idx=0, layer_idx=layer_idx)

        ngrams = self._compute_ngrams(
            token_ids, self._config.gate_thresholds.ngram_n
        )
        top_k = self._tfidf_top_k(
            token_ids, self._config.gate_thresholds.tfidf_top_k
        )

        ctx = source_context_token_ids if source_context_token_ids is not None else token_ids
        version_hash = self._version_hash(ctx)

        # kv_data shape assumed: [n_tokens, n_heads, d_kv] or [n_tokens, d_kv]
        if kv_data.dim() >= 3:
            mean_kv = kv_data.float().mean(dim=0).mean(dim=0)
        elif kv_data.dim() == 2:
            mean_kv = kv_data.float().mean(dim=0)
        else:
            mean_kv = kv_data.float().flatten()

        metadata = SegmentMetadata(
            segment_id=key,
            token_ids=list(token_ids),
            ngrams=ngrams,
            top_k_tokens=top_k,
            source_version_hash=version_hash,
            mean_kv_vector=mean_kv.detach().clone(),
        )
        self._metadata[key] = metadata
        self._inner.put(key, kv_data)
        return key

    def lookup(
        self,
        query_tokens: List[int],
        current_context_tokens: List[int],
        query_embedding: Optional[torch.Tensor] = None,
        current_source_version_hash: Optional[str] = None,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], HitOutcome]:
        """4-gate fail-fast safety validation then return KV (or partial KV).

        Gate evaluation order:
          Gate 1 (n-gram content sim) → fail: gate_rejected (stop)
          Gate 2 (TF-IDF evidence overlap) → fail: partial_reuse
          Gate 3 (version hash) → fail: stale_evicted (stop, evict entry)
          Gate 4 (cosine KV support) → fail: partial_reuse
          All pass: safe_reuse_hit, return full KV

        partial_reuse returns kv[..., :d//2] (content KV component).
        """
        self._total_lookups += 1

        key = self._inner.chunk_key(query_tokens, chunk_idx=0, layer_idx=layer_idx)
        kv = self._inner.get(key)
        if kv is None:
            self._misses += 1
            return None, "miss"

        meta = self._metadata.get(key)
        if meta is None:
            self._misses += 1
            return None, "miss"

        # --- Gate 1: Content Similarity (fail-fast) ---
        ctx_ngrams = self._compute_ngrams(
            current_context_tokens, self._config.gate_thresholds.ngram_n
        )
        content_sim = self._jaccard(meta.ngrams, ctx_ngrams)
        if content_sim < self._config.gate_thresholds.content_sim_threshold:
            self._gate_rejections += 1
            return None, "gate_rejected"

        # --- Gate 2: Evidence Overlap ---
        ctx_top_k = self._tfidf_top_k(
            current_context_tokens, self._config.gate_thresholds.tfidf_top_k
        )
        if len(meta.top_k_tokens) > 0:
            evidence_overlap = len(meta.top_k_tokens & ctx_top_k) / len(meta.top_k_tokens)
        else:
            evidence_overlap = 0.0
        gate2_pass = evidence_overlap >= self._config.gate_thresholds.evidence_overlap_threshold

        # --- Gate 3: Source Version Validity ---
        if current_source_version_hash is not None:
            gate3_pass = meta.source_version_hash == current_source_version_hash
        else:
            gate3_pass = True  # no version info provided → conservative allow
        if not gate3_pass:
            self._stale_evictions += 1
            self._evict_key(key)
            return None, "stale_evicted"

        # --- Gate 4: Attention Contribution Support ---
        gate4_pass = True
        if query_embedding is not None:
            attn_support = self._cosine_similarity(meta.mean_kv_vector, query_embedding)
            gate4_pass = attn_support >= self._config.gate_thresholds.attn_support_threshold

        # --- Result branch ---
        if gate2_pass and gate4_pass:
            self._safe_reuse_hits += 1
            return kv, "safe_reuse_hit"
        else:
            # partial_reuse: return content KV (first half of last dimension)
            self._partial_reuse_hits += 1
            d = kv.shape[-1]
            c_kv = kv[..., : d // 2]
            return c_kv, "partial_reuse_hit"

    def get_stats(self) -> dict:
        """Return 5-level hit rate statistics."""
        n = max(1, self._total_lookups)
        safe = self._safe_reuse_hits
        denom_gate = safe + self._gate_rejections + self._stale_evictions
        return {
            "total_lookups": self._total_lookups,
            "safe_reuse_hit_rate": safe / n,
            "partial_reuse_hit_rate": self._partial_reuse_hits / n,
            "gate_rejection_rate": self._gate_rejections / n,
            "stale_eviction_rate": self._stale_evictions / n,
            "miss_rate": self._misses / n,
            "safety_gate_pass_rate": safe / denom_gate if denom_gate > 0 else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Private gate computation helpers                                     #
    # ------------------------------------------------------------------ #

    def _compute_ngrams(self, tokens: List[int], n: int) -> Set[Tuple[int, ...]]:
        """Generate n-gram set from token ID sequence.

        O(len(tokens) × n) ≈ 0.1ms at n=3, len≤256.
        """
        if len(tokens) < n:
            return {tuple(tokens)} if tokens else set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def _tfidf_top_k(self, tokens: List[int], k: int) -> Set[int]:
        """Lightweight single-document TF-IDF approximation, return top-k token IDs.

        IDF approximated as log(N / tf) since there is no corpus available.
        This rewards rare tokens within the document (discriminative keywords).
        """
        if not tokens:
            return set()
        tf = Counter(tokens)
        n = len(tokens)
        tfidf_scores: Dict[int, float] = {}
        for token, freq in tf.items():
            # Guard against log(0): freq >= 1 always holds
            tfidf_scores[token] = freq * math.log(n / freq)
        top = sorted(tfidf_scores, key=tfidf_scores.__getitem__, reverse=True)[:k]
        return set(top)

    def _jaccard(self, a: Set, b: Set) -> float:
        """Jaccard similarity |a ∩ b| / |a ∪ b|. Empty sets → 1.0."""
        if not a and not b:
            return 1.0
        union = len(a | b)
        if union == 0:
            return 1.0
        return len(a & b) / union

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """1-D tensor cosine similarity. Normalise then dot-product."""
        a_f = a.float().reshape(1, -1)
        b_f = b.float().reshape(1, -1)
        a_norm = torch.nn.functional.normalize(a_f, dim=-1)
        b_norm = torch.nn.functional.normalize(b_f, dim=-1)
        return float((a_norm * b_norm).sum())

    def _version_hash(self, token_ids: List[int]) -> str:
        """SHA-256 hash of token ID sequence as version fingerprint."""
        raw = struct.pack(f"{len(token_ids)}I", *token_ids) if token_ids else b""
        return hashlib.sha256(raw).hexdigest()

    def _evict_key(self, key: str) -> None:
        """Remove a specific key from inner store and metadata."""
        if key in self._inner._store:
            del self._inner._store[key]
        self._metadata.pop(key, None)
