"""ARKV-style tri-state KV cache compressor (Activity C).

Classifies KV tokens into three tiers based on attention scores:
  - retain  (top    retain_ratio)  → stored as FP16 (near-lossless)
  - compress (middle compress_ratio) → INT4 via HadamardInt4Codec
  - evict   (bottom evict_ratio)  → discarded (low-attention, minimal impact)

Memory savings vs FP32 baseline:
  retain   20% × (2/4) = 0.10
  compress 40% × (1/4) = 0.10
  evict    40% ×  0    = 0.00
  total retained fraction = 0.20 (i.e., 80% savings)
"""

from typing import Dict, List, Tuple
import torch

from src.cache.compression import HadamardInt4Codec


class TriStateCompressor:
    """Tri-state KV classifier and compressor (retain / compress / evict)."""

    def __init__(
        self,
        codec: HadamardInt4Codec,
        retain_ratio: float = 0.20,
        evict_ratio: float = 0.40,
    ) -> None:
        self.codec = codec
        self.retain_ratio = retain_ratio
        self.evict_ratio = evict_ratio
        # compress_ratio is the remainder
        self.compress_ratio = 1.0 - retain_ratio - evict_ratio

    def classify(
        self,
        kv: torch.Tensor,
        attn_weights: torch.Tensor,
        layer_idx: int,
    ) -> Dict:
        """Partition tokens into retain / compress / evict index sets.

        Args:
            kv:           (n_tokens, kv_dim) float tensor
            attn_weights: (n_tokens,) float tensor — higher = more important
            layer_idx:    used to determine codec precision tier

        Returns dict with keys:
            retain_kv, compress_kv,
            retain_indices, compress_indices, evict_indices
        """
        n_tokens = kv.shape[0]
        sorted_indices = torch.argsort(attn_weights, descending=True)

        n_retain = max(1, int(n_tokens * self.retain_ratio))
        n_evict = max(0, int(n_tokens * self.evict_ratio))
        n_compress = n_tokens - n_retain - n_evict

        retain_indices = sorted_indices[:n_retain]
        compress_indices = sorted_indices[n_retain: n_retain + n_compress]
        evict_indices = sorted_indices[n_retain + n_compress:]

        retain_kv = kv[retain_indices]
        compress_kv = kv[compress_indices]

        return {
            "retain_kv": retain_kv,
            "compress_kv": compress_kv,
            "retain_indices": retain_indices,
            "compress_indices": compress_indices,
            "evict_indices": evict_indices,
        }

    def encode(
        self,
        kv: torch.Tensor,
        attn_weights: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> Dict:
        """Classify then compress; returns a storage dict.

        Storage dict keys:
            retain_kv        FP16 tensor for high-attention tokens
            compressed_kv    INT8 tensor (codec output) for mid-attention tokens
            retain_indices   LongTensor
            compress_indices LongTensor
            evict_indices    LongTensor
            n_tokens         int — original sequence length
            layer_idx        int
            tensor_id        int
        """
        classification = self.classify(kv, attn_weights, layer_idx)

        retain_kv_fp16 = classification["retain_kv"].to(torch.float16)
        compressed_kv = self.codec.encode(
            classification["compress_kv"], layer_idx, tensor_id
        )

        return {
            "retain_kv": retain_kv_fp16,
            "compressed_kv": compressed_kv,
            "retain_indices": classification["retain_indices"],
            "compress_indices": classification["compress_indices"],
            "evict_indices": classification["evict_indices"],
            "n_tokens": kv.shape[0],
            "layer_idx": layer_idx,
            "tensor_id": tensor_id,
        }

    def decode(
        self,
        storage: Dict,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Reconstruct full (n_tokens, kv_dim) tensor; evicted positions are zeros.

        Non-evicted tokens are placed at their original positions.
        """
        n_tokens: int = storage["n_tokens"]
        retain_kv: torch.Tensor = storage["retain_kv"].float()
        compressed_kv: torch.Tensor = storage["compressed_kv"]
        retain_indices: torch.Tensor = storage["retain_indices"]
        compress_indices: torch.Tensor = storage["compress_indices"]

        kv_dim = retain_kv.shape[-1]
        reconstructed = torch.zeros(n_tokens, kv_dim, dtype=torch.float32)

        # Place retained tokens (FP16 → FP32)
        reconstructed[retain_indices] = retain_kv

        # Decompress and place compressed tokens
        if compress_indices.numel() > 0:
            decompressed = self.codec.decode(compressed_kv, layer_idx, tensor_id).float()
            reconstructed[compress_indices] = decompressed

        # Evicted positions remain zero (already initialised above)
        return reconstructed

    def compression_ratio(
        self,
        retain_ratio: float = 0.20,
        evict_ratio: float = 0.40,
    ) -> float:
        """Fraction of FP32 memory retained after tri-state compression.

        retain tier  : stored as FP16 → 2 bytes per element  (factor 0.5 vs FP32)
        compress tier: stored as INT8 → 1 byte per element    (factor 0.25 vs FP32)
        evict tier   : not stored                             (factor 0)

        compress_ratio = 1 - retain_ratio - evict_ratio
        """
        compress_ratio = 1.0 - retain_ratio - evict_ratio
        retained_fraction = (
            retain_ratio * 0.5       # FP16 vs FP32
            + compress_ratio * 0.25  # INT8 vs FP32
        )
        return retained_fraction
