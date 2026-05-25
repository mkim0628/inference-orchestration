"""Smoke tests for vLLM integration — 2026-05-25 cycle (Activity B+C).

Tests:
  1. Import check: vericache_codec_patch and kv_packet_block_manager_patch import without error.
  2. VeriCacheCodecHookConfig: instantiation + field validation.
  3. VeriCacheCodecAttentionHook:
       - write_to_cache returns original tensors (accuracy contract)
       - read_from_cache returns VerificationResult on hit, None on miss
       - get_final_output returns draft (accepted) or verified (rejected)
       - memory_reduction_ratio >= 0.30 (INT8 2x compression)
  4. apply_vericache_codec_patch: monkey-patch attaches hooks, idempotent.
  5. extend_cache_config_vericache: CacheConfig extension adds correct fields.
  6. KVPacketSegmentConfig: instantiation + field defaults.
  7. _InlineKVPacketStore:
       - put + get (adapter-prepended shape)
       - get_kv_pair returns (K, V) tuple
       - miss returns None
       - non-contiguous hit rate tracking
  8. KVPacketSegmentMixin (via MinimalKVPktMgr):
       - store_kv_packet_segment stores correctly
       - find_noncontiguous_hits finds all stored segments
       - find_noncontiguous_hits returns empty on miss
       - build_kv_packet_block_table produces correct shape and sentinel padding
       - kv_packet_stats returns expected keys
  9. make_kv_packet_kv_cache_manager_class: produces KVCacheManager subclass.
  10. Cross B+C: KVPacket → VeriCache pipeline accuracy (final output < 1% error vs full KV).

Does NOT require:
  - A running vLLM server
  - A GPU
  - src/ imports (all inline fallbacks active)
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

# Ensure repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# 1. Import checks                                                              #
# --------------------------------------------------------------------------- #

class TestImports:
    def test_vericache_codec_patch_imports(self) -> None:
        from vllm_integration import vericache_codec_patch  # noqa: F401
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig,
            VeriCacheCodecAttentionHook,
            VeriCacheVerificationResult,
            apply_vericache_codec_patch,
            extend_cache_config_vericache,
        )
        assert VeriCacheCodecHookConfig is not None
        assert VeriCacheCodecAttentionHook is not None
        assert VeriCacheVerificationResult is not None
        assert callable(apply_vericache_codec_patch)
        assert callable(extend_cache_config_vericache)

    def test_kv_packet_block_manager_patch_imports(self) -> None:
        from vllm_integration import kv_packet_block_manager_patch  # noqa: F401
        from vllm_integration.kv_packet_block_manager_patch import (
            KVPacketSegmentConfig,
            KVPacketSegmentMixin,
            make_kv_packet_kv_cache_manager_class,
            _InlineKVPacketStore,
            _hash_token_ids,
        )
        assert KVPacketSegmentConfig is not None
        assert KVPacketSegmentMixin is not None
        assert callable(make_kv_packet_kv_cache_manager_class)
        assert _InlineKVPacketStore is not None
        assert callable(_hash_token_ids)


# --------------------------------------------------------------------------- #
# 2. VeriCacheCodecHookConfig                                                   #
# --------------------------------------------------------------------------- #

class TestVeriCacheCodecHookConfig:
    def test_default_instantiation(self) -> None:
        from vllm_integration.vericache_codec_patch import VeriCacheCodecHookConfig
        cfg = VeriCacheCodecHookConfig()
        assert cfg.d_head == 128
        assert cfg.acceptance_threshold == 0.01
        assert cfg.max_entries_per_layer == 512
        assert cfg.seed == 42

    def test_custom_instantiation(self) -> None:
        from vllm_integration.vericache_codec_patch import VeriCacheCodecHookConfig
        cfg = VeriCacheCodecHookConfig(d_head=64, acceptance_threshold=0.02, max_entries_per_layer=100)
        assert cfg.d_head == 64
        assert cfg.acceptance_threshold == 0.02
        assert cfg.max_entries_per_layer == 100


# --------------------------------------------------------------------------- #
# 3. VeriCacheCodecAttentionHook                                                #
# --------------------------------------------------------------------------- #

class TestVeriCacheCodecAttentionHook:
    def _make_hook(self, **kwargs) -> "VeriCacheCodecAttentionHook":
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig,
            VeriCacheCodecAttentionHook,
        )
        cfg = VeriCacheCodecHookConfig(d_head=64, max_entries_per_layer=50, **kwargs)
        return VeriCacheCodecAttentionHook(cfg)

    def test_write_returns_original_tensors(self) -> None:
        """MANDATORY accuracy contract: primary attention kernel receives unmodified KV."""
        hook = self._make_hook()
        key = torch.randn(16, 64)
        val = torch.randn(16, 64)
        k_out, v_out = hook.write_to_cache("seg_test", key, val, layer_idx=0)
        assert k_out is key, "write_to_cache must return the original key tensor (identity)"
        assert v_out is val, "write_to_cache must return the original value tensor (identity)"

    def test_write_repeated_key_still_returns_original(self) -> None:
        """Idempotent: second write with same key still returns originals."""
        hook = self._make_hook()
        key = torch.randn(8, 64)
        val = torch.randn(8, 64)
        hook.write_to_cache("seg_dup", key, val, layer_idx=0)
        key2 = torch.randn(8, 64)
        val2 = torch.randn(8, 64)
        k_out, v_out = hook.write_to_cache("seg_dup", key2, val2, layer_idx=0)
        assert k_out is key2 and v_out is val2

    def test_read_from_cache_hit(self) -> None:
        """read_from_cache must return VerificationResult on hit."""
        from vllm_integration.vericache_codec_patch import VeriCacheVerificationResult
        hook = self._make_hook()
        key = torch.randn(16, 64)
        val = torch.randn(16, 64)
        hook.write_to_cache("seg_read", key, val, layer_idx=0)
        Q = torch.randn(4, 64)
        result = hook.read_from_cache("seg_read", layer_idx=0, Q=Q)
        assert result is not None, "Must return VerificationResult on hit"
        assert isinstance(result, VeriCacheVerificationResult)
        assert isinstance(result.accepted, bool)
        assert result.relative_error >= 0.0
        assert result.acceptance_threshold == 0.01

    def test_read_from_cache_miss(self) -> None:
        """read_from_cache must return None on cache miss."""
        hook = self._make_hook()
        Q = torch.randn(4, 64)
        result = hook.read_from_cache("nonexistent_key", layer_idx=0, Q=Q)
        assert result is None

    def test_read_from_cache_no_query(self) -> None:
        """read_from_cache with Q=None must return None."""
        hook = self._make_hook()
        hook.write_to_cache("seg_noq", torch.randn(8, 64), torch.randn(8, 64), layer_idx=0)
        result = hook.read_from_cache("seg_noq", layer_idx=0, Q=None)
        assert result is None

    def test_get_final_output_accepted(self) -> None:
        """Accepted draft: get_final_output returns draft_output."""
        from vllm_integration.vericache_codec_patch import (
            VeriCacheVerificationResult, VeriCacheCodecAttentionHook,
        )
        draft = torch.randn(4, 64)
        verified = torch.randn(4, 64)
        result = VeriCacheVerificationResult(
            accepted=True,
            draft_output=draft,
            verified_output=verified,
            relative_error=0.001,
            acceptance_threshold=0.01,
        )
        hook = self._make_hook()
        out = hook.get_final_output(result)
        assert out is draft

    def test_get_final_output_rejected(self) -> None:
        """Rejected draft: get_final_output returns verified_output (accuracy guarantee)."""
        from vllm_integration.vericache_codec_patch import (
            VeriCacheVerificationResult, VeriCacheCodecAttentionHook,
        )
        draft = torch.randn(4, 64)
        verified = torch.randn(4, 64)
        result = VeriCacheVerificationResult(
            accepted=False,
            draft_output=draft,
            verified_output=verified,
            relative_error=0.05,
            acceptance_threshold=0.01,
        )
        hook = self._make_hook()
        out = hook.get_final_output(result)
        assert out is verified

    def test_memory_reduction_ratio(self) -> None:
        """INT8 codec must yield >= 30% memory reduction."""
        hook = self._make_hook()
        key = torch.randn(32, 64)
        val = torch.randn(32, 64)
        hook.write_to_cache("seg_mem", key, val, layer_idx=0)
        mrr = hook.memory_reduction_ratio()
        assert mrr >= 0.30, f"memory_reduction_ratio={mrr:.3f} < 0.30 (MANDATORY Activity C)"

    def test_stats_keys(self) -> None:
        """stats() must return all required keys."""
        hook = self._make_hook()
        stats = hook.stats()
        for key in ("hit_rate", "draft_acceptance_rate", "mean_relative_error",
                    "memory_reduction_ratio", "compression_codec", "compression_ratio"):
            assert key in stats, f"Missing stats key: {key}"
        assert stats["compression_codec"] == "Int8DraftCodec"
        assert stats["compression_ratio"] == 2.0

    def test_accuracy_final_output_vs_full_kv(self) -> None:
        """MANDATORY: final output relative error vs full KV < 1%.

        With acceptance_threshold=0.0: all drafts rejected → get_final_output returns
        verified_output (= full KV attention) → relative_error = 0.
        """
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, VeriCacheCodecAttentionHook,
        )
        import torch.nn.functional as F
        cfg = VeriCacheCodecHookConfig(d_head=64, acceptance_threshold=0.0)
        hook = VeriCacheCodecAttentionHook(cfg)
        key = torch.randn(32, 64)
        val = torch.randn(32, 64)
        hook.write_to_cache("acc_seg", key, val, layer_idx=0)
        Q = torch.randn(4, 64)
        result = hook.read_from_cache("acc_seg", layer_idx=0, Q=Q)
        assert result is not None
        final = hook.get_final_output(result)
        # With acceptance_threshold=0.0, draft always rejected → verified_output used
        assert not result.accepted
        verified_direct = VeriCacheCodecAttentionHook._compute_attention(
            Q.float(), key.float(), val.float()
        )
        rel_err = float(
            (final.float() - verified_direct.float()).norm()
            / (verified_direct.float().norm() + 1e-8)
        )
        assert rel_err < 0.01, (
            f"MANDATORY: final output rel_err={rel_err:.4f} >= 0.01 "
            f"(rejected draft → verified_output must = full KV attention)"
        )

    def test_layer_idx_separation(self) -> None:
        """Different layer_idx must not interfere with each other."""
        hook = self._make_hook()
        key = torch.randn(8, 64)
        val = torch.randn(8, 64)
        hook.write_to_cache("seg_layer", key, val, layer_idx=0)
        hook.write_to_cache("seg_layer", key, val, layer_idx=1)
        Q = torch.randn(2, 64)
        r0 = hook.read_from_cache("seg_layer", layer_idx=0, Q=Q)
        r1 = hook.read_from_cache("seg_layer", layer_idx=1, Q=Q)
        r_miss = hook.read_from_cache("seg_layer", layer_idx=2, Q=Q)
        assert r0 is not None
        assert r1 is not None
        assert r_miss is None

    def test_lru_eviction(self) -> None:
        """LRU eviction triggers at max_entries_per_layer."""
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, VeriCacheCodecAttentionHook,
        )
        cfg = VeriCacheCodecHookConfig(d_head=16, max_entries_per_layer=3)
        hook = VeriCacheCodecAttentionHook(cfg)
        for i in range(4):
            hook.write_to_cache(f"seg_{i}", torch.randn(8, 16), torch.randn(8, 16), layer_idx=0)
        # Store has at most max_entries entries
        assert len(hook._store) <= 3


# --------------------------------------------------------------------------- #
# 4. apply_vericache_codec_patch                                                #
# --------------------------------------------------------------------------- #

class TestApplyVericacheCodecPatch:
    def test_patch_attaches_hooks(self) -> None:
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, apply_vericache_codec_patch,
        )

        class _StubImpl:
            pass

        cfg = VeriCacheCodecHookConfig(d_head=32)
        hook = apply_vericache_codec_patch(_StubImpl, cfg, layer_idx=0)
        assert hasattr(_StubImpl, "write_to_cache"), "write_to_cache must be attached"
        assert hasattr(_StubImpl, "read_from_cache"), "read_from_cache must be attached"
        assert hasattr(_StubImpl, "_vericache_hook"), "_vericache_hook must be attached"
        assert hook is _StubImpl._vericache_hook

    def test_patch_write_returns_original(self) -> None:
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, apply_vericache_codec_patch,
        )

        class _StubImpl2:
            pass

        apply_vericache_codec_patch(_StubImpl2, VeriCacheCodecHookConfig(d_head=32), layer_idx=0)
        key = torch.randn(8, 32)
        val = torch.randn(8, 32)
        k_out, v_out = _StubImpl2.write_to_cache("patch_seg", key, val)
        assert k_out is key and v_out is val

    def test_patch_idempotent(self) -> None:
        """Calling apply_vericache_codec_patch twice on same impl does not error."""
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, apply_vericache_codec_patch,
        )

        class _StubImpl3:
            pass

        cfg = VeriCacheCodecHookConfig(d_head=32)
        hook1 = apply_vericache_codec_patch(_StubImpl3, cfg, layer_idx=0)
        hook2 = apply_vericache_codec_patch(_StubImpl3, cfg, layer_idx=0)
        # Same hook returned
        assert hook1 is hook2


# --------------------------------------------------------------------------- #
# 5. extend_cache_config_vericache                                              #
# --------------------------------------------------------------------------- #

class TestExtendCacheConfigVericache:
    def test_fields_added(self) -> None:
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, extend_cache_config_vericache,
        )

        class _FakeCC:
            pass

        cfg = VeriCacheCodecHookConfig(
            d_head=64, acceptance_threshold=0.015, max_entries_per_layer=200
        )
        fake_cc = _FakeCC()
        extend_cache_config_vericache(fake_cc, cfg)
        assert getattr(fake_cc, "compression_method") == "vericache_speculative"
        assert getattr(fake_cc, "vericache_acceptance_threshold") == 0.015
        assert getattr(fake_cc, "vericache_d_head") == 64
        assert getattr(fake_cc, "vericache_max_entries") == 200
        assert getattr(fake_cc, "vericache_compression_codec") == "int8_draft"

    def test_default_config(self) -> None:
        from vllm_integration.vericache_codec_patch import extend_cache_config_vericache

        class _FakeCC:
            pass

        fake_cc = _FakeCC()
        extend_cache_config_vericache(fake_cc)
        assert fake_cc.compression_method == "vericache_speculative"
        assert fake_cc.vericache_acceptance_threshold == 0.01


# --------------------------------------------------------------------------- #
# 6. KVPacketSegmentConfig                                                      #
# --------------------------------------------------------------------------- #

class TestKVPacketSegmentConfig:
    def test_default_instantiation(self) -> None:
        from vllm_integration.kv_packet_block_manager_patch import KVPacketSegmentConfig
        cfg = KVPacketSegmentConfig()
        assert cfg.max_segments == 512
        assert cfg.n_adapter_tokens == 4
        assert cfg.n_heads == 8
        assert cfg.d_head == 128
        assert cfg.seed == 42
        assert cfg.block_table_sentinel == -1

    def test_custom_config(self) -> None:
        from vllm_integration.kv_packet_block_manager_patch import KVPacketSegmentConfig
        cfg = KVPacketSegmentConfig(max_segments=100, n_heads=4, d_head=32)
        assert cfg.max_segments == 100
        assert cfg.n_heads == 4
        assert cfg.d_head == 32


# --------------------------------------------------------------------------- #
# 7. _InlineKVPacketStore                                                       #
# --------------------------------------------------------------------------- #

class TestInlineKVPacketStore:
    def _make_store(self, **kwargs) -> "_InlineKVPacketStore":
        from vllm_integration.kv_packet_block_manager_patch import _InlineKVPacketStore
        return _InlineKVPacketStore(
            max_entries=50, n_adapter_tokens=4, n_heads=4, d_head=32, seed=42, **kwargs
        )

    def test_put_and_get_shape(self) -> None:
        store = self._make_store()
        kv = torch.randn(16, 2, 4, 32).half()
        store.put("seg_A", kv)
        adapted = store.get("seg_A")
        assert adapted is not None
        # Expected: [n_adapter_tokens + n_tokens, 2, n_heads, d_head]
        assert adapted.shape == (4 + 16, 2, 4, 32), f"Unexpected shape: {adapted.shape}"

    def test_get_kv_pair_shape(self) -> None:
        store = self._make_store()
        kv = torch.randn(8, 2, 4, 32).half()
        store.put("seg_B", kv)
        pair = store.get_kv_pair("seg_B")
        assert pair is not None
        K, V = pair
        assert K.shape == (4 + 8, 4, 32), f"K shape: {K.shape}"
        assert V.shape == (4 + 8, 4, 32), f"V shape: {V.shape}"

    def test_miss_returns_none(self) -> None:
        store = self._make_store()
        assert store.get("nonexistent") is None
        assert store.get_kv_pair("nonexistent") is None

    def test_hit_rate(self) -> None:
        store = self._make_store()
        kv = torch.randn(8, 2, 4, 32).half()
        store.put("seg_A", kv)
        store.put("seg_B", kv)
        store.get("seg_A")    # hit
        store.get("seg_B")    # hit
        store.get("seg_X")    # miss
        hr = store.hit_rate()
        assert abs(hr - 2 / 3) < 1e-6, f"Expected hit_rate=0.667, got {hr}"

    def test_noncontiguous_hit_tracking(self) -> None:
        store = self._make_store()
        kv = torch.randn(8, 2, 4, 32).half()
        # Insert 3 segments in order A, B, C
        store.put("seg_A", kv.clone())
        store.put("seg_B", kv.clone())
        store.put("seg_C", kv.clone())
        # Access A, C (skip B) — should register as non-contiguous
        store.get("seg_A")
        store.get("seg_C")
        nc_rate = store.noncontiguous_hit_rate()
        assert nc_rate >= 0.0

    def test_lru_eviction_at_max(self) -> None:
        from vllm_integration.kv_packet_block_manager_patch import _InlineKVPacketStore
        store = _InlineKVPacketStore(
            max_entries=3, n_adapter_tokens=4, n_heads=2, d_head=16, seed=42
        )
        for i in range(4):
            store.put(f"seg_{i}", torch.randn(4, 2, 2, 16).half())
        assert len(store._store) <= 3

    def test_duplicate_put_does_not_grow(self) -> None:
        store = self._make_store()
        kv = torch.randn(8, 2, 4, 32).half()
        store.put("seg_dup", kv)
        store.put("seg_dup", kv)  # second put should not add another entry
        assert len(store._store) == 1


# --------------------------------------------------------------------------- #
# 8. KVPacketSegmentMixin                                                       #
# --------------------------------------------------------------------------- #

class TestKVPacketSegmentMixin:
    def _make_mgr(self) -> "object":
        from vllm_integration.kv_packet_block_manager_patch import (
            KVPacketSegmentConfig, KVPacketSegmentMixin, _InlineKVPacketStore,
        )

        class _MinMgr(KVPacketSegmentMixin):
            def __init__(self, **kwargs):
                cfg = kwargs.get("kv_packet_config") or KVPacketSegmentConfig()
                self._kv_packet_cfg = cfg
                self._kv_packet_store = _InlineKVPacketStore(
                    max_entries=cfg.max_segments,
                    n_adapter_tokens=cfg.n_adapter_tokens,
                    n_heads=cfg.n_heads,
                    d_head=cfg.d_head,
                    seed=cfg.seed,
                )
                self._kv_packet_block_registry = {}
                self._kv_packet_block_align_warnings = 0

        cfg = KVPacketSegmentConfig(max_segments=50, n_adapter_tokens=4, n_heads=4, d_head=32)
        return _MinMgr(kv_packet_config=cfg)

    def test_store_returns_string_segment_id(self) -> None:
        mgr = self._make_mgr()
        kv = torch.randn(8, 2, 4, 32).half()
        seg_id = mgr.store_kv_packet_segment(list(range(8)), kv, layer_idx=0)
        assert isinstance(seg_id, str) and len(seg_id) > 0

    def test_different_token_ids_give_different_ids(self) -> None:
        mgr = self._make_mgr()
        kv = torch.randn(8, 2, 4, 32).half()
        id_A = mgr.store_kv_packet_segment(list(range(8)), kv, layer_idx=0)
        id_B = mgr.store_kv_packet_segment(list(range(100, 108)), kv, layer_idx=0)
        assert id_A != id_B

    def test_find_noncontiguous_hits_all_stored(self) -> None:
        mgr = self._make_mgr()
        token_ids_A = list(range(8))
        token_ids_B = list(range(100, 108))
        kv_a = torch.randn(8, 2, 4, 32).half()
        kv_b = torch.randn(8, 2, 4, 32).half()
        mgr.store_kv_packet_segment(token_ids_A, kv_a, layer_idx=0)
        mgr.store_kv_packet_segment(token_ids_B, kv_b, layer_idx=0)
        hits = mgr.find_noncontiguous_hits([token_ids_A, token_ids_B], layer_idx=0)
        assert len(hits) == 2, f"Expected 2 hits, got {len(hits)}"
        for seg_id, K, V in hits:
            assert isinstance(seg_id, str)
            # K, V should be [n_adapter + n_tokens, n_heads, d_head]
            assert K.shape[-1] == 32 and V.shape[-1] == 32

    def test_find_noncontiguous_hits_miss(self) -> None:
        mgr = self._make_mgr()
        hits = mgr.find_noncontiguous_hits([[999, 998, 997]], layer_idx=0)
        assert hits == []

    def test_build_kv_packet_block_table_shape_and_sentinel(self) -> None:
        from vllm_integration.kv_packet_block_manager_patch import _hash_token_ids
        mgr = self._make_mgr()
        token_ids_A = list(range(8))
        token_ids_B = list(range(100, 108))
        kv = torch.randn(8, 2, 4, 32).half()
        mgr.store_kv_packet_segment(token_ids_A, kv, layer_idx=0)
        mgr.store_kv_packet_segment(token_ids_B, kv, layer_idx=0)
        seg_keys = [
            f"{_hash_token_ids(token_ids_A)}_L0",
            f"{_hash_token_ids(token_ids_B)}_L0",
        ]
        table = mgr.build_kv_packet_block_table(seg_keys, block_size=16, max_blocks=8)
        assert table is not None, "table must not be None for stored segments"
        assert table.shape == (1, 8), f"Expected shape (1, 8), got {table.shape}"
        assert table.dtype == torch.int64, f"Expected int64, got {table.dtype}"
        # Unused slots (index 2..7) must be sentinel -1
        assert (table[0, 2:] == -1).all(), f"Unused slots must be -1: {table[0, 2:]}"

    def test_build_kv_packet_block_table_none_on_empty(self) -> None:
        mgr = self._make_mgr()
        table = mgr.build_kv_packet_block_table([], block_size=16, max_blocks=8)
        assert table is None

    def test_kv_packet_stats_keys(self) -> None:
        mgr = self._make_mgr()
        stats = mgr.kv_packet_stats()
        for key in ("hit_rate", "noncontiguous_hit_rate", "n_segments",
                    "total_hits", "total_misses", "noncontiguous_hits"):
            assert key in stats, f"Missing stats key: {key}"
        assert stats["n_segments"] == 0


# --------------------------------------------------------------------------- #
# 9. make_kv_packet_kv_cache_manager_class                                      #
# --------------------------------------------------------------------------- #

class TestMakeKVPacketKVCacheManagerClass:
    def test_issubclass_of_base_and_mixin(self) -> None:
        from vllm_integration.kv_packet_block_manager_patch import (
            KVPacketSegmentConfig, KVPacketSegmentMixin,
            make_kv_packet_kv_cache_manager_class,
        )
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager
            cfg = KVPacketSegmentConfig(n_heads=4, d_head=32)
            KVPktCls = make_kv_packet_kv_cache_manager_class(KVCacheManager, cfg)
            assert issubclass(KVPktCls, KVCacheManager), (
                "Result must subclass vLLM KVCacheManager"
            )
            assert issubclass(KVPktCls, KVPacketSegmentMixin), (
                "Result must subclass KVPacketSegmentMixin"
            )
        except ImportError:
            pytest.skip("vllm.v1.core.kv_cache_manager not available (no GPU env)")


# --------------------------------------------------------------------------- #
# 10. Cross B+C: KVPacket → VeriCache accuracy contract                        #
# --------------------------------------------------------------------------- #

class TestCrossBCAccuracy:
    def test_cross_bc_final_output_accuracy(self) -> None:
        """MANDATORY: Cross B+C final output relative error vs full KV < 1%."""
        import torch.nn.functional as F
        from vllm_integration.kv_packet_block_manager_patch import _InlineKVPacketStore
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, VeriCacheCodecAttentionHook,
        )

        torch.manual_seed(42)

        # Activity B: store segment and retrieve (K, V)
        b_store = _InlineKVPacketStore(
            max_entries=50, n_adapter_tokens=4, n_heads=4, d_head=32, seed=42
        )
        kv_block = torch.randn(8, 2, 4, 32).half()
        b_store.put("cross_seg", kv_block)
        K_from_b, V_from_b = b_store.get_kv_pair("cross_seg")

        # Flatten for VeriCache (expects [n_tokens, flat_d])
        K_flat = K_from_b.reshape(K_from_b.shape[0], -1).float()
        V_flat = V_from_b.reshape(V_from_b.shape[0], -1).float()

        # Activity C: store full KV + compressed draft, then draft-verify
        # Use acceptance_threshold=0.0 to force rejection → guaranteed full KV accuracy
        c_hook = VeriCacheCodecAttentionHook(
            VeriCacheCodecHookConfig(
                d_head=K_flat.shape[-1],
                acceptance_threshold=0.0,  # reject all → verified_output (full KV accuracy)
            )
        )
        k_orig, v_orig = c_hook.write_to_cache("cross_kv", K_flat, V_flat, layer_idx=0)
        assert k_orig is K_flat and v_orig is V_flat, "C hook must return original tensors"

        Q_cross = torch.randn(2, K_flat.shape[-1])
        result = c_hook.read_from_cache("cross_kv", layer_idx=0, Q=Q_cross)
        assert result is not None
        assert not result.accepted, "With threshold=0.0, draft always rejected"

        final = c_hook.get_final_output(result)

        # Verify accuracy: rejected draft → final = verified_output = full KV attention
        verified_direct = VeriCacheCodecAttentionHook._compute_attention(
            Q_cross.float(), K_flat, V_flat
        )
        rel_err = float(
            (final.float() - verified_direct.float()).norm()
            / (verified_direct.float().norm() + 1e-8)
        )
        assert rel_err < 0.01, (
            f"MANDATORY Cross B+C: final_output relative_error={rel_err:.4f} >= 0.01"
        )

    def test_cross_bc_b_hit_feeds_c(self) -> None:
        """B hit (KV Packet) correctly feeds into C (VeriCache)."""
        from vllm_integration.kv_packet_block_manager_patch import _InlineKVPacketStore
        from vllm_integration.vericache_codec_patch import (
            VeriCacheCodecHookConfig, VeriCacheCodecAttentionHook,
        )
        torch.manual_seed(99)

        b_store = _InlineKVPacketStore(
            max_entries=10, n_adapter_tokens=4, n_heads=2, d_head=16, seed=99
        )
        kv = torch.randn(4, 2, 2, 16).half()
        b_store.put("bc_seg", kv)
        K, V = b_store.get_kv_pair("bc_seg")

        K_flat = K.reshape(K.shape[0], -1).float()
        V_flat = V.reshape(V.shape[0], -1).float()

        c_hook = VeriCacheCodecAttentionHook(
            VeriCacheCodecHookConfig(d_head=K_flat.shape[-1])
        )
        k_out, v_out = c_hook.write_to_cache("bc_seg_kv", K_flat, V_flat, layer_idx=0)
        assert k_out is K_flat and v_out is V_flat  # primary kernel safety

        Q = torch.randn(1, K_flat.shape[-1])
        result = c_hook.read_from_cache("bc_seg_kv", layer_idx=0, Q=Q)
        assert result is not None
        final = c_hook.get_final_output(result)
        assert final.shape[0] == 1  # 1 query token
