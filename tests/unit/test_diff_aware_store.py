"""Unit tests for DiffAwareSegmentStore (Activity B-1).

All tests use CPU tensors. torch.manual_seed(42) for reproducibility.
"""

import importlib
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cache.diff_aware_store import DiffAwareSegmentStore


class TestDiffAwareSegmentStore:
    """DiffAwareSegmentStore correctness and interface tests."""

    def setup_method(self):
        torch.manual_seed(42)

    def _make_store(self, block_size=64, diff_threshold=0.1, max_groups=10):
        return DiffAwareSegmentStore(
            block_size=block_size,
            diff_threshold=diff_threshold,
            max_groups=max_groups,
        )

    def test_register_master_and_get(self):
        """register_master then get('master:g1') must return the same tensor."""
        torch.manual_seed(42)
        store = self._make_store()
        master_kv = torch.randn(4, 64, 64)
        store.register_master(master_kv, "group1")
        result = store.get("master:group1")
        assert result is not None
        # Shape preserved
        assert result.shape == master_kv.shape

    def test_put_agent_kv_identical_returns_no_diff(self):
        """Identical agent_kv == master_kv → no diff blocks stored, all master_ref_blocks."""
        torch.manual_seed(42)
        store = self._make_store(diff_threshold=0.1)
        master_kv = torch.zeros(1, 64, 64)
        store.register_master(master_kv, "g1")
        store.put_agent_kv("a1", "g1", master_kv.clone())
        diff_entry = store._diffs["g1"]["a1"]
        assert len(diff_entry.diff_blocks) == 0, "Identical KV should have no diff blocks"
        assert len(diff_entry.master_ref_blocks) > 0, "All blocks should be master refs"

    def test_put_agent_kv_different_stores_diff(self):
        """Agent KV with L2 > threshold → diff blocks stored for differing blocks."""
        torch.manual_seed(42)
        store = self._make_store(diff_threshold=0.01)
        master_kv = torch.zeros(1, 64, 64)
        # Create agent with large difference
        agent_kv = torch.ones(1, 64, 64) * 10.0
        store.register_master(master_kv, "g1")
        store.put_agent_kv("a1", "g1", agent_kv)
        diff_entry = store._diffs["g1"]["a1"]
        assert len(diff_entry.diff_blocks) > 0, "Large difference should have diff blocks"

    def test_get_agent_kv_restores_original(self):
        """register_master + put_agent_kv + get_agent_kv → torch.allclose(original, restored)."""
        torch.manual_seed(42)
        store = self._make_store(diff_threshold=0.001)  # very low threshold → all diffs stored
        master_kv = torch.randn(1, 64, 64)
        agent_kv = torch.randn(1, 64, 64)
        store.register_master(master_kv, "g1")
        store.put_agent_kv("a1", "g1", agent_kv)
        restored = store.get_agent_kv("a1", "g1")
        assert restored is not None
        # atol=0.003 accounts for FP16 roundtrip quantization error in diff storage
        assert torch.allclose(agent_kv.half(), restored, atol=0.003), (
            f"Max diff: {(agent_kv.half() - restored).abs().max().item()}"
        )

    def test_evict_removes_group_and_all_agents(self):
        """Evicting a group removes master and all its agent diffs."""
        torch.manual_seed(42)
        store = self._make_store(max_groups=5)
        kv = torch.randn(1, 64, 64)
        store.register_master(kv, "g1")
        store.put_agent_kv("a1", "g1", kv.clone())
        store.put_agent_kv("a2", "g1", kv.clone())

        assert "g1" in store._masters
        assert "g1" in store._diffs

        store._evict_group("g1")

        assert "g1" not in store._masters
        assert "g1" not in store._diffs

    def test_diff_threshold_boundary(self):
        """L2 dist <= threshold → master ref; L2 dist > threshold → diff stored."""
        torch.manual_seed(42)
        store = self._make_store(block_size=64, diff_threshold=1.0)
        master_kv = torch.zeros(1, 64, 64)

        # Agent with exactly zero difference (L2 = 0 <= 1.0) → master ref
        agent_identical = torch.zeros(1, 64, 64)
        store.register_master(master_kv, "g_same")
        store.put_agent_kv("a_same", "g_same", agent_identical)
        diff_entry_same = store._diffs["g_same"]["a_same"]
        assert len(diff_entry_same.diff_blocks) == 0

        # Agent with large difference (L2 >> 1.0) → diff blocks
        store2 = self._make_store(block_size=64, diff_threshold=1.0)
        store2.register_master(master_kv, "g_diff")
        agent_diff = torch.ones(1, 64, 64) * 100.0  # L2 >> 1.0
        store2.put_agent_kv("a_diff", "g_diff", agent_diff)
        diff_entry_diff = store2._diffs["g_diff"]["a_diff"]
        assert len(diff_entry_diff.diff_blocks) > 0

    def test_max_groups_triggers_eviction(self):
        """max_groups=2: registering a 3rd group evicts the oldest."""
        torch.manual_seed(42)
        store = self._make_store(max_groups=2)
        kv = torch.randn(1, 32, 64)
        store.register_master(kv, "g1")
        store.register_master(kv, "g2")
        assert len(store._masters) == 2

        store.register_master(kv, "g3")
        assert len(store._masters) == 2
        assert "g1" not in store._masters, "Oldest group g1 should have been evicted"
        assert "g3" in store._masters

    def test_lru_order_on_access(self):
        """Accessing an old group via get() updates LRU order."""
        torch.manual_seed(42)
        store = self._make_store(max_groups=2)
        kv = torch.randn(1, 32, 64)
        store.register_master(kv, "g1")
        store.register_master(kv, "g2")

        # Access g1 to make it most recently used
        _ = store.get("master:g1")

        # Registering g3 should evict g2 (now oldest), not g1
        store.register_master(kv, "g3")
        assert "g1" in store._masters, "g1 (recently accessed) should NOT be evicted"
        assert "g2" not in store._masters, "g2 (least recently used) should be evicted"

    def test_diff_hit_stats_structure(self):
        """diff_hit_stats() must return dict with required keys."""
        torch.manual_seed(42)
        store = self._make_store()
        kv = torch.randn(1, 64, 64)
        store.register_master(kv, "g1")
        store.get("master:g1")

        stats = store.diff_hit_stats()
        required_keys = {"diff_hit_rate", "master_hit_rate", "overall_hit_rate", "n_groups"}
        assert required_keys.issubset(stats.keys()), (
            f"Missing keys: {required_keys - stats.keys()}"
        )

    def test_faiss_not_imported(self):
        """diff_aware_store.py must not import faiss."""
        module_file = Path(__file__).parent.parent.parent / "src" / "cache" / "diff_aware_store.py"
        content = module_file.read_text()
        assert "faiss" not in content, "faiss import found in diff_aware_store.py"

    def test_put_get_cachestore_interface(self):
        """put('agent:g1:a1', kv) → get('agent:g1:a1') must work via CacheStore interface."""
        torch.manual_seed(42)
        store = self._make_store(diff_threshold=0.001)
        master_kv = torch.randn(1, 64, 64)
        agent_kv = torch.randn(1, 64, 64)

        # Register master via put interface
        store.put("master:g1", master_kv)
        # Store agent via put interface
        store.put("agent:g1:a1", agent_kv)
        # Retrieve via get interface
        result = store.get("agent:g1:a1")
        assert result is not None
        assert result.shape == agent_kv.shape

    def test_memory_bytes_accounts_diff_blocks(self):
        """memory_bytes() must increase after storing agent diff blocks."""
        torch.manual_seed(42)
        store = self._make_store(diff_threshold=0.001)  # force all diffs to be stored
        master_kv = torch.zeros(1, 64, 64)
        agent_kv = torch.ones(1, 64, 64) * 10.0  # large difference

        store.register_master(master_kv, "g1")
        baseline_bytes = store.memory_bytes()

        store.put_agent_kv("a1", "g1", agent_kv)
        after_bytes = store.memory_bytes()

        assert after_bytes > baseline_bytes, (
            f"memory_bytes {after_bytes} should be > {baseline_bytes} after diff storage"
        )

    def test_reset_stats(self):
        """reset_stats() must zero all hit/miss counters."""
        torch.manual_seed(42)
        store = self._make_store()
        kv = torch.randn(1, 32, 64)
        store.register_master(kv, "g1")
        store.get("master:g1")   # generates a hit
        store.get("master:g99")  # generates a miss

        store.reset_stats()
        assert store.hit_rate() == 0.0
        assert store._diff_hits == 0
        assert store._master_hits == 0
        assert store._hits == 0
        assert store._misses == 0

    def test_cachestore_interface_all_methods(self):
        """All 6 CacheStore abstract methods must be callable without error."""
        torch.manual_seed(42)
        store = self._make_store()
        kv = torch.randn(1, 32, 64)

        # put
        store.put("master:g1", kv)
        # get
        result = store.get("master:g1")
        assert result is not None
        # hit_rate
        rate = store.hit_rate()
        assert isinstance(rate, float)
        # memory_bytes
        mem = store.memory_bytes()
        assert isinstance(mem, int)
        # evict
        freed = store.evict()
        assert isinstance(freed, int)
        # reset_stats
        store.reset_stats()
