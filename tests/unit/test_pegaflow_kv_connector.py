"""Unit tests for PegaFlowKVConnector and MockPegaFlowConnector (Activity A-2)."""

import pytest
import torch

from src.cache.pegaflow_kv_connector import (
    MockPegaFlowConnector,
    PegaFlowConnectorConfig,
    PegaFlowKVConnector,
    create_pegaflow_connector,
)
from src.cache.base import CacheStore


def _default_config(**kwargs) -> PegaFlowConnectorConfig:
    defaults = dict(socket_path="/tmp/pegaflow.sock", async_put=True, timeout_ms=100, use_mock=True, seed=42)
    defaults.update(kwargs)
    return PegaFlowConnectorConfig(**defaults)


def _make_tensor(n: int = 4) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32)


def test_mock_connector_cache_store_interface():
    """MockPegaFlowConnector must implement all CacheStore abstract methods."""
    config = _default_config()
    connector = MockPegaFlowConnector(config)
    assert isinstance(connector, CacheStore)

    # Verify all abstract methods exist and are callable
    assert callable(connector.put)
    assert callable(connector.get)
    assert callable(connector.evict)
    assert callable(connector.hit_rate)
    assert callable(connector.memory_bytes)
    assert callable(connector.reset_stats)


def test_mock_put_get_round_trip():
    config = _default_config()
    connector = MockPegaFlowConnector(config)
    t = _make_tensor()
    connector.put("key1", t)
    result = connector.get("key1")
    assert result is not None
    assert torch.allclose(result, t)


def test_mock_get_returns_none_on_miss():
    config = _default_config()
    connector = MockPegaFlowConnector(config)
    result = connector.get("nonexistent_key")
    assert result is None


def test_mock_hit_rate_tracking():
    config = _default_config()
    connector = MockPegaFlowConnector(config)
    t = _make_tensor()
    connector.put("k1", t)

    # 1 hit, 1 miss
    connector.get("k1")
    connector.get("k_miss")

    rate = connector.hit_rate()
    assert rate == pytest.approx(0.5, abs=1e-6)


def test_mock_evict_frees_memory():
    config = _default_config()
    connector = MockPegaFlowConnector(config)
    t = _make_tensor(16)
    connector.put("k1", t)

    before = connector.memory_bytes()
    freed = connector.evict()
    after = connector.memory_bytes()

    assert freed > 0
    assert after < before


def test_mock_delete_removes_key():
    config = _default_config()
    connector = MockPegaFlowConnector(config)
    t = _make_tensor()
    connector.put("k1", t)
    connector.delete("k1")
    result = connector.get("k1")
    assert result is None


def test_mock_reset_stats_clears_counters():
    config = _default_config()
    connector = MockPegaFlowConnector(config)
    t = _make_tensor()
    connector.put("k1", t)
    connector.get("k1")
    connector.get("miss")

    connector.reset_stats()
    assert connector.hit_rate() == 0.0


def test_connector_factory_uses_mock_when_configured():
    """create_pegaflow_connector returns MockPegaFlowConnector when use_mock=True."""
    config = _default_config(use_mock=True)
    connector = create_pegaflow_connector(config)
    assert isinstance(connector, MockPegaFlowConnector)
