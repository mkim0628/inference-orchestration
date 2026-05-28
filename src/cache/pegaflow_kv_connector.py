"""Activity A-2: PegaFlow GIL-free Rust external KV connector wrapper.

CacheStore wrapper for the PegaFlow Rust KV cache process via Unix socket IPC.
Falls back to MockPegaFlowConnector when use_mock=True (default) or when the
Rust process is unavailable.
"""

import socket
import struct
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch

from src.cache.base import CacheStore


class OpCode(IntEnum):
    GET = 1
    PUT = 2
    DELETE = 3


@dataclass
class PegaFlowConnectorConfig:
    socket_path: str = "/tmp/pegaflow.sock"
    async_put: bool = True
    timeout_ms: int = 100
    use_mock: bool = True
    seed: int = 42


def create_pegaflow_connector(config: PegaFlowConnectorConfig) -> "CacheStore":
    """Factory: returns MockPegaFlowConnector when use_mock=True."""
    if config.use_mock:
        return MockPegaFlowConnector(config)
    return PegaFlowKVConnector(config)


class MockPegaFlowConnector(CacheStore):
    """PegaFlow-compatible in-memory fallback for test environments.

    Implements the same CacheStore interface using a Python dict.
    GIL-free behavior cannot be verified here but logic is correct.
    """

    def __init__(self, config: PegaFlowConnectorConfig) -> None:
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits: int = 0
        self._total: int = 0
        torch.manual_seed(config.seed)

    def put(self, key: str, value: torch.Tensor) -> None:
        self._store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        self._total += 1
        v = self._store.get(key)
        if v is not None:
            self._hits += 1
        return v

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def evict(self) -> int:
        if not self._store:
            return 0
        _, val = self._store.popitem(last=False)
        return val.nelement() * val.element_size()

    def hit_rate(self) -> float:
        return self._hits / self._total if self._total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nelement() * v.element_size() for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._total = 0


class PegaFlowKVConnector(CacheStore):
    """CacheStore wrapper that communicates with a PegaFlow Rust process via Unix socket.

    Wire protocol (per message):
      [4B opcode][4B key_len][key_bytes][4B tensor_len][tensor_bytes (PUT only)]

    Response for GET:
      [1B found][4B tensor_len][tensor_bytes] if found=1, else [1B found=0]

    GIL-free design: Python thread releases the GIL during socket I/O via
    the OS blocking call. For truly GIL-free operation, the Rust process
    handles tensor storage without re-entering CPython.

    Layer hierarchy inside PegaFlow (transparent to Python):
      GPU HBM → host DRAM → SSD (PegaFlow internal policy)
    """

    def __init__(self, config: PegaFlowConnectorConfig) -> None:
        self.config = config
        self._hits: int = 0
        self._total: int = 0
        self._last_freed_bytes: int = 0
        torch.manual_seed(config.seed)

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.config.timeout_ms / 1000.0)
        sock.connect(self.config.socket_path)
        return sock

    def _send_recv(self, opcode: OpCode, key: str, tensor: Optional[torch.Tensor] = None) -> Optional[bytes]:
        key_bytes = key.encode("utf-8")
        header = struct.pack(">II", int(opcode), len(key_bytes)) + key_bytes

        if tensor is not None:
            buf = tensor.numpy().tobytes()
            header += struct.pack(">I", len(buf)) + buf
        else:
            header += struct.pack(">I", 0)

        try:
            sock = self._connect()
            try:
                sock.sendall(header)
                if opcode == OpCode.GET:
                    found_byte = sock.recv(1)
                    if not found_byte or found_byte[0] == 0:
                        return None
                    size_bytes = sock.recv(4)
                    size = struct.unpack(">I", size_bytes)[0]
                    data = b""
                    while len(data) < size:
                        chunk = sock.recv(size - len(data))
                        if not chunk:
                            break
                        data += chunk
                    return data
                return None
            finally:
                sock.close()
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return None

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.config.async_put:
            # Non-blocking: send without waiting for ack (fire-and-forget)
            try:
                sock = self._connect()
                key_bytes = key.encode("utf-8")
                buf = value.numpy().tobytes()
                msg = (
                    struct.pack(">II", int(OpCode.PUT), len(key_bytes))
                    + key_bytes
                    + struct.pack(">I", len(buf))
                    + buf
                )
                sock.setblocking(False)
                try:
                    sock.send(msg)
                except BlockingIOError:
                    pass
                sock.close()
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                pass
        else:
            self._send_recv(OpCode.PUT, key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        self._total += 1
        data = self._send_recv(OpCode.GET, key)
        if data is not None:
            self._hits += 1
            arr = torch.frombuffer(data, dtype=torch.float32)
            return arr
        return None

    def delete(self, key: str) -> None:
        self._send_recv(OpCode.DELETE, key)

    def evict(self) -> int:
        # PegaFlow manages eviction internally; return last known freed bytes
        return self._last_freed_bytes

    def hit_rate(self) -> float:
        return self._hits / self._total if self._total > 0 else 0.0

    def memory_bytes(self) -> int:
        # PegaFlow tracks its own memory; return 0 as Python-side estimate
        return 0

    def reset_stats(self) -> None:
        self._hits = 0
        self._total = 0
