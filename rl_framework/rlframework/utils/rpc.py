"""A small self-hosted TCP runtime for TensorDict request/response roles.

The protocol uses a JSON header plus raw tensor bytes.  It accepts only CPU
TensorDict leaves with well-known Torch dtypes; no arbitrary user objects or
pickle payloads are deserialized across the network.  It is intentionally a
small transport port, not a replacement for a cluster scheduler."""

from __future__ import annotations

import json
import socket
import socketserver
import struct
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from rlframework.data import NestedKey
from rlframework.utils.tensor import clone_to_cpu



_HEADER_SIZE = struct.Struct("!I")


_DTYPE_TO_NUMPY: Mapping[torch.dtype, np.dtype] = {
    torch.float32: np.dtype("float32"),
    torch.float64: np.dtype("float64"),
    torch.int64: np.dtype("int64"),
    torch.int32: np.dtype("int32"),
    torch.int16: np.dtype("int16"),
    torch.int8: np.dtype("int8"),
    torch.uint8: np.dtype("uint8"),
    torch.bool: np.dtype("bool"),
}


_NUMPY_TO_DTYPE: Mapping[str, torch.dtype] = {
    value.name: key for key, value in _DTYPE_TO_NUMPY.items()
}


Handler = Callable[
    [TensorDictBase, Mapping[str, Any]], Tuple[TensorDictBase, Mapping[str, Any]]
]


class TensorDictCodec:
    """Encode tensor leaves into a deterministic JSON-metadata binary payload."""

    @staticmethod
    def encode(batch: TensorDictBase) -> bytes:
        """Encode a TensorDict after detaching it from autograd and moving it to CPU."""
        source = clone_to_cpu(batch)
        leaves: List[Mapping[str, Any]] = []
        chunks: List[bytes] = []
        for key in source.keys(True, True):
            value = source.get(key)
            if value.dtype not in _DTYPE_TO_NUMPY:
                raise TypeError(
                    "remote runtime does not support TensorDict dtype {0}".format(
                        value.dtype
                    )
                )
            array = value.contiguous().numpy()
            leaves.append(
                {
                    "key": list(_key_to_tuple(key)),
                    "dtype": _DTYPE_TO_NUMPY[value.dtype].name,
                    "shape": list(array.shape),
                    "byte_length": int(array.nbytes),
                }
            )
            chunks.append(array.tobytes(order="C"))
        metadata = json.dumps(
            {"batch_size": list(source.batch_size), "leaves": leaves},
            separators=(",", ":"),
        ).encode("utf-8")
        return _HEADER_SIZE.pack(len(metadata)) + metadata + b"".join(chunks)

    @staticmethod
    def decode(payload: bytes) -> TensorDictBase:
        """Decode a trusted protocol payload into a CPU TensorDict with validated lengths."""
        if len(payload) < _HEADER_SIZE.size:
            raise ValueError("remote TensorDict payload is shorter than its header")
        metadata_length = _HEADER_SIZE.unpack(payload[: _HEADER_SIZE.size])[0]
        metadata_end = _HEADER_SIZE.size + metadata_length
        if metadata_end > len(payload):
            raise ValueError("remote TensorDict metadata length exceeds payload length")
        metadata = json.loads(payload[_HEADER_SIZE.size : metadata_end].decode("utf-8"))
        result = TensorDict({}, batch_size=metadata["batch_size"])
        offset = metadata_end
        for leaf in metadata["leaves"]:
            dtype_name = leaf["dtype"]
            if dtype_name not in _NUMPY_TO_DTYPE:
                raise TypeError(
                    "remote TensorDict payload contains unsupported dtype {0}".format(
                        dtype_name
                    )
                )
            byte_length = int(leaf["byte_length"])
            end = offset + byte_length
            if end > len(payload):
                raise ValueError("remote TensorDict leaf exceeds payload length")
            array = np.frombuffer(
                payload[offset:end], dtype=np.dtype(dtype_name)
            ).copy()
            tensor = torch.from_numpy(array.reshape(leaf["shape"]))
            key = tuple(leaf["key"])
            result.set(key[0] if len(key) == 1 else key, tensor)
            offset = end
        if offset != len(payload):
            raise ValueError("remote TensorDict payload contains trailing bytes")
        return result


def _key_to_tuple(key: NestedKey) -> Tuple[str, ...]:
    return (key,) if isinstance(key, str) else key


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    """Read exactly ``size`` bytes or raise if the peer closes the connection."""
    chunks: List[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("remote peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(
    connection: socket.socket, header: Mapping[str, Any], payload: bytes
) -> None:
    """Send one protocol frame with a JSON envelope and raw tensor payload."""
    encoded_header = json.dumps(dict(header), separators=(",", ":")).encode("utf-8")
    connection.sendall(
        _HEADER_SIZE.pack(len(encoded_header))
        + encoded_header
        + _HEADER_SIZE.pack(len(payload))
        + payload
    )


def _recv_frame(connection: socket.socket) -> Tuple[Mapping[str, Any], bytes]:
    """Receive one protocol frame and verify its length boundaries."""
    header_length = _HEADER_SIZE.unpack(_recv_exact(connection, _HEADER_SIZE.size))[0]
    header = json.loads(_recv_exact(connection, header_length).decode("utf-8"))
    payload_length = _HEADER_SIZE.unpack(_recv_exact(connection, _HEADER_SIZE.size))[0]
    return header, _recv_exact(connection, payload_length)


class _RequestHandler(socketserver.BaseRequestHandler):
    """Bridge socketserver's handler shape to ``TensorDictRpcServer``."""

    def handle(self) -> None:
        server = self.server.framework_server
        try:
            header, payload = _recv_frame(self.request)
            method = header["method"]
            handler = server.handlers[method]
            request = TensorDictCodec.decode(payload)
            response, metadata = handler(request, header.get("metadata", {}))
            _send_frame(
                self.request,
                {"ok": True, "metadata": dict(metadata)},
                TensorDictCodec.encode(response),
            )
        except Exception as error:
            _send_frame(
                self.request,
                {"ok": False, "error": "{0}: {1}".format(type(error).__name__, error)},
                b"",
            )


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TensorDictRpcServer:
    """Self-contained TCP server exposing named TensorDict request handlers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.handlers: Dict[str, Handler] = {}
        self._server = _ThreadingServer((host, port), _RequestHandler)
        self._server.framework_server = self
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        """Return the bound endpoint, including an OS-assigned ephemeral port."""
        host, port = self._server.server_address
        return str(host), int(port)

    def register(self, method: str, handler: Handler) -> None:
        """Register exactly one handler for a protocol method name."""
        if method in self.handlers:
            raise ValueError("duplicate TensorDict RPC method {0!r}".format(method))
        self.handlers[method] = handler

    def start(self) -> None:
        """Serve requests on a daemon thread; suitable for a local actor process too."""
        if self._thread is not None:
            raise RuntimeError("TensorDict RPC server is already running")
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="tensordict-rpc", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the server and release its socket."""
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class TensorDictRpcClient:
    """Short-lived-connection client; easy to deploy across process or machine boundaries."""

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        self._address = (host, port)
        self._timeout = timeout

    def request(
        self,
        method: str,
        batch: TensorDictBase,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[TensorDictBase, Mapping[str, Any]]:
        """Call a named handler and return its TensorDict payload plus JSON metadata."""
        with socket.create_connection(
            self._address, timeout=self._timeout
        ) as connection:
            _send_frame(
                connection,
                {"method": method, "metadata": dict(metadata or {})},
                TensorDictCodec.encode(batch),
            )
            header, payload = _recv_frame(connection)
        if not header.get("ok"):
            raise RuntimeError(
                "remote request failed: {0}".format(
                    header.get("error", "unknown error")
                )
            )
        return TensorDictCodec.decode(payload), header.get("metadata", {})
