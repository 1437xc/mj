"""TCP implementation of the replay-store interface."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from rlframework.replay_store import ReplayStore
from rlframework.utils.rpc import TensorDictRpcClient, TensorDictRpcServer


class RemoteReplayStoreClient(ReplayStore):
    """ReplayStore proxy backed by explicit TensorDict RPC operations."""

    def __init__(self, client: TensorDictRpcClient) -> None:
        self._client = client

    def add(self, records: Iterable[TensorDictBase]) -> None:
        for record in records:
            self._client.request("replay.add", record)

    def sample(self, batch_size: int) -> TensorDictBase:
        payload, _ = self._client.request(
            "replay.sample",
            TensorDict({}, batch_size=[]),
            {"batch_size": int(batch_size)},
        )
        return payload

    def __len__(self) -> int:
        _, metadata = self._client.request("replay.size", TensorDict({}, batch_size=[]))
        return int(metadata["size"])


def register_replay_store(server: TensorDictRpcServer, store: ReplayStore) -> None:
    """Expose one replay store through namespaced RPC handlers."""

    def add(
        payload: TensorDictBase, metadata: Mapping[str, Any]
    ) -> Tuple[TensorDictBase, Mapping[str, Any]]:
        store.add(
            [payload]
            if len(payload.batch_size) == 0
            else (payload[i] for i in range(payload.batch_size[0]))
        )
        return TensorDict({}, batch_size=[]), {"size": len(store)}

    def sample(
        payload: TensorDictBase, metadata: Mapping[str, Any]
    ) -> Tuple[TensorDictBase, Mapping[str, Any]]:
        return store.sample(int(metadata["batch_size"])), {"size": len(store)}

    def size(
        payload: TensorDictBase, metadata: Mapping[str, Any]
    ) -> Tuple[TensorDictBase, Mapping[str, Any]]:
        return TensorDict({}, batch_size=[]), {"size": len(store)}

    server.register("replay.add", add)
    server.register("replay.sample", sample)
    server.register("replay.size", size)


__all__ = ["RemoteReplayStoreClient", "register_replay_store"]
