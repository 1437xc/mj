"""TCP implementation of the weight-store interface."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from rlframework.utils.rpc import TensorDictRpcClient, TensorDictRpcServer
from rlframework.utils.tensor import clone_to_cpu
from rlframework.weight_store import VersionedWeights, WeightStore


class RemoteWeightStoreClient(WeightStore):
    """WeightStore proxy backed by namespaced TensorDict RPC operations."""

    def __init__(self, client: TensorDictRpcClient) -> None:
        self._client = client

    def publish(self, policy_id: str, parameters: TensorDictBase) -> VersionedWeights:
        _, metadata = self._client.request(
            "weights.publish", parameters, {"policy_id": policy_id}
        )
        return VersionedWeights(
            policy_id, int(metadata["version"]), clone_to_cpu(parameters)
        )

    def latest(self, policy_id: str) -> VersionedWeights:
        payload, metadata = self._client.request(
            "weights.latest", TensorDict({}, batch_size=[]), {"policy_id": policy_id}
        )
        return VersionedWeights(policy_id, int(metadata["version"]), payload)


def register_weight_store(server: TensorDictRpcServer, store: WeightStore) -> None:
    """Expose one weight store through namespaced RPC handlers."""

    def publish(
        payload: TensorDictBase, metadata: Mapping[str, Any]
    ) -> Tuple[TensorDictBase, Mapping[str, Any]]:
        value = store.publish(str(metadata["policy_id"]), payload)
        return TensorDict({}, batch_size=[]), {"version": value.version}

    def latest(
        payload: TensorDictBase, metadata: Mapping[str, Any]
    ) -> Tuple[TensorDictBase, Mapping[str, Any]]:
        value = store.latest(str(metadata["policy_id"]))
        return value.parameters, {"version": value.version}

    server.register("weights.publish", publish)
    server.register("weights.latest", latest)


__all__ = ["RemoteWeightStoreClient", "register_weight_store"]
