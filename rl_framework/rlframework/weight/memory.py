"""Thread-safe in-process versioned policy weight storage."""

from __future__ import annotations

import threading
from typing import Dict

from tensordict.tensordict import TensorDictBase

from rlframework.utils.tensor import clone_to_cpu
from rlframework.weight_store import VersionedWeights, WeightStore


class InMemoryWeightStore(WeightStore):
    """Thread-safe in-process latest-weight store used by local and TCP runtimes."""

    def __init__(self) -> None:
        self._values: Dict[str, VersionedWeights] = {}
        self._lock = threading.Lock()

    def publish(self, policy_id: str, parameters: TensorDictBase) -> VersionedWeights:
        with self._lock:
            previous = self._values.get(policy_id)
            version = 1 if previous is None else previous.version + 1
            value = VersionedWeights(policy_id, version, clone_to_cpu(parameters))
            self._values[policy_id] = value
            return value

    def latest(self, policy_id: str) -> VersionedWeights:
        with self._lock:
            try:
                value = self._values[policy_id]
            except KeyError as error:
                raise KeyError(
                    "no weights published for policy {0!r}".format(policy_id)
                ) from error
            return VersionedWeights(
                value.policy_id, value.version, clone_to_cpu(value.parameters)
            )
