"""Versioned weight payload and publication interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tensordict.tensordict import TensorDictBase


@dataclass(frozen=True)
class VersionedWeights:
    """Immutable parameter payload associated with a monotonically increasing version."""

    policy_id: str
    version: int
    parameters: TensorDictBase


class WeightStore(ABC):
    """Latest-weight port used by actors; it is not a historical league store."""

    @abstractmethod
    def publish(self, policy_id: str, parameters: TensorDictBase) -> VersionedWeights:
        """Publish a new immutable weight version."""

    @abstractmethod
    def latest(self, policy_id: str) -> VersionedWeights:
        """Read the most recently published version for a policy."""
