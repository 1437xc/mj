"""Replay storage interface, independent of storage and transport."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from tensordict.tensordict import TensorDictBase


class ReplayStore(ABC):
    """Narrow replay interface suitable for local and future remote stores."""

    @abstractmethod
    def add(self, records: Iterable[TensorDictBase]) -> None:
        """Append unbatched transition records."""

    @abstractmethod
    def sample(self, batch_size: int) -> TensorDictBase:
        """Draw a batch of independent transition records."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the currently retained transition count."""
