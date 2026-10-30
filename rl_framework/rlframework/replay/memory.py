"""In-process FIFO uniform replay storage."""

from __future__ import annotations

import random
import threading
from collections import deque
from typing import Deque, Iterable

from tensordict.tensordict import TensorDictBase

from rlframework.replay_store import ReplayStore
from rlframework.utils.tensor import clone_to_cpu, stack_records


class InMemoryReplayBuffer(ReplayStore):
    """Correct FIFO uniform replay for local DQN-style workflows."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self._records: Deque[TensorDictBase] = deque(maxlen=capacity)
        self._random = random.Random(seed)
        self._lock = threading.Lock()

    def add(self, records: Iterable[TensorDictBase]) -> None:
        with self._lock:
            for record in records:
                if record.batch_size:
                    raise ValueError(
                        "ReplayStore.add expects unbatched TensorDict records"
                    )
                self._records.append(clone_to_cpu(record))

    def sample(self, batch_size: int) -> TensorDictBase:
        with self._lock:
            if batch_size <= 0:
                raise ValueError("sample batch size must be positive")
            if len(self._records) < batch_size:
                raise ValueError(
                    "replay contains {0} records, need {1}".format(
                        len(self._records), batch_size
                    )
                )
            return stack_records(self._random.sample(list(self._records), batch_size))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
