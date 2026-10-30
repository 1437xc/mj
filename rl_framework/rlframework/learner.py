"""Batch learner protocol shared by training workflows."""

from __future__ import annotations

from typing import Mapping, Protocol


class BatchLearner(Protocol):
    """Learner port; updates return actual optimizer-step counts with metrics."""

    def update(self, batch) -> Mapping[str, float]:
        """Optimize one processed flat or sequence batch."""
