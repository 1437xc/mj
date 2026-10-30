"""Synchronous step collection using the shared decision lifecycle."""

from __future__ import annotations

from rlframework.collectors.sparse import SparseDecisionCollector


class SynchronousStepCollector(SparseDecisionCollector):
    """Require a result for each acted agent while keeping trajectories separate."""

    synchronous = True
