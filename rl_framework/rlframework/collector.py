"""Collector collaboration contracts; implementations live in ``collectors``.

Collectors expose ``collect``/``collect_episode`` and return a Collection. The
existing implementations use structural contracts rather than a common base
class; PolicyRef identifies a routed policy and its behavior weight version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rlframework.environment import AgentId, MultiEnvStep
from rlframework.policy import Policy



RewardResolver = Callable[[AgentId, MultiEnvStep], float]


@dataclass(frozen=True)
class PolicyRef:
    """A routing target and the version of its current behavior weights."""

    policy_id: str
    policy: Policy
    version: int = 0
