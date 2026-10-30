"""Sparse decision collection with explicit agent, episode and behavior boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from rlframework.collector import PolicyRef, RewardResolver
from rlframework.collectors.rewards import _active_reward
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.data import Collection, Trajectory, TrajectoryKey
from rlframework.environment import MultiEnv, MultiEnvStep
from rlframework.policy import PolicyOutput
from rlframework.utils.tensor import clone_to_cpu, stack_records


@dataclass
class _PendingDecision:
    observation: TensorDictBase
    output: PolicyOutput
    state_in: Optional[TensorDictBase]
    reference: PolicyRef
    episode_start: bool
    parameter_versions: tuple
    reward: float = 0.0
    ticks: int = 0


class SparseDecisionCollector:
    """Keep each decision pending until its next observation or actual episode end.

    ``collect`` preserves live episodes across rollout cuts without inventing a
    truncation. Rewards use a decision clock by default; ``env_step`` discounts
    each intervening tick. Lambda is always applied once per decision.
    Pending decisions must close before changing their behavior parameters.
    """

    synchronous = False

    def __init__(
        self,
        environment: MultiEnv,
        router: StaticPolicyRouter,
        reward_resolver: Optional[RewardResolver] = None,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        discount_clock: str = "decision",
        env_id: str = "env_0",
    ) -> None:
        if discount_clock not in ("decision", "env_step"):
            raise ValueError("discount_clock must be decision or env_step")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must lie in [0, 1]")
        self.environment, self.router = environment, router
        self.reward_resolver = reward_resolver or _active_reward
        self.gamma, self.gae_lambda, self.discount_clock = (
            gamma,
            gae_lambda,
            discount_clock,
        )
        self.env_id, self._episode_id, self._segment_id = env_id, -1, 0
        self._current = None
        self._pending, self._states, self._routes = {}, {}, {}
        self._seen, self._dead = set(), set()

    @property
    def has_pending(self) -> bool:
        """Whether unfinished sparse decisions prevent a behavior-weight update."""
        return bool(self._pending)

    def collect_episode(
        self, deterministic: bool = False, config: Optional[Mapping[str, object]] = None
    ) -> Collection:
        """Finish the current episode, or reset once if there is no live episode."""
        return self._collect(None, deterministic, config)

    def collect(
        self,
        min_env_steps: int,
        deterministic: bool = False,
        config: Optional[Mapping[str, object]] = None,
    ) -> Collection:
        """Advance joint steps and retain state at a live cut.

        If config supplies a seed, automatic resets within this call increment
        it once per episode. Other reset options are retained unchanged.
        """
        if min_env_steps <= 0:
            raise ValueError("min_env_steps must be positive")
        return self._collect(min_env_steps, deterministic, config)

    def _reset(self, config):
        self._current = self.environment.reset(config)
        self._episode_id += 1
        self._states.clear()
        self._routes.clear()
        self._seen.clear()
        self._dead.clear()
        if self._current.episode_done:
            raise ValueError("reset must return a live episode")

    def _collect(self, budget, deterministic, config):
        records = {}
        env_steps = agent_steps = completed = 0
        resets = 0
        while True:
            if self._current is None:
                reset_config = None if config is None else dict(config)
                # A step-budget rollout can reset many times. Reusing a fixed
                # seed would silently train on the same episode over and over.
                if reset_config is not None and "seed" in reset_config:
                    reset_config["seed"] = int(reset_config["seed"]) + resets
                self._reset(reset_config)
                resets += 1
            actions = {}
            for agent, step in self._current.agents.items():
                if step.terminated or step.truncated or agent in self._dead:
                    continue
                if agent in self._pending:
                    raise RuntimeError(
                        "agent received a new decision before its prior result"
                    )
                reference = self.router.resolve(agent, step)
                identity = (reference.policy_id, id(reference.policy))
                if agent in self._routes and self._routes[agent] != identity:
                    raise ValueError("policy routing cannot change inside an episode")
                self._routes[agent] = identity
                state = self._states.get(agent)
                if state is None:
                    state = reference.policy.initial_state(step.observation)
                with torch.no_grad():
                    output = reference.policy.act(
                        step.observation, state, deterministic
                    )
                state_in = state
                if state_in is None and output.next_state is not None:
                    raise ValueError("recurrent policies must implement initial_state")
                self._pending[agent] = _PendingDecision(
                    clone_to_cpu(step.observation),
                    output,
                    None if state_in is None else clone_to_cpu(state_in),
                    reference,
                    agent not in self._seen,
                    tuple(p._version for p in reference.policy.parameters()),
                )
                self._seen.add(agent)
                self._states[agent] = output.next_state
                actions[agent] = output.action
            following = self.environment.step(actions)
            env_steps += 1
            agent_steps += len(actions)
            if self.synchronous and not set(actions).issubset(following.agents):
                raise KeyError(
                    "synchronous environments must return a result for every acted agent"
                )
            global_terminated, global_truncated = _episode_flags(following)
            for agent, decision in list(self._pending.items()):
                if decision.parameter_versions != tuple(
                    p._version for p in decision.reference.policy.parameters()
                ):
                    raise RuntimeError(
                        "behavior weights changed while a sparse decision was pending"
                    )
                scale = (
                    self.gamma**decision.ticks
                    if self.discount_clock == "env_step"
                    else 1.0
                )
                decision.reward += scale * float(self.reward_resolver(agent, following))
                decision.ticks += 1
                next_step = following.agents.get(agent)
                if next_step is None and not following.episode_done:
                    continue
                terminated = global_terminated or (
                    next_step is not None and next_step.terminated
                )
                truncated = global_truncated or (
                    next_step is not None and next_step.truncated
                )
                if next_step is None and truncated and not terminated:
                    raise ValueError(
                        "truncated sparse decisions require the agent's true final observation"
                    )
                next_obs = (
                    next_step.observation
                    if next_step is not None
                    else decision.observation
                )
                key = (
                    self._episode_id,
                    agent,
                    decision.reference.policy_id,
                    decision.reference.version,
                )
                records.setdefault(key, []).append(
                    self._transition(decision, next_obs, terminated, truncated)
                )
                del self._pending[agent]
                if terminated or truncated:
                    self._dead.add(agent)
                    self._states.pop(agent, None)
            self._current = following
            if following.episode_done:
                completed += 1
                self._current = None
                self._states.clear()
                if budget is None:
                    break
            if budget is not None and env_steps >= budget:
                break
        trajectories = [
            Trajectory(
                TrajectoryKey(self.env_id, episode, agent, self._segment_id),
                policy_id,
                version,
                stack_records(rows),
            )
            for (episode, agent, policy_id, version), rows in records.items()
        ]
        for trajectory in trajectories:
            # A rollout cut keeps bootstrap discount but ends this GAE trace.
            trajectory.data.get("trace_discount")[-1] = 0.0
        self._segment_id += 1
        return Collection(trajectories, env_steps, agent_steps, completed)

    def _transition(self, decision, next_observation, terminated, truncated):
        discount = self.gamma ** (
            decision.ticks if self.discount_clock == "env_step" else 1
        )
        data = {
            "obs": decision.observation,
            "action": clone_to_cpu(decision.output.action),
            "reward": torch.tensor(decision.reward, dtype=torch.float32),
            "terminated": torch.tensor(bool(terminated)),
            "truncated": torch.tensor(bool(truncated)),
            "episode_start": torch.tensor(decision.episode_start),
            "discount": torch.tensor(
                0.0 if terminated else discount, dtype=torch.float32
            ),
            "trace_discount": torch.tensor(
                0.0 if terminated or truncated else discount * self.gae_lambda
            ),
            "next": {"obs": clone_to_cpu(next_observation)},
            "policy": {
                "version": torch.tensor(decision.reference.version),
                "extras": clone_to_cpu(decision.output.extras),
            },
        }
        if decision.output.next_state is not None:
            data["state_in"] = decision.state_in
            data["state_out"] = clone_to_cpu(decision.output.next_state)
        return TensorDict(data, batch_size=[])


def _episode_flags(step: MultiEnvStep):
    """Individual death is never promoted to global termination in a live game."""
    if not step.episode_done:
        return False, False
    keys = step.shared.keys(True, True)
    truncated = (
        bool(step.shared.get("truncated"))
        if "truncated" in keys
        else any(s.truncated for s in step.agents.values())
    )
    terminated = (
        bool(step.shared.get("terminated")) if "terminated" in keys else not truncated
    )
    if not terminated and not truncated:
        raise ValueError("episode_done requires termination or truncation")
    return terminated, truncated
