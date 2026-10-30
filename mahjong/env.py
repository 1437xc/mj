"""Translate judge events into private arrays at the framework boundary."""

import json
import torch
from tensordict import TensorDict
from rlframework.environment import MultiEnv, MultiEnvStep, AgentStep
from .gamecore.engine import GameCore
from .observation import SeatView
from .features import encode


class Environment(MultiEnv):
    def __init__(self, capacity=None):
        self.core = GameCore()

    def reset(self, config=None):
        self.seed = int((config or {}).get("seed", 0))
        self.commands = []
        self.core.r.seed(self.seed)
        self.views = [SeatView(i) for i in range(4)]
        self.offsets = [0] * 4
        self.core.reset()
        return self._result()

    def _result(self):
        for i in range(4):
            events = self.core.agents[i].events
            for event in events[self.offsets[i] :]:
                self.views[i].request2obs(event)
            self.offsets[i] = len(events)
        result = {}
        active = range(4) if self.core.done else self.core.obs
        for i in active:
            arrays = encode(self.views[i])
            result[str(i)] = AgentStep(
                TensorDict({k: torch.from_numpy(v) for k, v in arrays.items()}, []),
                terminated=self.core.done,
            )
        rewards = [(x / 32.0) for x in (self.core.reward or [0.0] * 4)]
        return MultiEnvStep(
            result, self.core.done, TensorDict({"rewards": torch.tensor(rewards)}, [])
        )

    def step(self, actions):
        if {int(p) for p in actions} != set(self.core.obs):
            raise ValueError("All and only the current decision seats must act")
        commands = {}
        for player, action in actions.items():
            i, index = int(player), int(action.get("action"))
            if index not in self.views[i].valid:
                raise ValueError("Illegal masked Mahjong action")
            commands[self.core.agent_names[i]] = self.views[i].action2response(index)
        self.commands.append(commands)
        self.core.step(commands)
        if (
            self.core.reward
            and min(self.core.reward) == -30
            and max(self.core.reward) == 10
        ):
            raise RuntimeError(
                "Judge rejected a candidate advertised as legal: "
                + json.dumps(
                    {
                        "seed": self.seed,
                        "invalid": self.core.invalid_info,
                        "commands": self.commands,
                    }
                )
            )
        return self._result()
