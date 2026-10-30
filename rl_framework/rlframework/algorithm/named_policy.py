"""Categorical actor-critic adapter for models consuming several named arrays."""

import torch
from tensordict import TensorDict

from rlframework.policy import ActionEvaluation, ActorCriticPolicy, PolicyOutput
from rlframework.utils.module import module_device


class NamedCategoricalPolicy(ActorCriticPolicy):
    """A model receives ``input_keys`` tensors and returns logits plus scalar value.

    The model owns feature interpretation. The policy owns masking and action
    probabilities, so the same model remains usable without TensorDict at deploy.
    """

    def __init__(self, module, input_keys=("obs",), mask_key="action_mask"):
        super().__init__()
        self.module, self.input_keys, self.mask_key = (
            module,
            tuple(input_keys),
            mask_key,
        )

    def _forward(self, observation):
        device = module_device(self.module)
        logits, value = self.module(
            *(observation.get(k).to(device).float() for k in self.input_keys)
        )
        mask = observation.get(self.mask_key).to(device).bool()
        if not mask.any(-1).all():
            raise ValueError("An active categorical observation has no legal action")
        return logits.masked_fill(~mask, -1e9), value

    def act(self, observation, state=None, deterministic=False):
        with torch.no_grad():
            logits, value = self._forward(observation.unsqueeze(0))
            distribution = torch.distributions.Categorical(logits=logits[0])
            action = logits[0].argmax() if deterministic else distribution.sample()
            return PolicyOutput(
                TensorDict({"action": action.cpu()}, []),
                TensorDict(
                    {
                        "logp": distribution.log_prob(action).cpu(),
                        "value": value[0].cpu(),
                    },
                    [],
                ),
            )

    def values(self, observation, state=None, episode_start=None):
        # Terminal masks need not contain an action; values do not use the mask.
        device = module_device(self.module)
        return self.module(
            *(observation.get(k).to(device).float() for k in self.input_keys)
        )[1]

    def evaluate_actions(self, observation, actions, state=None, episode_start=None):
        logits, value = self._forward(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        return ActionEvaluation(
            distribution.log_prob(actions.get("action").to(logits.device)),
            distribution.entropy(),
            value,
        )
