"""PyTorch module device and parameter transfer helpers."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn


def module_device(module: nn.Module) -> torch.device:
    """Return a module's execution device, including parameter-free modules.

    Environment adapters deliberately emit CPU ``TensorDict`` objects.  Policies
    use this helper at the actor boundary before passing an observation to a
    model, while the collector subsequently stores the resulting transition on
    CPU.  This keeps environment implementations device-agnostic.
    """
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def module_parameters(module: nn.Module) -> TensorDictBase:
    """Export a module state dict as a CPU TensorDict suitable for a WeightStore."""
    return TensorDict(
        {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.state_dict().items()
        },
        batch_size=[],
    )


def load_module_parameters(module: nn.Module, parameters: TensorDictBase) -> None:
    """Load a WeightStore payload into a module with PyTorch's strict checks."""
    state = {name: value for name, value in parameters.items()}
    module.load_state_dict(state, strict=True)
