"""Generic named-array adapters, independent of any feature language."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from rlframework.data import NestedKey, TensorDictSpec
from rlframework.utils.tensor import as_tensordict


class FeatureAdapter:
    @property
    def spec(self) -> TensorDictSpec:
        raise NotImplementedError

    def encode(self, raw_observation: Any) -> TensorDictBase:
        raise NotImplementedError


class ArrayFeatureAdapter(FeatureAdapter):
    """Validate an already encoded mapping/TensorDict at the environment boundary."""

    def __init__(self, spec: TensorDictSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> TensorDictSpec:
        return self._spec

    def encode(self, observation: Any) -> TensorDictBase:
        result = as_tensordict(observation, device=torch.device("cpu"))
        self.spec.validate(result)
        return result


class CallableFeatureAdapter(FeatureAdapter):
    """Wrap manual/generated encoders with the same named CPU array contract."""

    def __init__(
        self, encoders: Mapping[NestedKey, Callable], spec: TensorDictSpec
    ) -> None:
        if set(encoders) != set(spec.fields):
            raise ValueError("each declared field requires exactly one encoder")
        self._encoders = dict(encoders)
        self._spec = spec

    @property
    def spec(self) -> TensorDictSpec:
        return self._spec

    def encode(self, observation: Any) -> TensorDictBase:
        result = TensorDict({}, batch_size=[])
        for key, encoder in self._encoders.items():
            field = self.spec.fields[key]
            result.set(
                key,
                torch.as_tensor(encoder(observation), dtype=field.dtype, device="cpu"),
            )
        self.spec.validate(result)
        return result
