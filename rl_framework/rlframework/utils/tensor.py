"""TensorDict conversion, copying, stacking and specification helpers."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

from rlframework.data import FieldSpec, NestedKey, TensorDictSpec


def _tensorize(value: Any, device: Optional[torch.device] = None) -> Any:
    """Recursively convert leaves to tensors without adding a batch dimension."""
    if isinstance(value, TensorDictBase):
        return value.to(device) if device is not None else value
    if isinstance(value, MappingABC):
        return {key: _tensorize(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device) if device is not None else value
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(device) if device is not None else tensor


def as_tensordict(
    value: Union[TensorDictBase, Mapping[str, Any]],
    batch_size: Sequence[int] = (),
    device: Optional[torch.device] = None,
) -> TensorDictBase:
    """Return ``value`` as a TensorDict with the supplied explicit batch shape."""
    if isinstance(value, TensorDictBase):
        result = value
        if tuple(result.batch_size) != tuple(batch_size):
            raise ValueError(
                "TensorDict batch size does not match the requested batch size"
            )
        return result.to(device) if device is not None else result
    if not isinstance(value, MappingABC):
        raise TypeError(
            "a TensorDict boundary requires a mapping, not {0}".format(
                type(value).__name__
            )
        )
    return TensorDict(
        _tensorize(value, device), batch_size=list(batch_size), device=device
    )


def clone_to_cpu(batch: TensorDictBase) -> TensorDictBase:
    """Detach a batch from autograd and copy every leaf to contiguous CPU memory."""
    clone = batch.detach().clone().to("cpu")
    return clone.contiguous()


def stack_records(records: Iterable[TensorDictBase]) -> TensorDictBase:
    """Stack unbatched transition records into a concrete batch TensorDict."""
    items = list(records)
    if not items:
        raise ValueError("cannot stack an empty record sequence")
    return torch.stack(items, dim=0).to_tensordict()


def scalar_tensor(value: Union[bool, float, int], dtype: torch.dtype) -> torch.Tensor:
    """Create a CPU scalar tensor with an intentionally specified dtype."""
    return torch.tensor(value, dtype=dtype)


def flat_observation_spec(
    feature_sizes: Mapping[NestedKey, int], schema_id: str = "unspecified"
) -> TensorDictSpec:
    """Build the common float-vector observation spec from FDL rule lengths."""
    return TensorDictSpec(
        fields={
            key: FieldSpec((length,), torch.float32, "observation")
            for key, length in feature_sizes.items()
        },
        schema_id=schema_id,
    )
