"""Small structured distributions; action semantics stay out of learners."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def masked_categorical(logits, mask=None):
    """Build a categorical distribution, rejecting empty legal-action sets."""
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError("action mask must match logits shape")
        if not mask.bool().any(-1).all():
            raise ValueError("every active distribution must permit an action")
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
    return torch.distributions.Categorical(logits=logits)


class ConditionalCategorical:
    """A parent categorical with one child active for one parent value.

    Inactive children use the canonical placeholder zero and contribute no
    log-probability. Entropy is exact: H(parent) + P(active) H(child).
    A child may have no legal action only where its parent branch is illegal.
    """

    def __init__(
        self,
        parent_logits,
        child_logits,
        parent_mask=None,
        child_mask=None,
        active_value=0,
        parent_key="intent",
        child_key="direction",
    ):
        if parent_logits.shape[:-1] != child_logits.shape[:-1]:
            raise ValueError("parent and child batch shapes must match")
        if not 0 <= active_value < parent_logits.shape[-1]:
            raise ValueError("active parent value is out of range")
        self.parent = masked_categorical(parent_logits, parent_mask)
        self.active_value, self.parent_key, self.child_key = (
            active_value,
            parent_key,
            child_key,
        )
        self.active_possible = (
            torch.ones_like(parent_logits[..., active_value], dtype=torch.bool)
            if parent_mask is None
            else parent_mask[..., active_value].bool()
        )
        if child_mask is None:
            child_mask = torch.ones_like(child_logits, dtype=torch.bool)
        if child_mask.shape != child_logits.shape:
            raise ValueError("child mask must match child logits")
        if (self.active_possible & ~child_mask.bool().any(-1)).any():
            raise ValueError("an available parent branch needs a legal child action")
        # A finite dummy distribution avoids 0 * NaN for unavailable branches.
        safe_mask = torch.where(
            self.active_possible.unsqueeze(-1),
            child_mask.bool(),
            torch.ones_like(child_mask, dtype=torch.bool),
        )
        safe_logits = torch.where(
            self.active_possible.unsqueeze(-1),
            child_logits,
            torch.zeros_like(child_logits),
        )
        self.child = masked_categorical(safe_logits, safe_mask)

    def sample(self, deterministic=False):
        """Sample both heads and canonicalize the inactive child."""
        parent = self.parent.probs.argmax(-1) if deterministic else self.parent.sample()
        child = self.child.probs.argmax(-1) if deterministic else self.child.sample()
        child = torch.where(parent == self.active_value, child, torch.zeros_like(child))
        return TensorDict(
            {self.parent_key: parent, self.child_key: child}, batch_size=parent.shape
        )

    def log_prob(self, action):
        """Return joint log-probability; inactive placeholders are never evaluated."""
        parent, child = action.get(self.parent_key), action.get(self.child_key)
        active = parent == self.active_value
        safe_child = torch.where(active, child, self.child.probs.argmax(-1))
        return self.parent.log_prob(parent) + torch.where(
            active,
            self.child.log_prob(safe_child),
            torch.zeros_like(parent, dtype=self.parent.logits.dtype),
        )

    def entropy(self):
        """Integrate child entropy over the parent, independently of sampled actions."""
        return (
            self.parent.entropy()
            + self.parent.probs[..., self.active_value] * self.child.entropy()
        )
