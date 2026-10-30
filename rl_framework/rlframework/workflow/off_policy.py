"""Local replay insertion and off-policy learner scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from rlframework.learner import BatchLearner


@dataclass
class LocalOffPolicyWorkflow:
    """Append per-agent records to replay before the configured learner updates."""

    collector: object
    policy_id: str
    replay: object
    learner: BatchLearner
    batch_size: int
    updates_per_episode: int = 1

    def step(self):
        collected = self.collector.collect_episode()
        records = collected.records(self.policy_id)
        self.replay.add(records)
        metrics = {
            "env_steps": float(collected.env_steps),
            "agent_steps": float(collected.agent_steps),
            "workflow/transitions": float(len(records)),
            "gradient_updates": 0.0,
        }
        if len(self.replay) >= self.batch_size:
            updates = 0.0
            for _ in range(self.updates_per_episode):
                result = self.learner.update(self.replay.sample(self.batch_size))
                updates += result["gradient_updates"]
                metrics.update(result)
            metrics["gradient_updates"] = updates
        return metrics
