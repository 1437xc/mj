"""Synchronous whole-episode PPO; one learner process and N CPU actors."""

import argparse
import json
import os
import random
import time
from functools import partial
from pathlib import Path
import numpy as np
import torch
from rlframework.collectors.parallel import ParallelEpisodeCollector
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collector import PolicyRef
from rlframework.algorithm.ppo import PPOLearner, PPOConfig
from rlframework.processing.gae import GAEProcessor
from rlframework.workflow.on_policy import MultiPolicyOnPolicyWorkflow, OnPolicyStack
from .policy import make_policy, make_collector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "configs/default.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--actors", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    for key in ("iterations", "actors"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if any(cfg[k] < 1 for k in ("iterations", "actors", "episodes_per_actor")):
        parser.error("iterations, actors and episodes_per_actor must be positive")
    if cfg["capacity"] < 2 or (args.seconds is not None and args.seconds <= 0):
        parser.error("capacity must be >=2 and seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    policy = make_policy(cfg["hidden"])
    ppo = PPOConfig(**cfg["ppo"])
    learner = PPOLearner(policy, ppo, device=cfg["device"])
    start_iteration = 0
    counters = {
        "env_steps": 0,
        "agent_steps": 0,
        "gradient_updates": 0,
        "completed_episodes": 0,
    }
    episode_cursor = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if (
            state["config"]["hidden"] != cfg["hidden"]
            or state["config"]["capacity"] != cfg["capacity"]
        ):
            raise ValueError("Checkpoint feature/model configuration mismatch")
        policy.load_state_dict(state["policy"])
        learner.optimizer.load_state_dict(state["optimizer"])
        start_iteration = state["iteration"]
        counters = state["counters"]
        episode_cursor = state["episode_cursor"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
    router = StaticPolicyRouter({str(i): PolicyRef("shared", policy) for i in range(4)})
    factory = partial(
        make_collector,
        hidden=cfg["hidden"],
        capacity=cfg["capacity"],
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        seed=cfg["seed"],
    )
    start = time.monotonic()
    with ParallelEpisodeCollector(
        factory,
        router,
        actors=cfg["actors"],
        episodes_per_actor=cfg["episodes_per_actor"],
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        seed=cfg["seed"],
        timeout=cfg["actor_timeout"],
    ) as collector:
        collector.episode_cursor = episode_cursor
        workflow = MultiPolicyOnPolicyWorkflow(
            collector,
            {
                "shared": OnPolicyStack(
                    policy, GAEProcessor(ppo.gamma, ppo.gae_lambda), learner
                )
            },
        )
        with (args.output / "metrics.jsonl").open("w") as log:
            for iteration in range(
                start_iteration + 1, start_iteration + 1 + cfg["iterations"]
            ):
                before = {
                    k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
                }
                metrics = workflow.step()
                changed = any(
                    not torch.equal(before[k], v.detach().cpu())
                    for k, v in policy.state_dict().items()
                )
                if not changed:
                    raise RuntimeError("Training completed without a parameter update")
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("Non-finite training metric")
                for k in counters:
                    counters[k] += int(metrics[k])
                row = dict(
                    metrics,
                    iteration=iteration,
                    elapsed_seconds=time.monotonic() - start,
                    actor_stats=collector.last_actor_stats,
                    actor_pids=collector.actor_pids,
                    learner_pid=os.getpid(),
                    parameters_changed=changed,
                    terminal_rewards=[
                        float(t.data.get("reward").sum())
                        for t in workflow.last_collection.trajectories
                    ],
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "episodes": counters["completed_episodes"],
                            "seconds": row["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
                state = {
                    "policy": policy.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                    "config": cfg,
                    "iteration": iteration,
                    "counters": counters,
                    "episode_cursor": collector.episode_cursor,
                    "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                }
                temporary = args.output / "checkpoint.tmp"
                torch.save(state, temporary)
                temporary.replace(args.output / "checkpoint.pt")
                if args.seconds and time.monotonic() - start >= args.seconds:
                    break
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(counters, iteration=iteration, seconds=time.monotonic() - start),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
