"""Fixed-seed play evaluation against uniform legal-action opponents."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .env import Environment
from .policy import make_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--untrained", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu")
    cfg = state["config"]
    torch.manual_seed(cfg["seed"])
    policy = make_policy(cfg["hidden"])
    if not args.untrained:
        policy.load_state_dict(state["policy"])
    policy.eval()
    env = Environment(cfg["capacity"])
    records = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        seat = episode % 4
        controlled = [seat] if __package__ == "mahjong" else [seat, (seat + 2) % 4]
        rng = np.random.RandomState(seed)
        step = env.reset({"seed": seed})
        turns = 0
        while not step.episode_done:
            actions = {}
            for player, decision in step.agents.items():
                if int(player) in controlled:
                    with torch.no_grad():
                        actions[player] = policy.act(
                            decision.observation, deterministic=True
                        ).action
                else:
                    legal = (
                        torch.nonzero(
                            decision.observation["action_mask"], as_tuple=False
                        )
                        .flatten()
                        .numpy()
                    )
                    actions[player] = TensorDict(
                        {"action": torch.tensor(int(rng.choice(legal)))}, []
                    )
            step = env.step(actions)
            turns += 1
            if turns > 5000:
                raise RuntimeError("Evaluation did not terminate")
        rewards = step.shared["rewards"].tolist()
        records.append(
            {
                "seed": seed,
                "controlled_seats": controlled,
                "turns": turns,
                "rewards": rewards,
                "controlled_mean_reward": float(
                    np.mean([rewards[p] for p in controlled])
                ),
            }
        )
    scores = [x["controlled_mean_reward"] for x in records]
    result = {
        "game": __package__,
        "untrained": args.untrained,
        "opponent": "uniform over the same retained legal candidates",
        "episodes": args.episodes,
        "mean_reward": float(np.mean(scores)),
        "positive_reward_episodes": sum(x > 0 for x in scores),
        "zero_reward_episodes": sum(x == 0 for x in scores),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
