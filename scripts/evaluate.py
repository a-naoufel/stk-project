"""Seeded headless evaluation: python -m scripts.evaluate --agent heuristic."""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec

from stk_actor.actors import load_policy_actor, make_heuristic_actor
from stk_actor.pystk_actor import create_state, env_name, get_actor, get_wrappers

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["heuristic", "random", "ai", "deployed", "checkpoint"], default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tracks", default="lighthouse", help="Comma-separated track names")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--num-karts", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation.json")
    args = parser.parse_args()
    if args.agent == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required for --agent checkpoint")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    rows = []
    for track in args.tracks.split(","):
        env = gym.make(
            env_name, track=track.strip(), num_kart=args.num_karts,
            render_mode=None, max_episode_steps=args.max_steps,
            agent=AgentSpec(use_ai=args.agent == "ai"),
        )
        try:
            for wrapper in get_wrappers():
                env = wrapper(env)
            if args.agent == "checkpoint":
                actor = load_policy_actor(args.checkpoint, env.observation_space, env.action_space)
            elif args.agent == "deployed":
                actor = get_actor(ROOT / "stk_actor", env.observation_space, env.action_space)
            elif args.agent == "heuristic":
                actor = make_heuristic_actor()
            else:
                actor = lambda state, obs: env.action_space.sample()
            for seed in map(int, args.seeds.split(",")):
                env.action_space.seed(seed)
                obs, _ = env.reset(seed=seed)
                state = create_state()
                total, worst, durations = 0.0, 0.0, []
                for step in range(1, args.max_steps + 1):
                    start = time.perf_counter()
                    action = actor(state, obs)
                    elapsed = time.perf_counter() - start
                    durations.append(elapsed)
                    worst = max(worst, elapsed)
                    if not env.action_space.contains(action):
                        raise ValueError(f"Invalid action: {action}")
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        break
                row = {
                    "agent": args.agent, "track": track, "seed": seed,
                    "steps": step, "finished": bool(terminated),
                    "truncated": bool(truncated), "return": total,
                    "position": int(info["position"]), "distance": float(info["distance"]),
                    "mean_action_ms": float(np.mean(durations) * 1000),
                    "max_action_ms": worst * 1000,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
