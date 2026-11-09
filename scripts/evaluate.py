"""Seeded headless evaluation: python -m scripts.evaluate --agent heuristic."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec

from stk_actor.actors import load_policy_actor, make_heuristic_actor
from stk_actor.pystk_actor import create_state, env_name, get_actor, get_wrappers

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["heuristic", "random", "ai", "deployed", "checkpoint"], default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tracks", default="lighthouse", help="Comma-separated track names")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--num-karts", type=int, default=4)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation.json")
    args = parser.parse_args()
    if args.agent == "checkpoint" and not args.checkpoint:
        parser.error("--checkpoint is required for --agent checkpoint")
    if args.max_steps < 1 or args.num_karts < 1 or args.laps < 1:
        parser.error("max-steps, num-karts and laps must be positive")
    if args.difficulty not in {0, 1, 2}:
        parser.error("difficulty must be 0, 1 or 2")
    tracks = [track.strip() for track in args.tracks.split(",")]
    try:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
    except ValueError:
        parser.error("seeds must be comma-separated integers")
    if not all(tracks) or len(set(tracks)) != len(tracks) or len(set(seeds)) != len(seeds):
        parser.error("tracks and seeds must be nonempty and unique")
    checkpoint = args.checkpoint
    if args.agent == "deployed" and (ROOT / "stk_actor/pystk_actor.pth").is_file():
        checkpoint = ROOT / "stk_actor/pystk_actor.pth"
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "agent": args.agent, "tracks": tracks, "seeds": seeds,
        "max_steps": args.max_steps, "num_karts": args.num_karts,
        "laps": args.laps, "difficulty": args.difficulty,
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        "hostname": platform.node(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["pystk2-gymnasium", "gymnasium", "numpy", "torch", "stable-baselines3"]},
        "inference": "deterministic" if args.agent in {"checkpoint", "deployed"} else args.agent,
        "ai_latency_note": "Built-in AI runs inside simulator step; actor latency is not measured.",
    }
    write_json(args.output.with_suffix(".meta.json"), metadata)
    rows = []
    for track in tracks:
        env = gym.make(
            env_name, track=track, num_kart=args.num_karts,
            render_mode=None, max_episode_steps=args.max_steps,
            laps=args.laps, difficulty=args.difficulty,
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
            for seed in seeds:
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
                    "mean_action_ms": float(np.mean(durations) * 1000) if args.agent != "ai" else None,
                    "max_action_ms": worst * 1000 if args.agent != "ai" else None,
                }
                rows.append(row)
                write_json(args.output, rows)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["completed_races"] = len(rows)
    write_json(args.output.with_suffix(".meta.json"), metadata)
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
