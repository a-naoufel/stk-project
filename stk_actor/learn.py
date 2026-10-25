"""Train a CPU PPO starter: python -m stk_actor.learn --steps 100000."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import torch
from pystk2_gymnasium import AgentSpec
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .actors import FEATURE_VERSION, NET_ARCH
from .pystk_actor import env_name, get_wrappers, player_name

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track", default=None, help="Omit for random tracks")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-karts", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--resume", type=Path, help="Resume an SB3 .zip checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "stk_actor/pystk_actor.pth")
    args = parser.parse_args()
    if args.steps < 1 or args.n_steps < 2 or args.batch_size < 2:
        parser.error("steps must be positive; n-steps and batch-size must be >= 2")
    if args.n_steps % args.batch_size:
        parser.error("n-steps must be divisible by batch-size")
    torch.set_num_threads(args.threads)
    run = ROOT / "outputs" / datetime.now(timezone.utc).strftime("ppo-%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    env = gym.make(
        env_name, render_mode=None, track=args.track, num_kart=args.num_karts,
        agent=AgentSpec(use_ai=False, name=player_name),
    )
    try:
        for wrapper in get_wrappers():
            env = wrapper(env)
        env = Monitor(env, str(run / "monitor.csv"))
        if args.resume:
            model = PPO.load(args.resume, env=env, device="cpu")
        else:
            model = PPO(
                "MlpPolicy", env, device="cpu", seed=args.seed, verbose=1,
                n_steps=args.n_steps, batch_size=args.batch_size,
                policy_kwargs={"net_arch": NET_ARCH}, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            )
        callback = CheckpointCallback(
            save_freq=10_000, save_path=str(run / "checkpoints"), name_prefix="ppo"
        )
        model.learn(
            total_timesteps=args.steps, callback=callback,
            reset_num_timesteps=not bool(args.resume),
        )
        model.save(run / "ppo.zip")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"feature_version": FEATURE_VERSION, "net_arch": model.policy.net_arch,
             "policy_state_dict": model.policy.state_dict()},
            args.output,
        )
        metadata = {
            "seed": args.seed, "track": args.track, "num_karts": args.num_karts,
            "requested_steps": args.steps, "trained_steps": model.num_timesteps,
            "feature_version": FEATURE_VERSION, "output": str(args.output.resolve()),
            "resume": str(args.resume) if args.resume else None,
            "note": "PPO starter; evaluate before treating this as a competitive agent.",
        }
        (run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Inference weights: {args.output.resolve()}\nRun and resumable model: {run}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
