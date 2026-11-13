"""Generate expert trajectories and train DAgger or SAC with symmetric replay."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from scripts.evaluate import write_json
from scripts.study_env import (
    EVAL_OFFSETS_M, EVAL_SEED, EXPERT_COMMIT, MAX_STEPS, STEP_SIZE, TRACK,
    TRAIN_OFFSET_M, evaluate_policy, expert_action, expert_factory, lap_result, make_trial,
)
from stk_actor.actors import FEATURE_DIM, FEATURE_VERSION, NET_ARCH

ROOT = Path(__file__).resolve().parents[1]
DEMO_SEED = 12_300
DEMO_EPISODES = 32
ROUNDS = 8
BC_UPDATES = 2048
DAGGER_UPDATES = 384
BATCH_SIZE = 128
BC_LR = 1e-3
SAC_SETTINGS = {
    "learning_rate": 1e-4, "gamma": 0.995, "tau": 0.005, "ent_coef": 0.02,
    "learning_starts": 0, "train_freq": 4, "gradient_steps": 1,
}
FIELDS = ("observations", "actions", "next_observations", "dones", "rewards")


class SymmetricReplay(ReplayBuffer):
    """Draw a fixed share of each batch from demonstrations, with replacement."""

    def __init__(self, *args, offline_percent=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.offline_percent = offline_percent
        self.offline_draws = self.online_draws = 0

    def sample(self, batch_size, env=None):
        offline_count, remainder = divmod(batch_size * self.offline_percent, 100)
        if remainder:
            raise ValueError("Batch size cannot represent the requested replay split")
        online_count = batch_size - offline_count
        online = super().sample(online_count, env) if online_count else None
        indices = np.random.randint(len(self.offline["rewards"]), size=offline_count)
        offline = [self.to_torch(self.offline[key][indices]) for key in FIELDS]
        self.offline_draws += offline_count
        self.online_draws += online_count
        if online is None:
            return ReplayBufferSamples(*offline)
        return ReplayBufferSamples(*(torch.cat((online_data, offline_data))
                                     for online_data, offline_data in zip(online[:5], offline, strict=True)))


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "pysupertuxkart2", "pystk2-gymnasium", "torch", "stable-baselines3")
    }}


def trial_options(args):
    return {"track": args.track, "max_steps": args.max_steps, "eval_offsets": args.eval_offsets,
            "eval_seed": getattr(args, "eval_seed", EVAL_SEED), "observe_track_items": args.observe_track_items}


def generate(args, make_expert):
    args.output.mkdir(parents=True)
    env = make_trial(DEMO_SEED, **trial_options(args))
    rng = np.random.default_rng(DEMO_SEED)
    data = {key: [] for key in FIELDS}
    episodes = []
    try:
        for index in range(args.demo_transitions or DEMO_EPISODES):
            seed = DEMO_SEED + index
            offset = float(rng.uniform(-TRAIN_OFFSET_M, TRAIN_OFFSET_M))
            obs, _ = env.reset(seed=seed, options={"offset_m": offset})
            expert = make_expert(env.unwrapped)
            for step in range(1, args.max_steps + 1):
                action = expert_action(expert, env.unwrapped)
                new_obs, reward, finished, truncated, _ = env.step(action)
                for key, value in zip(FIELDS, (obs, action, new_obs, [finished], [reward])):
                    data[key].append(value)
                obs = new_obs
                if finished or truncated:
                    break
            episodes.append({"seed": seed, "offset_m": offset, **lap_result(env, finished, step)})
            if args.demo_transitions and len(data["rewards"]) >= args.demo_transitions:
                break
        path = args.output / "expert.npz"
        arrays = {key: np.asarray(value[:args.demo_transitions], dtype=np.float32)
                  for key, value in data.items()}
        np.savez_compressed(path, **arrays)
        test_laps = evaluate_policy(env, None, make_expert)
        write_json(args.output / "expert.json", {
            "commit": EXPERT_COMMIT, "agent": f"team{args.expert_team}", "track": args.track,
            "dataset_sha256": fingerprint(path), "transitions": len(arrays["rewards"]),
            "collected_transitions": len(data["rewards"]), "max_steps": args.max_steps,
            "episodes": episodes, "evaluation": test_laps, "runtime": runtime(),
        })
        completed = sum(row["finished"] for row in episodes)
        print(f"Expert dataset: {len(arrays['rewards'])} transitions, {completed}/{len(episodes)} completed", flush=True)
    finally:
        env.close()


def fit_actions(actor, optimizer, obs, actions, rng, updates):
    for _ in range(updates):
        indices = rng.integers(len(obs), size=BATCH_SIZE)
        prediction = actor(torch.from_numpy(obs[indices]), deterministic=True)
        loss = (prediction - torch.from_numpy(actions[indices])).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def train_dagger(env, model, dataset, make_expert, optimizer, rng, steps, beta_decay, updates):
    observations = np.empty((len(dataset["observations"]) + steps, FEATURE_DIM), dtype=np.float32)
    actions = np.empty((len(observations), 2), dtype=np.float32)
    size = len(dataset["observations"])
    observations[:size] = dataset["observations"]
    actions[:size] = dataset["actions"]
    obs, _ = env.reset()
    expert = make_expert(env.unwrapped)
    visits = []
    for round_index in range(ROUNDS):
        beta = beta_decay ** (round_index + 1)
        interventions = 0
        for _ in range(steps // ROUNDS):
            label = expert_action(expert, env.unwrapped)
            observations[size], actions[size] = obs, label
            size += 1
            use_expert = rng.random() < beta
            action = label if use_expert else model.predict(obs, deterministic=True)[0]
            interventions += int(use_expert)
            obs, _, finished, truncated, _ = env.step(action)
            if finished or truncated:
                obs, _ = env.reset()
                expert = make_expert(env.unwrapped)
        loss = fit_actions(model.actor, optimizer, observations[:size], actions[:size], rng, updates)
        visits.append({"round": round_index + 1, "beta": beta, "expert_actions": interventions,
                       "labels": size - len(dataset["observations"]), "last_mse": loss})
        print(json.dumps(visits[-1]), flush=True)
    return {"expert_queries": steps, "gradient_updates": ROUNDS * updates, "rounds": visits}


def train(args, make_expert):
    args.output.mkdir(parents=True)
    with np.load(args.dataset, allow_pickle=False) as arrays:
        dataset = {key: arrays[key] for key in FIELDS}
    dataset_sha = fingerprint(args.dataset)
    env = make_trial(args.seed, **trial_options(args))
    sac_settings = {**SAC_SETTINGS, "learning_rate": args.sac_learning_rate,
                    "ent_coef": args.sac_ent_coef, "gradient_steps": args.sac_gradient_steps}
    offline_percent = args.offline_percent if args.method == "symmetric" else 0
    start = time.perf_counter()
    try:
        model = SAC(
            "MlpPolicy", env, device="cpu", seed=args.seed, verbose=0,
            policy_kwargs={"net_arch": NET_ARCH}, batch_size=BATCH_SIZE,
            buffer_size=args.steps,
            replay_buffer_class=SymmetricReplay if offline_percent else ReplayBuffer,
            replay_buffer_kwargs={"offline_percent": offline_percent} if offline_percent else None,
            **sac_settings,
        )
        # All methods share the same supervised starting policy and exploration scale.
        with torch.no_grad():
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(-2.0)
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=BC_LR)
        bc_rng = np.random.default_rng(args.seed)
        loss = fit_actions(model.actor, optimizer, dataset["observations"], dataset["actions"],
                           bc_rng, args.bc_updates)
        digest = sha256()
        for parameter in model.actor.state_dict().values():
            digest.update(parameter.detach().cpu().numpy().tobytes())
        initial_actor_sha = digest.hexdigest()
        predict = lambda obs: model.predict(obs, deterministic=True)[0]
        bc = evaluate_policy(env, predict)
        print(f"BC: {sum(r['finished'] for r in bc)}/{len(bc)} finished, last MSE {loss:.5f}", flush=True)
        if args.method == "dagger":
            details = train_dagger(env, model, dataset, make_expert, optimizer,
                                   np.random.default_rng(args.seed + 1_000_000), args.steps,
                                   args.dagger_beta_decay, args.dagger_updates)
        else:
            replay = model.replay_buffer
            if offline_percent:
                replay.offline = dataset
            model.learn(total_timesteps=args.steps)
            details = {"expert_queries": 0, "gradient_updates": model._n_updates,
                       "offline_draws": getattr(replay, "offline_draws", 0),
                       "online_draws": getattr(replay, "online_draws", model._n_updates * BATCH_SIZE)}
        trained_seconds = time.perf_counter() - start
        laps = evaluate_policy(env, predict)
        model.save(args.output / "policy")
        write_json(args.output / "run.json", {
            "method": args.method, "seed": args.seed, "online_steps": args.steps,
            "track": args.track, "kart": "tux", "mode": "TIME_TRIAL", "num_karts": 1, "laps": 1,
            "step_size": STEP_SIZE, "max_steps": args.max_steps, "timeout_s": args.max_steps * STEP_SIZE,
            "train_offset_m": TRAIN_OFFSET_M, "eval_offsets_m": args.eval_offsets,
            "eval_seed": args.eval_seed, "offline_percent": offline_percent,
            "observe_track_items": args.observe_track_items,
            "reward": "progress-v1", "feature_version": FEATURE_VERSION,
            "dataset_sha256": dataset_sha, "expert_commit": EXPERT_COMMIT,
            "expert_team": args.expert_team,
            "net_arch": NET_ARCH, "batch_size": BATCH_SIZE, "threads": 1,
            "bc_updates": args.bc_updates, "bc_lr": BC_LR, "bc_last_mse": loss,
            "initial_actor_sha256": initial_actor_sha, "bc_evaluation": bc,
            "dagger_updates_per_round": args.dagger_updates, "dagger_beta_decay": args.dagger_beta_decay,
            "sac": sac_settings,
            "evaluation": laps, "fit_and_train_wall_s": trained_seconds,
            "runtime": runtime(),
            **details,
        })
        print(f"{args.method} seed {args.seed}: {sum(r['finished'] for r in laps)}/{len(laps)} finished", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["expert", "dagger", "symmetric", "sac"])
    parser.add_argument("--expert-repo", type=Path, default=ROOT / ".cache/time-trial/pystk_project")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16_384)
    parser.add_argument("--bc-updates", type=int, default=BC_UPDATES)
    parser.add_argument("--track", default=TRACK)
    parser.add_argument("--expert-team", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval-offsets", type=float, nargs="+", default=EVAL_OFFSETS_M)
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--demo-transitions", type=int)
    parser.add_argument("--observe-track-items", action="store_true")
    parser.add_argument("--offline-percent", type=int, choices=(0, 25, 50, 75, 100), default=50)
    parser.add_argument("--sac-learning-rate", type=float, default=SAC_SETTINGS["learning_rate"])
    parser.add_argument("--sac-ent-coef", type=float, default=SAC_SETTINGS["ent_coef"])
    parser.add_argument("--sac-gradient-steps", type=int, default=SAC_SETTINGS["gradient_steps"])
    parser.add_argument("--dagger-beta-decay", type=float, default=0.5)
    parser.add_argument("--dagger-updates", type=int, default=DAGGER_UPDATES)
    args = parser.parse_args()
    if args.method != "expert" and args.dataset is None:
        parser.error("--dataset is required for training")
    if args.steps <= 0 or args.steps % ROUNDS or args.bc_updates <= 0:
        parser.error("--steps must be a positive multiple of eight and --bc-updates must be positive")
    if args.max_steps <= 0 or (args.demo_transitions is not None and args.demo_transitions <= 0):
        parser.error("--max-steps and --demo-transitions must be positive")
    if not 0 < args.sac_learning_rate < float("inf") or not 0 <= args.sac_ent_coef < float("inf"):
        parser.error("SAC learning rate must be positive and entropy coefficient must be nonnegative")
    if args.sac_gradient_steps < 1 or args.dagger_updates < 1 or not 0 <= args.dagger_beta_decay <= 1:
        parser.error("Update counts must be positive and DAgger beta decay must be in [0, 1]")
    torch.set_num_threads(1)
    make_expert = expert_factory(args.expert_repo, args.expert_team) if args.method in {"expert", "dagger"} else None
    if args.method == "expert":
        generate(args, make_expert)
    else:
        train(args, make_expert)


if __name__ == "__main__":
    main()
