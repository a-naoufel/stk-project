"""Evaluate frozen solo time-trial policies under matched starting pose changes."""

import argparse
from contextlib import nullcontext
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess

import numpy as np
import pystk2
import torch
from stable_baselines3 import SAC

from scripts.evaluate import write_json
from scripts.stk_study import BATCH_SIZE, BC_LR, SAC_SETTINGS, fingerprint, fit_actions, runtime
from scripts.study_env import EVAL_SEED, MAX_STEPS, lap_result, make_trial
from stk_actor.actors import NET_ARCH

METHODS = ("bc", "dagger", "symmetric")
STARTS = tuple(
    {"race_seed": 60_000 + index, "offset_m": offset, "heading_deg": heading}
    for index, (offset, heading) in enumerate(
        product((-2.5, -1.25, 0.0, 1.25, 2.5), (-60.0, -30.0, 30.0, 60.0))
    )
)
NOMINAL_START = {"race_seed": EVAL_SEED + 5, "offset_m": 0.0, "heading_deg": 0.0}
FRAME_SIZE = (640, 360)


def restore_bc(env, seed, dataset_path, metadata):
    model = SAC("MlpPolicy", env, device="cpu", seed=seed, verbose=0,
                policy_kwargs={"net_arch": NET_ARCH}, batch_size=BATCH_SIZE,
                buffer_size=metadata["online_steps"], **SAC_SETTINGS)
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(-2.0)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=BC_LR)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        fit_actions(model.actor, optimizer, dataset["observations"], dataset["actions"],
                    np.random.default_rng(seed), metadata["bc_updates"])
    digest = sha256()
    for parameter in model.actor.state_dict().values():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    if digest.hexdigest() != metadata["initial_actor_sha256"]:
        raise ValueError(f"Reconstructed BC weights differ for seed {seed}")
    return model


def rollout(env, model, start, trace=None, video=None):
    obs, _ = env.reset(seed=start["race_seed"], options=start)
    race = env.unwrapped
    for step in range(1, MAX_STEPS + 1):
        action = model.predict(obs, deterministic=True)[0]
        obs, _, finished, truncated, _ = env.step(action)
        if trace:
            kart = race.world.karts[0]
            trace.writerow([
                step, race.world.time, kart.overall_distance, np.linalg.norm(kart.velocity_lc),
                (float(action[0]) + 1) / 2, float(action[1]), int(kart.is_on_road), *kart.location,
            ])
        if video:
            frame, = race._stk.race.render_data
            video.stdin.write(np.asarray(frame.image).tobytes())
        if finished or truncated:
            break
    return {**start, **lap_result(env, finished, step)}


def replay(env, model, out, render):
    encoder = nullcontext()
    if render:
        w, h = FRAME_SIZE
        encoder = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pixel_format", "rgb24", "-video_size", f"{w}x{h}", "-framerate", "10",
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-threads", "1",
             "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out.with_suffix(".mp4"))],
            stdin=subprocess.PIPE,
        )
    with out.with_suffix(".csv").open("w", newline="") as stream, encoder as video:
        trace = csv.writer(stream)
        trace.writerow(["step", "time_s", "distance_m", "speed_mps", "throttle", "steer", "on_road", "x", "y", "z"])
        row = rollout(env, model, NOMINAL_START, trace, video)
        if video:
            video.stdin.close()
            if video.wait():
                raise RuntimeError("FFmpeg did not finish the capture")
    return [row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render and not args.replay:
        parser.error("--render requires --replay")
    torch.set_num_threads(1)
    seed_dir = f"seed-{args.seed:02d}"
    source = args.campaign / ("dagger" if args.method == "bc" else args.method) / seed_dir
    metadata = json.loads((source / "run.json").read_text())
    dataset = args.campaign / "expert/expert.npz"
    if fingerprint(dataset) != metadata["dataset_sha256"]:
        raise ValueError("Expert data checksum does not match the original run")
    if runtime() != metadata["runtime"]:
        raise ValueError("Evaluation runtime differs from the original run")
    graphics = None
    if args.render:
        graphics = pystk2.GraphicsConfig.ld()
        graphics.screen_width, graphics.screen_height = FRAME_SIZE
        graphics.display = True
    env = make_trial(args.seed, graphics_config=graphics)
    try:
        if args.method == "bc":
            model = restore_bc(env, args.seed, dataset, metadata)
            policy_sha = metadata["initial_actor_sha256"]
        else:
            path = source / "policy.zip"
            model = SAC.load(path, device="cpu")
            policy_sha = fingerprint(path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.replay:
            rows = replay(env, model, args.output, args.render)
        else:
            rows = [rollout(env, model, start) for start in STARTS]
        write_json(args.output.with_suffix(".json"), {
            "method": args.method, "seed": args.seed, "rendered": args.render,
            "policy_sha256": policy_sha, "runtime": runtime(), "evaluation": rows,
        })
        print(f"{args.method} seed {args.seed}: {sum(r['finished'] for r in rows)}/{len(rows)} laps complete", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
