"""Prepare and run the all-track solo study using the shared SB3 trainer."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/STK_ALL_TRACKS_PROTOCOL.json"
CONTROLLERS = {1: "Tasty Crousteam", 2: "DemoPilote", 3: "Team L'Eclair",
               4: "The Winners", 5: "Donkey Bombs"}


def load_protocol():
    return json.loads(PROTOCOL.read_text())


def prepare_track(args, protocol):
    from scripts.evaluate import write_json
    from scripts.stk_study import generate, runtime, trial_options
    from scripts.study_env import evaluate_policy, expert_factory, EXPERT_COMMIT
    from scripts.study_env import make_trial

    args.output.mkdir()
    baselines = {}
    for team in CONTROLLERS:
        make_expert = expert_factory(args.expert_repo, team)
        args.expert_team = team
        track_output = args.output
        if team in protocol["expert_teams"]:
            args.output = track_output / f"team{team}"
            generate(args, make_expert)
            restricted = json.loads((args.output / "expert.json").read_text())["evaluation"]
            args.output = track_output
        else:
            env = make_trial(0, **trial_options(args))
            try:
                restricted = evaluate_policy(env, None, make_expert)
            finally:
                env.close()
        env = make_trial(0, **trial_options(args), native=True)
        try:
            native = evaluate_policy(env, None, make_expert, native=True)
        finally:
            env.close()
        baselines[team] = {"name": CONTROLLERS[team], "restricted": restricted, "native": native}
        print(f"{args.track} team {team}: {sum(r['finished'] for r in restricted)}/{len(restricted)} "
              f"restricted, {sum(r['finished'] for r in native)}/{len(native)} native", flush=True)
    write_json(args.output / "controllers.json", {"track": args.track, "runtime": runtime(),
               "expert_commit": EXPERT_COMMIT, "controllers": baselines})


def run_job(destination, marker, command):
    if (destination / marker).is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}-") as temporary:
        staging = Path(temporary)
        env.update(XDG_CONFIG_HOME=str(staging / "config"), XDG_DATA_HOME=str(staging / "data"))
        log = destination.with_name(f"{destination.name}.log")
        result = staging / "result"
        with log.open("w") as stream:
            subprocess.run([sys.executable, *command, "--output", str(result)],
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not (result / marker).is_file():
            raise RuntimeError(f"Job did not produce {marker}. See {log}")
        result.rename(destination)
    print(f"Completed {destination}", flush=True)


def jobs(args, protocol):
    common = ["--expert-repo", str(args.expert_repo), "--max-steps", str(protocol["max_steps"]),
              "--eval-offsets", *map(str, protocol["evaluation_offsets_m"])]
    for track in args.tracks or protocol["tracks"]:
        track_args = ["--track", track, *common]
        reference = args.campaign / "reference" / track
        if args.phase == "prepare":
            yield (reference, "controllers.json", ["-m", "scripts.stk_all_tracks", "prepare-track",
                   *track_args])
            continue
        if not (reference / "controllers.json").is_file():
            raise ValueError(f"Prepare {track} before training")
        for seed in args.seeds or protocol["training_seeds"]:
            for team in protocol["expert_teams"]:
                for method in protocol["methods"]:
                    output = args.campaign / "learners" / track / f"team{team}" / method / f"seed-{seed:02d}"
                    dataset = reference / f"team{team}" / "expert.npz"
                    yield (output, "run.json", ["-m", "scripts.stk_study", method, *track_args, "--observe-track-items",
                           "--seed", str(seed), "--expert-team", str(team), "--dataset", str(dataset),
                           "--steps", str(protocol["online_steps"]), "--bc-updates", str(protocol["bc_updates"])])


def main():
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for phase in ("prepare", "train"):
        command = sub.add_parser(phase)
        command.add_argument("campaign", type=Path)
        command.add_argument("--tracks", nargs="+", choices=protocol["tracks"])
        command.add_argument("--workers", type=int, default=1)
    sub.choices["train"].add_argument("--seeds", nargs="+", type=int, choices=protocol["training_seeds"])
    worker = sub.add_parser("prepare-track")
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--track", choices=protocol["tracks"], required=True)
    worker.add_argument("--max-steps", type=int, required=True)
    worker.add_argument("--eval-offsets", type=float, nargs="+", required=True)
    for command in sub.choices.values():
        command.add_argument("--expert-repo", type=Path, default=ROOT / ".cache/time-trial/pystk_project")
    args = parser.parse_args()
    args.expert_repo = args.expert_repo.resolve()
    if args.phase == "prepare-track":
        args.demo_transitions = protocol["demo_transitions"]
        args.observe_track_items = True
        prepare_track(args, protocol)
        return
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.tracks and len(set(args.tracks)) != len(args.tracks):
        parser.error("--tracks cannot contain duplicates")
    if args.phase == "train" and args.seeds and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds cannot contain duplicates")
    args.campaign = args.campaign.resolve()
    planned = list(jobs(args, protocol))
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, *job): job[0] for job in planned}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(futures[future])
                print(f"Failed {futures[future]}: {error}", file=sys.stderr, flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} of {len(planned)} jobs failed")


if __name__ == "__main__":
    main()
