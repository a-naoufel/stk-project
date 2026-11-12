"""Optuna search and paired replay-ratio follow-up for the solo track study."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from scripts.stk_all_tracks import load_protocol, run_job


DEV_TRACKS = ("lighthouse", "fortmagma", "sandtrack", "xr591")
DEV_SEEDS = (20, 21)
TEST_SEEDS = tuple(range(10, 20))
DEV_START = 52_000
TEST_START = 51_000
TRIALS = 12
RATIOS = (0, 25, 50, 75, 100)
DEFAULT = {
    "dagger": {"dagger_beta_decay": 0.5, "dagger_updates": 384},
    "symmetric": {"sac_learning_rate": 0.0001, "sac_ent_coef": 0.02, "sac_gradient_steps": 1},
}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def build_jobs(protocol, tracks, seeds, eval_seed, arm, method, params, offline):
    return [
        {"id": f"{arm}/{track}/team{team}/seed-{seed:02d}", "arm": arm, "track": track,
         "team": team, "seed": seed, "eval_seed": eval_seed, "method": method, "params": params,
         "offline": offline, "steps": protocol["online_steps"], "bc_updates": protocol["bc_updates"],
         "max_steps": protocol["max_steps"], "offsets": protocol["evaluation_offsets_m"]}
        for track in tracks for team in protocol["expert_teams"] for seed in seeds
    ]


def replay_jobs(protocol, tracks, seeds, eval_seed, prefix, params, ratios):
    return [job for ratio in ratios for job in build_jobs(
        protocol, tracks, seeds, eval_seed, f"{prefix}/replay-{ratio:03d}",
        "symmetric" if ratio else "sac", params, ratio)]


def worker(jobs, root, baseline, expert_repo, workers):
    def execute(job):
        dataset = baseline / "reference" / job["track"] / f"team{job['team']}" / "expert.npz"
        command = ["-m", "scripts.stk_study", job["method"], "--dataset", str(dataset),
                   "--track", job["track"], "--expert-team", str(job["team"]), "--seed", str(job["seed"]),
                   "--expert-repo", str(expert_repo), "--steps", str(job["steps"]),
                   "--bc-updates", str(job["bc_updates"]), "--max-steps", str(job["max_steps"]),
                   "--eval-seed", str(job["eval_seed"]), "--observe-track-items",
                   "--offline-percent", str(job["offline"]), "--eval-offsets", *map(str, job["offsets"])]
        for name, value in job["params"].items():
            command.extend(("--" + name.replace("_", "-"), str(value)))
        run_job(root / job["id"], "run.json", command)

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, job): job["id"] for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(futures[future])
                print(f"Failed {futures[future]}: {error}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} training jobs failed. See their logs.")


def optuna_studies(root):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return {method: optuna.create_study(
        study_name=method, storage=f"sqlite:///{root.resolve() / 'optuna.db'}", load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=112 + index), direction="minimize")
        for index, method in enumerate(DEFAULT)}


def search_plan(root, protocol):
    path = root / "search-jobs.json"
    if path.exists():
        return read(path)
    jobs = []
    for method, study in optuna_studies(root).items():
        if not study.trials:
            study.enqueue_trial(DEFAULT[method])
        while len([trial for trial in study.trials if trial.params]) < TRIALS:
            trial = study.ask()
            if method == "dagger":
                trial.suggest_float("dagger_beta_decay", 0.2, 0.85)
                trial.suggest_categorical("dagger_updates", [192, 384, 768])
            else:
                trial.suggest_float("sac_learning_rate", 1e-5, 3e-4, log=True)
                trial.suggest_float("sac_ent_coef", 0.001, 0.05, log=True)
                trial.suggest_categorical("sac_gradient_steps", [1, 2, 4])
        for trial in study.trials:
            if set(trial.params) != set(DEFAULT[method]):
                raise ValueError(f"Incomplete {method} trial {trial.number} in Optuna storage")
            jobs.extend(build_jobs(protocol, DEV_TRACKS, DEV_SEEDS, DEV_START,
                                   f"search/{method}/{trial.number:02d}", method, trial.params,
                                   50 if method == "symmetric" else 0))
    jobs.extend(replay_jobs(protocol, DEV_TRACKS, DEV_SEEDS, DEV_START, "screen", DEFAULT["symmetric"],
                            [ratio for ratio in RATIOS if ratio != 50]))
    write(path, jobs)
    return jobs


def confirmation_plan(root, protocol, best):
    tracks = [track for track in protocol["tracks"] if track not in DEV_TRACKS]
    jobs = replay_jobs(protocol, tracks, TEST_SEEDS, TEST_START, "test", best["symmetric"]["params"], RATIOS)
    for arm, method, params, offline in (
        ("dagger-tuned", "dagger", best["dagger"]["params"], 0),
        ("dagger-original", "dagger", DEFAULT["dagger"], 0),
        ("sac-original", "symmetric", DEFAULT["symmetric"], 50),
    ):
        jobs.extend(build_jobs(protocol, tracks, TEST_SEEDS, TEST_START, f"test/{arm}", method, params, offline))
    write(root / "test-jobs.json", jobs)
    return jobs


def records(jobs, root, baseline):
    from scripts.analyze_stk_study import check_laps

    scales, references, actor_hashes = {}, {}, {}
    rows = []
    for job in jobs:
        track, team = job["track"], job["team"]
        reference = baseline / "reference" / track
        if track not in scales:
            native = read(reference / "controllers.json")["controllers"]["3"]["native"]
            scales[track] = mean(row["capped_s"] for row in native)
        if (track, team) not in references:
            expert_dir = reference / f"team{team}"
            with (expert_dir / "expert.npz").open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            expert = read(expert_dir / "expert.json")
            if actual != expert["dataset_sha256"]:
                raise ValueError(f"Demonstration dataset changed for {track}, team {team}")
            references[track, team] = expert
        expert = references[track, team]
        run = read(root / job["id"] / "run.json")
        expected = {"track": track, "expert_team": team, "seed": job["seed"], "eval_seed": job["eval_seed"],
                    "method": job["method"], "online_steps": job["steps"], "offline_percent": job["offline"],
                    "bc_updates": job["bc_updates"], "max_steps": job["max_steps"], "observe_track_items": True,
                    "dataset_sha256": expert["dataset_sha256"], "runtime": expert["runtime"]}
        for name, value in expected.items():
            if run[name] != value:
                raise ValueError(f"{job['id']}: {name} is {run[name]}, expected {value}")
        key = (track, team, job["seed"], job["eval_seed"])
        if actor_hashes.setdefault(key, run["initial_actor_sha256"]) != run["initial_actor_sha256"]:
            raise ValueError(f"{job['id']}: initial actor differs across matched conditions")
        offsets = job["offsets"]
        starts = list(zip(range(job["eval_seed"], job["eval_seed"] + len(offsets)), offsets))
        check_laps(run["evaluation"], starts, job["max_steps"] * run["step_size"])
        check_laps(run["bc_evaluation"], starts, job["max_steps"] * run["step_size"])
        params = job["params"]
        if job["method"] == "dagger":
            if (run["dagger_beta_decay"] != params["dagger_beta_decay"]
                    or run["dagger_updates_per_round"] != params["dagger_updates"]
                    or run["gradient_updates"] != 8 * params["dagger_updates"]
                    or run["expert_queries"] != job["steps"]):
                raise ValueError(f"{job['id']}: DAgger schedule mismatch")
        else:
            if any(run["sac"][name.removeprefix("sac_")] != value for name, value in params.items()):
                raise ValueError(f"{job['id']}: SAC settings mismatch")
            updates = job["steps"] // 4 * params["sac_gradient_steps"]
            offline = updates * run["batch_size"] * job["offline"] // 100
            if (run["gradient_updates"], run["offline_draws"], run["online_draws"]) != (
                    updates, offline, updates * run["batch_size"] - offline):
                raise ValueError(f"{job['id']}: replay counts do not match the split")
        laps = run["evaluation"]
        capped = mean(row["capped_s"] for row in laps)
        rows.append({"arm": job["arm"], "track": track, "team": team, "seed": job["seed"],
                     "capped_s": capped, "normalized": capped * 100 / scales[track],
                     "finishes": sum(row["finished"] for row in laps), "laps": len(laps),
                     "wall_s": run["fit_and_train_wall_s"], "updates": run["gradient_updates"]})
    return rows


def summarize(rows):
    return {"runs": len(rows), "laps": sum(row["laps"] for row in rows),
            "finishes": sum(row["finishes"] for row in rows), "mean_capped_s": mean(row["capped_s"] for row in rows),
            "mean_normalized": mean(row["normalized"] for row in rows),
            "mean_wall_s": mean(row["wall_s"] for row in rows), "updates": rows[0]["updates"]}


def by_arm(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["arm"], []).append(row)
    return groups


def finish_search(root, rows):
    groups = by_arm(rows)
    best = {}
    trials = []
    for method, study in optuna_studies(root).items():
        for trial in study.trials:
            summary = summarize(groups[f"search/{method}/{trial.number:02d}"])
            study.tell(trial.number, summary["mean_normalized"], skip_if_finished=True)
            trials.append({"method": method, "trial": trial.number, "params": trial.params, **summary})
        best[method] = next(row for row in trials if row["method"] == method and row["trial"] == study.best_trial.number)
    write(root / "optuna-trials.json", trials)
    write(root / "selected.json", best)
    return best


def table(path, rows):
    with path.open("w", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)


def report(root, rows, *, screen=False):
    from scripts.analyze_stk_study import paired_difference
    import numpy as np

    groups = by_arm(rows)
    prefix = "screen" if screen else "test"
    ratio_arms = {ratio: f"{prefix}/replay-{ratio:03d}" for ratio in RATIOS}
    if screen:
        ratio_arms[50] = "search/symmetric/00"
    summaries = {arm: summarize(group) for arm, group in groups.items()}
    comparisons = {f"{ratio}/{100 - ratio} vs 50/50": (arm, ratio_arms[50])
                   for ratio, arm in ratio_arms.items() if ratio != 50}
    if not screen:
        comparisons.update({"Tuned 50/50 vs tuned DAgger": ("test/replay-050", "test/dagger-tuned"),
                            "Tuned DAgger vs original": ("test/dagger-tuned", "test/dagger-original"),
                            "Tuned 50/50 vs original": ("test/replay-050", "test/sac-original"),
                            "25/75 vs 0/100": ("test/replay-025", "test/replay-000"),
                            "25/75 vs tuned DAgger": ("test/replay-025", "test/dagger-tuned")})
    effects = {}
    for label, (left, right) in comparisons.items():
        keys = [(row["track"], row["team"], row["seed"]) for row in groups[left]]
        matched = {(row["track"], row["team"], row["seed"]): row for row in groups[right]}
        if len(set(keys)) != len(keys) or set(keys) != set(matched):
            raise ValueError(f"{label}: unmatched seed blocks")
        blocks = {}
        for key, row in zip(keys, groups[left], strict=True):
            blocks.setdefault(key[2], []).append(row["normalized"] - matched[key]["normalized"])
        differences = np.array([mean(values) for values in blocks.values()])
        effects[label] = {"mean_pp": float(differences.mean())}
        if not screen:
            effects[label]["interval_pp"] = paired_difference(differences)["bootstrap_95_ci_s"]
    write(root / f"{prefix}-summary.json", {"arms": summaries, "effects": effects})
    table(root / f"{prefix}-runs.csv", rows)
    tracks, seeds = len({row["track"] for row in rows}), len({row["seed"] for row in rows})
    title = "Development replay screen" if screen else "Held-out Optuna and replay results"
    lines = [f"# {title}", "", f"The comparison uses {tracks} tracks, both experts and {seeds} paired training seeds.",
             "Replay ratios are offline/online. Lower capped time is better.", "",
             "| SAC replay split | Finished laps | Capped mean | Mean normalized time | Updates per run |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for ratio, arm in ratio_arms.items():
        m = summaries[arm]
        lines.append(f"| {ratio}/{100 - ratio} | {m['finishes']}/{m['laps']} | {m['mean_capped_s']:.2f} s | "
                     f"{m['mean_normalized']:.2f}% | {m['updates']:,} |")
    if not screen:
        lines.extend(("", "| Other training condition | Finished laps | Capped mean | Mean normalized time | Updates per run |",
                      "| --- | ---: | ---: | ---: | ---: |"))
        for label, arm in (("Tuned DAgger", "test/dagger-tuned"), ("Original DAgger", "test/dagger-original"),
                           ("Original SAC 50/50", "test/sac-original")):
            m = summaries[arm]
            lines.append(f"| {label} | {m['finishes']}/{m['laps']} | {m['mean_capped_s']:.2f} s | "
                         f"{m['mean_normalized']:.2f}% | {m['updates']:,} |")
    lines.extend(("", "Each difference is in percentage points of the per-track reference time. Negative favors the first condition.", "",
                  "| Paired comparison | Mean difference | 95% bootstrap interval |",
                  "| --- | ---: | ---: |"))
    for label, effect in effects.items():
        interval = effect.get("interval_pp")
        ci = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "—"
        lines.append(f"| {label} | {effect['mean_pp']:+.2f} | {ci} |")
    if screen:
        lines.extend(("", "This development screen uses the original SAC settings and two training seeds. Results are descriptive."))
    else:
        lines.extend(("", "The intervals resample paired seed blocks. The ratio contrasts and comparisons involving the best observed ratio are exploratory.",
                      "All ratios use the same SAC settings selected at 50/50. At 100/0 the online rollouts preserve the update cadence and supply zero gradient samples."))
    (root / ("REPLAY_SCREEN.md" if screen else "RESULTS.md")).write_text("\n".join(lines) + "\n")


def run(root, baseline, dispatch):
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    if read(baseline / "protocol.json") != protocol:
        raise ValueError("The baseline campaign differs from the all-track protocol")
    search = search_plan(root, protocol)
    pilot = [job for job in search if job["arm"].startswith("screen/") or job["arm"] == "search/symmetric/00"]
    dispatch(pilot, "screen")
    report(root, records(pilot, root, baseline), screen=True)
    dispatch([job for job in search if job not in pilot], "search")
    trials = [job for job in search if job["arm"].startswith("search/")]
    best = finish_search(root, records(trials, root, baseline))
    confirmation = confirmation_plan(root, protocol, best)
    dispatch(confirmation, "test")
    report(root, records(confirmation, root, baseline))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expert-repo", type=Path, default=Path(".cache/time-trial/pystk_project"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--job-file", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root, baseline, expert_repo = (path.resolve() for path in (args.root, args.baseline, args.expert_repo))
    if args.job_file:
        worker(read(args.job_file), root, baseline, expert_repo, args.workers)
    else:
        run(root, baseline, lambda jobs, phase: worker(jobs, root, baseline, expert_repo, args.workers))


if __name__ == "__main__":
    main()
