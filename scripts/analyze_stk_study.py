"""Validate a finished time-trial campaign and compare paired training seeds."""

import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 17
BOOTSTRAP_DRAWS = 20_000


def read(path):
    return json.loads(path.read_text())


def metrics(laps):
    times = [r["lap_s"] for r in laps if r["finished"]]
    return {
        "laps": len(laps),
        "finishes": len(times),
        "completion": len(times) / len(laps),
        "mean_capped_s": float(np.mean([r["capped_s"] for r in laps])),
        "mean_finished_s": float(np.mean(times)) if times else None,
    }


def paired_difference(differences):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample = rng.choice(differences, (BOOTSTRAP_DRAWS, len(differences)), replace=True)
    lower, upper = np.quantile(sample.mean(axis=1), [0.025, 0.975])
    negatives = int((differences < -1e-9).sum())
    positives = int((differences > 1e-9).sum())
    non_ties = negatives + positives
    sign_p = min(1.0, 2 * sum(comb(non_ties, i) for i in range(min(negatives, positives) + 1))
                 / 2**non_ties)
    return {
        "paired_differences_s": differences.tolist(),
        "mean_difference_s": float(differences.mean()),
        "bootstrap_95_ci_s": [float(lower), float(upper)],
        "negative_pairs": negatives, "positive_pairs": positives,
        "ties": len(differences) - non_ties, "two_sided_sign_p": sign_p,
    }


def check_laps(laps, starts, cap):
    if [(r["seed"], r["offset_m"]) for r in laps] != starts:
        raise ValueError("Evaluation starts do not match the protocol")
    for row in laps:
        if row["finished"]:
            if not 0 < row["lap_s"] <= cap or row["capped_s"] != row["lap_s"]:
                raise ValueError(f"Invalid finish time: {row}")
        elif row["lap_s"] is not None or row["capped_s"] != cap:
            raise ValueError(f"Invalid timeout: {row}")


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"bc": "Behavior\ncloning", "dagger": "DAgger",
              "symmetric": "Symmetric\nSAC", "sac": "Online-only\nSAC"}
    order = list(labels)
    colors = ["#748595", "#136d70", "#a3492c", "#5f4b8b"]
    seeds = summary["training_seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), layout="constrained")
    for record in summary["by_seed"]:
        values = [record[name]["mean_capped_s"] for name in order]
        axes[0].plot(range(len(order)), values, color="#c0c7cb", linewidth=0.75, zorder=1)
        axes[0].scatter(range(len(order)), values, c=colors, s=24, alpha=0.75, zorder=2)
    axes[0].axhline(summary["expert"]["mean_capped_s"], linestyle="--", linewidth=1,
                    color="#333333", label="Team 3 expert")
    axes[0].set_ylabel("Mean capped lap time (s)")
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[0].set_xticks(range(len(order)), [labels[name] for name in order])
    axes[0].set_title("Lap times by training seed", fontsize=11)
    effect = summary["symmetric_minus_dagger"]
    axes[1].axvline(0, linestyle="--", linewidth=1, color="#333333")
    axes[1].scatter(effect["paired_differences_s"], range(len(seeds)), color=colors[2], s=24)
    lower, upper = effect["bootstrap_95_ci_s"]
    mean = effect["mean_difference_s"]
    axes[1].errorbar(mean, len(seeds), xerr=[[mean - lower], [upper - mean]],
                     fmt="D", markersize=5, capsize=4, color="#333333")
    axes[1].set_yticks(range(len(seeds) + 1), [f"Seed {seed}" for seed in seeds] + ["Mean"])
    axes[1].invert_yaxis()
    axes[1].set_title("Paired differences and 95% interval for the mean", fontsize=11)
    axes[1].set_xlabel("Symmetric SAC minus DAgger (s)")
    for ax in axes:
        ax.grid(axis="y", color="#e8e9ea")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Solo Lighthouse time trial  |  Lines connect the same training seed", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study
    protocol = read(root / "protocol.json")
    expert = read(root / "expert/expert.json")
    dataset_sha = sha256((root / "expert/expert.npz").read_bytes()).hexdigest()
    if dataset_sha != expert["dataset_sha256"]:
        raise ValueError("Expert dataset checksum mismatch")
    starts = list(zip(protocol["evaluation_seeds"], protocol["evaluation_offsets_m"], strict=True))
    cap = protocol["max_steps"] * protocol["step_size"]
    check_laps(expert["evaluation"], starts, cap)
    methods = protocol["methods"]
    seeds = protocol["training_seeds"]
    laps = {name: [] for name in ("bc", *methods)}
    by_seed = []
    for seed in seeds:
        pair = {name: read(root / name / f"seed-{seed:02d}" / "run.json") for name in methods}
        first = pair[methods[0]]
        record = {"seed": seed, "bc": metrics(first["bc_evaluation"])}
        laps["bc"].extend(first["bc_evaluation"])
        for method, run in pair.items():
            expected = {
                "method": method, "seed": seed, "online_steps": protocol["online_steps"],
                "bc_updates": protocol["bc_updates"], "dataset_sha256": dataset_sha,
                "initial_actor_sha256": first["initial_actor_sha256"],
                "bc_evaluation": first["bc_evaluation"],
                "runtime": expert["runtime"], "track": protocol["track"],
                "num_karts": 1, "laps": 1, "mode": "TIME_TRIAL", "max_steps": protocol["max_steps"],
            }
            for key, value in expected.items():
                if run[key] != value:
                    raise ValueError(f"{method}, seed {seed}: unexpected {key}")
            check_laps(run["evaluation"], starts, cap)
            check_laps(run["bc_evaluation"], starts, cap)
            if method in {"symmetric", "sac"}:
                expected_draws = run["gradient_updates"] * run["batch_size"]
                expected_offline = expected_draws // 2 if method == "symmetric" else 0
                if run["offline_draws"] != expected_offline or run["online_draws"] != expected_draws - expected_offline:
                    raise ValueError(f"Wrong replay proportions for {method}, seed {seed}")
            elif run["expert_queries"] != protocol["online_steps"]:
                raise ValueError(f"Missing DAgger labels for seed {seed}")
            record[method] = metrics(run["evaluation"])
            laps[method].extend(run["evaluation"])
        by_seed.append(record)

    means = {method: np.array([r[method]["mean_capped_s"] for r in by_seed]) for method in methods}
    summary = {
        "expert": metrics(expert["evaluation"]),
        "training_seeds": seeds,
        "dataset_sha256": dataset_sha,
        "by_seed": by_seed,
        "aggregate": {name: metrics(rows) for name, rows in laps.items()},
        "symmetric_minus_dagger": paired_difference(means["symmetric"] - means["dagger"]),
        "symmetric_minus_online_sac_mean_s": float((means["symmetric"] - means["sac"]).mean()),
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(summary, root / "seed-comparison.png")
    print(json.dumps({key: summary[key] for key in ("aggregate", "expert", "symmetric_minus_dagger",
                                                    "symmetric_minus_online_sac_mean_s")}, indent=2))


if __name__ == "__main__":
    main()
