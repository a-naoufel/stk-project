"""Validate and compare the paired held-out starting poses."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from scripts.analyze_stk_study import metrics, paired_difference, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign / "recovery"
    protocol = read(root / "protocol.json")
    all_laps = {method: [] for method in protocol["methods"]}
    seeds = []
    for seed in protocol["training_seeds"]:
        record = {"seed": seed}
        for method in all_laps:
            run = read(root / "laps" / f"{method}-seed-{seed:02d}.json")
            source = args.campaign / ("dagger" if method == "bc" else method) / f"seed-{seed:02d}"
            reference = read(source / "run.json")
            policy_sha = (reference["initial_actor_sha256"] if method == "bc"
                          else sha256((source / "policy.zip").read_bytes()).hexdigest())
            if (run["method"], run["seed"], run["rendered"], run["policy_sha256"], run["runtime"]) != (
                method, seed, False, policy_sha, reference["runtime"]
            ):
                raise ValueError(f"Unexpected policy or runtime for {method}, seed {seed}")
            laps = run["evaluation"]
            starts = [{key: row[key] for key in ("race_seed", "offset_m", "heading_deg")} for row in laps]
            if starts != protocol["starts"]:
                raise ValueError(f"Unmatched starts for {method}, seed {seed}")
            for row in laps:
                expected_time = row["lap_s"] if row["finished"] else protocol["timeout_s"]
                if row["capped_s"] != expected_time or not 0 < row["steps"] <= protocol["max_steps"]:
                    raise ValueError(f"Invalid result for {method}, seed {seed}: {row}")
            record[method] = metrics(laps)
            all_laps[method].extend(laps)
        seeds.append(record)

    summary = {
        "by_seed": seeds,
        "aggregate": {method: metrics(rows) for method, rows in all_laps.items()},
        "by_heading_magnitude": {
            str(degrees): {method: metrics([r for r in rows if abs(r["heading_deg"]) == degrees])
                           for method, rows in all_laps.items()}
            for degrees in sorted({abs(start["heading_deg"]) for start in protocol["starts"]})
        },
        "symmetric_minus_dagger": paired_difference(np.array([
            seed["symmetric"]["mean_capped_s"] - seed["dagger"]["mean_capped_s"] for seed in seeds
        ])),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_seed"}, indent=2))


if __name__ == "__main__":
    main()
