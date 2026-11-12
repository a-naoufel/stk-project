"""Report controller baselines and paired seed effects across the 21 tracks."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from scripts.analyze_stk_study import check_laps, metrics, paired_difference, read
from scripts.evaluate import write_json
from scripts.stk_all_tracks import CONTROLLERS, load_protocol


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_reference(root, protocol, starts, cap):
    references, table = {}, []
    for track in protocol["tracks"]:
        reference = read(root / "reference" / track / "controllers.json")
        if reference["track"] != track or set(reference["controllers"]) != set(map(str, CONTROLLERS)):
            raise ValueError(f"Wrong controller reference for {track}")
        references[track] = reference
        for team, controller in reference["controllers"].items():
            for mode in ("native", "restricted"):
                laps = controller[mode]
                check_laps(laps, starts, cap)
                table.append({"track": track, "team": team, "controller": controller["name"],
                              "actions": mode, **metrics(laps), "capped_sd_s": float(np.std(
                                  [r["capped_s"] for r in laps], ddof=1))})
    return references, table


def compare_historical_profile(source, baselines):
    """Join the supplied five-kart aggregates to the new solo references."""
    historic = {}
    track = None
    for line in source.read_text().splitlines():
        if line.startswith("## "):
            track = None
        match = re.fullmatch(r"### `([a-z0-9_]+)`", line)
        if match:
            track = match[1]
        if not track or not line.startswith("|"):
            continue
        columns = [cell.strip() for cell in line.strip("|").split("|")]
        if len(columns) != 6 or columns[0] not in CONTROLLERS.values():
            continue
        name, mean, sd, finished, _, rank = columns
        count, total = map(int, finished.split("/"))
        mean_s = float(mean) if count else None
        historic[track, name] = {
            "profile_finished": count, "profile_laps": total, "profile_finish_mean_s": mean_s,
            "profile_finish_sd_s": float(sd) if count > 1 else None,
            "profile_capped_s_approx": round(((mean_s or 0) * count + 200 * (total - count)) / total, 3),
            "profile_capped_rank": int(rank),
        }
    rows = []
    for solo in baselines:
        if solo["actions"] == "native":
            key = solo["track"], solo["controller"]
            rows.append({"track": key[0], "controller": key[1], **historic[key],
                         "solo_native_finished": solo["finishes"], "solo_native_capped_s": solo["mean_capped_s"],
                         "solo_native_capped_sd_s": solo["capped_sd_s"]})
    return rows


def load_policy_laps(root, protocol, references, starts, cap):
    for track, reference in references.items():
        for team in protocol["expert_teams"]:
            demo = root / "reference" / track / f"team{team}"
            expert = read(demo / "expert.json")
            digest = sha256((demo / "expert.npz").read_bytes()).hexdigest()
            if (expert["dataset_sha256"] != digest or expert["track"] != track
                    or expert["agent"] != f"team{team}" or expert["commit"] != reference["expert_commit"]
                    or expert["transitions"] != protocol["demo_transitions"]
                    or expert["evaluation"] != reference["controllers"][str(team)]["restricted"]):
                raise ValueError(f"Expert dataset mismatch: {track}, team {team}")
            for seed in protocol["training_seeds"]:
                first = None
                for method in protocol["methods"]:
                    folder = root / "learners" / track / f"team{team}" / method / f"seed-{seed:02d}"
                    run = read(folder / "run.json")
                    expected = {"track": track, "expert_team": team, "seed": seed, "method": method,
                                "online_steps": protocol["online_steps"], "bc_updates": protocol["bc_updates"],
                                "dataset_sha256": digest, "runtime": reference["runtime"],
                                "expert_commit": reference["expert_commit"], "num_karts": 1, "laps": 1,
                                "mode": "TIME_TRIAL", "step_size": protocol["step_size"],
                                "max_steps": protocol["max_steps"], "observe_track_items": True}
                    if any(run[key] != value for key, value in expected.items()):
                        raise ValueError(f"Run metadata mismatch: {folder}")
                    if method == "dagger":
                        if run["expert_queries"] != protocol["online_steps"]:
                            raise ValueError(f"Wrong query count: {folder}")
                    else:
                        draws = (protocol["online_steps"] // 4) * run["batch_size"]
                        offline = draws // 2 if method == "symmetric" else 0
                        if (run["offline_draws"], run["online_draws"]) != (offline, draws - offline):
                            raise ValueError(f"Wrong replay counts: {folder}")
                    check_laps(run["evaluation"], starts, cap)
                    check_laps(run["bc_evaluation"], starts, cap)
                    initialization = (run["initial_actor_sha256"], run["bc_evaluation"])
                    if first is None:
                        first = initialization
                        yield track, team, seed, "bc", run["bc_evaluation"]
                    elif initialization != first:
                        raise ValueError(f"BC mismatch: {folder}")
                    yield track, team, seed, method, run["evaluation"]


def contrast(values, *, percent=True):
    summary = paired_difference(values)
    if percent:
        summary = {key.removesuffix("_s") + "_pct" if key.endswith("_s") else key: value
                   for key, value in summary.items()}
    return summary


def summarize(root, protocol, references, laps):
    seeds = protocol["training_seeds"]
    teams = protocol["expert_teams"]
    methods = ["bc", *protocol["methods"]]
    tracks = protocol["tracks"]
    results, comparisons, percent = [], [], {}
    wins = {team: {method: {"native": 0, "restricted": 0} for method in methods} for team in teams}
    for track in tracks:
        controllers = references[track]["controllers"]
        baseline = metrics(controllers["3"]["native"])["mean_capped_s"]
        best_controller = {mode: min(metrics(controller[mode])["mean_capped_s"]
                                     for controller in controllers.values())
                           for mode in ("native", "restricted")}
        for team in teams:
            for method in methods:
                trials = [laps[track, team, seed, method] for seed in seeds]
                values = np.array([metrics(rows)["mean_capped_s"] for rows in trials])
                percent[track, team, method] = 100 * (values / baseline - 1)
                for mode, best in best_controller.items():
                    wins[team][method][mode] += int(values.mean() < best - 1e-9)
                results.append({"track": track, "expert_team": team, "method": method,
                                **metrics([row for trial in trials for row in trial]),
                                "training_seed_sd_s": float(values.std(ddof=1)),
                                "training_seed_sd_pct": float((100 * values / baseline).std(ddof=1)),
                                "mean_within_policy_start_sd_s": float(np.mean([
                                    np.std([row["capped_s"] for row in trial], ddof=1) for trial in trials])),
                                "mean_vs_team3_native_pct": float(percent[track, team, method].mean())})
                for other, controller in controllers.items():
                    for mode in ("native", "restricted"):
                        other_s = metrics(controller[mode])["mean_capped_s"]
                        effect = contrast(values - other_s, percent=False)
                        comparisons.append({"track": track, "expert_team": team, "method": method,
                                            "controller_team": other, "controller_actions": mode,
                                            "controller_capped_s": other_s,
                                            "learner_minus_controller_s": effect["mean_difference_s"],
                                            "ci_low_s": effect["bootstrap_95_ci_s"][0],
                                            "ci_high_s": effect["bootstrap_95_ci_s"][1],
                                            "learner_seed_wins": effect["negative_pairs"],
                                            "learner_seed_ties": effect["ties"]})
    write_csv(root / "track-methods.csv", results)
    write_csv(root / "learner-vs-controller.csv", comparisons)
    effects = {(track, team): percent[track, team, "symmetric"] - percent[track, team, "dagger"]
               for track in tracks for team in teams}
    by_track = {track: contrast(np.mean([effects[track, team] for team in teams], axis=0)) for track in tracks}
    track_effect_means = [effect["mean_difference_pct"] for effect in by_track.values()]
    summary = {
        "tracks": tracks, "training_seeds": seeds, "expert_teams": teams,
        "symmetric_minus_dagger_pct_of_native_team3": contrast(np.mean(list(effects.values()), axis=0)),
        "symmetric_minus_dagger_by_track": by_track,
        "symmetric_minus_dagger_by_track_and_expert": {
            track: {team: contrast(effects[track, team]) for team in teams} for track in tracks},
        "across_track_sd_of_mean_method_gap_pct": float(np.std(track_effect_means, ddof=1)),
        "team4_minus_team3_by_method": {
            method: contrast(np.mean([percent[track, 4, method] - percent[track, 3, method]
                                      for track in tracks], axis=0)) for method in methods},
        "expert_by_method_interaction": contrast(np.mean([
            effects[track, 4] - effects[track, 3] for track in tracks], axis=0)),
        "symmetric_minus_online_sac": contrast(np.mean([
            percent[track, team, "symmetric"] - percent[track, team, "sac"]
            for track in tracks for team in teams], axis=0)),
        "across_track_sd_of_mean_normalized_time_pct": {
            team: {method: float(np.std([percent[track, team, method].mean() for track in tracks], ddof=1))
                   for method in methods} for team in teams},
        "tracks_with_lower_mean_capped_time_than_best_solo_controller": wins,
    }
    write_json(root / "summary.json", summary)
    plot_effects(by_track, root / "track-effects.png")
    plot_variance(results, tracks, teams, root / "training-variance.png")
    write_report(root / "RESULTS.md", protocol, summary, results, methods)


def write_report(path, protocol, summary, results, methods):
    labels = {"bc": "Behavior cloning", "dagger": "DAgger", "symmetric": "Symmetric SAC", "sac": "Online-only SAC"}

    def interval(effect):
        low, high = effect["bootstrap_95_ci_pct"]
        return f"{effect['mean_difference_pct']:+.2f}% [{low:+.2f}, {high:+.2f}]"

    tracks = protocol["tracks"]
    seeds = protocol["training_seeds"]
    teams = protocol["expert_teams"]
    main = summary["symmetric_minus_dagger_pct_of_native_team3"]
    lines = [
        "# DAgger and symmetric replay in solo SuperTuxKart", "",
        f"The study covers {len(tracks)} tracks, {len(teams)} demonstration sources, and {len(seeds)} training seeds "
        f"per condition at {protocol['online_steps']:,} online steps. Each policy ran on "
        f"{len(protocol['evaluation_seeds'])} matched starts. The overall symmetric SAC minus DAgger gap was "
        f"**{interval(main)}** of Team L’Eclair’s solo capped lap time (mean and 95% paired bootstrap interval). "
        "Positive differences favor DAgger. Each track and source receives equal weight.", "",
        "## Completion, controller comparisons, and variability", "",
        "| Demonstrations | Learner | Finishes | Mean vs Team 3 native | Tracks faster than best native / restricted controller | Median within-track SD | Across-track SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for team in teams:
        for method in methods:
            rows = [row for row in results if row["expert_team"] == team and row["method"] == method]
            mean = np.mean([row["mean_vs_team3_native_pct"] for row in rows])
            within = np.median([row["training_seed_sd_pct"] for row in rows])
            between = summary["across_track_sd_of_mean_normalized_time_pct"][team][method]
            wins = summary["tracks_with_lower_mean_capped_time_than_best_solo_controller"][team][method]
            lines.append(f"| Team {team} | {labels[method]} | {sum(row['finishes'] for row in rows)}/"
                         f"{sum(row['laps'] for row in rows)} | {mean:+.2f}% | {wins['native']} / "
                         f"{wins['restricted']} | {within:.2f}% | {between:.2f}% |")
    lines += ["", "Controller wins use lower mean capped time on the fixed starts. The two SD columns describe "
              "variation across training seeds within a track and variation of normalized means across tracks. "
              "Read both alongside completion, since a timeout contributes the full capped time.", "",
              "## Effect of the demonstration source", "",
              "| Learner | Team 4 minus Team 3 |", "| --- | ---: |"]
    for method in methods:
        lines.append(f"| {labels[method]} | {interval(summary['team4_minus_team3_by_method'][method])} |")
    interaction = interval(summary["expert_by_method_interaction"])
    ablation = interval(summary["symmetric_minus_online_sac"])
    lines += ["", f"Switching from Team 3 to Team 4 changed the symmetric SAC minus DAgger gap by {interaction}. "
              f"The symmetric SAC minus online-only SAC contrast was {ablation}. Both comparisons use the same "
              "normalization and paired training seeds.", "", "## Track-level results", "",
              "| Track | Symmetric SAC minus DAgger |", "| --- | ---: |"]
    for track, effect in summary["symmetric_minus_dagger_by_track"].items():
        lines.append(f"| `{track}` | {interval(effect)} |")
    lines += ["", "The across-track SD of the mean method gaps was "
              f"{summary['across_track_sd_of_mean_method_gap_pct']:.2f}% of the reference time. "
              "The track-level intervals are exploratory.", "",
              "[Paired track effects](track-effects.png) · [Training-seed variance](training-variance.png) · "
              "[Per-track learner metrics](track-methods.csv) · [Solo controller baselines](controller-baselines.csv) · "
              "[Learner vs controller comparisons](learner-vs-controller.csv)", ""]
    path.write_text("\n".join(lines))


def plot_effects(by_track, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 8), layout="constrained")
    for i, (track, result) in enumerate(by_track.items()):
        low, high = result["bootstrap_95_ci_pct"]
        axis.hlines(i, low, high, color="#64748b")
        axis.scatter(result["mean_difference_pct"], i, color="#136d70", zorder=3)
    axis.axvline(0, color="#475569", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(by_track)), [track.replace("_", " ") for track in by_track])
    axis.invert_yaxis()
    axis.set_xlabel("Symmetric SAC minus DAgger (% of Team 3 solo capped time)")
    axis.set_title(f"Solo performance across {len(by_track)} tracks, averaged over both experts", loc="left")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_variance(results, tracks, teams, output):
    import matplotlib.pyplot as plt

    sd = {(row["track"], row["expert_team"], row["method"]): row["training_seed_sd_pct"]
          for row in results}
    columns = [(team, method) for team in teams for method in ("dagger", "symmetric")]
    values = [[sd[track, team, method] for team, method in columns] for track in tracks]
    fig, axis = plt.subplots(figsize=(7.8, 8), layout="constrained")
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0)
    axis.set_yticks(range(len(tracks)), [track.replace("_", " ") for track in tracks])
    labels = {"dagger": "DAgger", "symmetric": "Symmetric SAC"}
    axis.set_xticks(range(len(columns)), [f"Team {team}\n{labels[method]}" for team, method in columns])
    axis.set_title("Lap-time variation across training seeds", loc="left")
    fig.colorbar(image, ax=axis, label="SD of mean capped time (% of Team 3 solo capped time)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--controller-profile", type=Path, help="Join the historical five-kart summary")
    args = parser.parse_args()
    frozen = args.campaign / "protocol.json"
    protocol = read(frozen) if frozen.is_file() else load_protocol()
    starts = list(zip(protocol["evaluation_seeds"], protocol["evaluation_offsets_m"], strict=True))
    cap = protocol["max_steps"] * protocol["step_size"]
    references, baselines = read_reference(args.campaign, protocol, starts, cap)
    write_csv(args.campaign / "controller-baselines.csv", baselines)
    if args.controller_profile:
        write_csv(args.campaign / "historical-controller-context.csv",
                  compare_historical_profile(args.controller_profile, baselines))
    if not args.baselines_only:
        laps = {(track, team, seed, method): rows for track, team, seed, method, rows in
                load_policy_laps(args.campaign, protocol, references, starts, cap)}
        summarize(args.campaign, protocol, references, laps)


if __name__ == "__main__":
    main()
