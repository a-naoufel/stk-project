"""Summarize a fixed STK evaluation suite and export comparison plots."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bootstrap(values, rng, samples=5000):
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    protocol = json.loads((folder / "protocol.json").read_text())
    tracks, seeds = protocol["tracks"], protocol["seeds"]
    expected = {(track, seed) for track in tracks for seed in seeds}
    rng = np.random.default_rng(20261107)
    agents = {}
    all_rows = []
    for path in sorted((folder / "results").glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        rows = json.loads(path.read_text())
        if len(rows) != len(expected) or {(r["track"], r["seed"]) for r in rows} != expected:
            raise ValueError(f"Incomplete or duplicated races: {path}")
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        if "finished_at" not in meta:
            raise ValueError(f"Missing completion provenance: {path}")
        for row in rows:
            if not 1 <= row["steps"] <= protocol["max_steps"] or not 1 <= row["position"] <= protocol["num_karts"]:
                raise ValueError(f"Invalid race result: {row}")
            if not row["finished"] and not row["truncated"]:
                raise ValueError(f"Unfinished race without truncation: {row}")
        agents[path.stem] = {"rows": rows, "meta": meta}
        all_rows.extend({"model": path.stem, **row} for row in rows)
    if not agents:
        raise ValueError("No complete agent results")
    summaries = []
    for label, agent in agents.items():
        rows = agent["rows"]
        finished = [row for row in rows if row["finished"]]
        by_track = [{
            "track": track,
            "completion_rate": float(np.mean([row["finished"] for row in rows if row["track"] == track])),
            "mean_capped_steps": float(np.mean([row["steps"] for row in rows if row["track"] == track])),
        } for track in tracks]
        measured = [row for row in rows if row["mean_action_ms"] is not None]
        summaries.append({
            "model": label, "races": len(rows), "finished": len(finished),
            "completion_rate": len(finished) / len(rows),
            "completion_ci95": bootstrap([row["completion_rate"] for row in by_track], rng),
            "mean_position": float(np.mean([row["position"] for row in rows])),
            "mean_capped_steps": float(np.mean([row["steps"] for row in rows])),
            "capped_steps_ci95": bootstrap([row["mean_capped_steps"] for row in by_track], rng),
            "mean_finished_steps": float(np.mean([row["steps"] for row in finished])) if finished else None,
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_action_ms": float(np.mean([row["mean_action_ms"] for row in measured])) if measured else None,
            "max_action_ms": max(row["max_action_ms"] for row in measured) if measured else None,
            "checkpoint_sha256": agent["meta"]["checkpoint_sha256"], "by_track": by_track,
        })
    summaries.sort(key=lambda row: (-row["completion_rate"], row["mean_capped_steps"], row["mean_position"]))
    ppo = [row for row in summaries if row["model"].startswith("ppo-")]
    best = ppo[0]
    comparisons = {}
    for baseline in ["heuristic", "ai"]:
        if baseline not in agents:
            continue
        model_rows = {(row["track"], row["seed"]): row for row in agents[best["model"]]["rows"]}
        base_rows = {(row["track"], row["seed"]): row for row in agents[baseline]["rows"]}
        finish_deltas, step_deltas = [], []
        for track in tracks:
            finish_deltas.append(np.mean([int(model_rows[track, seed]["finished"]) - int(base_rows[track, seed]["finished"]) for seed in seeds]))
            step_deltas.append(np.mean([model_rows[track, seed]["steps"] - base_rows[track, seed]["steps"] for seed in seeds]))
        comparisons[baseline] = {
            "completion_difference_pp": float(np.mean(finish_deltas) * 100),
            "completion_difference_ci95_pp": [x * 100 for x in bootstrap(finish_deltas, rng)],
            "capped_steps_difference": float(np.mean(step_deltas)),
            "capped_steps_difference_ci95": bootstrap(step_deltas, rng),
        }
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
               "best_ppo": best["model"], "leaderboard": summaries,
               "best_ppo_vs_baselines": comparisons,
               "ci_method": "5000 bootstrap resamples of whole tracks, preserving the paired evaluation seeds; RNG seed 20261107"}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [key for key in summaries[0] if key not in {"by_track", "completion_ci95", "capped_steps_ci95"}]
    with (folder / "leaderboard.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summaries)
    with (folder / "races.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = [row["model"] for row in summaries]
    colors = ["#0d9488" if label == "ai" else "#d97706" if label == "heuristic" else "#2563eb" if label == best["model"] else "#94a3b8" for label in labels]
    y = np.arange(len(summaries))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(labels) * 0.36)), sharey=True, constrained_layout=True)
    axes[0].barh(y, [row["completion_rate"] * 100 for row in summaries], color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 112)
    axes[0].set_xlabel("Races completed (%)")
    for index, row in enumerate(summaries):
        axes[0].text(row["completion_rate"] * 100 + 1.2, index, f"{row['finished']}/{row['races']}", va="center", fontsize=9)
    axes[1].barh(y, [row["mean_capped_steps"] for row in summaries], color=colors)
    axes[1].set_xlabel(f"Mean steps (unfinished races = {protocol['max_steps']}; lower is better)")
    axes[0].set_title("Completion", loc="left", weight="bold")
    axes[1].set_title("Race length including failures", loc="left", weight="bold")
    fig.suptitle(f"STK {protocol['phase']}\n{len(tracks)} tracks × {len(seeds)} seeds ({seeds[0]}–{seeds[-1]}), {protocol['num_karts']} karts, one lap", weight="bold", fontsize=13)
    for suffix in ["png", "svg"]:
        fig.savefig(folder / f"comparison.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)

    matrix = np.array([[row["completion_rate"] * 100 for row in agent["by_track"]] for agent in summaries])
    fig, ax = plt.subplots(figsize=(15, max(5, len(labels) * 0.35)), constrained_layout=True)
    heatmap = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(y, labels)
    ax.set_xticks(range(len(tracks)), [track.replace("_", " ") for track in tracks], rotation=55, ha="right", fontsize=9)
    for i in range(len(labels)):
        for j in range(len(tracks)):
            ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=7, color="white" if matrix[i,j] >= 60 else "#172033")
    ax.set_title(f"Completion by track (%) - {protocol['phase']}", loc="left", weight="bold", pad=12)
    fig.colorbar(heatmap, ax=ax, label="Completion (%)", shrink=0.6)
    fig.savefig(folder / "track_completion.png", dpi=170, facecolor="white")
    plt.close(fig)

    lines = ["# STK evaluation", "", f"Phase: `{protocol['phase']}`. Training campaign: `{protocol['training_campaign']}`.", "",
             f"Each agent ran {len(expected)} matched races: {len(tracks)} race tracks × {len(seeds)} seeds ({', '.join(map(str,seeds))}), "
             f"one lap, four karts, difficulty 2, {protocol['max_steps']} steps maximum. PPO uses deterministic inference.", "",
             "Rank agents by completion rate, then mean steps with failures capped at the step limit, then mean final position. "
             "Final position includes unfinished races; mean steps on completed races alone can favor agents that fail difficult tracks.", "",
             "| Agent | Completed | Rate | Mean capped steps | Mean final position | Mean inference (ms) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        latency = f"{row['mean_action_ms']:.3f}" if row["mean_action_ms"] is not None else "N/A"
        lines.append(f"| {row['model']} | {row['finished']}/{row['races']} | {row['completion_rate']:.1%} | {row['mean_capped_steps']:.1f} | {row['mean_position']:.2f} | {latency} |")
    lines += ["", f"Best PPO by this phase's ranking: **{best['model']}**.", "",
              "![Completion and capped race length](comparison.png)", "", "![Completion by track](track_completion.png)", "",
              "## Paired comparisons", ""]
    for baseline, row in comparisons.items():
        low, high = row["completion_difference_ci95_pp"]
        lines.append(f"- {best['model']} minus {baseline}: {row['completion_difference_pp']:+.1f} percentage points completion "
                     f"(95% track-bootstrap interval {low:+.1f} to {high:+.1f}); "
                     f"{row['capped_steps_difference']:+.1f} mean capped steps.")
    lines += ["", "## Interpretation and limits", "",
              "- The benchmark chooses among 14 training seeds. A model selected on these results needs fresh evaluation seeds to estimate performance after selection.",
              "- Training sampled random tracks from this same track pool. Track performance is not a test of unseen-track generalization.",
              "- Bootstrap intervals resample whole tracks, preserving the matched seeds. They describe variation across these tracks, not a guarantee on arbitrary races.",
              "- The built-in AI controls the full game action set. The PPO and heuristic use acceleration and steering only. AI execution occurs inside the simulator, so its actor latency is not comparable and is not reported.",
              "- These are local races against three built-in opponents, not ML-Arena Elo or head-to-head races between all submitted agents.",
              "- Raw returns are retained in the CSV/JSON but do not establish completion or race quality on their own.",
              "- No deployed policy is changed by evaluation.", "",
              "## Files", "",
              "- `protocol.json`: fixed settings and model-selection rule.",
              "- `results/`: per-race JSON and runtime/checkpoint provenance.",
              "- `races.csv`, `leaderboard.csv`, `summary.json`: complete machine-readable results.",
              "- `models/seed-XX/`: backed-up final inference weights, resumable PPO model and training records.", ""]
    (folder / "report.md").write_text("\n".join(lines))
    print(json.dumps({"best_ppo": best["model"], "agents": len(summaries), "races": len(all_rows), "report": str(folder / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
