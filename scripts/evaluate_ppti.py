"""Evaluate final PPTI models on identical races and back up final weights."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from deploy_ppti import ROOT, run, ssh_command, ssh_options, source_archive

REMOTE_ROOT = "/temporary/21501292/stk-project"
TRACKS = [
    "abyss", "black_forest", "candela_city", "cocoa_temple", "cornfield_crossing",
    "fortmagma", "gran_paradiso_island", "hacienda", "lighthouse", "mines",
    "minigolf", "olivermath", "ravenbridge_mansion", "sandtrack", "scotland",
    "snowmountain", "snowtuxpeak", "stk_enterprise", "volcano_island", "xr591", "zengarden",
]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", default="benchmark-20261107")
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--tracks", default=",".join(TRACKS))
    parser.add_argument("--hosts", help="Optional comma-separated host names")
    parser.add_argument("--jobs", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.phase):
        parser.error("phase must be a simple directory name")
    tracks = args.tracks.split(",")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not set(tracks) <= set(TRACKS) or len(tracks) != len(set(tracks)) or len(seeds) != len(set(seeds)):
        parser.error("tracks must be available race tracks; tracks and seeds must be unique")
    if args.jobs < 1 or args.max_steps < 1:
        parser.error("jobs and max-steps must be positive")
    manifest = json.loads(args.manifest.read_text())
    hosts = sorted(args.hosts.split(",") if args.hosts else manifest["hosts"])
    if len(hosts) < 2 or not set(hosts) <= set(manifest["hosts"]):
        parser.error("select at least two hosts present in the training manifest")
    if any(manifest["hosts"][host]["state"] != "completed" for host in hosts):
        parser.error("all selected training runs must have completed")
    output = ROOT / "outputs/ppti-evaluation" / args.phase
    for name in ["results", "logs", "models"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": args.phase, "training_campaign": manifest["campaign"], "hosts": hosts,
        "tracks": tracks, "seeds": seeds, "races_per_agent": len(tracks) * len(seeds),
        "max_steps": args.max_steps, "num_karts": 4, "laps": 1, "difficulty": 2,
        "selection_rule": "completion rate descending, mean capped steps ascending, mean position ascending",
        "checkpoint_inference": "deterministic", "baselines": ["heuristic", "ai"],
        "tracks_note": "Training used random tracks from this same set; these are not held-out tracks.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if {k: v for k, v in previous.items() if k != "created_at"} != {k: v for k, v in protocol.items() if k != "created_at"}:
            parser.error("phase already has a different evaluation protocol")
    else:
        write_json(protocol_path, protocol)
    options = ssh_options(Path.home() / ".ssh/config", ROOT / ".cache/ppti/known_hosts", 10)
    transport = shlex.join(["ssh", *options])
    archive = source_archive([ROOT / "scripts/evaluate.py"])
    report = {}

    def evaluate(host):
        training = manifest["hosts"][host]
        job = training["job"]
        seed = training["seed"]
        model_dir = output / "models" / f"seed-{seed:02}"
        model_dir.mkdir(exist_ok=True)
        files = [job + suffix for suffix in [
            "/pystk_actor.pth", "/run/ppo.zip", "/run/run.json", "/run/monitor.csv",
            "/status.json", "/config.json",
        ]]
        run(["rsync", "-a", "-e", transport, *[f"{host}:{name}" for name in files], str(model_dir) + "/"], timeout=90)
        sha = hashlib.sha256((model_dir / "pystk_actor.pth").read_bytes()).hexdigest()
        run(ssh_command(options, None, host, f"tar -xzf - -C {shlex.quote(REMOTE_ROOT)}"), input=archive, timeout=40)
        remote_dir = f"{REMOTE_ROOT}/outputs/ppti-evaluation/{args.phase}"
        agents = [(f"ppo-seed-{seed:02}", "checkpoint")]
        if host == hosts[0]:
            agents.append(("heuristic", "heuristic"))
        if host == hosts[1]:
            agents.append(("ai", "ai"))
        collected = []
        for label, agent in agents:
            remote_output = f"{remote_dir}/{label}.json"
            command = [
                "bash", "scripts/run_ppti.sh", "python", "-u", "-m", "scripts.evaluate",
                "--agent", agent, "--tracks", args.tracks, "--seeds", args.seeds,
                "--max-steps", str(args.max_steps), "--output", remote_output,
            ]
            if agent == "checkpoint":
                command += ["--checkpoint", f"{job}/pystk_actor.pth"]
            remote = f"cd {shlex.quote(REMOTE_ROOT)} && " + shlex.join(command)
            with (output / "logs" / f"{label}.log").open("wb") as log:
                result = subprocess.run(ssh_command(options, None, host, remote), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if result.returncode:
                raise RuntimeError(f"{label} failed with exit {result.returncode}; see logs/{label}.log")
            run(["rsync", "-a", "-e", transport, f"{host}:{remote_output}", f"{host}:{remote_output[:-5]}.meta.json", str(output / "results") + "/"], timeout=60)
            rows = json.loads((output / "results" / f"{label}.json").read_text())
            metadata = json.loads((output / "results" / f"{label}.meta.json").read_text())
            expected = {(track, seed) for track in tracks for seed in seeds}
            actual = {(row["track"], row["seed"]) for row in rows}
            if len(rows) != len(expected) or actual != expected:
                raise RuntimeError(f"{label}: incomplete or duplicated evaluation rows")
            if agent == "checkpoint" and metadata["checkpoint_sha256"] != sha:
                raise RuntimeError(f"{label}: downloaded weights do not match evaluated checkpoint")
            if "finished_at" not in metadata:
                raise RuntimeError(f"{label}: missing completion metadata")
            completed = sum(row["finished"] for row in rows)
            print(f"{label} ({host}): {completed}/{len(rows)} races finished", flush=True)
            collected.append({"label": label, "races": len(rows), "finished": completed, "checkpoint_sha256": metadata["checkpoint_sha256"]})
        return {"state": "completed", "agents": collected, "seed": seed}

    with ThreadPoolExecutor(max_workers=min(args.jobs, len(hosts))) as pool:
        futures = {pool.submit(evaluate, host): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            try:
                report[host] = future.result()
            except Exception as exc:
                report[host] = {"state": "failed", "error": str(exc)}
                print(f"{host}: FAILED: {exc}", flush=True)
            write_json(output / "execution.json", report)
    if any(row["state"] != "completed" for row in report.values()):
        raise SystemExit("Some evaluations failed; see execution.json and per-agent logs")
    print(f"All evaluations complete: {output}")


if __name__ == "__main__":
    main()
