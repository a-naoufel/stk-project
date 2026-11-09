"""Launch detached PPO jobs on PPTI and inspect or download their results."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from deploy_ppti import ROOT, run, ssh_command, ssh_options, source_archive

REMOTE_ROOT = "/temporary/21501292/stk-project"


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def active_trainers():
    trainers = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            args = (process / "cmdline").read_bytes().split(bytes([0]))
            if b"stk_actor.learn" in args:
                trainers.append(int(process.name))
        except OSError:
            continue
    return trainers


def status(job):
    data = json.loads((job / "status.json").read_text()) if (job / "status.json").exists() else {"state": "starting"}
    if data["state"] == "running":
        try:
            args = (Path("/proc") / str(data["pid"]) / "cmdline").read_bytes().split(bytes([0]))
            alive = b"worker" in args and str(job).encode() in args
        except OSError:
            alive = False
        if not alive:
            data["state"] = "interrupted"
    log = job / "train.log"
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 32_000))
            tail = stream.read().decode(errors="replace")
        counts = re.findall(r"total_timesteps\s*\|\s*(\d+)", tail)
        data["logged_steps"] = int(counts[-1]) if counts else 0
        data["log_tail"] = tail[-1500:]
    checkpoints = list((job / "run/checkpoints").glob("ppo_*_steps.zip"))
    if checkpoints:
        latest = max(checkpoints, key=lambda path: int(path.stem.split("_")[-2]))
        data["latest_checkpoint"] = str(latest)
        data["checkpoint_steps"] = int(latest.stem.split("_")[-2])
    return data


def worker(job):
    config = json.loads((job / "config.json").read_text())
    os.chdir(ROOT)
    lock_path = ROOT / "outputs/.ppti-training.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            save_json(job / "status.json", {"state": "skipped", "reason": "Another managed training job is running"})
            return
        existing = active_trainers()
        if existing:
            save_json(job / "status.json", {"state": "skipped", "reason": f"Existing trainers: {existing}"})
            return
        data = {"state": "running", "pid": os.getpid(), "started_at": now(), **config}
        save_json(job / "status.json", data)
        command = [
            "bash", "scripts/run_ppti.sh", "python", "-u", "-m", "stk_actor.learn",
            "--steps", str(config["steps"]), "--seed", str(config["seed"]),
            "--threads", str(config["threads"]), "--run-dir", str(job / "run"),
            "--output", str(job / "pystk_actor.pth"),
        ]
        try:
            with (job / "train.log").open("ab", buffering=0) as log:
                result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            data.update(state="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
        except Exception as exc:
            data.update(state="failed", error=str(exc))
        data["finished_at"] = now()
        save_json(job / "status.json", data)


def start_job(job, config):
    if job.exists():
        return status(job)
    if not (ROOT / ".ppti-env").is_file():
        raise RuntimeError("PPTI runtime has not been configured on this host")
    job.mkdir(parents=True)
    save_json(job / "config.json", config)
    with (job / "supervisor.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            ["nohup", sys.executable, str(Path(__file__).resolve()), "worker", "--job", str(job)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=ROOT, close_fds=True,
        )
    return {"state": "starting", "pid": process.pid, **config}


def remote_action(host, options, action, job, config=None):
    command = ["python3", "scripts/train_ppti.py", action, "--job", str(job)]
    if config is not None:
        command += ["--config", json.dumps(config)]
    remote = f"cd {shlex.quote(REMOTE_ROOT)} && " + shlex.join(command)
    return json.loads(run(ssh_command(options, None, host, remote), timeout=35))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "status", "collect", "worker", "start", "inspect"])
    parser.add_argument("--campaign", default=datetime.now(timezone.utc).strftime("ppo-%Y%m%d-%H%M%S"))
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.action in {"worker", "start", "inspect"}:
        if args.job is None or not args.job.is_absolute() or not args.job.is_relative_to(ROOT / "outputs/ppti-training"):
            parser.error("job must be an absolute directory under outputs/ppti-training")
        if args.action == "worker":
            worker(args.job)
        elif args.action == "start":
            print(json.dumps(start_job(args.job, json.loads(args.config))))
        else:
            print(json.dumps(status(args.job)))
        return
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.campaign):
        parser.error("campaign must be a simple directory name")
    cache = ROOT / ".cache/ppti"
    cache.mkdir(parents=True, exist_ok=True)
    known_hosts = cache / "known_hosts"
    if not known_hosts.is_file():
        known_hosts.write_bytes((Path.home() / ".ssh/known_hosts").read_bytes())
    options = ssh_options(Path.home() / ".ssh/config", known_hosts, 8)
    if args.action == "launch":
        preflight = json.loads((ROOT / "outputs/ppti-training-preflight.json").read_text())
        selected = [host for host, state in preflight.items() if state.get("ready") and not state.get("active_trainers")]
        blocked = {host: state for host, state in preflight.items() if host not in selected}
        if not selected:
            raise SystemExit("No idle, configured hosts in outputs/ppti-training-preflight.json")
        folder = ROOT / "outputs/ppti-training" / args.campaign
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            raise SystemExit(f"Campaign already exists: {manifest_path}; use status to inspect it")
        manifest = {"campaign": args.campaign, "created_at": now(), "steps_per_host": args.steps, "threads": args.threads, "hosts": {}, "unavailable": blocked}
        archive = source_archive([ROOT / "scripts/train_ppti.py", ROOT / "stk_actor/learn.py"])

        def launch(host):
            config = {"host": host, "seed": int(host.rsplit("-", 1)[1]) - 1, "steps": args.steps, "threads": args.threads}
            job = Path(REMOTE_ROOT) / "outputs/ppti-training" / args.campaign / f"seed-{config['seed']:02}"
            run(ssh_command(options, None, host, f"tar -xzf - -C {shlex.quote(REMOTE_ROOT)}"), input=archive, timeout=40)
            return {"job": str(job), **remote_action(host, options, "start", job, config)}

        targets = selected
        action = launch
    else:
        if args.manifest is None:
            parser.error("status and collect require --manifest")
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        folder = manifest_path.parent
        targets = list(manifest["hosts"])

        def inspect(host):
            row = manifest["hosts"][host]
            data = {"job": row["job"], **remote_action(host, options, "inspect", row["job"])}
            if args.action == "collect":
                destination = folder / host
                destination.mkdir(exist_ok=True)
                exclusions = ["--exclude=*.tmp"]
                if data["state"] != "completed":
                    # SB3 writes checkpoint zips directly. Avoid the newest
                    # zip and final exports while the worker may write them.
                    exclusions += ["--exclude=pystk_actor.pth", "--exclude=run/ppo.zip"]
                    if data.get("latest_checkpoint"):
                        exclusions.append("--exclude=" + Path(data["latest_checkpoint"]).name)
                transfer = ["rsync", "-a", *exclusions, "-e", shlex.join(["ssh", *options]), f"{host}:{row['job']}/", str(destination) + "/"]
                run(transfer, timeout=180)
                data["local_results"] = str(destination)
            return data

        action = inspect
    save_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(action, host): host for host in targets}
        for future in as_completed(futures):
            host = futures[future]
            try:
                result = future.result()
                manifest["hosts"][host] = result
                print(f"{host}: {result['state']}, seed {result.get('seed', '?')}, {result.get('logged_steps', 0):,} logged steps", flush=True)
            except Exception as exc:
                previous = manifest["hosts"].get(host, {})
                manifest["hosts"][host] = {**previous, "state": "unreachable", "error": str(exc)}
                print(f"{host}: {exc}", flush=True)
            save_json(manifest_path, manifest)
    manifest["checked_at"] = now()
    save_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
