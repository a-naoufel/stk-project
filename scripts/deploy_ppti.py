"""Copy STK to PPTI hosts and verify or install the project runtime."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tarfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def project_files():
    # An explicit source list keeps local credentials, virtualenvs, the untracked
    # ~/ directory, caches, and old training outputs off the teaching machines.
    names = ["README.md", "Makefile", "pyproject.toml", "uv.lock", ".python-version"]
    for directory, patterns in {
        "docs": ["*.md"],
        "scripts": ["*.py", "*.sh", "uv"],
        "stk_actor": ["*.py", "pystk_actor.pth"],
        "tests": ["test_*.py"],
    }.items():
        for pattern in patterns:
            names.extend(str(path.relative_to(ROOT)) for path in (ROOT / directory).glob(pattern))
    paths = [ROOT / name for name in sorted(set(names))]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular project file: {path}")
    return paths


def source_archive(paths):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    return buffer.getvalue()


def ssh_options(config, known_hosts, timeout):
    options = []
    if config:
        options += ["-F", str(config)]
    return options + [
        "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
        "-o", "ConnectionAttempts=1", "-o", "ControlPath=none",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
    ]


def ssh_command(options, gateway, host, command):
    connection = ["ssh", *options]
    if gateway:
        proxy = shlex.join(["ssh", *options, "-W", "%h:%p", gateway])
        connection += ["-o", f"ProxyCommand={proxy}"]
    return [*connection, host, command]


def run(command, **kwargs):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
    )
    output = result.stdout.decode(errors="replace")
    if result.returncode:
        raise RuntimeError(output.strip() or f"Command exited {result.returncode}")
    return output


def inspect_command(destination):
    return "\n".join([
        "hostname; id -un",
        "ls -ld /temprary /temporary 2>/dev/null || true",
        f"ls -ld -- {shlex.quote(destination)} 2>/dev/null || true",
        'printf "PPTI_ENV_PATH=%s\\nVIRTUAL_ENV=%s\\n" "${PPTI_ENV_PATH:-}" "${VIRTUAL_ENV:-}"',
        "command -v python python3 pystk2 uv || true",
        "for base in /opt /usr/local; do",
        '  find "$base" -maxdepth 4 -name pyvenv.cfg -readable -print 2>/dev/null || true',
        "done",
    ])


RUNTIME_CHECK = """
import importlib.metadata as metadata
import sys
import gymnasium, numpy, torch, pystk2_gymnasium, stable_baselines3, sb3_contrib
print('Python:', sys.version.split()[0], 'at', sys.executable)
for name in ('pystk2-gymnasium', 'gymnasium', 'numpy', 'torch', 'stable-baselines3', 'sb3-contrib'):
    print(name + ':', metadata.version(name))
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Select the course Python 3.12 environment using --env PATH.')
if pystk2_gymnasium.__version__ != '0.10.0':
    raise SystemExit('This project requires pystk2-gymnasium 0.10.0.')
"""


def deploy(host, args, options, archive):
    remote = lambda command: ssh_command(options, args.gateway, host, command)
    if args.inspect:
        return run(remote(shlex.join(["bash", "-lc", inspect_command(args.remote_dir)])), timeout=90)

    destination = shlex.quote(args.remote_dir)
    copy = "\n".join([
        "set -eu", "umask 077",
        f"test ! -L {destination}",
        f"mkdir -p -- {destination}",
        f'test "$(stat -c %u -- {destination})" = "$(id -u)"',
        f"tar -xzf - -C {destination}",
    ])
    output = run(remote(copy), input=archive, timeout=120)
    setup = ["set -euo pipefail", f"cd -- {destination}", "chmod u+x scripts/run_ppti.sh"]
    if args.proxy:
        setup += [
            f"export HTTPS_PROXY={shlex.quote(args.proxy)}",
            f"export HTTP_PROXY={shlex.quote(args.proxy)}",
            f"printf '%s\\n' {shlex.quote(args.proxy)} > .ppti-proxy",
        ]
    if args.bootstrap:
        setup += [
            'export UV_CACHE_DIR="$PWD/.cache/uv"',
            'export UV_PYTHON_INSTALL_DIR="$PWD/.python"',
            'export UV_SYSTEM_CERTS=true',
            'mkdir -p outputs',
            'sh scripts/uv sync --frozen --python 3.12 --no-dev 2>&1 | tee outputs/ppti-install.log',
            'printf "%s\\n" "$PWD/.venv" > .ppti-env',
        ]
    if args.env:
        setup.append(f"printf '%s\\n' {shlex.quote(args.env)} > .ppti-env")
    setup += [
        "hostname",
        shlex.join(["bash", "scripts/run_ppti.sh", "python", "-c", RUNTIME_CHECK]),
        "bash scripts/run_ppti.sh make test",
    ]
    if not args.skip_race:
        setup.append("bash scripts/run_ppti.sh make check")
    setup.append(f"printf '%s\\n' {shlex.quote('READY: ' + args.remote_dir)}")
    return output + run(remote(shlex.join(["bash", "-lc", "\n".join(setup)])), timeout=3600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=16)
    parser.add_argument("--remote-dir", default="/temporary/21501292/stk-project")
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--env", help="Use an existing PPTI environment directory")
    runtime.add_argument("--bootstrap", action="store_true", help="Install locked dependencies and Python 3.12 in the scratch project")
    parser.add_argument("--proxy", help="Web proxy URL for runtime and simulator asset downloads")
    parser.add_argument("--gateway", help="Override the gateway; default: use your SSH host configuration")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh/config")
    parser.add_argument("--connect-timeout", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--inspect", action="store_true", help="Read remote paths and environment locations only")
    parser.add_argument("--dry-run", action="store_true", help="List hosts and source files without connecting")
    parser.add_argument("--skip-race", action="store_true", help="Run unit tests without downloading STK assets/racing")
    args = parser.parse_args()
    if not 1 <= args.first <= args.last <= 16:
        parser.error("Use host numbers between 1 and 16, with first <= last")
    if args.jobs < 1 or args.connect_timeout < 1:
        parser.error("jobs and connect-timeout must be positive")
    if args.gateway and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.gateway):
        parser.error("gateway must be an SSH hostname or alias")
    destination = PurePosixPath(args.remote_dir)
    if not destination.is_absolute() or len(destination.parts) < 3 or ".." in destination.parts:
        parser.error("remote-dir must be an absolute project subdirectory, not a storage root")
    if args.env and (not args.env.startswith("/") or "\n" in args.env):
        parser.error("env must be an absolute environment directory")
    if args.proxy:
        proxy = urlsplit(args.proxy)
        if proxy.scheme not in {"http", "https"} or not proxy.hostname or any(char in args.proxy for char in "\r\n"):
            parser.error("proxy must be an HTTP or HTTPS URL")
    if not args.ssh_config.is_file():
        parser.error(f"SSH config not found: {args.ssh_config}")

    hosts = [f"ppti-14-407-{number:02}" for number in range(args.first, args.last + 1)]
    paths = project_files()
    print(f"Hosts: {', '.join(hosts)}\nProject directory: {args.remote_dir}", flush=True)
    if args.dry_run:
        for path in paths:
            print(f"  {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
        print("Dry run: no SSH connections or remote writes.")
        return

    cache = ROOT / ".cache/ppti"
    cache.mkdir(parents=True, exist_ok=True)
    known_hosts = cache / "known_hosts"
    if not known_hosts.exists():
        existing = Path.home() / ".ssh/known_hosts"
        if existing.is_file():
            shutil.copyfile(existing, known_hosts)
        else:
            known_hosts.touch(mode=0o600)
    options = ssh_options(args.ssh_config, known_hosts, args.connect_timeout)
    # Respect the host's default ProxyJump/ProxyCommand configuration. Check
    # transport before attempting to copy anything to the selected machines.
    try:
        run(ssh_command(options, args.gateway, hosts[0], "true"), timeout=args.connect_timeout + 20)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"SSH unavailable for {hosts[0]}; no project files copied.\n{exc}") from exc

    archive = None if args.inspect else source_archive(paths)
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(hosts))) as pool:
        futures = {pool.submit(deploy, host, args, options, archive): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            try:
                print(f"\n[{host}]\n{future.result()}", flush=True)
            except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
                failures.append(host)
                print(f"\n[{host}] FAILED: {exc}", flush=True)
    if failures:
        raise SystemExit(f"Failed on {len(failures)} host(s): {', '.join(sorted(failures))}")
    print(f"{'Inspected' if args.inspect else 'Verified'} {len(hosts)} host(s).")


if __name__ == "__main__":
    main()
