# Run STK on PPTI room 14-407

The deployment script targets `ppti-14-407-01` through `ppti-14-407-16`
using the default gateway from your existing SSH config. It copies the
current source files, including local dependency changes, without copying
credentials, Git history, local Python installations, or previous training
outputs. An installed `stk_actor/pystk_actor.pth` is included if present.

PPTI documents `/temporary` as its scratch storage:
<https://mind.sorbonne-universite.fr/informations/logiciel/>.
The confirmed project destination is `/temporary/21501292/stk-project`,
which is also the script default. Use `--remote-dir` for another destination.
It does not use sudo. Prefer the teacher's existing environment with `--env`;
when none is available, `--bootstrap` installs the locked runtime under the
scratch project directory.

## Inspect before deployment

```bash
python3 scripts/deploy_ppti.py --first 1 --last 1 --inspect
python3 scripts/deploy_ppti.py --dry-run
```

The inspection prints storage availability, active Python commands, and
candidate preinstalled environments. Obtain the RL environment path from
the teacher's course/Mattermost instructions. `--env` is the directory
containing its `bin/python`, not the activation script.

The course activation command supplied on 6 November 2026 is:

```bash
source /users/Enseignants/piwowarski/mind-rl/.venv/bin/activate
```

It was checked on host 01: Python resolves to 3.13.5, while the installed
packages are under `lib/python3.11/site-packages`, and importing `gymnasium`
fails. The project therefore uses its own locked Python 3.12 runtime in
scratch storage. The teacher's environment was not modified.

## Deploy and verify

For the documented scratch location, using your PPTI login:

```bash
python3 scripts/deploy_ppti.py \
  --remote-dir /temporary/21501292/stk-project \
  --env /absolute/path/to/the/course-environment
```

Replace the example environment path with the actual preinstalled path.
If the correct environment is already activated by the remote login shell,
omit `--env`. The chosen explicit path is saved in `.ppti-env` on each host.
You can also set `PPTI_ENV_PATH` when running commands directly.

When the course environment is unavailable, install the locked dependencies
and Python 3.12 using the campus proxy:

```bash
python3 scripts/deploy_ppti.py --bootstrap \
  --proxy http://proxy.ufr-info-p6.jussieu.fr:3128
```

This puts the virtual environment in `.venv/`, Python in `.python/`, and the
package cache in `.cache/uv/` inside the scratch project. The selected runtime
and proxy are recorded in `.ppti-env` and `.ppti-proxy` on each host. The
runner reads these files without executing them as shell code.

Each host checks dependency imports, Python 3.12 and pystk2-gymnasium 0.10.0,
runs the unit tests, then runs the instructor's 100-step headless race check.
Use `--skip-race` to omit the simulator check. `--first N --last N` retries
one host; `--jobs` controls concurrent setups (default two).

The copy updates source files but retains existing `outputs/`, caches, and
checkpoints outside the copied source list. Back up a deployed checkpoint
before copying a different local checkpoint over it. Runtime checks report
installed dependency versions. `--bootstrap` installs packages into the
project's isolated scratch environment.

## Run on a configured host

```bash
ssh ppti-14-407-01
cd /temporary/21501292/stk-project
make test
make race
bash scripts/run_ppti.sh python -m stk_actor.learn \
  --steps 100000 --seed 0 --output outputs/seed0/pystk_actor.pth
```

Use a different seed/output directory on each machine. Setup does not start
long training runs. The runner uses the selected Python installation and puts
XDG caches under the project directory, with three CPU threads by default.
The Makefile detects `.ppti-env` and invokes the runner automatically for
normal commands. On a configured PPTI checkout, `make setup` checks the
selected runtime instead of creating another environment.

To view a race dashboard, connect from your computer with port forwarding:

```bash
ssh -L 8050:localhost:8050 ppti-14-407-01
cd /temporary/21501292/stk-project
make dashboard
```

Open <http://localhost:8050> and click **Run**. PPTI workstations may restart
daily; copy experiment results back to persistent storage regularly.

## Training jobs

On 6 November 2026, detached PPO runs were launched on hosts 01 and 04–16.
Campaign: `ppo-20261106-220242`. Each learns independently from scratch for
1,000,000 requested steps on random tracks, with four karts, three CPU
threads and a seed equal to the host number minus one. These are separate
learners; their weights are not shared between machines. SB3 rounds the
training budget up to complete its 1,024-step rollouts.

All 14 jobs completed successfully at 1,000,448 steps on 6 November, between
23:29 and 23:31 Paris time. Final inference weights and resumable models
are backed up locally under `outputs/ppti-evaluation/benchmark-20261107/models/`.
See [the completed evaluation](EVALUATION.md) for the model comparison.

For example, host 01 writes to:

```text
/temporary/21501292/stk-project/outputs/ppti-training/ppo-20261106-220242/seed-00/
```

Each job contains `config.json`, `status.json`, `train.log`, a `run/`
directory with `monitor.csv` and checkpoints every 10,000 steps, plus
`pystk_actor.pth` and `run/ppo.zip` after successful completion. Jobs survive
SSH disconnection; a workstation reboot interrupts them. Existing deployed
weights are not used or overwritten by these runs.

Check progress from your computer:

```bash
python3 scripts/train_ppti.py status \
  --manifest outputs/ppti-training/ppo-20261106-220242/manifest.json
```

Download logs and saved results into the local campaign directory:

```bash
python3 scripts/train_ppti.py collect \
  --manifest outputs/ppti-training/ppo-20261106-220242/manifest.json
```

While a job is running, collection omits its newest checkpoint zip and final
exports to avoid copying files being written. Older checkpoint zips are
included. Once completed, all final results are copied. Status checks report
an unreachable host separately from a failed or interrupted training job.

To follow host 01's log directly:

```bash
ssh ppti-14-407-01
cd /temporary/21501292/stk-project
tail -f outputs/ppti-training/ppo-20261106-220242/seed-00/train.log
```

`scripts/train_ppti.py launch` uses the host preflight saved in
`outputs/ppti-training-preflight.json`, checks for existing trainers and
refuses to start a second managed job on the same machine. Use a new
`--campaign` for a later experiment. Evaluate finished checkpoints across
tracks and seeds before selecting one to deploy.

## Setup status — 6 November 2026

The destination is `/temporary/21501292/stk-project` on each machine.
Connections use the user's default SSH configuration (`ProxyJump ppti`).
Earlier gateway timeouts were resolved after the user established their
default SSH connection.

Hosts 01 and 04–16 passed all four actor tests and the instructor's 100-step
Lighthouse race check: mean/max action time 0.1/0.2 ms, peak memory 176–179 MB.
Hosts 02 and 03 timed out on repeated connection attempts.

The installed runtime is Python 3.12.14, pystk2-gymnasium 0.10.0,
gymnasium 1.3.0, numpy 2.2.6, torch 2.12.0+cpu, SB3 2.9.0 and
SB3-Contrib 2.9.0. The official STK 1.5 assets are cached inside the project.
The dependency files preserve the user's existing numpy 2.2.6 change.

Each machine installed independently through the campus proxy. To retry an
unavailable host when it is up, run on your computer:

```bash
python3 scripts/deploy_ppti.py --first 2 --last 2 \
  --bootstrap \
  --proxy http://proxy.ufr-info-p6.jussieu.fr:3128
```

On host 01, plain `make setup`, `make test`, `make smoke` and `make race`
also passed. The isolated PPO smoke run trained for 256 steps and exported
`outputs/smoke/pystk_actor.pth`; the resumable run is under
`outputs/ppo-20261106-203042-901338/`. The heuristic completed the full
Lighthouse race in 587 steps, finishing fourth.

The default agent remains the heuristic. The smoke checkpoint is a pipeline
check, without evidence of useful racing performance. No ML-Arena upload or
deployment was performed.
