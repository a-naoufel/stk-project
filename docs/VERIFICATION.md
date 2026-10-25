# Local verification — 23 October 2026

Environment: Python 3.12.15, pystk2-gymnasium 0.10.0, PySuperTuxKart2 0.7.4,
torch 2.12.0+cpu, SB3/SB3-Contrib 2.9.0, gymnasium 1.3.0, numpy 2.5.3.
The official STK assets have been downloaded and extracted successfully.

| Check | Result |
| --- | --- |
| `make test` | All four tests passed: variable-length features, steering, action bounds, and checkpoint reload/version rejection |
| `./scripts/uv sync --frozen` | Passed; all 46 installed packages match the lockfile |
| `make check` | Instructor's flat loader ran the heuristic for 100 steps on Lighthouse; mean/max action latency 0.7/1.4 ms, reported peak memory 172 MB |
| Heuristic full race | Lighthouse, seed 0, four karts: finished in 597 steps, position 4, distance 880.51, no truncation |
| `make smoke` | PPO trained for 256 steps and exported `outputs/smoke/pystk_actor.pth` plus a resumable SB3 model |
| Learned flat upload | The instructor's validator loaded that smoke checkpoint in a temporary flat folder and ran 100 steps; mean/max action latency 1.6/3.8 ms, reported peak memory 432 MB |
| `make package` | Created `dist/stk-agent.zip` with `agent.py`, `actors.py`, and `pystk_actor.py` at the ZIP root |
| `git diff --check` | Passed |

The instructor's `agent.py` is unchanged (SHA-256
`1271c7308fd5b854c79dea1755031353a6c2f98c502de62c3e1be62b7b569d1c`).
The default agent remains the heuristic; the smoke weights were not copied
into `stk_actor/`. These local setup checks do not establish competitive PPO
performance. No submission was uploaded or deployed.

Race metrics: `outputs/heuristic-lighthouse-seed0.json`.
Smoke run: `outputs/ppo-20261023-182208-487663/`.

Git branch: `codex/bootstrap`. The instructor's remote is named `upstream`;
the destination folder's existing `origin` remote was retained during relocation.
Changes are left uncommitted for review. The internship repository was not modified for this project.
