# STK project — Master MIND

Bootstrapped from the [instructor's STK template](https://git.isir.upmc.fr/bpiwowar/pystk2-project-template).
The original platform adapter `stk_actor/agent.py` is unchanged.

```bash
cd /home/naoufel/Documents/stk-project
make test       # feature/action and checkpoint checks
make check      # instructor's flat-upload check and 100-step race
make race       # run the current agent headlessly
make dashboard  # visual dashboard at http://127.0.0.1:8050; starts paused
```

When the dashboard terminal shows `Racing: 0step`, open
<http://localhost:8050> and click **Run** to start the race, or **Step** to
advance once. For a headless race that starts immediately, use `make race`.

Python 3.12 and the CPU runtime are installed in `.venv`. The dependencies
match the template's ML-Arena runtime: pystk2-gymnasium 0.10.0, torch 2.12,
SB3/SB3-Contrib 2.9, gymnasium 1.3.0, numpy 2.5.3. `uv.lock` pins transitive
dependencies. `./scripts/uv` uses system uv if available, otherwise the local
`.tools/uv` binary. Recreate the environment with `make setup`.

The first simulator run downloads the official STK assets into the user's
cache. Graphics are optional: the normal checks and training run headlessly.

## Agent and training

With no `stk_actor/pystk_actor.pth`, the agent uses a small path-following
heuristic. It is a baseline for getting the project running, not a trained
policy or a claim of competitive performance. Both agents use the same
32 bounded features and two controls (throttle and steer); advanced controls
remain off. Feature and action code lives in `stk_actor/actors.py`.

```bash
make smoke     # 256 PPO steps, isolated checkpoint under outputs/smoke/
make train     # 100,000 PPO steps on random tracks, CPU
```

`make train` writes deployment weights to `stk_actor/pystk_actor.pth`.
Once that file exists, the current agent automatically loads it. Keep or move
an existing checkpoint before training if you want to preserve it.
Logs, episode monitor CSV, configuration, a resumable `ppo.zip`, and periodic
checkpoints go to `outputs/ppo-*/`. The smoke test leaves the deployed
heuristic in place. A short smoke run does not produce a useful trained agent.

For a specific experiment:

```bash
.venv/bin/python -m stk_actor.learn --steps 100000 --seed 1 --track lighthouse
.venv/bin/python -m stk_actor.learn --steps 100000 --resume outputs/RUN/ppo.zip
.venv/bin/python -m scripts.evaluate --agent deployed --tracks lighthouse,zengarden --seeds 0,1,2
.venv/bin/python -m scripts.evaluate --agent checkpoint --checkpoint outputs/smoke/pystk_actor.pth --seeds 0 --max-steps 100
```

Evaluation records completion, truncation, distance, position, return, and
action latency in `outputs/evaluation.json`. Use `--output` to keep separate
results, and `--agent random`, `heuristic`, or `ai` for comparison. The AI
option gives control to the game's built-in bot.

## Submission preparation

```bash
make check
make package   # creates dist/stk-agent.zip locally
```

The ZIP contains the flat agent files selected by the instructor's rules;
training code, caches and credentials are excluded. ML-Arena sign-in,
enrollment and submission are still yours to complete. No files have been
uploaded, and no submission has been deployed. The inherited `scripts/submit.py`
performs real uploads/deployment; it is not called by any Makefile target.

See [course context and next experiments](docs/COURSE_CONTEXT.md),
[the verified local results](docs/VERIFICATION.md),
[the AI assistance log](docs/AI_USE.md), and [the instructor README](docs/UPSTREAM_README.md).
Checkpoints are intentionally versionable: commit the chosen `pystk_actor.pth`
when you decide which model to hand in. Keep `.env` and API keys out of Git.
