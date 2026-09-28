# PySTK2-gymnasium project template

This project template contains a basic structure for a PySTK2 reinforcement learning agent.
For information about the PySTK2 gymnasium environment, please look at the [corresponding github page](https://github.com/bpiwowar/pystk2-gymnasium)

## Setup

### On PPTI machines

A Python environment with all the required libraries is already installed:
**do not create your own**, just activate it (in every new terminal):

```sh
source PPTI_ENV_PATH/bin/activate
```

and define the `PYSTK2` and `PYTHON` aliases used in the commands below:

```sh
alias PYSTK2=pystk2
alias PYTHON=python
```

### On your own computer

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment
(see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/)).

```sh
# To be run from the base directory: creates a virtual environment in .venv
uv venv
# Install the PySTK2 gymnasium environment with the race runner (cli) and
# the web dashboard (web) -- add any other library you need
uv pip install "pystk2-gymnasium[cli,web]"
```

Then define the `PYSTK2` and `PYTHON` aliases used in the commands below (in every new
terminal); `uv run` runs the command within this environment:

```sh
alias PYSTK2="uv run pystk2"
alias PYTHON="uv run python"
```

## Structure

**Warning**: all the imports should be relative within your module (see `learn.py` for an example).

### `actors.py`

Contains the actors used throughout your project

### `learn.py`

Contains the code to train your actor

### `pystk_actor.py`

This Python file (**don't change its name**) should contain:

- `env_name`: The base environment name
- `player_name`: The actor name (displayed on top of the kart)
- `get_wrappers()` *(optional)*: Returns a list of extra wrappers applied to the environment
- `get_actor(module_dir, observation_space, action_space)`: Returns a callable `actor(state, obs) -> action`. The `module_dir` parameter is the path to the directory containing this file (use it to load saved weights, e.g. `module_dir / "pystk_actor.pth"`)
- `create_state()` *(optional)*: Called once per race to create a fresh state object; the state is then passed as the first argument to the actor on every step. Return `None` for stateless agents (the default if not defined).

### Unified actor interface

All actors use the same signature: `actor(state, obs) -> action`.

The `create_state()` function determines whether the agent carries state:

**Stateless** (`create_state` returns `None`):

```python
# pystk_actor.py
def create_state():
    return None

def get_actor(module_dir, observation_space, action_space):
    def actor(state, obs):  # state is None
        return action_space.sample()
    return actor
```

**Stateful** (`create_state` returns a dict or any object):

```python
# pystk_actor.py
def create_state():
    return {"step": 0, "history": []}

def get_actor(module_dir, observation_space, action_space):
    def actor(state, obs):
        state["step"] += 1
        state["history"].append(obs)
        return action_space.sample()
    return actor
```

## Learn your model

```sh
# To be run from the base directory
PYTHONPATH=. PYTHON -m stk_actor.learn
```

This should create the `pystk_actor.pth` file (**don't change its name**) that contains the parameters of your model. Load it in `get_actor` using `module_dir / "pystk_actor.pth"`.

## Testing the actor

To test your agent locally:

```sh
PYTHONPATH=. PYSTK2 race --hide stk_actor
```

You can also compare different actors in the same race:

```sh
PYTHONPATH=. PYSTK2 race --hide stk_actor other_actor
```

### Interacting with the race (web dashboard)

To see what your agent is doing, use the `--web` option: it starts a web
server displaying the track map (3D view with the karts positions) and
information about each kart (observations, rewards).

```sh
PYTHONPATH=. PYSTK2 race --hide --web stk_actor
```

Then open <http://localhost:8050> in your browser (use `--web-port PORT` to
change the port, e.g. if several people share the same machine). The race
starts **paused**; use the buttons to control it:

- **Step**: advance the race by one step
- **Run** / **Pause**: run the race continuously / pause it
- **Stop**: end the race

Other useful options: `--track TRACK` (choose the track), `--num-karts N`,
`--laps N`, `--max-steps N`, and `--record race.mp4` to save a video of the race
(see `PYSTK2 race --help` for the full list).

## Submit your work on the `evaluation` branch

1. **Commit your final work**

   ```bash
   # Add files
   git add ...
   git commit -m "Final version for evaluation"
   ```

   Ensure that all relevant files are versioned with `git status` -- in particular the `pystk_actor.pth`.

2. **Switch (or create) the `evaluation` branch**

   ```bash
   git fetch origin
   git checkout -B evaluation origin/evaluation || git checkout -b evaluation
   ```


3. **Merge your work and push**

   ```bash
   git merge main        # or your working branch
   git push origin evaluation
   ```

   In case of problem with the `pystk_actor.pth` file, use this
   ```bash
    # overwrite binary file
    git checkout main -- stk_actor/pystk_actor.pth
    git add stk_actor/pystk_actor.pth
   ```

4. **Check online** that the `evaluation` branch contains your latest commit.
