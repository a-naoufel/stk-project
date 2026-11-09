"""Solo Lighthouse time trial shared by the expert and both learners."""

from hashlib import sha256
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pystk2
from pystk2_gymnasium import AgentSpec
from pystk2_gymnasium.envs import STKRaceEnv

from stk_actor.actors import DrivingActions, KartFeatures
from stk_actor.learn import ProgressReward

TRACK = "lighthouse"
STEP_SIZE = 0.1
MAX_STEPS = 1500
TIMEOUT_S = MAX_STEPS * STEP_SIZE
TRAIN_OFFSET_M = 1.5
EVAL_OFFSETS_M = (-2.0, -1.75, -1.25, -0.75, -0.25, 0.0, 0.25, 0.75, 1.25, 1.75, 2.0)
EVAL_SEED = 50_000
EXPERT_COMMIT = "ddf2cb6573850f890bba26d4e5f82888d9dac82a"
EXPERT_FILES = {
    "agents/kart_agent.py": "829a2218e6a53d3316a945298915783fafb0639015893ccea9ebeb325c2a6813",
    "agents/team3/agent3.py": "d052178b445f5807337f2c1730bb5a37fed57475551f11fd2dadf644c34502ad",
    "agents/team3/Pilot.py": "216d1d61704e0506f6739a024217b1a7e7693caddfe386d323324ba45e62fdbd",
    "agents/team3/Items.py": "59a43103e9f0b0eded32759627998aafee8c63af296c510a3c4f915fcd9174a5",
    "agents/team3/Rescue.py": "41fd1925fc8c966fa2b8c5b635dafa5e7151920184a07ce906d52e07f9cbef18",
}


class SoloTrial(STKRaceEnv):
    def __init__(self, seed, *, graphics_config=None):
        super().__init__(
            track=TRACK, laps=1, num_kart=1, difficulty=2,
            step_size=STEP_SIZE, use_subprocess=False,
            agent=AgentSpec(kart="tux", rank_start=0), graphics_config=graphics_config,
        )
        self.starts = np.random.default_rng(seed)

    def reset_race(self, random, *, options=None):
        super().reset_race(random, options=options)
        self.config.mode = pystk2.RaceConfig.RaceMode.TIME_TRIAL

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = int(self.starts.integers(2**31))
        offset = options["offset_m"] if options and "offset_m" in options else float(
            self.starts.uniform(-TRAIN_OFFSET_M, TRAIN_OFFSET_M)
        )
        heading = (options or {}).get("heading_deg", 0.0)
        _, info = super().reset(seed=seed, options=options)
        # The countdown overwrites the grid pose. Start control at GO.
        while self.world.phase in (pystk2.WorldState.Phase.READY_PHASE,
                                  pystk2.WorldState.Phase.SET_PHASE):
            self.race_step(pystk2.Action())
            self.world_update(False)
        if offset or heading:
            kart = self.world.karts[0]
            location = np.asarray(kart.location)
            x, _, z = np.asarray(kart.front) - location
            sideways = np.array([z, 0.0, -x])
            position = location + offset * sideways / np.linalg.norm(sideways)
            # Observations expose world-to-kart rotation in w,x,y,z order.
            # The pose setter expects the inverse in x,y,z,w order.
            w, x, y, z = map(float, kart.rotation)
            angle = np.deg2rad(heading) / 2
            c, s = np.cos(angle), np.sin(angle)
            rotation = (-c * x - s * z, -c * y + s * w, -c * z + s * x, c * w + s * y)
            pystk2.WorldState.set_kart_location(0, tuple(position), rotation, 0.0)
            self.world_update(False)
        obs = self.get_observation(0, False)
        self.expert_obs = obs
        return obs, info

    def step(self, action):
        result = super().step(action)
        self.expert_obs = result[0]
        return result


def make_trial(seed, *, graphics_config=None):
    race = gym.wrappers.TimeLimit(SoloTrial(seed, graphics_config=graphics_config), MAX_STEPS)
    return DrivingActions(KartFeatures(ProgressReward(race)))


def expert_factory(repo):
    """Use Team 3 from the instructor's checkout without loading its vendored env."""
    source = Path(repo).resolve() / "src"
    for name, expected in EXPERT_FILES.items():
        if sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{source / name} differs from expert commit {EXPERT_COMMIT}")
    sys.path.append(str(source))
    from agents.team3.agent3 import Agent3

    return Agent3


def expert_action(expert, race):
    raw = expert.choose_action({
        **race.expert_obs, "items_position": (), "items_type": (),
    })
    action = np.array([2 * float(raw["acceleration"]) - 1, float(raw["steer"])], dtype=np.float32)
    if not np.isfinite(action).all():
        raise ValueError(f"Expert returned a non-finite action: {action}")
    return np.clip(action, -1.0, 1.0)


def lap_result(env, finished, steps):
    seconds = float(env.unwrapped.world.karts[0].finish_time) if finished else None
    return {
        "finished": bool(finished), "steps": steps, "lap_s": seconds,
        "capped_s": seconds if finished else TIMEOUT_S,
    }


def evaluate_policy(env, policy, make_expert=None):
    rows = []
    for index, offset in enumerate(EVAL_OFFSETS_M):
        seed = EVAL_SEED + index
        obs, _ = env.reset(seed=seed, options={"offset_m": offset})
        expert = make_expert(env.unwrapped) if make_expert else None
        for step in range(1, MAX_STEPS + 1):
            action = expert_action(expert, env.unwrapped) if expert else policy(obs)
            obs, _, finished, truncated, _ = env.step(action)
            if finished or truncated:
                break
        rows.append({"seed": seed, "offset_m": offset, **lap_result(env, finished, step)})
    return rows
