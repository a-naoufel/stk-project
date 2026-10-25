"""Shared features, actions, baseline, and CPU policy loading."""

import math
from pathlib import Path

import gymnasium as gym
import numpy as np

FEATURE_VERSION = 1
N_PATHS = 5
FEATURE_DIM = 32
NET_ARCH = [64, 64]


class KartFeatures(gym.ObservationWrapper):
    """Bounded features: endpoints (15), widths (5), masks (5), center (3),
    velocity (3), energy (1). Version this layout when changing it.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(FEATURE_DIM,), dtype=np.float32
        )

    def observation(self, obs):
        ends = np.asarray(obs["paths_end"], dtype=np.float32).reshape(-1, 3)
        widths = np.asarray(obs["paths_width"], dtype=np.float32).reshape(-1)
        count = min(N_PATHS, len(ends), len(widths))
        paths = np.zeros((N_PATHS, 3), dtype=np.float32)
        width = np.zeros(N_PATHS, dtype=np.float32)
        mask = np.zeros(N_PATHS, dtype=np.float32)
        paths[:count] = ends[:count] / 50.0
        width[:count] = widths[:count] / 20.0
        mask[:count] = 1.0
        features = np.concatenate([
            paths.ravel(), width, mask,
            np.asarray(obs["center_path"], dtype=np.float32) / 20.0,
            np.asarray(obs["velocity"], dtype=np.float32) / 30.0,
            np.asarray(obs["energy"], dtype=np.float32).reshape(1) / 10.0,
        ])
        return np.clip(np.nan_to_num(features), -1.0, 1.0).astype(np.float32)


class DrivingActions(gym.ActionWrapper):
    """Normalized throttle and steering. Other controls are off for this starter."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def action(self, action):
        throttle, steer = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        return {
            "acceleration": np.array([(throttle + 1.0) / 2.0], dtype=np.float32),
            "steer": np.array([steer], dtype=np.float32),
            "brake": 0, "drift": 0, "nitro": 0, "rescue": 0, "fire": 0,
        }


def make_heuristic_actor():
    """A transparent path follower to check the simulator before learning."""

    def actor(state, obs):
        paths = obs[:15].reshape(N_PATHS, 3) * 50.0
        valid = obs[20:25] > 0
        candidates = paths[valid & (paths[:, 2] > 6.0)]
        if len(candidates) == 0:
            candidates = paths[valid]
        target = candidates[0] if len(candidates) else np.array([0, 0, 10])
        angle = math.atan2(float(target[0]), max(float(target[2]), 0.1))
        steer = np.clip(2.0 * angle, -1.0, 1.0)
        speed = float(np.linalg.norm(obs[28:31]) * 30.0)
        target_speed = 16.0 if abs(angle) < 0.25 else 9.0
        throttle = 1.0 if speed < target_speed else 0.15
        return np.array([2.0 * throttle - 1.0, steer], dtype=np.float32)

    return actor


def load_policy_actor(path: Path, observation_space, action_space):
    """Load weights on CPU; fail explicitly for incompatible checkpoints."""
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy

    torch.set_num_threads(3)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint feature version does not match actors.py")
    policy = ActorCriticPolicy(
        observation_space, action_space, lr_schedule=lambda _: 0.0,
        net_arch=payload["net_arch"],
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.to("cpu")
    policy.set_training_mode(False)

    def actor(state, obs):
        action, _ = policy.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return actor
