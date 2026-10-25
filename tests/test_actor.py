import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

torch.set_num_threads(3)

from stk_actor.actors import (
    FEATURE_DIM, FEATURE_VERSION, NET_ARCH, DrivingActions, KartFeatures,
    load_policy_actor, make_heuristic_actor,
)


class DummyKart(gym.Env):
    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Dict({
        "acceleration": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
        "steer": gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
        **{name: gym.spaces.Discrete(2) for name in ["brake", "drift", "nitro", "rescue", "fire"]},
    })


class ActorTests(unittest.TestCase):
    def observation(self, endpoints):
        return {
            "paths_end": endpoints, "paths_width": [[10]] * len(endpoints),
            "center_path": [0, 0, 0], "velocity": [0, 0, 0], "energy": [0],
        }

    def test_variable_paths_and_finite_features(self):
        wrapper = KartFeatures(DummyKart())
        for endpoints in [[], [[0, 0, 10]], [[1e9, float("nan"), 10]] * 8]:
            vector = wrapper.observation(self.observation(endpoints))
            self.assertEqual(vector.shape, (FEATURE_DIM,))
            self.assertTrue(wrapper.observation_space.contains(vector))

    def test_steering_and_zero_padding(self):
        wrapper = KartFeatures(DummyKart())
        actor = make_heuristic_actor()
        for x, sign in [(-5, -1), (0, 0), (5, 1)]:
            obs = wrapper.observation(self.observation([[x, 0, 10]]))
            action = actor(None, obs)
            self.assertEqual(np.sign(action[1]), sign)
            self.assertTrue(DrivingActions(wrapper).action_space.contains(action))
        self.assertTrue(np.isfinite(actor(None, wrapper.observation(self.observation([])))).all())

    def test_actions_match_simulator_space(self):
        raw = DummyKart()
        wrapper = DrivingActions(raw)
        for action in [[-1, -1], [1, 1], [0, 0], [100, -100]]:
            self.assertTrue(raw.action_space.contains(wrapper.action(action)))
        self.assertEqual(float(wrapper.action([-1, 0])["acceleration"][0]), 0)
        self.assertEqual(float(wrapper.action([1, 0])["acceleration"][0]), 1)

    def test_checkpoint_reproduces_policy_and_rejects_feature_change(self):
        env = DrivingActions(KartFeatures(DummyKart()))
        policy = ActorCriticPolicy(
            env.observation_space, env.action_space, lambda _: 0.0, net_arch=NET_ARCH
        )
        obs = np.zeros(FEATURE_DIM, dtype=np.float32)
        expected, _ = policy.predict(obs, deterministic=True)
        payload = {"feature_version": FEATURE_VERSION, "net_arch": NET_ARCH,
                   "policy_state_dict": policy.state_dict()}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pystk_actor.pth"
            torch.save(payload, path)
            actor = load_policy_actor(path, env.observation_space, env.action_space)
            np.testing.assert_allclose(actor(None, obs), expected, atol=1e-6)
            payload["feature_version"] = -1
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_policy_actor(path, env.observation_space, env.action_space)


if __name__ == "__main__":
    unittest.main()
