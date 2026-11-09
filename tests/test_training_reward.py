from types import SimpleNamespace
import unittest

import gymnasium as gym
from pystk2 import WorldState

from stk_actor.learn import ProgressReward


class RaceTrace(gym.Env):
    """A single controlled kart driven by a scripted sequence of states."""

    kart_ix = 0
    num_kart = 4
    track = SimpleNamespace(length=100.0)
    config = SimpleNamespace(step_size=0.1)

    def reset(self, **kwargs):
        self.world = SimpleNamespace(phase=WorldState.Phase.READY_PHASE, karts=[None])
        self.set_kart(-100.0, 0.0)
        return None, {}

    def set_kart(self, distance, x, on_road=True):
        self.world.karts[0] = SimpleNamespace(
            overall_distance=distance, location=(x, 0.0, 0.0), is_on_road=on_road,
        )

    def step(self, action):
        distance, x, on_road, finished, truncated = action
        self.set_kart(distance, x, on_road)
        return None, 100.0, finished, truncated, {"position": 1}


class ProgressRewardTests(unittest.TestCase):
    def setUp(self):
        self.race = RaceTrace()
        self.env = ProgressReward(self.race)
        self.env.reset()
        self.race.world.phase = WorldState.Phase.RACE_PHASE

    def reward(self, distance, x, on_road=True, finished=False, truncated=False):
        return self.env.step((distance, x, on_road, finished, truncated))[1]

    def test_start_line_wrap_and_repeated_progress(self):
        self.assertAlmostEqual(self.reward(0.5, 0.5), 0.03)
        self.assertAlmostEqual(self.reward(99.5, -0.5), -0.02)
        self.assertAlmostEqual(self.reward(0.5, 0.5), -0.02)
        self.assertAlmostEqual(self.reward(1.0, 1.0), 0.03)

    def test_projection_jump_and_rescue_earn_no_progress(self):
        self.assertAlmostEqual(self.reward(2.0, 2.0), 0.18)
        self.assertAlmostEqual(self.reward(42.0, 2.05), -0.02)
        self.assertAlmostEqual(self.reward(2.0, 2.05), -0.02)
        self.assertAlmostEqual(self.reward(5.0, 5.0, on_road=False), -0.10)
        self.assertAlmostEqual(self.reward(1.0, -12.0), -0.02)
        self.assertAlmostEqual(self.reward(2.0, -11.0), -0.02)
        self.assertAlmostEqual(self.reward(3.0, -10.0), 0.08)

    def test_countdown_reset_and_finish(self):
        self.reward(2.0, 2.0)
        self.env.reset()
        self.assertEqual(self.reward(-100.0, 0.0), 0.0)
        self.race.world.phase = WorldState.Phase.RACE_PHASE
        self.assertAlmostEqual(self.reward(2.0, 2.0), 0.18)
        self.assertAlmostEqual(self.reward(2.0, 2.0, truncated=True), -0.02)
        self.assertAlmostEqual(self.reward(2.0, 2.0, finished=True), 11.48)
