import unittest

from gymnasium.spaces import Box
import numpy as np

from scripts.stk_study import SymmetricReplay


class SymmetricReplayTests(unittest.TestCase):
    def test_asymmetric_splits_and_offline_only_sampling(self):
        space = Box(-1, 1, shape=(1,), dtype=np.float32)
        for percent in (25, 75, 100):
            with self.subTest(offline_percent=percent):
                buffer = SymmetricReplay(4, space, space, device="cpu", offline_percent=percent)
                if percent < 100:
                    value = np.array([[-0.5]], dtype=np.float32)
                    buffer.add(value, value, value, np.array([-1.0]), np.array([False]), [{}])
                buffer.offline = {field: np.array([[1.0]], dtype=np.float32) for field in (
                    "observations", "actions", "next_observations", "dones", "rewards")}
                batch = buffer.sample(128)
                expected_offline = 128 * percent // 100
                self.assertEqual(buffer.offline_draws, expected_offline)
                self.assertEqual(buffer.online_draws, 128 - expected_offline)
                self.assertEqual(int((batch.rewards == 1).sum()), expected_offline)
                self.assertEqual(int((batch.rewards == -1).sum()), 128 - expected_offline)

    def test_equal_sources_preserve_transitions_and_timeout_mask(self):
        space = Box(-1, 1, shape=(1,), dtype=np.float32)
        buffer = SymmetricReplay(4, space, space, device="cpu")
        buffer.add(
            np.array([[-0.5]]), np.array([[-0.25]]), np.array([[-1.0]]),
            np.array([-0.02]), np.array([True]), [{"TimeLimit.truncated": True}],
        )
        buffer.offline = {
            "observations": np.array([[0.5]], dtype=np.float32),
            "actions": np.array([[1.0]], dtype=np.float32),
            "next_observations": np.array([[0.75]], dtype=np.float32),
            "dones": np.array([[1.0]], dtype=np.float32),
            "rewards": np.array([[10.0]], dtype=np.float32),
        }
        batch = buffer.sample(128)
        self.assertEqual(buffer.online_draws, 64)
        self.assertEqual(buffer.offline_draws, 64)
        for field, online, offline in [
            (batch.observations, -0.5, 0.5), (batch.actions, -1.0, 1.0),
            (batch.next_observations, -0.25, 0.75), (batch.dones, 0, 1),
            (batch.rewards, -0.02, 10.0),
        ]:
            np.testing.assert_allclose(field[:64], online)
            np.testing.assert_allclose(field[64:], offline)
