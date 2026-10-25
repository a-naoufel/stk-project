"""Course/ML-Arena entry point. Package imports must be relative."""

from pathlib import Path

from .actors import DrivingActions, KartFeatures, load_policy_actor, make_heuristic_actor

env_name = "supertuxkart/full-v0"
player_name = "Naoufel-STK"


def get_wrappers():
    return [KartFeatures, DrivingActions]


def create_state():
    return None


def get_actor(module_dir: Path, observation_space, action_space):
    checkpoint = Path(module_dir) / "pystk_actor.pth"
    if checkpoint.is_file():
        return load_policy_actor(checkpoint, observation_space, action_space)
    return make_heuristic_actor()
