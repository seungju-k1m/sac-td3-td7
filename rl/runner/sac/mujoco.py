import os
import json
import gymnasium as gym
from copy import deepcopy

import pandas as pd
from rl import SAVE_DIR
from rl.runner.runner import run_rl
from rl.core.logger import setup_logger
from rl.utils.miscellaneous import convert_dict_as_param
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


def run_mujoco(
    env_id: str, exp_name: str, print_mode: bool = False, n_rollouts: int = 1, **kwargs
):
    """Run mujoco."""
    params = convert_dict_as_param(deepcopy(locals()))
    if print_mode:
        print("-" * 5 + "[SAC]" + "-" * 5)
        print(" " + pd.Series(params).to_string().replace("\n", "\n "))
        print()
    exp_dir = SAVE_DIR / "mujoco" / env_id / exp_name
    os.makedirs(exp_dir, exist_ok=True)
    with open(exp_dir / "config.json", "w") as file_handler:
        json.dump(params, file_handler, indent=4)
    logger = setup_logger(str(exp_dir / "training.log"))

    def make_env():
        def _make_env():
            env = gym.make(env_id)
            return env

        return _make_env

    env = gym.vector.AsyncVectorEnv([make_env() for _ in range(n_rollouts)])
    env = RecordEpisodeStatistics(env, 5)
    eval_env = gym.make(env_id)
    run_rl(env, eval_env, exp_dir, logger, print_mode=print_mode, n_eval=10, **kwargs)
