import os
import json
import metaworld
import gymnasium as gym
from copy import deepcopy

from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

import pandas as pd
from rl import SAVE_DIR
from rl.runner.runner import run_rl
from rl.core.logger import setup_logger
from rl.utils.miscellaneous import convert_dict_as_param


def run_metaworld(
    exp_name: str,
    benchmark: str = "ML1",
    task_id: str = "pick-place-v2",
    task_idx: int = 0,
    print_mode: bool = False,
    n_rollouts: int = 1,
    **kwargs
):
    """Run mujoco."""
    params = convert_dict_as_param(deepcopy(locals()))
    if print_mode:
        print("-" * 5 + "[SAC]" + "-" * 5)
        print(" " + pd.Series(params).to_string().replace("\n", "\n "))
        print()
    exp_dir = SAVE_DIR / "metaworld" / benchmark / task_id / exp_name
    os.makedirs(exp_dir, exist_ok=True)
    with open(exp_dir / "config.json", "w") as file_handler:
        json.dump(params, file_handler, indent=4)
    logger = setup_logger(str(exp_dir / "training.log"))
    benchmark = getattr(metaworld, benchmark)(task_id)
    task = benchmark.train_tasks[task_idx]
    eval_env = benchmark.train_classes[task_id]()
    eval_env.set_task(task)

    def make_env():
        def _make_env():
            env = benchmark.train_classes[task_id]()
            env.set_task(task)
            env = TimeLimit(env, 500)
            # env = RecordEpisodeStatistics(TimeLimit(env, 500), 5)
            return env

        return _make_env

    env = gym.vector.AsyncVectorEnv([make_env() for _ in range(n_rollouts)])
    env = RecordEpisodeStatistics(env, 5)
    eval_env = make_env()()
    run_rl(env, eval_env, exp_dir, logger, print_mode=print_mode, n_eval=1, **kwargs)
