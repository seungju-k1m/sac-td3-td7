"""Run RL Algorithm."""

from copy import deepcopy
import random
from pathlib import Path
from logging import Logger

import torch
import numpy as np
import gymnasium as gym

from rl.agent.base import Agent
from rl.rollout import Rollout
from rl.runner.run import logging, run_train_ops, test_agent
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


def run_rl_w_checkpoint(
    env: gym.Env,
    agent: Agent,
    logger: Logger,
    base_dir: Path,
    n_inital_exploration_steps: int = 25_000,
    n_iteration: int = 10_000_000,
    batch_size: int = 256,
    replay_buffer_size: int = 1_000_000,
    seed: int = 777,
    max_episode_per_one_chpt: int = 20,
    reset_weight: float = 0.9,
    eval_period: int = 20_000,
    **kwargs,
) -> None:
    """Run SAC Algorithm."""
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set Rollout
    eval_env = gym.make(env.spec)

    env = RecordEpisodeStatistics(env, 1)
    eval_env = RecordEpisodeStatistics(eval_env, 16)
    rollout = Rollout(env, replay_buffer_size)

    # Miscellaneous
    train_flag = False
    iteration = 0

    # Checkpoint
    best_min_return = -1e8
    checkpoint_agent = deepcopy(agent)

    # Run RL
    start_logging = True
    update_checkpoint_agent = True
    best_return = -1e8
    test_info = test_agent(eval_env, checkpoint_agent, True)
    while iteration < n_iteration:
        train_infos: list[dict] = list()
        min_return = 1e8
        sum_episode_length = 0

        for idx in range(max_episode_per_one_chpt):
            # One Episode
            done = False
            while not done:
                iteration += 1
                done = rollout.sample()
                if train_flag is False:
                    if len(rollout.replay_buffer) >= n_inital_exploration_steps:
                        rollout.set_sampler(agent)
                        train_flag = True
                    else:
                        continue
                if iteration % eval_period == 0:
                    if update_checkpoint_agent:
                        test_info = test_agent(eval_env, checkpoint_agent, True)
                        if test_info["perf/mean"] > best_return:
                            best_return = test_info["perf/mean"]
                            checkpoint_agent.save(base_dir / "best.pkl")
                        update_checkpoint_agent = False
            episode_return: float = rollout.env.return_queue[-1][0]
            episode_length: float = rollout.env.length_queue[-1][0]
            sum_episode_length += episode_length
            min_return = min(episode_return, min_return)
            if min_return < best_min_return:
                break
        if idx == max_episode_per_one_chpt - 1:
            best_min_return = min_return
            checkpoint_agent = deepcopy(agent)
            checkpoint_agent.save(base_dir / "ckpt.pkl")
            if train_flag:
                update_checkpoint_agent = True
        if train_flag:
            train_infos = run_train_ops(sum_episode_length, rollout, agent, batch_size)
            # best_min_return *= reset_weight
            rollout_info = {
                "rollout/return": episode_return,
                "rollout/episode_length": episode_length,
            }
            logging(
                iteration, logger, train_infos, test_info, rollout_info, start_logging
            )
            if start_logging:
                start_logging = False
            # agent.save(base_dir / "model.pkl")
            sum_episode_length = 0
