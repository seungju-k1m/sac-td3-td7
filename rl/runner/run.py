"""Run RL Algorithm."""

import pdb
import random
from pathlib import Path
from logging import Logger

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


from rl.agent.base import Agent
from rl.rollout import Rollout


@torch.no_grad()
def test_agent(
    env: RecordEpisodeStatistics,
    agent: Agent,
    deterministic: bool = True,
    n_episodes: int = 16,
) -> None:
    """Test agent."""
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.sample(obs, deterministic)
            next_obs, _, truncated, terminated, _ = env.step(action)
            obs = next_obs
            done = truncated or terminated
    mean = sum(env.return_queue) / len(env.return_queue)
    min_return, max_return = min(env.return_queue), max(env.return_queue)
    info = {"perf/mean": mean[0], "perf/min": min_return[0], "perf/max": max_return[0]}
    return info

def _get_mean(eles: list[float | None]) -> float:
    """Get mean."""
    eles = [ele for ele in eles if ele is not None]
    mean = sum(eles) / len(eles) if len(eles) != 0 else -1e6
    return mean

def logging(iteration: int, logger: Logger, train_infos: list[dict], test_info: dict, rollout_info: dict, start_logging: bool, **kwargs) -> None:
    """Logging."""
    train_keies = list(train_infos[0].keys())
    logging_info = {
        key: _get_mean(list(map(lambda x: x[key], train_infos)))
        for key in train_keies
    }
    if start_logging:
        logger.info(
            ",".join(
                ["iteration"]
                + sorted(
                    list(logging_info.keys())
                    + list(rollout_info.keys())
                    + list(test_info.keys())
                )
            )
        )
    logging_info.update(rollout_info)
    logging_info.update(test_info)
    logging_info = {key: value for key, value in sorted(logging_info.items())}
    stats_string = ",".join(
        [f"{value:.4f}" for value in logging_info.values()]
    )
    logger.info(f"{iteration},{stats_string}")

def run_train_ops(n_ops: int, rollout: Rollout, agent: Agent, batch_size: int) -> list[dict]:
    """Run Train Ops."""
    train_infos = list()
    for _ in range(n_ops):
        batch = rollout.get_batch(batch_size)
        train_info = agent.train_ops(batch)
        train_infos.append(train_info)
    return train_infos


def run_rl(
    env: gym.Env,
    agent: Agent,
    logger: Logger,
    base_dir: Path,
    n_inital_exploration_steps: int = 25_000,
    n_iteration: int = 10_000_000,
    batch_size: int = 256,
    replay_buffer_size: int = 1_000_000,
    n_grad_step: int = 1,
    seed: int = 777,
    eval_period: int = 5000,
    **kwargs,
) -> None:
    """Run SAC Algorithm."""
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set Rollout
    eval_env = gym.make(env.spec)

    env = RecordEpisodeStatistics(env, 2)
    eval_env = RecordEpisodeStatistics(eval_env, 16)
    rollout = Rollout(env, replay_buffer_size)

    # Miscellaneous
    train_flag = False
    iteration = 0
    start_logging = True
    best_return = -1e8
    # Run RL
    test_info = test_agent(eval_env, agent, True)
    while iteration < n_iteration:
        done = False
        train_infos: list[dict] = list()
        # One Episode.
        while not done:
            done = rollout.sample()
            if train_flag is False:
                if len(rollout.replay_buffer) >= n_inital_exploration_steps:
                    rollout.set_sampler(agent)
                    train_flag = True
                else:
                    continue
            train_infos += run_train_ops(n_grad_step, rollout, agent, batch_size)
            iteration += n_grad_step
            if iteration % eval_period == 0:
                test_info = test_agent(eval_env, agent, True)
                if test_info["perf/mean"] > best_return:
                    best_return = test_info["perf/mean"]
                    agent.save(base_dir / "best.pkl")
        episode_return: float = rollout.env.return_queue[-1][0]
        episode_length: float = rollout.env.length_queue[-1][0]
        if len(train_infos) > 0:
            rollout_info = {
                "rollout/return": episode_return,
                "rollout/episode_length": episode_length
            }
            logging(iteration, logger, train_infos, test_info, rollout_info, start_logging)
            if start_logging:
                start_logging = False
            agent.save(base_dir / "model.pkl")
            train_infos.clear()

