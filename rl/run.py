"""Run RL Algorithm."""

from copy import deepcopy
import random
from time import time
from pathlib import Path
from logging import Logger

import torch
import numpy as np
import gymnasium as gym

from tqdm import tqdm

from rl.agent.base import Agent
from rl.rollout import Rollout


@torch.no_grad()
def test_agent(
    env: gym.vector.AsyncVectorEnv,
    agent: Agent,
    deterministic: bool = True,
    n_episodes: int = 16,
) -> None:
    """Test agent."""
    returns = []
    obs, _ = env.reset()
    n_envs = env.num_envs
    b_rewards = {idx: [] for idx in range(n_envs)}
    returns = list()
    flag = True
    while flag:
        action = agent.sample(obs, deterministic)
        next_obs, rewards, truncateds, terminateds, _ = env.step(action)
        obs = next_obs
        for ii, (reward, truncated, termianted) in enumerate(
            zip(rewards, truncateds, terminateds)
        ):
            b_rewards[ii].append(reward)
            if truncated or termianted:
                returns.append(sum(b_rewards[ii]))
                b_rewards[ii].clear()
        if len(returns) >= n_episodes:
            flag = False
    mean = sum(returns) / len(returns)
    min_return, max_return = min(returns), max(returns)
    info = {"perf/mean": mean, "perf/min": min_return, "perf/max": max_return}
    return info


def run_rl(
    env: gym.Env,
    agent: Agent,
    logger: Logger,
    base_dir: Path,
    n_epochs: int = 1_000,
    iteration_per_epoch: int = 1_000,
    n_inital_exploration_steps: int = 25_000,
    batch_size: int = 256,
    replay_buffer_size: int = 1_000_000,
    n_grad_step: int = 1,
    n_skip_steps: int = 1,
    eval_period: int = 1,
    n_episodes_eval: int = 8,
    seed: int = 777,
    **kwargs,
) -> None:
    """Run SAC Algorithm."""
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set Rollout
    rollout = Rollout(env, replay_buffer_size, n_skip_steps=n_skip_steps)
    eval_env = gym.make_vec(env.spec, 8)

    # Miscellaneous
    start_logging = False
    best_performance = -float("inf")
    iterator = tqdm(range(n_epochs))
    start_time = time()
    train_infos: list[dict] = list()
    train_flag = False
    test_info = test_agent(eval_env, agent, n_episodes=n_episodes_eval)
    iteration = 0
    for epoch in iterator:
        for inner_iteration in range(iteration_per_epoch):
            rollout.sample()
            if train_flag is False:
                if len(rollout.replay_buffer) >= n_inital_exploration_steps:
                    rollout.set_sampler(agent)
                    train_flag = True
                else:
                    continue
            # Run Train ops
            for _ in range(n_grad_step):
                batch = rollout.get_batch(batch_size)
                train_info = agent.train_ops(batch)
                train_infos.append(train_info)
                iteration += 1
        if epoch % eval_period == 0 and train_flag:
            test_info = test_agent(
                eval_env, agent, n_episodes=n_episodes_eval
            )
            if test_info["perf/mean"] > best_performance:
                best_performance = test_info["perf/mean"]
                agent.save(base_dir / "best.pkl")
        # Logging.
        if len(train_infos) > 0:
            train_keies = list(train_infos[0].keys())
            logging_info = {
                key: sum(list(map(lambda x: x[key], train_infos))) / len(train_infos)
                for key in train_keies
            }
            rollout_info = {
                "rollout/returns": float(
                    sum(rollout.env.return_queue) / len(rollout.env.return_queue)
                ),
                "rollout/episode_length": float(
                    sum(rollout.env.length_queue) / len(rollout.env.length_queue)
                ),
            }
            if not start_logging:
                start_logging = True
                logger.info(
                    ", ".join(
                        ["epoch"]
                        + sorted(
                            list(logging_info.keys())
                            + list(test_info.keys())
                            + list(rollout_info.keys())
                        )
                        + ["elasped_time"]
                    )
                )
            logging_info.update(test_info)
            logging_info.update(rollout_info)
            logging_info = {key: value for key, value in sorted(logging_info.items())}
            elasped_time = time() - start_time
            stats_string = ", ".join(
                [f"{value:.4f}" for value in logging_info.values()]
            )
            stats_string = stats_string + f", {elasped_time:.1f}"
            logger.info(f"{epoch}, {stats_string}")
            train_infos.clear()
            agent.save(base_dir / "model.pkl")
