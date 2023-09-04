"""Run RL Algorithm."""

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
    env: gym.vector.AsyncVectorEnv, agent: Agent, deterministic: bool = True
) -> None:
    """Test agent."""
    returns = []
    obs, _ = env.reset()
    n_envs = env.num_envs
    b_rewards = {idx: [] for idx in range(n_envs)}
    returns = list()
    for _ in range(250):
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
    mean = sum(returns) / len(returns)
    min_return, max_return = min(returns), max(returns)
    info = {"perf/mean": mean, "perf/min": min_return, "perf/max": max_return}
    return info


def run_rl(
    env: gym.Env,
    eval_env: gym.Env,
    agent: Agent,
    logger: Logger,
    base_dir: Path,
    n_epochs: int = 1_000,
    epoch_length: int = 1_000,
    n_inital_exploration_steps: int = 10_000,
    batch_size: int = 256,
    replay_buffer_size: int = 1_000_000,
    n_grad_step: int = 1,
    seed: int = 777,
    print_mode: bool = True,
    n_steps: int = 1,
    period_eval: int = 1,
    **kwargs,
) -> None:
    """Run SAC Algorithm."""
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set Rollout
    rollout = Rollout(env, replay_buffer_size, n_steps=n_steps)

    # Miscellaneous
    start_logging = False
    best_performance = -float("inf")
    iterator = tqdm(range(n_epochs)) if print_mode else range(n_epochs)
    start_time = time()
    train_infos: list[dict] = list()
    train_flag = False
    test_info = test_agent(eval_env, agent)
    iteration = 0
    for epoch in iterator:
        for _ in range(epoch_length):
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
        if epoch % period_eval == 0 and train_flag:
            test_info = test_agent(eval_env, agent)
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
