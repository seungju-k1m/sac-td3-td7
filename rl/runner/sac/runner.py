import random
from time import time
from pathlib import Path
from logging import Logger

import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl.agent.sac import SAC
from rl.core.sampler import Rollout


def test_agent(
    env: gym.Env, agent: SAC, n_episode: int = 32, deterministic: bool = True
) -> None:
    """."""
    returns = []
    for _ in range(n_episode):
        obs, _ = env.reset(seed=np.random.randint(1, 100000000000))
        rewards = []
        flag = True
        while flag:
            action = agent.sample(obs, deterministic)
            next_obs, reward, terminated, done, _ = env.step(action)
            rewards.append(reward)
            obs = next_obs
            if terminated or done:
                returns.append(sum(rewards))
                flag = False
    mean = sum(returns) / len(returns)
    min_return, max_return = min(returns), max(returns)
    info = {"perf/mean": mean, "perf/min": min_return, "perf/max": max_return}
    return info


def run_rl(
    env: RecordEpisodeStatistics,
    eval_env: gym.Env,
    exp_dir: Path,
    logger: Logger,
    n_epochs: int = 1_000,
    epoch_length: int = 1_000,
    n_inital_exploration_steps: int = 10_000,
    batch_size: int = 256,
    replay_buffer_size: int = 1_000_000,
    seed: int = 777,
    auto_tmp: bool = False,
    tmp: float = 0.2,
    print_mode: bool = False,
    n_eval: int = 10,
    device: str = "cpu",
    **agent_kwargs,
) -> None:
    """Run SAC Algorithm."""
    tmp = "auto" if auto_tmp else tmp
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rollout = Rollout(env, replay_buffer_size, device)
    agent = SAC(
        env.action_space.shape[-1],
        env.observation_space.shape[-1],
        tmp=tmp,
        device=device,
        **agent_kwargs,
    )
    init_logger = False
    best_performance = 0.0
    iterator = tqdm(range(n_epochs)) if print_mode else range(n_epochs)
    start_time = time()
    infos: list[dict] = list()
    for epoch in iterator:
        for _ in range(epoch_length):
            rollout.sample()
            if len(rollout.replay_buffer) > n_inital_exploration_steps:
                rollout.set_sampler(agent)
            else:
                continue
            batch = rollout.get_batch(batch_size)
            info = agent.train_ops(batch)
            infos.append(info)
        if len(infos) >= epoch_length * 5:
            keies = list(infos[0].keys())
            logging_info = {
                key: sum(list(map(lambda x: x[key], infos))) / len(infos)
                for key in keies
            }
            rollout_info = {
                "rollout/returns": float(
                    sum(rollout.env.return_queue) / len(rollout.env.return_queue)
                ),
                "rollout/episode_length": float(
                    sum(rollout.env.length_queue) / len(rollout.env.length_queue)
                ),
            }
            test_info = test_agent(eval_env, agent, n_eval)
            if not init_logger:
                init_logger = True
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
            if test_info["perf/mean"] > best_performance:
                best_performance = test_info["perf/mean"]
                agent.save(exp_dir / "best.pkl")
            infos.clear()
