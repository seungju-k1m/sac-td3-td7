import json
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

from rl.agent.sac import SAC
from rl.core.sampler import Rollout
from rl.core.logger import setup_logger
from rl.utils.miscellaneous import convert_dict_as_param


def test_agent(
    env: gym.Env, agent: SAC, n_episode: int = 32, deterministic: bool = True
) -> None:
    """."""
    returns = []
    for _ in range(n_episode):
        obs, _ = env.reset(seed=np.random.randint(1, 10000000))
        rewards = []
        flag = True
        # for j in range(1000):
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


def run_sac(
    env_id: str,
    exp_dir: Path,
    n_epochs: int = 1000,
    epoch_length: int = 1000,
    n_inital_exploration_steps: int = 10_000,
    batch_size: int = 256,
    replay_buffer_size: int = 1_000_000,
    **agent_kwargs,
) -> None:
    """Run SAC Algorithm."""
    params = convert_dict_as_param(deepcopy(locals()))
    print("-" * 5 + "[SAC]" + "-" * 5)
    print(" " + pd.Series(params).to_string().replace("\n", "\n "))
    print()
    with open(exp_dir / "config.json", "w") as file_handler:
        json.dump(params, file_handler, indent=4)
    train_logger = setup_logger(str(exp_dir / "training.log"))
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    rollout = Rollout(env, replay_buffer_size)
    agent = SAC(env, **agent_kwargs)
    n_transition: int = 0
    init_logger = False
    best_performance = 0.0
    for epoch in tqdm(range(n_epochs)):
        infos: list[dict] = list()
        for _ in range(epoch_length):
            rollout.sample()
            n_transition += 1
            if n_transition == n_inital_exploration_steps:
                rollout.set_sampler(agent)
            if n_transition < n_inital_exploration_steps:
                continue
            batch = rollout.get_batch(batch_size)
            info = agent.train_ops(batch)
            infos.append(info)
        if len(infos) > 0:
            keies = list(infos[0].keys())
            logging_info = {
                key: sum(list(map(lambda x: x[key], infos))) / len(infos)
                for key in keies
            }
            test_info = test_agent(eval_env, agent, 8)
            if not init_logger:
                init_logger = True
                train_logger.info(
                    ", ".join(
                        ["epoch"]
                        + sorted(list(logging_info.keys()) + list(test_info.keys()))
                    )
                )
            logging_info.update(test_info)
            logging_info = {key: value for key, value in sorted(logging_info.items())}
            stats_string = ", ".join(
                [f"{value:.4f}" for value in logging_info.values()]
            )
            train_logger.info(f"{epoch}, {stats_string}")
            if test_info["perf/mean"] > best_performance:
                best_performance = test_info["perf/mean"]
                agent.save(exp_dir / "best.pkl")
