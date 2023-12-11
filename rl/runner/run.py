"""Run RL Algorithm."""

from pathlib import Path
from logging import Logger

import torch
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
from tqdm import tqdm


from rl.agent.abc import Agent
from rl.replay_buffer.base import REPLAYBUFFER
from rl.rollout import Rollout
from rl.utils import NoStdStreams


@torch.no_grad()
def test_agent(
    env: RecordEpisodeStatistics,
    agent: Agent,
    deterministic: bool = True,
    n_episodes: int = 16,
) -> dict[str, float]:
    """Test agent."""
    with NoStdStreams():
        for idx in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.sample(obs, deterministic)
                next_obs, _, truncated, terminated, _ = env.step(action)
                obs = next_obs
                done = truncated or terminated
        mean = sum(env.return_queue) / len(env.return_queue)
        min_return, max_return = min(env.return_queue), max(env.return_queue)
        info = {
            "perf/mean": mean[0],
            "perf/min": min_return[0],
            "perf/max": max_return[0],
        }
        return info


def _calculate_mean_with_dirty_eles(eles: list[float | None]) -> float:
    """Get mean."""
    eles = [ele for ele in eles if ele is not None]
    mean = sum(eles) / len(eles) if len(eles) != 0 else -1e6
    return mean


def logging(
    iteration: int,
    logger: Logger,
    train_infos: list[dict],
    test_info: dict,
    rollout_info: dict,
    start_logging: bool,
    **kwargs,
) -> None:
    """Logging."""
    train_keies = list(train_infos[0].keys())
    logging_info = {
        key: _calculate_mean_with_dirty_eles(list(map(lambda x: x[key], train_infos)))
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
    stats_string = ",".join([f"{value:.4f}" for value in logging_info.values()])
    logger.info(f"{iteration},{stats_string}")


def run_train_ops(
    n_ops: int, rollout: Rollout, agent: Agent, batch_size: int
) -> list[dict]:
    """Run Train Ops."""
    train_infos = list()
    for _ in range(n_ops):
        batch = rollout.get_batch(batch_size)
        train_info = agent.train_ops(batch, replay_buffer=rollout.replay_buffer)
        train_infos.append(train_info)
    return train_infos


def run_rl(
    env: gym.Env,
    agent: Agent,
    replay_buffer: REPLAYBUFFER,
    logger: Logger,
    n_initial_exploration_steps: int = 25_000,
    n_iteration: int = 10_000_000,
    batch_size: int = 256,
    n_grad_step: int = 1,
    eval_period: int = 10_000,
    show_progressbar: bool = True,
    record_video: bool = True,
    seed: int = 42,
    **kwargs,
) -> None:
    """Run SAC Algorithm."""
    # Extract base_dir from logger.
    base_dir = Path(logger.name).parent

    # Set Rollout
    render_mode = "rgb_array" if record_video else None
    eval_env = gym.make(env.spec, render_mode=render_mode)
    eval_env.reset(seed=seed + 100)

    env = RecordEpisodeStatistics(env, 2)
    eval_env = RecordEpisodeStatistics(eval_env, 16)

    if record_video:
        video_dir = base_dir / "video"
        video_dir.mkdir(exist_ok=True, parents=True)

        # Only record last episode when evaluation.
        def epi_trigger(x) -> bool:
            if x % 16 == 0:
                return True
            else:
                False

        eval_env = RecordVideo(eval_env, str(video_dir), episode_trigger=epi_trigger)
    rollout = Rollout(env, replay_buffer)

    # Miscellaneous
    train_flag = False
    iteration = 0
    start_logging = True
    best_return = -1e8

    # Progress bar
    if show_progressbar:
        progress_bar = tqdm(
            range(0, n_iteration),
        )
        progress_bar.set_description("Iteration")
    # Run RL
    test_info = test_agent(eval_env, agent, True)
    while iteration < n_iteration:
        done = False
        train_infos: list[dict] = list()
        # One Episode.
        while not done:
            done = rollout.sample()
            if train_flag is False:
                if len(rollout.replay_buffer) >= n_initial_exploration_steps:
                    rollout.set_sampler(agent)
                    train_flag = True
                else:
                    continue
            train_infos += run_train_ops(n_grad_step, rollout, agent, batch_size)
            iteration += n_grad_step
            if show_progressbar:
                progress_bar.update(n_grad_step)
                progress_bar.set_postfix(test_info)
            if iteration % eval_period == 0:
                test_info = test_agent(eval_env, agent, deterministic=True)
                if test_info["perf/mean"] > best_return:
                    best_return = test_info["perf/mean"]
                    agent.save(base_dir / "best.pkl")
        episode_return: float = rollout.env.return_queue[-1][0]
        episode_length: float = rollout.env.length_queue[-1][0]
        if len(train_infos) > 0:
            rollout_info = {
                "rollout/return": episode_return,
                "rollout/episode_length": episode_length,
            }
            logging(
                iteration, logger, train_infos, test_info, rollout_info, start_logging
            )
            if start_logging:
                start_logging = False
            agent.save(base_dir / "model.pkl")
            train_infos.clear()
