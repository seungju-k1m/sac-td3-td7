"""Run RL Algorithm."""

from copy import deepcopy
from pathlib import Path
from logging import Logger

import torch
import gymnasium as gym
from tqdm import tqdm
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from rl.agent.abc import Agent
from rl.replay_buffer.base import REPLAYBUFFER
from rl.rollout import Rollout
from rl.runner.run import logging, run_train_ops, test_agent


def run_rl_w_ckpt(
    env: gym.Env,
    agent: Agent,
    replay_buffer: REPLAYBUFFER,
    logger: Logger,
    n_inital_exploration_steps: int = 25_000,
    n_iteration: int = 10_000_000,
    batch_size: int = 256,
    max_episodes_per_single_ckpt: int = 20,
    reset_weight: float = 0.9,
    eval_period: int = 10_000,
    show_progressbar: bool = True,
    record_video: bool = True,
    use_gpu: bool = False,
    seed: int = 42,
    **kwargs,
) -> None:
    """Run SAC Algorithm."""

    # Extract base_dir from logger.
    base_dir = Path(logger.name).parent
    # Set seed

    # Set Rollout
    render_mode = "rgb_array" if record_video else None
    eval_env = gym.make(env.spec, render_mode=render_mode)
    eval_env.reset(seed=seed + 100)

    env = RecordEpisodeStatistics(env, 1)
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

    # Miscellaneous
    train_flag = False
    iteration = 0

    # Checkpoint
    best_min_return = -1e8
    update_steps_before_ckpt = int(75e4)
    ckpt_agent: Agent = deepcopy(agent)
    if use_gpu:
        agent.to(torch.device("cuda"))
        ckpt_agent.to(torch.device("cuda"))

    rollout = Rollout(env, replay_buffer)

    # Progress bar
    if show_progressbar:
        progress_bar = tqdm(
            range(0, n_iteration),
        )
        progress_bar.set_description("Iteration")

    # Run RL
    start_logging = True
    best_return = -1e8
    test_info = test_agent(eval_env, ckpt_agent, True)
    current_max_episode_per_one_ckpt = 1
    sum_episode_length = 0
    eval_iteration = 0
    while iteration < n_iteration:
        train_infos: list[dict] = list()
        current_agent_min_return = 1e8

        # Collect data with fixed agent.
        for idx in range(current_max_episode_per_one_ckpt):
            done = False
            while not done:
                done = rollout.sample()
                if train_flag is False:
                    if len(rollout.replay_buffer) >= n_inital_exploration_steps:
                        rollout.set_sampler(agent)
                        train_flag = True
                    else:
                        continue

                # Evaluate ckpt agent.
                if train_flag and iteration > eval_iteration:
                    eval_iteration = iteration + eval_period
                    test_info = test_agent(eval_env, ckpt_agent, deterministic=True)
                    if test_info["perf/mean"] > best_return:
                        best_return = test_info["perf/mean"]
                        ckpt_agent.save(base_dir / "best.pkl")
            episode_return: float = rollout.env.return_queue[-1][0]
            episode_length: float = rollout.env.length_queue[-1][0]
            sum_episode_length += episode_length
            current_agent_min_return = min(episode_return, current_agent_min_return)

            # If minimum performance of agnet is lower than best return,
            # collecting data with current agent stops.
            if current_agent_min_return < best_min_return:
                break

        # Update checkpoint agent.
        if (
            current_agent_min_return >= best_min_return
            and idx == current_max_episode_per_one_ckpt - 1
            and train_flag
        ):
            best_min_return = current_agent_min_return
            ckpt_agent.load_state_dict(agent)
            ckpt_agent.save(base_dir / "ckpt.pkl")

        # Train ops
        if train_flag:
            train_infos = run_train_ops(sum_episode_length, rollout, agent, batch_size)
            # best_min_return *= reset_weight
            rollout_info = {
                "rollout/return": episode_return,
                "rollout/episode_length": episode_length,
            }
            iteration += sum_episode_length
            logging(
                iteration, logger, train_infos, test_info, rollout_info, start_logging
            )
            if show_progressbar:
                progress_bar.update(sum_episode_length)
                info = {
                    "best_min_return": best_min_return,
                    "current_min_return": current_agent_min_return,
                }
                info.update(test_info)
                progress_bar.set_postfix(info)

            if iteration > update_steps_before_ckpt:
                current_max_episode_per_one_ckpt = max_episodes_per_single_ckpt
                best_min_return *= reset_weight
                reset_weight = 1.0

            if start_logging:
                start_logging = False
            sum_episode_length = 0
