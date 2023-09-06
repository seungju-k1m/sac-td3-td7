"""Rollout."""
import random
from copy import deepcopy

import torch
import numpy as np
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics


from rl.sampler import SAMPLER, RandomSampler
from rl.utils.annotation import TRANSITION, OBSERVATION, BATCH, ACTION, REWARD, DONE


class Rollout:
    """Rollout Worker."""

    def __init__(
        self,
        env: RecordEpisodeStatistics,
        replay_buffer_size: int = 1_000_000,
    ):
        """Initialize."""
        self.env = env
        self.replay_buffer: list[TRANSITION] = list()
        self.count_replay_buffer: int = 0
        self.replay_buffer_size: int = replay_buffer_size
        self.sampler = RandomSampler(self.env.action_space)
        self.obs: OBSERVATION = self.env.reset()[0]
        self.need_reset: bool = True
        self.n_episode: int = 0

    def set_sampler(self, sampler: SAMPLER) -> None:
        """Set sampler."""
        self.sampler = sampler

    def get_batch(self, batch_size: int, use_torch: bool = True) -> BATCH:
        """Return batch for train ops."""
        transitions = random.sample(self.replay_buffer, batch_size)
        obs: OBSERVATION = np.stack(list(map(lambda x: x[0], transitions)), 0)
        action: ACTION = np.stack(list(map(lambda x: x[1], transitions)), 0)
        reward: REWARD = (
            np.array(list(map(lambda x: x[2], transitions)))
            .astype(np.float32)
            .reshape(-1, 1)
        )
        next_obs: OBSERVATION = np.stack(list(map(lambda x: x[3], transitions)), 0)
        done: DONE = (
            np.array(list(map(lambda x: x[4], transitions)))
            .astype(np.float32)
            .reshape(-1, 1)
        )
        if use_torch:
            obs = torch.Tensor(obs)
            action = torch.Tensor(action)
            reward = torch.Tensor(reward)
            next_obs = torch.Tensor(next_obs)
            done = torch.Tensor(done)
        return dict(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def sample(self) -> bool:
        """Sample action using policy."""
        if self.need_reset:
            self.need_reset = False
            self.obs = self.env.reset()[0]
        action = self.sampler.sample(self.obs)
        next_obs, reward, truncated, terminated, info = self.env.step(action)
        done = truncated or terminated
        self.replay_buffer.append(
            deepcopy([self.obs, action, reward, next_obs, 1.0 - float(done)])
        )
        self.obs = next_obs
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)
        if done:
            self.obs = self.env.reset()[0]
        return done
