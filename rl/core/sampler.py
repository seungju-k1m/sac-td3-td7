import random
from abc import ABC
from copy import deepcopy
from typing import TypeVar

import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np
import torch

from rl.utils.annotation import ACTION, BATCH, DONE, OBSERVATION, REWARD, TRANSITION

SAMPLER = TypeVar("SAMPLER", bound="Sampler")


class Sampler(ABC):
    def sample(self, **kwargs) -> ACTION:
        raise NotImplementedError("!!")


class RandomSampler(Sampler):
    def __init__(self, action_space: gym.Space) -> None:
        self.action_space = action_space
        self.action_space.seed(777)

    def sample(self, *args, **kwargs) -> ACTION:
        return self.action_space.sample()


class Rollout:
    def __init__(
        self,
        env: RecordEpisodeStatistics,
        replay_buffer_size: int = 1_000_000,
        device: str = "mps",
    ):
        self.env = env
        self.replay_buffer: list[TRANSITION] = list()
        self.count_replay_buffer: int = 0
        self.replay_buffer_size: int = replay_buffer_size
        self.sampler = RandomSampler(self.env.action_space)
        self.need_reset = True
        self.obs: OBSERVATION = self.env.reset()[0]
        self.device = torch.device(device)

    def set_sampler(self, sampler: SAMPLER) -> None:
        """Set sampler."""
        self.sampler = sampler

    def get_batch(self, batch_size: int, use_torch: bool = True) -> BATCH:
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
            obs = torch.Tensor(obs).to(self.device)
            action = torch.Tensor(action).to(self.device)
            reward = torch.Tensor(reward).to(self.device)
            next_obs = torch.Tensor(next_obs).to(self.device)
            done = torch.Tensor(done).to(self.device)
        return dict(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def sample(self) -> None:
        obs = deepcopy(self.obs)
        n_dim = obs.shape[0]
        action = self.sampler.sample(obs).reshape(n_dim, -1)
        next_obs, reward, terminated, done, info = self.env.step(action)
        _done = done + terminated
        float_done = 1.0 - _done.astype(np.float32)
        obs = obs.astype(np.float32)
        next_obs = next_obs.astype(np.float32)
        action = action.astype(np.float32)
        self.replay_buffer += [
            [o, a, r, n, d]
            for o, a, r, n, d in zip(obs, action, reward, next_obs, float_done)
        ]
        self.count_replay_buffer += len(action)
        self.obs = next_obs
        if self.count_replay_buffer > self.replay_buffer_size:
            self.replay_buffer.pop(0)
            for _ in range(len(action)):
                self.replay_buffer.pop(0)
            self.count_replay_buffer -= len(action)
