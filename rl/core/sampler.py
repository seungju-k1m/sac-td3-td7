import random
from abc import ABC
from copy import deepcopy
from typing import TypeVar

import gymnasium as gym
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
    def __init__(self, env: gym.Env, replay_buffer_size: int = 1_000_000):
        self.env = env
        self.replay_buffer: list[TRANSITION] = list()
        self.count_replay_buffer: int = 0
        self.replay_buffer_size: int = replay_buffer_size
        self.sampler = RandomSampler(self.env.action_space)
        self.need_reset = True
        self.obs: OBSERVATION
        self._returns: list[float] = list()
        self._rewards: list[float] = list()

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
            obs = torch.Tensor(obs)
            action = torch.Tensor(action)
            reward = torch.Tensor(reward)
            next_obs = torch.Tensor(next_obs)
            done = torch.Tensor(done)
        return dict(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def sample(self) -> None:
        if self.need_reset:
            obs, _ = self.env.reset(seed=np.random.randint(1, 100000000))
            self.need_reset = False
        else:
            obs = deepcopy(self.obs)
        action = self.sampler.sample(obs)
        next_obs, reward, terminated, done, _ = self.env.step(action)
        self._rewards.append(reward)
        _done = done or terminated
        float_done = 1.0 - float(_done)
        self.replay_buffer.append(
            deepcopy(
                [
                    obs.astype(np.float32),
                    action.astype(np.float32),
                    reward,
                    next_obs.astype(np.float32),
                    float_done,
                ]
            )
        )
        self.count_replay_buffer += 1
        self.obs = next_obs
        if _done:
            self.need_reset = True
            self._returns.append(sum(self._rewards))
            self._rewards.clear()
        if self.count_replay_buffer > self.replay_buffer_size:
            self.replay_buffer.pop(0)
            self.count_replay_buffer -= 1

    def print(self) -> None:
        """."""
        print(sum(self._returns) / (len(self._returns) + 1e-6))
        self._returns.clear()
