"""Rollout."""
import random
from copy import deepcopy

import torch
import numpy as np
import gymnasium as gym

from rl.sampler import SAMPLER, RandomSampler
from rl.utils.annotation import TRANSITION, OBSERVATION, BATCH, ACTION, REWARD, DONE


class Rollout:
    """Rollout Worker."""

    def __init__(
        self,
        env: gym.vector.AsyncVectorEnv,
        replay_buffer_size: int = 1_000_000,
        n_steps: int = 1,
    ):
        """Initialize."""
        self.env = env
        self.n_steps = n_steps
        self.replay_buffer: list[TRANSITION] = list()
        self.count_replay_buffer: int = 0
        self.replay_buffer_size: int = replay_buffer_size
        self.sampler = RandomSampler(self.env.action_space)
        self.obs: OBSERVATION = self.env.reset()[0]
        self._returns: list[float] = list()
        self._rewards: list[float] = list()

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

    def sample(self) -> None:
        """Sample action using policy."""
        action = self.sampler.sample(self.obs)
        # TODO: If length is not completely divided by n_steps
        reward = np.zeros(self.env.num_envs)
        for _ in range(self.n_steps):
            next_obs, _reward, truncated, terminated, infos = self.env.step(action)
            reward += _reward
        np_action = np.array(list(action.values())).T
        for ob, aa, rwd, no, ter, trunc in zip(
            self.obs, np_action, reward, next_obs, terminated, truncated
        ):
            if trunc:
                continue
            float_done = 1.0 - float(ter)
            self.replay_buffer.append(
                deepcopy(
                    [
                        ob.astype(np.float32),
                        aa.astype(np.float32),
                        rwd,
                        no.astype(np.float32),
                        float_done,
                    ]
                )
            )
        self.obs = next_obs
        if self.count_replay_buffer > self.replay_buffer_size:
            for _ in range(len(action)):
                self.replay_buffer.pop(0)
