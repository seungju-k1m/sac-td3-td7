"""Replay Buffer."""
from typing import Any

import numpy as np
import gymnasium as gym
import torch

from rl.replay_buffer.base import BaseReplayBuffer
from rl.utils.annotation import BATCH


class LAPReplayBuffer(BaseReplayBuffer):
    """LAP."""

    def __init__(
        self,
        replay_buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        **kwargs
    ) -> None:
        """Initialize."""
        super().__init__(replay_buffer_size, **kwargs)
        self.ptr = 0
        self.size = 0
        self.ind: list[int]
        self.state = np.zeros((replay_buffer_size, observation_space.shape[-1]))
        self.action = np.zeros((replay_buffer_size, action_space.shape[-1]))
        self.next_state = np.zeros((replay_buffer_size, observation_space.shape[-1]))
        self.reward = np.zeros((replay_buffer_size, 1))
        self.float_done = np.zeros([replay_buffer_size, 1])
        self.priority = torch.zeros(replay_buffer_size)
        self.max_priority = 1

    def append(self, transition: list[Any]) -> None:
        """Append transition."""
        assert len(transition) == 5
        obs, action, reward, next_obs, float_done = transition
        self.state[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_obs
        self.float_done[self.ptr] = float_done
        self.priority[self.ptr] = self.max_priority
        self.ptr = (self.ptr + 1) % self.replay_buffer_size
        self.size = min(self.size + 1, self.replay_buffer_size)

    def sample(self, batch_size: int, use_torch: bool = True) -> BATCH:
        """Sample."""
        current_priority_sum = torch.cumsum(self.priority[: self.size], 0)
        value = (
            torch.rand(
                batch_size,
            )
            * current_priority_sum[-1]
        )
        ind = torch.searchsorted(current_priority_sum, value).cpu().data.numpy()
        if use_torch:
            state = torch.Tensor(self.state[ind])
            action = torch.Tensor(self.action[ind])
            reward = torch.Tensor(self.reward[ind])
            next_state = torch.Tensor(self.next_state[ind])
            done = torch.Tensor(self.float_done[ind])
        self.ind = ind
        return dict(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def update_priority(self, priority: torch.Tensor) -> None:
        """Update priority."""
        self.priority[self.ind] = priority
        self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        """Reset max proirity."""
        self.max_priority = float(self.priority[: self.size].max())

    def __len__(self) -> int:
        return self.ptr
