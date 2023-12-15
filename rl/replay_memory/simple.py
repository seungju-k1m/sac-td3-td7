"""Replay Buffer."""
from typing import Any

import numpy as np
import torch

from rl.replay_memory.base import BaseReplayMemory
from rl.utils.annotation import BATCH
from rl.utils.miscellaneous import get_action_bias_scale, get_state_action_dims


class SimpleReplayMemory(BaseReplayMemory):
    """LAP."""

    def __init__(self, replay_buffer_size: int, env_id: str, **kwargs) -> None:
        """Initialize."""
        super().__init__(replay_buffer_size, **kwargs)
        self.ptr = 0
        self.size = 0
        self.ind: list[int]
        state_dim, action_dim = get_state_action_dims(env_id)
        self.action_bias, self.action_scale = get_action_bias_scale(env_id)
        self.state = np.zeros((replay_buffer_size, state_dim))
        self.action = np.zeros((replay_buffer_size, action_dim))
        self.next_state = np.zeros((replay_buffer_size, state_dim))
        self.reward = np.zeros((replay_buffer_size, 1))
        self.float_done = np.zeros([replay_buffer_size, 1])

    def append(self, transition: list[Any]) -> None:
        """Append transition."""
        assert len(transition) == 5
        obs, action, reward, next_obs, float_done = transition
        action = action / self.action_scale - self.action_bias
        self.state[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_obs
        self.float_done[self.ptr] = float_done
        self.ptr = (self.ptr + 1) % self.replay_buffer_size
        self.size = min(self.size + 1, self.replay_buffer_size)

    def sample(self, batch_size: int, use_torch: bool = True) -> BATCH:
        """Sample."""
        priority = torch.ones(self.size)
        current_priority_sum = torch.cumsum(priority, 0)
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
        return dict(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )

    def __len__(self) -> int:
        return self.ptr
