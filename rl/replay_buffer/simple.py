from collections import deque
import random
from typing import Any

import numpy as np
import torch
from rl.replay_buffer.base import BaseReplayBuffer
from rl.utils.annotation import ACTION, BATCH, DONE, STATE, REWARD


class SimpleReplayBuffer(BaseReplayBuffer):
    """Simple form."""

    def __init__(self, replay_buffer_size: int, **kwargs) -> None:
        super().__init__(replay_buffer_size, **kwargs)
        self.queue = deque(maxlen=self.replay_buffer_size)

    def sample(self, batch_size: int, use_torch: bool = True) -> BATCH:
        """Sample."""
        transitions = random.sample(self.queue, batch_size)
        state: STATE = np.stack(list(map(lambda x: x[0], transitions)), 0)
        action: ACTION = np.stack(list(map(lambda x: x[1], transitions)), 0)
        reward: REWARD = (
            np.array(list(map(lambda x: x[2], transitions)))
            .astype(np.float32)
            .reshape(-1, 1)
        )
        next_state: STATE = np.stack(list(map(lambda x: x[3], transitions)), 0)
        done: DONE = (
            np.array(list(map(lambda x: x[4], transitions)))
            .astype(np.float32)
            .reshape(-1, 1)
        )
        if use_torch:
            state = torch.Tensor(state)
            action = torch.Tensor(action)
            reward = torch.Tensor(reward)
            next_state = torch.Tensor(next_state)
            done = torch.Tensor(done)
        return dict(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )

    def append(self, transition: list[Any]) -> None:
        """Append."""
        self.queue.append(transition)

    def __len__(self) -> int:
        return len(self.queue)
