"""Sampler for off-policy reinforcement learning."""

from abc import ABC
from typing import TypeVar

import gymnasium as gym

from rl.utils.annotation import ACTION


SAMPLER = TypeVar("SAMPLER", bound="Sampler")


class Sampler(ABC):
    """Abstract Sampler."""

    def sample(self, **kwargs) -> ACTION:
        """Sample must be implemented."""
        raise NotImplementedError("!!")


class RandomSampler(Sampler):
    """Random Sampler.

    It is used when initialization.
    """

    def __init__(self, action_space: gym.Space) -> None:
        """Initialize."""
        self.action_space = action_space
        self.action_space.seed(777)

    def sample(self, *args, **kwargs) -> ACTION:
        """Randomly sample using gym.space."""
        d_actions = self.action_space.sample()
        return d_actions
