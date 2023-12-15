"""Miscellaneous Code."""

import os
import logging
import random
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch


def convert_dict_as_param(d_str_value: dict[str, Any]) -> dict[str, Any]:
    """Convert dict as param format."""
    param: dict[str, Any] = dict()
    for key, value in d_str_value.items():
        if not isinstance(value, dict):
            param[key] = value
        else:
            param.update(value)
    return param


def setup_logger(path: str, level=logging.DEBUG) -> logging.Logger:
    """Set up logger."""
    logger = logging.getLogger(path)
    logger.setLevel(level)

    if os.path.isfile(path):
        os.remove(path)
    file_handler = logging.FileHandler(path)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


def fix_seed(seed: int) -> None:
    """Fix seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def clamp(x: float, min_x: float, max_x: float) -> float:
    """Clamp x within range."""
    return max(min_x, min(x, max_x))


def get_state_action_dims(env_id: str) -> tuple[int, int]:
    """Return state and action dimension."""
    assert env_id in gym.registry
    env = gym.make(env_id)
    return env.observation_space.shape[0], env.action_space.shape[0]


def get_action_bias_scale(env_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return action bias and scale."""
    action_space = gym.make(env_id).action_space
    action_upper_bound: np.ndarray = action_space.high
    action_lower_bound: np.ndarray = action_space.low
    action_bias = (action_lower_bound + action_upper_bound) / 2.0
    action_scale = (action_upper_bound - action_lower_bound) / 2.0
    return action_bias, action_scale


class NoStdStreams:
    """Prevent function from prinint to console."""

    def __init__(self, stdout=None, stderr=None):
        """Initialize."""
        self.devnull = open(os.devnull, "w")
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        """Enter."""
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit."""
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
