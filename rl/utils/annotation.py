"""Annotation."""

import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch


MATH_E = math.e
EPS = 1e-6

STATE = np.ndarray | torch.TensorType | dict[str, torch.TensorType | np.ndarray]
ACTION = np.ndarray | torch.TensorType
REWARD = np.ndarray | float
DONE = np.ndarray | torch.Tensor
PATH = str | Path

TRANSITION = tuple[np.ndarray, float, np.ndarray, np.ndarray, float]


class BATCH(TypedDict):
    """BATCH consists of obs, action, reward, next_obs, and done."""

    state: STATE
    next_state: STATE
    action: ACTION
    reward: REWARD
    done: DONE
