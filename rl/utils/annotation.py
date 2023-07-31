import math
from typing import TypedDict

from pathlib import Path
import numpy as np
import torch


MATH_E = math.e
EPS = 1e-6

OBSERVATION = np.ndarray | torch.TensorType | dict[str, torch.TensorType | np.ndarray]
ACTION = np.ndarray | torch.TensorType
REWARD = np.ndarray | float
DONE = np.ndarray | torch.Tensor
PATH = str | Path

TRANSITION = tuple[np.ndarray, float, np.ndarray, np.ndarray, float]


class BATCH(TypedDict):
    obs: OBSERVATION
    next_obs: OBSERVATION
    action: ACTION
    reward: REWARD
    done: DONE
