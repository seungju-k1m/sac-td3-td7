# MAKINAROCKS CONFIDENTIAL
# ________________________
#
# [2017] - [2023] MakinaRocks Co., Ltd.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of MakinaRocks Co., Ltd. and its suppliers, if any.
# The intellectual and technical concepts contained herein are
# proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
# covered by U.S. and Foreign Patents, patents in process, and
# are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained
# from MakinaRocks Co., Ltd.
"""Annotation."""

import math
from pathlib import Path
from typing import TypedDict

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
    """BATCH consists of obs, action, reward, next_obs, and done."""

    obs: OBSERVATION
    next_obs: OBSERVATION
    action: ACTION
    reward: REWARD
    done: DONE
