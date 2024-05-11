from typing import Any, Callable
from rl.nn.abc import ACTOR, CRITIC, WORLDMODEL
from rl.nn.sale import AvgL1Norm, SALEEncoder

import torch
from torch import nn
import torch.nn.functional as F


class WorldModel(SALEEncoder, WORLDMODEL):
    """World Model."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int = 256,
        hdim: int = 256,
        activ: Callable[..., Any] = F.elu,
    ):
        """Init."""
        super().__init__(state_dim, action_dim, zs_dim, hdim, activ)
        self.reward_zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.reward_zsa2 = nn.Linear(hdim, hdim)
        self.reward_zsa3 = nn.Linear(hdim, 1)

    def predict_reward(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward."""
        zsa = self.activ(self.reward_zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.reward_zsa2(zsa))
        reward = self.reward_zsa3(zsa)
        return reward


class LatentActor(ACTOR):
    """Actor with latent variable."""

    def __init__(
        self,
        action_dim: int,
        zs_dim: int = 256,
        hdim: int = 256,
        activ: Callable = F.relu,
    ):
        """Initialize."""
        super().__init__()
        self.activ = activ
        self.l0 = nn.Linear(zs_dim, hdim)
        self.l1 = nn.Linear(zs_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def inference_mean(self, zs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        a = AvgL1Norm(self.l0(zs))
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class LatentCritic(CRITIC):
    """Critic with latent variable."""

    def __init__(
        self,
        action_dim: int,
        zs_dim: int = 256,
        hdim: int = 256,
        activ: Callable = F.elu,
    ):
        """Initialize."""
        super().__init__()
        self.activ = activ
        self.q01 = nn.Linear(zs_dim + action_dim, hdim)
        self.q1 = nn.Linear(zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

    def estimate_q_value(
        self,
        action: torch.Tensor,
        zs: torch.Tensor,
        zsa: torch.Tensor,
    ) -> torch.Tensor:
        """Forward."""
        sa = torch.cat([zs, action], 1)
        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, zsa], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)
        return q1
