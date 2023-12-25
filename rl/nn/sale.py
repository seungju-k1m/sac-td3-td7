"""Functions for making neural networks."""

from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F

from . import ACTOR, CRITIC, ENCODER


def AvgL1Norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply AvgL1Norm."""
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class SALEEncoder(ENCODER):
    """Encoder for learning a pair of state and action embedding."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int = 256,
        hdim: int = 256,
        activ: Callable = F.elu,
    ):
        """Initialize."""
        super().__init__()

        self.activ = activ
        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Forward obs embedding."""
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def encode_state_action(
        self, zs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward state-action embedding."""
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class SALEActor(ACTOR):
    """TD7 Actor."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int = 256,
        hdim: int = 256,
        activ: Callable = F.relu,
    ):
        """Initialize."""
        super().__init__()
        self.activ = activ
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def inference_mean(self, obs: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        a = AvgL1Norm(self.l0(obs))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class SALECritic(CRITIC):
    """TD7 Critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        zs_dim: int = 256,
        hdim: int = 256,
        activ: Callable = F.elu,
    ):
        """Initialize."""
        super().__init__()
        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

    def estimate_q_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        zsa: torch.Tensor,
        zs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward."""
        sa = torch.cat([obs, action], 1)
        embeddings = torch.cat([zsa, zs], 1)
        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)
        return q1
