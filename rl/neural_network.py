"""Functions for making neural networks."""

from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F

from rl.utils.annotation import ACTION, STATE


def calculate_grad_norm(neural_network: nn.Module) -> float:
    """Calculate grad norm."""
    total_norm = 0.0
    for param in neural_network.parameters():
        if param.grad is not None:
            total_norm += torch.norm(param.grad.data, p=2)
    return total_norm.item()


def clip_grad_norm(neural_network: nn.Module, max_norm: float = 1.0) -> None:
    """Clip norm of gradient."""
    if max_norm == float("inf"):
        return
    torch.nn.utils.clip_grad_norm_(neural_network.parameters(), max_norm)


def AvgL1Norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply AvgL1Norm."""
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int] = [256, 256],
    action_fn: str | None = "ReLU",
) -> nn.Module:
    """Make FeedForward neural network."""
    action_fn = getattr(nn, action_fn)() if isinstance(action_fn, str) else action_fn
    input_hidden_sizes = [input_dim] + hidden_sizes
    output_hidden_sizes = hidden_sizes + [output_dim]
    _nns: list[nn.Module] = list()
    for input_hs, output_hs in zip(input_hidden_sizes, output_hidden_sizes):
        linear = nn.Linear(input_hs, output_hs)
        nn.init.xavier_normal_(linear.weight.data)
        nn.init.zeros_(linear.bias.data)
        _nns.append(linear)
        if action_fn is not None:
            _nns.append(action_fn)
    _nns.pop(-1)
    return nn.Sequential(*_nns)


class MLPPolicy(nn.Module):
    """Base Policy uses Multi-Layer Perceptrons."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | int = 256,
        action_fn: str | nn.Module = "ReLU",
    ) -> None:
        """Initialize."""
        super().__init__()
        hidden_sizes = (
            [hidden_sizes] * 2 if isinstance(hidden_sizes, int) else hidden_sizes
        )
        action_fn = (
            getattr(nn, action_fn)() if isinstance(action_fn, str) else action_fn
        )

        self.mlp = make_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            action_fn=action_fn,
        )

    def forward(self, state: STATE) -> torch.Tensor:
        """Forward"""
        return self.mlp(state)


class MLPQ(nn.Module):
    """Base Q function uses Multi-Layer Perceptrons."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | int = 256,
        action_fn: str | nn.Module = "ReLU",
    ) -> None:
        """Initialize."""
        super().__init__()
        hidden_sizes = (
            [hidden_sizes] * 2 if isinstance(hidden_sizes, int) else hidden_sizes
        )
        action_fn = (
            getattr(nn, action_fn)() if isinstance(action_fn, str) else action_fn
        )

        self.mlp = make_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            action_fn=action_fn,
        )

    def forward(self, state: STATE, action: ACTION) -> torch.Tensor:
        """Forward"""
        state_action = torch.cat([state, action], -1)
        return self.mlp(state_action)


class Encoder(nn.Module):
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


class SALEActor(nn.Module):
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

    def forward(self, obs: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        a = AvgL1Norm(self.l0(obs))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class SALECritic(nn.Module):
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

    def forward(
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
