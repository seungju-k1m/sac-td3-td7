"""MLP Policy and Q Function."""

import torch
from torch import nn

from rl.utils.annotation import ACTION, STATE
from rl.nn import ACTOR, CRITIC


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int] = [256, 256],
    action_fn: str | None = "ReLU",
    init_weight: None | str = None,
    init_bias: None | str = None,
) -> nn.Module:
    """Make FeedForward neural network."""
    init_weight = (
        nn.init.xavier_normal_ if init_weight is None else getattr(nn.init, init_weight)
    )
    init_bias = nn.init.zeros_ if init_bias is None else getattr(nn.init, init_bias)
    action_fn = getattr(nn, action_fn)() if isinstance(action_fn, str) else action_fn
    input_hidden_sizes = [input_dim] + hidden_sizes
    output_hidden_sizes = hidden_sizes + [output_dim]
    _nns: list[nn.Module] = list()
    for input_hs, output_hs in zip(input_hidden_sizes, output_hidden_sizes):
        linear = nn.Linear(input_hs, output_hs)
        init_weight(linear.weight.data)
        init_bias(linear.bias.data)
        _nns.append(linear)
        if action_fn is not None:
            _nns.append(action_fn)
    _nns.pop(-1)
    return nn.Sequential(*_nns)


class MLPActor(ACTOR):
    """Actor uses Multi-Layer Perceptrons."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | int = 256,
        **mlp_kwargs,
    ) -> None:
        """Initialize."""
        super().__init__()
        hidden_sizes = (
            [hidden_sizes] * 2 if isinstance(hidden_sizes, int) else hidden_sizes
        )
        self.mlp = make_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **mlp_kwargs,
        )

    def inference_mean(self, state: STATE) -> torch.Tensor:
        """Inference mean"""
        return self.mlp(state)

    def inference_mean_logvar(self, state: STATE) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference mean and logvar."""
        output: torch.Tensor = self.mlp(state)
        mean, logvar = output.chunk(2, -1)
        return mean, logvar

    def inference_policy(self, state: STATE) -> torch.Tensor:
        """Inference policy in discrete action space."""
        return self.mlp(state)


class MLPCritic(CRITIC):
    """Critic uses Multi-Layer Perceptrons."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | int = 256,
        **mlp_kwargs,
    ) -> None:
        """Initialize."""
        super().__init__()
        hidden_sizes = (
            [hidden_sizes] * 2 if isinstance(hidden_sizes, int) else hidden_sizes
        )

        self.mlp = make_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            **mlp_kwargs,
        )

    def estimate_q_value(self, state: STATE, action: ACTION) -> torch.Tensor:
        """Inference Q"""
        state_action = torch.cat([state, action], -1)
        return self.mlp(state_action)

    def estimate_state_value(self, state: STATE) -> torch.Tensor:
        return self.mlp(state)
