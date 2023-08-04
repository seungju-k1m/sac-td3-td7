import torch
from torch import nn


def calculate_grad_norm(neural_network: nn.Module) -> float:
    """Calculate grad norm."""
    total_norm = 0.0
    for param in neural_network.parameters():
        if param.grad is not None:
            total_norm += torch.norm(param.grad.data, p=2)
    return total_norm.item()


def clip_grad_norm(neural_network: nn.Module, max_norm: float = 1.0) -> None:
    if max_norm == float("inf"):
        return
    torch.nn.utils.clip_grad_norm_(neural_network.parameters(), max_norm)


def make_feedforward(
    input_dim: int,
    output_dim: int,
    hidden_sizes: list[int],
    act_fn: str | None = None,
    skip_last_act: bool = False,
    device: str | torch.device = "mps",
) -> nn.Module:
    """Make FeedForward neural network."""
    device = torch.device(device) if isinstance(device, str) else device
    act_fn = getattr(nn, act_fn)() if isinstance(act_fn, str) else act_fn
    input_hidden_sizes = [input_dim] + hidden_sizes
    output_hidden_sizes = hidden_sizes + [output_dim]
    _nns: list[nn.Module] = list()
    for input_hs, output_hs in zip(input_hidden_sizes, output_hidden_sizes):
        linear = nn.Linear(input_hs, output_hs).to(device)
        nn.init.xavier_normal_(linear.weight.data)
        nn.init.zeros_(linear.bias.data)
        _nns.append(linear)
        if act_fn is not None:
            _nns.append(act_fn)
    if skip_last_act:
        _nns.pop(-1)
    return nn.Sequential(*_nns)
