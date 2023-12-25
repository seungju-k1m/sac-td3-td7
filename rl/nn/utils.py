"""Functions for making neural networks."""
from functools import wraps
from typing import Any, Callable, ParamSpec

import torch
from torch import nn

from .abc import CRITIC, ACTOR, ENCODER

P = ParamSpec("P")


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


def annotate_make_nn(
    func: Callable[P, tuple[ACTOR, CRITIC, CRITIC]]
) -> Callable[P, tuple[ACTOR, CRITIC, CRITIC]]:
    """Annotation for function generating neural network of SAC."""

    @wraps(func)
    def wrapper(
        *args: P.args, **kwargs: P.kwargs
    ) -> Callable[[Any], tuple[ACTOR, CRITIC, CRITIC]]:
        """Wrapper."""
        return func(*args, **kwargs)

    return wrapper


def annotate_make_nn_td7(
    func: Callable[P, tuple[ACTOR, CRITIC, CRITIC, ENCODER]]
) -> Callable[P, tuple[ACTOR, CRITIC, CRITIC, ENCODER]]:
    """Annotation for function generating neural network of SAC."""

    @wraps(func)
    def wrapper(
        *args: P.args, **kwargs: P.kwargs
    ) -> tuple[ACTOR, CRITIC, CRITIC, ENCODER]:
        """Wrapper."""
        return func(*args, **kwargs)

    return wrapper
