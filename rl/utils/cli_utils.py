"""Utility code related to CLI."""

from typing import Callable
import click
import yaml


def configure(
    ctx: click.core.Context, param: click.core.Option, path: str | None
) -> None:
    """Update hyper-parameter corresponding to params."""
    if path is None:
        return
    with open(path) as file_handler:
        options = yaml.load(file_handler, yaml.FullLoader)
    ctx.default_map = options


def common_params_for_rl_alg(func) -> Callable:
    """Common Params for Off-policy RL Algorithms."""

    @click.option(
        "-c",
        "--config",
        type=click.Path(),
        callback=configure,
        is_eager=True,
        expose_value=False,
        help="Read option defaults from the specified INI file",
        show_default=True,
    )
    @click.option(
        "--run-name",
        default="",
        type=str,
        help="Run experiment would be saved in save/<ALG_NAME>/<run_name>-<timestamp>",
        show_default=True,
    )
    @click.option(
        "--env-id",
        default="Hopper-v4",
        type=str,
        help="Env Id.",
        show_default=True,
    )
    @click.option(
        "--discount-factor",
        type=click.FLOAT,
        default=0.99,
        show_default=True,
        help="Discount Factor.",
    )
    @click.option(
        "--n-iteration",
        type=click.INT,
        default=5_000_000,
        show_default=True,
        help="# of train iteration.",
    )
    @click.option(
        "--replay-buffer-size",
        type=click.INT,
        default=1_000_000,
        show_default=True,
        help="Max size of replay memory.",
    )
    @click.option(
        "--n-initial-exploration-steps",
        type=click.INT,
        default=25_000,
        show_default=True,
        help="# of transitions which random policy collects.",
    )
    @click.option(
        "--eval-period",
        type=click.INT,
        default=10_000,
        show_default=True,
        help="Every eval period, evaluate agent performance.",
    )
    @click.option(
        "--batch-size",
        type=click.INT,
        default=256,
        show_default=True,
        help="Batch size.",
    )
    @click.option(
        "--record-video",
        type=click.BOOL,
        default=False,
        show_default=True,
        help="Record Video.",
        is_flag=True,
    )
    @click.option("--seed", type=click.INT, default=42, show_default=True, help="Seed.")
    def wrapper(*args, **kwargs) -> Callable:
        return func(*args, **kwargs)

    return wrapper
