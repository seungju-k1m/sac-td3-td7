"""Command Line Interface."""
import click
from rl.agent import run_sac
from rl.agent.td3 import run_td3

from rl.utils.cli_utils import configure


@click.command(name="sac")
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
@click.argument("env-id", type=click.STRING)
@click.argument("rl-run-name", type=click.STRING)
@click.option(
    "--tmp",
    type=float,
    default=0.2,
    help="Temperature for balancing exploration and exploitation.",
    show_default=True,
)
@click.option(
    "--auto-tmp",
    type=bool,
    default=False,
    is_flag=True,
)
@click.option("--action-fn", type=click.STRING, default="ReLU", show_default=True)
@click.option("--reward-scale", type=click.FLOAT, default=1.0, show_default=True)
@click.option("--discount-factor", type=click.FLOAT, default=0.99, show_default=True)
# For Traiuning
@click.option(
    "--n-iteration",
    type=click.INT,
    default=10_000_000,
    show_default=True,
    help="# of iteration.",
)
@click.option(
    "--batch-size", type=click.INT, default=256, show_default=True, help="Batch size."
)
@click.option(
    "--use-checkpoint",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use Checkpoint",
    is_flag=True,
)
def cli_run_sac(**kwargs):
    """Run SAC Algorithm."""
    run_sac(**kwargs)


@click.command(name="td3")
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
@click.argument("env-id", type=click.STRING)
@click.argument("rl-run-name", type=click.STRING)
@click.option("--action-fn", type=click.STRING, default="ReLU", show_default=True)
@click.option("--discount-factor", type=click.FLOAT, default=0.99, show_default=True)
# For Traiuning
@click.option(
    "--n-iteration",
    type=click.INT,
    default=10_000_000,
    show_default=True,
    help="# of iteration.",
)
@click.option(
    "--batch-size", type=click.INT, default=256, show_default=True, help="Batch size."
)
@click.option(
    "--use-checkpoint",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use Checkpoint",
    is_flag=True,
)
@click.option(
    "--use-lap",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use LAP.",
    is_flag=True,
)
def cli_run_td3(**kwargs):
    """Run SAC Algorithm."""
    run_td3(**kwargs)
