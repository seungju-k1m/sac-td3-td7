"""Command Line Interface."""
import click
from rl.agent import run_sac

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
@click.option(
    "--policy-reg-coeff",
    type=click.FLOAT,
    default=0.0,
    help="Coefficient for regulating policy.",
    show_default=True,
)
@click.option(
    "--policy-sto-reg-coeff",
    type=click.FLOAT,
    default=0.0,
    help="Coefficient for regulating policy.",
    show_default=True,
)
@click.option(
    "--n-epochs",
    type=click.INT,
    default=1_000,
    show_default=True,
)
@click.option("--action-fn", type=click.STRING, default="ReLU6", show_default=True)
@click.option("--reward-scale", type=click.FLOAT, default=1.0, show_default=True)
@click.option("--discount-factor", type=click.FLOAT, default=0.99, show_default=True)
@click.option("--penalty-action-mag", type=click.FLOAT, default=0.0, show_default=True)
@click.option(
    "--n-skip-steps", type=click.INT, default=1, help="# of steps.", show_default=True
)
@click.option(
    "--n-rollouts",
    type=click.INT,
    default=1,
    show_default=True,
    help="# of rollouts for sampling.",
)
@click.option(
    "--eval-period",
    type=click.INT,
    default=1,
    show_default=True,
    help="Period evaluation.",
)
def cli_run_sac(**kwargs):
    """Run SAC Algorithm."""
    run_sac(**kwargs)
