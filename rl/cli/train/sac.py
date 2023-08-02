import click

from rl.runner.sac import run_sac
from rl.utils.miscellaneous import configure


default_cfg_path = "config/hopper.json"


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False),
    default=default_cfg_path,
    callback=configure,
    is_eager=True,
    expose_value=False,
    help="Read option defaults from the specified INI file",
    show_default=True,
)
@click.option(
    "--exp-name",
    type=click.STRING,
    # required=True,
    help="The experiment-name for logging.",
)
@click.option(
    "--env-id",
    type=click.STRING,
    default="Hopper-v4",
    help="The env id registered in gym.",
    show_default=True,
)
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
    help="Coefficient for regulating policy.(squre of mean and log_std).",
    show_default=True,
)
@click.option(
    "--n-epochs",
    type=click.INT,
    default=1_000,
    show_default=True,
)
@click.option("--seed", type=click.INT, default=777, help="Seed", show_default=True)
def sac(**agent_kwargs) -> None:
    """Soft Actor Critic."""
    run_sac(print_mode=True, **agent_kwargs)
