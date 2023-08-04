import click

from rl.runner.mujoco import run_mujoco
from rl.utils.miscellaneous import configure


default_cfg_path = "config/mujoco/hopper.json"


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
@click.option(
    "--action-fn",
    type=click.STRING,
    default="ReLU",
    show_default=True,
    help="The activation function.",
)
@click.option(
    "--device", type=click.STRING, default="cpu", show_default=True, help="Device."
)
@click.option(
    "--n-rollouts", type=click.INT, default=5, show_default=True, help="# of rollouts."
)
def mujoco(**agent_kwargs) -> None:
    """Soft Actor Critic."""
    run_mujoco(print_mode=True, **agent_kwargs)
