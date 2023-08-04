import click
from rl.runner.metaworld import run_metaworld


@click.command()
@click.option(
    "--exp-name",
    type=click.STRING,
    # required=True,
    help="The experiment-name for logging.",
)
@click.option(
    "--benchmark",
    type=click.Choice(
        ["ML1", "MT1", "ML10", "ML45", "MT10", "MT50"],
    ),
    default="MT1",
    show_default=True,
    help="Benchmark in MetaWorld.",
)
@click.option(
    "--task-id",
    type=click.STRING,
    default="pick-place-v2",
    help="The env id registered in gym.",
    show_default=True,
)
@click.option(
    "--task-idx", type=click.INT, default=0, show_default=True, help="The task index."
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
    default=500,
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
def metaworld(**agent_kwargs) -> None:
    """Run Soft Actor Critic in meta world."""
    run_metaworld(print_mode=True, **agent_kwargs)
