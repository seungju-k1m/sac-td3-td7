"""Command Line Interface."""
import os
import click
import ray
from rl import SEEDS
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
    help="Temperature for balancing exploration and exploitation. \
If use <auto-tmp>, it doesn't work.",
    show_default=True,
)
@click.option(
    "--auto-tmp",
    type=bool,
    default=False,
    is_flag=True,
)
@click.option(
    "--action-fn",
    type=click.STRING,
    default="ReLU",
    show_default=True,
    help="Activation function.",
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
    default=10_000_000,
    show_default=True,
    help="# of iteration.",
)
@click.option(
    "--batch-size", type=click.INT, default=256, show_default=True, help="Batch size."
)
# Know-how
@click.option(
    "--use-checkpoint",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use Checkpoint",
    is_flag=True,
)
@click.option(
    "--valid-benchmark",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Running multiple identical experiments in parallel \
with only different seeds means.",
    is_flag=True,
)
@click.option("--seed", type=click.INT, default=777, show_default=True, help="Seed.")
def cli_run_sac(valid_benchmark: bool, **kwargs):
    """Run SAC Algorithm."""
    if valid_benchmark:
        n_cpus = -1 if os.cpu_count() < 8 else 8
        ray.init(num_cpus=n_cpus)
        remote_fn_sac = ray.remote(run_sac)
        if "seed" in kwargs.keys():
            kwargs.pop("seed")
        ray_objs = [
            remote_fn_sac.remote(
                seed=seed,
                benchmark_idx=idx + 1,
                **kwargs,
            )
            for idx, seed in enumerate(SEEDS)
        ]
        ray.get(ray_objs)
    else:
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
@click.option(
    "--valid-benchmark",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Running multiple identical experiments in parallel \
with only different seeds means.",
    is_flag=True,
)
@click.option("--seed", type=click.INT, default=777, show_default=True, help="Seed.")
def cli_run_td3(valid_benchmark: bool, **kwargs):
    """Run SAC Algorithm."""
    if valid_benchmark:
        n_cpus = -1 if os.cpu_count() < 8 else 8
        ray.init(num_cpus=n_cpus)
        remote_fn_td3 = ray.remote(run_td3)
        if "seed" in kwargs.keys():
            kwargs.pop("seed")
        ray_objs = [
            remote_fn_td3.remote(
                seed=seed,
                benchmark_idx=idx + 1,
                **kwargs,
            )
            for idx, seed in enumerate(SEEDS)
        ]
        ray.get(ray_objs)
    else:
        run_td3(**kwargs)
