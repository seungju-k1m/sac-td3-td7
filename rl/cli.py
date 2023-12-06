"""Command Line Interface."""
import os
import click
import ray


from rl import SEEDS
from rl.agent import run_sac
from rl.agent.td3 import run_td3
from rl.agent.td7 import run_td7

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
    default=-1.0,
    help="Temperature for balancing exploration and exploitation. \
If tmp is negative, `auto_tmp_mode` works.",
    show_default=True,
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
    default=5_000_000,
    show_default=True,
    help="# of iteration.",
)
@click.option(
    "--n-initial-exploration-steps",
    type=click.INT,
    default=10_000,
    show_default=True,
    help="# of transition using randon policy.",
)
@click.option(
    "--batch-size", type=click.INT, default=256, show_default=True, help="Batch size."
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
@click.option(
    "--record-video",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Record Video.",
    is_flag=True,
)
@click.option(
    "--use-gpu",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use GPU.",
    is_flag=True,
)
@click.option("--seed", type=click.INT, default=42, show_default=True, help="Seed.")
def cli_run_sac(valid_benchmark: bool, **kwargs):
    """Run SAC Algorithm.

    Examples:

        >>> sac Ant-v4 ant@auto\n
        >>> # If you want to record video while training\n
        >>> sac Ant-v4 ant@auto --record-video\n
        >>> # If you want to run several experiments
        in parallel with only different seeds,\n
        >>> sac Ant-v4 ant@auto --valid-benchmark
    """
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
                show_progressbar=False,
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
@click.option(
    "--record-video",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Record Video.",
    is_flag=True,
)
@click.option(
    "--use-gpu",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use GPU.",
    is_flag=True,
)
@click.option("--seed", type=click.INT, default=777, show_default=True, help="Seed.")
def cli_run_td3(valid_benchmark: bool, **kwargs):
    """Run TD3 Algorithm."""
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
                show_progressbar=False,
                **kwargs,
            )
            for idx, seed in enumerate(SEEDS)
        ]
        ray.get(ray_objs)
    else:
        run_td3(**kwargs)


@click.command(name="td7")
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
    "--without-policy-checkpoint",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use Checkpoint",
    is_flag=True,
)
@click.option(
    "--without-lap",
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
@click.option(
    "--record-video",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Record Video.",
    is_flag=True,
)
@click.option(
    "--use-gpu",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use GPU.",
    is_flag=True,
)
@click.option("--seed", type=click.INT, default=777, show_default=True, help="Seed.")
def cli_run_td7(valid_benchmark: bool, **kwargs):
    """Run TD7 Algorithm."""
    if valid_benchmark:
        n_cpus = -1 if os.cpu_count() < 8 else 8
        ray.init(num_cpus=n_cpus)
        remote_fn_td3 = ray.remote(run_td7)
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
        run_td7(**kwargs)
