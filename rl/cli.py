"""Command Line Interface."""
from copy import deepcopy
import os
import click
import pandas as pd
import ray


from rl import SEEDS
from rl.agent import run_sac, run_td3, run_td7
from rl.replayer import Replayer

from rl.utils.cli_utils import configure
from rl.utils.miscellaneous import convert_dict_as_param


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
    help="# of train iteration.",
)
@click.option(
    "--n-initial-exploration-steps",
    type=click.INT,
    default=25_000,
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
@click.option("--seed", type=click.INT, default=42, show_default=True, help="Seed.")
def cli_run_sac(valid_benchmark: bool, **kwargs):
    """
    Run SAC Algorithm.

    Examples :

    # Train SAC Agent with Ant-v4 Env\n
    >>> rl sac Ant-v4 ant@auto\n\n
    # Train SAC Agent with fixed temperature\n
    >>> rl sac Ant-v4 ant@tmp20 --tmp 0.2\n
    # If you want to record video while training\n\n
    >>> rl sac Ant-v4 ant@auto --record-video\n
    # If you want to run several experiments
    in parallel with only different seeds\n
    >>> rl sac Ant-v4 ant@auto --valid-benchmark
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
    """
    Run TD3 Algorithm.

    Examples :

    # Train TD3 Agent with Ant-v4 Env\n
    >>> rl td3 Ant-v4 td3\n\n
    # If you want to record video while training\n\n
    >>> rl td3 Ant-v4 ant@auto --record-video\n
    # If you want to run several experiments
    in parallel with only different seeds\n
    >>> rl td3 Ant-v4 ant@auto --valid-benchmark
    """
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
    """
    Run TD7 Algorithm.

    Examples :

    # Train TD7 Agent with Ant-v4 Env\n
    >>> rl td7 Ant-v4 td3\n\n
    # If you want to record video while training\n\n
    >>> rl td7 Ant-v4 ant@auto --record-video\n
    # If you want to run several experiments
    in parallel with only different seeds\n
    >>> rl td7 Ant-v4 ant@auto --valid-benchmark
    """
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


@click.command(name="replay")
@click.option(
    "--root-dir",
    required=True,
    type=click.Path(exists=True),
    help="RL Agent dir. save/<rl_alg>/*",
)
@click.option(
    "--use-ckpt-model",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use Checkpoint model.",
    is_flag=True,
)
@click.option(
    "--stochastic",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Sample action in stochastic way.",
    is_flag=True,
)
@click.option("--seed", type=click.INT, default=42, show_default=True, help="Seed.")
@click.option(
    "--n-episodes", type=click.INT, default=8, show_default=True, help="# of episodes."
)
@click.option(
    "--video-dir",
    type=click.Path(exists=True),
    show_default=True,
    default=None,
    help="Video directory. If you don't set `video_dir`, replay vidoes \
are saved in root_dir/replayer",
)
def cli_replay_agent(n_episodes: int, stochastic: bool, **kwargs) -> None:
    """
    Replay Trained Agent.

    Examples:

    >>> replay --root-dir save/<RL_ALG>/<Train_DIR>\n
    # If you want to sample action in stochastic way,\n
    >>> replay --root-dir save/<RL_ALG>/<Train_DIR> --stochastic\n
    # If you want to record 16 episodes,\n
    >>> replay --root-dir save/<RL_ALG>/<Train_DIR> --n-episodes\n

    """
    params = convert_dict_as_param(deepcopy(locals()))
    print("-" * 5 + "[Replay Agent]" + "-" * 5)
    print(" " + pd.Series(params).to_string().replace("\n", "\n "))
    print()
    replayer = Replayer(**kwargs)
    replayer.run(n_episodes, stochastic)
