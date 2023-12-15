"""Command Line Interface."""
from copy import deepcopy
import click
import pandas as pd


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
@click.option(
    "--run-name",
    default=None,
    type=str,
    help="Run experiment would be saved in save/<ALG_NAME>/<run_name>-<timestamp>",
    show_default=True,
)
@click.option(
    "--env-id",
    default="Hopper-v4",
    type=str,
    help="Run experiment would be saved in save/<ALG_NAME>/<run_name>-<timestamp>",
    show_default=True,
)
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
    "--record-video",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Record Video.",
    is_flag=True,
)
@click.option("--seed", type=click.INT, default=42, show_default=True, help="Seed.")
def cli_run_sac(**kwargs):
    """
    Run SAC Algorithm.

    Examples :
    """
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
@click.option(
    "--run-name",
    default=None,
    type=str,
    help="Run experiment would be saved in save/<ALG_NAME>/<run_name>-<timestamp>",
    show_default=True,
)
@click.option(
    "--env-id",
    default="Hopper-v4",
    type=str,
    help="Run experiment would be saved in save/<ALG_NAME>/<run_name>-<timestamp>",
    show_default=True,
)
@click.option("--action-fn", type=click.STRING, default="ReLU", show_default=True)
@click.option("--discount-factor", type=click.FLOAT, default=0.99, show_default=True)
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
def cli_run_td3(**kwargs):
    """
    Run TD3 Algorithm.

    Examples :
    """
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
@click.option(
    "--run-name",
    default=None,
    type=str,
    help="Run experiment would be saved in save/<ALG_NAME>/<run_name>-<timestamp>",
    show_default=True,
)
@click.option(
    "--env-id",
    default="Hopper-v4",
    type=str,
    help="Run experiment would be saved in save/<ALG_NAME>/<run_name>-<timestamp>",
    show_default=True,
)
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
    """
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
