"""Command Line Interface."""
from copy import deepcopy
import click
import pandas as pd


from rl.docs import SAC_HELP, TD3_HELP, TD7_HELP
from rl.agent import run_sac, run_td3, run_td7
from rl.replayer import Replayer

from rl.utils.cli_utils import common_params_for_rl_alg
from rl.utils.miscellaneous import convert_dict_as_param

@click.command(name="sac", help=SAC_HELP)
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
@common_params_for_rl_alg
def cli_run_sac(**kwargs):
    """Run SAC Algorithm."""
    run_sac(**kwargs)


@click.command(name="td3", help=TD3_HELP)
@click.option("--action-fn", type=click.STRING, default="ReLU", show_default=True)
@click.option(
    "--use-gpu",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use GPU.",
    is_flag=True,
)
@common_params_for_rl_alg
def cli_run_td3(**kwargs):
    """Run TD3 Algorithm."""
    run_td3(**kwargs)


@click.command(name="td7", help=TD7_HELP)
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
    "--use-gpu",
    type=click.BOOL,
    default=False,
    show_default=True,
    help="Use GPU.",
    is_flag=True,
)
@common_params_for_rl_alg
def cli_run_td7(**kwargs):
    """Run TD7 Algorithm."""
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
    """Replay Trained Agent.

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
