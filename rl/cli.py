"""Command Line Interface."""
from copy import deepcopy
import click
import pandas as pd


from rl.agent import run_sac, run_td3, run_td7
from rl.replayer import Replayer

from rl.utils.cli_utils import common_params_for_rl_alg
from rl.utils.miscellaneous import convert_dict_as_param


@click.command(name="sac")
@click.option(
    "--tmp",
    type=float,
    default=-1.0,
    help="Temperature for balancing exploration and exploitation. \
If tmp is negative, `auto_tmp_mode` works.",
    show_default=True,
)
@common_params_for_rl_alg
def cli_run_sac(**kwargs):
    """
    Soft Actor Critic (SAC) Algorithm.

    Here is paper address:
    https://arxiv.org/abs/1801.01290

    Notes:

        - SAC is one of the most famous off-policy
    reinforcement learning algorithm. It is often used as baseline.

        - One of the key hyper-parameter for SAC is temperature, `tmp`.
    If you set the `tmp` as negative, auto-tmp mode works, which automatically
    tuning the temperature during training.

    Examples:

        # If you want to run SAC alg with fixed temperature.\n
        >>> python cli.py rl sac --env-id Ant-v4 --tmp 0.2 --n-iteration 1_000_000\n
        # If you want to record video while training.\n
        >>> python cli.py rl sac --env-id Ant-v4 --record-video"""
    run_sac(**kwargs)


@click.command(name="td3")
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
    """
    Twin Delayed Deep Deterministic Policy Gradient Algorithm (TD3).

    Here is paper address:
    https://arxiv.org/pdf/1802.09477.pdf

    Notes:

        - TD3 is a family of DDPG style algorithms.
    This algorithm handles the overestimation of value function introducing
    two state-action value functions.

    Examples:

        # If you want to run TD3 Algorithm for Ant-v4 of mujoco env with `td3@
    ant` run name.\n
        >>> python cli.py rl td3 --env-id Ant-v4 --run-name td3@ant\n
    """
    run_td3(**kwargs)


@click.command(name="td7")
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
@click.option(
    "--exploration-noise",
    type=click.FLOAT,
    default=0.1,
    show_default=True,
    help="Exploration noise.",
)
@click.option(
    "--target-policy-noise",
    type=click.FLOAT,
    default=0.2,
    show_default=True,
    help="Noise for Target Network.",
)
@click.option(
    "--max-episodes-per-single-ckpt",
    type=click.INT,
    default=20,
    show_default=True,
    help="# of episodes for updating single ckpt.",
)
@common_params_for_rl_alg
def cli_run_td7(**kwargs):
    """
    TD7 is variant of TD3 algorithm with 4 additions.

    Here is paper address:
    https://arxiv.org/pdf/2306.02451.pdf

    Notes:

        - State-Action Learned Embedding (SALE) Network.

        - LAP Replay Memory.

        - Checkpoint Policy Strategy.

        - Behavior Cloning Term for offline RL.


    Examples:

        >>> python cli.py rl td7 --env-id Ant-v4 --run-name td7@ant\n
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
