import os
import click
import torch
import random
import numpy as np
from rl import SAVE_DIR

from rl.runner.sac import run_sac


@click.command()
@click.option(
    "--exp-name",
    type=click.STRING,
    required=True,
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
def sac(
    exp_name: str,
    env_id: str,
    seed: int,
    auto_tmp: bool,
    tmp: float,
    **agent_kwargs,
) -> None:
    """Soft Actor Critic."""
    # Fix seed.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    exp_dir = SAVE_DIR / env_id / exp_name
    os.makedirs(exp_dir, exist_ok=True)
    tmp = "auto" if auto_tmp else tmp
    run_sac(env_id, exp_dir, tmp=tmp, **agent_kwargs)
