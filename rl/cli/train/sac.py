import os
import random
import click
import numpy as np
import torch
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
    type=click.FLOAT,
    default=0.2,
    help="Temperature for balancing exploration and exploitation.",
    show_default=True,
)
@click.option(
    "--policy-reg-coeff",
    type=click.FLOAT,
    default=1e-3,
    help="Coefficient for regulating policy.(squre of mean and log_std).",
    show_default=True,
)
@click.option("--seed", type=click.INT, default=777, help="Seed", show_default=True)
def sac(
    exp_name: str,
    env_id: str,
    seed: int,
    **agent_kwargs,
) -> None:
    """Soft Actor Critic."""
    # Fix seed.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    exp_dir = SAVE_DIR / exp_name
    os.makedirs(exp_dir, exist_ok=True)
    run_sac(env_id, exp_dir, **agent_kwargs)
