import click
from rl.cli.train.sac.mujoco import mujoco
from rl.cli.train.sac.metaworld import metaworld


@click.group()
def sac():
    """Soft Actor Critic."""


sac.add_command(mujoco)
sac.add_command(metaworld)

__all__ = ["sac"]
