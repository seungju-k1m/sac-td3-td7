import click

from rl.cli.train.sacs.mujoco import mujoco


@click.group()
def sacs():
    """Run SAC Algorithms."""


sacs.add_command(mujoco)

__all__ = ["sacs"]
