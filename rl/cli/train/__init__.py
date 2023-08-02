import click

from rl.cli.train.sac import sac
from rl.cli.train.sacs import sacs


@click.group()
def train() -> None:
    """Train reinforcement learning algorith."""


train.add_command(sac)
train.add_command(sacs)

__all__ = ["train"]
