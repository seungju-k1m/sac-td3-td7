import click

from rl.cli.train.sac import sac


@click.group()
def train() -> None:
    """Train reinforcement learning algorith."""


train.add_command(sac)

__all__ = ["train"]
