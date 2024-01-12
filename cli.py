import click
from click_repl import register_repl

from rl.cli import cli_run_sac, cli_run_td3, cli_run_td7, cli_replay_agent


@click.group()
def main():
    """CLI."""


@click.group()
def rl() -> None:
    """
    Run Off-Policy Reinforcement Learning Algorithms.


    Examples:

        # I recommend reading docstring of each command.

        >>> rl sac --help

        # Run default setting

        >>> rl sac

        # You can load hyper-parameter via *.yaml

        >> rl sac -c config/common.yaml

        # You can specify the run name.

        >>> rl sac --run-name Ant@Seed42

        # You want to record the video while training,

        >>> rl sac --record-video"""


rl.add_command(cli_run_sac)
rl.add_command(cli_run_td3)
rl.add_command(cli_run_td7)

main.add_command(rl)
main.add_command(cli_replay_agent)

if __name__ == "__main__":
    register_repl(main)
    main()
