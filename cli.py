import click
from click_repl import register_repl

from rl.cli import cli_run_sac, cli_run_td3, cli_run_td7, cli_replay_agent


@click.group()
def main():
    """CLI. Hello"""


@click.group()
def rl() -> None:
    """Off-Policy RL: SAC, TD3 and TD7."""


rl.add_command(cli_run_sac)
rl.add_command(cli_run_td3)
rl.add_command(cli_run_td7)

main.add_command(rl)
main.add_command(cli_replay_agent)

if __name__ == "__main__":
    register_repl(main)
    main()
