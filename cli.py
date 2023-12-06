import click
from click_repl import register_repl

from rl.cli import cli_run_sac, cli_run_td3, cli_run_td7


@click.group()
def main():
    """CLI."""


main.add_command(cli_run_sac)
main.add_command(cli_run_td3)
main.add_command(cli_run_td7)

if __name__ == "__main__":
    register_repl(main)
    main()
