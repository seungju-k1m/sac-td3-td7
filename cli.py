import click


from rl.cli import cli_run_sac, cli_run_td3


@click.group()
def main():
    """CLI."""


main.add_command(cli_run_sac)
main.add_command(cli_run_td3)

if __name__ == "__main__":
    main()
