import click


from rl.cli import cli_run_sac


@click.group()
def main():
    """CLI."""


main.add_command(cli_run_sac)

if __name__ == "__main__":
    main()
