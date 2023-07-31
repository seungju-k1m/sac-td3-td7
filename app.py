import click


from rl.cli.train import train


@click.group()
def main():
    """CLI."""


main.add_command(train)

if __name__ == "__main__":
    main()
