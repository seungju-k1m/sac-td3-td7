"""Utility code related to CLI."""

import json
import collections
from typing import Any

import click


class OrderedGroup(click.Group):
    """Order group."""

    def __init__(self, name: Any = None, commands: Any = None, **attrs):
        """Initialize."""
        super().__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx: click.Context) -> Any:
        """Update commands."""
        return self.commands


def configure(ctx: click.core.Context, param: click.core.Option, path: str | None):
    """Configure."""
    if path is None:
        return
    with open(path) as file_handler:
        options = json.load(file_handler)
    ctx.default_map = options


def run_cmd_callback(cmd: click.Command, *args, **kwargs) -> None:
    """Run click command's callback."""
    default_kwargs = {
        param.name: param.default
        for param in cmd.params
        if isinstance(param, click.core.Option)
    }
    if "config" in default_kwargs.keys():
        if kwargs["config"] is not None:
            with open(kwargs["config"]) as file_handler:
                default_kwargs = json.load(file_handler)
    idx = 0
    default_args = {}
    for param in cmd.params:
        if isinstance(param, click.core.Argument):
            default_args[param.name] = args[idx]
            idx += 1
    default_kwargs.update(kwargs)
    default_kwargs.update(default_args)
    if "config" in default_kwargs:
        del default_kwargs["config"]
    cmd.callback(**default_kwargs)
