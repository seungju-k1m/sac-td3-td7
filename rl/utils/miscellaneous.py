import json
from typing import Any
from pathlib import PosixPath

import click


def convert_dict_as_param(d_str_value: dict[str, Any]) -> dict[str, Any]:
    """Convert dict as param format."""
    param: dict[str, Any] = dict()
    for key, value in d_str_value.items():
        if not isinstance(value, dict):
            if isinstance(value, PosixPath):
                param[key] = str(value)
            else:
                param[key] = value
        else:
            param.update(value)
    return param


def configure(ctx: click.core.Context, param: click.core.Option, path: str):
    """Configure."""
    with open(path) as file_handler:
        options = json.load(file_handler)
    ctx.default_map = options
