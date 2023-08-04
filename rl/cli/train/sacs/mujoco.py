import ray
import json
import click
from copy import deepcopy

from rl.runner.mujoco import run_mujoco


@click.command()
@click.option(
    "-c", "--config", multiple=True, type=click.Path(exists=True), required=True
)
def mujoco(
    config: tuple[str],
) -> None:
    """Soft Actor Critic."""
    # Fix seed.
    experiments: list[dict] = list()
    with open("config/mujoco/default.json") as file_handler:
        default_config = json.load(file_handler)
    for path in config:
        with open(path) as file_handler:
            experiment = deepcopy(default_config)
            _experiment = json.load(file_handler)
            experiment.update(_experiment)
            experiments.append(experiment)
    n_experiment = len(experiments)
    if n_experiment > 1:
        ray.init()
        remote_run_sac = ray.remote(run_mujoco)
        ray_objs = [remote_run_sac.remote(**experiment) for experiment in experiments]
        ray.get(ray_objs)
    else:
        experiment = experiments[0]
        experiment["print_mode"] = True
        run_mujoco(**experiment)
