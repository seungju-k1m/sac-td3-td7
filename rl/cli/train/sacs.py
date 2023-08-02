import ray
import json
import click

from rl.runner.sac import run_sac


@click.command()
@click.option(
    "-c", "--config", multiple=True, type=click.Path(exists=True), required=True
)
def sacs(
    config: tuple[str],
) -> None:
    """Soft Actor Critic."""
    # Fix seed.
    experiments: list[dict] = list()
    with open("config/default.json") as file_handler:
        default_config = json.load(file_handler)
    for path in config:
        with open(path) as file_handler:
            experiment = json.load(file_handler)
            experiment.update(default_config)
            experiments.append(experiment)
    n_experiment = len(experiments)
    if n_experiment > 1:
        ray.init()
        remote_run_sac = ray.remote(run_sac)
        ray_objs = [remote_run_sac.remote(**experiment) for experiment in experiments]
        ray.get(ray_objs)
    else:
        experiment = experiments[0]
        experiment["print_mode"] = True
        run_sac(**experiment)
