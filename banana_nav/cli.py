import pathlib

import click
from unityagents import UnityEnvironment

from banana_nav import navigation


HERE = pathlib.Path(__file__).absolute().parent
DEFAULT_PATH_TO_BANANA_ENV = HERE.parent.joinpath("Banana_Linux/Banana.x86_64")


@click.group()
def cli():
    pass


@cli.command(
    "demo-dqn",
    help="Run a demo of Banana Navigation agent - trained or random (if no weights provided)")
@click.argument(
    "PATH_WEIGHTS",
    required=False,
    type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.option(
    "--unity-banana-env", "-u",
    type=click.Path(dir_okay=False, file_okay=True, exists=True),
    default=DEFAULT_PATH_TO_BANANA_ENV,
    help=f"Path to Unity Banana Environment executable, default: {DEFAULT_PATH_TO_BANANA_ENV}")
def demo(path_weights, unity_banana_env):
    env = UnityEnvironment(file_name=str(unity_banana_env))
    if path_weights is None:
        click.echo("Using Random agent")
    else:
        click.echo(f"Loading trained agent model weights from {path_weights.absolute()}")
    score = navigation.demo(env, path_weights)
    click.echo(
        "Episode completed with {'random' if path_weights is None else 'trained'} agent. "
        f"Score: {score:2f}")
    env.close()
