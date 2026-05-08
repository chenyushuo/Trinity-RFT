import typer
from typing_extensions import Annotated


def studio_command(
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="The port for Trinity-Studio."),
    ] = 8501,
) -> None:
    """Run studio to manage configurations."""
    from trinity.manager.config_manager import ConfigManager

    ConfigManager.run(port)
