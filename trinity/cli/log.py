import os
from typing import Optional

import typer
from typing_extensions import Annotated

from trinity.common.config import load_config


def log_command(
    log_dir: Annotated[
        str,
        typer.Option(
            "--log-dir",
            "-d",
            help="Path to the log directory. If provided, it will be used directly and ignore --config.",
        ),
    ] = "",
    config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="Path to the config file. If provided, it will automatically locate the log directory based on the config.",
        ),
    ] = "",
    keyword: Annotated[
        Optional[str],
        typer.Option(
            "--keyword",
            "-k",
            help="Only track log files containing the keyword in their filenames.",
        ),
    ] = None,
    level: Annotated[
        str,
        typer.Option("--level", "-l", help="The minimum log level to display in real-time."),
    ] = "INFO",
    last_n_lines: Annotated[
        int,
        typer.Option("--last-n-lines", "-n", help="Number of last lines to display when starting."),
    ] = 0,
    search_pattern: Annotated[
        Optional[str],
        typer.Option(
            "--search-pattern",
            "-p",
            help="The pattern to search in log files. Only search for history logs and display all lines containing the pattern.",
        ),
    ] = None,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output."),
    ] = False,
) -> None:
    """Monitor log files in real-time."""
    from trinity.manager.log_manager import LogManager

    if not config and not log_dir:
        raise typer.BadParameter("Either --config or --log-dir must be provided.")
    if not log_dir:
        cfg = load_config(config)
        checkpoint_job_dir = cfg.get_checkpoint_job_dir()
        # we do not use check_and_update here because user may use this command
        # in another environment
        log_dir = os.path.join(checkpoint_job_dir, "log")

    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    log_manager = LogManager(
        log_dir=log_dir,
        keyword=keyword,
        min_level=level,
        color_output=not no_color,
        last_n_lines=last_n_lines,
        search_pattern=search_pattern,
    )
    log_manager.monitor()
