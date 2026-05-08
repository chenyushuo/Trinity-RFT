import os
import traceback
from typing import Optional

import typer
from typing_extensions import Annotated

from trinity.common.constants import PLUGIN_DIRS_ENV_VAR

perf_app = typer.Typer(help="Performance testing tools.")


@perf_app.command("run")
def perf_run(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to the config file."),
    ],
    module: Annotated[
        str,
        typer.Option(
            "--module", "-m", help="Perf module to run. Currently only supports 'explorer'."
        ),
    ] = "explorer",
    output_path: Annotated[
        str,
        typer.Option("--output-path", "-o", help="Path to the output JSON file."),
    ] = "./perf/output.json",
    monitor_interval: Annotated[
        float,
        typer.Option("--monitor-interval", help="Resource sampling interval in seconds."),
    ] = 2.0,
    total_steps: Annotated[
        int,
        typer.Option("--total-steps", help="Total steps to run the explorer for."),
    ] = 5,
    timeout: Annotated[
        Optional[float],
        typer.Option(
            "--timeout", help="Optional timeout in seconds for prepare, sync and explore calls."
        ),
    ] = None,
    plugin_dir: Annotated[
        Optional[str],
        typer.Option("--plugin-dir", help="Path to the directory containing plugin modules."),
    ] = None,
) -> None:
    """Run performance benchmark."""
    if module != "explorer":
        raise typer.BadParameter("Only --module explorer is supported for now.")

    from trinity.perf import (
        ExplorerPerfOptions,
        run_explorer_perf,
        write_explorer_perf_output,
    )

    try:
        if plugin_dir:
            os.environ[PLUGIN_DIRS_ENV_VAR] = plugin_dir

        options = ExplorerPerfOptions(
            config_path=config,
            output_path=output_path,
            monitor_interval=monitor_interval,
            total_steps=total_steps,
            timeout=timeout,
        )
        payload = run_explorer_perf(options)
        write_explorer_perf_output(output_path, payload)
    except Exception:  # noqa: BLE001
        payload = {
            "status": {
                "success": False,
                "error": traceback.format_exc(),
            },
            "data": None,
        }
        write_explorer_perf_output(output_path, payload)

    if not payload["status"]["success"]:
        typer.echo(f"Failed to run perf: {payload['status']['error']}")


@perf_app.command("view")
def perf_view(
    report: Annotated[
        str,
        typer.Option("--report", "-r", help="Path to the perf report JSON file."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port used by the Streamlit report viewer."),
    ] = 8503,
) -> None:
    """Open the Streamlit perf report viewer."""
    from trinity.perf.report_viewer import launch_report_viewer

    launch_report_viewer(report, port)
