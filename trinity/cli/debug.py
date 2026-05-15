import asyncio
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import ray
import typer
from typing_extensions import Annotated

from trinity.common.config import Config, load_config
from trinity.common.constants import DEBUG_NAMESPACE, PLUGIN_DIRS_ENV_VAR
from trinity.utils.log import get_logger
from trinity.utils.plugin_loader import load_plugins

if TYPE_CHECKING:
    from trinity.common.models.model import ModelWrapper


logger = get_logger(__name__)


async def create_debug_models(config: Config) -> None:
    from trinity.common.models.allocator import Allocator

    allocator = Allocator(config.explorer)
    rollout_models, auxiliary_models = await allocator.create_all_models()
    logger.info(
        "----------------------------------------------------\n"
        "Inference models started successfully for debugging.\n"
        "Press Ctrl+C to exit.\n"
        "----------------------------------------------------"
    )
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down debug models...")
        # wait for all models to shutdown
        models = rollout_models + [model for sublist in auxiliary_models for model in sublist]
        await asyncio.gather(*[model.shutdown() for model in models])
        ray.shutdown()


async def get_debug_models(config: Config) -> Tuple["ModelWrapper", List["ModelWrapper"]]:
    from trinity.common.models.allocator import Allocator

    allocator = Allocator(config.explorer)
    rollout_model = allocator.get_model(config.explorer.rollout_model, "rollout", 0)
    auxiliary_models = [
        allocator.get_model(auxiliary_model_config, f"auxiliary_{index}", 0)
        for index, auxiliary_model_config in enumerate(config.explorer.auxiliary_models)
    ]
    await asyncio.gather(
        rollout_model.prepare(),
        *[auxiliary_model.prepare() for auxiliary_model in auxiliary_models],
    )
    return rollout_model, auxiliary_models


def debug(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to the config file."),
    ],
    module: Annotated[
        str,
        typer.Option(
            "--module",
            "-m",
            help="The module to debug: 'inference_model', 'workflow', or 'viewer'.",
        ),
    ],
    plugin_dir: Annotated[
        Optional[str],
        typer.Option("--plugin-dir", help="Path to the directory containing plugin modules."),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", "-o", help="The output directory for debug files."),
    ] = "debug_output",
    disable_overwrite: Annotated[
        bool,
        typer.Option("--disable-overwrite", help="Disable overwriting the output directory."),
    ] = False,
    enable_profiling: Annotated[
        bool,
        typer.Option("--enable-profiling", help="Whether to use viztracer for workflow profiling."),
    ] = False,
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="The port for Experience Viewer."),
    ] = 8502,
) -> None:
    """Debug a workflow implementation."""
    valid_modules = ("inference_model", "workflow", "viewer")
    if module not in valid_modules:
        raise typer.BadParameter(f"Only support {valid_modules} for debugging, got '{module}'")

    if plugin_dir:
        os.environ[PLUGIN_DIRS_ENV_VAR] = plugin_dir
    load_plugins()
    cfg = load_config(config)
    cfg.mode = "explore"
    cfg.ray_namespace = DEBUG_NAMESPACE
    cfg.explorer.rollout_model.engine_num = 1
    for auxiliary_model_config in cfg.explorer.auxiliary_models:
        auxiliary_model_config.engine_num = 1
    cfg.check_and_update()
    sys.path.insert(0, os.getcwd())
    ray.init(
        namespace=cfg.ray_namespace,
        runtime_env={"env_vars": cfg.get_envs()},
        ignore_reinit_error=True,
    )

    if module == "inference_model":
        asyncio.run(create_debug_models(cfg))

    elif module == "workflow":
        from trinity.explorer.workflow_runner import DebugWorkflowRunner

        runner = DebugWorkflowRunner(cfg, output_dir, enable_profiling, disable_overwrite)
        logger.info("Preparing debug workflow runner...")
        asyncio.run(runner.prepare())
        logger.info("Running debug workflow runner...")
        asyncio.run(runner.debug())

    elif module == "viewer":
        from trinity.buffer.viewer import SQLExperienceViewer

        output_dir_abs = os.path.abspath(output_dir).rstrip("/")

        SQLExperienceViewer.run_viewer(
            model_path=cfg.model.model_path,
            db_url=f"sqlite:///{os.path.join(output_dir_abs, 'debug_buffer.db')}",
            table_name="debug_buffer",
            schema_type="experience",
            port=port,
        )
