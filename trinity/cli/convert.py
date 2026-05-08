import os
from typing import Optional

import typer
from typing_extensions import Annotated


def convert_command(
    checkpoint_dir: Annotated[
        str,
        typer.Option("--checkpoint-dir", "-c", help="The path to the checkpoint directory."),
    ],
    base_model_dir: Annotated[
        Optional[str],
        typer.Option("--base-model-dir", "-b", help="The path to the base model."),
    ] = None,
) -> None:
    """Convert model checkpoints to huggingface format."""
    from trinity.manager.checkpoint_converter import Converter

    dir_path = checkpoint_dir
    if "global_step_" in dir_path:
        while not os.path.basename(dir_path).startswith("global_step_"):
            dir_path = os.path.dirname(dir_path)
    converter = Converter(base_model_dir)
    converter.convert(dir_path)
