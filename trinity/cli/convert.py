import os
from typing import List, Optional

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
    step: Annotated[
        Optional[str],
        typer.Option(
            "--step",
            "-s",
            help="Specific step number(s) to convert. Comma-separated (e.g., 100,200,300) or repeated (-s 100 -s 200).",
        ),
    ] = None,
) -> None:
    """Convert model checkpoints to huggingface format."""
    from trinity.manager.checkpoint_converter import Converter

    converter = Converter(base_model_dir)

    # Parse step parameter (supports both "100,200,300" and multiple -s flags)
    step_list: List[int] = []
    if step:
        # Split by comma and/or whitespace, then convert to int
        for part in step.replace(",", " ").split():
            try:
                step_list.append(int(part))
            except ValueError:
                typer.echo(f"[ERROR] Invalid step number: {part}", err=True)
                raise typer.Exit(code=1)

    if step_list:
        _convert_multi_steps(step_list, checkpoint_dir, converter)
    else:
        # Original behavior: convert all or a single checkpoint
        dir_path = checkpoint_dir
        if "global_step_" in dir_path:
            while not os.path.basename(dir_path).startswith("global_step_"):
                dir_path = os.path.dirname(dir_path)
        converter.convert(dir_path)


def _convert_multi_steps(step_list: List[int], checkpoint_dir: str, converter) -> None:
    # When --step is provided, convert only the specified step checkpoints
    # Resolve the root directory that contains global_step_* folders
    root_dir = checkpoint_dir
    if "global_step_" in root_dir:
        while not os.path.basename(root_dir).startswith("global_step_"):
            root_dir = os.path.dirname(root_dir)
        # Go one level up to the parent that contains global_step_* dirs
        root_dir = os.path.dirname(root_dir)

    succeeded: List[int] = []
    failed: List[tuple] = []  # (step, reason)

    for s in step_list:
        step_dir = os.path.join(root_dir, f"global_step_{s}")
        if not os.path.isdir(step_dir):
            reason = f"Directory not found: {step_dir}"
            typer.echo(f"[ERROR] Step {s}: {reason}", err=True)
            failed.append((s, reason))
            continue
        try:
            converter.convert(step_dir)
            succeeded.append(s)
        except Exception as e:
            typer.echo(f"[ERROR] Step {s}: {e}", err=True)
            failed.append((s, str(e)))

    # Print summary report
    typer.echo("\n" + "=" * 50)
    typer.echo("Conversion Report")
    typer.echo("=" * 50)
    typer.echo(f"Total requested: {len(step_list)}")
    typer.echo(f"Succeeded:       {len(succeeded)}")
    typer.echo(f"Failed:          {len(failed)}")
    if succeeded:
        typer.echo(f"\nSucceeded steps: {', '.join(str(s) for s in succeeded)}")
    if failed:
        typer.echo("\nFailed steps:")
        for s, reason in failed:
            typer.echo(f"  Step {s}: {reason}")
    typer.echo("=" * 50)

    if failed:
        raise typer.Exit(code=1)
