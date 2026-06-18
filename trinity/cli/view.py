import os
from typing import Optional

import typer
from typing_extensions import Annotated


def _resolve_db_url(url: str) -> str:
    """Normalize a database specifier into a connection URL.

    Accepts a plain path to a ``.db`` file (relative or absolute) and converts
    it to a ``sqlite:///`` URL. Anything that already looks like a DB URL
    (``sqlite:///``, ``postgresql://``, ...) is passed through unchanged.
    """
    if "://" in url:
        return url
    return "sqlite:///" + os.path.abspath(os.path.expanduser(url))


def _resolve_from_config(config_path: str) -> tuple[str, str, str]:
    """Resolve (db_url, table_name, tokenizer_path) from a Trinity config file.

    Reads the experience pipeline's saved input. The viewer is only usable when
    the pipeline actually writes a SQL database, so requests are rejected when
    ``save_input`` is disabled or ``input_save_path`` is not a database URL.
    """
    from trinity.buffer.storage.queue import is_database_url
    from trinity.common.config import load_config

    config = load_config(config_path)
    pipeline_cfg = config.data_processor.experience_pipeline

    if not pipeline_cfg.save_input:
        raise typer.BadParameter(
            f"Cannot view from {config_path}: "
            "data_processor.experience_pipeline.save_input is False, "
            "so no input database is produced."
        )
    save_path = pipeline_cfg.input_save_path
    if save_path is None:
        # save_input is on but no path given: the pipeline writes a SQLite DB
        # under the buffer cache dir. Resolve the same default so `view` works
        # without a full config validation pass.
        from trinity.buffer.pipelines.experience_pipeline import default_input_save_path

        save_path = default_input_save_path(config)
    if not is_database_url(save_path):
        raise typer.BadParameter(
            f"Cannot view from {config_path}: "
            "data_processor.experience_pipeline.input_save_path "
            f"({save_path!r}) is not a database URL, so no SQL database is produced."
        )

    tokenizer_path = config.model.model_path
    if not tokenizer_path:
        raise typer.BadParameter(f"Cannot view from {config_path}: model.model_path is not set.")

    _warn_if_renamed(config)

    # Mirrors ExperiencePipeline: the SQL input writer is created with table
    # name "pipeline_input" (see trinity/buffer/pipelines/experience_pipeline.py).
    return save_path, "pipeline_input", tokenizer_path


def _warn_if_renamed(config) -> None:
    """Warn if the resolved path may not match where the DB was actually written.

    At run time, ``GlobalConfigValidator`` auto-renames an experiment (appending a
    timestamp to ``name``) when ``continue_from_checkpoint`` is False and the job
    directory already exists and is non-empty. Since ``view`` resolves the clean
    (pre-rename) path, the database in that case lives under a timestamped sibling
    ``<name>_<timestamp>/`` and the clean path may be stale or point at a different run.
    """
    import glob

    job_dir = config.get_checkpoint_job_dir()
    if config.continue_from_checkpoint or not os.path.isdir(job_dir) or not os.listdir(job_dir):
        return

    parent = os.path.dirname(job_dir)
    candidates = sorted(
        d for d in glob.glob(os.path.join(parent, f"{config.name}_*")) if os.path.isdir(d)
    )
    hint = ""
    if candidates:
        hint = " Likely candidate(s): " + ", ".join(os.path.basename(d) for d in candidates)

    typer.secho(
        "Warning: experiment was likely auto-renamed at run time "
        "(continue_from_checkpoint=False and the job directory already exists). "
        f"The resolved database path is under {job_dir!r}, but the actual database "
        f"is probably under a timestamped sibling {config.name}_<timestamp>/{hint} "
        "Pass --url directly at the database file to avoid ambiguity.",
        fg=typer.colors.YELLOW,
        err=True,
    )


def view_command(
    url: Annotated[
        Optional[str],
        typer.Option(
            "--url",
            help=(
                "Database specifier for the experience table. Accepts either a DB URL "
                "(e.g. sqlite:////path/to/debug_buffer.db) or a plain path to a .db file "
                "(relative or absolute), which is converted to a sqlite URL automatically. "
                "Optional when --config is given; a value here overrides the config."
            ),
        ),
    ] = None,
    table: Annotated[
        Optional[str],
        typer.Option("--table", help="Name of the experience table to monitor."),
    ] = None,
    tokenizer: Annotated[
        Optional[str],
        typer.Option(
            "--tokenizer",
            help="Tokenizer/model path used to decode token ids in the viewer.",
        ),
    ] = None,
    config: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            help=(
                "Path to a Trinity config file. When set, --url/--tokenizer/--table are "
                "inferred from it (experience pipeline input database + model.model_path); "
                "explicit CLI args still override the config."
            ),
        ),
    ] = None,
    schema: Annotated[
        str,
        typer.Option(
            "--schema",
            help="Schema type of the table. Supported values: experience, sft.",
        ),
    ] = "experience",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="The port for Experience Viewer."),
    ] = 8502,
) -> None:
    """Run the Streamlit viewer to inspect an experience table."""
    schema = schema.lower()
    if schema not in {"experience", "sft"}:
        raise typer.BadParameter("--schema only supports 'experience' or 'sft'.")

    db_url: Optional[str] = None
    table_name: Optional[str] = None
    tokenizer_path: Optional[str] = None

    if config is not None:
        db_url, table_name, tokenizer_path = _resolve_from_config(config)

    # Explicit CLI arguments take precedence over config-derived values.
    if url is not None:
        db_url = _resolve_db_url(url)
    if table is not None:
        table_name = table
    if tokenizer is not None:
        tokenizer_path = tokenizer

    if db_url is None:
        raise typer.BadParameter("--url is required (or provide --config).")
    if table_name is None:
        raise typer.BadParameter("--table is required (or provide --config).")
    if tokenizer_path is None:
        raise typer.BadParameter("--tokenizer is required (or provide --config).")

    from trinity.buffer.viewer import SQLExperienceViewer

    SQLExperienceViewer.run_viewer(
        model_path=tokenizer_path,
        db_url=db_url,
        table_name=table_name,
        schema_type=schema,
        port=port,
    )
