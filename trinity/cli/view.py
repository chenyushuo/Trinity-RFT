import typer
from typing_extensions import Annotated


def view_command(
    url: Annotated[
        str,
        typer.Option(
            "--url",
            help="Database URL for the experience table, for example sqlite:////path/to/debug_buffer.db.",
        ),
    ],
    table: Annotated[
        str,
        typer.Option("--table", help="Name of the experience table to monitor."),
    ],
    tokenizer: Annotated[
        str,
        typer.Option(
            "--tokenizer",
            help="Tokenizer/model path used to decode token ids in the viewer.",
        ),
    ],
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
    from trinity.buffer.viewer import SQLExperienceViewer

    schema = schema.lower()
    if schema not in {"experience", "sft"}:
        raise typer.BadParameter("--schema only supports 'experience' or 'sft'.")

    SQLExperienceViewer.run_viewer(
        model_path=tokenizer,
        db_url=url,
        table_name=table,
        schema_type=schema,
        port=port,
    )
