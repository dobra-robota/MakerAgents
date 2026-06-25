"""Typer CLI entry point for MakerAgents."""

from __future__ import annotations

import typer

from makeragents.run import build_run_metadata, create_run_folder

app = typer.Typer(help="MakerAgents command-line interface.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """MakerAgents command-line interface."""


@app.command()
def run(
    city: str = typer.Option(..., help="City to research, e.g. 'Łodz'."),
    community: str = typer.Option(
        ..., help="Community to research, e.g. 'senior citizens'."
    ),
    max_opportunities: int = typer.Option(
        5, min=1, help="Maximum number of opportunities to surface."
    ),
) -> None:
    """Create a stubbed run folder for a city-plus-community run."""

    metadata = build_run_metadata(
        city=city,
        community=community,
        max_opportunities=max_opportunities,
    )
    run_dir = create_run_folder(metadata)
    typer.echo(f"Created run folder: {run_dir}")


if __name__ == "__main__":
    app()
