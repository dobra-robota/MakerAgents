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


@app.command()
def report(
    run_dir: str = typer.Argument(..., help="Path to the run folder, e.g. runs/20250101-120000-lodz-senior-citizens"),
) -> None:
    """Re-render the final report from existing on-disk run state."""
    from pathlib import Path

    from makeragents.agents.report import ReportAgent

    run_path = Path(run_dir)
    if not run_path.is_dir():
        typer.echo(f"Error: run folder not found: {run_dir}", err=True)
        raise typer.Exit(code=1)

    agent = ReportAgent()
    report_path = agent.generate(run_path)
    typer.echo(f"Report generated: {report_path}")
    typer.echo(f"  Appendix written to: {run_path / 'appendix'}")


if __name__ == "__main__":
    app()
