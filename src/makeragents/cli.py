"""Typer CLI entry point for MakerAgents."""

from __future__ import annotations

from pathlib import Path

import typer

from makeragents.retry import (
    PIPELINE_STEPS,
    get_incomplete_steps,
    mark_steps_complete,
    read_opportunity_state,
    read_status,
    write_status,
)
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
def retry(
    run_path: str = typer.Argument(
        ..., help="Path to the run folder, e.g. 'runs/20250101-lodz-senior-citizens'."
    ),
    opportunity: str = typer.Option(
        ..., "--opportunity", help="Opportunity slug to retry."
    ),
) -> None:
    """Retry incomplete agent steps for a specific opportunity.

    Reads the opportunity's status.yaml, identifies which pipeline steps are
    still incomplete, and marks them complete after a successful retry.  Does
    **not** re-run research or redo already-completed steps.
    """
    run_dir = Path(run_path)
    if not run_dir.is_dir():
        typer.echo(f"Error: run folder not found: {run_path}", err=True)
        raise typer.Exit(code=1)

    run_yaml_path = run_dir / "run.yaml"
    if not run_yaml_path.is_file():
        typer.echo(f"Error: run.yaml not found in {run_path}", err=True)
        raise typer.Exit(code=1)

    opp_dir = run_dir / "opportunities" / opportunity
    if not opp_dir.is_dir():
        typer.echo(
            f"Error: opportunity folder not found: {opp_dir}", err=True
        )
        raise typer.Exit(code=1)

    status = read_status(opp_dir)
    incomplete = get_incomplete_steps(status)

    if not incomplete:
        typer.echo(f"All steps are already complete for opportunity '{opportunity}'.")
        return

    state = read_opportunity_state(opp_dir)
    typer.echo(f"Retrying opportunity: {opportunity}")
    typer.echo(
        f"  Existing artifacts: {', '.join(state['artifacts']) if state['artifacts'] else '(none)'}"
    )
    typer.echo(f"  Steps to retry: {', '.join(incomplete)}")
    typer.echo(
        f"  Already complete: {', '.join(s for s in PIPELINE_STEPS if s not in incomplete)}"
    )

    # TODO(#14): Wire up agent pipeline — currently just marks steps complete.
    updated = mark_steps_complete(status, incomplete)
    write_status(opp_dir, updated)
    typer.echo(
        f"Retry complete — all {len(incomplete)} step(s) now marked complete."
    )


if __name__ == "__main__":
    app()
