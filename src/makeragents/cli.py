"""Typer CLI entry point for MakerAgents."""

from __future__ import annotations

from pathlib import Path

import typer

from makeragents.agents.report import ReportAgent
from makeragents.config import load_config
from makeragents.orchestrator import PipelineRunner
from makeragents.retry import (
    PIPELINE_STEPS,
    _RETRYABLE_STEPS,
    get_incomplete_steps,
    load_opportunity_for_retry,
    mark_steps_complete,
    read_opportunity_state,
    read_status,
    run_retry_step,
    write_status,
)
from makeragents.run import build_run_metadata, create_run_folder
from makeragents.schemas import ScoreValue
from makeragents.sources.registry import (
    RUN_REGISTRY_RELATIVE_PATH,
    SourceRegistry,
    load_registry,
)

app = typer.Typer(help="MakerAgents command-line interface.", no_args_is_help=True)

# -- sub-command groups -------------------------------------------------------

sources_app = typer.Typer(help="Inspect and manage source trust registries.")
app.add_typer(sources_app, name="sources")


@app.callback()
def main() -> None:
    """MakerAgents command-line interface."""


# -- run ----------------------------------------------------------------------


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
    """Research a city + community and produce a ranked opportunity report."""

    # 1. Load config and validate required API keys.
    config = load_config()
    if not config.deepseek_api_key:
        typer.echo(
            "Error: DEEPSEEK_API_KEY is not set.  "
            "Please set it in your environment or .mise.toml.",
            err=True,
        )
        raise typer.Exit(code=1)

    # 2. Build metadata and create the run folder.
    metadata = build_run_metadata(
        city=city,
        community=community,
        max_opportunities=max_opportunities,
    )
    run_dir = create_run_folder(metadata)

    # 3. Run the full pipeline.
    runner = PipelineRunner(config=config)
    final_report = runner.run(run_dir, metadata)

    # 4. Echo a short summary.
    typer.echo(f"Run directory: {run_dir}")
    typer.echo(f"Final report:  {final_report}")
    typer.echo(
        f"Completed run for '{city} / {community}' "
        f"(max opportunities: {max_opportunities})."
    )


# -- report -------------------------------------------------------------------


@app.command()
def report(
    run_path: str = typer.Argument(
        ..., help="Path to the run directory, e.g. 'runs/20250626-120000-lodz-senior-citizens'."
    ),
) -> None:
    """Re-render final-report.md from on-disk run state.

    This reads existing run artifacts (run.yaml, source registry, evidence
    items, and opportunities) and regenerates final-report.md without
    calling any agents or search providers.
    """
    run_dir = Path(run_path)
    run_yaml = run_dir / "run.yaml"
    if not run_yaml.exists():
        typer.echo(f"Error: No run found at {run_path}", err=True)
        raise typer.Exit(code=1)

    agent = ReportAgent()
    dest = agent.generate(run_dir)
    typer.echo(f"Report written: {dest}")


# -- retry --------------------------------------------------------------------


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

    # Separate retryable steps (maker/taker/mediator/cost_checker) from
    # pre-processing steps (research/evidence/opportunity) which cannot be
    # re-run from on-disk state alone (PRD §15).
    retryable = [s for s in incomplete if s in _RETRYABLE_STEPS]
    skipped = [s for s in incomplete if s not in _RETRYABLE_STEPS]

    typer.echo(f"Retrying opportunity: {opportunity}")
    typer.echo(
        f"  Existing artifacts: {', '.join(state['artifacts']) if state['artifacts'] else '(none)'}"
    )

    if skipped:
        typer.echo(
            f"  Skipping (pre-processing, cannot be re-run): {', '.join(skipped)}"
        )

    if not retryable:
        if skipped:
            typer.echo(
                "All incomplete steps are pre-processing steps that cannot be "
                "re-run from on-disk state."
            )
        else:
            typer.echo(
                f"All steps are already complete for opportunity '{opportunity}'."
            )
        return

    typer.echo(f"  Steps to retry: {', '.join(retryable)}")
    typer.echo(
        f"  Already complete: {', '.join(s for s in PIPELINE_STEPS if s not in incomplete)}"
    )

    # Load on-disk state needed by downstream agents.
    try:
        opp, evidence_items = load_opportunity_for_retry(opp_dir, run_dir)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # Execute each incomplete retryable step in pipeline order, persisting
    # progress after each successful step.
    failed = False
    for step in retryable:
        typer.echo(f"  → Running {step} agent ...")
        try:
            opp = run_retry_step(
                step=step,
                opportunity=opp,
                evidence_items=evidence_items,
                opp_dir=opp_dir,
                run_dir=run_dir,
            )
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  ✗ {step} agent failed: {exc}", err=True)
            failed = True
            # Stop early — do not run downstream steps on a broken
            # opportunity state.
            break

        # Mark this step complete immediately so progress is durable.
        status = mark_steps_complete(status, [step])
        write_status(opp_dir, status)
        typer.echo(f"  ✓ {step} complete")

    if not failed:
        # Re-render the final report so it picks up the new outputs.
        typer.echo("  → Re-generating final-report.md ...")
        try:
            report_agent = ReportAgent()
            dest = report_agent.generate(run_dir)
            typer.echo(f"  ✓ Report written: {dest}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  ✗ Report generation failed: {exc}", err=True)
            failed = True

    if failed:
        completed_now = [
            s for s in retryable
            if s in status.get("steps", {})
            and status["steps"][s] == "complete"
            and s in incomplete
        ]
        typer.echo(
            f"Retry finished with failures — "
            f"{len(completed_now)} step(s) completed before error."
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"Retry complete — {len(retryable)} step(s) successfully executed."
    )


# -- sources ------------------------------------------------------------------


@sources_app.command(name="list")
def sources_list(
    run: str | None = typer.Option(
        None,
        "--run",
        help="Run directory whose source registry to list "
        "(default: packaged registry).",
    ),
) -> None:
    """List known sources and their trust scores."""
    _print_registry(_resolve_registry(run))


@sources_app.command(name="trust")
def sources_trust(
    domain: str = typer.Argument(..., help="Domain to set a trust score for."),
    score: int = typer.Option(
        ..., "--score", min=0, max=100, help="Trust score (0-100)."
    ),
    run: str | None = typer.Option(
        None,
        "--run",
        help="Run directory whose source registry to update.",
    ),
) -> None:
    """Set or override a domain's trust score in the registry.

    Requires ``--run`` to target a specific run's registry.  Without
    ``--run`` an error is shown.
    """
    if not run:
        typer.echo(
            "Error: --run is required.  Provide the run directory path to "
            "update its source registry (e.g. --run runs/<run-id>).",
            err=True,
        )
        raise typer.Exit(code=1)

    run_dir = Path(run)
    registry_path = run_dir / RUN_REGISTRY_RELATIVE_PATH
    try:
        registry = load_registry(registry_path if registry_path.is_file() else None)
    except FileNotFoundError:
        # No registry file yet — start from the packaged default.
        registry = load_registry()

    registry.domains[domain] = ScoreValue(score)
    save_path = registry.persist_to_run(run_dir)
    typer.echo(
        f"Trust score set: {domain} → {score} (saved to {save_path})"
    )


# -- helpers ------------------------------------------------------------------


def _print_registry(registry: SourceRegistry) -> None:
    """Print a human-readable summary of *registry*."""

    typer.echo("Source Trust Registry")
    typer.echo("=" * 40)
    typer.echo()
    typer.echo(
        f"Default unknown-domain score: {registry.default_unknown_domain_score}"
    )
    typer.echo()

    typer.echo("Source-type baseline scores")
    typer.echo("-" * 30)
    for stype in sorted(registry.source_type_defaults, key=lambda t: t.value):
        score = registry.source_type_defaults[stype]
        typer.echo(f"  {stype.value:<24s} {score:5.1f}")
    typer.echo()

    if registry.domains:
        typer.echo("Per-domain overrides")
        typer.echo("-" * 30)
        for domain in sorted(registry.domains):
            score = registry.domains[domain]
            typer.echo(f"  {domain:<32s} {score:5.1f}")
        typer.echo()
    else:
        typer.echo("Per-domain overrides: (none)")
        typer.echo()


def _resolve_registry(run_path: str | None) -> SourceRegistry:
    """Load the registry from *run_path* or the packaged default."""
    if run_path:
        registry_file = Path(run_path) / RUN_REGISTRY_RELATIVE_PATH
        if registry_file.is_file():
            return load_registry(registry_file)
        typer.echo(
            f"Warning: no registry found at {registry_file}; "
            "using packaged default.",
            err=True,
        )
    return load_registry()


if __name__ == "__main__":
    app()
