"""Retry and resume helpers: status tracking and disk-state reading."""

from __future__ import annotations

from pathlib import Path

import yaml

STATUS_YAML_FILENAME = "status.yaml"
RUN_YAML_FILENAME = "run.yaml"

PIPELINE_STEPS: list[str] = [
    "research",
    "evidence",
    "opportunity",
    "maker",
    "taker",
    "mediator",
    "cost_checker",
]


def read_status(opportunity_dir: Path) -> dict:
    """Read ``status.yaml`` from *opportunity_dir*.

    Returns a default all-incomplete status dict when the file does not exist.
    """
    status_path = opportunity_dir / STATUS_YAML_FILENAME
    if not status_path.is_file():
        return {
            "opportunity_id": opportunity_dir.name,
            "steps": {step: "incomplete" for step in PIPELINE_STEPS},
        }
    return yaml.safe_load(status_path.read_text(encoding="utf-8"))


def write_status(opportunity_dir: Path, status: dict) -> None:
    """Write *status* as ``status.yaml`` inside *opportunity_dir*."""
    opportunity_dir.mkdir(parents=True, exist_ok=True)
    status_path = opportunity_dir / STATUS_YAML_FILENAME
    status_path.write_text(
        yaml.safe_dump(status, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def get_incomplete_steps(status: dict) -> list[str]:
    """Return pipeline step names whose recorded state is ``"incomplete"``."""
    steps: dict = status.get("steps", {})
    return [name for name, state in steps.items() if state == "incomplete"]


def read_opportunity_state(opportunity_dir: Path) -> dict:
    """Return a summary of on-disk artifacts for *opportunity_dir*.

    The returned dict includes the opportunity slug and a list of file names
    present in the directory.  Does not parse file contents.
    """
    slug = opportunity_dir.name
    artifacts: list[str] = []
    if opportunity_dir.is_dir():
        artifacts = sorted(
            p.name for p in opportunity_dir.iterdir() if p.is_file()
        )
    return {"slug": slug, "artifacts": artifacts}


def mark_steps_complete(status: dict, steps: list[str]) -> dict:
    """Return a copy of *status* with every step in *steps* set to ``"complete"``."""
    updated = dict(status)
    updated_steps = dict(updated.get("steps", {}))
    for step in steps:
        if step in updated_steps:
            updated_steps[step] = "complete"
    updated["steps"] = updated_steps
    return updated
