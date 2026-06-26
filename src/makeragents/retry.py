"""Retry and resume helpers: status tracking and disk-state reading."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

STATUS_YAML_FILENAME = "status.yaml"

# TODO: Keep in sync with agent graph in AGENTS.md/PRD.md §6
PIPELINE_STEPS: list[str] = [
    "research",
    "evidence",
    "opportunity",
    "maker",
    "taker",
    "mediator",
    "cost_checker",
]


def read_status(opportunity_dir: Path) -> dict[str, Any]:
    """Read ``status.yaml`` from *opportunity_dir*.

    Returns a default all-incomplete status dict when the file does not exist
    or when the YAML is unreadable / corrupt.
    """
    status_path = opportunity_dir / STATUS_YAML_FILENAME
    if not status_path.is_file():
        return _default_status(opportunity_dir)
    try:
        return yaml.safe_load(status_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        print(
            f"Warning: corrupt status.yaml in {opportunity_dir}, using defaults",
            file=sys.stderr,
        )
        return _default_status(opportunity_dir)


def _default_status(opportunity_dir: Path) -> dict[str, Any]:
    return {
        "opportunity_id": opportunity_dir.name,
        "steps": {step: "incomplete" for step in PIPELINE_STEPS},
    }


def write_status(opportunity_dir: Path, status: dict[str, Any]) -> None:
    """Write *status* as ``status.yaml`` inside *opportunity_dir*."""
    opportunity_dir.mkdir(parents=True, exist_ok=True)
    status_path = opportunity_dir / STATUS_YAML_FILENAME
    yaml_text = yaml.safe_dump(status, sort_keys=False, allow_unicode=True)
    tmp_path = status_path.with_suffix(".tmp")
    tmp_path.write_text(yaml_text, encoding="utf-8")
    tmp_path.rename(status_path)


def get_incomplete_steps(status: dict[str, Any]) -> list[str]:
    """Return pipeline step names whose recorded state is ``"incomplete"``."""
    steps: dict[str, Any] = status.get("steps", {})
    return [name for name, state in steps.items() if state == "incomplete"]


def read_opportunity_state(opportunity_dir: Path) -> dict[str, Any]:
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


def mark_steps_complete(status: dict[str, Any], steps: list[str]) -> dict[str, Any]:
    """Return a copy of *status* with every step in *steps* set to ``"complete"``."""
    updated = dict(status)
    updated_steps = dict(updated.get("steps", {}))
    for step in steps:
        if step not in PIPELINE_STEPS:
            print(
                f"Warning: unknown pipeline step '{step}'",
                file=sys.stderr,
            )
        if step in updated_steps:
            updated_steps[step] = "complete"
    updated["steps"] = updated_steps
    return updated
