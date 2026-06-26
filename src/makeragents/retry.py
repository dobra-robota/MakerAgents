"""Retry and resume helpers: status tracking and disk-state reading."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from makeragents.agents.cost_checker import CostCheckerAgent
from makeragents.agents.maker import MakerAgent
from makeragents.agents.mediator import MediatorAgent
from makeragents.agents.taker import TakerAgent
from makeragents.schemas import Confidence, EvidenceItem, Opportunity, ScoreSet

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


# Steps that can be re-run from on-disk state (pre-processing steps cannot).
_RETRYABLE_STEPS: frozenset[str] = frozenset(
    {"maker", "taker", "mediator", "cost_checker"}
)


def load_opportunity_for_retry(
    opp_dir: Path,
    run_dir: Path,
) -> tuple[Opportunity, list[EvidenceItem]]:
    """Load the opportunity and evidence items from disk for a retry run.

    Reads ``opportunity.yaml`` and ``evidence/evidence.json``, then
    populates any scores already persisted by completed agent steps
    (maker.json, taker.json) so downstream agents can pick them up.
    """
    opp_yaml = opp_dir / "opportunity.yaml"
    if not opp_yaml.is_file():
        raise FileNotFoundError(f"opportunity.yaml not found in {opp_dir}")

    opp_data = yaml.safe_load(opp_yaml.read_text(encoding="utf-8")) or {}
    opportunity = Opportunity.model_validate(opp_data)

    # Load existing maker scores (for use by taker / mediator / cost_checker
    # when maker is already complete from a prior run).
    maker_file = opp_dir / "maker.json"
    if maker_file.is_file():
        try:
            maker_data = json.loads(maker_file.read_text(encoding="utf-8"))
            opportunity = _apply_maker_scores(opportunity, maker_data)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # Load existing taker scores (for use by mediator / cost_checker).
    taker_file = opp_dir / "taker.json"
    if taker_file.is_file() and opportunity.scores is not None:
        try:
            taker_data = json.loads(taker_file.read_text(encoding="utf-8"))
            opportunity = _apply_taker_scores(opportunity, taker_data)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # Load evidence items.
    evidence_items: list[EvidenceItem] = []
    evidence_file = run_dir / "evidence" / "evidence.json"
    if evidence_file.is_file():
        try:
            evidence_data = json.loads(evidence_file.read_text(encoding="utf-8"))
            evidence_items = [
                EvidenceItem.model_validate(item) for item in evidence_data
            ]
        except (json.JSONDecodeError, OSError):
            pass

    return opportunity, evidence_items


def run_retry_step(
    *,
    step: str,
    opportunity: Opportunity,
    evidence_items: list[EvidenceItem],
    opp_dir: Path,
    run_dir: Path,
) -> Opportunity:
    """Execute a single pipeline *step* and return the (possibly updated) opportunity.

    The opportunity returned may have updated ``scores`` when *step* is
    ``"maker"`` or ``"taker"`` so that downstream agents can use them.
    """
    slug = opp_dir.name

    if step == "maker":
        return _run_maker_step(opportunity, evidence_items, run_dir)

    if step == "taker":
        return _run_taker_step(opportunity, evidence_items, slug, run_dir)

    if step == "mediator":
        _run_mediator_step(opportunity, run_dir)
        return opportunity

    if step == "cost_checker":
        _run_cost_checker_step(opportunity, run_dir)
        return opportunity

    raise ValueError(f"Unknown retry step: {step}")


# ------------------------------------------------------------------ helpers


def _apply_maker_scores(
    opportunity: Opportunity, maker_data: dict[str, Any]
) -> Opportunity:
    """Build a :class:`ScoreSet` from *maker_data* and attach it to *opportunity*."""
    scores = ScoreSet(
        validity_score=maker_data.get("validity_score", 0.0),
        maker_score=maker_data.get("maker_score", 0.0),
        maker_confidence=Confidence(maker_data.get("maker_confidence", "low")),
        taker_score=0.0,
        taker_confidence=Confidence.LOW,
        people_helped_score=maker_data.get("people_helped_score", 0.0),
        severity_score=maker_data.get("severity_score", 0.0),
        impact_score=maker_data.get("impact_score", 0.0),
        intervention_ease_score=maker_data.get("intervention_ease_score", 0.0),
        harm_risk_score=maker_data.get("harm_risk_score", 0.0),
        ability_to_act_score=maker_data.get("ability_to_act_score", 0.0),
        rank_score=maker_data.get("rank_score", 0.0),
    )
    return opportunity.model_copy(update={"scores": scores})


def _apply_taker_scores(
    opportunity: Opportunity, taker_data: dict[str, Any]
) -> Opportunity:
    """Update the opportunity's scores with taker-specific fields."""
    if opportunity.scores is None:
        return opportunity
    risk_breakdown: dict[str, Any] = taker_data.get("risk_breakdown", {})
    updated_scores = opportunity.scores.model_copy(
        update={
            "taker_score": taker_data.get("taker_score", 0.0),
            "taker_confidence": Confidence(
                taker_data.get("taker_confidence", "low")
            ),
            "harm_risk_score": risk_breakdown.get("harm_risk", 0.0),
        }
    )
    return opportunity.model_copy(update={"scores": updated_scores})


def _run_maker_step(
    opportunity: Opportunity,
    evidence_items: list[EvidenceItem],
    run_dir: Path,
) -> Opportunity:
    """Run the Maker Agent and return the opportunity with updated scores."""
    agent = MakerAgent()
    result = agent.run(opportunity, evidence_items)
    agent.save_output(result, run_dir)

    scores = ScoreSet(
        validity_score=result.validity_score,
        maker_score=result.maker_score,
        maker_confidence=result.maker_confidence,
        taker_score=0.0,
        taker_confidence=Confidence.LOW,
        people_helped_score=result.people_helped_score,
        severity_score=result.severity_score,
        impact_score=result.impact_score,
        intervention_ease_score=result.intervention_ease_score,
        harm_risk_score=result.harm_risk_score,
        ability_to_act_score=result.ability_to_act_score,
        rank_score=result.rank_score,
    )
    return opportunity.model_copy(update={"scores": scores})


def _run_taker_step(
    opportunity: Opportunity,
    evidence_items: list[EvidenceItem],
    slug: str,
    run_dir: Path,
) -> Opportunity:
    """Run the Taker Agent and return the opportunity with updated scores."""
    agent = TakerAgent()
    output, updated_opportunity = agent.analyze_and_update(
        opportunity, evidence_items
    )
    TakerAgent.save_output(output, slug, run_dir)
    return updated_opportunity


def _run_mediator_step(
    opportunity: Opportunity,
    run_dir: Path,
) -> None:
    """Run the Mediator Agent and persist its output."""
    agent = MediatorAgent()
    result = agent.run(opportunity)
    agent.save_output(result, run_dir)


def _run_cost_checker_step(
    opportunity: Opportunity,
    run_dir: Path,
) -> None:
    """Run the Cost Checker Agent and persist its output."""
    agent = CostCheckerAgent()
    estimate = agent.estimate(opportunity)
    agent.write_artifacts(estimate, run_dir)
