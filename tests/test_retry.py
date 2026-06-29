"""Tests for retry.py status tracking and CLI retry command."""

import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from makeragents.retry import (
    PIPELINE_STEPS,
    STATUS_YAML_FILENAME,
    RetryPrerequisiteError,
    get_incomplete_steps,
    load_opportunity_for_retry,
    mark_steps_complete,
    read_opportunity_state,
    read_status,
    run_retry_step,
    write_status,
)
from tests.conftest import _invoke_in


# ------------------------------------------------------------------ read_status


class TestReadStatus:
    def test_reads_existing_status_yaml(self, tmp_path: Path) -> None:
        opp_dir = tmp_path / "test-opp"
        opp_dir.mkdir()
        (opp_dir / STATUS_YAML_FILENAME).write_text(
            yaml.safe_dump(
                {
                    "opportunity_id": "test-opp",
                    "steps": {"research": "complete", "evidence": "incomplete"},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        status = read_status(opp_dir)
        assert status["opportunity_id"] == "test-opp"
        assert status["steps"]["research"] == "complete"
        assert status["steps"]["evidence"] == "incomplete"

    def test_returns_default_when_missing(self, tmp_path: Path) -> None:
        opp_dir = tmp_path / "new-opp"
        opp_dir.mkdir()
        status = read_status(opp_dir)
        assert status["opportunity_id"] == "new-opp"
        assert all(v == "incomplete" for v in status["steps"].values())
        assert list(status["steps"].keys()) == PIPELINE_STEPS

    def test_returns_default_when_dir_missing(self, tmp_path: Path) -> None:
        status = read_status(tmp_path / "no-such-dir")
        assert status["opportunity_id"] == "no-such-dir"
        assert all(v == "incomplete" for v in status["steps"].values())

    def test_returns_default_on_corrupt_yaml(self, tmp_path: Path) -> None:
        opp_dir = tmp_path / "corrupt-opp"
        opp_dir.mkdir()
        (opp_dir / STATUS_YAML_FILENAME).write_text(
            "{{{ invalid: yaml: :::", encoding="utf-8"
        )
        status = read_status(opp_dir)
        assert status["opportunity_id"] == "corrupt-opp"
        assert all(v == "incomplete" for v in status["steps"].values())


# ----------------------------------------------------------------- write_status


class TestWriteStatus:
    def test_writes_status_yaml(self, tmp_path: Path) -> None:
        opp_dir = tmp_path / "test-opp"
        opp_dir.mkdir()
        status = {
            "opportunity_id": "test-opp",
            "steps": {"research": "complete"},
        }
        write_status(opp_dir, status)
        written = yaml.safe_load(
            (opp_dir / STATUS_YAML_FILENAME).read_text(encoding="utf-8")
        )
        assert written == status

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        opp_dir = tmp_path / "auto-created"
        status = {"opportunity_id": "auto-created", "steps": {}}
        write_status(opp_dir, status)
        assert opp_dir.is_dir()
        assert (opp_dir / STATUS_YAML_FILENAME).is_file()


# --------------------------------------------------------- get_incomplete_steps


class TestGetIncompleteSteps:
    def test_returns_only_incomplete(self) -> None:
        status = {
            "steps": {
                "research": "complete",
                "evidence": "incomplete",
                "opportunity": "complete",
                "maker": "incomplete",
            }
        }
        incomplete = get_incomplete_steps(status)
        assert incomplete == ["evidence", "maker"]

    def test_returns_empty_when_all_complete(self) -> None:
        status = {"steps": {"research": "complete", "evidence": "complete"}}
        assert get_incomplete_steps(status) == []

    def test_handles_missing_steps_key(self) -> None:
        assert get_incomplete_steps({}) == []
        assert get_incomplete_steps({"opportunity_id": "x"}) == []


# --------------------------------------------------------- read_opportunity_state


class TestReadOpportunityState:
    def test_returns_summary_of_existing_artifacts(self, tmp_path: Path) -> None:
        opp_dir = tmp_path / "test-opp"
        opp_dir.mkdir()
        (opp_dir / "search-results.json").write_text("{}")
        (opp_dir / "evidence.json").write_text("[]")
        (opp_dir / "status.yaml").write_text("steps: {}")
        state = read_opportunity_state(opp_dir)
        assert state["slug"] == "test-opp"
        assert "evidence.json" in state["artifacts"]
        assert "search-results.json" in state["artifacts"]
        assert "status.yaml" in state["artifacts"]

    def test_returns_empty_artifacts_when_dir_missing(self, tmp_path: Path) -> None:
        state = read_opportunity_state(tmp_path / "no-such-dir")
        assert state["slug"] == "no-such-dir"
        assert state["artifacts"] == []


# ---------------------------------------------------------- mark_steps_complete


class TestMarkStepsComplete:
    def test_marks_specified_steps_complete(self) -> None:
        status = {
            "opportunity_id": "opp-1",
            "steps": {
                "research": "incomplete",
                "evidence": "incomplete",
                "opportunity": "incomplete",
            },
        }
        updated = mark_steps_complete(status, ["research", "opportunity"])
        assert updated["opportunity_id"] == "opp-1"
        assert updated["steps"]["research"] == "complete"
        assert updated["steps"]["evidence"] == "incomplete"
        assert updated["steps"]["opportunity"] == "complete"

    def test_does_not_mutate_original(self) -> None:
        status = {"steps": {"research": "incomplete"}}
        mark_steps_complete(status, ["research"])
        assert status["steps"]["research"] == "incomplete"

    def test_ignores_unknown_steps(self) -> None:
        status = {"steps": {"research": "incomplete"}}
        updated = mark_steps_complete(status, ["research", "not-a-step"])
        assert updated["steps"]["research"] == "complete"


# ---------------------------------------------------------------- CLI commands


# Shared fixture helpers -------------------------------------------------------


def _make_valid_evidence_item(
    eid: str = "ev-001", domain: str = "example.com"
) -> dict:
    """Return a dict suitable for EvidenceItem.model_validate."""
    return {
        "id": eid,
        "source_url": f"https://{domain}/article/{eid}",
        "source_domain": domain,
        "source_type": "local_news",
        "evidence_type": "news_report",
        "snippet": f"Snippet for {eid} about community issues.",
        "language": "en",
        "claim_classification": "evidence_based",
        "trust_score": 75.0,
        "recency": "2025-01-01",
        "confidence": "high",
    }


def _make_valid_opportunity_dict(slug: str) -> dict:
    """Return a dict suitable for Opportunity.model_validate."""
    return {
        "id": slug,
        "title": f"Test Opportunity: {slug}",
        "type": "public_guide",
        "pain_summary": "Example pain point for testing retry.",
        "who_benefits": ["senior citizens"],
        "vulnerable_groups": ["elderly"],
        "evidence_ids": ["ev-001"],
        "speculative": False,
        "scores": None,
        "verdict": None,
    }


def _setup_retry_run(
    tmp_path: Path,
    *,
    slug: str,
    steps: dict | None = None,
    with_opportunity: bool = True,
    with_evidence: bool = True,
) -> Path:
    """Create a run directory with all artifacts needed for retry.

    Returns the run directory path.
    """
    run_dir = tmp_path / "runs" / "dummy-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {"run_id": "dummy-run", "city": "Testville", "community": "testers"},
        ),
        encoding="utf-8",
    )

    opp_dir = run_dir / "opportunities" / slug
    opp_dir.mkdir(parents=True)

    if with_opportunity:
        (opp_dir / "opportunity.yaml").write_text(
            yaml.safe_dump(
                _make_valid_opportunity_dict(slug),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    if with_evidence:
        evidence_dir = run_dir / "evidence"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "evidence.json").write_text(
            json.dumps([_make_valid_evidence_item()]),
            encoding="utf-8",
        )

    if steps is None:
        steps = {s: "incomplete" for s in PIPELINE_STEPS}
    write_status(opp_dir, {"opportunity_id": slug, "steps": steps})

    return run_dir


# ---------------------------------------------------------- load_opportunity_for_retry


class TestLoadOpportunityForRetry:
    def test_loads_opportunity_and_evidence(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(tmp_path, slug="test-opp")
        opp_dir = run_dir / "opportunities" / "test-opp"
        opportunity, evidence = load_opportunity_for_retry(opp_dir, run_dir)
        assert opportunity.id == "test-opp"
        assert opportunity.title == "Test Opportunity: test-opp"
        assert opportunity.type.value == "public_guide"
        assert len(evidence) == 1
        assert evidence[0].id == "ev-001"

    def test_raises_when_opportunity_yaml_missing(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(
            tmp_path, slug="test-opp", with_opportunity=False
        )
        opp_dir = run_dir / "opportunities" / "test-opp"
        with pytest.raises(FileNotFoundError):
            load_opportunity_for_retry(opp_dir, run_dir)

    def test_loads_existing_maker_scores(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(tmp_path, slug="test-opp")
        opp_dir = run_dir / "opportunities" / "test-opp"
        # Pre-write maker.json with known scores
        (opp_dir / "maker.json").write_text(
            json.dumps(
                {
                    "opportunity_id": "test-opp",
                    "maker_score": 72.5,
                    "maker_confidence": "medium",
                    "people_helped_score": 50.0,
                    "severity_score": 60.0,
                    "impact_score": 55.0,
                    "validity_score": 75.0,
                    "intervention_ease_score": 85.0,
                    "harm_risk_score": 15.0,
                    "ability_to_act_score": 30.0,
                    "rank_score": 62.0,
                }
            ),
            encoding="utf-8",
        )
        opportunity, _evidence = load_opportunity_for_retry(opp_dir, run_dir)
        assert opportunity.scores is not None
        assert opportunity.scores.maker_score == 72.5

    def test_handles_missing_evidence(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(
            tmp_path, slug="test-opp", with_evidence=False
        )
        opp_dir = run_dir / "opportunities" / "test-opp"
        _opportunity, evidence = load_opportunity_for_retry(opp_dir, run_dir)
        assert evidence == []


# --------------------------------------------------------------- run_retry_step


class TestRunRetryStep:
    def test_run_maker_step_writes_output(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(tmp_path, slug="test-opp")
        opp_dir = run_dir / "opportunities" / "test-opp"
        opportunity, evidence = load_opportunity_for_retry(opp_dir, run_dir)

        updated = run_retry_step(
            step="maker",
            opportunity=opportunity,
            evidence_items=evidence,
            opp_dir=opp_dir,
            run_dir=run_dir,
        )
        assert updated.scores is not None
        assert (opp_dir / "maker.json").is_file()
        assert (opp_dir / "maker.md").is_file()

    def test_run_taker_step_requires_maker_artifact(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(tmp_path, slug="test-opp")
        opp_dir = run_dir / "opportunities" / "test-opp"
        opportunity, evidence = load_opportunity_for_retry(opp_dir, run_dir)

        with pytest.raises(RetryPrerequisiteError, match="maker.json"):
            run_retry_step(
                step="taker",
                opportunity=opportunity,
                evidence_items=evidence,
                opp_dir=opp_dir,
                run_dir=run_dir,
            )

    def test_run_full_pipeline(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(tmp_path, slug="test-opp")
        opp_dir = run_dir / "opportunities" / "test-opp"
        opportunity, evidence = load_opportunity_for_retry(opp_dir, run_dir)

        # Run maker → taker → mediator → cost_checker in sequence
        opportunity = run_retry_step(
            step="maker",
            opportunity=opportunity,
            evidence_items=evidence,
            opp_dir=opp_dir,
            run_dir=run_dir,
        )
        assert (opp_dir / "maker.json").is_file()

        opportunity = run_retry_step(
            step="taker",
            opportunity=opportunity,
            evidence_items=evidence,
            opp_dir=opp_dir,
            run_dir=run_dir,
        )
        assert (opp_dir / "taker.json").is_file()

        opportunity = run_retry_step(
            step="mediator",
            opportunity=opportunity,
            evidence_items=evidence,
            opp_dir=opp_dir,
            run_dir=run_dir,
        )
        assert (opp_dir / "mediator.json").is_file()

        opportunity = run_retry_step(
            step="cost_checker",
            opportunity=opportunity,
            evidence_items=evidence,
            opp_dir=opp_dir,
            run_dir=run_dir,
        )
        assert (opp_dir / "cost.json").is_file()

    def test_raises_for_unknown_step(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(tmp_path, slug="test-opp")
        opp_dir = run_dir / "opportunities" / "test-opp"
        opportunity, evidence = load_opportunity_for_retry(opp_dir, run_dir)

        with pytest.raises(ValueError, match="Unknown retry step"):
            run_retry_step(
                step="research",
                opportunity=opportunity,
                evidence_items=evidence,
                opp_dir=opp_dir,
                run_dir=run_dir,
            )


# ---------------------------------------------------------------- CLI commands


class TestRetryCLI:
    def test_retry_runs_only_missing_steps(self, tmp_path: Path) -> None:
        """Partial completion: only incomplete retryable steps run."""
        run_dir = _setup_retry_run(
            tmp_path,
            slug="test-opp",
            steps={
                "research": "complete",
                "evidence": "complete",
                "opportunity": "complete",
                "maker": "incomplete",
                "taker": "incomplete",
                "mediator": "incomplete",
                "cost_checker": "complete",
            },
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "test-opp",
        )
        assert result.exit_code == 0, result.output
        assert "Retrying opportunity: test-opp" in result.output
        assert "maker" in result.output
        assert "taker" in result.output
        assert "mediator" in result.output
        assert "Retry complete" in result.output

        # Verify steps were marked complete
        opp_dir = run_dir / "opportunities" / "test-opp"
        status = read_status(opp_dir)
        assert status["steps"]["maker"] == "complete"
        assert status["steps"]["taker"] == "complete"
        assert status["steps"]["mediator"] == "complete"
        # Already-complete steps untouched
        assert status["steps"]["cost_checker"] == "complete"

        # Output files should exist
        assert (opp_dir / "maker.json").is_file()
        assert (opp_dir / "taker.json").is_file()
        assert (opp_dir / "mediator.json").is_file()


    def test_retry_does_not_call_research_agent(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(
            tmp_path,
            slug="test-opp",
            steps={
                "research": "complete",
                "evidence": "complete",
                "opportunity": "complete",
                "maker": "incomplete",
                "taker": "complete",
                "mediator": "complete",
                "cost_checker": "complete",
            },
        )
        with mock.patch(
            "makeragents.agents.research.ResearchAgent.search",
            side_effect=AssertionError("ResearchAgent must not run"),
        ):
            result = _invoke_in(
                tmp_path,
                "retry",
                str(run_dir.relative_to(tmp_path)),
                "--opportunity",
                "test-opp",
            )
        assert result.exit_code == 0, result.output
        status = read_status(run_dir / "opportunities" / "test-opp")
        assert status["steps"]["maker"] == "complete"

    def test_retry_errors_on_missing_evidence_artifact(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(
            tmp_path,
            slug="test-opp",
            steps={
                "research": "complete",
                "evidence": "complete",
                "opportunity": "complete",
                "maker": "incomplete",
                "taker": "complete",
                "mediator": "complete",
                "cost_checker": "complete",
            },
            with_evidence=False,
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "test-opp",
        )
        assert result.exit_code == 1
        assert "Cannot retry maker" in result.output
        assert "evidence/evidence.json" in result.output

    def test_retry_errors_on_missing_mediator_artifact(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(
            tmp_path,
            slug="test-opp",
            steps={
                "research": "complete",
                "evidence": "complete",
                "opportunity": "complete",
                "maker": "complete",
                "taker": "complete",
                "mediator": "complete",
                "cost_checker": "incomplete",
            },
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "test-opp",
        )
        assert result.exit_code == 1
        assert "Cannot retry cost_checker" in result.output
        assert "mediator.json" in result.output

    def test_retry_skips_when_all_complete(self, tmp_path: Path) -> None:
        run_dir = _setup_retry_run(
            tmp_path,
            slug="all-done",
            steps={s: "complete" for s in PIPELINE_STEPS},
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "all-done",
        )
        assert result.exit_code == 0, result.output
        assert "already complete" in result.output

    def test_retry_skips_preprocessing_steps(self, tmp_path: Path) -> None:
        """Research/evidence/opportunity are skipped even when incomplete."""
        run_dir = _setup_retry_run(
            tmp_path,
            slug="test-opp",
            steps={
                "research": "incomplete",
                "evidence": "incomplete",
                "opportunity": "incomplete",
                "maker": "complete",
                "taker": "complete",
                "mediator": "complete",
                "cost_checker": "complete",
            },
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "test-opp",
        )
        assert result.exit_code == 0, result.output
        assert "Skipping (pre-processing, cannot be re-run)" in result.output
        assert "research" in result.output
        assert "evidence" in result.output
        assert "opportunity" in result.output
        assert "cannot be re-run from on-disk state" in result.output

    def test_retry_regenerates_report_after_success(self, tmp_path: Path) -> None:
        """After successful retry, final-report.md is regenerated."""
        run_dir = _setup_retry_run(
            tmp_path,
            slug="test-opp",
            steps={
                "research": "complete",
                "evidence": "complete",
                "opportunity": "complete",
                "maker": "incomplete",
                "taker": "incomplete",
                "mediator": "incomplete",
                "cost_checker": "incomplete",
            },
        )
        # Remove stub report to prove regeneration creates a new one.
        report_path = run_dir / "final-report.md"
        report_path.unlink(missing_ok=True)

        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "test-opp",
        )
        assert result.exit_code == 0, result.output
        assert "Report written:" in result.output
        assert report_path.is_file()
        content = report_path.read_text(encoding="utf-8")
        # Should contain run metadata and the opportunity title
        assert "Testville" in content
        assert "testers" in content
        assert "Test Opportunity" in content

    def test_retry_handles_missing_opportunity_yaml(self, tmp_path: Path) -> None:
        """Retry without opportunity.yaml exits with error."""
        run_dir = _setup_retry_run(
            tmp_path, slug="test-opp", with_opportunity=False
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "test-opp",
        )
        assert result.exit_code == 1
        assert "opportunity.yaml not found" in result.output

    def test_retry_handles_missing_run_folder(self, tmp_path: Path) -> None:
        result = _invoke_in(
            tmp_path,
            "retry",
            "runs/no-such-run",
            "--opportunity",
            "anything",
        )
        assert result.exit_code == 1
        assert "run folder not found" in result.output

    def test_retry_handles_missing_run_yaml(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "no-yaml"
        run_dir.mkdir(parents=True)
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "anything",
        )
        assert result.exit_code == 1
        assert "run.yaml not found" in result.output

    def test_retry_handles_missing_opportunity(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "dummy-run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.yaml").write_text(
            yaml.safe_dump({"run_id": "dummy-run"}),
            encoding="utf-8",
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "no-such-slug",
        )
        assert result.exit_code == 1
        assert "opportunity folder not found" in result.output

    def test_retry_exits_on_agent_failure(self, tmp_path: Path) -> None:
        """When an agent step raises, retry stops and exits with error."""
        run_dir = _setup_retry_run(
            tmp_path,
            slug="test-opp",
            steps={
                "research": "complete",
                "evidence": "complete",
                "opportunity": "complete",
                "maker": "incomplete",
                "taker": "incomplete",
                "mediator": "complete",
                "cost_checker": "complete",
            },
        )

        # Simulate a failure during step execution.
        with mock.patch(
            "makeragents.cli.run_retry_step",
            side_effect=RuntimeError("simulated step failure"),
        ):
            result = _invoke_in(
                tmp_path,
                "retry",
                str(run_dir.relative_to(tmp_path)),
                "--opportunity",
                "test-opp",
            )
        # CliRunner captures SystemExit; exit_code reflects the exit code.
        assert result.exit_code == 1
        assert "simulated step failure" in result.output
