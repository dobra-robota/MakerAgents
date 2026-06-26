"""Tests for retry.py status tracking and CLI retry command."""

from pathlib import Path

import yaml

from makeragents.retry import (
    PIPELINE_STEPS,
    STATUS_YAML_FILENAME,
    get_incomplete_steps,
    mark_steps_complete,
    read_opportunity_state,
    read_status,
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


class TestRetryCLI:
    def _make_run_with_opportunity(
        self, tmp_path: Path, *, slug: str, steps: dict | None = None
    ) -> Path:
        """Create a minimal run folder with an opportunity and its status.yaml."""
        run_dir = tmp_path / "runs" / "dummy-run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.yaml").write_text(
            yaml.safe_dump({"run_id": "dummy-run", "city": "X", "community": "Y"}),
            encoding="utf-8",
        )
        opp_dir = run_dir / "opportunities" / slug
        opp_dir.mkdir(parents=True)
        if steps is None:
            steps = {s: "incomplete" for s in PIPELINE_STEPS}
        write_status(opp_dir, {"opportunity_id": slug, "steps": steps})
        return run_dir

    def test_retry_command_reports_incomplete_steps(self, tmp_path: Path) -> None:
        run_dir = self._make_run_with_opportunity(
            tmp_path, slug="senior-services-guide"
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "senior-services-guide",
        )
        assert result.exit_code == 0, result.output
        assert "Retrying opportunity: senior-services-guide" in result.output
        assert "Steps to retry:" in result.output
        assert "Retry complete" in result.output

    def test_retry_command_updates_status_after_success(self, tmp_path: Path) -> None:
        run_dir = self._make_run_with_opportunity(
            tmp_path,
            slug="senior-services-guide",
            steps={
                "research": "complete",
                "evidence": "complete",
                "opportunity": "incomplete",
                "maker": "incomplete",
                "taker": "incomplete",
                "mediator": "incomplete",
                "cost_checker": "incomplete",
            },
        )
        result = _invoke_in(
            tmp_path,
            "retry",
            str(run_dir.relative_to(tmp_path)),
            "--opportunity",
            "senior-services-guide",
        )
        assert result.exit_code == 0, result.output
        # After retry, all steps should be complete
        opp_dir = run_dir / "opportunities" / "senior-services-guide"
        status = read_status(opp_dir)
        incomplete = get_incomplete_steps(status)
        assert incomplete == [], f"Expected no incomplete steps, got: {incomplete}"

    def test_retry_skips_when_all_complete(self, tmp_path: Path) -> None:
        run_dir = self._make_run_with_opportunity(
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
