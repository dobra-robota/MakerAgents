"""CLI integration tests (no API calls)."""

import json
from pathlib import Path
from unittest import mock

import yaml

from makeragents.config import AppConfig
from makeragents.search.providers import ProviderResponse, SearchResult
from makeragents.sources.registry import (
    RUN_REGISTRY_RELATIVE_PATH,
    SourceRegistry,
    load_registry,
)
from tests.conftest import _invoke_in, app, runner


# -- run ----------------------------------------------------------------------

def test_run_help_shows_search_volume_options() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "--queries-per-run" in result.output
    assert "Number of search queries" in result.output
    assert "[default: 10]" in result.output
    assert "--results-per-query" in result.output
    assert "Number of search results" in result.output
    assert "[default: 5]" in result.output


def _mock_config() -> AppConfig:
    """Return a config with a fake API key for testing."""
    return AppConfig(deepseek_api_key="test-key")


def _mock_search_response(query: str, *, count: int = 10) -> ProviderResponse:
    """Return a deterministic, network-free provider response."""
    return ProviderResponse(
        provider="duckduckgo",
        results=[
            SearchResult(
                title=f"Result for {query}",
                url="https://example.test/result",
                snippet=f"Snippet for {query}",
            )
        ],
        raw={"query": query, "count": count, "items": [{"title": "raw"}]},
    )


def test_run_command_creates_run_folder_and_writes_search_results(
    tmp_path: Path,
) -> None:
    with (
        mock.patch(
            "makeragents.cli.load_config", return_value=_mock_config()
        ) as mock_load,
        mock.patch(
            "makeragents.search.client.SearchClient.search",
            side_effect=_mock_search_response,
        ) as mock_search,
    ):

        result = _invoke_in(
            tmp_path,
            "run",
            "--city",
            "Łodz",
            "--community",
            "senior citizens",
        )

    assert result.exit_code == 0, result.output

    # Verify load_config was called.
    mock_load.assert_called_once()

    # Verify ResearchAgent.search used the configured search client without
    # touching the network.
    assert mock_search.called

    # Verify run folder was created.
    runs_root = tmp_path / "runs"
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.endswith("-lodz-senior-citizens")
    registry_path = run_dir / RUN_REGISTRY_RELATIVE_PATH
    assert registry_path.is_file()
    assert load_registry(registry_path).model_dump() == load_registry().model_dump()


    report_text = (run_dir / "final-report.md").read_text(encoding="utf-8")
    assert "No research has been performed yet" in report_text
    assert not (run_dir / "evidence").exists()
    assert not (run_dir / "opportunities").exists()

    search_payload = json.loads(
        (run_dir / "sources" / "search-results.json").read_text(
            encoding="utf-8"
        )
    )
    first_result = search_payload["query_results"][0]
    assert first_result["query"] == "senior citizens problems in Łodz"
    assert first_result["provider"] == "duckduckgo"
    assert first_result["results"] == [
        {
            "title": "Result for senior citizens problems in Łodz",
            "url": "https://example.test/result",
            "snippet": "Snippet for senior citizens problems in Łodz",
        }
    ]
    assert first_result["raw"] == {
        "query": "senior citizens problems in Łodz",
        "count": 5,
        "items": [{"title": "raw"}],
    }
    parsed = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert parsed["city"] == "Łodz"
    assert parsed["community"] == "senior citizens"
    assert parsed["max_opportunities"] == 5
    assert parsed["queries_per_run"] == 10
    assert parsed["results_per_query"] == 5
    assert "timestamp" in parsed
    assert parsed["run_id"] == run_dir.name

    # Verify summary output.
    output = result.output
    assert "Run directory:" in output
    assert "Final report:" in output
    assert "Completed run for 'Łodz / senior citizens'" in output


def test_run_command_honors_search_volume_options(tmp_path: Path) -> None:
    with (
        mock.patch(
            "makeragents.cli.load_config", return_value=_mock_config()
        ),
        mock.patch(
            "makeragents.cli.PipelineRunner"
        ) as mock_runner_cls,
    ):
        mock_runner = mock.MagicMock()
        mock_runner.run.return_value = str(
            tmp_path / "runs" / "fake" / "final-report.md"
        )
        mock_runner_cls.return_value = mock_runner

        result = _invoke_in(
            tmp_path,
            "run",
            "--city",
            "Berlin",
            "--community",
            "cyclists",
            "--queries-per-run",
            "12",
            "--results-per-query",
            "7",
        )

    assert result.exit_code == 0, result.output
    run_dir = next((tmp_path / "runs").iterdir())
    parsed = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert parsed["queries_per_run"] == 12
    assert parsed["results_per_query"] == 7

    metadata = mock_runner.run.call_args.args[1]
    assert metadata.queries_per_run == 12
    assert metadata.results_per_query == 7


def test_run_command_honors_max_opportunities(tmp_path: Path) -> None:
    with (
        mock.patch(
            "makeragents.cli.load_config", return_value=_mock_config()
        ),
        mock.patch(
            "makeragents.search.client.SearchClient.search",
            side_effect=_mock_search_response,
        ),
    ):

        result = _invoke_in(
            tmp_path,
            "run",
            "--city",
            "Berlin",
            "--community",
            "cyclists",
            "--max-opportunities",
            "8",
        )

    assert result.exit_code == 0, result.output
    run_dir = next((tmp_path / "runs").iterdir())
    parsed = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert parsed["max_opportunities"] == 8


def test_run_command_rejects_invalid_max_opportunities(tmp_path: Path) -> None:
    result = _invoke_in(
        tmp_path,
        "run",
        "--city",
        "Berlin",
        "--community",
        "cyclists",
        "--max-opportunities",
        "0",
    )

    assert result.exit_code != 0


def test_run_command_rejects_invalid_queries_per_run(tmp_path: Path) -> None:
    result = _invoke_in(
        tmp_path,
        "run",
        "--city",
        "Berlin",
        "--community",
        "cyclists",
        "--queries-per-run",
        "0",
    )

    assert result.exit_code != 0


def test_run_command_rejects_invalid_results_per_query(tmp_path: Path) -> None:
    result = _invoke_in(
        tmp_path,
        "run",
        "--city",
        "Berlin",
        "--community",
        "cyclists",
        "--results-per-query",
        "0",
    )

    assert result.exit_code != 0


def test_run_command_rejects_missing_api_key(tmp_path: Path) -> None:
    """``maker run`` exits with an error when DEEPSEEK_API_KEY is not set."""
    with mock.patch(
        "makeragents.cli.load_config",
        return_value=AppConfig(deepseek_api_key=None),
    ):
        result = _invoke_in(
            tmp_path,
            "run",
            "--city",
            "Berlin",
            "--community",
            "cyclists",
        )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "DEEPSEEK_API_KEY" in result.output


# -- sources ------------------------------------------------------------------


def test_sources_list_prints_packaged_registry() -> None:
    """``maker sources list`` prints source types, scores, and the default."""
    result = runner.invoke(app, ["sources", "list"])
    assert result.exit_code == 0, result.output

    output = result.output
    assert "Source Trust Registry" in output
    assert "Default unknown-domain score" in output
    assert "government" in output
    assert "academic" in output
    assert "reddit" in output
    assert "anonymous_social" in output
    assert "Per-domain overrides: (none)" in output


def test_sources_list_run_registry(tmp_path: Path) -> None:
    """``maker sources list --run`` lists a run-specific registry."""
    run_dir = tmp_path / "runs" / "20250626-120000-lodz-senior-citizens"
    registry = SourceRegistry(
        domains={"example.com": 90, "untrusted.org": 15}
    )
    registry.persist_to_run(run_dir)

    result = runner.invoke(
        app, ["sources", "list", "--run", str(run_dir)]
    )
    assert result.exit_code == 0, result.output

    output = result.output
    assert "Per-domain overrides" in output
    assert "example.com" in output
    assert "90" in output
    assert "untrusted.org" in output
    assert "15" in output


def test_sources_list_falls_back_when_no_run_registry(tmp_path: Path) -> None:
    """``maker sources list --run`` falls back to packaged when no registry file."""
    run_dir = tmp_path / "runs" / "no-registry"
    run_dir.mkdir(parents=True)

    result = runner.invoke(
        app, ["sources", "list", "--run", str(run_dir)]
    )
    # Should succeed (uses packaged default) but emit a warning.
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output


def test_sources_trust_updates_run_registry(tmp_path: Path) -> None:
    """``maker sources trust`` persists a domain score into the run registry."""
    run_dir = tmp_path / "runs" / "myrun"
    run_dir.mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "sources", "trust", "example.com",
            "--score", "75",
            "--run", str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Trust score set: example.com" in result.output

    # Verify persistence on disk.
    reloaded = load_registry(run_dir / RUN_REGISTRY_RELATIVE_PATH)
    assert reloaded.domains["example.com"] == 75


def test_sources_trust_rejects_missing_run() -> None:
    """``maker sources trust`` without --run shows an error."""
    result = runner.invoke(
        app,
        ["sources", "trust", "example.com", "--score", "50"],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "--run" in result.output


def test_sources_trust_rejects_invalid_score(tmp_path: Path) -> None:
    """``maker sources trust`` rejects out-of-range scores."""
    run_dir = tmp_path / "runs" / "myrun"
    run_dir.mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "sources", "trust", "example.com",
            "--score", "150",
            "--run", str(run_dir),
        ],
    )
    assert result.exit_code != 0


# -- report -------------------------------------------------------------------


def test_report_regenerates_from_on_disk_state(tmp_path: Path) -> None:
    """``maker report`` re-renders final-report.md from existing run artifacts."""
    run_dir = tmp_path / "runs" / "testrun"

    # Create a run-like directory with run.yaml.
    registry = SourceRegistry(domains={"trusted.example": 85})
    registry.persist_to_run(run_dir)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "testrun",
                "city": "Testville",
                "community": "testers",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "max_opportunities": 3,
            }
        ),
        encoding="utf-8",
    )

    # Add an evidence item.
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "ev-001.json").write_text(
        json.dumps(
            {
                "id": "ev-001",
                "evidence_type": "claim",
                "source_domain": "trusted.example",
                "trust_score": 85,
                "snippet": "A test claim snippet.",
            }
        ),
        encoding="utf-8",
    )

    # Add an opportunity.
    opp_dir = run_dir / "opportunities" / "opp-001"
    opp_dir.mkdir(parents=True)
    (opp_dir / "opportunity.json").write_text(
        json.dumps(
            {
                "id": "opp-001",
                "title": "Test Opportunity",
                "type": "public_guide",
                "pain_summary": "Example pain.",
                "verdict": "MANUAL_POC",
                "scores": {"rank_score": 72.5},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "Report written:" in result.output

    # Verify the report was written and contains expected sections.
    report_path = run_dir / "final-report.md"
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "Testville" in report
    assert "testers" in report
    assert "Run ID" in report or "run_id" in report or "testrun" in report


def test_report_empty_run_writes_minimal_report(tmp_path: Path) -> None:
    """``maker report`` works on a minimal run directory with only run.yaml."""
    run_dir = tmp_path / "runs" / "emptyrun"
    run_dir.mkdir(parents=True)
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "emptyrun",
                "city": "Ghost Town",
                "community": "nobody",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "max_opportunities": 1,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["report", str(run_dir)])
    assert result.exit_code == 0, result.output

    report = (run_dir / "final-report.md").read_text(encoding="utf-8")
    assert "Ghost Town" in report
    assert "No valid opportunities found" in report
