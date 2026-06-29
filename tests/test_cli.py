"""CLI integration tests (no API calls)."""

import json
import re
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
from makeragents.search.providers import ProviderResponse, SearchResult
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


def test_run_command_creates_run_folder_and_runs_pipeline(
    tmp_path: Path,
) -> None:
    with (
        mock.patch(
            "makeragents.cli.load_config", return_value=_mock_config()
        ),
        mock.patch(
            "makeragents.cli.PipelineRunner",
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
            "Łodz",
            "--community",
            "senior citizens",
        )

    assert result.exit_code == 0, result.output
    assert mock_runner.run.called

    runs_root = tmp_path / "runs"
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.endswith("-lodz-senior-citizens")
    registry_path = run_dir / RUN_REGISTRY_RELATIVE_PATH
    assert registry_path.is_file()
    assert load_registry(registry_path).model_dump() == load_registry().model_dump()

    metadata_arg = mock_runner.run.call_args.args[1]
    assert metadata_arg.city == "Łodz"
    assert metadata_arg.community == "senior citizens"
    assert metadata_arg.max_opportunities == 5

    parsed = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert parsed["city"] == "Łodz"
    assert parsed["run_id"] == run_dir.name

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


def test_run_command_writes_full_artifact_tree_with_mocked_search(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """``maker run`` writes the expected local artifact tree without network."""

    class FakeSearchClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def search(self, query: str, *, count: int = 10) -> ProviderResponse:
            results = [
                SearchResult(
                    title="Łodz senior transport support audit",
                    url="https://lodz.example.gov/senior-transport",
                    snippet=(
                        "2026 city audit: senior citizens in Łodz report "
                        "long waits and inaccessible transport to appointments."
                    ),
                ),
                SearchResult(
                    title="Families describe transport gaps",
                    url="https://forum.lodz.example/senior-mobility",
                    snippet=(
                        "Forum complaint from families says senior citizens "
                        "need clearer transport help and appointment guidance."
                    ),
                ),
            ][:count]
            return ProviderResponse(
                provider="mock",
                results=results,
                raw={"query": query, "result_count": len(results)},
            )

    class FakeLLMClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def chat_json(self, messages, **kwargs):
            prompt = "\n".join(message.content for message in messages)
            evidence_ids = re.findall(r"(?:\*\*|\[)(EVID-[^\]*]+)(?:\*\*|\])", prompt)

            if "Research Agent" in prompt:
                return {
                    "queries": [
                        {
                            "query": (
                                "Łodz senior citizens accessible transport help"
                            ),
                            "language": "en",
                        }
                    ]
                }

            if "Evidence Agent" in prompt:
                return {
                    "items": [
                        {
                            "snippet_index": 0,
                            "evidence_type": "official_statement",
                            "language": "en",
                            "confidence": "high",
                            "recency": "2026",
                            "claim_classification": "evidence_based",
                        },
                        {
                            "snippet_index": 1,
                            "evidence_type": "complaint",
                            "language": "en",
                            "confidence": "medium",
                            "recency": "recent",
                            "claim_classification": "inference",
                        },
                    ]
                }

            if "Opportunity Agent" in prompt:
                return {
                    "opportunities": [
                        {
                            "id": "accessible-senior-transport-guide",
                            "title": "Accessible senior transport guide",
                            "type": "public_guide",
                            "pain_summary": (
                                "Senior citizens in Łodz need clearer, "
                                "accessible transport guidance for appointments."
                            ),
                            "who_benefits": [
                                "senior citizens",
                                "families and carers",
                            ],
                            "vulnerable_groups": [
                                "mobility-limited senior citizens"
                            ],
                            "evidence_ids": evidence_ids,
                            "speculative": False,
                        }
                    ]
                }

            if "Maker Agent" in prompt:
                return {
                    "value_add_summary": (
                        "A concise guide would help senior citizens and carers "
                        "find vetted transport options with clear constraints."
                    ),
                    "score": 82,
                    "confidence": "high",
                    "evidence_ids": evidence_ids,
                    "claims": [
                        {
                            "text": "The city audit supports the access need.",
                            "classification": "evidence_based",
                            "evidence_id": evidence_ids[0],
                        }
                    ],
                }

            if "Taker Agent" in prompt:
                return {
                    "risk_summary": (
                        "Main risks are stale advice and gatekeeping; keep the "
                        "guide source-cited and human-reviewed."
                    ),
                    "score": 18,
                    "confidence": "medium",
                    "evidence_ids": evidence_ids,
                    "claims": [
                        {
                            "text": "Guidance must avoid false authority.",
                            "classification": "inference",
                            "evidence_id": evidence_ids[0],
                        }
                    ],
                }

            if "Mediator Agent" in prompt:
                return {
                    "comparison": (
                        "The value-add case is stronger than the bounded risks."
                    ),
                    "verdict": "MANUAL_POC",
                    "do_no_harm": {
                        "vulnerable_groups": "Mobility-limited senior citizens",
                        "negative_side_effects": "Outdated guidance",
                        "abuse_risks": "Low if sources stay cited",
                        "legal_concerns": "None identified",
                        "misinformation_risks": "Mitigate with update dates",
                        "dependency_risks": "Keep alternatives visible",
                        "false_authority_risks": "Label as informational",
                        "safeguards": "Review before publication",
                    },
                    "safe_intervention_shape": (
                        "Run a manual POC guide with citations and review dates."
                    ),
                    "evidence_too_weak": False,
                }

            if "Cost Checker Agent" in prompt:
                return {
                    "poc_type": "public_guide",
                    "cost_range": "$0–$50",
                    "time_est": "1 weekend",
                    "risk_level": "low",
                    "first_actions": [
                        "Collect current transport contacts.",
                        "Draft a cited one-page guide.",
                        "Review with two local carers.",
                    ],
                }

            raise AssertionError(f"Unhandled prompt:\n{prompt}")

    def run_with_registry(self, run_dir: Path, metadata):
        SourceRegistry().persist_to_run(run_dir)
        return original_run(self, run_dir, metadata)

    def save_maker(self, result, opp_dir: Path):
        opp_dir = Path(opp_dir)
        json_path = opp_dir / "maker.json"
        md_path = opp_dir / "maker.md"
        json_path.write_text(
            json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(self._to_markdown(result), encoding="utf-8")
        return json_path, md_path

    def save_taker(output, opportunity_slug: str, run_dir: Path):
        opp_dir = Path(run_dir) / "opportunities" / opportunity_slug
        opp_dir.mkdir(parents=True, exist_ok=True)
        json_path = opp_dir / "taker.json"
        md_path = opp_dir / "taker.md"
        json_path.write_text(
            json.dumps(output.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(output.to_markdown(), encoding="utf-8")
        return json_path, md_path

    def save_mediator(self, result, opp_dir: Path):
        opp_dir = Path(opp_dir)
        json_path = opp_dir / "mediator.json"
        md_path = opp_dir / "mediator.md"
        json_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(self._to_markdown(result), encoding="utf-8")
        return json_path, md_path

    def save_cost(self, estimate, opp_dir: Path):
        opp_dir = Path(opp_dir)
        json_path = opp_dir / "cost.json"
        md_path = opp_dir / "cost.md"
        json_path.write_text(
            json.dumps(estimate.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        md_path.write_text(self._to_markdown(estimate), encoding="utf-8")
        return json_path, md_path

    from makeragents.agents.cost_checker import CostCheckerAgent
    from makeragents.agents.maker import MakerAgent
    from makeragents.agents.mediator import MediatorAgent
    from makeragents.agents.taker import TakerAgent
    from makeragents.orchestrator import PipelineRunner

    original_run = PipelineRunner.run
    monkeypatch.setattr("makeragents.cli.load_config", _mock_config)
    monkeypatch.setattr("makeragents.orchestrator.LLMClient", FakeLLMClient)
    monkeypatch.setattr("makeragents.agents.research.SearchClient", FakeSearchClient)
    monkeypatch.setattr(PipelineRunner, "run", run_with_registry)
    monkeypatch.setattr(MakerAgent, "save_output", save_maker)
    monkeypatch.setattr(TakerAgent, "save_output", save_taker)
    monkeypatch.setattr(MediatorAgent, "save_output", save_mediator)
    monkeypatch.setattr(CostCheckerAgent, "save_output", save_cost, raising=False)

    result = _invoke_in(
        tmp_path,
        "run",
        "--city",
        "Łodz",
        "--community",
        "senior citizens",
        "--max-opportunities",
        "1",
    )

    assert result.exit_code == 0, result.output
    run_dir = next((tmp_path / "runs").iterdir())

    assert (run_dir / "run.yaml").is_file()
    assert (run_dir / "sources" / "search-results.json").is_file()
    assert (run_dir / "sources" / "source-registry.yaml").is_file()
    assert (run_dir / "evidence" / "evidence.json").is_file()
    assert (run_dir / "appendix" / "rejected-opportunities.md").is_file()
    assert (run_dir / "appendix" / "incomplete-opportunities.md").is_file()
    assert (run_dir / "final-report.md").is_file()

    opportunity_dirs = [
        path
        for path in (run_dir / "opportunities").iterdir()
        if path.is_dir() and (path / "opportunity.yaml").is_file()
    ]
    assert opportunity_dirs
    complete_dirs = [
        path
        for path in opportunity_dirs
        if all(
            (path / name).is_file()
            for name in (
                "maker.json",
                "maker.md",
                "taker.json",
                "taker.md",
                "mediator.json",
                "mediator.md",
                "cost.json",
                "cost.md",
            )
        )
    ]
    assert complete_dirs

    report = (run_dir / "final-report.md").read_text(encoding="utf-8")
    assert "Łodz" in report
    assert "senior citizens" in report
    assert "Accessible senior transport guide" in report
    assert "Evidence References" in report
    assert "EVID-" in report
    assert "POC Cost Estimate" in report
    assert "$0–$50" in report


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
