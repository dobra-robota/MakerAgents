"""Tests for the Report Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from makeragents.agents.report import (
    LoadedOpportunity,
    ReportAgent,
    _missing_steps,
    _reject_reason,
)
from makeragents.cli import app
from makeragents.schemas import Verdict


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _make_run_dir(tmp_path: Path, run_id: str = "20250101-120000-lodz-test") -> Path:
    """Create a minimal run folder with run.yaml."""
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    _write_yaml(
        run_dir / "run.yaml",
        {
            "run_id": run_id,
            "city": "Lodz",
            "community": "test community",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "max_opportunities": 5,
        },
    )
    return run_dir


def _make_opportunity_yaml(
    opp_dir: Path,
    opp_id: str,
    title: str = "Test Opportunity",
    verdict: str | None = None,
    scores: dict | None = None,
    speculative: bool = False,
    evidence_ids: list[str] | None = None,
) -> None:
    """Write an opportunity.yaml file."""
    opp_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "id": opp_id,
        "title": title,
        "type": "public_guide",
        "pain_summary": f"Pain summary for {title}",
        "who_benefits": ["residents", "senior citizens"],
        "vulnerable_groups": ["elderly"],
        "evidence_ids": evidence_ids or ["EVID-001"],
        "speculative": speculative,
    }
    if verdict:
        data["verdict"] = verdict
    if scores:
        data["scores"] = scores
    _write_yaml(opp_dir / "opportunity.yaml", data)


def _make_maker_json(
    opp_dir: Path,
    opp_id: str,
    maker_score: float = 75.0,
    rank_score: float = 65.0,
    evidence_ids: list[str] | None = None,
) -> None:
    """Write a maker.json file."""
    opp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        opp_dir / "maker.json",
        {
            "opportunity_id": opp_id,
            "maker_score": maker_score,
            "maker_confidence": "high",
            "people_helped_score": 70.0,
            "severity_score": 60.0,
            "impact_score": 65.0,
            "validity_score": 80.0,
            "intervention_ease_score": 50.0,
            "harm_risk_score": 20.0,
            "ability_to_act_score": 55.0,
            "rank_score": rank_score,
            "evidence_ids": evidence_ids or ["EVID-001"],
            "summary": "Test maker summary",
        },
    )


def _make_taker_json(
    opp_dir: Path,
    opp_id: str,
    taker_score: float = 25.0,
) -> None:
    """Write a taker.json file."""
    opp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        opp_dir / "taker.json",
        {
            "opportunity_id": opp_id,
            "taker_score": taker_score,
            "taker_confidence": "medium",
        },
    )


def _make_mediator_json(
    opp_dir: Path,
    opp_id: str,
    verdict: str = "MANUAL_POC",
) -> None:
    """Write a mediator.json file."""
    opp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        opp_dir / "mediator.json",
        {
            "opportunity_id": opp_id,
            "verdict": verdict,
            "maker_score": 75.0,
            "taker_score": 25.0,
            "balance_summary": "Maker substantially outweighs Taker.",
            "do_no_harm": {
                "vulnerable_groups_affected": ["elderly"],
                "possible_negative_side_effects": "May create dependency.",
                "safeguards_required_before_poc": "Validate with community.",
            },
            "recommended_intervention_shape": "Test manually with small-scale public guide.",
            "summary": "Mediator analysis summary.",
        },
    )


def _make_cost_json(opp_dir: Path, opp_id: str) -> None:
    """Write a cost.json file."""
    opp_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        opp_dir / "cost.json",
        {
            "opportunity_id": opp_id,
            "poc_type": "public_guide",
            "cost_estimate_usd": "$0–$50",
            "time_estimate": "1 weekend",
            "risk_level": "low",
            "first_3_actions": ["Research", "Draft guide", "Publish"],
        },
    )


def _make_evidence_index(run_dir: Path) -> None:
    """Write an evidence.json file in the evidence folder."""
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        ev_dir / "evidence.json",
        [
            {
                "id": "EVID-001",
                "source_url": "https://example.gov/report",
                "source_domain": "example.gov",
                "source_type": "government",
                "evidence_type": "official_statement",
                "snippet": "Test evidence snippet.",
                "language": "en",
                "claim_classification": "evidence_based",
                "trust_score": 85.0,
                "recency": "2025-01",
                "confidence": "high",
            },
            {
                "id": "EVID-002",
                "source_url": "https://news.example.com/article",
                "source_domain": "news.example.com",
                "source_type": "major_news",
                "evidence_type": "news_report",
                "snippet": "Another test snippet.",
                "language": "en",
                "claim_classification": "evidence_based",
                "trust_score": 70.0,
                "recency": "2025-01",
                "confidence": "medium",
            },
        ],
    )


def _make_full_opportunity(
    run_dir: Path,
    slug: str,
    opp_id: str,
    title: str = "Test Opportunity",
    verdict: str = "MANUAL_POC",
    rank_score: float = 65.0,
    maker_score: float = 75.0,
    taker_score: float = 25.0,
    speculative: bool = False,
    evidence_ids: list[str] | None = None,
) -> Path:
    """Create a complete opportunity folder with all artifacts."""
    opp_dir = run_dir / "opportunities" / slug
    _make_opportunity_yaml(
        opp_dir, opp_id, title=title, verdict=verdict, speculative=speculative,
        evidence_ids=evidence_ids,
    )
    _make_maker_json(opp_dir, opp_id, maker_score=maker_score,
                     rank_score=rank_score, evidence_ids=evidence_ids)
    _make_taker_json(opp_dir, opp_id, taker_score=taker_score)
    _make_mediator_json(opp_dir, opp_id, verdict=verdict)
    _make_cost_json(opp_dir, opp_id)
    return opp_dir


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _invoke_in(tmp_path: Path, *args: str):
    """Invoke the CLI with the CWD set to tmp_path."""
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        runner = CliRunner()
        return runner.invoke(app, list(args))
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Tests: categorization
# ---------------------------------------------------------------------------


class TestCategorization:
    def test_ranked_opportunity(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1",
            slug="opp-1",
            title="Ranked Opp",
            scores={"rank_score": 65.0, "maker_score": 75.0},
            verdict="MANUAL_POC",
            has_maker=True,
            has_taker=True,
            has_mediator=True,
            has_cost=True,
        )
        assert opp.is_complete
        assert not opp.is_rejected
        assert opp.is_ranked

    def test_rejected_ignored(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1", scores={"rank_score": 25.0},
            verdict="IGNORE", has_maker=True, has_taker=True,
        )
        assert not opp.is_ranked
        assert opp.is_rejected

    def test_rejected_do_not_touch(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1", scores={"rank_score": 30.0},
            verdict="DO_NOT_TOUCH", has_maker=True, has_taker=True,
        )
        assert not opp.is_ranked
        assert opp.is_rejected

    def test_rejected_non_intervention(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1", scores={"rank_score": 40.0},
            verdict="NON_INTERVENTION", has_maker=True, has_taker=True,
        )
        assert not opp.is_ranked
        assert opp.is_rejected

    def test_incomplete_no_scores(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1", scores={},
            verdict="MANUAL_POC",
        )
        assert not opp.is_complete
        assert not opp.is_ranked
        assert not opp.is_rejected

    def test_incomplete_no_verdict(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1",
            scores={"rank_score": 65.0}, verdict=None,
        )
        assert not opp.is_complete
        assert not opp.is_ranked

    def test_incomplete_status_steps(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1",
            scores={"rank_score": 65.0}, verdict="MANUAL_POC",
            status_incomplete_steps=["Maker Agent"],
        )
        assert not opp.is_complete
        assert not opp.is_ranked

    def test_categorize_splits_correctly(self) -> None:
        ranked = LoadedOpportunity(
            opportunity_id="OPP-R", slug="opp-r",
            scores={"rank_score": 65.0}, verdict="BUILD_POC",
            has_maker=True, has_taker=True, has_mediator=True, has_cost=True,
        )
        rejected = LoadedOpportunity(
            opportunity_id="OPP-X", slug="opp-x",
            scores={"rank_score": 25.0}, verdict="IGNORE",
            has_maker=True, has_taker=True,
        )
        incomplete = LoadedOpportunity(
            opportunity_id="OPP-I", slug="opp-i",
        )

        r, x, i = ReportAgent._categorize([ranked, rejected, incomplete])
        assert len(r) == 1
        assert r[0].opportunity_id == "OPP-R"
        assert len(x) == 1
        assert x[0].opportunity_id == "OPP-X"
        assert len(i) == 1
        assert i[0].opportunity_id == "OPP-I"


# ---------------------------------------------------------------------------
# Tests: ranking order
# ---------------------------------------------------------------------------


class TestRankingOrder:
    def test_sorted_by_rank_score_descending(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)

        _make_full_opportunity(run_dir, "opp-alpha", "OPP-A",
                               title="Alpha", rank_score=45.0)
        _make_full_opportunity(run_dir, "opp-beta", "OPP-B",
                               title="Beta", rank_score=80.0)
        _make_full_opportunity(run_dir, "opp-gamma", "OPP-C",
                               title="Gamma", rank_score=62.0)

        agent = ReportAgent()
        opps = agent._discover_and_load(run_dir)
        ranked, _, _ = agent._categorize(opps)
        # Sort by rank_score descending (same as generate does)
        ranked.sort(key=lambda o: o.rank_score or 0.0, reverse=True)

        assert len(ranked) == 3
        assert ranked[0].title == "Beta"
        assert ranked[1].title == "Gamma"
        assert ranked[2].title == "Alpha"


# ---------------------------------------------------------------------------
# Tests: report content (populated)
# ---------------------------------------------------------------------------


class TestReportContent:
    def test_header_includes_city_community(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")
        _make_evidence_index(run_dir)

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "# Final Report: Lodz / test community" in content
        assert "Run ID" in content
        assert "Generated:" in content

    def test_ranking_formula_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "## Ranking Formula" in content
        assert "people_helped_score" in content
        assert "low_harm_score = 100 - harm_risk_score" in content

    def test_opportunity_fields_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1", title="My Opportunity",
                               rank_score=72.5, maker_score=82.0, taker_score=18.0)
        _make_evidence_index(run_dir)

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "My Opportunity" in content
        assert "rank_score: 72.5" in content
        assert "Pain Summary" in content
        assert "Who Benefits" in content
        assert "Maker Score" in content
        assert "Taker Score" in content
        assert "Validity Score" in content
        assert "Impact Estimate" in content
        assert "Intervention Ease" in content
        assert "Harm Risk" in content
        assert "Ability to Act" in content
        assert "Mediator Summary" in content
        assert "POC Cost Estimate" in content
        assert "Do No Harm Summary" in content
        assert "Evidence References" in content

    def test_recommended_next_action_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "Recommended Next Action" in content
        assert "Test manually with small-scale public guide" in content

    def test_poc_cost_estimate_table(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "POC Type" in content
        assert "public_guide" in content
        assert "$0–$50" in content
        assert "1 weekend" in content
        assert "low" in content
        assert "First 3 Actions" in content

    def test_evidence_referenced_not_inlined(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1",
                               evidence_ids=["EVID-001", "EVID-002"])
        _make_evidence_index(run_dir)

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        # Evidence is referenced by ID, not with full snippet content
        assert "`EVID-001`" in content
        assert "`EVID-002`" in content
        assert "example.gov" in content
        assert "news.example.com" in content
        # The full snippet should NOT appear in the report
        assert "Test evidence snippet." not in content

    def test_top_evidence_sources_section(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1",
                               evidence_ids=["EVID-001", "EVID-002"])
        _make_evidence_index(run_dir)

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "## Top Evidence Sources" in content
        assert "example.gov" in content
        assert "85" in content  # trust score

    def test_appendix_links_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "## Appendix" in content
        assert "rejected-opportunities.md" in content
        assert "incomplete-opportunities.md" in content

    def test_vulnerable_groups_section(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "#### Vulnerable Groups" in content
        assert "elderly" in content


# ---------------------------------------------------------------------------
# Tests: rejected and incomplete
# ---------------------------------------------------------------------------


class TestRejectedAndIncomplete:
    def test_rejected_in_appendix_not_main(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-good", "OPP-GOOD", verdict="BUILD_POC",
                               title="Good Opp", rank_score=70.0)
        _make_full_opportunity(run_dir, "opp-bad", "OPP-BAD", verdict="IGNORE",
                               title="Bad Opp", rank_score=20.0)

        agent = ReportAgent()
        agent.generate(run_dir)

        main_report = (run_dir / "final-report.md").read_text(encoding="utf-8")

        # Good opp in main report, bad opp NOT
        assert "Good Opp" in main_report
        assert "Bad Opp" not in main_report

        # Bad opp in appendix
        appendix = (run_dir / "appendix" / "rejected-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "Bad Opp" in appendix
        assert "IGNORE" in appendix

    def test_incomplete_in_appendix(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-good", "OPP-GOOD", title="Good Opp")
        # Create an incomplete opportunity (only opportunity.yaml, no other files)
        opp_dir = run_dir / "opportunities" / "opp-incomplete"
        _make_opportunity_yaml(opp_dir, "OPP-INC", title="Incomplete Opp")

        agent = ReportAgent()
        agent.generate(run_dir)

        appendix = (run_dir / "appendix" / "incomplete-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "Incomplete Opp" in appendix
        assert "Maker Agent" in appendix

    def test_do_not_touch_goes_to_rejected(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-dnt", "OPP-DNT",
                               verdict="DO_NOT_TOUCH", title="DNT Opp",
                               rank_score=10.0)

        agent = ReportAgent()
        agent.generate(run_dir)

        appendix = (run_dir / "appendix" / "rejected-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "DNT Opp" in appendix
        assert "DO_NOT_TOUCH" in appendix

    def test_non_intervention_goes_to_rejected(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-ni", "OPP-NI",
                               verdict="NON_INTERVENTION", title="NI Opp",
                               rank_score=35.0)

        agent = ReportAgent()
        agent.generate(run_dir)

        appendix = (run_dir / "appendix" / "rejected-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "NI Opp" in appendix
        assert "NON_INTERVENTION" in appendix


# ---------------------------------------------------------------------------
# Tests: no-valid-opportunities path
# ---------------------------------------------------------------------------


class TestNoValidOpportunities:
    def test_empty_opportunities_produces_explanatory_report(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "No valid opportunities found" in content
        assert "What was searched" in content
        assert "Lodz" in content
        assert "test community" in content
        assert "Why evidence was insufficient" in content
        assert "Recommended next search direction" in content

    def test_empty_but_with_sources_dir(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        sources_dir = run_dir / "sources"
        sources_dir.mkdir()
        (sources_dir / "search-results.json").write_text("[]")

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "No valid opportunities found" in content
        assert "source file" in content.lower()

    def test_empty_with_evidence_items(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_evidence_index(run_dir)

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "No valid opportunities found" in content
        assert "evidence items" in content.lower()
        assert "example.gov" in content

    def test_all_rejected_produces_no_valid(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1", verdict="IGNORE",
                               title="Rejected 1", rank_score=20.0)
        _make_full_opportunity(run_dir, "opp-2", "OPP-2", verdict="DO_NOT_TOUCH",
                               title="Rejected 2", rank_score=15.0)

        agent = ReportAgent()
        report_path = agent.generate(run_dir)
        content = Path(report_path).read_text(encoding="utf-8")

        assert "No valid opportunities found" in content

        # Rejected still appear in appendix
        appendix = (run_dir / "appendix" / "rejected-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "Rejected 1" in appendix
        assert "Rejected 2" in appendix

    def test_appendix_files_created_when_empty(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)

        agent = ReportAgent()
        agent.generate(run_dir)

        assert (run_dir / "appendix" / "rejected-opportunities.md").is_file()
        assert (run_dir / "appendix" / "incomplete-opportunities.md").is_file()


# ---------------------------------------------------------------------------
# Tests: appendix content
# ---------------------------------------------------------------------------


class TestAppendixContent:
    def test_rejected_appendix_format(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-ign", "OPP-IGN", verdict="IGNORE",
                               title="Ignored Opp", rank_score=20.0)

        agent = ReportAgent()
        agent.generate(run_dir)

        content = (run_dir / "appendix" / "rejected-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "# Rejected Opportunities" in content
        assert "Ignored Opp" in content
        assert "IGNORE" in content
        assert "Maker score too low" in content

    def test_incomplete_appendix_format(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        opp_dir = run_dir / "opportunities" / "opp-inc"
        _make_opportunity_yaml(opp_dir, "OPP-INC", title="Incomplete Opp")
        # Only opportunity.yaml, no other files

        agent = ReportAgent()
        agent.generate(run_dir)

        content = (run_dir / "appendix" / "incomplete-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "# Incomplete Opportunities" in content
        assert "Incomplete Opp" in content
        assert "Maker Agent" in content
        assert "Scoring" in content

    def test_empty_rejected_appendix(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        agent.generate(run_dir)

        content = (run_dir / "appendix" / "rejected-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "No opportunities were rejected" in content

    def test_empty_incomplete_appendix(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        agent.generate(run_dir)

        content = (run_dir / "appendix" / "incomplete-opportunities.md").read_text(
            encoding="utf-8"
        )
        assert "No incomplete opportunities" in content


# ---------------------------------------------------------------------------
# Tests: loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_loads_run_metadata(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        metadata = ReportAgent._load_run_metadata(run_dir)

        assert metadata["city"] == "Lodz"
        assert metadata["community"] == "test community"
        assert "timestamp" in metadata

    def test_returns_empty_for_missing_run_yaml(self, tmp_path: Path) -> None:
        metadata = ReportAgent._load_run_metadata(tmp_path)
        assert metadata == {}

    def test_discovers_opportunities(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-a", "OPP-A")
        _make_full_opportunity(run_dir, "opp-b", "OPP-B")

        agent = ReportAgent()
        opps = agent._discover_and_load(run_dir)
        assert len(opps) == 2
        ids = {o.opportunity_id for o in opps}
        assert ids == {"OPP-A", "OPP-B"}

    def test_skips_non_dirs_in_opportunities(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        opps_root = run_dir / "opportunities"
        opps_root.mkdir()
        # Create a non-directory file that should be skipped
        (opps_root / "readme.md").write_text("# Readme")

        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        agent = ReportAgent()
        opps = agent._discover_and_load(run_dir)
        assert len(opps) == 1

    def test_loads_evidence_index(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_evidence_index(run_dir)

        index = ReportAgent._load_evidence_index(run_dir)
        assert len(index) == 2
        assert "EVID-001" in index
        assert index["EVID-001"]["source_domain"] == "example.gov"
        assert index["EVID-001"]["trust_score"] == 85.0

    def test_returns_empty_evidence_when_missing(self, tmp_path: Path) -> None:
        index = ReportAgent._load_evidence_index(tmp_path)
        assert index == {}

    def test_loads_status_yaml_when_present(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        opp_dir = run_dir / "opportunities" / "opp-inc"
        _make_opportunity_yaml(opp_dir, "OPP-INC", title="Incomplete")
        _write_yaml(
            opp_dir / "status.yaml",
            {"incomplete_steps": ["Maker Agent", "Taker Agent"]},
        )

        agent = ReportAgent()
        opps = agent._discover_and_load(run_dir)
        assert len(opps) == 1
        assert opps[0].status_incomplete_steps == ["Maker Agent", "Taker Agent"]


# ---------------------------------------------------------------------------
# Tests: CLI command
# ---------------------------------------------------------------------------


class TestCLIReportCommand:
    def test_report_command_succeeds(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1", title="CLI Test Opp")

        result = _invoke_in(tmp_path, "report", str(run_dir))

        assert result.exit_code == 0, result.output
        assert "Report generated" in result.output

    def test_report_command_creates_appendix(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1")

        _invoke_in(tmp_path, "report", str(run_dir))

        assert (run_dir / "appendix" / "rejected-opportunities.md").is_file()
        assert (run_dir / "appendix" / "incomplete-opportunities.md").is_file()

    def test_report_command_missing_dir(self, tmp_path: Path) -> None:
        result = _invoke_in(tmp_path, "report", str(tmp_path / "nonexistent"))
        assert result.exit_code == 1

    def test_report_command_with_evidence(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        _make_full_opportunity(run_dir, "opp-1", "OPP-1",
                               evidence_ids=["EVID-001"])
        _make_evidence_index(run_dir)

        result = _invoke_in(tmp_path, "report", str(run_dir))
        assert result.exit_code == 0, result.output

        content = (run_dir / "final-report.md").read_text(encoding="utf-8")
        assert "Top Evidence Sources" in content


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_reject_reason_ignore(self) -> None:
        assert "too low" in _reject_reason("IGNORE")

    def test_reject_reason_do_not_touch(self) -> None:
        assert "exploitation" in _reject_reason("DO_NOT_TOUCH").lower()

    def test_reject_reason_non_intervention(self) -> None:
        assert "not the right actor" in _reject_reason("NON_INTERVENTION")

    def test_reject_reason_unknown(self) -> None:
        assert _reject_reason(None) == "Unknown reason."
        assert _reject_reason("SOMETHING_ELSE") == "Unknown reason."

    def test_missing_steps_complete(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1",
            scores={"rank_score": 65.0}, verdict="BUILD_POC",
            has_maker=True, has_taker=True, has_mediator=True, has_cost=True,
        )
        assert _missing_steps(opp) == []

    def test_missing_steps_from_status(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1",
            status_incomplete_steps=["Research Agent"],
        )
        assert _missing_steps(opp) == ["Research Agent"]

    def test_missing_steps_partial(self) -> None:
        opp = LoadedOpportunity(
            opportunity_id="OPP-1", slug="opp-1",
            has_maker=True,
        )
        steps = _missing_steps(opp)
        assert "Taker Agent" in steps
        assert "Mediator Agent" in steps
        assert "Cost Checker Agent" in steps
        assert "Scoring" in steps
        assert "Verdict" in steps


# ---------------------------------------------------------------------------
# Tests: scoring module
# ---------------------------------------------------------------------------


class TestScoringFunctions:
    def test_compute_low_harm_score(self) -> None:
        from makeragents.scoring import compute_low_harm_score

        assert compute_low_harm_score(harm_risk_score=20.0) == 80.0
        assert compute_low_harm_score(harm_risk_score=0.0) == 100.0
        assert compute_low_harm_score(harm_risk_score=100.0) == 0.0

    def test_compute_rank_score(self) -> None:
        from makeragents.scoring import compute_rank_score

        result = compute_rank_score(
            people_helped_score=70,
            severity_score=60,
            validity_score=80,
            intervention_ease_score=50,
            harm_risk_score=20,
            ability_to_act_score=55,
        )
        expected = (
            70 * 0.22 + 60 * 0.20 + 80 * 0.18 + 50 * 0.14
            + 80 * 0.14 + 55 * 0.12
        )
        assert result == round(expected, 2)

    def test_compute_rank_score_matches_scoreset(self) -> None:
        from makeragents.scoring import compute_rank_score
        from makeragents.schemas import ScoreSet

        args = {
            "people_helped_score": 60.0,
            "severity_score": 50.0,
            "validity_score": 70.0,
            "intervention_ease_score": 60.0,
            "harm_risk_score": 30.0,
            "ability_to_act_score": 50.0,
        }
        sc = compute_rank_score(**args)
        ss = ScoreSet.calculate_rank_score(**args)
        assert sc == ss
