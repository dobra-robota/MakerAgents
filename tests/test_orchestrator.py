"""Tests for the pipeline orchestrator (mocked agents, no live calls)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock


from makeragents.agents.maker import MakerResult
from makeragents.agents.taker import TakerOutput
from makeragents.config import AppConfig
from makeragents.orchestrator import PipelineRunner
from makeragents.run import build_run_metadata, create_run_folder
from makeragents.schemas import (
    Confidence,
    EvidenceItem,
    EvidenceType,
    Opportunity,
    OpportunityType,
    SourceType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence_item(idx: int) -> EvidenceItem:
    """Build a minimal evidence item for tests."""
    return EvidenceItem(
        id=f"ev-{idx:03d}",
        source_url=f"https://example.com/{idx}",
        source_domain="example.com",
        source_type=SourceType.LOCAL_NEWS,
        evidence_type=EvidenceType.CLAIM,
        snippet=f"Test snippet {idx}",
        language="en",
        claim_classification="evidence_based",
        trust_score=75.0,
        recency="2025-01-01",
        confidence=Confidence.MEDIUM,
    )


def _make_opportunity(idx: int) -> Opportunity:
    """Build a minimal opportunity for tests."""
    return Opportunity(
        id=f"opp-{idx:03d}",
        title=f"Test Opportunity {idx}",
        type=OpportunityType.PUBLIC_GUIDE,
        pain_summary=f"Pain summary {idx}",
        who_benefits=["community members"],
        evidence_ids=[f"ev-{idx:03d}"],
    )


# ---------------------------------------------------------------------------
# Tests — topology and artifact layout
# ---------------------------------------------------------------------------


class TestOrchestratorTopology:
    """Verify the pipeline runs the correct sequence and writes artifacts."""

    def test_no_opportunities_returns_existing_report_path(self) -> None:
        """When there are no opportunities, per-opportunity stages are skipped."""
        runner = PipelineRunner(
            config=AppConfig(deepseek_api_key="test-key"),
        )
        metadata = build_run_metadata(city="Lodz", community="senior")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_folder(metadata, base_dir=Path(tmp))

            # Mock pre-opportunity agents to return empty results
            with (
                mock.patch(
                    "makeragents.orchestrator.ResearchAgent"
                ) as mock_research,
                mock.patch(
                    "makeragents.orchestrator.EvidenceAgent"
                ) as mock_evidence,
                mock.patch(
                    "makeragents.orchestrator.OpportunityAgent"
                ) as mock_opp,
            ):
                # Setup mocks
                mock_search = mock.MagicMock()
                mock_search.query_results = []
                mock_research.return_value.search.return_value = mock_search

                mock_evidence.return_value.process.return_value = []
                mock_evidence.return_value.save_evidence.return_value = None

                mock_opp.return_value.process.return_value = []

                result = runner.run(run_dir, metadata)

            assert result == str(run_dir / "final-report.md")
            assert (run_dir / "final-report.md").exists()

    def test_writes_opportunity_artifacts(self) -> None:
        """Per-opportunity artifacts are written to the §15 layout."""
        evidence_items = [_make_evidence_item(i) for i in range(3)]
        opp = _make_opportunity(0)

        runner = PipelineRunner(
            config=AppConfig(deepseek_api_key="test-key"),
        )
        metadata = build_run_metadata(city="Lodz", community="senior")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_folder(metadata, base_dir=Path(tmp))

            runner._write_opportunity_artifacts(opp, run_dir)

            # Check README.md
            readme = run_dir / "README.md"
            assert readme.exists()
            content = readme.read_text()
            assert opp.title in content
            assert opp.pain_summary in content

            # Check opportunity.yaml
            opp_yaml = run_dir / "opportunity.yaml"
            assert opp_yaml.exists()


class TestOrchestratorSequencing:
    """Verify Maker/Taker sequencing and scoped evidence selection."""

    def test_taker_runs_after_maker_artifacts_with_selected_evidence(self) -> None:
        """Taker receives Maker scores only after maker artifacts are saved."""
        opp = _make_opportunity(0)
        evidence = [_make_evidence_item(0), _make_evidence_item(1)]
        runner = PipelineRunner(
            config=AppConfig(deepseek_api_key="test-key"),
        )
        metadata = build_run_metadata(city="Lodz", community="senior")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_folder(metadata, base_dir=Path(tmp))
            opp_dir = run_dir / "opportunities" / "opp-000"

            call_order = []

            def capture_maker(*a, **kw):
                call_order.append("maker_run")
                return MakerResult(
                    opportunity_id=opp.id,
                    maker_score=75.0,
                    maker_confidence=Confidence.MEDIUM,
                    people_helped_score=70.0,
                    severity_score=60.0,
                    impact_score=65.0,
                    validity_score=80.0,
                    intervention_ease_score=50.0,
                    harm_risk_score=20.0,
                    ability_to_act_score=55.0,
                    rank_score=0.0,
                    evidence_ids=["ev-000"],
                    summary="Maker summary",
                )

            def save_maker(result, path):
                call_order.append("maker_save")
                assert path == opp_dir
                json_path = path / "maker.json"
                md_path = path / "maker.md"
                json_path.write_text("{}", encoding="utf-8")
                md_path.write_text("maker", encoding="utf-8")
                return json_path, md_path

            def capture_taker(opportunity, selected_evidence, *a, **kw):
                call_order.append("taker_run")
                assert "maker_save" in call_order
                assert opportunity.scores is not None
                assert opportunity.scores.maker_score == 75.0
                assert [item.id for item in selected_evidence] == ["ev-000"]
                assert opportunity.evidence_ids == ["ev-000"]
                return TakerOutput(
                    opportunity_id=opp.id,
                    taker_score=30.0,
                    taker_confidence="medium",
                    risk_breakdown={
                        "extraction_risk": 30.0,
                        "gatekeeping_risk": 10.0,
                        "false_authority_risk": 10.0,
                        "dependency_risk": 10.0,
                        "harm_risk": 10.0,
                    },
                    evidence_ids=["ev-000"],
                    summary="Risk summary",
                )

            def save_taker(result, path):
                call_order.append("taker_save")
                assert path == opp_dir

            maker_agent = mock.MagicMock()
            maker_agent.run_with_llm.side_effect = capture_maker
            maker_agent.save_output.side_effect = save_maker

            taker_agent = mock.MagicMock()
            taker_agent.run_with_llm.side_effect = capture_taker
            taker_agent.save_output.side_effect = save_taker

            with (
                mock.patch(
                    "makeragents.orchestrator.MakerAgent",
                    return_value=maker_agent,
                ),
                mock.patch(
                    "makeragents.orchestrator.TakerAgent",
                    return_value=taker_agent,
                ),
            ):
                runner._process_one_opportunity(
                    opp, evidence, run_dir, "Lodz", "senior"
                )

            assert call_order == [
                "maker_run",
                "maker_save",
                "taker_run",
                "taker_save",
            ]


class TestOrchestratorStatusTracking:
    """Verify status.yaml is written correctly per opportunity."""

    def test_status_yaml_written(self) -> None:
        """status.yaml tracks all pipeline steps."""
        opp = _make_opportunity(0)
        evidence = [_make_evidence_item(0)]
        runner = PipelineRunner(
            config=AppConfig(deepseek_api_key="test-key"),
        )
        metadata = build_run_metadata(city="Lodz", community="senior")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_folder(metadata, base_dir=Path(tmp))
            opp_dir = run_dir / "opportunities" / "opp-000"

            maker_agent = mock.MagicMock()
            maker_agent.run_with_llm.return_value = MakerResult(
                opportunity_id=opp.id,
                maker_score=75.0,
                maker_confidence=Confidence.MEDIUM,
                people_helped_score=70.0,
                severity_score=60.0,
                impact_score=65.0,
                validity_score=80.0,
                intervention_ease_score=50.0,
                harm_risk_score=20.0,
                ability_to_act_score=55.0,
                rank_score=0.0,
                evidence_ids=["ev-000"],
            )

            def save_maker(result, path):
                json_path = path / "maker.json"
                md_path = path / "maker.md"
                json_path.write_text("{}", encoding="utf-8")
                md_path.write_text("maker", encoding="utf-8")
                return json_path, md_path

            maker_agent.save_output.side_effect = save_maker

            taker_agent = mock.MagicMock()
            taker_agent.run_with_llm.return_value = TakerOutput(
                opportunity_id=opp.id,
                taker_score=30.0,
                taker_confidence="medium",
                risk_breakdown={
                    "extraction_risk": 20.0,
                    "gatekeeping_risk": 20.0,
                    "false_authority_risk": 20.0,
                    "dependency_risk": 20.0,
                    "harm_risk": 20.0,
                },
                evidence_ids=["ev-000"],
                summary="Risk",
            )
            taker_agent.save_output.return_value = None

            with (
                mock.patch(
                    "makeragents.orchestrator.MakerAgent",
                    return_value=maker_agent,
                ),
                mock.patch(
                    "makeragents.orchestrator.TakerAgent",
                    return_value=taker_agent,
                ),
            ):
                runner._process_one_opportunity(
                    opp, evidence, run_dir, "Lodz", "senior"
                )

            # Status file exists and stops at Taker for this issue scope.
            status_path = opp_dir / "status.yaml"
            assert status_path.exists()

            import yaml
            status = yaml.safe_load(status_path.read_text())
            assert status["opportunity_id"] == opp.id
            for step in ["research", "evidence", "opportunity", "maker", "taker"]:
                assert status["steps"][step] == "complete", f"{step} not complete"
            for step in ["mediator", "cost_checker"]:
                assert status["steps"][step] == "incomplete", f"{step} should not run"
