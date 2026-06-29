"""Tests for the pipeline orchestrator (mocked agents, no live calls)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from makeragents.agents.maker import MakerResult
from makeragents.agents.mediator import MediatorResult
from makeragents.agents.taker import TakerOutput
from makeragents.config import AppConfig
from makeragents.orchestrator import PipelineRunner
from makeragents.run import (
    build_run_metadata,
    create_run_folder,
    opportunity_artifact_slug,
)
from makeragents.schemas import (
    Confidence,
    EvidenceItem,
    EvidenceType,
    Opportunity,
    OpportunityType,
    RunMetadata,
    SourceType,
    Verdict,
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

    def test_no_opportunities_returns_existing_report_stub(self) -> None:
        """When no opportunities are generated, the existing report stub remains."""
        runner = PipelineRunner(
            config=AppConfig(deepseek_api_key="test-key"),
        )
        metadata = build_run_metadata(city="Lodz", community="senior")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_folder(metadata, base_dir=Path(tmp))

            # Mock all agents to return empty results
            with (
                mock.patch.object(runner, "run", wraps=runner.run) as wrapped_run,
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

            # Should still run and return a report path
            assert "final-report.md" in result
            assert (run_dir / "final-report.md").exists()

    def test_run_writes_maker_artifacts_with_selected_evidence(self) -> None:
        """Generated opportunities get Maker artifacts using cited evidence only."""
        evidence_items = [_make_evidence_item(i) for i in range(3)]
        opp_a = _make_opportunity(0).model_copy(
            update={"evidence_ids": ["ev-000", "ev-002"]}
        )
        opp_b = _make_opportunity(1)
        captured: dict[str, list[str]] = {}

        def make_result(
            opportunity: Opportunity,
            supporting_evidence: list[EvidenceItem],
            **_: object,
        ) -> MakerResult:
            evidence_ids = [item.id for item in supporting_evidence]
            captured[opportunity.id] = evidence_ids
            return MakerResult(
                opportunity_id=opportunity.id,
                maker_score=75.0,
                maker_confidence=Confidence.MEDIUM,
                people_helped_score=70.0,
                severity_score=60.0,
                impact_score=65.0,
                validity_score=80.0,
                intervention_ease_score=50.0,
                harm_risk_score=20.0,
                ability_to_act_score=55.0,
                rank_score=64.3,
                evidence_ids=evidence_ids,
                summary=f"Maker summary for {opportunity.id}",
            )

        def save_result(
            result: MakerResult, opportunity_dir: Path
        ) -> tuple[Path, Path]:
            opportunity_dir.mkdir(parents=True, exist_ok=True)
            json_path = opportunity_dir / "maker.json"
            md_path = opportunity_dir / "maker.md"
            json_path.write_text(json.dumps(result.to_json_dict()), encoding="utf-8")
            md_path.write_text(result.summary, encoding="utf-8")
            return json_path, md_path

        runner = PipelineRunner(config=AppConfig(deepseek_api_key="test-key"))
        metadata = build_run_metadata(
            city="Lodz", community="senior", max_opportunities=2
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_folder(metadata, base_dir=Path(tmp))
            maker_agent = mock.MagicMock()
            maker_agent.run_with_llm.side_effect = make_result
            maker_agent.save_output.side_effect = save_result
            with (
                mock.patch("makeragents.orchestrator.ResearchAgent") as mock_research,
                mock.patch("makeragents.orchestrator.EvidenceAgent") as mock_evidence,
                mock.patch("makeragents.orchestrator.OpportunityAgent") as mock_opp,
                mock.patch(
                    "makeragents.orchestrator.MakerAgent",
                    return_value=maker_agent,
                ),
                mock.patch("makeragents.orchestrator.TakerAgent") as mock_taker,
                mock.patch("makeragents.orchestrator.MediatorAgent") as mock_mediator,
                mock.patch("makeragents.orchestrator.CostCheckerAgent") as mock_cost,
                mock.patch("makeragents.orchestrator.ReportAgent") as mock_report,
            ):
                mock_search = mock.MagicMock()
                mock_search.query_results = []
                mock_research.return_value.search.return_value = mock_search
                mock_evidence.return_value.process.return_value = evidence_items
                mock_evidence.return_value.save_evidence.return_value = None
                mock_opp.return_value.process.return_value = [opp_a, opp_b]

                runner.run(run_dir, metadata)

            assert captured == {
                opp_a.id: ["ev-000", "ev-002"],
                opp_b.id: ["ev-001"],
            }
            for opp in (opp_a, opp_b):
                opp_dir = run_dir / "opportunities" / opportunity_artifact_slug(opp)
                maker_data = json.loads(
                    (opp_dir / "maker.json").read_text(encoding="utf-8")
                )
                assert maker_data["maker_score"] == 75.0
                assert maker_data["maker_confidence"] == "medium"
                assert maker_data["rank_score"] == 64.3
                assert maker_data["summary"] == f"Maker summary for {opp.id}"
                assert maker_data["evidence_ids"] == captured[opp.id]
                for score_key in [
                    "people_helped_score",
                    "severity_score",
                    "impact_score",
                    "validity_score",
                    "intervention_ease_score",
                    "harm_risk_score",
                    "ability_to_act_score",
                ]:
                    assert score_key in maker_data

                opportunity_data = yaml.safe_load(
                    (opp_dir / "opportunity.yaml").read_text(encoding="utf-8")
                )
                assert opportunity_data["scores"]["maker_score"] == 75.0
                assert opportunity_data["scores"]["rank_score"] == 64.3

            mock_taker.assert_not_called()
            mock_mediator.assert_not_called()
            mock_cost.assert_not_called()
            mock_report.assert_not_called()

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


class TestOrchestratorConcurrency:
    """Verify Maker/Taker run in parallel and opportunities are concurrent."""

    def test_maker_taker_parallel(self) -> None:
        """Maker and Taker are submitted concurrently."""
        opp = _make_opportunity(0)
        evidence = [_make_evidence_item(0)]
        runner = PipelineRunner(
            config=AppConfig(deepseek_api_key="test-key"),
        )
        metadata = build_run_metadata(city="Lodz", community="senior")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_folder(metadata, base_dir=Path(tmp))
            opp_dir = run_dir / "opportunities" / opportunity_artifact_slug(opp)
            opp_dir.mkdir(parents=True)

            call_order = []

            def capture_maker(*a, **kw):
                call_order.append("maker")
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
                )

            def capture_taker(*a, **kw):
                call_order.append("taker")
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
                    evidence_ids=[],
                    summary="Risk summary",
                )

            maker_agent = mock.MagicMock()
            maker_agent.run_with_llm.side_effect = capture_maker
            maker_agent.save_output.return_value = None

            taker_agent = mock.MagicMock()
            taker_agent.run_with_llm.side_effect = capture_taker
            taker_agent.save_output.return_value = None

            mediator = mock.MagicMock()
            mediator_result = MediatorResult(
                opportunity_id=opp.id,
                verdict=Verdict.WATCH,
                maker_score=75.0,
                taker_score=30.0,
                balance_summary="Balanced",
                evidence_ids=[],
                summary="Summary",
            )
            mediator.run_with_llm.return_value = mediator_result
            mediator.save_output.return_value = None

            cost = mock.MagicMock()
            cost.run_with_llm.return_value = mock.MagicMock()
            cost.save_output.return_value = None

            with (
                mock.patch(
                    "makeragents.orchestrator.MakerAgent",
                    return_value=maker_agent,
                ),
                mock.patch(
                    "makeragents.orchestrator.TakerAgent",
                    return_value=taker_agent,
                ),
                mock.patch(
                    "makeragents.orchestrator.MediatorAgent",
                    return_value=mediator,
                ),
                mock.patch(
                    "makeragents.orchestrator.CostCheckerAgent",
                    return_value=cost,
                ),
            ):
                runner._process_one_opportunity(
                    opp, evidence, run_dir, "Lodz", "senior"
                )

            # Both maker and taker were called
            assert "maker" in call_order
            assert "taker" in call_order


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
            opp_dir = run_dir / "opportunities" / opportunity_artifact_slug(opp)
            opp_dir.mkdir(parents=True)

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
            )
            maker_agent.save_output.return_value = None

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
                evidence_ids=[],
                summary="Risk",
            )
            taker_agent.save_output.return_value = None

            mediator = mock.MagicMock()
            mediator_result = MediatorResult(
                opportunity_id=opp.id,
                verdict=Verdict.WATCH,
                maker_score=75.0,
                taker_score=30.0,
                balance_summary="Balanced",
                evidence_ids=[],
                summary="Summary",
            )
            mediator.run_with_llm.return_value = mediator_result
            mediator.save_output.return_value = None

            cost = mock.MagicMock()
            cost.run_with_llm.return_value = mock.MagicMock()
            cost.save_output.return_value = None

            with (
                mock.patch(
                    "makeragents.orchestrator.MakerAgent",
                    return_value=maker_agent,
                ),
                mock.patch(
                    "makeragents.orchestrator.TakerAgent",
                    return_value=taker_agent,
                ),
                mock.patch(
                    "makeragents.orchestrator.MediatorAgent",
                    return_value=mediator,
                ),
                mock.patch(
                    "makeragents.orchestrator.CostCheckerAgent",
                    return_value=cost,
                ),
            ):
                runner._process_one_opportunity(
                    opp, evidence, run_dir, "Lodz", "senior"
                )

            # Status file exists and has all steps complete
            status_path = opp_dir / "status.yaml"
            assert status_path.exists()

            import yaml
            status = yaml.safe_load(status_path.read_text())
            assert status["opportunity_id"] == opp.id
            for step in [
                "research", "evidence", "opportunity",
                "maker", "taker", "mediator", "cost_checker",
            ]:
                assert status["steps"][step] == "complete", f"{step} not complete"
