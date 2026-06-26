"""Tests for the Cost Checker Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from makeragents.agents.cost_checker import CostCheckerAgent, CostEstimate
from makeragents.schemas import (
    Confidence,
    Opportunity,
    OpportunityType,
    POCType,
    ScoreSet,
)


def _make_opportunity(
    id_: str = "OPP-test",
    o_type: OpportunityType = OpportunityType.PUBLIC_GUIDE,
    maker: float = 70.0,
    speculative: bool = False,
) -> Opportunity:
    return Opportunity(
        id=id_,
        title="Test",
        type=o_type,
        pain_summary="pain",
        who_benefits=["residents"],
        evidence_ids=["EVID-001"],
        speculative=speculative,
        scores=ScoreSet(
            validity_score=70.0,
            maker_score=maker,
            maker_confidence=Confidence.MEDIUM,
            taker_score=30.0,
            taker_confidence=Confidence.MEDIUM,
            people_helped_score=60.0,
            severity_score=50.0,
            impact_score=55.0,
            intervention_ease_score=60.0,
            harm_risk_score=30.0,
            ability_to_act_score=50.0,
            rank_score=50.0,
        ),
    )


class TestPOCTypeMapping:
    def test_all_opportunity_types_mapped(self) -> None:
        for o_type in OpportunityType:
            poc = CostCheckerAgent.opportunity_type_to_poc_type(o_type)
            assert isinstance(poc, POCType)

    def test_public_guide(self) -> None:
        assert (
            CostCheckerAgent.opportunity_type_to_poc_type(OpportunityType.PUBLIC_GUIDE)
            == POCType.PUBLIC_GUIDE
        )

    def test_software_tooling_maps_to_prototype(self) -> None:
        assert (
            CostCheckerAgent.opportunity_type_to_poc_type(OpportunityType.SOFTWARE_TOOLING)
            == POCType.SOFTWARE_PROTOTYPE
        )

    def test_dashboard(self) -> None:
        assert (
            CostCheckerAgent.opportunity_type_to_poc_type(OpportunityType.TRANSPARENCY_DASHBOARD)
            == POCType.DASHBOARD
        )

    def test_community_support_maps_to_coordination(self) -> None:
        assert (
            CostCheckerAgent.opportunity_type_to_poc_type(OpportunityType.COMMUNITY_SUPPORT_PROCESS)
            == POCType.COORDINATION_PROCESS
        )


class TestCostMap:
    def test_all_poc_types_have_entries(self) -> None:
        agent = CostCheckerAgent()
        for o_type in OpportunityType:
            usd, time_, risk = agent.map_opportunity_type(o_type)
            assert usd.startswith("$")
            assert len(time_) > 0
            assert risk in ("low", "medium", "high")

    def test_public_guide_cost(self) -> None:
        agent = CostCheckerAgent()
        usd, time_, risk = agent.map_opportunity_type(OpportunityType.PUBLIC_GUIDE)
        assert usd == "$0–$50"
        assert "weekend" in time_
        assert risk == "low"

    def test_manual_service_cost(self) -> None:
        agent = CostCheckerAgent()
        usd, time_, risk = agent.map_opportunity_type(OpportunityType.MANUAL_SERVICE)
        assert "$" in usd
        assert "weekend" in time_
        assert risk == "medium"

    def test_software_prototype_cost(self) -> None:
        agent = CostCheckerAgent()
        usd, time_, risk = agent.map_opportunity_type(OpportunityType.SOFTWARE_TOOLING)
        assert "$500–$10000" == usd
        assert risk == "high"


class TestEstimateMethod:
    @pytest.fixture
    def agent(self) -> CostCheckerAgent:
        return CostCheckerAgent()

    def test_returns_cost_estimate(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        assert isinstance(est, CostEstimate)
        assert est.opportunity_id == "OPP-test"

    def test_correct_poc_type(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity(o_type=OpportunityType.PUBLIC_GUIDE)
        est = agent.estimate(opp)
        assert est.poc_type == POCType.PUBLIC_GUIDE

    def test_cost_usd_present(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        assert est.cost_estimate_usd.startswith("$")

    def test_time_estimate_present(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        assert len(est.time_estimate) > 0

    def test_risk_level_valid(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        assert est.risk_level in ("low", "medium", "high")

    def test_low_maker_adds_note(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity(maker=20.0)
        est = agent.estimate(opp)
        assert "not be justified" in est.notes

    def test_speculative_adds_note(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity(speculative=True)
        est = agent.estimate(opp)
        assert "speculative" in est.notes

    def test_no_notes_when_healthy(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity(maker=70.0)
        est = agent.estimate(opp)
        assert est.notes == ""


class TestFirst3Actions:
    @pytest.fixture
    def agent(self) -> CostCheckerAgent:
        return CostCheckerAgent()

    def test_returns_3_actions(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        assert len(est.first_3_actions) == 3

    def test_all_non_empty(self, agent: CostCheckerAgent) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        for action in est.first_3_actions:
            assert len(action) > 0

    def test_different_per_type(self, agent: CostCheckerAgent) -> None:
        guide = agent.estimate(_make_opportunity(o_type=OpportunityType.PUBLIC_GUIDE))
        sw = agent.estimate(_make_opportunity(o_type=OpportunityType.SOFTWARE_TOOLING))
        assert guide.first_3_actions != sw.first_3_actions


class TestCostEstimateDict:
    def test_to_dict(self) -> None:
        est = CostEstimate(
            opportunity_id="OPP-1",
            poc_type=POCType.PUBLIC_GUIDE,
            cost_estimate_usd="$0–$50",
            time_estimate="1 weekend",
            risk_level="low",
            first_3_actions=["a", "b", "c"],
            notes="test",
        )
        d = est.to_dict()
        assert d["opportunity_id"] == "OPP-1"
        assert d["poc_type"] == "public_guide"


class TestArtifactWriting:
    @pytest.fixture
    def agent(self) -> CostCheckerAgent:
        return CostCheckerAgent()

    def test_writes_files(self, agent: CostCheckerAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        json_path, md_path = agent.write_artifacts(est, tmp_path)
        assert json_path.exists()
        assert md_path.exists()

    def test_json_contents(self, agent: CostCheckerAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        json_path, _ = agent.write_artifacts(est, tmp_path)
        data = json.loads(json_path.read_text())
        assert data["poc_type"] == "public_guide"

    def test_md_contents(self, agent: CostCheckerAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        est = agent.estimate(opp)
        _, md_path = agent.write_artifacts(est, tmp_path)
        content = md_path.read_text()
        assert "OPP-test" in content
        assert "First 3 Actions" in content
