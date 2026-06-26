"""Tests for the Mediator Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from makeragents.agents.mediator import MediatorAgent, MediatorResult
from makeragents.run import slugify
from makeragents.schemas import (
    Confidence,
    Opportunity,
    OpportunityType,
    ScoreSet,
    Verdict,
)


def _make_opportunity(
    id_: str = "OPP-test",
    o_type: OpportunityType = OpportunityType.PUBLIC_GUIDE,
    maker: float = 70.0,
    taker: float = 30.0,
    speculative: bool = False,
    evidence_ids: list[str] | None = None,
    vulnerable: list[str] | None = None,
) -> Opportunity:
    return Opportunity(
        id=id_,
        title="Test",
        type=o_type,
        pain_summary="pain",
        who_benefits=["residents"],
        vulnerable_groups=vulnerable or [],
        evidence_ids=evidence_ids or ["EVID-001"],
        speculative=speculative,
        scores=ScoreSet(
            validity_score=70.0,
            maker_score=maker,
            maker_confidence=Confidence.MEDIUM,
            taker_score=taker,
            taker_confidence=Confidence.MEDIUM,
            people_helped_score=60.0,
            severity_score=50.0,
            impact_score=55.0,
            intervention_ease_score=60.0,
            harm_risk_score=30.0,
            ability_to_act_score=50.0,
            rank_score=ScoreSet.calculate_rank_score(
                people_helped_score=60.0,
                severity_score=50.0,
                validity_score=70.0,
                intervention_ease_score=60.0,
                harm_risk_score=30.0,
                ability_to_act_score=50.0,
            ),
        ),
    )


class TestVerdictRules:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_do_not_touch_high_taker_low_maker(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=40.0, taker=85.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.DO_NOT_TOUCH

    def test_ignore_low_maker(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=20.0, taker=10.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.IGNORE

    def test_watch_speculative_low_maker(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=40.0, taker=20.0, speculative=True)
        result = agent.run(opp)
        assert result.verdict == Verdict.WATCH

    def test_build_poc_software(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(
            maker=60.0, taker=20.0, o_type=OpportunityType.SOFTWARE_TOOLING
        )
        result = agent.run(opp)
        assert result.verdict == Verdict.BUILD_POC

    def test_manual_poc_non_software(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=60.0, taker=20.0, o_type=OpportunityType.PUBLIC_GUIDE)
        result = agent.run(opp)
        assert result.verdict == Verdict.MANUAL_POC

    def test_research_more(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=45.0, taker=50.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.RESEARCH_MORE

    def test_non_intervention_default(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=35.0, taker=65.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.NON_INTERVENTION

    def test_taker_at_threshold(self, agent: MediatorAgent) -> None:
        # taker >= 80, maker just under 50 → DO_NOT_TOUCH
        opp = _make_opportunity(maker=49.0, taker=80.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.DO_NOT_TOUCH

    def test_maker_at_threshold(self, agent: MediatorAgent) -> None:
        # maker >= 50, taker < 40 → MANUAL_POC
        opp = _make_opportunity(maker=50.0, taker=39.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.MANUAL_POC

    def test_speculative_bypasses_build_poc(self, agent: MediatorAgent) -> None:
        # speculative=True with maker >= 50 and taker < 40 would hit
        # rule 4 (BUILD_POC/MANUAL_POC), but should be RESEARCH_MORE
        opp = _make_opportunity(
            maker=60.0, taker=20.0, speculative=True, o_type=OpportunityType.PUBLIC_GUIDE
        )
        result = agent.run(opp)
        assert result.verdict == Verdict.RESEARCH_MORE


class TestDoNoHarm:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_all_fields_present(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        dnh = result.do_no_harm
        expected_keys = {
            "vulnerable_groups_affected",
            "possible_negative_side_effects",
            "abuse_or_exploitation_risks",
            "legal_or_tos_concerns",
            "trust_and_misinformation_risks",
            "dependency_risks",
            "gatekeeping_risks",
            "false_authority_risks",
            "safeguards_required_before_poc",
        }
        assert set(dnh.keys()) == expected_keys

    def test_vulnerable_groups_from_opportunity(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(vulnerable=["elderly", "children"])
        result = agent.run(opp)
        assert "elderly" in result.do_no_harm["vulnerable_groups_affected"]


class TestBalanceSummary:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_maker_outweighs(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=80.0, taker=20.0)
        result = agent.run(opp)
        assert "substantially outweighs" in result.balance_summary

    def test_taker_outweighs(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=20.0, taker=80.0)
        result = agent.run(opp)
        assert "substantially outweighs" in result.balance_summary

    def test_balanced(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=50.0, taker=50.0)
        result = agent.run(opp)
        assert "roughly balanced" in result.balance_summary


class TestSaveOutput:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_writes_files(self, agent: MediatorAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        json_path, md_path = agent.save_output(result, tmp_path)
        assert json_path.exists()
        assert md_path.exists()

    def test_json_serializable(self, agent: MediatorAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        json_path, _ = agent.save_output(result, tmp_path)
        data = json.loads(json_path.read_text())
        assert data["verdict"] == "MANUAL_POC"
        assert "do_no_harm" in data

    def test_md_contains_verdict(self, agent: MediatorAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        _, md_path = agent.save_output(result, tmp_path)
        content = md_path.read_text()
        assert "MANUAL_POC" in content


class TestMediatorResult:
    def test_model_dump_json(self) -> None:
        result = MediatorResult(
            opportunity_id="OPP-1",
            verdict=Verdict.MANUAL_POC,
            maker_score=65.0,
            taker_score=25.0,
            balance_summary="Maker outweighs",
            do_no_harm={"test": "data"},
            recommended_intervention_shape="Try it",
            evidence_ids=["EVID-001"],
            summary="Test",
        )
        d = result.model_dump(mode="json")
        assert d["verdict"] == "MANUAL_POC"
        assert d["maker_score"] == 65.0
