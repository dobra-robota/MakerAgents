from __future__ import annotations

from pathlib import Path

from makeragents.agents.cost_checker import CostCheckerAgent, CostEstimate
from makeragents.agents.maker import MakerAgent, MakerResult
from makeragents.agents.mediator import MediatorAgent, MediatorResult
from makeragents.agents.opportunity import OpportunityAgent
from makeragents.agents.report import ReportAgent
from makeragents.agents.taker import TakerAgent, TakerOutput
from makeragents.run import opportunity_artifact_slug
from makeragents.schemas import (
    Confidence,
    Opportunity,
    OpportunityType,
    POCType,
    Verdict,
)


def _opportunity() -> Opportunity:
    return Opportunity(
        id="OPP-Senior Services Guide",
        title="Senior Services Guide",
        type=OpportunityType.PUBLIC_GUIDE,
        pain_summary="Residents cannot find services.",
        who_benefits=["senior residents"],
        evidence_ids=["ev-001"],
    )


def _maker_result(opportunity_id: str) -> MakerResult:
    return MakerResult(
        opportunity_id=opportunity_id,
        maker_score=75.0,
        maker_confidence=Confidence.MEDIUM,
        people_helped_score=70.0,
        severity_score=60.0,
        impact_score=65.0,
        validity_score=80.0,
        intervention_ease_score=50.0,
        harm_risk_score=20.0,
        ability_to_act_score=55.0,
        rank_score=62.1,
        evidence_ids=["ev-001"],
        summary="Useful public guide.",
    )


def _taker_output(opportunity_id: str) -> TakerOutput:
    return TakerOutput(
        opportunity_id=opportunity_id,
        taker_score=30.0,
        taker_confidence="medium",
        risk_breakdown={
            "extraction_risk": 10.0,
            "gatekeeping_risk": 20.0,
            "false_authority_risk": 30.0,
            "dependency_risk": 40.0,
            "harm_risk": 30.0,
        },
        evidence_ids=["ev-001"],
        summary="No exploitation instructions; risks are manageable.",
    )


def _mediator_result(opportunity_id: str) -> MediatorResult:
    return MediatorResult(
        opportunity_id=opportunity_id,
        verdict=Verdict.MANUAL_POC,
        maker_score=75.0,
        taker_score=30.0,
        balance_summary="Maker outweighs Taker.",
        do_no_harm={"safeguards_required_before_poc": "Validate first."},
        recommended_intervention_shape="Manual guide pilot.",
        evidence_ids=["ev-001"],
        summary="Proceed with safeguards.",
    )


def _cost_estimate(opportunity_id: str) -> CostEstimate:
    return CostEstimate(
        opportunity_id=opportunity_id,
        poc_type=POCType.PUBLIC_GUIDE,
        cost_estimate_usd="$0-$500",
        time_estimate="1-2 weeks",
        risk_level="low",
        first_3_actions=["Draft", "Review", "Publish"],
        notes="Use existing public information.",
    )


def test_opportunity_artifact_slug_uses_normalized_id() -> None:
    opportunity = _opportunity()

    assert opportunity_artifact_slug(opportunity) == "opp-senior-services-guide"
    assert opportunity_artifact_slug(opportunity.id) == "opp-senior-services-guide"


def test_all_opportunity_artifacts_share_normalized_directory(tmp_path: Path) -> None:
    opportunity = _opportunity()
    expected_dir = tmp_path / "opportunities" / "opp-senior-services-guide"

    opportunity_path = OpportunityAgent._persist_opportunity(opportunity, tmp_path)
    maker_path, _ = MakerAgent().save_output(_maker_result(opportunity.id), tmp_path)
    taker_path, _ = TakerAgent.save_output(_taker_output(opportunity.id), tmp_path)
    mediator_path, _ = MediatorAgent().save_output(
        _mediator_result(opportunity.id), tmp_path
    )
    cost_path, _ = CostCheckerAgent().write_artifacts(
        _cost_estimate(opportunity.id), tmp_path
    )

    assert opportunity_path.parent == expected_dir
    assert maker_path.parent == expected_dir
    assert taker_path.parent == expected_dir
    assert mediator_path.parent == expected_dir
    assert cost_path.parent == expected_dir
    assert {path.name for path in expected_dir.iterdir()} >= {
        "opportunity.yaml",
        "maker.json",
        "taker.json",
        "mediator.json",
        "cost.json",
    }

    loaded = ReportAgent()._discover_and_load(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].slug == "opp-senior-services-guide"
    assert loaded[0].opportunity_id == opportunity.id
    assert loaded[0].has_maker is True
    assert loaded[0].has_taker is True
    assert loaded[0].has_mediator is True
    assert loaded[0].has_cost is True
