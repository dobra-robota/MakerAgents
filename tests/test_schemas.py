import pytest
from pydantic import ValidationError

from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    Opportunity,
    OpportunityType,
    RunMetadata,
    ScoreSet,
    SourceType,
    Verdict,
)


def test_schema_validation_accepts_core_run_artifacts() -> None:
    run = RunMetadata(
        run_id="20260625-lodz-senior-citizens",
        city="Łodz",
        community="senior citizens",
    )
    evidence = EvidenceItem(
        id="EV-001",
        source_url="https://lodz.example.gov/senior-services",
        source_domain="lodz.example.gov",
        source_type=SourceType.GOVERNMENT,
        evidence_type=EvidenceType.OFFICIAL_STATEMENT,
        snippet="City page describes senior services and contact points.",
        language="en",
        claim_classification=ClaimClassification.EVIDENCE_BASED,
        trust_score=85,
        recency="current",
        confidence=Confidence.HIGH,
    )
    rank_score = ScoreSet.calculate_rank_score(
        people_helped_score=70,
        severity_score=80,
        validity_score=85,
        intervention_ease_score=60,
        harm_risk_score=20,
        ability_to_act_score=75,
    )
    scores = ScoreSet(
        validity_score=85,
        maker_score=76,
        maker_confidence=Confidence.MEDIUM,
        taker_score=24,
        taker_confidence=Confidence.LOW,
        people_helped_score=70,
        severity_score=80,
        impact_score=72,
        intervention_ease_score=60,
        harm_risk_score=20,
        ability_to_act_score=75,
        rank_score=rank_score,
    )
    opportunity = Opportunity(
        id="senior-services-guide",
        title="Plain-language senior services guide",
        type=OpportunityType.PUBLIC_GUIDE,
        pain_summary="Residents may struggle to find the right city service contact.",
        who_benefits=["senior citizens", "caregivers"],
        vulnerable_groups=["older adults"],
        evidence_ids=[evidence.id],
        speculative=False,
        scores=scores,
        verdict=Verdict.MANUAL_POC,
    )

    assert run.max_opportunities == 5
    assert run.queries_per_run == 10
    assert run.results_per_query == 5
    assert evidence.source_url.unicode_string() == "https://lodz.example.gov/senior-services"
    assert scores.rank_score == 75.3
    assert opportunity.evidence_ids == ["EV-001"]
    assert opportunity.type is OpportunityType.PUBLIC_GUIDE


def test_scores_reject_values_outside_zero_to_one_hundred() -> None:
    with pytest.raises(ValidationError):
        ScoreSet(
            validity_score=101,
            maker_score=76,
            maker_confidence=Confidence.MEDIUM,
            taker_score=24,
            taker_confidence=Confidence.LOW,
            people_helped_score=70,
            severity_score=80,
            impact_score=72,
            intervention_ease_score=60,
            harm_risk_score=20,
            ability_to_act_score=75,
            rank_score=75,
        )
