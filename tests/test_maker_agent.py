"""Tests for the Maker Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic import HttpUrl

from makeragents.agents.maker import MakerAgent, MakerResult
from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    Opportunity,
    OpportunityType,
    SourceType,
)


def _make_evidence(
    id_: str = "EVID-001",
    trust: float = 80.0,
    confidence: Confidence = Confidence.HIGH,
    domain: str = "example.gov",
    e_type: EvidenceType = EvidenceType.OFFICIAL_STATEMENT,
) -> EvidenceItem:
    return EvidenceItem(
        id=id_,
        source_url=HttpUrl(f"https://{domain}/report"),
        source_domain=domain,
        source_type=SourceType.GOVERNMENT,
        evidence_type=e_type,
        snippet="Test evidence snippet for scoring",
        language="en",
        claim_classification=ClaimClassification.EVIDENCE_BASED,
        trust_score=trust,
        recency="recent",
        confidence=confidence,
    )


def _make_opportunity(
    id_: str = "OPP-test",
    title: str = "Test Opportunity",
    o_type: OpportunityType = OpportunityType.PUBLIC_GUIDE,
    pain: str = "Community pain point",
    beneficiaries: list[str] | None = None,
    vulnerable: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    speculative: bool = False,
) -> Opportunity:
    return Opportunity(
        id=id_,
        title=title,
        type=o_type,
        pain_summary=pain,
        who_benefits=beneficiaries or ["residents"],
        vulnerable_groups=vulnerable or [],
        evidence_ids=evidence_ids or [],
        speculative=speculative,
    )


class TestMakerAgentRun:
    @pytest.fixture
    def agent(self) -> MakerAgent:
        return MakerAgent()

    def test_produces_scores_in_range(self, agent: MakerAgent) -> None:
        opp = _make_opportunity()
        evidence = [_make_evidence()]
        result = agent.run(opp, evidence)
        assert 0.0 <= result.maker_score <= 100.0
        assert 0.0 <= result.people_helped_score <= 100.0
        assert 0.0 <= result.severity_score <= 100.0
        assert 0.0 <= result.impact_score <= 100.0
        assert 0.0 <= result.validity_score <= 100.0
        assert 0.0 <= result.intervention_ease_score <= 100.0
        assert 0.0 <= result.harm_risk_score <= 100.0
        assert 0.0 <= result.ability_to_act_score <= 100.0
        assert 0.0 <= result.rank_score <= 100.0

    def test_cites_evidence_ids(self, agent: MakerAgent) -> None:
        opp = _make_opportunity(evidence_ids=["EVID-001", "EVID-002"])
        evidence = [
            _make_evidence(id_="EVID-001"),
            _make_evidence(id_="EVID-002"),
        ]
        result = agent.run(opp, evidence)
        assert set(result.evidence_ids) == {"EVID-001", "EVID-002"}

    def test_returns_empty_when_no_evidence_ids_specified(self, agent: MakerAgent) -> None:
        opp = _make_opportunity(evidence_ids=[])
        evidence = [_make_evidence(id_="EVID-001"), _make_evidence(id_="EVID-002")]
        result = agent.run(opp, evidence)
        # Empty evidence_ids means no evidence was linked — return empty
        assert len(result.evidence_ids) == 0

    def test_returns_empty_for_nonexistent_evidence_ids(self, agent: MakerAgent) -> None:
        opp = _make_opportunity(evidence_ids=["NONEXISTENT"])
        evidence = [_make_evidence(id_="EVID-001"), _make_evidence(id_="EVID-002")]
        result = agent.run(opp, evidence)
        assert len(result.evidence_ids) == 0

    def test_claim_classifications_recorded(self, agent: MakerAgent) -> None:
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run(opp, evidence)
        assert "EVID-001" in result.claim_classifications
        assert result.claim_classifications["EVID-001"] == "evidence_based"

    def test_higher_trust_yields_higher_validity(self, agent: MakerAgent) -> None:
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        low = agent.run(opp, [_make_evidence(id_="EVID-001", trust=30.0)])
        high = agent.run(opp, [_make_evidence(id_="EVID-001", trust=90.0)])
        assert high.validity_score > low.validity_score

    def test_speculative_reduces_scores(self, agent: MakerAgent) -> None:
        opp_norm = _make_opportunity(speculative=False)
        opp_spec = _make_opportunity(speculative=True)
        evidence = [_make_evidence()]
        normal = agent.run(opp_norm, evidence)
        speculative = agent.run(opp_spec, evidence)
        # Speculative should have lower maker_score
        assert speculative.maker_score <= normal.maker_score

    def test_vulnerable_groups_increase_severity_and_harm(self, agent: MakerAgent) -> None:
        opp = _make_opportunity(vulnerable=["elderly", "disabled"])
        evidence = [_make_evidence()]
        result = agent.run(opp, evidence)
        assert result.severity_score > 20.0  # baseline gives some
        assert result.harm_risk_score >= 0.0

    def test_empty_evidence_low_confidence(self, agent: MakerAgent) -> None:
        opp = _make_opportunity()
        result = agent.run(opp, [])
        assert result.maker_confidence == Confidence.LOW

    def test_rank_score_computed(self, agent: MakerAgent) -> None:
        opp = _make_opportunity()
        result = agent.run(opp, [_make_evidence()])
        assert result.rank_score > 0.0
        # Verify it matches the documented formula
        expected = (
            result.people_helped_score * 0.22
            + result.severity_score * 0.20
            + result.validity_score * 0.18
            + result.intervention_ease_score * 0.14
            + (100 - result.harm_risk_score) * 0.14
            + result.ability_to_act_score * 0.12
        )
        assert abs(result.rank_score - round(expected, 2)) < 0.1


class TestMakerAgentSaveOutput:
    @pytest.fixture
    def agent(self) -> MakerAgent:
        return MakerAgent()

    def test_writes_json_and_md(self, agent: MakerAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        evidence = [_make_evidence()]
        result = agent.run(opp, evidence)
        json_path, md_path = agent.save_output(result, tmp_path)
        assert json_path.exists()
        assert md_path.exists()

    def test_json_is_valid(self, agent: MakerAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp, [_make_evidence()])
        json_path, _ = agent.save_output(result, tmp_path)
        data = json.loads(json_path.read_text())
        assert data["opportunity_id"] == "OPP-test"
        assert "maker_score" in data

    def test_md_contains_opportunity_id(self, agent: MakerAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp, [_make_evidence()])
        _, md_path = agent.save_output(result, tmp_path)
        content = md_path.read_text()
        assert "OPP-test" in content


class TestMakerResult:
    def test_to_json_dict(self) -> None:
        result = MakerResult(
            opportunity_id="OPP-1",
            maker_score=75.0,
            maker_confidence=Confidence.HIGH,
            people_helped_score=70.0,
            severity_score=60.0,
            impact_score=65.0,
            validity_score=80.0,
            intervention_ease_score=50.0,
            harm_risk_score=20.0,
            ability_to_act_score=55.0,
            rank_score=62.5,
            evidence_ids=["EVID-001"],
            summary="Test",
        )
        d = result.to_json_dict()
        assert d["opportunity_id"] == "OPP-1"
        assert d["maker_score"] == 75.0
        assert d["maker_confidence"] == "high"
