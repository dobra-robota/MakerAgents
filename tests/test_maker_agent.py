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


# ---------------------------------------------------------------------------
# LLM-backed run_with_llm tests (mocked, no live API calls)
# ---------------------------------------------------------------------------


import json as _json


def _llm_maker_response(
    *,
    score: float = 75.0,
    confidence: str = "medium",
    value_add_summary: str = "A well-designed intervention would add genuine value.",
    claims: list[dict] | None = None,
    evidence_ids: list[str] | None = None,
) -> dict:
    """Build a mock LLM JSON response matching the maker prompt's expected schema."""
    return {
        "value_add_summary": value_add_summary,
        "score": score,
        "confidence": confidence,
        "evidence_ids": evidence_ids or ["EVID-001"],
        "claims": claims or [
            {
                "text": "A specific observation about community pain",
                "classification": "evidence_based",
                "evidence_id": "EVID-001",
            },
            {
                "text": "An inference about root causes",
                "classification": "inference",
                "evidence_id": "EVID-001",
            },
        ],
    }


def _make_mock_llm_client(
    response: dict | None = None,
) -> "LLMClient":
    """Return an LLMClient with a mocked HTTP transport that returns the given JSON."""
    from makeragents.config import AppConfig
    from makeragents.llm import LLMClient

    import httpx

    payload = response or _llm_maker_response()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": _json.dumps(payload),
                        },
                    }
                ],
                "model": "deepseek-chat",
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    cfg = AppConfig(deepseek_api_key="test-key")
    return LLMClient(config=cfg, http_client=http_client)


class TestMakerAgentRunWithLLM:
    @pytest.fixture
    def agent(self) -> MakerAgent:
        return MakerAgent(llm_client=_make_mock_llm_client())

    @pytest.fixture
    def agent_no_llm(self) -> MakerAgent:
        """Agent without LLM — should fall back to deterministic."""
        return MakerAgent(llm_client=None)

    def test_produces_value_add_argument(self, agent: MakerAgent) -> None:
        """LLM-backed run produces a non-empty value_add_argument."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence, city="Łodz", community="seniors")
        assert result.value_add_argument != ""
        assert "well-designed intervention" in result.value_add_argument.lower()

    def test_produces_claims(self, agent: MakerAgent) -> None:
        """LLM-backed run produces validated claims."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        assert len(result.claims) >= 1
        assert result.claims[0]["classification"] in (
            "evidence_based", "inference", "assumption", "unknown",
        )

    def test_score_in_range(self, agent: MakerAgent) -> None:
        """Score from LLM stays within 0–100."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        assert 0.0 <= result.maker_score <= 100.0

    def test_confidence_from_llm(self, agent: MakerAgent) -> None:
        """Confidence from LLM JSON is used when valid."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        # Default mock returns "medium"
        assert result.maker_confidence == Confidence.MEDIUM

    def test_high_confidence_from_llm(self) -> None:
        """High confidence from LLM is parsed correctly."""
        llm = _make_mock_llm_client(_llm_maker_response(confidence="high"))
        agent = MakerAgent(llm_client=llm)
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        assert result.maker_confidence == Confidence.HIGH

    def test_invalid_confidence_falls_back(self) -> None:
        """Invalid confidence string falls back to deterministic confidence."""
        llm = _make_mock_llm_client(_llm_maker_response(confidence="bogus"))
        agent = MakerAgent(llm_client=llm)
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        # Deterministic with 1 HIGH evidence should yield HIGH confidence
        assert result.maker_confidence == Confidence.HIGH

    def test_filters_unknown_evidence_ids(self, agent: MakerAgent) -> None:
        """Evidence IDs not in the provided evidence list are filtered out."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        # LLM may cite EVID-001 but not EVID-999
        assert "EVID-999" not in result.evidence_ids

    def test_claims_include_evidence_id_and_classification(self, agent: MakerAgent) -> None:
        """Each claim dict has text, classification, and evidence_id keys."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        for claim in result.claims:
            assert "text" in claim
            assert "classification" in claim
            assert "evidence_id" in claim

    def test_invalid_claim_classification_maps_to_unknown(self) -> None:
        """Claims with invalid classifications are mapped to 'unknown'."""
        llm = _make_mock_llm_client(_llm_maker_response(
            claims=[
                {
                    "text": "Some claim",
                    "classification": "bogus_class",
                    "evidence_id": "EVID-001",
                },
            ],
        ))
        agent = MakerAgent(llm_client=llm)
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        assert result.claims[0]["classification"] == "unknown"

    def test_out_of_range_score_clamped(self) -> None:
        """LLM scores outside 0–100 are clamped."""
        llm = _make_mock_llm_client(_llm_maker_response(score=250.0))
        agent = MakerAgent(llm_client=llm)
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        assert result.maker_score == 100.0

    def test_negative_score_clamped(self) -> None:
        llm = _make_mock_llm_client(_llm_maker_response(score=-50.0))
        agent = MakerAgent(llm_client=llm)
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        assert result.maker_score == 0.0

    def test_falls_back_to_deterministic_without_llm(self, agent_no_llm: MakerAgent) -> None:
        """When no LLM client, run_with_llm returns deterministic result."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent_no_llm.run_with_llm(opp, evidence)
        assert result.value_add_argument == ""
        assert result.claims == []

    def test_falls_back_on_llm_error(self) -> None:
        """When LLM raises, fall back to deterministic result."""
        def handler(request):
            raise RuntimeError("simulated network failure")

        import httpx
        from makeragents.config import AppConfig
        from makeragents.llm import LLMClient

        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(transport=transport)
        cfg = AppConfig(deepseek_api_key="test-key")
        llm = LLMClient(config=cfg, http_client=http_client)
        agent = MakerAgent(llm_client=llm)

        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        # Should fall back to deterministic
        assert result.maker_score >= 0.0
        assert result.evidence_ids == ["EVID-001"]

    def test_value_add_in_markdown(self, agent: MakerAgent, tmp_path: Path) -> None:
        """maker.md includes the value-add argument section."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        _, md_path = agent.save_output(result, tmp_path)
        content = md_path.read_text()
        assert "## Value-Add Argument" in content
        assert "well-designed intervention" in content.lower()

    def test_claims_in_markdown(self, agent: MakerAgent, tmp_path: Path) -> None:
        """maker.md includes a claims table."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        _, md_path = agent.save_output(result, tmp_path)
        content = md_path.read_text()
        assert "## Claims" in content
        assert "evidence_based" in content

    def test_value_add_in_json(self, agent: MakerAgent, tmp_path: Path) -> None:
        """maker.json includes value_add_argument and claims."""
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run_with_llm(opp, evidence)
        json_path, _ = agent.save_output(result, tmp_path)
        data = _json.loads(json_path.read_text())
        assert "value_add_argument" in data
        assert data["value_add_argument"] != ""
        assert "claims" in data
        assert len(data["claims"]) >= 1

    def test_deterministic_still_works_as_before(self) -> None:
        """Existing run() method still works without LLM."""
        agent = MakerAgent()
        opp = _make_opportunity(evidence_ids=["EVID-001"])
        evidence = [_make_evidence(id_="EVID-001")]
        result = agent.run(opp, evidence)
        assert result.maker_score > 0.0
        assert result.value_add_argument == ""
        assert result.claims == []
