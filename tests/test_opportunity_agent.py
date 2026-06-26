"""Tests for the Opportunity Agent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from makeragents.agents.opportunity import (
    OpportunityAgent,
    _derive_beneficiaries,
    _derive_title,
    _map_evidence_to_opportunity_type,
    _OPPORTUNITY_TYPE_MAP,
    _same_theme,
)
from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    OpportunityType,
    SourceType,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def gov_evidence() -> EvidenceItem:
    """Single evidence item from a government source about senior services."""
    return EvidenceItem(
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


@pytest.fixture
def news_evidence() -> EvidenceItem:
    """Single evidence item from a news source about senior wait times."""
    return EvidenceItem(
        id="EV-002",
        source_url="https://lodz-news.example.com/wait-times",
        source_domain="lodz-news.example.com",
        source_type=SourceType.LOCAL_NEWS,
        evidence_type=EvidenceType.NEWS_REPORT,
        snippet="Report documents long wait times for senior services in Lodz.",
        language="en",
        claim_classification=ClaimClassification.EVIDENCE_BASED,
        trust_score=60,
        recency="current",
        confidence=Confidence.HIGH,
    )


@pytest.fixture
def forum_evidence() -> EvidenceItem:
    """Single evidence item from a forum about senior transport issues."""
    return EvidenceItem(
        id="EV-003",
        source_url="https://forum.lodz.example.com/transport",
        source_domain="forum.lodz.example.com",
        source_type=SourceType.FORUM,
        evidence_type=EvidenceType.COMPLAINT,
        snippet="Forum discusses lack of accessible transport for elderly residents.",
        language="en",
        claim_classification=ClaimClassification.INFERENCE,
        trust_score=40,
        recency="recent",
        confidence=Confidence.MEDIUM,
    )


@pytest.fixture
def same_domain_evidence(gov_evidence: EvidenceItem) -> EvidenceItem:
    """A second evidence item on the same government domain."""
    return EvidenceItem(
        id="EV-004",
        source_url="https://lodz.example.gov/benefits",
        source_domain="lodz.example.gov",
        source_type=SourceType.GOVERNMENT,
        evidence_type=EvidenceType.OFFICIAL_STATEMENT,
        snippet="City releases new benefit programme for senior citizens.",
        language="en",
        claim_classification=ClaimClassification.EVIDENCE_BASED,
        trust_score=85,
        recency="current",
        confidence=Confidence.HIGH,
    )


@pytest.fixture
def varied_evidence(
    gov_evidence: EvidenceItem,
    news_evidence: EvidenceItem,
    forum_evidence: EvidenceItem,
) -> list[EvidenceItem]:
    """Mixed evidence from different sources and domains."""
    return [gov_evidence, news_evidence, forum_evidence]


# ------------------------------------------------------------------
# Unit tests: _same_theme
# ------------------------------------------------------------------


class TestSameTheme:
    def test_same_domain_is_true(self, gov_evidence: EvidenceItem) -> None:
        same = EvidenceItem(
            id="EV-OTHER",
            source_url="https://lodz.example.gov/other",
            source_domain="lodz.example.gov",
            source_type=SourceType.GOVERNMENT,
            evidence_type=EvidenceType.OFFICIAL_STATEMENT,
            snippet="Another page on the same domain.",
            language="en",
            claim_classification=ClaimClassification.EVIDENCE_BASED,
            trust_score=80,
            recency="current",
            confidence=Confidence.HIGH,
        )
        assert _same_theme(gov_evidence, same) is True

    def test_different_domain_is_false(
        self, gov_evidence: EvidenceItem, news_evidence: EvidenceItem
    ) -> None:
        assert _same_theme(gov_evidence, news_evidence) is False


# ------------------------------------------------------------------
# Unit tests: _map_evidence_to_opportunity_type
# ------------------------------------------------------------------


class TestMapEvidenceType:
    def test_maps_known_types(self) -> None:
        assert (
            _map_evidence_to_opportunity_type("complaint")
            is OpportunityType.ADVOCACY_REPORT
        )
        assert (
            _map_evidence_to_opportunity_type("official_statement")
            is OpportunityType.PUBLIC_GUIDE
        )
        assert (
            _map_evidence_to_opportunity_type("statistic")
            is OpportunityType.OPEN_DATA_RESOURCE
        )

    def test_falls_back_to_default_for_unknown(self) -> None:
        assert (
            _map_evidence_to_opportunity_type("unknown")
            is OpportunityType.COMMUNITY_SUPPORT_PROCESS
        )

    def test_every_evidence_type_in_map(self) -> None:
        """All EvidenceType values should have a mapping."""
        for ev_type in EvidenceType:
            mapped = _map_evidence_to_opportunity_type(ev_type.value)
            assert isinstance(mapped, OpportunityType)


# ------------------------------------------------------------------
# Unit tests: _derive_beneficiaries
# ------------------------------------------------------------------


class TestDeriveBeneficiaries:
    def test_finds_keyword_matches(self, gov_evidence: EvidenceItem) -> None:
        ev = gov_evidence.model_copy(
            update={"snippet": "Guide for senior citizens and their family members."}
        )
        result = _derive_beneficiaries([ev])
        assert "senior citizens" in result
        assert "families" in result

    def test_falls_back_when_no_keywords_match(self, news_evidence: EvidenceItem) -> None:
        ev = news_evidence.model_copy(
            update={"snippet": "General information about city infrastructure."}
        )
        result = _derive_beneficiaries([ev])
        assert result == ["community members"]


# ------------------------------------------------------------------
# Unit tests: _derive_title
# ------------------------------------------------------------------


class TestDeriveTitle:
    def test_uses_snippet_when_long_enough(self) -> None:
        snippet = "Long wait times for senior services"
        opp_type = OpportunityType.ADVOCACY_REPORT
        ev = _make_ev("EV-TEST", snippet)
        title = _derive_title(opp_type, [ev])
        assert title.startswith("Advocacy report:")
        assert "wait" in title

    def test_falls_back_to_generic_title_for_short_snippet(self) -> None:
        ev = _make_ev("EV-TEST", "Short.")
        title = _derive_title(OpportunityType.PUBLIC_GUIDE, [ev])
        assert title == "Public information guide"

    def test_falls_back_for_unknown_type(self) -> None:
        ev = _make_ev("EV-TEST", "Short.")
        title = _derive_title(OpportunityType.COORDINATION_PROCESS, [ev])
        assert title == "Coordination process"

    def test_title_does_not_exceed_max_length(self) -> None:
        """Very long snippet should produce a title capped at 120 characters."""
        long_snippet = "A " + "very " * 30 + "long snippet that would produce an excessively verbose title exceeding the maximum allowed length"
        ev = _make_ev("EV-LONG", long_snippet)
        title = _derive_title(OpportunityType.PUBLIC_GUIDE, [ev])
        assert len(title) <= 120


# ------------------------------------------------------------------
# Integration tests: OpportunityAgent.process
# ------------------------------------------------------------------


class TestOpportunityAgentProcess:
    def test_generates_opportunities_from_evidence(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        agent = OpportunityAgent()
        opportunities = agent.process(varied_evidence, tmp_path)

        assert len(opportunities) >= 1
        for opp in opportunities:
            assert opp.id.startswith("OPP-")
            assert opp.title
            assert opp.pain_summary
            assert opp.who_benefits
            assert isinstance(opp.speculative, bool)

    def test_speculative_flag_for_single_evidence(
        self, gov_evidence: EvidenceItem, tmp_path: Path
    ) -> None:
        agent = OpportunityAgent()
        opportunities = agent.process([gov_evidence], tmp_path)

        assert len(opportunities) == 1
        assert opportunities[0].speculative is True

    def test_multiple_evidence_same_domain_is_not_speculative(
        self,
        gov_evidence: EvidenceItem,
        same_domain_evidence: EvidenceItem,
        tmp_path: Path,
    ) -> None:
        """Two items from the same domain group together and become non-speculative."""
        agent = OpportunityAgent()
        evidence_list = [gov_evidence, same_domain_evidence]
        opportunities = agent.process(evidence_list, tmp_path)

        # Both have the same domain, so they group into one opportunity.
        assert len(opportunities) == 1
        assert opportunities[0].speculative is False
        assert len(opportunities[0].evidence_ids) >= 2

    def test_respects_max_opportunities(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        agent = OpportunityAgent(max_opportunities=1)
        opportunities = agent.process(varied_evidence, tmp_path)

        assert len(opportunities) <= 1

    def test_respects_lower_max_opportunities(
        self, tmp_path: Path
    ) -> None:
        """Use many evidence items from different domains to produce multiple groups."""
        items = [
            _make_ev("EV-A", "Alpha domain content.", "alpha.com"),
            _make_ev("EV-B", "Beta domain content.", "beta.com"),
            _make_ev("EV-C", "Gamma domain content.", "gamma.com"),
        ]
        agent = OpportunityAgent(max_opportunities=2)
        opportunities = agent.process(items, tmp_path)

        assert len(opportunities) == 2

    def test_writes_opportunity_yaml_to_run_folder(
        self, gov_evidence: EvidenceItem, tmp_path: Path
    ) -> None:
        agent = OpportunityAgent()
        agent.process([gov_evidence], tmp_path)

        # Check that opportunity.yaml was written somewhere under opportunities/.
        opp_dir = tmp_path / "opportunities"
        assert opp_dir.is_dir()

        yaml_files = list(opp_dir.rglob("opportunity.yaml"))
        assert len(yaml_files) == 1

        parsed = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8"))
        assert parsed["id"].startswith("OPP-")
        assert parsed["speculative"] is True

    def test_multiple_opportunities_each_have_own_folder(
        self, tmp_path: Path
    ) -> None:
        """Different-domain items produce separate folders."""
        items = [
            _make_ev("EV-A", "Alpha domain content.", "alpha.com"),
            _make_ev("EV-B", "Beta domain content.", "beta.com"),
        ]
        agent = OpportunityAgent()
        agent.process(items, tmp_path)

        opp_dir = tmp_path / "opportunities"
        folders = [d for d in opp_dir.iterdir() if d.is_dir()]
        assert len(folders) == 2
        for folder in folders:
            yaml_path = folder / "opportunity.yaml"
            assert yaml_path.is_file()

    def test_empty_evidence_returns_empty_list(self, tmp_path: Path) -> None:
        agent = OpportunityAgent()
        opportunities = agent.process([], tmp_path)
        assert opportunities == []

    def test_default_max_opportunities_is_five(self) -> None:
        agent = OpportunityAgent()
        assert agent.max_opportunities == 5

    def test_evidence_ids_are_preserved_in_opportunity(
        self,
        gov_evidence: EvidenceItem,
        same_domain_evidence: EvidenceItem,
        tmp_path: Path,
    ) -> None:
        agent = OpportunityAgent()
        evidence_list = [gov_evidence, same_domain_evidence]
        opportunities = agent.process(evidence_list, tmp_path)

        assert len(opportunities) == 1
        opp = opportunities[0]
        assert gov_evidence.id in opp.evidence_ids
        assert same_domain_evidence.id in opp.evidence_ids

    def test_opportunity_type_reflects_evidence_type(
        self, tmp_path: Path
    ) -> None:
        """Official statement evidence should produce PUBLIC_GUIDE opportunity type."""
        ev = _make_ev(
            "EV-TYPE",
            "Official city statement about senior services.",
            domain="lodz.example.gov",
            evidence_type=EvidenceType.OFFICIAL_STATEMENT,
        )
        agent = OpportunityAgent()
        opportunities = agent.process([ev], tmp_path)
        assert len(opportunities) == 1
        assert opportunities[0].type is OpportunityType.PUBLIC_GUIDE


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_ev(
    ev_id: str,
    snippet: str,
    domain: str = "example.com",
    evidence_type: EvidenceType = EvidenceType.CLAIM,
) -> EvidenceItem:
    return EvidenceItem(
        id=ev_id,
        source_url=f"https://{domain}/page",
        source_domain=domain,
        source_type=SourceType.UNKNOWN,
        evidence_type=evidence_type,
        snippet=snippet,
        language="en",
        claim_classification=ClaimClassification.UNKNOWN,
        trust_score=50,
        recency="unknown",
        confidence=Confidence.LOW,
    )


# ------------------------------------------------------------------
# LLM-backed tests (mocked, no live calls)
# ------------------------------------------------------------------

import json

import httpx

from makeragents.config import AppConfig
from makeragents.llm import ChatMessage, LLMClient


# --- Mock helpers ---

_LLM_JSON_RESPONSE = {
    "opportunities": [
        {
            "id": "OPP-LLM-001",
            "title": "Better senior transport information",
            "type": "public_guide",
            "pain_summary": "Seniors struggle to find accessible transport options in Lodz.",
            "who_benefits": ["senior citizens", "family caregivers"],
            "vulnerable_groups": ["senior citizens", "people with disabilities"],
            "evidence_ids": ["EV-001", "EV-003"],
            "speculative": False,
        },
        {
            "id": "OPP-LLM-002",
            "title": "Reduce clinic wait times",
            "type": "advocacy_report",
            "pain_summary": "Long wait times at public clinics affect patient health.",
            "who_benefits": ["patients", "healthcare workers"],
            "vulnerable_groups": ["low-income households"],
            "evidence_ids": ["EV-002"],
            "speculative": True,
        },
    ]
}


def _mock_llm_response(
    payload: dict | None = None,
) -> tuple[LLMClient, httpx.MockTransport]:
    """Build an LLMClient backed by a mock HTTP transport."""
    payload = payload if payload is not None else _LLM_JSON_RESPONSE

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": "deepseek-chat",
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    cfg = AppConfig(deepseek_api_key="test-key")
    llm = LLMClient(config=cfg, http_client=client)
    return llm, transport


def _mock_error_client() -> LLMClient:
    """Build an LLMClient that always returns HTTP 500."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"Internal Server Error")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    cfg = AppConfig(deepseek_api_key="test-key")
    return LLMClient(config=cfg, http_client=client)


class TestLLMOpportunityAgent:
    """Integration-style tests with a mocked LLM backend."""

    def test_llm_derives_opportunities(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """LLM-backed path produces candidate opportunities with beneficiaries and type."""
        llm, _transport = _mock_llm_response()
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process(varied_evidence, tmp_path)

        assert len(opportunities) >= 1
        for opp in opportunities:
            assert opp.id.startswith("OPP-")
            assert opp.title
            assert opp.pain_summary
            assert len(opp.who_benefits) >= 1
            assert isinstance(opp.type, OpportunityType)

    def test_llm_opportunities_have_speculative_flag(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """LLM opportunities respect the speculative flag from LLM output."""
        llm, _transport = _mock_llm_response()
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process(varied_evidence, tmp_path)

        # At least one opportunity should be speculative (only 1 evidence ID).
        speculative_opps = [o for o in opportunities if o.speculative]
        assert len(speculative_opps) >= 1

    def test_llm_single_evidence_becomes_speculative(
        self, gov_evidence: EvidenceItem, tmp_path: Path
    ) -> None:
        """Opportunity backed by <2 evidence IDs is forced speculative."""
        single_source_payload = {
            "opportunities": [
                {
                    "id": "OPP-LLM-001",
                    "title": "Single-source opportunity",
                    "type": "public_guide",
                    "pain_summary": "Derived from one evidence item.",
                    "who_benefits": ["residents"],
                    "vulnerable_groups": [],
                    "evidence_ids": ["EV-001"],
                    "speculative": False,
                }
            ]
        }
        llm, _transport = _mock_llm_response(single_source_payload)
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process([gov_evidence], tmp_path)

        assert len(opportunities) == 1
        assert opportunities[0].speculative is True
        assert "unknown — speculative opportunity" in opportunities[0].vulnerable_groups

    def test_llm_enforces_max_opportunities(
        self, tmp_path: Path
    ) -> None:
        """LLM path respects max_opportunities limit."""
        many_opps_payload = {
            "opportunities": [
                {
                    "id": f"OPP-LLM-{i:03d}",
                    "title": f"Opportunity {i}",
                    "type": "public_guide",
                    "pain_summary": f"Pain point {i}.",
                    "who_benefits": ["residents"],
                    "vulnerable_groups": [],
                    "evidence_ids": ["EV-A", "EV-B"],
                    "speculative": False,
                }
                for i in range(1, 8)
            ]
        }
        llm, _transport = _mock_llm_response(many_opps_payload)
        agent = OpportunityAgent(
            max_opportunities=3,
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        items = [
            _make_ev("EV-A", "Alpha.", "alpha.com"),
            _make_ev("EV-B", "Beta.", "beta.com"),
        ]
        opportunities = agent.process(items, tmp_path)
        assert len(opportunities) == 3

    def test_llm_writes_opportunity_yaml(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """LLM-backed opportunities are persisted as YAML on disk."""
        llm, _transport = _mock_llm_response()
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        agent.process(varied_evidence, tmp_path)

        opp_dir = tmp_path / "opportunities"
        assert opp_dir.is_dir()
        yaml_files = list(opp_dir.rglob("opportunity.yaml"))
        assert len(yaml_files) >= 1

        for yf in yaml_files:
            parsed = yaml.safe_load(yf.read_text(encoding="utf-8"))
            assert "id" in parsed
            assert "title" in parsed
            assert "type" in parsed
            assert "who_benefits" in parsed
            assert "speculative" in parsed

    def test_llm_fallback_to_heuristic_when_no_client(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """When no LLM client is provided, agent falls back to heuristic path."""
        agent = OpportunityAgent(
            max_opportunities=3,
            # No llm_client passed — should fall back
        )
        opportunities = agent.process(varied_evidence, tmp_path)

        # Heuristic path groups by domain → 3 opportunities from 3 domains.
        assert len(opportunities) >= 1
        for opp in opportunities:
            assert opp.id.startswith("OPP-")

    def test_llm_fallback_on_api_error(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """LLM errors trigger graceful fallback to heuristic."""
        llm = _mock_error_client()
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process(varied_evidence, tmp_path)

        # Should fall back to heuristic clustering (≥1 opportunity).
        assert len(opportunities) >= 1
        for opp in opportunities:
            assert opp.id.startswith("OPP-")

    def test_llm_fallback_on_empty_response(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """Empty LLM opportunities list triggers fallback."""
        llm, _transport = _mock_llm_response({"opportunities": []})
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process(varied_evidence, tmp_path)

        assert len(opportunities) >= 1
        for opp in opportunities:
            assert opp.id.startswith("OPP-")

    def test_llm_missing_opportunities_key(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """Response without 'opportunities' key triggers fallback."""
        llm, _transport = _mock_llm_response({"unrelated": 42})
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process(varied_evidence, tmp_path)

        assert len(opportunities) >= 1
        for opp in opportunities:
            assert opp.id.startswith("OPP-")

    def test_llm_opportunity_with_unrecognised_type_is_skipped(
        self, varied_evidence: list[EvidenceItem], tmp_path: Path
    ) -> None:
        """Bad opportunity type from LLM is silently skipped."""
        bad_type_payload = {
            "opportunities": [
                {
                    "id": "OPP-BAD",
                    "title": "Bad type opp",
                    "type": "nonexistent_type_xyz",
                    "pain_summary": "Should be skipped.",
                    "who_benefits": ["nobody"],
                    "vulnerable_groups": [],
                    "evidence_ids": ["EV-001"],
                    "speculative": True,
                },
                {
                    "id": "OPP-GOOD",
                    "title": "Good opp",
                    "type": "public_guide",
                    "pain_summary": "Should be kept.",
                    "who_benefits": ["residents"],
                    "vulnerable_groups": [],
                    "evidence_ids": ["EV-001", "EV-002"],
                    "speculative": False,
                },
            ]
        }
        llm, _transport = _mock_llm_response(bad_type_payload)
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process(varied_evidence, tmp_path)

        # Only the valid one should be included.
        assert len(opportunities) == 1
        assert opportunities[0].title == "Good opp"

    def test_llm_invalid_evidence_ids_filtered(
        self, gov_evidence: EvidenceItem, tmp_path: Path
    ) -> None:
        """LLM-referenced evidence IDs that don't match input are dropped."""
        payload = {
            "opportunities": [
                {
                    "id": "OPP-LLM-001",
                    "title": "Test opp",
                    "type": "public_guide",
                    "pain_summary": "Testing evidence ID filtering.",
                    "who_benefits": ["residents"],
                    "vulnerable_groups": [],
                    "evidence_ids": ["EV-FAKE", "EV-001"],
                    "speculative": False,
                }
            ]
        }
        llm, _transport = _mock_llm_response(payload)
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process([gov_evidence], tmp_path)

        assert len(opportunities) == 1
        # Only EV-001 should survive.
        assert opportunities[0].evidence_ids == ["EV-001"]
        # Single valid evidence → speculative.
        assert opportunities[0].speculative is True

    def test_llm_fuzzy_type_mapping(
        self, gov_evidence: EvidenceItem, same_domain_evidence: EvidenceItem, tmp_path: Path
    ) -> None:
        """Fuzzy type names from LLM are mapped to correct OpportunityType."""
        payload = {
            "opportunities": [
                {
                    "id": "OPP-LLM-001",
                    "title": "Open data opp",
                    "type": "open data",
                    "pain_summary": "Data should be open.",
                    "who_benefits": ["citizens"],
                    "vulnerable_groups": [],
                    "evidence_ids": ["EV-001", "EV-004"],
                    "speculative": False,
                }
            ]
        }
        llm, _transport = _mock_llm_response(payload)
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process(
            [gov_evidence, same_domain_evidence], tmp_path
        )

        assert len(opportunities) == 1
        assert opportunities[0].type is OpportunityType.OPEN_DATA_RESOURCE

    def test_llm_empty_evidence_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """Empty evidence list returns [] even with LLM client."""
        llm, _transport = _mock_llm_response()
        agent = OpportunityAgent(
            llm_client=llm,
            city="Lodz",
            community="senior citizens",
        )
        opportunities = agent.process([], tmp_path)
        assert opportunities == []
