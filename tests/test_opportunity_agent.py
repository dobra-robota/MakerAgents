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
