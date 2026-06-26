"""Tests for the Evidence Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from makeragents.agents.evidence import (
    ConflictResult,
    EvidenceAgent,
    _classify_claim,
    _classify_evidence_type,
    _classify_source_type,
    _estimate_confidence,
    _extract_domain,
    _jaccard_similarity,
    _normalize_snippet,
)
from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    SourceType,
)
from makeragents.search.providers import SearchResult
from makeragents.sources.registry import SourceRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(url: str, snippet: str) -> SearchResult:
    return SearchResult(title="Test", url=url, snippet=snippet)


def _make_evidence_item(
    id_: str = "EVID-00000001-0001",
    url: str = "https://example.gov/report",
    domain: str = "example.gov",
    src_type: SourceType = SourceType.GOVERNMENT,
    e_type: EvidenceType = EvidenceType.OFFICIAL_STATEMENT,
    snippet: str = "Official report on community needs",
    trust: float = 85.0,
    confidence: Confidence = Confidence.HIGH,
) -> EvidenceItem:
    return EvidenceItem(
        id=id_,
        source_url=url,  # type: ignore[arg-type]
        source_domain=domain,
        source_type=src_type,
        evidence_type=e_type,
        snippet=snippet,
        language="en",
        claim_classification=ClaimClassification.EVIDENCE_BASED,
        trust_score=trust,  # type: ignore[arg-type]
        recency="recent",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestExtractDomain:
    def test_basic_domain(self) -> None:
        assert _extract_domain("https://example.gov/report") == "example.gov"

    def test_strips_www(self) -> None:
        assert _extract_domain("https://www.example.com/page") == "example.com"

    def test_no_scheme(self) -> None:
        assert _extract_domain("example.org/path") == "example.org"

    def test_lowercases(self) -> None:
        assert _extract_domain("HTTPS://Example.COM") == "example.com"


class TestNormalizeSnippet:
    def test_lowercases(self) -> None:
        assert _normalize_snippet("Hello WORLD") == "hello world"

    def test_strips_punctuation(self) -> None:
        result = _normalize_snippet("Hello, world! How's it going?")
        assert result == "hello world hows it going"

    def test_collapses_whitespace(self) -> None:
        assert _normalize_snippet("  too   many   spaces  ") == "too many spaces"


class TestJaccardSimilarity:
    def test_identical(self) -> None:
        assert _jaccard_similarity("a b c", "a b c") == 1.0

    def test_disjoint(self) -> None:
        assert _jaccard_similarity("a b c", "d e f") == 0.0

    def test_partial_overlap(self) -> None:
        result = _jaccard_similarity("a b c", "b c d")
        assert 0.0 < result < 1.0

    def test_empty(self) -> None:
        assert _jaccard_similarity("", "") == 0.0


class TestClassifyEvidenceType:
    def test_statistic(self) -> None:
        assert (
            _classify_evidence_type("According to a study, 45% of seniors", SourceType.ACADEMIC)
            == EvidenceType.STATISTIC
        )

    def test_complaint(self) -> None:
        assert (
            _classify_evidence_type("I complained about the terrible service", SourceType.REDDIT)
            == EvidenceType.COMPLAINT
        )

    def test_official_statement(self) -> None:
        assert (
            _classify_evidence_type("The ministry announced new regulations", SourceType.GOVERNMENT)
            == EvidenceType.OFFICIAL_STATEMENT
        )

    def test_news_report(self) -> None:
        assert (
            _classify_evidence_type("Reported by local news agency yesterday", SourceType.MAJOR_NEWS)
            == EvidenceType.NEWS_REPORT
        )

    def test_first_hand(self) -> None:
        assert (
            _classify_evidence_type("I experienced long wait times at the clinic", SourceType.FORUM)
            == EvidenceType.FIRST_HAND_ACCOUNT
        )

    def test_second_hand(self) -> None:
        assert (
            _classify_evidence_type("Someone told me the buses never run on time", SourceType.REDDIT)
            == EvidenceType.SECOND_HAND_ACCOUNT
        )

    def test_fallback_government(self) -> None:
        assert (
            _classify_evidence_type("Some generic text", SourceType.GOVERNMENT)
            == EvidenceType.OFFICIAL_STATEMENT
        )

    def test_fallback_news(self) -> None:
        assert (
            _classify_evidence_type("Some generic text", SourceType.MAJOR_NEWS)
            == EvidenceType.NEWS_REPORT
        )

    def test_fallback_forum(self) -> None:
        assert (
            _classify_evidence_type("Some generic text", SourceType.FORUM)
            == EvidenceType.CLAIM
        )

    def test_fallback_unknown(self) -> None:
        assert (
            _classify_evidence_type("Some generic text", SourceType.UNKNOWN)
            == EvidenceType.UNKNOWN
        )


class TestClassifySourceType:
    def test_government(self) -> None:
        assert _classify_source_type("city.gov") == SourceType.GOVERNMENT

    def test_academic(self) -> None:
        assert _classify_source_type("university.edu") == SourceType.ACADEMIC

    def test_reddit(self) -> None:
        assert _classify_source_type("reddit.com") == SourceType.REDDIT

    def test_major_news(self) -> None:
        assert _classify_source_type("bbc.com") == SourceType.MAJOR_NEWS

    def test_unknown(self) -> None:
        assert _classify_source_type("random-website.info") == SourceType.UNKNOWN


class TestClassifyClaim:
    def test_statistic_is_evidence_based(self) -> None:
        # STATISTIC with trust >= 50 returns EVIDENCE_BASED
        assert _classify_claim(EvidenceType.STATISTIC, 60.0) == ClaimClassification.EVIDENCE_BASED

    def test_statistic_low_trust_falls_through(self) -> None:
        # STATISTIC with trust < 50 falls through to trust-based thresholds
        assert _classify_claim(EvidenceType.STATISTIC, 40.0) == ClaimClassification.ASSUMPTION

    def test_high_trust_is_evidence_based(self) -> None:
        assert _classify_claim(EvidenceType.CLAIM, 70.0) == ClaimClassification.EVIDENCE_BASED

    def test_medium_trust_is_inference(self) -> None:
        assert _classify_claim(EvidenceType.CLAIM, 50.0) == ClaimClassification.INFERENCE

    def test_low_trust_is_assumption(self) -> None:
        assert _classify_claim(EvidenceType.CLAIM, 30.0) == ClaimClassification.ASSUMPTION

    def test_very_low_trust_is_unknown(self) -> None:
        assert _classify_claim(EvidenceType.CLAIM, 20.0) == ClaimClassification.UNKNOWN


class TestEstimateConfidence:
    def test_high_confidence(self) -> None:
        assert _estimate_confidence(85.0, EvidenceType.STATISTIC) == Confidence.HIGH

    def test_medium_confidence(self) -> None:
        assert _estimate_confidence(55.0, EvidenceType.CLAIM) == Confidence.MEDIUM

    def test_low_confidence(self) -> None:
        assert _estimate_confidence(20.0, EvidenceType.UNKNOWN) == Confidence.LOW


# ---------------------------------------------------------------------------
# Integration tests for EvidenceAgent
# ---------------------------------------------------------------------------


class TestEvidenceAgentProcess:
    @pytest.fixture
    def registry(self) -> SourceRegistry:
        return SourceRegistry()

    @pytest.fixture
    def agent(self, registry: SourceRegistry) -> EvidenceAgent:
        return EvidenceAgent(registry=registry)

    def test_process_assigns_ids(self, agent: EvidenceAgent) -> None:
        results = [
            _make_search_result("https://example.gov/a", "Official policy on housing"),
            _make_search_result("https://nytimes.com/b", "News report on housing crisis"),
        ]
        items = agent.process(results, run_id="20250101-testrun")
        assert len(items) == 2
        assert items[0].id == "EVID-20250101-0001"
        assert items[1].id == "EVID-20250101-0002"

    def test_process_deduplicates_exact_url(self, agent: EvidenceAgent) -> None:
        results = [
            _make_search_result("https://example.gov/a", "Same URL twice"),
            _make_search_result("https://example.gov/a", "Same URL twice"),
        ]
        items = agent.process(results)
        assert len(items) == 1

    def test_process_deduplicates_near_duplicate_snippets(self, agent: EvidenceAgent) -> None:
        results = [
            _make_search_result("https://a.com/1", "The housing crisis in Łodz is severe"),
            _make_search_result("https://b.com/2", "The housing crisis in lodz is severe!"),
        ]
        items = agent.process(results)
        assert len(items) == 1

    def test_process_classifies_types(self, agent: EvidenceAgent) -> None:
        results = [
            _make_search_result("https://city.gov/report", "The ministry announced new funding"),
            _make_search_result("https://reddit.com/r/lodz", "I complained about the buses"),
        ]
        items = agent.process(results)
        types = {i.evidence_type for i in items}
        assert EvidenceType.OFFICIAL_STATEMENT in types
        assert EvidenceType.COMPLAINT in types

    def test_process_applies_trust_scores(self, agent: EvidenceAgent) -> None:
        results = [_make_search_result("https://example.gov/doc", "Government report")]
        items = agent.process(results)
        # .gov domain gets government source type score (85 by default)
        assert items[0].trust_score >= 80.0

    def test_process_empty(self, agent: EvidenceAgent) -> None:
        assert agent.process([]) == []


class TestEvidenceAgentValidityScore:
    @pytest.fixture
    def agent(self) -> EvidenceAgent:
        return EvidenceAgent()

    def test_empty_returns_zero(self, agent: EvidenceAgent) -> None:
        assert agent.calculate_validity_score([]) == 0.0

    def test_single_item(self, agent: EvidenceAgent) -> None:
        items = [_make_evidence_item(trust=80.0)]
        assert agent.calculate_validity_score(items) == 80.0

    def test_weighted_average(self, agent: EvidenceAgent) -> None:
        items = [
            _make_evidence_item(trust=90.0, confidence=Confidence.HIGH, domain="a.gov"),
            _make_evidence_item(trust=50.0, confidence=Confidence.LOW, domain="b.org"),
        ]
        score = agent.calculate_validity_score(items)
        # Two domains get diversity bonus (+10), then low-confidence penalty applies
        assert 45.0 < score < 75.0

    def test_diversity_bonus(self, agent: EvidenceAgent) -> None:
        items = [
            _make_evidence_item(domain="site-a.gov", trust=70.0),
            _make_evidence_item(domain="site-b.org", trust=70.0),
        ]
        score = agent.calculate_validity_score(items)
        # Two distinct domains = diversity bonus of +10, avg=70, result ~80
        assert score >= 75.0

    def test_low_confidence_penalty(self, agent: EvidenceAgent) -> None:
        items = [
            _make_evidence_item(domain="a.gov", trust=80.0, confidence=Confidence.LOW),
            _make_evidence_item(domain="b.org", trust=80.0, confidence=Confidence.LOW),
            _make_evidence_item(domain="c.edu", trust=80.0, confidence=Confidence.HIGH),
        ]
        score = agent.calculate_validity_score(items)
        # Should be penalized for multiple low-confidence items
        assert score < 80.0


class TestEvidenceAgentConflictDetection:
    @pytest.fixture
    def agent(self) -> EvidenceAgent:
        return EvidenceAgent()

    def test_no_conflict_when_only_official(self, agent: EvidenceAgent) -> None:
        items = [
            _make_evidence_item(
                id_="EVID-001", url="https://city.gov/r", domain="city.gov",
                src_type=SourceType.GOVERNMENT, trust=85.0,
            ),
        ]
        result = agent.detect_conflicts(items)
        assert not result.has_conflict

    def test_conflict_when_large_trust_gap(self, agent: EvidenceAgent) -> None:
        items = [
            _make_evidence_item(
                id_="EVID-001", url="https://city.gov/r", domain="city.gov",
                src_type=SourceType.GOVERNMENT, trust=85.0,
            ),
            _make_evidence_item(
                id_="EVID-002", url="https://reddit.com/r/lodz", domain="reddit.com",
                src_type=SourceType.REDDIT, trust=20.0,
            ),
        ]
        result = agent.detect_conflicts(items)
        assert result.has_conflict
        assert "city.gov" in result.official_domains
        assert "reddit.com" in result.community_domains

    def test_no_conflict_when_small_gap(self, agent: EvidenceAgent) -> None:
        items = [
            _make_evidence_item(
                id_="EVID-001", url="https://city.gov/r", domain="city.gov",
                src_type=SourceType.GOVERNMENT, trust=60.0,
            ),
            _make_evidence_item(
                id_="EVID-002", url="https://reddit.com/r/lodz", domain="reddit.com",
                src_type=SourceType.REDDIT, trust=45.0,
            ),
        ]
        result = agent.detect_conflicts(items)
        assert not result.has_conflict


class TestEvidenceAgentSave:
    @pytest.fixture
    def agent(self) -> EvidenceAgent:
        return EvidenceAgent()

    def test_save_creates_file(self, agent: EvidenceAgent, tmp_path: Path) -> None:
        items = [_make_evidence_item(id_="EVID-001")]
        dest = agent.save_evidence(items, tmp_path)
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert len(data) == 1
        assert data[0]["id"] == "EVID-001"


class TestConflictResult:
    def test_to_dict(self) -> None:
        cr = ConflictResult(
            has_conflict=True,
            official_domains=["gov.pl"],
            community_domains=["reddit.com"],
            conflicting_evidence_ids=["EVID-001", "EVID-002"],
            description="Tension detected",
        )
        d = cr.model_dump(mode="json")
        assert d["has_conflict"] is True
        assert d["official_domains"] == ["gov.pl"]
        assert d["conflicting_evidence_ids"] == ["EVID-001", "EVID-002"]

    def test_default_no_conflict(self) -> None:
        cr = ConflictResult()
        assert not cr.has_conflict
        assert cr.model_dump(mode="json")["has_conflict"] is False
