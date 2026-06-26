"""Evidence Agent: classifies, deduplicates, and scores research evidence.

Takes raw search results and produces validated :class:`EvidenceItem` entries
with trust scores, evidence type classification, and conflict detection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import Field

from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    MakerAgentsModel,
    ScoreValue,
    SourceType,
)
from makeragents.sources.registry import SourceRegistry, load_registry

if TYPE_CHECKING:
    from makeragents.search.providers import SearchResult

_MAX_SNIPPET_LENGTH = 500
_JACCARD_THRESHOLD = 0.75
_VALIDITY_DIVERSITY_BONUS = 10.0
_LOW_CONFIDENCE_PENALTY = 0.7

# Keywords that strongly hint at evidence types, ordered by priority.
_EVIDENCE_TYPE_HINTS: list[tuple[list[str], EvidenceType]] = [
    (["percent", "rate of", "according to a study", "survey found", "statistics show"], EvidenceType.STATISTIC),
    (["i complained", "my complaint", "filed a complaint", "terrible service", "so frustrating", "never again"], EvidenceType.COMPLAINT),
    (["according to the government", "official report", "ministry announced", "regulation states", "policy states"], EvidenceType.OFFICIAL_STATEMENT),
    (["reported by", "news agency", "breaking news", "according to sources", "press release"], EvidenceType.NEWS_REPORT),
    (["i experienced", "i went through", "my experience", "i saw", "in my case", "personally, i"], EvidenceType.FIRST_HAND_ACCOUNT),
    (["someone told me", "my friend said", "i heard that", "people say", "they told me", "a relative mentioned"], EvidenceType.SECOND_HAND_ACCOUNT),
]

_SOURCE_TYPE_DOMAIN_HINTS: list[tuple[str, SourceType]] = [
    (".gov", SourceType.GOVERNMENT),
    (".edu", SourceType.ACADEMIC),
    (".ac.", SourceType.ACADEMIC),
    ("reddit.com", SourceType.REDDIT),
    ("twitter.com", SourceType.ANONYMOUS_SOCIAL),
    ("x.com", SourceType.ANONYMOUS_SOCIAL),
    ("facebook.com", SourceType.ANONYMOUS_SOCIAL),
    ("bbc.com", SourceType.MAJOR_NEWS),
    ("reuters.com", SourceType.MAJOR_NEWS),
    ("apnews.com", SourceType.MAJOR_NEWS),
    ("nytimes.com", SourceType.MAJOR_NEWS),
    ("theguardian.com", SourceType.MAJOR_NEWS),
    ("cnn.com", SourceType.MAJOR_NEWS),
    ("ngo.", SourceType.NGO),
    # NOTE: .org is deliberately absent because it covers too many non-NGO sites
    # (Wikipedia, python.org, etc.); specific NGO domains should be added individually.
]


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL, lowercased and stripped of 'www.'."""
    parsed = urlparse(url)
    if parsed.netloc:
        domain = parsed.netloc
    elif parsed.path:
        domain = parsed.path.split("/")[0]
    else:
        domain = url.split("/")[0]
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _normalize_snippet(text: str) -> str:
    """Normalize a snippet for deduplication: lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two normalized strings."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _classify_evidence_type(snippet: str, source_type: SourceType) -> EvidenceType:
    """Classify evidence type from snippet content using keyword heuristics."""
    snippet_lower = snippet.lower()
    # Check for numeric percentage patterns (e.g. "45%", "12.5 %") before keyword matching.
    if re.search(r"\d+(?:\.\d+)?\s*%", snippet_lower):
        return EvidenceType.STATISTIC
    for keywords, etype in _EVIDENCE_TYPE_HINTS:
        if any(kw in snippet_lower for kw in keywords):
            return etype
    # Fallback based on source type
    if source_type in (SourceType.GOVERNMENT,):
        return EvidenceType.OFFICIAL_STATEMENT
    if source_type in (SourceType.MAJOR_NEWS, SourceType.LOCAL_NEWS):
        return EvidenceType.NEWS_REPORT
    if source_type in (SourceType.REDDIT, SourceType.FORUM, SourceType.ANONYMOUS_SOCIAL):
        return EvidenceType.CLAIM
    return EvidenceType.UNKNOWN


def _classify_source_type(domain: str) -> SourceType:
    """Classify source type from domain using heuristic matching."""
    for hint, stype in _SOURCE_TYPE_DOMAIN_HINTS:
        if hint in domain:
            return stype
    return SourceType.UNKNOWN


def _classify_claim(e_type: EvidenceType, trust_score: float) -> ClaimClassification:
    """Derive claim classification from evidence type and trust score."""
    if e_type in (EvidenceType.STATISTIC, EvidenceType.OFFICIAL_STATEMENT) and trust_score >= 50:
        return ClaimClassification.EVIDENCE_BASED
    if trust_score >= 70:
        return ClaimClassification.EVIDENCE_BASED
    if trust_score >= 50:
        return ClaimClassification.INFERENCE
    if trust_score >= 30:
        return ClaimClassification.ASSUMPTION
    return ClaimClassification.UNKNOWN


def _estimate_confidence(trust_score: float, e_type: EvidenceType) -> Confidence:
    """Estimate confidence level from trust score and evidence type."""
    quality_types = {EvidenceType.STATISTIC, EvidenceType.OFFICIAL_STATEMENT, EvidenceType.NEWS_REPORT}
    base = trust_score
    if e_type in quality_types:
        base = min(base + 10, 100)
    if e_type in (EvidenceType.CLAIM, EvidenceType.UNKNOWN):
        base = max(base - 5, 0)
    if base >= 70:
        return Confidence.HIGH
    if base >= 40:
        return Confidence.MEDIUM
    return Confidence.LOW


class ConflictResult(MakerAgentsModel):
    """Detected conflict between official and community evidence."""

    has_conflict: bool = False
    official_domains: list[str] = Field(default_factory=list)
    community_domains: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    description: str = Field(default="")


class EvidenceAgent:
    """Classifies, deduplicates, and scores evidence from search results."""

    def __init__(self, registry: SourceRegistry | None = None) -> None:
        self._registry = registry if registry is not None else load_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        results: list[SearchResult],
        *,
        run_id: str = "unknown",
        language: str = "en",
    ) -> list[EvidenceItem]:
        """Process raw search results into scored, deduplicated evidence items."""
        items: list[EvidenceItem] = []
        for r in results:
            try:
                items.append(self._to_evidence_item(r, run_id, language))
            except Exception:
                continue  # skip malformed URLs and other construction errors
        items = self._deduplicate(items)
        items = self._assign_ids(items, run_id)
        return items

    def calculate_validity_score(self, items: list[EvidenceItem]) -> float:
        """Compute an aggregate validity score from evidence items.

        NOTE(v0): Uses weighted-average heuristic instead of the PRD §10 model
        (source_count + trust + recency + corroboration - conflict_penalty).
        Will align when recency and corroboration are implemented.

        Uses a weighted average of trust scores with a diversity bonus
        for multi-source evidence and a penalty for low-confidence items.
        """
        if not items:
            return 0.0

        weights = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}
        weighted_sum = 0.0
        weight_total = 0.0

        for item in items:
            w = weights.get(item.confidence.value, 0.5)
            weighted_sum += item.trust_score * w
            weight_total += w

        if weight_total == 0:
            return 0.0

        avg = weighted_sum / weight_total

        # Diversity bonus: multiple distinct domains increase validity.
        domains = {item.source_domain for item in items}
        if len(domains) >= 2:
            avg = min(avg + _VALIDITY_DIVERSITY_BONUS, 100.0)

        # Low-confidence penalty.
        low_conf = sum(1 for item in items if item.confidence == Confidence.LOW)
        if low_conf > 0 and len(items) > 1:
            penalty = _LOW_CONFIDENCE_PENALTY ** low_conf
            avg = avg * max(penalty, 0.5)

        return round(avg, 1)

    def detect_conflicts(self, items: list[EvidenceItem]) -> ConflictResult:
        """Detect narrative conflicts between official and community sources."""
        official_types = {SourceType.GOVERNMENT, SourceType.ACADEMIC, SourceType.MAJOR_NEWS, SourceType.LOCAL_NEWS}
        community_types = {SourceType.REDDIT, SourceType.FORUM, SourceType.ANONYMOUS_SOCIAL}

        official = [i for i in items if i.source_type in official_types]
        community = [i for i in items if i.source_type in community_types]

        if not official or not community:
            return ConflictResult()

        # Conflict exists when trust-score gap between groups is large.
        off_avg = sum(i.trust_score for i in official) / len(official)
        comm_avg = sum(i.trust_score for i in community) / len(community)

        if abs(off_avg - comm_avg) < 25:
            return ConflictResult()

        return ConflictResult(
            has_conflict=True,
            official_domains=list({i.source_domain for i in official}),
            community_domains=list({i.source_domain for i in community}),
            conflicting_evidence_ids=[i.id for i in items if i.id],
            description=(
                f"Official source average trust: {off_avg:.1f} vs "
                f"community source average trust: {comm_avg:.1f}. "
                f"Gap of {abs(off_avg - comm_avg):.1f} points suggests narrative tension."
            ),
        )

    def save_evidence(self, items: list[EvidenceItem], run_dir: Path | str) -> Path:
        """Persist evidence items as JSON under ``<run_dir>/evidence/evidence.json``."""
        run_dir = Path(run_dir)
        evidence_dir = run_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        dest = evidence_dir / "evidence.json"
        payload = [item.model_dump(mode="json") for item in items]
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_evidence_item(
        self,
        result: SearchResult,
        run_id: str,
        language: str,
        *,
        source_type_hint: SourceType | None = None,
    ) -> EvidenceItem:
        """Convert a single search result into a preliminary EvidenceItem."""
        domain = _extract_domain(result.url)
        src_type = source_type_hint if source_type_hint is not None else _classify_source_type(domain)
        e_type = _classify_evidence_type(result.snippet, src_type)
        trust = self._registry.score_for_domain(domain, src_type.value)
        snippet = result.snippet[: _MAX_SNIPPET_LENGTH]
        confidence = _estimate_confidence(trust, e_type)
        claim_class = _classify_claim(e_type, trust)

        return EvidenceItem(
            id="TEMP",  # assigned after dedup
            source_url=result.url,  # type: ignore[arg-type]
            source_domain=domain,
            source_type=src_type,
            evidence_type=e_type,
            snippet=snippet,
            language=language,
            claim_classification=claim_class,
            trust_score=trust,  # type: ignore[arg-type]
            recency="unknown",  # TODO: extract recency from search result metadata (dates, HTTP headers)
            confidence=confidence,
        )

    def _deduplicate(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        """Remove duplicates: exact URL match or Jaccard-similar snippets."""
        seen_urls: set[str] = set()
        seen_snippets: list[str] = []
        unique: list[EvidenceItem] = []

        for item in items:
            url = str(item.source_url)
            if url in seen_urls:
                continue
            norm = _normalize_snippet(item.snippet)
            is_dup = False
            for seen in seen_snippets:
                if _jaccard_similarity(norm, seen) >= _JACCARD_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_urls.add(url)
            seen_snippets.append(norm)
            unique.append(item)

        return unique

    def _assign_ids(self, items: list[EvidenceItem], run_id: str) -> list[EvidenceItem]:
        """Assign sequential evidence IDs to items."""
        prefix = run_id[:8] if len(run_id) >= 8 else run_id
        for i, item in enumerate(items, start=1):
            item.id = f"EVID-{prefix}-{i:04d}"
        return items
