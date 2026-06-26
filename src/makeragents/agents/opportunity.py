"""Opportunity Agent: turns evidence items into candidate opportunities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

from makeragents.run import slugify
from makeragents.schemas import EvidenceItem, Opportunity, OpportunityType


class OpportunityAgent:
    """Groups evidence by theme and generates candidate Opportunities.

    The agent takes a list of EvidenceItem objects (already processed by the
    Evidence Agent), clusters them by thematic similarity, and produces one
    Opportunity per cluster. Opportunities backed by fewer than 2 evidence
    items are marked ``speculative``.

    Generated opportunities are written to the run folder as
    ``opportunities/<slug>/opportunity.yaml``.
    """

    def __init__(
        self,
        max_opportunities: int = 5,
    ) -> None:
        self.max_opportunities = max_opportunities

    def process(
        self,
        evidence_items: Sequence[EvidenceItem],
        run_dir: Path,
    ) -> list[Opportunity]:
        """Convert evidence items into candidate opportunities.

        Parameters
        ----------
        evidence_items:
            Normalised evidence items produced by the Evidence Agent.
        run_dir:
            Root of the run folder where opportunity artifacts are written.

        Returns
        -------
        list[Opportunity]
            Opportunities derived from the evidence, respecting
            ``max_opportunities``.
        """
        groups = self._group_by_theme(evidence_items)
        opportunities: list[Opportunity] = []

        for group in groups:
            if len(opportunities) >= self.max_opportunities:
                break
            opportunity = self._build_opportunity(group)
            self._persist_opportunity(opportunity, run_dir)
            opportunities.append(opportunity)

        return opportunities

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_theme(
        evidence_items: Sequence[EvidenceItem],
    ) -> list[list[EvidenceItem]]:
        """Group evidence items by basic thematic similarity.

        The current implementation uses a simple heuristic: items that share
        the same ``source_domain`` are considered thematically related. A
        production agent would use embedding similarity or LLM-based clustering.
        """
        seen: set[str] = set()
        groups: list[list[EvidenceItem]] = []

        for item in evidence_items:
            if item.id in seen:
                continue

            group = [item]
            seen.add(item.id)

            for other in evidence_items:
                if other.id in seen:
                    continue
                if _same_theme(item, other):
                    group.append(other)
                    seen.add(other.id)

            groups.append(group)

        return groups

    @staticmethod
    def _build_opportunity(evidence_group: list[EvidenceItem]) -> Opportunity:
        """Derive a single Opportunity from a cluster of evidence items."""

        # Build a slug-based id from the first evidence snippet.
        seed = evidence_group[0].snippet.split(".", 1)[0].strip()
        opp_id = f"OPP-{slugify(seed)}" if seed else "OPP-unknown"

        # Truncate extremely long IDs (slugs from long snippets).
        if len(opp_id) > 80:
            opp_id = opp_id[:80].rstrip("-")

        # Pick the most common (or first) evidence type as the opportunity type.
        type_counts: dict[OpportunityType, int] = {}
        for ev in evidence_group:
            mapped = _map_evidence_to_opportunity_type(ev.evidence_type.value)
            type_counts[mapped] = type_counts.get(mapped, 0) + 1
        best_type = max(type_counts, key=type_counts.get)

        # Pain summary: join the first few unique snippets.
        seen_snippets: set[str] = set()
        parts: list[str] = []
        for ev in evidence_group:
            if ev.snippet not in seen_snippets:
                seen_snippets.add(ev.snippet)
                parts.append(ev.snippet)
                if len(parts) >= 3:
                    break
        pain_summary = " | ".join(parts)

        # Who benefits and vulnerable groups are placeholders until an LLM
        # or more sophisticated heuristic is plugged in.
        who_benefits = _derive_beneficiaries(evidence_group)

        evidence_ids = [ev.id for ev in evidence_group]

        # Speculative when fewer than 2 evidence items support the opportunity.
        speculative = len(evidence_group) < 2

        vulnerable: list[str] = []
        if speculative:
            vulnerable.append("unknown — speculative opportunity")

        return Opportunity(
            id=opp_id,
            title=_derive_title(best_type, evidence_group),
            type=best_type,
            pain_summary=pain_summary,
            who_benefits=who_benefits,
            vulnerable_groups=vulnerable,
            evidence_ids=evidence_ids,
            speculative=speculative,
        )

    @staticmethod
    def _persist_opportunity(opportunity: Opportunity, run_dir: Path) -> Path:
        """Write a single opportunity as ``opportunities/<slug>/opportunity.yaml``.

        Returns the path of the written YAML file.
        """
        slug = slugify(opportunity.id)
        opp_dir = run_dir / "opportunities" / slug
        opp_dir.mkdir(parents=True, exist_ok=True)

        payload = opportunity.model_dump(mode="json")
        dest = opp_dir / "opportunity.yaml"
        with dest.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        return dest


# ------------------------------------------------------------------
# Module-level helper functions
# ------------------------------------------------------------------

_OPPORTUNITY_TYPE_MAP: dict[str, OpportunityType] = {
    "complaint": OpportunityType.ADVOCACY_REPORT,
    "official_statement": OpportunityType.PUBLIC_GUIDE,
    "news_report": OpportunityType.COMMUNITY_SUPPORT_PROCESS,
    "first_hand_account": OpportunityType.MANUAL_SERVICE,
    "second_hand_account": OpportunityType.MANUAL_SERVICE,
    "statistic": OpportunityType.OPEN_DATA_RESOURCE,
    "claim": OpportunityType.ADVOCACY_REPORT,
    "unknown": OpportunityType.COMMUNITY_SUPPORT_PROCESS,  # explicit fallback
}

_OPPORTUNITY_TITLES: dict[OpportunityType, str] = {
    OpportunityType.PUBLIC_GUIDE: "Public information guide",
    OpportunityType.COORDINATION_PROCESS: "Coordination process",
    OpportunityType.ADVOCACY_REPORT: "Advocacy report",
    OpportunityType.TRANSPARENCY_DASHBOARD: "Transparency dashboard",
    OpportunityType.MANUAL_SERVICE: "Manual service",
    OpportunityType.COMMUNITY_SUPPORT_PROCESS: "Community support process",
    OpportunityType.SOFTWARE_TOOLING: "Software tooling",
    OpportunityType.INSTITUTION_FACING_REPORT: "Institution-facing report",
    OpportunityType.OPEN_DATA_RESOURCE: "Open data resource",
}


def _same_theme(a: EvidenceItem, b: EvidenceItem) -> bool:
    """Heuristic thematic grouping — same domain implies same theme."""
    return a.source_domain == b.source_domain


def _map_evidence_to_opportunity_type(evidence_type: str) -> OpportunityType:
    """Map an evidence type string to its best-guess OpportunityType."""
    return _OPPORTUNITY_TYPE_MAP.get(
        evidence_type, OpportunityType.COMMUNITY_SUPPORT_PROCESS
    )


def _derive_title(
    opp_type: OpportunityType,
    evidence_group: list[EvidenceItem],
) -> str:
    """Build a descriptive title from the opportunity type and evidence.

    Falls back to the generic type-based title when the snippet is too short
    to extract a meaningful label.
    """
    generic = _OPPORTUNITY_TITLES.get(opp_type, "Opportunity")
    seed = evidence_group[0].snippet.strip()
    if len(seed) < 10:
        return generic
    title_words = seed.split()[:8]
    title = generic + ": " + " ".join(title_words).rstrip(".,;:!")
    return title[:120]


def _derive_beneficiaries(
    evidence_group: list[EvidenceItem],
) -> list[str]:
    """Extract beneficiary labels from evidence snippets.

    This is a heuristic placeholder that looks for common demographic or
    role keywords. A production agent would use an LLM.
    """
    keywords = {
        "senior": "senior citizens",
        "elderly": "senior citizens",
        "older": "senior citizens",
        "youth": "youth",
        "student": "students",
        "parent": "parents",
        "family": "families",
        "patient": "patients",
        "resident": "residents",
        "citizen": "citizens",
        "worker": "workers",
        "employee": "employees",
        "immigrant": "immigrants",
        "refugee": "refugees",
        "disabled": "people with disabilities",
        "low-income": "low-income households",
        "small business": "small business owners",
    }

    found: list[str] = []
    seen_labels: set[str] = set()
    combined = " ".join(ev.snippet.lower() for ev in evidence_group)

    for keyword, label in keywords.items():
        if keyword in combined and label not in seen_labels:
            found.append(label)
            seen_labels.add(label)

    return found or ["community members"]
