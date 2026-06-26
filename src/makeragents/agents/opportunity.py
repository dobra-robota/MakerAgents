"""Opportunity Agent: turns evidence items into candidate opportunities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import yaml

from makeragents.llm import ChatMessage, LLMClient
from makeragents.prompts import load_prompt
from makeragents.run import slugify
from makeragents.schemas import EvidenceItem, Opportunity, OpportunityType

logger = logging.getLogger(__name__)

# Marker prefix for auto-generated slugs.
_SLUG_PREFIX = "OPP-LLM-"


class OpportunityAgent:
    """Groups evidence by theme and generates candidate Opportunities.

    The agent takes a list of EvidenceItem objects (already processed by the
    Evidence Agent), derives candidate opportunities via LLM-backed analysis,
    and produces one Opportunity per cluster. When the LLM path is unavailable
    or returns empty results, falls back to heuristic thematic clustering.

    Opportunities backed by fewer than 2 independent sources are marked
    ``speculative`` (PRD §7.3, §10). When evidence is weak but potential
    impact is high, weak-evidence opportunities are allowed but flagged.

    Generated opportunities are written to the run folder as
    ``opportunities/<slug>/opportunity.yaml``.
    """

    def __init__(
        self,
        max_opportunities: int = 5,
        *,
        llm_client: LLMClient | None = None,
        city: str = "",
        community: str = "",
    ) -> None:
        self.max_opportunities = max_opportunities
        self._llm_client = llm_client
        self._city = city
        self._community = community

    def process(
        self,
        evidence_items: Sequence[EvidenceItem],
        run_dir: Path,
    ) -> list[Opportunity]:
        """Convert evidence items into candidate opportunities.

        Tries LLM-backed derivation first; falls back to the heuristic
        thematic-clustering path when LLM is unavailable or returns no results.

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
        if not evidence_items:
            return []

        # --- LLM-backed path ---
        llm_opportunities = self._llm_derive_opportunities(evidence_items)
        if llm_opportunities:
            opportunities: list[Opportunity] = []
            for opp in llm_opportunities:
                if len(opportunities) >= self.max_opportunities:
                    break
                self._persist_opportunity(opp, run_dir)
                opportunities.append(opp)
            return opportunities

        # --- Heuristic fallback ---
        logger.info("LLM path unavailable or returned no results; falling back to heuristic clustering.")
        groups = self._group_by_theme(evidence_items)
        opportunities = []

        for group in groups:
            if len(opportunities) >= self.max_opportunities:
                break
            opportunity = self._build_opportunity(group)
            self._persist_opportunity(opportunity, run_dir)
            opportunities.append(opportunity)

        return opportunities


    def _llm_derive_opportunities(
        self,
        evidence_items: Sequence[EvidenceItem],
    ) -> list[Opportunity]:
        """Derive candidate opportunities via LLM-backed analysis.

        Returns an empty list when the LLM client is not configured,
        the call fails, or the model returns zero opportunities — the
        caller then falls back to the heuristic path.
        """
        if self._llm_client is None:
            return []

        evidence_summary = _build_evidence_summary(evidence_items)
        try:
            prompt = load_prompt(
                "opportunity",
                city=self._city or "unknown",
                community=self._community or "unknown",
                max_opportunities=str(self.max_opportunities),
                evidence_summary=evidence_summary,
            )
        except FileNotFoundError:
            logger.warning("Opportunity prompt file not found; falling back to heuristic.")
            return []

        try:
            response = self._llm_client.chat_json(
                [ChatMessage("user", prompt)],
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning("LLM call failed: %s; falling back to heuristic.", exc)
            return []

        raw_opps = response.get("opportunities")
        if not isinstance(raw_opps, list) or not raw_opps:
            logger.warning("LLM returned no opportunities; falling back to heuristic.")
            return []

        opportunities: list[Opportunity] = []
        valid_evidence_ids = {ev.id for ev in evidence_items}

        for idx, raw in enumerate(raw_opps):
            if not isinstance(raw, dict):
                continue
            if len(opportunities) >= self.max_opportunities:
                break

            opp = _parse_llm_opportunity(
                raw,
                index=idx,
                valid_evidence_ids=valid_evidence_ids,
            )
            if opp is not None:
                opportunities.append(opp)

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


# ------------------------------------------------------------------
# LLM-backed helpers
# ------------------------------------------------------------------


def _build_evidence_summary(evidence_items: Sequence[EvidenceItem]) -> str:
    """Build a compact markdown summary of evidence for the LLM prompt."""
    lines: list[str] = []
    for ev in evidence_items:
        lines.append(
            f"- **{ev.id}** [{ev.evidence_type.value}] "
            f"({ev.source_domain}, trust={ev.trust_score}, "
            f"confidence={ev.confidence.value}): {ev.snippet}"
        )
    return "\n".join(lines)


# Value-based type names for OpportunityType enum members.
_OPPORTUNITY_TYPE_VALUES: set[str] = {t.value for t in OpportunityType}


def _parse_llm_opportunity(
    raw: dict,
    *,
    index: int,
    valid_evidence_ids: set[str],
) -> Opportunity | None:
    """Parse and validate a single opportunity from LLM JSON output.

    Returns ``None`` for any input that cannot be mapped to a valid
    :class:`Opportunity` — the caller simply skips it.
    """
    opp_id = raw.get("id", f"{_SLUG_PREFIX}{index + 1:03d}")
    title = raw.get("title", "")
    opp_type_raw = raw.get("type", "")
    pain_summary = raw.get("pain_summary", "")
    who_benefits = raw.get("who_benefits", [])
    vulnerable_groups = raw.get("vulnerable_groups", [])
    evidence_ids = raw.get("evidence_ids", [])
    speculative = raw.get("speculative", False)

    # --- Required fields ---
    if not title or not pain_summary:
        logger.debug("LLM opportunity %d missing title or pain_summary; skipping.", index + 1)
        return None

    # --- Opportunity type ---
    opp_type = _resolve_opportunity_type(opp_type_raw)
    if opp_type is None:
        logger.debug("LLM opportunity %d unrecognised type %r; skipping.", index + 1, opp_type_raw)
        return None

    # --- Evidence ID validation ---
    filtered_evidence_ids = [eid for eid in evidence_ids if eid in valid_evidence_ids]
    if not isinstance(evidence_ids, list) or len(filtered_evidence_ids) == 0:
        logger.debug("LLM opportunity %d has no valid evidence IDs; marking speculative.", index + 1)
        speculative = True

    # --- Beneficiary validation ---
    if not isinstance(who_benefits, list) or len(who_benefits) == 0:
        who_benefits = ["community members"]

    # --- Vulnerable groups ---
    if not isinstance(vulnerable_groups, list):
        vulnerable_groups = []

    # --- Speculative check: ≥2 independent sources rule (PRD §7.3) ---
    if len(filtered_evidence_ids) < 2 and not speculative:
        speculative = True
        if not vulnerable_groups:
            vulnerable_groups = []
        if "unknown — speculative opportunity" not in vulnerable_groups:
            vulnerable_groups.append("unknown — speculative opportunity")

    # --- Coerce simple types ---
    title = str(title).strip()[:200]
    pain_summary = str(pain_summary).strip()
    who_benefits = [str(b).strip() for b in who_benefits if str(b).strip()]
    if not who_benefits:
        who_benefits = ["community members"]
    vulnerable_groups = [str(v).strip() for v in vulnerable_groups if str(v).strip()]
    evidence_id_strs = [str(eid).strip() for eid in filtered_evidence_ids if str(eid).strip()]

    try:
        return Opportunity(
            id=str(opp_id).strip() or f"{_SLUG_PREFIX}{index + 1:03d}",
            title=title,
            type=opp_type,
            pain_summary=pain_summary,
            who_benefits=who_benefits,
            vulnerable_groups=vulnerable_groups,
            evidence_ids=evidence_id_strs,
            speculative=bool(speculative),
        )
    except Exception as exc:
        logger.warning("Failed to construct Opportunity from LLM output: %s", exc)
        return None


def _resolve_opportunity_type(raw: str) -> OpportunityType | None:
    """Resolve a raw type string to an :class:`OpportunityType` member.

    Returns ``None`` when the string does not match any known type.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in _OPPORTUNITY_TYPE_VALUES:
        return OpportunityType(value)
    # Fuzzy: handle common LLM output variants.
    _fuzzy_map: dict[str, str] = {
        "open data": "open_data_resource",
        "open data resource": "open_data_resource",
        "public guide": "public_guide",
        "advocacy report": "advocacy_report",
        "coordination process": "coordination_process",
        "transparency dashboard": "transparency_dashboard",
        "manual service": "manual_service",
        "community support process": "community_support_process",
        "community support": "community_support_process",
        "software tooling": "software_tooling",
        "software": "software_tooling",
        "institution facing report": "institution_facing_report",
        "institution report": "institution_facing_report",
    }
    mapped = _fuzzy_map.get(value)
    if mapped is not None:
        return OpportunityType(mapped)
    return None
