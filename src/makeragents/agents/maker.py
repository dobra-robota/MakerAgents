"""Maker Agent: scores value-add potential and produces the Maker argument.

Takes an Opportunity and supporting evidence, produces component scores
and a structured Maker argument saved as both JSON and Markdown files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from makeragents.run import opportunity_artifact_slug
from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    Opportunity,
    OpportunityType,
    ScoreSet,
)

if TYPE_CHECKING:
    from makeragents.llm import LLMClient

# ---------------------------------------------------------------------------
# Scoring weights for the maker_score composite
# NOTE(v0): Weights are tuned to match the rank_score proportions for
# people_helped (~0.175) and severity (~0.20) — see ScoreSet.
# ---------------------------------------------------------------------------
_W_PEOPLE = 0.175
_W_SEVERITY = 0.20
_W_VALIDITY = 0.20
_W_LOW_HARM = 0.175
_W_EASE = 0.14
_W_ABILITY = 0.11

# Valid claim classifications
_VALID_CLAIM_CLASSIFICATIONS: set[str] = {
    "evidence_based",
    "inference",
    "assumption",
    "unknown",
}

# Valid confidence levels
_VALID_CONFIDENCE_LEVELS: set[str] = {"low", "medium", "high"}


def _count_high_confidence(items: list[EvidenceItem]) -> int:
    return sum(1 for i in items if i.confidence == Confidence.HIGH)


def _avg_trust(items: list[EvidenceItem]) -> float:
    if not items:
        return 0.0
    return sum(i.trust_score for i in items) / len(items)


@dataclass
class MakerResult:
    """Structured output from the Maker Agent."""

    opportunity_id: str
    maker_score: float
    maker_confidence: Confidence
    people_helped_score: float
    severity_score: float
    impact_score: float
    validity_score: float
    intervention_ease_score: float
    harm_risk_score: float
    ability_to_act_score: float
    rank_score: float
    claim_classifications: dict[str, str] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    summary: str = ""
    value_add_argument: str = ""
    claims: list[dict[str, str]] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "maker_score": self.maker_score,
            "maker_confidence": self.maker_confidence.value,
            "people_helped_score": self.people_helped_score,
            "severity_score": self.severity_score,
            "impact_score": self.impact_score,
            "validity_score": self.validity_score,
            "intervention_ease_score": self.intervention_ease_score,
            "harm_risk_score": self.harm_risk_score,
            "ability_to_act_score": self.ability_to_act_score,
            "rank_score": self.rank_score,
            "claim_classifications": self.claim_classifications,
            "evidence_ids": self.evidence_ids,
            "summary": self.summary,
            "value_add_argument": self.value_add_argument,
            "claims": self.claims,
        }


class MakerAgent:
    """Creates the value-add argument and assigns Maker scores.

    Deterministic scoring is used by default via ``run()`` (no LLM).
    When an ``LLMClient`` is provided, ``run_with_llm()`` generates a
    genuinely generative value-add argument with disciplined claim
    classification and evidence citation per PRD §7.4.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        opportunity: Opportunity,
        evidence: list[EvidenceItem],
    ) -> MakerResult:
        """Score an opportunity and produce a Maker argument."""
        cited = self._cited_evidence(opportunity, evidence)
        evidence_ids = [item.id for item in cited]

        # Component scores
        validity = self._score_validity(cited)
        people = self._score_people_helped(opportunity, cited)
        severity = self._score_severity(opportunity, cited)
        impact = round((people * 0.5) + (severity * 0.5), 1)
        ease = self._score_intervention_ease(opportunity, cited)
        harm = self._score_harm_risk(opportunity, cited)
        ability = self._score_ability_to_act(opportunity, cited)

        low_harm = 100.0 - harm
        # NOTE: people and severity are used directly (not via impact) to
        # avoid double-counting people_helped. See PR #27.
        maker_score = round(
            people * _W_PEOPLE
            + severity * _W_SEVERITY
            + validity * _W_VALIDITY
            + low_harm * _W_LOW_HARM
            + ease * _W_EASE
            + ability * _W_ABILITY,
            1,
        )
        maker_score = max(0.0, min(100.0, maker_score))

        confidence = self._maker_confidence(cited, validity, harm)
        rank = ScoreSet.calculate_rank_score(
            people_helped_score=people,
            severity_score=severity,
            validity_score=validity,
            intervention_ease_score=ease,
            harm_risk_score=harm,
            ability_to_act_score=ability,
        )

        classifications: dict[str, str] = {}
        for item in cited:
            classifications[item.id] = item.claim_classification.value

        summary = self._build_summary(
            opportunity, maker_score, confidence, evidence_ids
        )

        return MakerResult(
            opportunity_id=opportunity.id,
            maker_score=maker_score,
            maker_confidence=confidence,
            people_helped_score=people,
            severity_score=severity,
            impact_score=impact,
            validity_score=validity,
            intervention_ease_score=ease,
            harm_risk_score=harm,
            ability_to_act_score=ability,
            rank_score=rank,
            claim_classifications=classifications,
            evidence_ids=evidence_ids,
            summary=summary,
        )

    def run_with_llm(
        self,
        opportunity: Opportunity,
        evidence: list[EvidenceItem],
        *,
        city: str = "",
        community: str = "",
    ) -> MakerResult:
        """Score an opportunity using LLM-backed value-add argument generation.

        Uses ``load_prompt`` and the configured ``LLMClient`` to produce
        a genuinely generative value-add argument with disciplined claim
        classification and evidence citation. Falls back to deterministic
        ``run()`` if no LLM client is configured or if the LLM call fails.
        """
        # Gather cited evidence
        cited = self._cited_evidence(opportunity, evidence)
        evidence_ids = [item.id for item in cited]
        evidence_ids_set = set(evidence_ids)

        # Start with deterministic scores as the base
        deterministic = self.run(opportunity, evidence)

        if self._llm_client is None:
            return deterministic

        # Build opportunity summary for the prompt
        opp_summary_lines = [
            f"ID: {opportunity.id}",
            f"Title: {opportunity.title}",
            f"Type: {opportunity.type.value}",
            f"Pain: {opportunity.pain_summary}",
            f"Who benefits: {', '.join(opportunity.who_benefits)}",
        ]
        if opportunity.vulnerable_groups:
            opp_summary_lines.append(
                f"Vulnerable groups: {', '.join(opportunity.vulnerable_groups)}"
            )
        if opportunity.speculative:
            opp_summary_lines.append("⚠️ This opportunity is marked speculative.")
        if cited:
            opp_summary_lines.append("\nEvidence:")
            for item in cited:
                opp_summary_lines.append(
                    f"- [{item.id}] ({item.claim_classification.value}) "
                    f"{item.snippet[:200]}"
                )
        opportunity_summary = "\n".join(opp_summary_lines)

        try:
            from makeragents.prompts import load_prompt

            prompt = load_prompt(
                "maker",
                city=city,
                community=community,
                opportunity_summary=opportunity_summary,
            )

            messages: list = []
            from makeragents.llm import ChatMessage

            messages.append(ChatMessage("system", prompt))
            messages.append(
                ChatMessage(
                    "user",
                    "Generate the maker argument JSON as instructed.",
                )
            )

            llm_result = self._llm_client.chat_json(messages)

            # Validate and extract structured output
            return self._build_llm_result(
                opportunity=opportunity,
                deterministic=deterministic,
                llm_result=llm_result,
                evidence_ids_set=evidence_ids_set,
            )
        except Exception:
            # Fall back to deterministic on any LLM failure
            return deterministic

    def save_output(
        self,
        result: MakerResult,
        run_dir: Path | str,
    ) -> tuple[Path, Path]:
        """Write maker.json and maker.md to the opportunity folder."""
        opp_dir = (
            Path(run_dir)
            / "opportunities"
            / opportunity_artifact_slug(result.opportunity_id)
        )
        opp_dir.mkdir(parents=True, exist_ok=True)

        json_path = opp_dir / "maker.json"
        json_path.write_text(
            json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        md_path = opp_dir / "maker.md"
        md_path.write_text(self._to_markdown(result), encoding="utf-8")

        return json_path, md_path

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _cited_evidence(
        self, opportunity: Opportunity, evidence: list[EvidenceItem]
    ) -> list[EvidenceItem]:
        # Only use cited evidence; empty evidence_ids means no evidence was linked
        if not opportunity.evidence_ids:
            return []
        by_id = {e.id: e for e in evidence}
        return [by_id[eid] for eid in opportunity.evidence_ids if eid in by_id]

    def _score_validity(self, cited: list[EvidenceItem]) -> float:
        if not cited:
            return 0.0
        return round(_avg_trust(cited), 1)

    def _score_people_helped(self, opp: Opportunity, cited: list[EvidenceItem]) -> float:
        # NOTE(v0): Scores are count-based proxies. Population magnitude and issue intensity are not yet modeled.
        base = min(len(opp.who_benefits) * 25.0, 100.0)
        bonus = min(len(cited) * 3.0, 25.0)
        return round(min(base + bonus, 100.0), 1)

    def _score_severity(self, opp: Opportunity, cited: list[EvidenceItem]) -> float:
        # NOTE(v0): Scores are count-based proxies. Population magnitude and issue intensity are not yet modeled.
        base = min(len(opp.vulnerable_groups) * 20.0, 60.0)
        evidence_bonus = min(len(cited) * 3.0, 20.0)
        score = base + evidence_bonus + 20.0  # baseline
        if opp.speculative:
            score *= 0.7
        return round(min(score, 100.0), 1)

    def _score_intervention_ease(self, opp: Opportunity, cited: list[EvidenceItem]) -> float:
        type_baselines: dict[OpportunityType, float] = {
            OpportunityType.PUBLIC_GUIDE: 85.0,
            OpportunityType.OPEN_DATA_RESOURCE: 75.0,
            OpportunityType.ADVOCACY_REPORT: 65.0,
            OpportunityType.COORDINATION_PROCESS: 55.0,
            OpportunityType.COMMUNITY_SUPPORT_PROCESS: 50.0,
            OpportunityType.MANUAL_SERVICE: 45.0,
            OpportunityType.INSTITUTION_FACING_REPORT: 40.0,
            OpportunityType.SOFTWARE_TOOLING: 30.0,
            OpportunityType.TRANSPARENCY_DASHBOARD: 35.0,
        }
        base = type_baselines.get(opp.type, 50.0)
        evidence_bonus = min(len(cited) * 2.0, 15.0)
        score = base + evidence_bonus
        if opp.speculative:
            score *= 0.8
        return round(min(score, 100.0), 1)

    def _score_harm_risk(self, opp: Opportunity, cited: list[EvidenceItem]) -> float:
        base = min(len(opp.vulnerable_groups) * 15.0, 50.0)
        if opp.speculative:
            base += 10.0
        evidence_reduction = min(len(cited) * 2.0, 15.0)
        return round(max(min(base - evidence_reduction, 100.0), 0.0), 1)

    def _score_ability_to_act(self, opp: Opportunity, cited: list[EvidenceItem]) -> float:
        base = min(len(cited) * 12.0, 80.0)
        if opp.speculative:
            base *= 0.7
        return round(min(base + 10.0, 100.0), 1)  # +10 baseline

    # HIGH confidence requires: strong validity (≥70), majority high-confidence evidence (≥50%), and acceptable harm risk (<40)
    def _maker_confidence(
        self, cited: list[EvidenceItem], validity: float, harm: float
    ) -> Confidence:
        if not cited:
            return Confidence.LOW
        high_ratio = _count_high_confidence(cited) / len(cited)
        if validity >= 70 and high_ratio >= 0.5 and harm < 40:
            return Confidence.HIGH
        if validity >= 40:
            return Confidence.MEDIUM
        return Confidence.LOW

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _build_llm_result(
        self,
        *,
        opportunity: Opportunity,
        deterministic: MakerResult,
        llm_result: dict[str, Any],
        evidence_ids_set: set[str],
    ) -> MakerResult:
        """Build a :class:`MakerResult` from the LLM's JSON output.

        Validates claim classifications, filters unknown evidence IDs,
        and merges the LLM value-add argument with deterministic scores.
        """
        value_add_arg = str(llm_result.get("value_add_summary", ""))
        score = llm_result.get("score")
        confidence_raw = llm_result.get("confidence", "medium")

        # Validate score range
        try:
            score_val = float(score) if score is not None else deterministic.maker_score
        except (ValueError, TypeError):
            score_val = deterministic.maker_score
        score_val = max(0.0, min(100.0, score_val))

        # Validate confidence
        if (
            isinstance(confidence_raw, str)
            and confidence_raw.lower() in _VALID_CONFIDENCE_LEVELS
        ):
            confidence = Confidence(confidence_raw.lower())
        else:
            confidence = deterministic.maker_confidence

        # Validate and collect claims
        raw_claims: list[dict[str, Any]] = (
            llm_result.get("claims")
            if isinstance(llm_result.get("claims"), list)
            else []
        )
        claims: list[dict[str, str]] = []
        for c in raw_claims:
            if not isinstance(c, dict):
                continue
            text = str(c.get("text", ""))
            cls_raw = str(c.get("classification", "")).lower()
            eid = str(c.get("evidence_id", ""))
            if cls_raw not in _VALID_CLAIM_CLASSIFICATIONS:
                cls_raw = "unknown"
            claims.append({
                "text": text,
                "classification": cls_raw,
                "evidence_id": eid,
            })

        # Collect evidence IDs from LLM (filter to known IDs)
        llm_eids: list[str] = (
            llm_result.get("evidence_ids")
            if isinstance(llm_result.get("evidence_ids"), list)
            else []
        )
        cited_ids = [
            eid for eid in llm_eids if isinstance(eid, str) and eid in evidence_ids_set
        ]

        # Build claim classifications from claims
        claim_classifications: dict[str, str] = {}
        for claim in claims:
            eid = claim.get("evidence_id", "")
            if eid:
                claim_classifications[eid] = claim["classification"]

        summary = self._build_summary(
            opportunity, score_val, confidence, cited_ids
        )

        return MakerResult(
            opportunity_id=opportunity.id,
            maker_score=score_val,
            maker_confidence=confidence,
            people_helped_score=deterministic.people_helped_score,
            severity_score=deterministic.severity_score,
            impact_score=deterministic.impact_score,
            validity_score=deterministic.validity_score,
            intervention_ease_score=deterministic.intervention_ease_score,
            harm_risk_score=deterministic.harm_risk_score,
            ability_to_act_score=deterministic.ability_to_act_score,
            rank_score=deterministic.rank_score,
            claim_classifications=claim_classifications,
            evidence_ids=cited_ids,
            summary=summary,
            value_add_argument=value_add_arg,
            claims=claims,
        )

    def _build_summary(
        self,
        opp: Opportunity,
        maker_score: float,
        confidence: Confidence,
        evidence_ids: list[str],
    ) -> str:
        parts = [
            f"Opportunity: {opp.title}",
            f"Type: {opp.type.value}",
            f"Pain: {opp.pain_summary}",
            f"Benefits: {', '.join(opp.who_benefits)}",
            f"Maker Score: {maker_score:.1f}/100 ({confidence.value} confidence)",
            f"Evidence cited: {len(evidence_ids)} items",
        ]
        if opp.vulnerable_groups:
            parts.append(f"Vulnerable groups: {', '.join(opp.vulnerable_groups)}")
        if opp.speculative:
            parts.append("⚠️ Marked speculative — limited evidence base")
        return "\n".join(parts)

    def _to_markdown(self, result: MakerResult) -> str:
        lines = [
            f"# Maker Analysis: {result.opportunity_id}",
            "",
            "## Summary",
            "",
            result.summary,
            "",
        ]

        # Include LLM-generated value-add argument if present
        if result.value_add_argument:
            lines.extend([
                "## Value-Add Argument",
                "",
                result.value_add_argument,
                "",
            ])

        # Include claims table if present
        if result.claims:
            lines.extend([
                "## Claims",
                "",
                "| Claim | Classification | Evidence ID |",
                "|-------|----------------|-------------|",
            ])
            for claim in result.claims:
                text = claim.get("text", "").replace("|", "\\|")
                cls_ = claim.get("classification", "")
                eid = claim.get("evidence_id", "")
                lines.append(f"| {text} | {cls_} | `{eid}` |")
            lines.append("")

        lines.extend([
            "## Scores",
            "",
            "| Score | Value |",
            "|-------|-------|",
            f"| Maker Score | {result.maker_score:.1f} |",
            f"| Confidence | {result.maker_confidence.value} |",
            f"| People Helped | {result.people_helped_score:.1f} |",
            f"| Severity | {result.severity_score:.1f} |",
            f"| Impact | {result.impact_score:.1f} |",
            f"| Validity | {result.validity_score:.1f} |",
            f"| Intervention Ease | {result.intervention_ease_score:.1f} |",
            f"| Harm Risk | {result.harm_risk_score:.1f} |",
            f"| Ability to Act | {result.ability_to_act_score:.1f} |",
            f"| **Rank Score** | **{result.rank_score:.1f}** |",
            "",
            "## Claim Classification",
            "",
        ])
        for eid, cls in result.claim_classifications.items():
            lines.append(f"- `{eid}`: {cls}")

        lines.extend([
            "",
            "## Evidence Cited",
            "",
        ])
        for eid in result.evidence_ids:
            lines.append(f"- `{eid}`")

        return "\n".join(lines) + "\n"
