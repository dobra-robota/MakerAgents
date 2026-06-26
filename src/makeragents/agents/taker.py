"""Defensive red-team Taker Agent for MakerAgents.

The Taker Agent evaluates an opportunity purely through a defensive
lens: it identifies exploitation risk, extraction risk, and harm
potential **without providing exploitation instructions**.

Usage
-----
    taker = TakerAgent()
    result = taker.analyze(opportunity, evidence_items)
    taker.save_output(opportunity_slug, run_dir)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    Opportunity,
    ScoreSet,
)

# Risk category keys required in TakerOutput.risk_breakdown.
_RISK_CATEGORIES = (
    "extraction_risk",
    "gatekeeping_risk",
    "false_authority_risk",
    "dependency_risk",
    "harm_risk",
)

# Phrases that must NOT appear in any Taker output summary (defensive only).
_PROHIBITED_PATTERNS = (
    "how to exploit",
    "steps to exploit",
    "step by step exploit",
    "exploitation instructions",
    "how to abuse",
    "how to take advantage",
)

# Evidence classification thresholds used for scoring heuristics.
_LOW_EVIDENCE_RISK_BOOST = 10
_MEDIUM_EVIDENCE_RISK_BOOST = 5


class TakerOutput:
    """Structured output produced by the Taker Agent.

    Attributes
    ----------
    opportunity_id : str
        The ``id`` of the analysed opportunity.
    taker_score : float
        Exploitability risk score 0–100 (higher = worse).
    taker_confidence : str
        One of ``"low"``, ``"medium"``, ``"high"``.
    risk_breakdown : dict[str, float]
        Per-category risk scores 0–100.
    evidence_ids : list[str]
        Evidence IDs cited during analysis.
    summary : str
        Defensive summary of identified risks.
    """

    def __init__(
        self,
        *,
        opportunity_id: str,
        taker_score: float,
        taker_confidence: str,
        risk_breakdown: dict[str, float],
        evidence_ids: list[str],
        summary: str,
    ) -> None:
        self.opportunity_id = opportunity_id
        self.taker_score = taker_score
        self.taker_confidence = taker_confidence
        self.risk_breakdown = risk_breakdown
        self.evidence_ids = list(evidence_ids)
        self.summary = summary

        self._validate()

    def _validate(self) -> None:
        """Guard against common output contract violations."""
        if not 0 <= self.taker_score <= 100:
            raise ValueError(
                f"taker_score must be in 0–100, got {self.taker_score}"
            )
        missing = set(_RISK_CATEGORIES) - set(self.risk_breakdown.keys())
        if missing:
            raise ValueError(
                f"risk_breakdown missing required categories: {missing}"
            )
        for category, score in self.risk_breakdown.items():
            if not 0 <= score <= 100:
                raise ValueError(
                    f"risk_breakdown[{category!r}] must be 0–100, got {score}"
                )
        for pattern in _PROHIBITED_PATTERNS:
            if pattern in self.summary.lower():
                raise ValueError(
                    f"Summary contains prohibited exploitation language "
                    f"(matched pattern: {pattern!r})"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "opportunity_id": self.opportunity_id,
            "taker_score": self.taker_score,
            "taker_confidence": self.taker_confidence,
            "risk_breakdown": dict(self.risk_breakdown),
            "evidence_ids": list(self.evidence_ids),
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        """Render a readable taker.md report."""
        lines = [
            f"# Taker Analysis: {self.opportunity_id}",
            "",
            "## Summary",
            "",
            self.summary,
            "",
            "## Scores",
            "",
            f"- **Taker Score (exploitability):** {self.taker_score}/100",
            f"- **Confidence:** {self.taker_confidence}",
            "",
            "## Risk Breakdown",
            "",
        ]
        for category, score in self.risk_breakdown.items():
            label = category.replace("_", " ").title()
            lines.append(f"- **{label}:** {score}/100")
        lines += [
            "",
            "## Evidence Cited",
            "",
        ]
        for eid in self.evidence_ids:
            lines.append(f"- `{eid}`")
        lines.append("")
        lines.append(
            "---\n"
            "*This analysis was produced by the Taker Agent (defensive "
            "red-team). It identifies risks defensively and does not "
            "provide exploitation instructions.*"
        )
        return "\n".join(lines) + "\n"


class TakerAgent:
    """Defensive red-team analysis for a single Opportunity.

    The agent operates entirely through deterministic heuristics derived
    from the opportunity's scores and linked evidence so that tests can
    run without real LLM or API calls.
    """

    def analyze(
        self,
        opportunity: Opportunity,
        evidence_items: list[EvidenceItem] | None = None,
    ) -> TakerOutput:
        """Run the Taker analysis on *opportunity*.

        Parameters
        ----------
        opportunity : Opportunity
            The opportunity to analyse. Must have ``scores`` populated
            (which the Maker Agent would have filled in).
        evidence_items : list[EvidenceItem] | None
            Supporting evidence items used for risk scoring. If
            ``None``, defaults to an empty list.

        Returns
        -------
        TakerOutput
            The structured analysis result.
        """
        if evidence_items is None:
            evidence_items = []

        risk_breakdown = self._score_risks(opportunity, evidence_items)
        taker_score = self._calculate_taker_score(risk_breakdown)
        taker_confidence = self._determine_confidence(opportunity, evidence_items)
        summary = self._build_summary(
            opportunity,
            risk_breakdown,
            taker_score,
            taker_confidence,
        )
        evidence_ids = sorted(
            {ev.id for ev in evidence_items if ev.id}
            | set(opportunity.evidence_ids)
        )

        return TakerOutput(
            opportunity_id=opportunity.id,
            taker_score=taker_score,
            taker_confidence=taker_confidence,
            risk_breakdown=risk_breakdown,
            evidence_ids=evidence_ids,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal scoring heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_risks(
        opportunity: Opportunity,
        evidence_items: list[EvidenceItem],
    ) -> dict[str, float]:
        """Score each risk category 0–100 using deterministic heuristics.

        The heuristics consider:
        - What vulnerable groups are involved.
        - The speculative flag (weaker claims invite exploitation).
        - Number and quality of evidence items.
        - The current ``harm_risk_score`` if already set on the
          opportunity.
        """

        # --- Base scores derived from opportunity attributes ---

        has_vulnerable = bool(opportunity.vulnerable_groups)
        is_speculative = opportunity.speculative

        # Count evidence by classification.
        weak_evidence_count = sum(
            1
            for ev in evidence_items
            if ev.claim_classification
            in (ClaimClassification.ASSUMPTION, ClaimClassification.UNKNOWN)
        )
        evidence_based_count = sum(
            1
            for ev in evidence_items
            if ev.claim_classification == ClaimClassification.EVIDENCE_BASED
        )

        # Inherit previous harm_risk_score if available.
        existing_harm_risk: float = 0.0
        if opportunity.scores is not None:
            existing_harm_risk = opportunity.scores.harm_risk_score

        # --- Category-specific heuristics ---

        # Extraction risk: vulnerable groups + weak evidence = higher risk.
        extraction_risk = 30.0
        if has_vulnerable:
            extraction_risk += 20.0
        if is_speculative:
            extraction_risk += 15.0
        if weak_evidence_count > 0:
            extraction_risk += weak_evidence_count * _LOW_EVIDENCE_RISK_BOOST
        if evidence_based_count >= 2:
            extraction_risk -= 10.0

        # Gatekeeping risk: opportunities that act as intermediaries.
        gatekeeping_risk = 20.0
        if opportunity.type in (
            "coordination_process",
            "community_support_process",
            "manual_service",
        ):
            gatekeeping_risk += 20.0
        if has_vulnerable:
            gatekeeping_risk += 15.0
        if weak_evidence_count > 0:
            gatekeeping_risk += weak_evidence_count * _MEDIUM_EVIDENCE_RISK_BOOST

        # False authority risk: advocacy and report types risk appearing
        # as authoritative without verification.
        false_authority_risk = 15.0
        if opportunity.type in (
            "advocacy_report",
            "transparency_dashboard",
            "public_guide",
        ):
            false_authority_risk += 25.0
        if is_speculative:
            false_authority_risk += 20.0
        if evidence_based_count < 2:
            false_authority_risk += 10.0

        # Dependency risk: software/tooling and coordination types create
        # ongoing dependency on the entity providing them.
        dependency_risk = 15.0
        if opportunity.type in (
            "software_tooling",
            "coordination_process",
            "open_data_resource",
        ):
            dependency_risk += 25.0
        if has_vulnerable:
            dependency_risk += 15.0
        if is_speculative:
            dependency_risk += 10.0

        # Harm risk: how harmful exploitation would be for affected groups.
        harm_risk = max(existing_harm_risk, 20.0)
        if has_vulnerable:
            harm_risk += 20.0
        if is_speculative:
            harm_risk += 10.0
        if weak_evidence_count > 0:
            harm_risk += weak_evidence_count * _LOW_EVIDENCE_RISK_BOOST

        # Clamp each category to 0–100.
        raw_scores = {
            "extraction_risk": extraction_risk,
            "gatekeeping_risk": gatekeeping_risk,
            "false_authority_risk": false_authority_risk,
            "dependency_risk": dependency_risk,
            "harm_risk": harm_risk,
        }
        clamped = {
            key: max(0.0, min(100.0, score))
            for key, score in raw_scores.items()
        }

        return clamped

    @staticmethod
    def _calculate_taker_score(
        risk_breakdown: dict[str, float],
    ) -> float:
        """Aggregate per-category scores into a single 0–100 taker_score.

        Weighted average giving slightly more weight to harm_risk and
        extraction_risk, which are the most critical from a red-team
        perspective.
        """
        weights = {
            "extraction_risk": 0.25,
            "gatekeeping_risk": 0.15,
            "false_authority_risk": 0.15,
            "dependency_risk": 0.15,
            "harm_risk": 0.30,
        }
        assert set(weights) == set(_RISK_CATEGORIES), \
            "Weight keys must match risk categories"
        weighted_sum = sum(
            risk_breakdown.get(category, 0.0) * weight
            for category, weight in weights.items()
        )
        return round(weighted_sum, 1)

    @staticmethod
    def _determine_confidence(
        opportunity: Opportunity,
        evidence_items: list[EvidenceItem],
    ) -> str:
        """Derive confidence from evidence strength and opportunity scores.
        # TODO: Consider weighting by evidence trust_score, not just evidence_based count
        """
        evidence_based_count = sum(
            1
            for ev in evidence_items
            if ev.claim_classification == ClaimClassification.EVIDENCE_BASED
        )
        total_evidence = len(evidence_items)

        if evidence_based_count >= 3 or total_evidence >= 4:
            return "high"
        if evidence_based_count >= 1 or total_evidence >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _build_summary(
        opportunity: Opportunity,
        risk_breakdown: dict[str, float],
        taker_score: float,
        taker_confidence: str,
    ) -> str:
        """Build a defensive tone summary of identified risks.

        This method must **never** produce exploitation instructions.
        Every phrase is defensive — it flags what *could* go wrong,
        not how to achieve it.
        """
        high_risks = [
            (cat, score)
            for cat, score in risk_breakdown.items()
            if score >= 60
        ]
        medium_risks = [
            (cat, score)
            for cat, score in risk_breakdown.items()
            if 30 <= score < 60
        ]

        lines: list[str] = []

        lines.append(
            f"Taker analysis for **{opportunity.title}** "
            f"(score: {taker_score}/100, confidence: {taker_confidence})."
        )

        if opportunity.vulnerable_groups:
            lines.append(
                f"The opportunity targets {len(opportunity.vulnerable_groups)} "
                f"vulnerable group(s): {', '.join(opportunity.vulnerable_groups)}."
            )

        if high_risks:
            items = [
                f"{cat.replace('_', ' ').title()} ({score}/100)"
                for cat, score in high_risks
            ]
            lines.append(f"High-risk categories: {'; '.join(items)}.")
        else:
            lines.append("No high-risk categories identified.")

        if medium_risks:
            items = [
                f"{cat.replace('_', ' ').title()} ({score}/100)"
                for cat, score in medium_risks
            ]
            lines.append(f"Medium-risk categories: {'; '.join(items)}.")

        lines.append(
            "Defensive recommendations: ensure robust oversight, "
            "verify all claims with independent sources, and "
            "avoid creating single points of failure or dependency "
            "for vulnerable populations."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Output persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_output(
        output: TakerOutput,
        opportunity_slug: str,
        run_dir: Path,
    ) -> tuple[Path, Path]:
        """Write ``taker.json`` and ``taker.md`` to the opportunity folder.

        Parameters
        ----------
        output : TakerOutput
            The analysis result to persist.
        opportunity_slug : str
            Slug identifying the opportunity within the run.
        run_dir : Path
            Path to the run directory (``runs/<run-id>/``).

        Returns
        -------
        tuple[Path, Path]
            ``(json_path, md_path)`` of the written files.
        """
        opportunity_dir = run_dir / "opportunities" / opportunity_slug
        opportunity_dir.mkdir(parents=True, exist_ok=True)

        json_path = opportunity_dir / "taker.json"
        json_path.write_text(
            json.dumps(output.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        md_path = opportunity_dir / "taker.md"
        md_path.write_text(output.to_markdown(), encoding="utf-8")

        return json_path, md_path

    # ------------------------------------------------------------------
    # Convenience: analyse & update opportunity scores
    # ------------------------------------------------------------------

    def analyze_and_update(
        self,
        opportunity: Opportunity,
        evidence_items: list[EvidenceItem] | None = None,
    ) -> tuple[TakerOutput, Opportunity]:
        """Analyse and return a copy of *opportunity* with updated scores.

        The returned ``Opportunity`` will have its ``scores.taker_score``,
        ``scores.taker_confidence``, ``scores.harm_risk_score``, and
        ``scores.rank_score`` updated.
        """
        output = self.analyze(opportunity, evidence_items)

        if opportunity.scores is None:
            raise ValueError(
                "Opportunity must have scores set before Taker analysis. "
                "The Maker Agent should have populated scores first."
            )

        new_harm_risk = output.risk_breakdown["harm_risk"]
        new_rank = ScoreSet.calculate_rank_score(
            people_helped_score=opportunity.scores.people_helped_score,
            severity_score=opportunity.scores.severity_score,
            validity_score=opportunity.scores.validity_score,
            intervention_ease_score=opportunity.scores.intervention_ease_score,
            harm_risk_score=new_harm_risk,
            ability_to_act_score=opportunity.scores.ability_to_act_score,
        )

        new_scores = opportunity.scores.model_copy(
            update={
                "taker_score": output.taker_score,
                "taker_confidence": Confidence(output.taker_confidence),
                "harm_risk_score": new_harm_risk,
                "rank_score": new_rank,
            }
        )
        updated_opportunity = opportunity.model_copy(
            update={"scores": new_scores}
        )
        return output, updated_opportunity
