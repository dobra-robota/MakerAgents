"""Mediator Agent: compares Maker/Taker arguments and assigns a verdict.

Reads maker.json and taker.json from the opportunity folder, compares them,
assigns a Verdict, produces a Do No Harm section, and recommends a safe
intervention shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from makeragents.llm import ChatMessage
from makeragents.prompts import load_prompt
from makeragents.run import opportunity_artifact_slug
from makeragents.schemas import (
    MakerAgentsModel,
    NonEmptyString,
    Opportunity,
    OpportunityType,
    ScoreValue,
    Verdict,
)


# ---------------------------------------------------------------------------
# Verdict determination (heuristic rules)
# ---------------------------------------------------------------------------

def _determine_verdict(
    maker_score: float,
    taker_score: float,
    speculative: bool,
    opp_type: OpportunityType,
) -> Verdict:
    """Apply heuristic verdict rules in priority order."""
    # Rule 1: DO_NOT_TOUCH for high exploitation + low value
    if taker_score >= 80 and maker_score < 50:
        return Verdict.DO_NOT_TOUCH

    # Rule 2: IGNORE for very low maker signal
    if maker_score < 30:
        return Verdict.IGNORE

    # Rule 3: WATCH for speculative with modest maker
    if speculative and maker_score < 50:
        return Verdict.WATCH

    # Rule 4: BUILD_POC or MANUAL_POC for strong maker with low taker
    if maker_score >= 50 and taker_score < 40:
        # Speculative opportunities should not trigger POC recommendations.
        if speculative:
            return Verdict.RESEARCH_MORE
        if opp_type in (OpportunityType.SOFTWARE_TOOLING, OpportunityType.TRANSPARENCY_DASHBOARD):
            return Verdict.BUILD_POC
        return Verdict.MANUAL_POC

    # Rule 5: RESEARCH_MORE for promising but uncertain
    if maker_score >= 40 and taker_score < 60:
        return Verdict.RESEARCH_MORE

    # Rule 6: Default to NON_INTERVENTION
    return Verdict.NON_INTERVENTION


def _build_do_no_harm(opportunity: Opportunity) -> dict:
    """Build a Do No Harm section from opportunity metadata."""
    # NOTE(v0): Most fields contain generic placeholder text.
    # When LLM integration arrives, this should produce opportunity-specific analysis.
    has_vulnerable = bool(opportunity.vulnerable_groups)
    return {
        "vulnerable_groups_affected": (
            opportunity.vulnerable_groups
            if has_vulnerable
            else ["none identified explicitly"]
        ),
        "possible_negative_side_effects": (
            "Interventions may inadvertently shift burden to other groups, "
            "create dependency, or mask underlying structural problems."
        ),
        "abuse_or_exploitation_risks": (
            "Any well-intentioned service can be co-opted by bad actors. "
            "Gatekeeping, data misuse, and dependency creation are the primary risks."
        ),
        "legal_or_tos_concerns": (
            "Verify local regulations, data-protection laws, and platform "
            "terms of service before proceeding. v0 does not assess legal risk."
        ),
        "trust_and_misinformation_risks": (
            "Published guides or dashboards must cite evidence clearly. "
            "Unverified claims risk spreading misinformation."
        ),
        "dependency_risks": (
            "Communities may become reliant on an external intervention. "
            "Plan for sustainability and local ownership from the start."
        ),
        "gatekeeping_risks": (
            "A service or tool could become a gatekeeper if access is "
            "controlled by a single actor. Design for open access."
        ),
        "false_authority_risks": (
            "Publishing analysis may create perceived authority. "
            "Always caveat limitations and invite community correction."
        ),
        "safeguards_required_before_poc": (
            "1. Validate findings with community members directly. "
            "2. Consult local organisations already working on this issue. "
            "3. Establish a feedback mechanism before launching. "
            "4. Document evidence sources and confidence levels transparently."
        ),
    }


def _build_recommended_shape(verdict: Verdict, opp_type: OpportunityType) -> str:
    """Recommend a safe intervention shape based on verdict and type."""
    if verdict in (Verdict.DO_NOT_TOUCH, Verdict.IGNORE):
        return "No intervention recommended."
    if verdict == Verdict.WATCH:
        return "Monitor for additional evidence before acting."
    if verdict == Verdict.RESEARCH_MORE:
        return "Conduct lightweight validation (interviews, surveys, or additional data collection)."
    if verdict == Verdict.MANUAL_POC:
        return f"Test manually with a small-scale {opp_type.value.replace('_', ' ')} before building anything."
    if verdict == Verdict.BUILD_POC:
        return f"Build a minimal {opp_type.value.replace('_', ' ')} prototype and test with real users."
    if verdict == Verdict.NON_INTERVENTION:
        return "This issue is real but we are not the right actor to intervene."
    return "Undetermined — review manually."


def _balance_summary(maker_score: float, taker_score: float) -> str:
    diff = maker_score - taker_score
    if diff > 30:
        return f"Maker substantially outweighs Taker (+{diff:.0f}): value-add case is strong with manageable risk."
    if diff > 10:
        return f"Maker moderately outweighs Taker (+{diff:.0f}): proceed with caution."
    if diff < -30:
        return f"Taker substantially outweighs Maker ({diff:.0f}): exploitation risk dominates value-add potential."
    if diff < 0:
        return f"Taker slightly outweighs Maker ({diff:.0f}): proceed only with strong safeguards."
    return f"Maker and Taker are roughly balanced ({diff:.0f}): needs careful evaluation."


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

class MediatorResult(MakerAgentsModel):
    """Structured output from the Mediator Agent."""

    opportunity_id: NonEmptyString
    verdict: Verdict
    maker_score: ScoreValue
    taker_score: ScoreValue
    balance_summary: str = ""
    do_no_harm: dict = Field(default_factory=dict)
    recommended_intervention_shape: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# LLM response helpers
# ---------------------------------------------------------------------------

_VERDICT_MAP: dict[str, Verdict] = {
    v.value: v for v in Verdict
}


def _parse_llm_mediation(raw: dict) -> dict:
    """Parse and normalise an LLM mediation JSON response.

    Maps the prompt's output keys (PRD §7.6) to the internal
    ``MediatorResult`` field names used by ``_to_markdown``.
    """
    dnh_raw: dict = raw.get("do_no_harm", {})
    return {
        "comparison": raw.get("comparison", ""),
        "verdict": _VERDICT_MAP.get(
            raw.get("verdict", "").strip().upper(), Verdict.RESEARCH_MORE,
        ),
        "do_no_harm": {
            "vulnerable_groups_affected": dnh_raw.get("vulnerable_groups", ""),
            "possible_negative_side_effects": dnh_raw.get("negative_side_effects", ""),
            "abuse_or_exploitation_risks": dnh_raw.get("abuse_risks", ""),
            "legal_or_tos_concerns": dnh_raw.get("legal_concerns", ""),
            "trust_and_misinformation_risks": dnh_raw.get("misinformation_risks", ""),
            "dependency_risks": dnh_raw.get("dependency_risks", ""),
            "gatekeeping_risks": dnh_raw.get("dependency_risks", ""),
            "false_authority_risks": dnh_raw.get("false_authority_risks", ""),
            "safeguards_required_before_poc": dnh_raw.get("safeguards", ""),
        },
        "safe_intervention_shape": raw.get("safe_intervention_shape", ""),
        "evidence_too_weak": raw.get("evidence_too_weak", False),
    }
class MediatorAgent:
    """Compares Maker and Taker arguments and assigns a verdict."""

    def run(self, opportunity: Opportunity) -> MediatorResult:
        """Analyze an opportunity that has completed Maker and Taker scores."""
        scores = opportunity.scores
        if scores is None:
            return MediatorResult(
                opportunity_id=opportunity.id,
                verdict=Verdict.RESEARCH_MORE,
                maker_score=0.0,
                taker_score=0.0,
                balance_summary="No scores available — recommend gathering more evidence.",
                evidence_ids=list(opportunity.evidence_ids),
                summary="Insufficient data for mediation.",
            )

        maker = float(scores.maker_score)
        taker = float(scores.taker_score)
        verdict = _determine_verdict(maker, taker, opportunity.speculative, opportunity.type)
        do_no_harm = _build_do_no_harm(opportunity)
        rec_shape = _build_recommended_shape(verdict, opportunity.type)
        balance = _balance_summary(maker, taker)

        summary = (
            f"Opportunity: {opportunity.title}\n"
            f"Maker: {maker:.1f} | Taker: {taker:.1f}\n"
            f"Verdict: {verdict.value}\n"
            f"Balance: {balance}\n"
            f"Recommendation: {rec_shape}"
        )

        return MediatorResult(
            opportunity_id=opportunity.id,
            verdict=verdict,
            maker_score=maker,
            taker_score=taker,
            balance_summary=balance,
            do_no_harm=do_no_harm,
            recommended_intervention_shape=rec_shape,
            evidence_ids=list(opportunity.evidence_ids),
            summary=summary,
        )


    # ------------------------------------------------------------------
    # LLM-backed mediation
    # ------------------------------------------------------------------

    def run_with_llm(
        self,
        *,
        city: str,
        community: str,
        opportunity: Opportunity,
        maker_summary: str,
        taker_summary: str,
        llm_client: object,
    ) -> MediatorResult:
        """Run LLM-backed mediation: compare Maker/Taker, assign verdict, produce Do No Harm.

        Args:
            city: The city for this run.
            community: The community for this run.
            opportunity: The :class:`~makeragents.schemas.Opportunity` under review.
            maker_summary: Text summary of the Maker argument.
            taker_summary: Text summary of the Taker argument.
            llm_client: An :class:`LLMClient` instance (from ``makeragents.llm``).

        Returns:
            A :class:`MediatorResult` populated from the LLM response.

        Raises:
            LLMProviderError: If the LLM call fails or returns unparseable JSON.
        """
        opp_info = (
            f"Opportunity: {opportunity.title}\n"
            f"Type: {opportunity.type.value}\n"
            f"Pain: {opportunity.pain_summary}\n"
            f"Beneficiaries: {', '.join(opportunity.who_benefits)}\n"
            f"Speculative: {opportunity.speculative}"
        )

        prompt = load_prompt(
            "mediator",
            city=city,
            community=community,
            opportunity_summary=opp_info,
            maker_summary=maker_summary,
            taker_summary=taker_summary,
        )

        response = llm_client.chat_json(
            [ChatMessage("user", prompt)],
            temperature=0.3,
        )

        parsed = _parse_llm_mediation(response)

        scores = opportunity.scores
        maker = float(scores.maker_score) if scores else 0.0
        taker = float(scores.taker_score) if scores else 0.0
        balance = _balance_summary(maker, taker)

        return MediatorResult(
            opportunity_id=opportunity.id,
            verdict=parsed["verdict"],
            maker_score=maker,
            taker_score=taker,
            balance_summary=balance,
            do_no_harm=parsed["do_no_harm"],
            recommended_intervention_shape=parsed["safe_intervention_shape"],
            evidence_ids=list(opportunity.evidence_ids),
            summary=parsed["comparison"],
        )

    def save_output(
        self,
        result: MediatorResult,
        run_dir: Path | str,
    ) -> tuple[Path, Path]:
        """Write mediator.json and mediator.md to the opportunity folder."""
        opp_dir = (
            Path(run_dir)
            / "opportunities"
            / opportunity_artifact_slug(result.opportunity_id)
        )
        opp_dir.mkdir(parents=True, exist_ok=True)

        json_path = opp_dir / "mediator.json"
        json_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        md_path = opp_dir / "mediator.md"
        md_path.write_text(self._to_markdown(result), encoding="utf-8")

        return json_path, md_path

    # ------------------------------------------------------------------
    # Markdown output
    # ------------------------------------------------------------------

    def _to_markdown(self, result: MediatorResult) -> str:
        dnh = result.do_no_harm
        lines = [
            f"# Mediator Report: {result.opportunity_id}",
            "",
            "## Verdict",
            "",
            f"**{result.verdict.value}**",
            "",
            "## Score Balance",
            "",
            f"| Metric | Score |",
            f"|--------|-------|",
            f"| Maker Score | {result.maker_score:.1f} |",
            f"| Taker Score | {result.taker_score:.1f} |",
            "",
            result.balance_summary,
            "",
            "## Recommended Intervention Shape",
            "",
            result.recommended_intervention_shape,
            "",
            "## Do No Harm",
            "",
        ]
        for key in [
            "vulnerable_groups_affected",
            "possible_negative_side_effects",
            "abuse_or_exploitation_risks",
            "legal_or_tos_concerns",
            "trust_and_misinformation_risks",
            "dependency_risks",
            "gatekeeping_risks",
            "false_authority_risks",
            "safeguards_required_before_poc",
        ]:
            label = key.replace("_", " ").title()
            value = dnh.get(key, "")
            if isinstance(value, list):
                value = ", ".join(value)
            lines.append(f"### {label}")
            lines.append("")
            lines.append(str(value))
            lines.append("")

        if result.evidence_ids:
            lines.append("## Evidence Cited")
            lines.append("")
            for eid in result.evidence_ids:
                lines.append(f"- `{eid}`")
            lines.append("")

        return "\n".join(lines) + "\n"
