"""Cost Checker Agent: estimates POC cost, time, risk, and first actions.

Takes an Opportunity with a completed mediator verdict and produces
structured cost estimates as both JSON and Markdown under the
opportunity folder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from makeragents.run import opportunity_artifact_slug
from makeragents.schemas import Opportunity, OpportunityType, POCType, Verdict

if TYPE_CHECKING:
    from makeragents.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpportunityType → POCType mapping
# ---------------------------------------------------------------------------

_OPPORTUNITY_TO_POC: dict[OpportunityType, POCType] = {
    OpportunityType.PUBLIC_GUIDE: POCType.PUBLIC_GUIDE,
    OpportunityType.COORDINATION_PROCESS: POCType.COORDINATION_PROCESS,
    OpportunityType.ADVOCACY_REPORT: POCType.ADVOCACY_REPORT,
    OpportunityType.TRANSPARENCY_DASHBOARD: POCType.DASHBOARD,
    OpportunityType.MANUAL_SERVICE: POCType.MANUAL_SERVICE,
    OpportunityType.COMMUNITY_SUPPORT_PROCESS: POCType.COORDINATION_PROCESS,
    OpportunityType.SOFTWARE_TOOLING: POCType.SOFTWARE_PROTOTYPE,
    OpportunityType.INSTITUTION_FACING_REPORT: POCType.ADVOCACY_REPORT,
    OpportunityType.OPEN_DATA_RESOURCE: POCType.OPEN_DATA_RESOURCE,
}

# ---------------------------------------------------------------------------
# Cost map: POCType → (usd_range, time_estimate, risk_level)
# ---------------------------------------------------------------------------

_COST_MAP: dict[POCType, tuple[str, str, str]] = {
    POCType.PUBLIC_GUIDE: ("$0–$50", "1 weekend", "low"),
    POCType.MANUAL_SERVICE: ("$50–$300", "1–2 weekends", "medium"),
    POCType.ADVOCACY_REPORT: ("$0–$100", "1–2 weeks", "low"),
    POCType.DASHBOARD: ("$200–$2000", "2–4 weeks", "medium"),
    POCType.AUTOMATION: ("$300–$5000", "2–8 weeks", "high"),
    POCType.SOFTWARE_PROTOTYPE: ("$500–$10000", "4–12 weeks", "high"),
    POCType.COORDINATION_PROCESS: ("$0–$200", "1–4 weeks", "medium"),
    POCType.OPEN_DATA_RESOURCE: ("$0–$150", "1–3 weeks", "low"),
}

# ---------------------------------------------------------------------------
# First-3-actions templates per POC type
# ---------------------------------------------------------------------------

_ACTION_TEMPLATES: dict[POCType, list[str]] = {
    POCType.PUBLIC_GUIDE: [
        "Collect and verify the top 3 questions or pain points from community sources.",
        "Draft the guide content with clear, actionable steps and evidence references.",
        "Publish the guide in an accessible format and invite community feedback.",
    ],
    POCType.MANUAL_SERVICE: [
        "Document the manual workflow: inputs, process steps, outputs, and time required.",
        "Run a small trial (3–5 cases) and record outcomes, pain points, and costs.",
        "Collect feedback from trial participants and adjust the process.",
    ],
    POCType.ADVOCACY_REPORT: [
        "Compile key evidence and statistics from verified sources.",
        "Draft the report with clear findings, data visualisation, and recommendations.",
        "Share the draft with relevant organisations and incorporate feedback before publishing.",
    ],
    POCType.DASHBOARD: [
        "Identify the key metrics and data sources needed for the dashboard.",
        "Build a minimal prototype (spreadsheet or simple web view) with real data.",
        "Test with 3–5 stakeholders and iterate on the visualisation and data freshness.",
    ],
    POCType.AUTOMATION: [
        "Map the current manual process end-to-end with inputs, steps, and outputs.",
        "Identify the highest-impact step to automate first and build a minimal script.",
        "Test the automation on a small batch and measure time/cost savings.",
    ],
    POCType.SOFTWARE_PROTOTYPE: [
        "Define the minimum viable feature set (1–2 core features only).",
        "Build a working prototype with placeholder data and minimal UI.",
        "Test with 3–5 real users and collect structured feedback.",
    ],
    POCType.COORDINATION_PROCESS: [
        "Map the current coordination gaps: who needs to talk to whom, and where it breaks.",
        "Design a lightweight coordination workflow (shared doc, regular check-in, or simple tool).",
        "Pilot the workflow with 3–5 participants and measure improvement.",
    ],
    POCType.OPEN_DATA_RESOURCE: [
        "Identify the most valuable dataset and confirm it can be shared legally.",
        "Clean, structure, and document the dataset with a clear schema and license.",
        "Publish the dataset with a README and invite community contributions.",
    ],
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostEstimate:
    """Structured cost estimate for an opportunity's POC."""

    opportunity_id: str
    poc_type: POCType
    cost_estimate_usd: str
    time_estimate: str
    risk_level: str
    first_3_actions: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "poc_type": self.poc_type.value,
            "cost_estimate_usd": self.cost_estimate_usd,
            "time_estimate": self.time_estimate,
            "risk_level": self.risk_level,
            "first_3_actions": self.first_3_actions,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CostCheckerAgent:
    """Estimates POC cost, time, risk, and first actions for an opportunity."""

    @staticmethod
    def opportunity_type_to_poc_type(opp_type: OpportunityType) -> POCType:
        """Map an OpportunityType to the corresponding POCType."""
        try:
            return _OPPORTUNITY_TO_POC[opp_type]
        except KeyError:
            raise KeyError(f"No POC type mapping for opportunity type: {opp_type}")

    @staticmethod
    def map_opportunity_type(opp_type: OpportunityType) -> tuple[str, str, str]:
        """Return (usd_range, time_estimate, risk_level) for an opportunity type."""
        poc = _OPPORTUNITY_TO_POC.get(opp_type)
        if poc is None:
            raise KeyError(f"No POC type mapping for: {opp_type}")
        cost_info = _COST_MAP.get(poc)
        if cost_info is None:
            raise KeyError(f"No cost info for POC type: {poc}")
        return cost_info

    @staticmethod
    def generate_first_3_actions(poc_type: POCType) -> list[str]:
        """Return first-3-actions template for a POC type."""
        return list(_ACTION_TEMPLATES.get(poc_type, [
            "Research the problem space and existing solutions.",
            "Design a minimal approach and gather required resources.",
            "Execute a small trial and gather feedback.",
        ]))

    def estimate(self, opportunity: Opportunity) -> CostEstimate:
        """Produce a cost estimate for an opportunity."""
        poc_type = self.opportunity_type_to_poc_type(opportunity.type)
        usd, time_, risk = self.map_opportunity_type(opportunity.type)
        actions = self.generate_first_3_actions(poc_type)

        notes = ""
        if opportunity.scores and opportunity.scores.maker_score < 30:
            notes = "Low maker score — POC may not be justified."
        elif opportunity.speculative:
            notes = "Opportunity is speculative — validate evidence before investing."

        return CostEstimate(
            opportunity_id=opportunity.id,
            poc_type=poc_type,
            cost_estimate_usd=usd,
            time_estimate=time_,
            risk_level=risk,
            first_3_actions=actions,
            notes=notes,
        )

    @staticmethod
    def _build_opportunity_summary(opportunity: Opportunity) -> str:
        """Build a plain-text summary of the opportunity for the LLM prompt."""
        parts = [
            f"Opportunity ID: {opportunity.id}",
            f"Title: {opportunity.title}",
            f"Type: {opportunity.type.value}",
            f"Pain summary: {opportunity.pain_summary}",
        ]
        if opportunity.who_benefits:
            parts.append(f"Who benefits: {', '.join(opportunity.who_benefits)}")
        if opportunity.vulnerable_groups:
            parts.append(
                f"Vulnerable groups: {', '.join(opportunity.vulnerable_groups)}"
            )
        if opportunity.scores:
            parts.append(f"Maker score: {opportunity.scores.maker_score}")
            parts.append(f"Taker score: {opportunity.scores.taker_score}")
        if opportunity.speculative:
            parts.append("WARNING: This opportunity is speculative.")
        return "\n".join(parts)

    def run_with_llm(
        self,
        llm: LLMClient,
        opportunity: Opportunity,
        *,
        city: str,
        community: str,
        verdict: str,
        intervention_shape: str,
    ) -> CostEstimate:
        """Estimate POC cost using the LLM with the cost_checker prompt.

        Falls back to the heuristic :meth:`estimate` if the LLM call
        fails or returns unparseable output.

        Args:
            llm: An :class:`~makeragents.llm.client.LLMClient` instance.
            opportunity: The opportunity to estimate costs for.
            city: The run city (e.g. ``"Łodz"``).
            community: The run community (e.g. ``"senior citizens"``).
            verdict: The mediator verdict string.
            intervention_shape: The mediator safe intervention shape.

        Returns:
            A :class:`CostEstimate` with LLM-generated estimates.
        """
        from makeragents.llm.client import ChatMessage
        from makeragents.prompts import load_prompt

        opportunity_summary = self._build_opportunity_summary(opportunity)

        try:
            prompt = load_prompt(
                "cost_checker",
                city=city,
                community=community,
                opportunity_summary=opportunity_summary,
                verdict=verdict,
                intervention_shape=intervention_shape,
            )
        except FileNotFoundError:
            logger.warning(
                "cost_checker prompt not found; falling back to heuristic estimate"
            )
            return self.estimate(opportunity)

        try:
            result = llm.chat_json(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning(
                "LLM call failed for %s: %s; falling back to heuristic estimate",
                opportunity.id,
                exc,
            )
            return self.estimate(opportunity)

        return self._parse_llm_response(result, opportunity)

    def _parse_llm_response(
        self,
        result: dict,
        opportunity: Opportunity,
    ) -> CostEstimate:
        """Parse the LLM JSON response into a :class:`CostEstimate`.

        Falls back to the heuristic :meth:`estimate` if the response is
        missing required fields or contains unrecognised values.
        """
        # Non-actionable verdicts → N/A cost, no POC recommended.
        non_actionable = {"IGNORE", "DO_NOT_TOUCH", "NON_INTERVENTION"}
        if result.get("cost_range", "") == "N/A":
            return CostEstimate(
                opportunity_id=opportunity.id,
                poc_type=POCType.PUBLIC_GUIDE,
                cost_estimate_usd="N/A",
                time_estimate="N/A",
                risk_level="N/A",
                first_3_actions=[],
                notes="Mediator verdict does not recommend a POC.",
            )

        try:
            poc_type = POCType(result.get("poc_type", ""))
        except (ValueError, KeyError):
            logger.warning(
                "Unrecognised poc_type %r in LLM response; falling back",
                result.get("poc_type"),
            )
            return self.estimate(opportunity)

        cost_range = result.get("cost_range", "")
        time_est = result.get("time_est", "")
        risk_level = result.get("risk_level", "medium")
        first_actions = result.get("first_actions", [])

        if not cost_range or not time_est or not isinstance(first_actions, list):
            logger.warning(
                "LLM response missing required fields; falling back"
            )
            return self.estimate(opportunity)

        notes = ""
        if opportunity.scores and opportunity.scores.maker_score < 30:
            notes = "Low maker score — POC may not be justified."
        elif opportunity.speculative:
            notes = "Opportunity is speculative — validate evidence before investing."

        # Take only the first 3 actions.
        actions = first_actions[:3]
        while len(actions) < 3:
            actions.append("(additional action not provided by LLM)")

        return CostEstimate(
            opportunity_id=opportunity.id,
            poc_type=poc_type,
            cost_estimate_usd=cost_range,
            time_estimate=time_est,
            risk_level=risk_level,
            first_3_actions=actions,
            notes=notes,
        )

    def write_artifacts(
        self,
        estimate: CostEstimate,
        run_dir: Path | str,
    ) -> tuple[Path, Path]:
        """Write cost.json and cost.md to the opportunity folder."""
        opp_dir = (
            Path(run_dir)
            / "opportunities"
            / opportunity_artifact_slug(estimate.opportunity_id)
        )
        opp_dir.mkdir(parents=True, exist_ok=True)

        cost_json = opp_dir / "cost.json"
        cost_json.write_text(
            json.dumps(estimate.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        cost_md = opp_dir / "cost.md"
        cost_md.write_text(self._to_markdown(estimate), encoding="utf-8")

        return cost_json, cost_md

    def _to_markdown(self, estimate: CostEstimate) -> str:
        lines = [
            f"# Cost Estimate: {estimate.opportunity_id}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| POC Type | {estimate.poc_type.value} |",
            f"| Cost (USD) | {estimate.cost_estimate_usd} |",
            f"| Time | {estimate.time_estimate} |",
            f"| Risk Level | {estimate.risk_level} |",
            "",
            "## First 3 Actions",
            "",
        ]
        for i, action in enumerate(estimate.first_3_actions, start=1):
            lines.append(f"{i}. {action}")

        if estimate.notes:
            lines.extend(["", "## Notes", "", estimate.notes])

        return "\n".join(lines) + "\n"
