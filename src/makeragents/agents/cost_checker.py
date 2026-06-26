"""Cost Checker Agent: estimates POC cost, time, risk, and first actions.

Takes an Opportunity with a completed mediator verdict and produces
structured cost estimates as both JSON and Markdown under the
opportunity folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from makeragents.schemas import Opportunity, OpportunityType, POCType, Verdict

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

    def write_artifacts(
        self,
        estimate: CostEstimate,
        run_dir: Path | str,
    ) -> tuple[Path, Path]:
        """Write cost.json and cost.md to the opportunity folder."""
        opp_dir = Path(run_dir) / "opportunities" / estimate.opportunity_id
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
