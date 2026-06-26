"""Report Agent: reads on-disk run state and produces ranked final report.

The Report Agent consumes opportunity folders full of JSON artifacts
(make.json, taker.json, mediator.json, cost.json, opportunity.yaml) and
produces:

* ``final-report.md`` — ranked executive summary
* ``appendix/rejected-opportunities.md``
* ``appendix/incomplete-opportunities.md``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from makeragents.run import slugify
from makeragents.schemas import Verdict


# ---------------------------------------------------------------------------
# Intermediate typed structures for loaded opportunity data
# ---------------------------------------------------------------------------


@dataclass
class LoadedOpportunity:
    """All the on-disk data for one opportunity, parsed into typed fields."""

    opportunity_id: str
    slug: str
    title: str = ""
    pain_summary: str = ""
    who_benefits: list[str] = field(default_factory=list)
    vulnerable_groups: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    speculative: bool = False

    # Scores (from opportunity.yaml or maker.json)
    scores: dict[str, Any] = field(default_factory=dict)
    verdict: str | None = None

    # Mediator fields
    mediator_summary: str = ""
    do_no_harm: dict[str, Any] = field(default_factory=dict)
    recommended_intervention_shape: str = ""

    # Cost fields
    poc_type: str = ""
    cost_estimate_usd: str = ""
    time_estimate: str = ""
    risk_level: str = ""
    first_3_actions: list[str] = field(default_factory=list)

    # Status tracking
    has_maker: bool = False
    has_taker: bool = False
    has_mediator: bool = False
    has_cost: bool = False
    status_incomplete_steps: list[str] = field(default_factory=list)

    @property
    def rank_score(self) -> float | None:
        """Extract rank_score from scores dict if present."""
        raw = self.scores.get("rank_score")
        if raw is not None:
            return float(raw)
        return None

    @property
    def is_complete(self) -> bool:
        """An opportunity is complete when scores and verdict are present."""
        return bool(
            self.scores
            and self.verdict is not None
            and not self.status_incomplete_steps
        )

    @property
    def is_rejected(self) -> bool:
        """Rejected when verdict is IGNORE, DO_NOT_TOUCH, or NON_INTERVENTION."""
        return self.verdict in {
            Verdict.IGNORE.value,
            Verdict.DO_NOT_TOUCH.value,
            Verdict.NON_INTERVENTION.value,
        }

    @property
    def is_ranked(self) -> bool:
        """Ranked when complete, has rank_score, and not rejected."""
        return (
            self.is_complete
            and self.rank_score is not None
            and not self.is_rejected
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReportAgent:
    """Reads on-disk run artifacts and produces the ranked final report."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, run_dir: Path | str) -> str:
        """Generate ``final-report.md`` and appendix files; return its path."""
        run_path = Path(run_dir)

        metadata = self._load_run_metadata(run_path)
        opportunities = self._discover_and_load(run_path)
        ranked, rejected, incomplete = self._categorize(opportunities)
        evidence_index = self._load_evidence_index(run_path)

        # Sort ranked by rank_score descending
        ranked.sort(key=lambda o: o.rank_score or 0.0, reverse=True)

        # Generate full report
        if not ranked:
            report_md = self._no_valid_opportunities_report(
                metadata, run_path, evidence_index
            )
        else:
            report_md = self._build_full_report(
                metadata, ranked, rejected, incomplete, evidence_index
            )

        # Write final-report.md
        report_path = run_path / "final-report.md"
        report_path.write_text(report_md, encoding="utf-8")

        # Write appendix files
        appendix_dir = run_path / "appendix"
        appendix_dir.mkdir(parents=True, exist_ok=True)

        (appendix_dir / "rejected-opportunities.md").write_text(
            self._build_appendix_rejected(rejected), encoding="utf-8"
        )
        (appendix_dir / "incomplete-opportunities.md").write_text(
            self._build_appendix_incomplete(incomplete), encoding="utf-8"
        )

        return str(report_path)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_run_metadata(run_path: Path) -> dict[str, Any]:
        """Return the contents of ``run.yaml`` as a dict."""
        run_yaml = run_path / "run.yaml"
        if run_yaml.is_file():
            return yaml.safe_load(run_yaml.read_text(encoding="utf-8")) or {}
        return {}

    def _discover_and_load(self, run_path: Path) -> list[LoadedOpportunity]:
        """Find every opportunity folder and load its on-disk state."""
        opps_root = run_path / "opportunities"
        if not opps_root.is_dir():
            return []

        results: list[LoadedOpportunity] = []
        for opp_dir in sorted(opps_root.iterdir()):
            if not opp_dir.is_dir():
                continue
            loaded = self._load_one_opportunity(opp_dir)
            if loaded is not None:
                results.append(loaded)
        return results

    @staticmethod
    def _load_one_opportunity(opp_dir: Path) -> LoadedOpportunity | None:
        """Parse all artifact files in a single opportunity folder."""
        opp_yaml = opp_dir / "opportunity.yaml"
        if not opp_yaml.is_file():
            return None

        opp_data = yaml.safe_load(opp_yaml.read_text(encoding="utf-8")) or {}

        loaded = LoadedOpportunity(
            opportunity_id=opp_data.get("id", opp_dir.name),
            slug=opp_dir.name,
            title=opp_data.get("title", ""),
            pain_summary=opp_data.get("pain_summary", ""),
            who_benefits=opp_data.get("who_benefits", []),
            vulnerable_groups=opp_data.get("vulnerable_groups", []),
            evidence_ids=opp_data.get("evidence_ids", []),
            speculative=opp_data.get("speculative", False),
            scores=opp_data.get("scores") or {},
            verdict=opp_data.get("verdict"),
        )

        # Load maker.json
        maker_file = opp_dir / "maker.json"
        if maker_file.is_file():
            loaded.has_maker = True
            try:
                maker_data = json.loads(maker_file.read_text(encoding="utf-8"))
                # Merge scores from maker into our scores dict
                for key in (
                    "maker_score",
                    "maker_confidence",
                    "people_helped_score",
                    "severity_score",
                    "impact_score",
                    "validity_score",
                    "intervention_ease_score",
                    "harm_risk_score",
                    "ability_to_act_score",
                    "rank_score",
                ):
                    if key in maker_data:
                        loaded.scores[key] = maker_data[key]
                # Merge evidence_ids
                if maker_data.get("evidence_ids"):
                    loaded.evidence_ids = list(
                        dict.fromkeys(loaded.evidence_ids + maker_data["evidence_ids"])
                    )
            except (json.JSONDecodeError, OSError):
                pass

        # Load taker.json
        taker_file = opp_dir / "taker.json"
        if taker_file.is_file():
            loaded.has_taker = True
            try:
                taker_data = json.loads(taker_file.read_text(encoding="utf-8"))
                for key in ("taker_score", "taker_confidence"):
                    if key in taker_data:
                        loaded.scores[key] = taker_data[key]
            except (json.JSONDecodeError, OSError):
                pass

        # Load mediator.json
        mediator_file = opp_dir / "mediator.json"
        if mediator_file.is_file():
            loaded.has_mediator = True
            try:
                med_data = json.loads(mediator_file.read_text(encoding="utf-8"))
                if med_data.get("verdict") and not loaded.verdict:
                    loaded.verdict = med_data["verdict"]
                loaded.mediator_summary = med_data.get("summary", "")
                loaded.do_no_harm = med_data.get("do_no_harm", {})
                loaded.recommended_intervention_shape = med_data.get(
                    "recommended_intervention_shape", ""
                )
                # Merge scores
                for key in ("maker_score", "taker_score"):
                    if key in med_data:
                        loaded.scores[key] = med_data[key]
            except (json.JSONDecodeError, OSError):
                pass

        # Load cost.json
        cost_file = opp_dir / "cost.json"
        if cost_file.is_file():
            loaded.has_cost = True
            try:
                cost_data = json.loads(cost_file.read_text(encoding="utf-8"))
                loaded.poc_type = cost_data.get("poc_type", "")
                loaded.cost_estimate_usd = cost_data.get("cost_estimate_usd", "")
                loaded.time_estimate = cost_data.get("time_estimate", "")
                loaded.risk_level = cost_data.get("risk_level", "")
                loaded.first_3_actions = cost_data.get("first_3_actions", [])
            except (json.JSONDecodeError, OSError):
                pass

        # Load status.yaml (optional, from issue #14)
        status_file = opp_dir / "status.yaml"
        if status_file.is_file():
            try:
                status_data = yaml.safe_load(status_file.read_text(encoding="utf-8"))
                if isinstance(status_data, dict):
                    incomplete = status_data.get("incomplete_steps", [])
                    if isinstance(incomplete, list):
                        loaded.status_incomplete_steps = incomplete
            except (yaml.YAMLError, OSError):
                pass

        return loaded

    @staticmethod
    def _load_evidence_index(run_path: Path) -> dict[str, dict[str, Any]]:
        """Load evidence items from the evidence folder, indexed by ID."""
        evidence_dir = run_path / "evidence"
        if not evidence_dir.is_dir():
            return {}

        index: dict[str, dict[str, Any]] = {}
        for ev_file in evidence_dir.glob("*.json"):
            try:
                items = json.loads(ev_file.read_text(encoding="utf-8"))
                if isinstance(items, list):
                    for item in items:
                        eid = item.get("id")
                        if eid:
                            index[eid] = item
                elif isinstance(items, dict):
                    eid = items.get("id")
                    if eid:
                        index[eid] = items
            except (json.JSONDecodeError, OSError):
                pass
        return index

    # ------------------------------------------------------------------
    # Categorization
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize(
        opportunities: list[LoadedOpportunity],
    ) -> tuple[list[LoadedOpportunity], list[LoadedOpportunity], list[LoadedOpportunity]]:
        """Split opportunities into ranked, rejected, and incomplete groups."""
        ranked: list[LoadedOpportunity] = []
        rejected: list[LoadedOpportunity] = []
        incomplete: list[LoadedOpportunity] = []

        for opp in opportunities:
            if opp.is_ranked:
                ranked.append(opp)
            elif opp.is_rejected:
                rejected.append(opp)
            else:
                incomplete.append(opp)

        return ranked, rejected, incomplete

    # ------------------------------------------------------------------
    # Report builders
    # ------------------------------------------------------------------

    def _build_full_report(
        self,
        metadata: dict[str, Any],
        ranked: list[LoadedOpportunity],
        rejected: list[LoadedOpportunity],
        incomplete: list[LoadedOpportunity],
        evidence_index: dict[str, dict[str, Any]],
    ) -> str:
        """Construct the complete final-report.md content."""
        city = metadata.get("city", "Unknown")
        community = metadata.get("community", "Unknown")
        timestamp = metadata.get("timestamp", "Unknown")

        lines: list[str] = []

        # --- Header ---
        lines.extend(
            [
                f"# Final Report: {city} / {community}",
                "",
                f"- **Run ID:** `{metadata.get('run_id', 'N/A')}`",
                f"- **City:** {city}",
                f"- **Community:** {community}",
                f"- **Generated:** {timestamp}",
                f"- **Ranked opportunities:** {len(ranked)}",
                f"- **Rejected opportunities:** {len(rejected)}",
                f"- **Incomplete opportunities:** {len(incomplete)}",
                "",
            ]
        )

        # --- Ranking Formula ---
        lines.extend(self._ranking_formula_section())
        lines.append("")

        # --- Ranked Opportunities ---
        lines.append("## Ranked Opportunities")
        lines.append("")

        for idx, opp in enumerate(ranked, start=1):
            lines.extend(self._render_ranked_opportunity(idx, opp, evidence_index))

        # --- Top Evidence Sources ---
        lines.extend(self._top_evidence_sources_section(ranked, evidence_index))

        # --- Appendix Links ---
        lines.extend(
            [
                "## Appendix",
                "",
                f"- [Rejected Opportunities](appendix/rejected-opportunities.md) ({len(rejected)})",
                f"- [Incomplete Opportunities](appendix/incomplete-opportunities.md) ({len(incomplete)})",
                "",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _ranking_formula_section() -> list[str]:
        """Return the ranking formula section lines."""
        return [
            "## Ranking Formula",
            "",
            "```text",
            "rank_score =",
            "  people_helped_score     * 0.22 +",
            "  severity_score          * 0.20 +",
            "  validity_score          * 0.18 +",
            "  intervention_ease_score * 0.14 +",
            "  low_harm_score          * 0.14 +",
            "  ability_to_act_score    * 0.12",
            "```",
            "",
            "Where `low_harm_score = 100 - harm_risk_score`.",
        ]

    # pylint: disable=too-many-locals
    def _render_ranked_opportunity(
        self,
        idx: int,
        opp: LoadedOpportunity,
        evidence_index: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Render a single ranked opportunity section."""
        rank_score = opp.rank_score or 0.0
        maker_score = opp.scores.get("maker_score", "—")
        maker_conf = opp.scores.get("maker_confidence", "—")
        taker_score = opp.scores.get("taker_score", "—")
        taker_conf = opp.scores.get("taker_confidence", "—")
        validity = opp.scores.get("validity_score", "—")
        impact = opp.scores.get("impact_score", "—")
        intervention_ease = opp.scores.get("intervention_ease_score", "—")
        harm_risk = opp.scores.get("harm_risk_score", "—")
        ability_to_act = opp.scores.get("ability_to_act_score", "—")
        people_helped = opp.scores.get("people_helped_score", "—")
        severity = opp.scores.get("severity_score", "—")

        lines: list[str] = [
            f"### {idx}. {opp.title} — rank_score: {rank_score:.1f}",
            "",
            f"**Opportunity ID:** `{opp.opportunity_id}`",
            f"**Verdict:** {opp.verdict or '—'}",
            "",
            "#### Pain Summary",
            "",
            opp.pain_summary or "—",
            "",
            "#### Who Benefits",
            "",
            ", ".join(opp.who_benefits) if opp.who_benefits else "—",
            "",
            "#### Scores",
            "",
            "| Score | Value |",
            "|-------|-------|",
            f"| Maker Score | {maker_score} |",
            f"| Maker Confidence | {maker_conf} |",
            f"| Taker Score | {taker_score} |",
            f"| Taker Confidence | {taker_conf} |",
            f"| Validity Score | {validity} |",
            f"| People Helped | {people_helped} |",
            f"| Severity | {severity} |",
            f"| Impact Estimate | {impact} |",
            f"| Intervention Ease | {intervention_ease} |",
            f"| Harm Risk | {harm_risk} |",
            f"| Ability to Act | {ability_to_act} |",
            f"| **Rank Score** | **{rank_score:.1f}** |",
            "",
        ]

        # Mediator summary
        if opp.mediator_summary:
            lines.extend(
                [
                    "#### Mediator Summary",
                    "",
                    opp.mediator_summary,
                    "",
                ]
            )

        # Recommended next action
        if opp.recommended_intervention_shape:
            lines.extend(
                [
                    "#### Recommended Next Action",
                    "",
                    opp.recommended_intervention_shape,
                    "",
                ]
            )

        # POC cost estimate
        if opp.has_cost:
            lines.extend(
                [
                    "#### POC Cost Estimate",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| POC Type | {opp.poc_type} |",
                    f"| Cost (USD) | {opp.cost_estimate_usd} |",
                    f"| Time | {opp.time_estimate} |",
                    f"| Risk Level | {opp.risk_level} |",
                    "",
                ]
                + (
                    [
                        "**First 3 Actions:**",
                        "",
                    ]
                    + [f"{i}. {a}" for i, a in enumerate(opp.first_3_actions, 1)]
                    + [""]
                )
                if opp.first_3_actions
                else []
            )

        # Do No Harm summary
        if opp.do_no_harm:
            lines.extend(
                [
                    "#### Do No Harm Summary",
                    "",
                ]
            )
            for key, value in opp.do_no_harm.items():
                label = key.replace("_", " ").title()
                if isinstance(value, list):
                    value = ", ".join(value)
                lines.append(f"- **{label}:** {value}")
            lines.append("")

        # Evidence references
        if opp.evidence_ids:
            lines.extend(
                [
                    "#### Evidence References",
                    "",
                ]
            )
            for eid in opp.evidence_ids:
                ev = evidence_index.get(eid, {})
                domain = ev.get("source_domain", "—")
                trust = ev.get("trust_score", "—")
                lines.append(f"- `{eid}` — {domain} (trust: {trust})")
            lines.append("")

        # Vulnerable groups
        if opp.vulnerable_groups:
            lines.extend(
                [
                    "#### Vulnerable Groups",
                    "",
                    ", ".join(opp.vulnerable_groups),
                    "",
                ]
            )

        lines.append("---")
        lines.append("")
        return lines

    @staticmethod
    def _top_evidence_sources_section(
        ranked: list[LoadedOpportunity],
        evidence_index: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Build the top evidence sources section across all ranked opportunities."""
        lines: list[str] = [
            "## Top Evidence Sources",
            "",
        ]

        seen: set[str] = set()
        sources: list[tuple[str, str, float]] = []

        for opp in ranked:
            for eid in opp.evidence_ids:
                if eid in seen:
                    continue
                seen.add(eid)
                ev = evidence_index.get(eid, {})
                domain = ev.get("source_domain", "—")
                trust = float(ev.get("trust_score", 0))
                sources.append((eid, domain, trust))

        if not sources:
            lines.append("_No evidence sources found._")
            lines.append("")
            return lines

        # Sort by trust score descending, then by ID
        sources.sort(key=lambda x: (-x[2], x[0]))

        lines.append("| Evidence ID | Source Domain | Trust Score |")
        lines.append("|-------------|---------------|-------------|")
        for eid, domain, trust in sources:
            lines.append(f"| `{eid}` | {domain} | {trust:.1f} |")
        lines.append("")

        return lines

    def _no_valid_opportunities_report(
        self,
        metadata: dict[str, Any],
        run_path: Path,
        evidence_index: dict[str, dict[str, Any]],
    ) -> str:
        """Produce an explanatory report when no ranked opportunities exist."""
        city = metadata.get("city", "Unknown")
        community = metadata.get("community", "Unknown")
        timestamp = metadata.get("timestamp", "Unknown")
        run_id = metadata.get("run_id", "N/A")

        lines: list[str] = [
            f"# Final Report: {city} / {community}",
            "",
            "## Result: No valid opportunities found",
            "",
            f"- **Run ID:** `{run_id}`",
            f"- **City:** {city}",
            f"- **Community:** {community}",
            f"- **Generated:** {timestamp}",
            "",
            "### What was searched",
            "",
            f"The run investigated **{city}** focusing on the **{community}** community.",
            "",
            "### Sources found",
            "",
        ]

        # Summarize sources from evidence
        if evidence_index:
            domains = sorted(
                {ev.get("source_domain", "unknown") for ev in evidence_index.values()}
            )
            lines.append(f"Found **{len(evidence_index)}** evidence items across "
                         f"**{len(domains)}** unique source domains:")
            lines.append("")
            for domain in domains:
                lines.append(f"- {domain}")
            lines.append("")
        else:
            # Try to find sources from sources/ directory
            sources_dir = run_path / "sources"
            if sources_dir.is_dir():
                source_files = list(sources_dir.glob("*"))
                if source_files:
                    lines.append(
                        f"Found **{len(source_files)}** source file(s), "
                        "but no evidence items were extracted."
                    )
                    lines.append("")
                else:
                    lines.append(
                        "No source files were collected during the research phase."
                    )
                    lines.append("")
            else:
                lines.append(
                    "No evidence items or source files were found for this run."
                )
                lines.append("")

        lines.extend(
            [
                "### Why evidence was insufficient",
                "",
                "The run did not produce any ranked opportunities. Possible reasons:",
                "",
                "- Evidence items collected were insufficient to form concrete opportunities.",
                "- All generated opportunities were rejected (IGNORE, DO_NOT_TOUCH, or "
                "NON_INTERVENTION verdicts).",
                "- Some opportunities may be incomplete — check `appendix/incomplete-opportunities.md`.",
                "",
                "### Recommended next search direction",
                "",
                f"- Broaden the search terms for the **{community}** community in **{city}**.",
                "- Try alternative search providers or include regional-language sources.",
                "- Check community-specific forums, local news outlets, or NGO reports.",
                "- Consider refining the community definition to be more specific or broader.",
                "",
                "---",
                "",
                "## Appendix",
                "",
                "- [Incomplete Opportunities](appendix/incomplete-opportunities.md)",
                "- [Rejected Opportunities](appendix/rejected-opportunities.md)",
                "",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _build_appendix_rejected(
        rejected: list[LoadedOpportunity],
    ) -> str:
        """Build the rejected-opportunities.md appendix content."""
        lines: list[str] = [
            "# Rejected Opportunities",
            "",
            f"**Total rejected:** {len(rejected)}",
            "",
        ]

        if not rejected:
            lines.append("_No opportunities were rejected._")
            lines.append("")
            return "\n".join(lines)

        for opp in rejected:
            reason = _reject_reason(opp.verdict)
            lines.extend(
                [
                    f"### {opp.title}",
                    "",
                    f"- **ID:** `{opp.opportunity_id}`",
                    f"- **Verdict:** {opp.verdict or '—'}",
                    f"- **Reason:** {reason}",
                    "",
                ]
            )
            if opp.pain_summary:
                lines.extend(["**Pain Summary:**", "", opp.pain_summary, ""])

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _build_appendix_incomplete(
        incomplete: list[LoadedOpportunity],
    ) -> str:
        """Build the incomplete-opportunities.md appendix content."""
        lines: list[str] = [
            "# Incomplete Opportunities",
            "",
            f"**Total incomplete:** {len(incomplete)}",
            "",
        ]

        if not incomplete:
            lines.append("_No incomplete opportunities._")
            lines.append("")
            return "\n".join(lines)

        for opp in incomplete:
            missing = _missing_steps(opp)
            lines.extend(
                [
                    f"### {opp.title}",
                    "",
                    f"- **ID:** `{opp.opportunity_id}`",
                    f"- **Scores present:** {'Yes' if opp.scores else 'No'}",
                    f"- **Verdict:** {opp.verdict or '—'}",
                    f"- **Incomplete steps:** {', '.join(missing) if missing else 'Unknown'}",
                    "",
                ]
            )
            if opp.pain_summary:
                lines.extend(["**Pain Summary:**", "", opp.pain_summary, ""])

            lines.append("---")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Appendix helpers
# ---------------------------------------------------------------------------


def _reject_reason(verdict: str | None) -> str:
    """Return a human-readable reason for a rejection verdict."""
    reasons: dict[str, str] = {
        Verdict.IGNORE.value: "Maker score too low to justify action.",
        Verdict.DO_NOT_TOUCH.value: (
            "High exploitation risk with insufficient value-add potential."
        ),
        Verdict.NON_INTERVENTION.value: (
            "Issue is real but we are not the right actor to intervene."
        ),
    }
    return reasons.get(verdict or "", "Unknown reason.")


def _missing_steps(opp: LoadedOpportunity) -> list[str]:
    """Return a list of missing agent steps for an incomplete opportunity."""
    if opp.status_incomplete_steps:
        return opp.status_incomplete_steps

    missing: list[str] = []
    if not opp.has_maker:
        missing.append("Maker Agent")
    if not opp.has_taker:
        missing.append("Taker Agent")
    if not opp.has_mediator:
        missing.append("Mediator Agent")
    if not opp.has_cost:
        missing.append("Cost Checker Agent")
    if not opp.scores:
        missing.append("Scoring")
    if not opp.verdict:
        missing.append("Verdict")

    # When no missing steps were detected but we also have no status hints
    # and no obvious completion indicators, fall back to "Unknown".
    if not missing and not opp.scores and not opp.verdict and not opp.has_maker:
        return ["Unknown"]
    return missing
