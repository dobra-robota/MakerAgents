"""Report Agent: re-renders final-report.md from on-disk run state.

The Report Agent reads an existing run folder and regenerates the
``final-report.md`` without invoking any external agents, searches, or
LLM calls.  It is used by the ``maker report`` CLI command.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from makeragents.run import FINAL_REPORT_FILENAME, RUN_YAML_FILENAME
from makeragents.sources.registry import RUN_REGISTRY_RELATIVE_PATH


class ReportAgent:
    """Re-render a final report from on-disk run artifacts."""

    def __init__(self, run_path: Path | str) -> None:
        self.run_path = Path(run_path)

    def render(self) -> str:
        """Read on-disk artifacts and return a complete Markdown report."""

        run_yaml = self._read_run_yaml()
        registry = self._read_registry()
        evidence_items = self._read_evidence()
        opportunities = self._read_opportunities()

        lines: list[str] = []
        lines.append(
            f"# Final report: {run_yaml.get('city', '?')} / "
            f"{run_yaml.get('community', '?')}\n"
        )
        lines.append(
            f"- Run ID: `{run_yaml.get('run_id', 'unknown')}`\n"
            f"- Created at: {run_yaml.get('timestamp', 'unknown')}\n"
            f"- Max opportunities: {run_yaml.get('max_opportunities', '?')}\n"
            f"\n"
        )

        # Source trust registry summary
        if registry:
            lines.append("## Source Trust Registry\n\n")
            default = registry.get("default_unknown_domain_score", "?")
            lines.append(f"- **Default unknown-domain score**: {default}\n\n")
            type_defaults = registry.get("source_type_defaults", {})
            if type_defaults:
                lines.append("### Source-type baseline scores\n\n")
                for stype, score in sorted(type_defaults.items()):
                    lines.append(f"- `{stype}`: {score}\n")
                lines.append("\n")
            domain_overrides = registry.get("domains", {})
            if domain_overrides:
                lines.append("### Per-domain overrides\n\n")
                for domain, score in sorted(domain_overrides.items()):
                    lines.append(f"- `{domain}`: {score}\n")
                lines.append("\n")

        # Evidence summary
        lines.append("## Evidence\n\n")
        if evidence_items:
            lines.append(f"{len(evidence_items)} evidence item(s) collected.\n\n")
            for item in evidence_items:
                lines.append(
                    f"- **{item.get('id', '?')}** "
                    f"`{item.get('evidence_type', '?')}` "
                    f"(source: {item.get('source_domain', '?')}, "
                    f"trust: {item.get('trust_score', '?')})\n"
                )
                snippet = item.get("snippet", "")
                if snippet:
                    lines.append(f"  > {snippet[:200]}\n")
                lines.append("\n")
        else:
            lines.append("_No evidence items collected._\n\n")

        # Opportunities
        lines.append("## Opportunities\n\n")
        if opportunities:
            lines.append(f"{len(opportunities)} opportunity/ies surfaced.\n\n")
            for opp in opportunities:
                lines.append(f"### {opp.get('title', opp.get('id', '?'))}\n\n")
                lines.append(f"- **Type**: `{opp.get('type', '?')}`\n")
                lines.append(f"- **Pain summary**: {opp.get('pain_summary', 'N/A')}\n")
                verdict = opp.get("verdict")
                if verdict:
                    lines.append(f"- **Verdict**: {verdict}\n")
                scores = opp.get("scores")
                if scores:
                    lines.append("- **Scores**:\n")
                    score_keys = [
                        "validity_score",
                        "impact_score",
                        "maker_score",
                        "taker_score",
                        "intervention_ease_score",
                        "harm_risk_score",
                        "ability_to_act_score",
                        "rank_score",
                    ]
                    for key in score_keys:
                        val = scores.get(key)
                        if val is not None:
                            lines.append(f"  - `{key}`: {val}\n")
                lines.append("\n")
        else:
            lines.append("_No opportunities surfaced._\n\n")

        return "".join(lines)

    def write_report(self) -> Path:
        """Render and write ``final-report.md`` into the run folder."""
        report = self.render()
        dest = self.run_path / FINAL_REPORT_FILENAME
        dest.write_text(report, encoding="utf-8")
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_run_yaml(self) -> dict:
        path = self.run_path / RUN_YAML_FILENAME
        if not path.is_file():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _read_registry(self) -> dict:
        path = self.run_path / RUN_REGISTRY_RELATIVE_PATH
        if not path.is_file():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _read_evidence(self) -> list[dict]:
        evidence_dir = self.run_path / "evidence"
        if not evidence_dir.is_dir():
            return []
        items: list[dict] = []
        for fpath in sorted(evidence_dir.glob("*.json")):
            try:
                items.append(json.loads(fpath.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                logging.warning("Skipping unreadable evidence file: %s", fpath)
                continue
        return items

    def _read_opportunities(self) -> list[dict]:
        opp_dir = self.run_path / "opportunities"
        if not opp_dir.is_dir():
            return []
        items: list[dict] = []
        for slug_dir in sorted(opp_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            opp_json = slug_dir / "opportunity.json"
            if opp_json.is_file():
                try:
                    items.append(json.loads(opp_json.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    logging.warning("Skipping unreadable opportunity file: %s", opp_json)
                    continue
        return items
