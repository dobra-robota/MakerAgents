"""Pipeline orchestrator: wires agents end-to-end via the PRD §6 topology.

Research → Evidence → Opportunity
  → Maker / Taker (parallel, per opportunity)
    → Mediator → Cost Checker → Report

Writes the complete PRD §15 folder layout for a run, including per-opportunity
artifacts and ``status.yaml`` for resumability.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from makeragents.agents.cost_checker import CostCheckerAgent
from makeragents.agents.evidence import EvidenceAgent
from makeragents.agents.maker import MakerAgent, MakerResult
from makeragents.agents.mediator import MediatorAgent
from makeragents.agents.opportunity import OpportunityAgent
from makeragents.agents.report import ReportAgent
from makeragents.agents.research import ResearchAgent
from makeragents.agents.taker import TakerAgent, TakerOutput
from makeragents.config import AppConfig, load_config
from makeragents.llm.client import LLMClient
from makeragents.retry import PIPELINE_STEPS, write_status
from makeragents.run import opportunity_artifact_slug
from makeragents.schemas import EvidenceItem, Opportunity, RunMetadata, ScoreSet
from makeragents.search.providers import SearchResult

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs the full MakerAgents pipeline for a city + community pair."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        self._config = cfg
        self._llm = llm_client if llm_client is not None else LLMClient(config=cfg)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        run_dir: Path,
        metadata: RunMetadata,
    ) -> str:
        """Execute Research → Evidence → Opportunity → Maker.

        Args:
            run_dir: Path to the run folder created by
                :func:`~makeragents.run.create_run_folder`.
            metadata: The run metadata (city, community, max_opportunities).

        Returns:
            The existing ``final-report.md`` stub path as a string. Report
            generation belongs to a later pipeline stage.
        """
        city = metadata.city
        community = metadata.community

        # 1. Research
        logger.info("Step 1/7: Research — generating and executing queries")
        research = ResearchAgent(llm_client=self._llm, config=self._config)
        search_output = research.search(run_dir, city, community)

        # Flatten search results for evidence agent
        all_search_results: list[SearchResult] = []
        for qr in search_output.query_results:
            all_search_results.extend(qr.results)

        # 2. Evidence
        logger.info("Step 2/7: Evidence — classifying search results")
        evidence = EvidenceAgent(llm_client=self._llm)
        evidence_items = evidence.process(
            all_search_results,
            city=city,
            community=community,
        )
        evidence.save_evidence(evidence_items, run_dir)

        # 3. Opportunity
        logger.info("Step 3/7: Opportunity — deriving candidate opportunities")
        opportunity_agent = OpportunityAgent(
            metadata.max_opportunities,
            llm_client=self._llm,
            city=city,
            community=community,
        )
        opportunities = opportunity_agent.process(evidence_items, run_dir)

        # 4. Per-opportunity: Maker only.
        max_opps = min(len(opportunities), metadata.max_opportunities)
        selected = opportunities[:max_opps]

        if selected:
            self._process_maker_opportunities(
                selected, evidence_items, run_dir, city, community
            )
        else:
            logger.warning(
                "No opportunities derived — skipping per-opportunity steps"
            )

        # Report generation is intentionally left to a later pipeline stage.
        return str(run_dir / "final-report.md")


    # ------------------------------------------------------------------
    # Per-opportunity processing
    # ------------------------------------------------------------------

    def _process_maker_opportunities(
        self,
        opportunities: list[Opportunity],
        evidence_items: list[EvidenceItem],
        run_dir: Path,
        city: str,
        community: str,
    ) -> None:
        """Run Maker for each opportunity and persist maker artifacts."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._process_maker_opportunity,
                    opp,
                    evidence_items,
                    run_dir,
                    city,
                    community,
                ): opp
                for opp in opportunities
            }
            for future in concurrent.futures.as_completed(futures):
                opp = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(
                        "Maker stage for opportunity %s failed — "
                        "continuing with others",
                        opp.id,
                    )

    def _process_maker_opportunity(
        self,
        opp: Opportunity,
        evidence_items: list[EvidenceItem],
        run_dir: Path,
        city: str,
        community: str,
    ) -> MakerResult:
        """Run Maker for one opportunity using only its cited evidence."""
        slug = opportunity_artifact_slug(opp)
        opp_dir = run_dir / "opportunities" / slug
        opp_dir.mkdir(parents=True, exist_ok=True)

        supporting_evidence = self._select_supporting_evidence(opp, evidence_items)
        maker_agent = MakerAgent(llm_client=self._llm)
        maker_result = maker_agent.run_with_llm(
            opp,
            supporting_evidence,
            city=city,
            community=community,
        )
        maker_agent.save_output(maker_result, opp_dir)
        self._write_maker_scores(opp, opp_dir, maker_result)

        status = {
            "opportunity_id": opp.id,
            "slug": slug,
            "steps": {step: "incomplete" for step in PIPELINE_STEPS},
        }
        for step in ("research", "evidence", "opportunity", "maker"):
            status["steps"][step] = "complete"
        write_status(opp_dir, status)

        return maker_result

    @staticmethod
    def _select_supporting_evidence(
        opp: Opportunity,
        evidence_items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        """Return evidence items cited by ``opp.evidence_ids``, preserving order."""
        by_id = {item.id: item for item in evidence_items}
        return [by_id[eid] for eid in opp.evidence_ids if eid in by_id]

    def _process_opportunities(
        self,
        opportunities: list[Opportunity],
        evidence_items: list[EvidenceItem],
        run_dir: Path,
        city: str,
        community: str,
    ) -> None:
        """Process opportunities concurrently: Maker/Taker → Mediator → Cost."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._process_one_opportunity,
                    opp,
                    evidence_items,
                    run_dir,
                    city,
                    community,
                ): opp
                for opp in opportunities
            }
            for future in concurrent.futures.as_completed(futures):
                opp = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(
                        "Opportunity %s failed — continuing with others",
                        opp.id,
                    )

    def _process_one_opportunity(
        self,
        opp: Opportunity,
        evidence_items: list[EvidenceItem],
        run_dir: Path,
        city: str,
        community: str,
    ) -> None:
        """Run Maker/Taker → Mediator → Cost for a single opportunity."""
        slug = opportunity_artifact_slug(opp)
        opp_dir = run_dir / "opportunities" / slug
        opp_dir.mkdir(parents=True, exist_ok=True)

        # Write README.md and opportunity.yaml
        self._write_opportunity_artifacts(opp, opp_dir)

        # Status tracking
        status = {
            "opportunity_id": opp.id,
            "slug": slug,
            "steps": {step: "incomplete" for step in PIPELINE_STEPS},
        }
        for step in ("research", "evidence", "opportunity"):
            status["steps"][step] = "complete"
        write_status(opp_dir, status)

        # 4a. Maker + Taker in parallel
        logger.info("  Opportunity %s: Maker + Taker (parallel)", opp.id)

        maker_agent = MakerAgent(llm_client=self._llm)
        taker_agent = TakerAgent(llm_client=self._llm)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            maker_future = executor.submit(
                maker_agent.run_with_llm,
                opp,
                evidence_items,
                city=city,
                community=community,
            )
            taker_future = executor.submit(
                taker_agent.run_with_llm,
                opp,
                evidence_items,
                city=city,
                community=community,
            )
            maker_result: MakerResult = maker_future.result()
            taker_output: TakerOutput = taker_future.result()

        maker_agent.save_output(maker_result, opp_dir)
        taker_agent.save_output(taker_output, opp_dir)

        status["steps"]["maker"] = "complete"
        status["steps"]["taker"] = "complete"
        write_status(opp_dir, status)

        # 4b. Mediator
        logger.info("  Opportunity %s: Mediator", opp.id)
        mediator = MediatorAgent()
        maker_summary = getattr(
            maker_result, "value_add_argument", str(maker_result)
        )
        taker_summary = getattr(taker_output, "summary", str(taker_output))
        try:
            mediation = mediator.run_with_llm(
                city=city,
                community=community,
                opportunity=opp,
                maker_summary=maker_summary,
                taker_summary=taker_summary,
                llm_client=self._llm,
            )
        except Exception:
            logger.warning(
                "Mediator LLM call failed — falling back to heuristic"
            )
            mediation = mediator.run(opp)
        mediator.save_output(mediation, opp_dir)
        status["steps"]["mediator"] = "complete"
        write_status(opp_dir, status)

        # 4c. Cost Checker
        logger.info("  Opportunity %s: Cost Checker", opp.id)
        cost = CostCheckerAgent()
        verdict = getattr(mediation, "verdict", None)
        verdict_str = verdict.value if hasattr(verdict, "value") else str(verdict)
        intervention = getattr(
            mediation, "safe_intervention_shape", ""
        )
        try:
            estimate = cost.run_with_llm(
                self._llm,
                opp,
                city=city,
                community=community,
                verdict=verdict_str,
                intervention_shape=intervention,
            )
        except Exception:
            logger.warning(
                "Cost Checker LLM call failed — falling back to heuristic"
            )
            estimate = cost.estimate(opp)
        cost.save_output(estimate, opp_dir)
        status["steps"]["cost_checker"] = "complete"
        write_status(opp_dir, status)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_opportunity_artifacts(opp: Opportunity, opp_dir: Path) -> None:
        """Write README.md and opportunity.yaml for an opportunity."""
        readme = (
            f"# {opp.title}\n\n"
            f"- **ID**: `{opp.id}`\n"
            f"- **Type**: `{opp.type.value}`\n"
            f"- **Speculative**: {opp.speculative}\n\n"
            f"## Pain Summary\n\n{opp.pain_summary}\n\n"
        )
        if opp.who_benefits:
            readme += "## Who Benefits\n\n" + "\n".join(
                f"- {g}" for g in opp.who_benefits
            ) + "\n\n"
        if opp.vulnerable_groups:
            readme += "## Vulnerable Groups\n\n" + "\n".join(
                f"- {g}" for g in opp.vulnerable_groups
            ) + "\n\n"
        if opp.evidence_ids:
            readme += "## Evidence\n\n" + "\n".join(
                f"- `{eid}`" for eid in opp.evidence_ids
            ) + "\n\n"
        (opp_dir / "README.md").write_text(readme, encoding="utf-8")

        opp_yaml = opp.model_dump(mode="json")
        (opp_dir / "opportunity.yaml").write_text(
            yaml.safe_dump(opp_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    @staticmethod
    def _scores_from_maker_result(result: MakerResult) -> ScoreSet:
        """Build a score set from Maker output with downstream scores unset."""
        return ScoreSet(
            validity_score=result.validity_score,
            maker_score=result.maker_score,
            maker_confidence=result.maker_confidence,
            taker_score=0.0,
            taker_confidence="low",
            people_helped_score=result.people_helped_score,
            severity_score=result.severity_score,
            impact_score=result.impact_score,
            intervention_ease_score=result.intervention_ease_score,
            harm_risk_score=result.harm_risk_score,
            ability_to_act_score=result.ability_to_act_score,
            rank_score=result.rank_score,
        )

    @staticmethod
    def _write_maker_scores(
        opp: Opportunity,
        opp_dir: Path,
        result: MakerResult,
    ) -> None:
        """Persist Maker scores back into ``opportunity.yaml``."""
        scores = PipelineRunner._scores_from_maker_result(result)
        payload = opp.model_copy(update={"scores": scores}).model_dump(mode="json")
        dest = opp_dir / "opportunity.yaml"
        if dest.is_file():
            existing = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
            if isinstance(existing, dict):
                existing["scores"] = payload["scores"]
                payload = existing
        dest.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
