"""Pipeline orchestrator: wires agents end-to-end via the PRD §6 topology.

Research → Evidence → Opportunity
  → Maker (per opportunity)
    → Taker (after Maker scores and artifacts exist)

Writes the complete PRD §15 folder layout for a run, including per-opportunity
artifacts and ``status.yaml`` for resumability.
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path

import yaml

from makeragents.agents.evidence import EvidenceAgent
from makeragents.agents.maker import MakerAgent, MakerResult
from makeragents.agents.opportunity import OpportunityAgent
from makeragents.agents.research import ResearchAgent
from makeragents.agents.taker import TakerAgent, TakerOutput
from makeragents.config import AppConfig, load_config
from makeragents.llm.client import LLMClient
from makeragents.retry import PIPELINE_STEPS, write_status
from makeragents.run import opportunity_artifact_slug
from makeragents.schemas import (
    Confidence,
    EvidenceItem,
    Opportunity,
    RunMetadata,
    ScoreSet,
)
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
        """Execute the full pipeline and return the final report path.

        Args:
            run_dir: Path to the run folder created by
                :func:`~makeragents.run.create_run_folder`.
            metadata: The run metadata (city, community, max_opportunities).

        Returns:
            The path to ``final-report.md`` as a string.
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

        # 4. Per-opportunity: Maker → Taker. Downstream stages are separate
        # issue scopes and are intentionally not run here.
        max_opps = min(len(opportunities), metadata.max_opportunities)
        selected = opportunities[:max_opps]

        if selected:
            self._process_opportunities(
                selected, evidence_items, run_dir, city, community
            )
        else:
            logger.warning(
                "No opportunities derived — skipping per-opportunity steps"
            )

        return str(run_dir / "final-report.md")

    # ------------------------------------------------------------------
    # Per-opportunity processing
    # ------------------------------------------------------------------

    def _process_opportunities(
        self,
        opportunities: list[Opportunity],
        evidence_items: list[EvidenceItem],
        run_dir: Path,
        city: str,
        community: str,
    ) -> None:
        """Process opportunities concurrently: Maker → Taker."""
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
        """Run Maker, then Taker, for a single opportunity."""

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

        # 4a. Maker must finish and persist artifacts before Taker runs.
        logger.info("  Opportunity %s: Maker", opp.id)

        maker_agent = MakerAgent(llm_client=self._llm)
        maker_result: MakerResult = maker_agent.run_with_llm(
            opp,
            evidence_items,
            city=city,
            community=community,
        )
        maker_json, maker_md = maker_agent.save_output(maker_result, opp_dir)
        if not (maker_json.is_file() and maker_md.is_file()):
            raise RuntimeError(
                f"Maker artifacts missing for opportunity {opp.id}"
            )

        status["steps"]["maker"] = "complete"
        write_status(opp_dir, status)

        scored_opp = self._with_maker_scores(opp, maker_result)
        selected_evidence = self._select_evidence_by_ids(
            evidence_items,
            maker_result.evidence_ids or opp.evidence_ids,
        )
        taker_opp = scored_opp.model_copy(
            update={"evidence_ids": [item.id for item in selected_evidence]}
        )

        # 4b. Taker runs only after Maker scores/artifacts are available.
        logger.info("  Opportunity %s: Taker", opp.id)
        taker_agent = TakerAgent(llm_client=self._llm)
        maker_summary = (
            maker_result.value_add_argument
            or maker_result.summary
            or str(maker_result)
        )
        taker_output: TakerOutput = taker_agent.run_with_llm(
            taker_opp,
            selected_evidence,
            city=city,
            community=community,
            maker_summary=maker_summary,
        )
        taker_agent.save_output(taker_output, opp_dir)

        status["steps"]["taker"] = "complete"
        write_status(opp_dir, status)

    @staticmethod
    def _with_maker_scores(
        opp: Opportunity,
        result: MakerResult,
    ) -> Opportunity:
        """Return *opp* with Maker scores attached for Taker prerequisites."""
        scores = ScoreSet(
            validity_score=result.validity_score,
            maker_score=result.maker_score,
            maker_confidence=result.maker_confidence,
            taker_score=0.0,
            taker_confidence=Confidence.LOW,
            people_helped_score=result.people_helped_score,
            severity_score=result.severity_score,
            impact_score=result.impact_score,
            intervention_ease_score=result.intervention_ease_score,
            harm_risk_score=result.harm_risk_score,
            ability_to_act_score=result.ability_to_act_score,
            rank_score=result.rank_score,
        )
        return opp.model_copy(update={"scores": scores})

    @staticmethod
    def _select_evidence_by_ids(
        evidence_items: list[EvidenceItem],
        evidence_ids: list[str],
    ) -> list[EvidenceItem]:
        """Return evidence items matching *evidence_ids*, preserving input order."""
        wanted = set(evidence_ids)
        if not wanted:
            return []
        return [item for item in evidence_items if item.id in wanted]


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
