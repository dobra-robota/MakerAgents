"""Tests for the Mediator Agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from makeragents.agents.mediator import MediatorAgent, MediatorResult
from makeragents.run import slugify
from makeragents.schemas import (
    Confidence,
    Opportunity,
    OpportunityType,
    ScoreSet,
    Verdict,
)


def _make_opportunity(
    id_: str = "OPP-test",
    o_type: OpportunityType = OpportunityType.PUBLIC_GUIDE,
    maker: float = 70.0,
    taker: float = 30.0,
    speculative: bool = False,
    evidence_ids: list[str] | None = None,
    vulnerable: list[str] | None = None,
) -> Opportunity:
    return Opportunity(
        id=id_,
        title="Test",
        type=o_type,
        pain_summary="pain",
        who_benefits=["residents"],
        vulnerable_groups=vulnerable or [],
        evidence_ids=evidence_ids or ["EVID-001"],
        speculative=speculative,
        scores=ScoreSet(
            validity_score=70.0,
            maker_score=maker,
            maker_confidence=Confidence.MEDIUM,
            taker_score=taker,
            taker_confidence=Confidence.MEDIUM,
            people_helped_score=60.0,
            severity_score=50.0,
            impact_score=55.0,
            intervention_ease_score=60.0,
            harm_risk_score=30.0,
            ability_to_act_score=50.0,
            rank_score=ScoreSet.calculate_rank_score(
                people_helped_score=60.0,
                severity_score=50.0,
                validity_score=70.0,
                intervention_ease_score=60.0,
                harm_risk_score=30.0,
                ability_to_act_score=50.0,
            ),
        ),
    )


class TestVerdictRules:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_do_not_touch_high_taker_low_maker(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=40.0, taker=85.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.DO_NOT_TOUCH

    def test_ignore_low_maker(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=20.0, taker=10.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.IGNORE

    def test_watch_speculative_low_maker(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=40.0, taker=20.0, speculative=True)
        result = agent.run(opp)
        assert result.verdict == Verdict.WATCH

    def test_build_poc_software(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(
            maker=60.0, taker=20.0, o_type=OpportunityType.SOFTWARE_TOOLING
        )
        result = agent.run(opp)
        assert result.verdict == Verdict.BUILD_POC

    def test_manual_poc_non_software(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=60.0, taker=20.0, o_type=OpportunityType.PUBLIC_GUIDE)
        result = agent.run(opp)
        assert result.verdict == Verdict.MANUAL_POC

    def test_research_more(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=45.0, taker=50.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.RESEARCH_MORE

    def test_non_intervention_default(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=35.0, taker=65.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.NON_INTERVENTION

    def test_taker_at_threshold(self, agent: MediatorAgent) -> None:
        # taker >= 80, maker just under 50 → DO_NOT_TOUCH
        opp = _make_opportunity(maker=49.0, taker=80.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.DO_NOT_TOUCH

    def test_maker_at_threshold(self, agent: MediatorAgent) -> None:
        # maker >= 50, taker < 40 → MANUAL_POC
        opp = _make_opportunity(maker=50.0, taker=39.0)
        result = agent.run(opp)
        assert result.verdict == Verdict.MANUAL_POC

    def test_speculative_bypasses_build_poc(self, agent: MediatorAgent) -> None:
        # speculative=True with maker >= 50 and taker < 40 would hit
        # rule 4 (BUILD_POC/MANUAL_POC), but should be RESEARCH_MORE
        opp = _make_opportunity(
            maker=60.0, taker=20.0, speculative=True, o_type=OpportunityType.PUBLIC_GUIDE
        )
        result = agent.run(opp)
        assert result.verdict == Verdict.RESEARCH_MORE


class TestDoNoHarm:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_all_fields_present(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        dnh = result.do_no_harm
        expected_keys = {
            "vulnerable_groups_affected",
            "possible_negative_side_effects",
            "abuse_or_exploitation_risks",
            "legal_or_tos_concerns",
            "trust_and_misinformation_risks",
            "dependency_risks",
            "gatekeeping_risks",
            "false_authority_risks",
            "safeguards_required_before_poc",
        }
        assert set(dnh.keys()) == expected_keys

    def test_vulnerable_groups_from_opportunity(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(vulnerable=["elderly", "children"])
        result = agent.run(opp)
        assert "elderly" in result.do_no_harm["vulnerable_groups_affected"]


class TestBalanceSummary:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_maker_outweighs(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=80.0, taker=20.0)
        result = agent.run(opp)
        assert "substantially outweighs" in result.balance_summary

    def test_taker_outweighs(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=20.0, taker=80.0)
        result = agent.run(opp)
        assert "substantially outweighs" in result.balance_summary

    def test_balanced(self, agent: MediatorAgent) -> None:
        opp = _make_opportunity(maker=50.0, taker=50.0)
        result = agent.run(opp)
        assert "roughly balanced" in result.balance_summary


class TestSaveOutput:
    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    def test_writes_files(self, agent: MediatorAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        json_path, md_path = agent.save_output(result, tmp_path)
        assert json_path.exists()
        assert md_path.exists()

    def test_json_serializable(self, agent: MediatorAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        json_path, _ = agent.save_output(result, tmp_path)
        data = json.loads(json_path.read_text())
        assert data["verdict"] == "MANUAL_POC"
        assert "do_no_harm" in data

    def test_md_contains_verdict(self, agent: MediatorAgent, tmp_path: Path) -> None:
        opp = _make_opportunity()
        result = agent.run(opp)
        _, md_path = agent.save_output(result, tmp_path)
        content = md_path.read_text()
        assert "MANUAL_POC" in content




class TestRunWithLLM:
    """Tests for the LLM-backed run_with_llm method."""

    @pytest.fixture
    def agent(self) -> MediatorAgent:
        return MediatorAgent()

    @pytest.fixture
    def opportunity(self) -> Opportunity:
        return _make_opportunity(
            id_="OPP-llm",
            maker=65.0,
            taker=25.0,
        )

    @staticmethod
    def _make_llm_client(json_response: dict) -> object:
        """Return a mock LLMClient whose chat_json returns *json_response*."""
        import httpx

        payload = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps(json_response),
                },
            }],
            "model": "deepseek-chat",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        from makeragents.config import AppConfig
        from makeragents.llm.client import LLMClient

        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(transport=transport)
        cfg = AppConfig(deepseek_api_key="test-key")
        return LLMClient(config=cfg, http_client=http_client)

    _LLM_DNH = {
        "vulnerable_groups": "Elderly residents with limited digital access.",
        "negative_side_effects": "May shift burden to family caregivers.",
        "abuse_risks": "Low — information-only intervention.",
        "legal_concerns": "No PII collection; GDPR safe.",
        "misinformation_risks": "Moderate — must cite sources.",
        "dependency_risks": "Low — designed for self-service.",
        "false_authority_risks": "Must disclaim official endorsement.",
        "safeguards": "Community review panel before publishing.",
    }

    _FULL_RESPONSE = {
        "comparison": "Maker value-add is strong (65) vs manageable Taker risk (25).",
        "verdict": "MANUAL_POC",
        "do_no_harm": _LLM_DNH,
        "safe_intervention_shape": "Publish a community-reviewed public guide.",
        "evidence_too_weak": False,
    }

    def test_run_with_llm_returns_mediator_result(
        self, agent: MediatorAgent, opportunity: Opportunity,
    ) -> None:
        llm = self._make_llm_client(self._FULL_RESPONSE)
        result = agent.run_with_llm(
            city="Łodz",
            community="senior citizens",
            opportunity=opportunity,
            maker_summary="Maker: strong value-add case.",
            taker_summary="Taker: minimal risk.",
            llm_client=llm,
        )
        assert isinstance(result, MediatorResult)
        assert result.verdict == Verdict.MANUAL_POC
        assert result.maker_score == 65.0
        assert result.taker_score == 25.0
        assert "strong" in result.summary

    def test_do_no_harm_populated_from_llm(
        self, agent: MediatorAgent, opportunity: Opportunity,
    ) -> None:
        llm = self._make_llm_client(self._FULL_RESPONSE)
        result = agent.run_with_llm(
            city="Łodz",
            community="senior citizens",
            opportunity=opportunity,
            maker_summary="Maker summary",
            taker_summary="Taker summary",
            llm_client=llm,
        )
        dnh = result.do_no_harm
        expected_keys = {
            "vulnerable_groups_affected",
            "possible_negative_side_effects",
            "abuse_or_exploitation_risks",
            "legal_or_tos_concerns",
            "trust_and_misinformation_risks",
            "dependency_risks",
            "gatekeeping_risks",
            "false_authority_risks",
            "safeguards_required_before_poc",
        }
        assert set(dnh.keys()) == expected_keys
        # Verify LLM content flowed through
        assert "Elderly" in dnh["vulnerable_groups_affected"]
        assert "burden" in dnh["possible_negative_side_effects"]
        assert "GDPR" in dnh["legal_or_tos_concerns"]
        assert "Community review" in dnh["safeguards_required_before_poc"]

    def test_save_output_after_run_with_llm(
        self, agent: MediatorAgent, opportunity: Opportunity, tmp_path: Path,
    ) -> None:
        llm = self._make_llm_client(self._FULL_RESPONSE)
        result = agent.run_with_llm(
            city="Łodz",
            community="senior citizens",
            opportunity=opportunity,
            maker_summary="Maker summary",
            taker_summary="Taker summary",
            llm_client=llm,
        )
        json_path, md_path = agent.save_output(result, tmp_path)
        assert json_path.exists()
        assert md_path.exists()

        # Check JSON content
        data = json.loads(json_path.read_text())
        assert data["verdict"] == "MANUAL_POC"
        assert "do_no_harm" in data

        # Check Markdown content
        md_content = md_path.read_text()
        assert "MANUAL_POC" in md_content
        assert "# Mediator Report" in md_content
        assert "## Do No Harm" in md_content
        assert "Elderly" in md_content
        assert "GDPR" in md_content

    def test_default_verdict_for_unrecognised_string(
        self, agent: MediatorAgent, opportunity: Opportunity,
    ) -> None:
        resp = dict(self._FULL_RESPONSE, verdict="BOGUS")
        llm = self._make_llm_client(resp)
        result = agent.run_with_llm(
            city="x",
            community="y",
            opportunity=opportunity,
            maker_summary="m",
            taker_summary="t",
            llm_client=llm,
        )
        assert result.verdict == Verdict.RESEARCH_MORE

    def test_evidence_too_weak_flag_in_llm_response(
        self, agent: MediatorAgent, opportunity: Opportunity,
    ) -> None:
        """The evidence_too_weak key is parsed but not stored directly;
        the LLM should convey this through the comparison text and verdict."""
        resp = dict(self._FULL_RESPONSE, evidence_too_weak=True, verdict="WATCH")
        llm = self._make_llm_client(resp)
        result = agent.run_with_llm(
            city="x",
            community="y",
            opportunity=opportunity,
            maker_summary="m",
            taker_summary="t",
            llm_client=llm,
        )
        # Weak evidence should be reflected in verdict choice
        assert result.verdict == Verdict.WATCH

    def test_prompts_loaded_with_correct_city_community(
        self, agent: MediatorAgent, opportunity: Opportunity,
    ) -> None:
        """Sanity check: the loaded prompt includes our substitutions."""
        resp = dict(self._FULL_RESPONSE)
        llm = self._make_llm_client(resp)
        result = agent.run_with_llm(
            city="Gdynia",
            community="students",
            opportunity=opportunity,
            maker_summary="M",
            taker_summary="T",
            llm_client=llm,
        )
        assert result.verdict == Verdict.MANUAL_POC

    def test_passes_opportunity_type_to_llm(
        self, agent: MediatorAgent, opportunity: Opportunity,
    ) -> None:
        """The opportunity's type should be part of the prompt."""
        resp = dict(self._FULL_RESPONSE)
        llm = self._make_llm_client(resp)
        opp = _make_opportunity(id_="OPP-TYPE", o_type=OpportunityType.PUBLIC_GUIDE)
        result = agent.run_with_llm(
            city="x", community="y",
            opportunity=opp,
            maker_summary="M", taker_summary="T",
            llm_client=llm,
        )
        assert result.verdict == Verdict.MANUAL_POC

    def test_no_scores_fallback(
        self, agent: MediatorAgent,
    ) -> None:
        """When scores are None, maker/taker are set to zero."""
        resp = dict(self._FULL_RESPONSE)
        llm = self._make_llm_client(resp)
        opp = _make_opportunity(id_="OPP-NOSCORE")
        opp.scores = None
        result = agent.run_with_llm(
            city="x", community="y",
            opportunity=opp,
            maker_summary="M", taker_summary="T",
            llm_client=llm,
        )
        assert result.maker_score == 0.0
        assert result.taker_score == 0.0

class TestMediatorResult:
    def test_model_dump_json(self) -> None:
        result = MediatorResult(
            opportunity_id="OPP-1",
            verdict=Verdict.MANUAL_POC,
            maker_score=65.0,
            taker_score=25.0,
            balance_summary="Maker outweighs",
            do_no_harm={"test": "data"},
            recommended_intervention_shape="Try it",
            evidence_ids=["EVID-001"],
            summary="Test",
        )
        d = result.model_dump(mode="json")
        assert d["verdict"] == "MANUAL_POC"
        assert d["maker_score"] == 65.0
