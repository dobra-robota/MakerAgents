"""Tests for the prompt loader and discipline scaffolding."""

from __future__ import annotations

import pytest

from makeragents.prompts import load_prompt


# ---------------------------------------------------------------------------
# Basic loader tests
# ---------------------------------------------------------------------------


class TestLoadPrompt:
    def test_loads_research_prompt(self) -> None:
        result = load_prompt(
            "research",
            city="Łodz",
            community="senior citizens",
            max_queries="10",
            languages="English (en), Polish (pl)",
        )
        assert "Research Agent" in result
        assert "Łodz" in result
        assert "senior citizens" in result
        assert "10" in result
        # No leftover placeholder
        assert "${" not in result

    def test_loads_evidence_prompt(self) -> None:
        result = load_prompt(
            "evidence",
            city="Łodz",
            community="senior citizens",
            snippets="- Snippet 1: ...",
        )
        assert "Evidence Agent" in result
        assert "Snippet 1" in result

    def test_loads_opportunity_prompt(self) -> None:
        result = load_prompt(
            "opportunity",
            city="Łodz",
            community="senior citizens",
            max_opportunities="5",
            evidence_summary="- ev-001: ...",
        )
        assert "Opportunity Agent" in result
        assert "ev-001" in result

    def test_loads_maker_prompt(self) -> None:
        result = load_prompt(
            "maker",
            city="Łodz",
            community="senior citizens",
            opportunity_summary="Test opportunity",
        )
        assert "Maker Agent" in result
        assert "Test opportunity" in result

    def test_loads_taker_prompt(self) -> None:
        result = load_prompt(
            "taker",
            city="Łodz",
            community="senior citizens",
            opportunity_summary="Test opportunity",
            maker_summary="Maker says...",
        )
        assert "Taker Agent" in result
        assert "Maker says..." in result

    def test_loads_mediator_prompt(self) -> None:
        result = load_prompt(
            "mediator",
            city="Łodz",
            community="senior citizens",
            opportunity_summary="Test opportunity",
            maker_summary="Maker says...",
            taker_summary="Taker says...",
        )
        assert "Mediator Agent" in result
        assert "Taker says..." in result

    def test_loads_cost_checker_prompt(self) -> None:
        result = load_prompt(
            "cost_checker",
            city="Łodz",
            community="senior citizens",
            opportunity_summary="Test opportunity",
            verdict="WATCH",
            intervention_shape="Monitor and research more",
        )
        assert "Cost Checker Agent" in result
        assert "WATCH" in result

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            load_prompt("nonexistent", city="x", community="y")


# ---------------------------------------------------------------------------
# Discipline tests
# ---------------------------------------------------------------------------


class TestDiscipline:
    """Verify discipline clauses appear in rendered prompts."""

    def test_shared_discipline_in_every_prompt(self) -> None:
        for name in (
            "research", "evidence", "opportunity", "maker",
            "taker", "mediator", "cost_checker",
        ):
            result = load_prompt(
                name,
                city="x",
                community="y",
                max_queries="10",
                snippets="...",
                max_opportunities="5",
                evidence_summary="...",
                opportunity_summary="...",
                maker_summary="...",
                taker_summary="...",
                verdict="WATCH",
                intervention_shape="...",
            )
            assert "Prompt Discipline" in result, f"Missing in {name}"
            assert "evidence_based" in result, f"Missing in {name}"
            assert "unsupported claims" in result, f"Missing in {name}"
            assert "invent sources" in result, f"Missing in {name}"

    def test_taker_has_safety_constraint(self) -> None:
        result = load_prompt(
            "taker",
            city="x",
            community="y",
            opportunity_summary="...",
            maker_summary="...",
        )
        assert "CRITICAL SAFETY CONSTRAINT" in result
        assert "exploitation instructions" in result
        assert "attack vectors" in result

    def test_non_taker_prompts_lack_safety_constraint(self) -> None:
        """Only the Taker prompt gets the extra safety discipline."""
        for name in (
            "research", "evidence", "opportunity", "maker",
            "mediator", "cost_checker",
        ):
            result = load_prompt(
                name,
                city="x",
                community="y",
                max_queries="10",
                snippets="...",
                max_opportunities="5",
                evidence_summary="...",
                opportunity_summary="...",
                maker_summary="...",
                taker_summary="...",
                verdict="WATCH",
                intervention_shape="...",
            )
            assert "CRITICAL SAFETY CONSTRAINT" not in result, (
                f"Safety constraint leaked into {name}"
            )


# ---------------------------------------------------------------------------
# Variable substitution edge cases
# ---------------------------------------------------------------------------


class TestVariableSubstitution:
    def test_unknown_variables_left_intact(self) -> None:
        """safe_substitute leaves unbound variables unchanged."""
        result = load_prompt(
            "research",
            city="Łodz",
            community="senior citizens",
            max_queries="10",
            languages="English (en)",
        )
        # Unknown placeholders are left as-is with safe_substitute.
        # The research prompt uses ${city}, ${community}, ${max_queries},
        # ${languages} — all supplied, so there should be nothing left.
        assert "${" not in result

    def test_substitution_with_dollar_sign(self) -> None:
        """Variables containing literal $ are handled by safe_substitute."""
        result = load_prompt(
            "cost_checker",
            city="x",
            community="y",
            opportunity_summary="Budget: $100",
            verdict="WATCH",
            intervention_shape="...",
        )
        assert "Budget: $100" in result
