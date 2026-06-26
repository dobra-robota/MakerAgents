"""Tests for user override parsing, application, and markdown formatting."""

from __future__ import annotations

import pytest

from makeragents.overrides import (
    UserOverride,
    apply_score_overrides,
    apply_trust_overrides,
    apply_verdict_override,
    format_override_markdown,
    has_overrides,
    parse_overrides,
    strip_overrides,
)
from makeragents.schemas import (
    Confidence,
    Opportunity,
    OpportunityType,
    ScoreSet,
    Verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores(**overrides: float) -> ScoreSet:
    defaults: dict[str, float] = {
        "validity_score": 61,
        "maker_score": 70,
        "maker_confidence": Confidence.MEDIUM,
        "taker_score": 30,
        "taker_confidence": Confidence.LOW,
        "people_helped_score": 70,
        "severity_score": 80,
        "impact_score": 72,
        "intervention_ease_score": 60,
        "harm_risk_score": 20,
        "ability_to_act_score": 75,
        "rank_score": 0,
    }
    defaults["rank_score"] = ScoreSet.calculate_rank_score(
        people_helped_score=defaults["people_helped_score"],
        severity_score=defaults["severity_score"],
        validity_score=defaults["validity_score"],
        intervention_ease_score=defaults["intervention_ease_score"],
        harm_risk_score=defaults["harm_risk_score"],
        ability_to_act_score=defaults["ability_to_act_score"],
    )
    defaults.update(overrides)  # type: ignore[arg-type]
    return ScoreSet(**defaults)


def _make_opportunity(
    *,
    scores: ScoreSet | None = None,
    verdict: Verdict | None = Verdict.RESEARCH_MORE,
) -> Opportunity:
    return Opportunity(
        id="senior-services-guide",
        title="Plain-language senior services guide",
        type=OpportunityType.PUBLIC_GUIDE,
        pain_summary="Residents may struggle to find the right city service contact.",
        who_benefits=["senior citizens", "caregivers"],
        vulnerable_groups=["older adults"],
        evidence_ids=["EV-001"],
        speculative=False,
        scores=scores or _make_scores(),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# parse_overrides
# ---------------------------------------------------------------------------


class TestParseOverrides:
    def test_parses_single_override(self) -> None:
        md = (
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
            "> Reason: user manually confirmed source relevance.\n"
        )
        overrides = parse_overrides(md)
        assert len(overrides) == 1
        ov = overrides[0]
        assert ov.field_name == "validity_score"
        assert ov.old_value == "61"
        assert ov.new_value == "75"
        assert ov.reason == "user manually confirmed source relevance."

    def test_parses_multiple_overrides(self) -> None:
        md = (
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
            "> Reason: confirmed source.\n"
            "\n"
            "> User Override:\n"
            "> verdict changed from RESEARCH_MORE to BUILD_POC.\n"
            "> Reason: community urgently needs this.\n"
        )
        overrides = parse_overrides(md)
        assert len(overrides) == 2
        assert overrides[0].field_name == "validity_score"
        assert overrides[1].field_name == "verdict"

    def test_parses_all_override_fields(self) -> None:
        for field in ("validity_score", "ranking", "verdict", "source_trust"):
            md = (
                f"> User Override:\n"
                f"> {field} changed from old to new.\n"
                f"> Reason: test.\n"
            )
            overrides = parse_overrides(md)
            assert len(overrides) == 1
            assert overrides[0].field_name == field

    def test_attaches_source(self) -> None:
        md = (
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
            "> Reason: confirmed.\n"
        )
        overrides = parse_overrides(md, source="/path/to/file.md")
        assert len(overrides) == 1
        assert overrides[0].source == "/path/to/file.md"

    def test_empty_text_returns_empty_list(self) -> None:
        assert parse_overrides("") == []
        assert parse_overrides("Plain markdown without overrides.\n") == []

    def test_malformed_blocks_are_skipped(self) -> None:
        md = (
            "> User Override:\n"
            "> bad_field changed from 1 to 2.\n"
            "> Reason: test.\n"
        )
        # bad_field is not in the regex group; the regex won't match at all
        assert parse_overrides(md) == []

    def test_missing_reason_line_is_skipped(self) -> None:
        md = (
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
        )
        assert parse_overrides(md) == []

    def test_extra_whitespace_in_reason_is_stripped(self) -> None:
        md = (
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
            "> Reason:   padded reason  \n"
        )
        overrides = parse_overrides(md)
        assert overrides[0].reason == "padded reason"

    def test_values_with_spaces(self) -> None:
        md = (
            "> User Override:\n"
            "> ranking changed from 5th place to 2nd place.\n"
            "> Reason: better fit for community.\n"
        )
        overrides = parse_overrides(md)
        assert overrides[0].old_value == "5th place"
        assert overrides[0].new_value == "2nd place"


# ---------------------------------------------------------------------------
# apply_score_overrides
# ---------------------------------------------------------------------------


class TestApplyScoreOverrides:
    def test_overrides_validity_score_and_recalculates_rank(self) -> None:
        scores = _make_scores(validity_score=61)
        original_rank = scores.rank_score

        ov = UserOverride(
            field_name="validity_score",
            old_value="61",
            new_value="85",
            reason="confirmed",
        )
        result = apply_score_overrides(scores, [ov])

        assert result.validity_score == 85
        # rank should be recalculated (higher validity → higher rank)
        assert result.rank_score > original_rank
        # other fields unchanged
        assert result.people_helped_score == 70
        assert result.severity_score == 80

    def test_overrides_ranking_forces_rank_score(self) -> None:
        scores = _make_scores()
        original_validity = scores.validity_score

        ov = UserOverride(
            field_name="ranking", old_value="70", new_value="99", reason="force rank"
        )
        result = apply_score_overrides(scores, [ov])

        assert result.rank_score == 99
        # validity unchanged when ranking is overridden directly
        assert result.validity_score == original_validity

    def test_ranking_override_takes_precedence_over_validity(self) -> None:
        """When both ranking and validity are overridden, ranking wins."""
        scores = _make_scores(validity_score=61)

        overrides = [
            UserOverride("validity_score", "61", "85", "confirmed"),
            UserOverride("ranking", "70", "55", "manual rank"),
        ]
        result = apply_score_overrides(scores, overrides)
        assert result.validity_score == 85
        assert result.rank_score == 55

    def test_non_numeric_override_is_ignored(self) -> None:
        scores = _make_scores(validity_score=61)

        ov = UserOverride(
            field_name="validity_score",
            old_value="61",
            new_value="not-a-number",
            reason="oops",
        )
        result = apply_score_overrides(scores, [ov])
        assert result.validity_score == 61  # unchanged

    def test_clamps_out_of_range_scores(self) -> None:
        scores = _make_scores(validity_score=50)

        ov = UserOverride("validity_score", "50", "999", "too high")
        result = apply_score_overrides(scores, [ov])
        assert result.validity_score == 100

        ov2 = UserOverride("validity_score", "50", "-50", "too low")
        result2 = apply_score_overrides(scores, [ov2])
        assert result2.validity_score == 0

    def test_no_matching_overrides_returns_unchanged_scores(self) -> None:
        scores = _make_scores()
        ov = UserOverride("verdict", "x", "y", "wrong type")
        result = apply_score_overrides(scores, [ov])
        assert result.validity_score == scores.validity_score
        assert result.rank_score == scores.rank_score


# ---------------------------------------------------------------------------
# apply_verdict_override
# ---------------------------------------------------------------------------


class TestApplyVerdictOverride:
    def test_applies_verdict(self) -> None:
        opp = _make_opportunity(verdict=Verdict.RESEARCH_MORE)
        ov = UserOverride("verdict", "RESEARCH_MORE", "BUILD_POC", "urgent")
        result = apply_verdict_override(opp, [ov])
        assert result == Verdict.BUILD_POC

    def test_last_verdict_wins(self) -> None:
        opp = _make_opportunity(verdict=Verdict.RESEARCH_MORE)
        overrides = [
            UserOverride("verdict", "RESEARCH_MORE", "BUILD_POC", "first"),
            UserOverride("verdict", "RESEARCH_MORE", "DO_NOT_TOUCH", "last"),
        ]
        result = apply_verdict_override(opp, overrides)
        assert result == Verdict.DO_NOT_TOUCH

    def test_returns_none_when_no_verdict_override(self) -> None:
        opp = _make_opportunity()
        ov = UserOverride("validity_score", "61", "75", "score only")
        result = apply_verdict_override(opp, [ov])
        assert result is None

    def test_invalid_verdict_value_is_ignored(self) -> None:
        opp = _make_opportunity(verdict=Verdict.RESEARCH_MORE)
        ov = UserOverride("verdict", "RESEARCH_MORE", "INVALID_VERDICT", "bad")
        result = apply_verdict_override(opp, [ov])
        assert result is None


# ---------------------------------------------------------------------------
# apply_trust_overrides
# ---------------------------------------------------------------------------


class TestApplyTrustOverrides:
    def test_applies_source_trust_override(self) -> None:
        ov = UserOverride("source_trust", "50", "90", "more trustworthy")
        result = apply_trust_overrides(50.0, [ov])
        assert result == 90.0

    def test_last_source_trust_wins(self) -> None:
        overrides = [
            UserOverride("source_trust", "50", "90", "first"),
            UserOverride("source_trust", "50", "75", "last"),
        ]
        result = apply_trust_overrides(50.0, overrides)
        assert result == 75.0

    def test_clamps_out_of_range(self) -> None:
        ov = UserOverride("source_trust", "50", "150", "too high")
        assert apply_trust_overrides(50.0, [ov]) == 100.0

        ov2 = UserOverride("source_trust", "50", "-10", "too low")
        assert apply_trust_overrides(50.0, [ov2]) == 0.0

    def test_returns_original_when_no_override(self) -> None:
        ov = UserOverride("validity_score", "61", "75", "wrong field")
        assert apply_trust_overrides(42.0, [ov]) == 42.0

    def test_non_numeric_is_ignored(self) -> None:
        ov = UserOverride("source_trust", "50", "high", "not a number")
        assert apply_trust_overrides(50.0, [ov]) == 50.0


# ---------------------------------------------------------------------------
# format_override_markdown
# ---------------------------------------------------------------------------


class TestFormatOverrideMarkdown:
    def test_formats_single_override(self) -> None:
        ov = UserOverride(
            field_name="validity_score",
            old_value="61",
            new_value="75",
            reason="confirmed source.",
        )
        result = format_override_markdown([ov])
        assert "**User Override Applied:**" in result
        assert "`validity_score`: 61 → 75" in result
        assert "Reason: confirmed source." in result

    def test_formats_multiple_overrides(self) -> None:
        overrides = [
            UserOverride("validity_score", "61", "75", "confirmed"),
            UserOverride("verdict", "RESEARCH_MORE", "BUILD_POC", "urgent"),
        ]
        result = format_override_markdown(overrides)
        assert "`validity_score`: 61 → 75" in result
        assert "`verdict`: RESEARCH_MORE → BUILD_POC" in result

    def test_empty_overrides_returns_empty_string(self) -> None:
        assert format_override_markdown([]) == ""


# ---------------------------------------------------------------------------
# has_overrides / strip_overrides
# ---------------------------------------------------------------------------


class TestHasAndStripOverrides:
    def test_has_overrides_detects_presence(self) -> None:
        md = (
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
            "> Reason: test.\n"
        )
        assert has_overrides(md) is True

    def test_has_overrides_returns_false_for_plain_text(self) -> None:
        assert has_overrides("Just some markdown.\n") is False
        assert has_overrides("") is False

    def test_strip_overrides_removes_override_blocks(self) -> None:
        md = (
            "Some intro text.\n\n"
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
            "> Reason: test.\n\n"
            "Trailing content.\n"
        )
        result = strip_overrides(md)
        assert "User Override:" not in result
        assert "Some intro text." in result
        assert "Trailing content." in result

    def test_strip_overrides_with_no_overrides(self) -> None:
        md = "Plain markdown.\n"
        assert strip_overrides(md) == "Plain markdown."


# ---------------------------------------------------------------------------
# Re-render preservation (integration-style)
# ---------------------------------------------------------------------------


class TestRerenderPreservation:
    """Simulate a re-render workflow: parse overrides → apply → re-emit."""

    def test_roundtrip_preserves_overrides(self) -> None:
        """Overrides are parsed from existing markdown, applied, and
        re-emitted in an 'applied' block."""
        existing_report = (
            "# Opportunity: senior-services-guide\n\n"
            "Scores:\n"
            "- validity_score: 61\n"
            "- rank_score: 75.3\n\n"
            "> User Override:\n"
            "> validity_score changed from 61 to 75.\n"
            "> Reason: user manually confirmed source relevance.\n"
        )

        # Parse the existing overrides
        overrides = parse_overrides(existing_report, source="opp.md")

        # Apply them
        scores = _make_scores(validity_score=61)
        new_scores = apply_score_overrides(scores, overrides)

        # Re-render: strip old overrides, emit new content + applied markers
        base = strip_overrides(existing_report)
        applied_md = format_override_markdown(overrides)
        rerendered = f"{base}\n\n{applied_md}".strip() + "\n"

        assert new_scores.validity_score == 75
        assert "User Override:" not in base  # original stripped
        assert "**User Override Applied:**" in rerendered
        assert "`validity_score`: 61 → 75" in rerendered

    def test_multiple_overrides_flow_through_to_ranking(self) -> None:
        """Multiple overrides affecting validity → ranking flow through."""
        overrides = [
            UserOverride("validity_score", "61", "95", "much higher validity"),
        ]

        scores = _make_scores(validity_score=61)
        original_rank = scores.rank_score

        new_scores = apply_score_overrides(scores, overrides)

        # Higher validity → higher rank
        assert new_scores.rank_score > original_rank
        assert new_scores.validity_score == 95

        # Format should show the override
        md = format_override_markdown(overrides)
        assert "61 → 95" in md
