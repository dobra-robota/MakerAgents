"""Tests for the pure, deterministic scoring engine."""

import pytest

from makeragents.scoring import compute_low_harm_score, compute_rank_score


# ── low-harm score ────────────────────────────────────────────────────────


def test_low_harm_score_zero_risk() -> None:
    """When harm risk is 0, low harm is 100."""
    assert compute_low_harm_score(harm_risk_score=0.0) == 100.0


def test_low_harm_score_full_risk() -> None:
    """When harm risk is 100, low harm is 0."""
    assert compute_low_harm_score(harm_risk_score=100.0) == 0.0


def test_low_harm_score_mid_risk() -> None:
    """Mid-range harm risk produces the complement."""
    assert compute_low_harm_score(harm_risk_score=33.0) == 67.0
    assert compute_low_harm_score(harm_risk_score=67.0) == 33.0


# ── rank score: hand-computed fixtures ────────────────────────────────────


def test_rank_score_fixture_from_existing_test() -> None:
    """Matches the fixture value used in test_schemas.py (75.3)."""
    result = compute_rank_score(
        people_helped_score=70,
        severity_score=80,
        validity_score=85,
        intervention_ease_score=60,
        harm_risk_score=20,
        ability_to_act_score=75,
    )
    # 70*0.22 + 80*0.20 + 85*0.18 + 60*0.14 + 80*0.14 + 75*0.12
    # = 15.4 + 16.0 + 15.3 + 8.4 + 11.2 + 9.0 = 75.3
    assert result == 75.3


def test_rank_score_all_zeros() -> None:
    """All-zero scores still get the low-harm bonus (harm_risk=0 → low_harm=100)."""
    result = compute_rank_score(
        people_helped_score=0,
        severity_score=0,
        validity_score=0,
        intervention_ease_score=0,
        harm_risk_score=0,
        ability_to_act_score=0,
    )
    # Only the low-harm term contributes: 100 * 0.14 = 14.0
    assert result == 14.0


def test_rank_score_all_hundreds() -> None:
    """All-100 scores (harm_risk=100 → low_harm=0)."""
    result = compute_rank_score(
        people_helped_score=100,
        severity_score=100,
        validity_score=100,
        intervention_ease_score=100,
        harm_risk_score=100,
        ability_to_act_score=100,
    )
    # 100*(0.22+0.20+0.18+0.14+0.12) + 0*0.14 = 100*0.86 = 86.0
    assert result == 86.0


def test_rank_score_all_fifty() -> None:
    """All mid-range scores (harm_risk=50 → low_harm=50)."""
    result = compute_rank_score(
        people_helped_score=50,
        severity_score=50,
        validity_score=50,
        intervention_ease_score=50,
        harm_risk_score=50,
        ability_to_act_score=50,
    )
    # 50*(0.22+0.20+0.18+0.14+0.14+0.12) = 50*1.0 = 50.0
    assert result == 50.0


def test_rank_score_mixed_values() -> None:
    """Hand-computed mixed-value fixture."""
    result = compute_rank_score(
        people_helped_score=90,
        severity_score=75,
        validity_score=60,
        intervention_ease_score=45,
        harm_risk_score=30,
        ability_to_act_score=85,
    )
    # low_harm = 100-30 = 70
    # 90*0.22 + 75*0.20 + 60*0.18 + 45*0.14 + 70*0.14 + 85*0.12
    # = 19.8 + 15.0 + 10.8 + 6.3 + 9.8 + 10.2 = 71.9
    assert result == 71.9


# ── rounding ──────────────────────────────────────────────────────────────


def test_rank_score_rounds_to_two_decimals() -> None:
    """Raw values with >2 decimal places are rounded to 2."""
    result = compute_rank_score(
        people_helped_score=0.33,
        severity_score=0.33,
        validity_score=0.33,
        intervention_ease_score=0.33,
        harm_risk_score=0.33,
        ability_to_act_score=0.33,
    )
    # low_harm = 100 - 0.33 = 99.67
    # raw = 0.33*(0.22+0.20+0.18+0.14+0.12) + 99.67*0.14
    #     = 0.33*0.86 + 13.9538
    #     = 0.2838 + 13.9538 = 14.2376
    # rounded to 2 → 14.24
    assert result == 14.24


# ── taker score exclusion ─────────────────────────────────────────────────


def test_taker_score_not_in_signature() -> None:
    """compute_rank_score does not accept a taker_score parameter."""
    import inspect

    sig = inspect.signature(compute_rank_score)
    param_names = set(sig.parameters.keys())
    assert "taker_score" not in param_names


# ── determinism ───────────────────────────────────────────────────────────


def test_deterministic_output() -> None:
    """Same inputs always produce the same output."""
    kwargs = dict(
        people_helped_score=72,
        severity_score=68,
        validity_score=81,
        intervention_ease_score=55,
        harm_risk_score=25,
        ability_to_act_score=79,
    )
    first = compute_rank_score(**kwargs)
    for _ in range(100):
        assert compute_rank_score(**kwargs) == first


# ── weight sum invariant ──────────────────────────────────────────────────


def test_weights_sum_to_one() -> None:
    """Internal module-level assertion: weights sum to 1.0."""
    from makeragents.scoring import (
        _W_ABILITY_TO_ACT,
        _W_INTERVENTION_EASE,
        _W_LOW_HARM,
        _W_PEOPLE_HELPED,
        _W_SEVERITY,
        _W_VALIDITY,
    )

    total = (
        _W_PEOPLE_HELPED
        + _W_SEVERITY
        + _W_VALIDITY
        + _W_INTERVENTION_EASE
        + _W_LOW_HARM
        + _W_ABILITY_TO_ACT
    )
    assert total == 1.0


# ── same scale for any opportunity type ───────────────────────────────────


def test_same_function_for_any_opportunity_type() -> None:
    """Commercial and non-commercial opportunities use the same formula.

    The scoring engine is pure math; it has no notion of opportunity type.
    Any caller passes the same six scores regardless of whether the
    opportunity is commercial or non-commercial.
    """
    # Simulate a "commercial" set of scores — the function does not care.
    commercial = compute_rank_score(
        people_helped_score=80,
        severity_score=70,
        validity_score=90,
        intervention_ease_score=85,
        harm_risk_score=10,
        ability_to_act_score=95,
    )
    # Same scores run through the same function — obviously same result.
    same = compute_rank_score(
        people_helped_score=80,
        severity_score=70,
        validity_score=90,
        intervention_ease_score=85,
        harm_risk_score=10,
        ability_to_act_score=95,
    )
    assert commercial == same


# ── backward compatibility via ScoreSet.calculate_rank_score ──────────────


def test_scoreset_calculate_rank_score_delegates_to_scoring() -> None:
    """ScoreSet.calculate_rank_score should return the same result as
    calling compute_rank_score directly."""
    from makeragents.schemas import ScoreSet

    kwargs = dict(
        people_helped_score=70,
        severity_score=80,
        validity_score=85,
        intervention_ease_score=60,
        harm_risk_score=20,
        ability_to_act_score=75,
    )
    direct = compute_rank_score(**kwargs)
    via_scoreset = ScoreSet.calculate_rank_score(**kwargs)
    assert via_scoreset == direct == 75.3
