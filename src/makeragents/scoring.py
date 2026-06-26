"""Pure, deterministic scoring engine for MakerAgents opportunity ranking.

This module provides standalone functions for computing the harm-reduction
and rank scores used throughout the agent pipeline.  All functions are
pure (no I/O) and deterministic so that results can be reproduced from
the same numeric inputs.
"""

from __future__ import annotations

# ── weight constants (per PRD §Ranking formula) ──────────────────────────
_W_PEOPLE_HELPED = 0.22
_W_SEVERITY = 0.20
_W_VALIDITY = 0.18
_W_INTERVENTION_EASE = 0.14
_W_LOW_HARM = 0.14
_W_ABILITY_TO_ACT = 0.12

# Ensure the weights sum to 1.0 exactly (floating-point equality is safe
# here because these are short, terminating decimal fractions).
_sum_weights = sum(
    [
        _W_PEOPLE_HELPED,
        _W_SEVERITY,
        _W_VALIDITY,
        _W_INTERVENTION_EASE,
        _W_LOW_HARM,
        _W_ABILITY_TO_ACT,
    ]
)
if abs(_sum_weights - 1.0) >= 1e-9:
    raise ValueError(f"Weight sum must equal 1.0, got {_sum_weights}")


def compute_low_harm_score(*, harm_risk_score: float) -> float:
    """Return the harm-reduction complement of *harm_risk_score*.

    This is a separate public function (rather than inline arithmetic) for
    semantic clarity: the "Do No Harm" score is a named concept in the PRD
    (§11), and keeping it as its own function makes that relationship
    explicit in callers and tests.

    ``low_harm_score = 100 - harm_risk_score``

    Args:
        harm_risk_score: Harm risk component score in range [0.0, 100.0].

    Returns:
        float: Low-harm score in range [0.0, 100.0].
    """
    return 100.0 - harm_risk_score


def compute_rank_score(
    *,
    people_helped_score: float,
    severity_score: float,
    validity_score: float,
    intervention_ease_score: float,
    harm_risk_score: float,
    ability_to_act_score: float,
) -> float:
    """Compute the weighted rank score from the six component scores.

    The formula (per PRD) is::

        rank_score =
          people_helped_score     * 0.22 +
          severity_score          * 0.20 +
          validity_score          * 0.18 +
          intervention_ease_score * 0.14 +
          low_harm_score          * 0.14 +
          ability_to_act_score    * 0.12

    where ``low_harm_score = 100 - harm_risk_score``.

    The taker score is deliberately **not** included in this formula; it
    is displayed separately and does not affect the rank score.

    Args:
        people_helped_score: Component score in range [0.0, 100.0].
        severity_score: Component score in range [0.0, 100.0].
        validity_score: Component score in range [0.0, 100.0].
        intervention_ease_score: Component score in range [0.0, 100.0].
        harm_risk_score: Component score in range [0.0, 100.0].
        ability_to_act_score: Component score in range [0.0, 100.0].

    Returns
    -------
    float
        Rank score rounded to 2 decimal places.
    """
    low_harm = compute_low_harm_score(harm_risk_score=harm_risk_score)

    raw = (
        people_helped_score * _W_PEOPLE_HELPED
        + severity_score * _W_SEVERITY
        + validity_score * _W_VALIDITY
        + intervention_ease_score * _W_INTERVENTION_EASE
        + low_harm * _W_LOW_HARM
        + ability_to_act_score * _W_ABILITY_TO_ACT
    )

    return round(raw, 2)
