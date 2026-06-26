"""Pure, deterministic scoring engine for MakerAgents opportunity ranking.

This module provides standalone functions for computing the harm-reduction
and rank scores used throughout the agent pipeline.  All functions are
pure (no I/O) and deterministic so that results can be reproduced from
the same numeric inputs.

The actual formula implementation lives in :class:`makeragents.schemas.ScoreSet`
to avoid duplication; the functions here are thin delegation wrappers.
"""

from __future__ import annotations

from makeragents.schemas import ScoreSet


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
        float: Low-harm score in range [0.0, 100.0], rounded to 2 decimal places.
    """
    return round(100.0 - harm_risk_score, 2)


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

    Delegates to :meth:`ScoreSet.calculate_rank_score` to avoid duplicating
    the formula across modules.

    Args:
        people_helped_score: Component score in range [0.0, 100.0].
        severity_score: Component score in range [0.0, 100.0].
        validity_score: Component score in range [0.0, 100.0].
        intervention_ease_score: Component score in range [0.0, 100.0].
        harm_risk_score: Component score in range [0.0, 100.0].
        ability_to_act_score: Component score in range [0.0, 100.0].

    Returns:
        float: Rank score rounded to 2 decimal places.
    """
    return ScoreSet.calculate_rank_score(
        people_helped_score=people_helped_score,
        severity_score=severity_score,
        validity_score=validity_score,
        intervention_ease_score=intervention_ease_score,
        harm_risk_score=harm_risk_score,
        ability_to_act_score=ability_to_act_score,
    )
