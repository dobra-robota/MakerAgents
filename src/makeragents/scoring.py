"""Ranking and harm-score helpers for MakerAgents.

Provides the documented rank_score and low_harm_score formulas consumed
by the Report Agent and other components that need to re-derive rankings.
"""

from __future__ import annotations


def compute_low_harm_score(*, harm_risk_score: float) -> float:
    """Compute the low-harm score from the raw harm-risk score.

    ``low_harm_score = 100 - harm_risk_score``
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
    """Calculate the documented weighted ranking score.

    Formula from PRD §15:

        rank_score =
          people_helped_score     * 0.22 +
          severity_score          * 0.20 +
          validity_score          * 0.18 +
          intervention_ease_score * 0.14 +
          (100 - harm_risk_score) * 0.14 +
          ability_to_act_score    * 0.12
    """
    low_harm = 100.0 - harm_risk_score
    return round(
        people_helped_score * 0.22
        + severity_score * 0.20
        + validity_score * 0.18
        + intervention_ease_score * 0.14
        + low_harm * 0.14
        + ability_to_act_score * 0.12,
        2,
    )
