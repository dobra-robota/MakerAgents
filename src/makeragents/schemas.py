"""Core Pydantic schemas for MakerAgents run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from makeragents.scoring import compute_low_harm_score, compute_rank_score

NonEmptyString = Annotated[str, Field(min_length=1)]
ScoreValue = Annotated[float, Field(ge=0, le=100)]


class MakerAgentsModel(BaseModel):
    """Base model for strict, serializable MakerAgents schemas."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Confidence(str, Enum):
    """Allowed confidence labels for evidence and scored arguments."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClaimClassification(str, Enum):
    """How strongly a claim is grounded in available evidence."""

    EVIDENCE_BASED = "evidence_based"
    INFERENCE = "inference"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    """Evidence categories collected during research."""

    CLAIM = "claim"
    COMPLAINT = "complaint"
    OFFICIAL_STATEMENT = "official_statement"
    NEWS_REPORT = "news_report"
    FIRST_HAND_ACCOUNT = "first_hand_account"
    SECOND_HAND_ACCOUNT = "second_hand_account"
    STATISTIC = "statistic"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """Known source categories used for trust scoring."""

    GOVERNMENT = "government"
    ACADEMIC = "academic"
    MAJOR_NEWS = "major_news"
    LOCAL_NEWS = "local_news"
    NGO = "ngo"
    COMPANY_OFFICIAL = "company_official"
    FORUM = "forum"
    REDDIT = "reddit"
    ANONYMOUS_SOCIAL = "anonymous_social"
    UNKNOWN = "unknown"


class OpportunityType(str, Enum):
    """Opportunity shapes MakerAgents can recommend for review."""

    PUBLIC_GUIDE = "public_guide"
    COORDINATION_PROCESS = "coordination_process"
    ADVOCACY_REPORT = "advocacy_report"
    TRANSPARENCY_DASHBOARD = "transparency_dashboard"
    MANUAL_SERVICE = "manual_service"
    COMMUNITY_SUPPORT_PROCESS = "community_support_process"
    SOFTWARE_TOOLING = "software_tooling"
    INSTITUTION_FACING_REPORT = "institution_facing_report"
    OPEN_DATA_RESOURCE = "open_data_resource"


class POCType(str, Enum):
    """Proof-of-concept types for cost estimates."""

    MANUAL_SERVICE = "manual_service"
    PUBLIC_GUIDE = "public_guide"
    DASHBOARD = "dashboard"
    AUTOMATION = "automation"
    ADVOCACY_REPORT = "advocacy_report"
    SOFTWARE_PROTOTYPE = "software_prototype"
    COORDINATION_PROCESS = "coordination_process"
    OPEN_DATA_RESOURCE = "open_data_resource"


class Verdict(str, Enum):
    """Allowed mediator verdicts for an opportunity."""

    IGNORE = "IGNORE"
    WATCH = "WATCH"
    RESEARCH_MORE = "RESEARCH_MORE"
    MANUAL_POC = "MANUAL_POC"
    BUILD_POC = "BUILD_POC"
    DO_NOT_TOUCH = "DO_NOT_TOUCH"
    NON_INTERVENTION = "NON_INTERVENTION"


class RunMetadata(MakerAgentsModel):
    """Metadata that identifies a single city-plus-community run."""

    run_id: NonEmptyString
    city: NonEmptyString
    community: NonEmptyString
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    max_opportunities: int = Field(default=5, ge=1)
    output_dir: NonEmptyString = "runs"


class EvidenceItem(MakerAgentsModel):
    """A normalized public-source evidence item."""

    id: NonEmptyString
    source_url: HttpUrl
    source_domain: NonEmptyString
    source_type: SourceType
    evidence_type: EvidenceType
    snippet: NonEmptyString
    language: NonEmptyString
    claim_classification: ClaimClassification
    trust_score: ScoreValue
    recency: NonEmptyString
    confidence: Confidence


class ScoreSet(MakerAgentsModel):
    """Scores used to rank and explain an opportunity."""

    validity_score: ScoreValue
    maker_score: ScoreValue
    maker_confidence: Confidence
    taker_score: ScoreValue
    taker_confidence: Confidence
    people_helped_score: ScoreValue
    severity_score: ScoreValue
    impact_score: ScoreValue
    intervention_ease_score: ScoreValue
    harm_risk_score: ScoreValue
    ability_to_act_score: ScoreValue
    rank_score: ScoreValue

    @staticmethod
    def calculate_rank_score(
        *,
        people_helped_score: float,
        severity_score: float,
        validity_score: float,
        intervention_ease_score: float,
        harm_risk_score: float,
        ability_to_act_score: float,
    ) -> float:
        """Calculate the documented ranking score from component scores.

        Delegates to the pure ``makeragents.scoring.compute_rank_score``
        function so that the formula lives in a single, testable location.
        """
        return compute_rank_score(
            people_helped_score=people_helped_score,
            severity_score=severity_score,
            validity_score=validity_score,
            intervention_ease_score=intervention_ease_score,
            harm_risk_score=harm_risk_score,
            ability_to_act_score=ability_to_act_score,
        )


class Opportunity(MakerAgentsModel):
    """A candidate intervention or resource grounded in evidence."""

    id: NonEmptyString
    title: NonEmptyString
    type: OpportunityType
    pain_summary: NonEmptyString
    who_benefits: list[NonEmptyString] = Field(min_length=1)
    vulnerable_groups: list[NonEmptyString] = Field(default_factory=list)
    evidence_ids: list[NonEmptyString] = Field(default_factory=list)
    speculative: bool = False
    scores: ScoreSet | None = None
    verdict: Verdict | None = None
