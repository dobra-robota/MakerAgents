"""MakerAgents package foundation."""

from makeragents.config import AppConfig, load_config
from makeragents.search import ProviderResponse, SearchClient, SearchResult
from makeragents.schemas import (
    ClaimClassification,
    Confidence,
    EvidenceItem,
    EvidenceType,
    Opportunity,
    OpportunityType,
    POCType,
    RunMetadata,
    ScoreSet,
    SourceType,
    Verdict,
)

__all__ = [
    "AppConfig",
    "ClaimClassification",
    "Confidence",
    "EvidenceItem",
    "EvidenceType",
    "Opportunity",
    "OpportunityType",
    "POCType",
    "ProviderResponse",
    "RunMetadata",
    "ScoreSet",
    "SearchClient",
    "SearchResult",
    "SourceType",
    "Verdict",
    "load_config",
]

__version__ = "0.1.0"
