"""MakerAgents package foundation."""

from makeragents.config import AppConfig, load_config
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
from makeragents.sources import SourceRegistry, load_registry

__all__ = [
    "AppConfig",
    "ClaimClassification",
    "Confidence",
    "EvidenceItem",
    "EvidenceType",
    "Opportunity",
    "OpportunityType",
    "POCType",
    "RunMetadata",
    "ScoreSet",
    "SourceRegistry",
    "SourceType",
    "Verdict",
    "load_config",
    "load_registry",
]

__version__ = "0.1.0"
